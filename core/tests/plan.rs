//! The plan: fit arithmetic, labels, and the command-line program.

use kvscope_core::gpu;
use kvscope_core::plan::{plan, Basis, Labelled, PlanRequest};
use kvscope_core::units::{GIB, MIB};
use kvscope_core::{parse_config, KvDtype, KvOptions, MlaStorage};
use std::process::Command;

const GPTOSS: &str = include_str!("fixtures/gpt-oss-120b.json");

fn request(weights_gib: Option<f64>) -> PlanRequest {
    PlanRequest {
        model_id: Some("openai/gpt-oss-120b".into()),
        seq_len: 32 * 1024,
        users: 16,
        gpu: gpu::find("H100-80GB").unwrap(),
        gpu_count: 1,
        kv: KvOptions {
            dtype: KvDtype::Bf16,
            mla: MlaStorage::Latent,
        },
        kv_dtype_basis: Basis::Assumed,
        weights_bytes: weights_gib.map(|g| g * GIB),
        overhead_bytes_per_gpu: Labelled::new(1.0 * GIB, Basis::Assumed),
        utilisation: Labelled::new(0.9, Basis::Assumed),
    }
}

#[test]
fn fit_arithmetic() {
    let layout = parse_config(GPTOSS).unwrap();
    let p = plan(&layout, &request(Some(40.0)));
    // One 32k request: 18 full layers x 32768 tokens + 18 sliding x 128, at 2 KiB/token.
    let per_layer = (18.0 * 32768.0 + 18.0 * 128.0) * 2048.0;
    let uniform = 36.0 * 32768.0 * 2048.0;
    assert_eq!(p.per_request_bytes.per_layer, per_layer);
    assert_eq!(p.per_request_bytes.uniform, uniform);
    assert_eq!(p.all_users_bytes.uniform, 16.0 * uniform);

    let f = p.fit.as_ref().unwrap();
    let budget = 81559.0 * MIB * 0.9 - 40.0 * GIB - 1.0 * GIB;
    assert!((f.kv_budget_bytes.value - budget).abs() < 1.0);
    assert_eq!(
        f.max_requests.per_layer,
        (budget / per_layer).floor() as u64
    );
    assert_eq!(f.max_requests.uniform, (budget / uniform).floor() as u64);
    assert_eq!(f.users_fit.per_layer, f.max_requests.per_layer >= 16);
}

#[test]
fn labels() {
    let layout = parse_config(GPTOSS).unwrap();
    let p = plan(&layout, &request(Some(40.0)));
    assert_eq!(p.gpu_memory_bytes.basis, Basis::Measured); // from nvidia-smi
    assert_eq!(p.kv_dtype.basis, Basis::Assumed); // config has no dtype
    assert_eq!(p.uniform_waste_fraction.basis, Basis::Calculated);
    let f = p.fit.as_ref().unwrap();
    assert_eq!(f.weights_bytes.basis, Basis::Input);
    assert_eq!(f.overhead_bytes.basis, Basis::Assumed);
    assert_eq!(p.measured.len(), 3);

    let nominal = PlanRequest {
        gpu: gpu::find("RTX4060-8GB").unwrap(),
        ..request(None)
    };
    assert_eq!(
        plan(&layout, &nominal).gpu_memory_bytes.basis,
        Basis::Assumed
    );
}

#[test]
fn no_weights_no_fit_and_overfull_gpu_fits_nothing() {
    let layout = parse_config(GPTOSS).unwrap();
    assert!(plan(&layout, &request(None)).fit.is_none());
    // Measured gpt-oss-120b baseline on H200: 76,900.6 MiB (weights + CUDA context).
    let heavy = plan(&layout, &request(Some(76900.6 / 1024.0)));
    let f = heavy.fit.unwrap();
    assert!(f.kv_budget_bytes.value < 0.0);
    assert_eq!(f.max_requests.per_layer, 0);
    assert!(!f.users_fit.per_layer);
}

fn kvscope(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kvscope"))
        .args(args)
        .output()
        .expect("run kvscope")
}

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gpt-oss-120b.json"
);

#[test]
fn cli_json_carries_labels() {
    let out = kvscope(&[
        "plan",
        "--config",
        FIXTURE,
        "--model",
        "openai/gpt-oss-120b",
        "--context",
        "32k",
        "--users",
        "16",
        "--weights-gib",
        "40",
        "--json",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["seq_len"], 32768);
    assert_eq!(v["gpu_memory_bytes"]["basis"], "measured");
    assert_eq!(v["kv_dtype"]["basis"], "assumed");
    assert_eq!(v["fit"]["weights_bytes"]["basis"], "input");
    assert!(v["measured"][0]["source"]
        .as_str()
        .unwrap()
        .starts_with("results/"));
}

#[test]
fn cli_errors_exit_2_with_a_message() {
    let out = kvscope(&[
        "plan",
        "--config",
        FIXTURE,
        "--context",
        "4096",
        "--gpu",
        "H100",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown GPU 'H100'"));

    let out = kvscope(&["plan", "--config", "no/such/file.json", "--context", "4096"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
}
