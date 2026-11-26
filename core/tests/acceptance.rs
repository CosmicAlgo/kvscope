//! Acceptance tests: the planner's calculations against values measured by the
//! Python tracer on H100 (V1) and H200 (V2). Each measured value names its
//! source file. "MB" in those files is 1024-based (MiB).

use kvscope_core::kv::{layer_bytes, tokens_held};
use kvscope_core::units::{KIB, MIB};
use kvscope_core::{
    density_bytes_per_token_per_layer, parse_config, sequence_kv, KvDtype, KvOptions, LayerKind,
    MlaStorage,
};

const BF16: KvOptions = KvOptions {
    dtype: KvDtype::Bf16,
    mla: MlaStorage::Latent,
};

fn round_to(x: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (x * f).round() / f
}

#[test]
fn pythia_is_exactly_8_kib_per_token_per_layer() {
    // Measured: 8.00 KB/token/layer, V1 and V2
    // (results/h100_v1/comparative_analysis.json, results/h200_v2/mha_baseline_profile.json).
    let m = parse_config(include_str!("fixtures/pythia-1.4b-deduped.json")).unwrap();
    assert_eq!(density_bytes_per_token_per_layer(&m, BF16), 8.0 * KIB);
}

/// gpt-oss-120b, V2 prompt 0 (results/h200_v2/gptoss_profile.json,
/// per_prompt[0].tracer.kv_cache.per_layer_mb): 18 layers of 4.211 MB and
/// 18 of 0.248 MB. That prompt's length was not recorded; a full-attention
/// layer holds 2 KiB per token, so 4.211 MiB implies 2156 cached tokens.
const GPTOSS_FULL_MIB: f64 = 4.211;
const GPTOSS_SLIDING_MIB: f64 = 0.248;

fn gptoss() -> kvscope_core::ModelLayout {
    parse_config(include_str!("fixtures/gpt-oss-120b.json")).unwrap()
}

#[test]
fn gptoss_layer_sizes_match_measurement() {
    let m = gptoss();
    let seq_len = (GPTOSS_FULL_MIB * MIB / (2.0 * KIB)).round() as u64;
    assert_eq!(seq_len, 2156);

    let full = layer_bytes(&m.layers[1], seq_len, BF16) / MIB;
    let sliding = layer_bytes(&m.layers[0], seq_len, BF16) / MIB;
    assert_eq!(round_to(full, 3), GPTOSS_FULL_MIB);
    // The planner gives a sliding layer the full 128-token window (0.250 MiB).
    // Hugging Face transformers held 127 tokens (0.248 MiB): one token less.
    // Compared in whole tokens, because the measured value is rounded to 3 decimals.
    let measured_tokens = (GPTOSS_SLIDING_MIB * MIB / (2.0 * KIB)).round() as u64;
    assert_eq!(measured_tokens, 127);
    assert_eq!(tokens_held(&m.layers[0], seq_len), 128);
    assert_eq!(sliding, 0.25);
}

#[test]
fn gptoss_uniform_allocation_wastes_47_percent_at_the_measured_length() {
    let m = gptoss();
    let kv = sequence_kv(&m, 2156, BF16);
    // Report: 18 x 4.211 + 18 x 0.248 = 80.3 MB used vs 36 x 4.211 = 151.6 MB.
    assert_eq!(round_to(kv.per_layer_total / MIB, 1), 80.3);
    assert_eq!(round_to(kv.uniform_total / MIB, 1), 151.6);
    assert_eq!((kv.waste_fraction() * 100.0).round(), 47.0);

    // The same figure from the measured layer sizes alone.
    let measured =
        1.0 - (18.0 * GPTOSS_FULL_MIB + 18.0 * GPTOSS_SLIDING_MIB) / (36.0 * GPTOSS_FULL_MIB);
    assert_eq!((measured * 100.0).round(), 47.0);
}

#[test]
fn gptoss_waste_follows_half_of_one_minus_window_over_length() {
    let m = gptoss();
    for seq_len in [128u64, 1000, 2156, 32 * 1024, 128 * 1024] {
        let expected = 0.5 * (1.0 - 128.0 / seq_len as f64);
        let got = sequence_kv(&m, seq_len, BF16).waste_fraction();
        assert!(
            (got - expected).abs() < 1e-12,
            "seq_len {seq_len}: {got} vs {expected}"
        );
    }
    // Shorter than the window: nothing to save.
    assert_eq!(sequence_kv(&m, 100, BF16).waste_fraction(), 0.0);
}

#[test]
fn gemma4_31b_density_matches_and_hf_did_not_trim_sliding_layers() {
    let m = parse_config(include_str!("fixtures/gemma-4-31b-it.json")).unwrap();
    // Measured 14.67 KB/token/layer in V1 and V2 = (50 x 16 + 10 x 8) / 60.
    let density = density_bytes_per_token_per_layer(&m, BF16) / KIB;
    assert_eq!(round_to(density, 2), 14.67);

    // V2 prompt 0 (results/h200_v2/gemma4_profile.json): sliding layers 34.406 MB,
    // global layers 17.203 MB. A global layer holds 8 KiB per token, so the
    // sequence was 2202 tokens long.
    let seq_len = (17.203 * MIB / (8.0 * KIB)).round() as u64;
    assert_eq!(seq_len, 2202);
    let sliding = m
        .layers
        .iter()
        .find(|l| matches!(l, LayerKind::Sliding { .. }))
        .unwrap();
    // Measured sliding size = all 2202 tokens: the 1024-token window was not applied.
    let untrimmed = seq_len as f64 * 16.0 * KIB / MIB;
    assert_eq!(round_to(untrimmed, 3), 34.406);
    // Per-layer allocation (what the planner reports) holds one window.
    assert_eq!(layer_bytes(sliding, seq_len, BF16), 1024.0 * 16.0 * KIB);
}

#[test]
fn qwen36_35b_density_matches() {
    let m = parse_config(include_str!("fixtures/qwen3.6-35b-a3b.json")).unwrap();
    // Measured 2.00 KB/token/layer over the 10 attention layers
    // (results/h200_v2/qwen36_profile.json; 4.242 MB per layer on prompt 0).
    assert_eq!(density_bytes_per_token_per_layer(&m, BF16), 2.0 * KIB);
    let seq_len = (4.242 * MIB / (2.0 * KIB)).round() as u64;
    let kv = sequence_kv(&m, seq_len, BF16);
    assert_eq!(kv.kv_layers, 10);
    assert_eq!(round_to(kv.per_layer_total / MIB / 10.0, 3), 4.242);
}

#[test]
fn glm_mla_latent_versus_expanded() {
    let m = parse_config(include_str!("fixtures/glm-4.7-flash.json")).unwrap();
    let latent = density_bytes_per_token_per_layer(&m, BF16);
    let expanded = density_bytes_per_token_per_layer(
        &m,
        KvOptions {
            mla: MlaStorage::Expanded,
            ..BF16
        },
    );
    assert_eq!(latent, (512.0 + 64.0) * 2.0); // 1.125 KiB
    assert_eq!(expanded, 20.0 * KIB); // 20 heads x (256 + 256) x 2 bytes
                                      // V1 measured 82.012 MB per layer (results/h100_v1/glm47flash_profile.json).
                                      // Under expanded storage that is a whole number of tokens; prompt length
                                      // was not recorded, so this is a consistency check only.
    let tokens = 82.012 * MIB / expanded;
    assert!((tokens - tokens.round()).abs() < 0.05, "{tokens}");
}

#[test]
fn kv_dtype_scales_linearly() {
    let m = gptoss();
    let bf16 = sequence_kv(&m, 4096, BF16).per_layer_total;
    let fp8 = sequence_kv(
        &m,
        4096,
        KvOptions {
            dtype: KvDtype::Fp8,
            ..BF16
        },
    )
    .per_layer_total;
    assert_eq!(fp8 * 2.0, bf16);
}
