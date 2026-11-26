//! `kvscope` command-line program.

use clap::{Parser, Subcommand, ValueEnum};
use kvscope_core::gpu::{self, CATALOG};
use kvscope_core::plan::{self, Basis, ByPolicy, Labelled, Plan, PlanRequest};
use kvscope_core::units::{fmt_bytes, GIB, KIB};
use kvscope_core::{parse_config, KvDtype, KvOptions, MlaStorage};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "kvscope",
    version,
    about = "KV-cache capacity planning for LLM inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// KV memory for a workload, and how many requests fit on a GPU.
    Plan(PlanArgs),
    /// List the GPUs the planner knows, with the source of each memory size.
    Gpus,
}

#[derive(clap::Args)]
struct PlanArgs {
    /// Path to the model's config.json.
    #[arg(long)]
    config: std::path::PathBuf,
    /// Hugging Face model id, used to look up measured values (e.g. openai/gpt-oss-120b).
    #[arg(long)]
    model: Option<String>,
    /// Tokens per request, prompt plus output (e.g. 4096, 32k).
    #[arg(long, value_parser = plan::parse_tokens)]
    context: u64,
    /// Concurrent requests.
    #[arg(long, default_value_t = 1)]
    users: u64,
    /// GPU name; see `kvscope gpus`.
    #[arg(long, default_value = "H100-80GB")]
    gpu: String,
    /// GPU memory in GiB, instead of a catalog entry.
    #[arg(long)]
    gpu_mem_gib: Option<f64>,
    /// Number of GPUs (memory is pooled, e.g. tensor parallelism).
    #[arg(long, default_value_t = 1)]
    gpus: u32,
    /// KV-cache dtype. Default: the config's dtype (bf16 if it has none).
    #[arg(long)]
    kv_dtype: Option<KvDtype>,
    /// How MLA layers are stored.
    #[arg(long, value_enum, default_value_t = Mla::Latent)]
    mla: Mla,
    /// Model weight memory in GiB. Needed to say how many requests fit.
    #[arg(long)]
    weights_gib: Option<f64>,
    /// Memory per GPU for the CUDA context and activations, in GiB.
    #[arg(long)]
    overhead_gib: Option<f64>,
    /// Fraction of GPU memory the engine may use.
    #[arg(long)]
    utilisation: Option<f64>,
    /// Print the plan as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mla {
    /// Latent + RoPE key part (MLA-aware engines).
    Latent,
    /// Full per-head K/V (as Hugging Face transformers stored it).
    Expanded,
}

const DEFAULT_OVERHEAD_GIB: f64 = 1.0;
const DEFAULT_UTILISATION: f64 = 0.90;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Plan(args) => run_plan(args),
        Command::Gpus => {
            for g in CATALOG {
                println!(
                    "{:<13} {:>10}   {}",
                    g.name,
                    fmt_bytes(g.memory_bytes),
                    g.source
                );
            }
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kvscope: {e}");
            ExitCode::from(2)
        }
    }
}

fn run_plan(args: PlanArgs) -> Result<(), String> {
    let text = std::fs::read_to_string(&args.config)
        .map_err(|e| format!("cannot read {}: {e}", args.config.display()))?;
    let layout = parse_config(&text).map_err(|e| format!("{}: {e}", args.config.display()))?;

    let gpu = match args.gpu_mem_gib {
        Some(gib) if gib > 0.0 => gpu::Gpu {
            name: "custom",
            memory_bytes: gib * GIB,
            source: "input: --gpu-mem-gib",
        },
        Some(_) => return Err("--gpu-mem-gib must be positive".into()),
        None => gpu::find(&args.gpu)
            .ok_or_else(|| format!("unknown GPU '{}'; see `kvscope gpus`", args.gpu))?,
    };
    if args.gpus == 0 {
        return Err("--gpus must be at least 1".into());
    }
    let (dtype, dtype_basis) = match args.kv_dtype {
        Some(d) => (d, Basis::Input),
        None if layout.notes.iter().any(|n| n.contains("assuming a bf16")) => {
            (layout.config_dtype, Basis::Assumed)
        }
        None => (layout.config_dtype, Basis::Calculated),
    };
    let utilisation = match args.utilisation {
        Some(u) if u > 0.0 && u <= 1.0 => Labelled::new(u, Basis::Input),
        Some(_) => return Err("--utilisation must be in (0, 1]".into()),
        None => Labelled::new(DEFAULT_UTILISATION, Basis::Assumed),
    };
    let overhead = match args.overhead_gib {
        Some(g) if g >= 0.0 => Labelled::new(g * GIB, Basis::Input),
        Some(_) => return Err("--overhead-gib cannot be negative".into()),
        None => Labelled::new(DEFAULT_OVERHEAD_GIB * GIB, Basis::Assumed),
    };
    let request = PlanRequest {
        model_id: args.model.clone(),
        seq_len: args.context,
        users: args.users,
        gpu,
        gpu_count: args.gpus,
        kv: KvOptions {
            dtype,
            mla: match args.mla {
                Mla::Latent => MlaStorage::Latent,
                Mla::Expanded => MlaStorage::Expanded,
            },
        },
        kv_dtype_basis: dtype_basis,
        weights_bytes: args.weights_gib.map(|g| g * GIB),
        overhead_bytes_per_gpu: overhead,
        utilisation,
    };
    let p = plan::plan(&layout, &request);
    if args.json {
        let json = serde_json::to_string_pretty(&p).map_err(|e| e.to_string())?;
        println!("{json}");
    } else {
        print!("{}", render(&p));
    }
    Ok(())
}

fn tag(basis: Basis) -> &'static str {
    match basis {
        Basis::Calculated => "[calculated]",
        Basis::Measured => "[measured]",
        Basis::Assumed => "[assumed]",
        Basis::Input => "[input]",
    }
}

fn row(out: &mut String, label: &str, v: ByPolicy<String>, basis: Basis) {
    out.push_str(&format!(
        "  {label:<26} {:>14} {:>14}   {}\n",
        v.per_layer,
        v.uniform,
        tag(basis)
    ));
}

fn render(p: &Plan) -> String {
    let mut out = String::new();
    let l = &p.layers;
    out.push_str(&format!(
        "kvscope plan: {} ({})\n",
        p.model_id.as_deref().unwrap_or("model from --config"),
        p.model_type
    ));
    let mut kinds = Vec::new();
    if l.full > 0 {
        kinds.push(format!("{} full", l.full));
    }
    if l.sliding > 0 {
        let w: Vec<String> = l.windows.iter().map(u64::to_string).collect();
        kinds.push(format!("{} sliding (window {})", l.sliding, w.join("/")));
    }
    if l.latent > 0 {
        kinds.push(format!("{} MLA ({})", l.latent, p.mla_storage));
    }
    if l.no_kv > 0 {
        kinds.push(format!("{} without KV", l.no_kv));
    }
    out.push_str(&format!("  layers: {} = {}\n", l.total, kinds.join(" + ")));
    out.push_str(&format!(
        "  KV dtype: {} {}\n",
        p.kv_dtype.value,
        tag(p.kv_dtype.basis)
    ));
    out.push_str(&format!(
        "  workload: {} tokens per request x {} requests [input]\n",
        p.seq_len, p.users
    ));
    out.push_str(&format!(
        "  GPU: {} x {} = {} {} ({})\n\n",
        p.gpu_count,
        p.gpu.name,
        fmt_bytes(p.gpu_memory_bytes.value),
        tag(p.gpu_memory_bytes.basis),
        p.gpu.source
    ));

    out.push_str(&format!(
        "  KV per token per KV layer: {:.3} KiB (no window cap) {}\n\n",
        p.density_bytes_per_token_per_layer.value / KIB,
        tag(p.density_bytes_per_token_per_layer.basis)
    ));
    out.push_str(&format!(
        "  {:<26} {:>14} {:>14}\n",
        "KV cache", "per-layer", "uniform"
    ));
    let bytes = |b: ByPolicy<f64>| ByPolicy {
        per_layer: fmt_bytes(b.per_layer),
        uniform: fmt_bytes(b.uniform),
    };
    row(
        &mut out,
        "one request",
        bytes(p.per_request_bytes),
        Basis::Calculated,
    );
    row(
        &mut out,
        &format!("{} requests", p.users),
        bytes(p.all_users_bytes),
        Basis::Calculated,
    );
    out.push_str(&format!(
        "  uniform allocation wastes {:.1}% of its KV memory {}\n\n",
        p.uniform_waste_fraction.value * 100.0,
        tag(p.uniform_waste_fraction.basis)
    ));

    match &p.fit {
        Some(f) => {
            out.push_str(&format!(
                "  Fit: {} x {:.2} - weights {} {} - overhead {} {}\n",
                fmt_bytes(p.gpu_memory_bytes.value),
                f.utilisation.value,
                fmt_bytes(f.weights_bytes.value),
                tag(f.weights_bytes.basis),
                fmt_bytes(f.overhead_bytes.value),
                tag(f.overhead_bytes.basis)
            ));
            out.push_str(&format!(
                "      utilisation {:.2} {}\n",
                f.utilisation.value,
                tag(f.utilisation.basis)
            ));
            if f.kv_budget_bytes.value <= 0.0 {
                out.push_str(
                    "  Nothing fits: the weights and overhead use all available memory.\n\n",
                );
            } else {
                out.push_str(&format!(
                    "  memory left for KV: {} {}\n",
                    fmt_bytes(f.kv_budget_bytes.value),
                    tag(f.kv_budget_bytes.basis)
                ));
                let n = |b: ByPolicy<u64>| ByPolicy {
                    per_layer: b.per_layer.to_string(),
                    uniform: b.uniform.to_string(),
                };
                row(
                    &mut out,
                    "requests that fit",
                    n(f.max_requests),
                    Basis::Calculated,
                );
                let yn = |b: bool| {
                    if b {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    }
                };
                row(
                    &mut out,
                    &format!("{} requests fit?", p.users),
                    ByPolicy {
                        per_layer: yn(f.users_fit.per_layer),
                        uniform: yn(f.users_fit.uniform),
                    },
                    Basis::Calculated,
                );
                out.push('\n');
            }
        }
        None => out.push_str("  Fit: give --weights-gib to see how many requests fit.\n\n"),
    }

    if !p.measured.is_empty() {
        out.push_str(
            "  Measured by KVScope for this model (Hugging Face transformers, batch 1):\n",
        );
        for m in &p.measured {
            out.push_str(&format!(
                "    {} {}: {}, {}, {} [measured; {}]\n",
                m.value, m.quantity, m.gpu, m.date, m.engine, m.source
            ));
        }
        out.push('\n');
    }
    if !p.notes.is_empty() {
        out.push_str("  Notes:\n");
        for n in &p.notes {
            out.push_str(&format!("    - {n}\n"));
        }
    }
    out
}
