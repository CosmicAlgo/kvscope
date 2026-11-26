//! Engine settings and a Slurm batch script for a plan.
//!
//! Flags were checked against each engine's documentation (November 2026):
//! vLLM engine arguments, llama.cpp `tools/server/README.md`, the Ollama FAQ
//! and SGLang server arguments. Where a setting cannot be expressed, the
//! output says so instead of guessing.

use crate::dtype::KvDtype;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchSpec {
    /// Hugging Face model id, or a placeholder.
    pub model: String,
    pub seq_len: u64,
    pub users: u64,
    pub gpu_count: u32,
    pub kv_dtype: KvDtype,
    pub utilisation: f64,
    /// The model has sliding-window layers.
    pub has_sliding: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineCommand {
    pub engine: &'static str,
    pub command: String,
    pub notes: Vec<String>,
}

pub fn vllm(s: &LaunchSpec) -> EngineCommand {
    let mut cmd = format!(
        "vllm serve {} --max-model-len {} --max-num-seqs {} --gpu-memory-utilization {:.2}",
        s.model, s.seq_len, s.users, s.utilisation
    );
    let mut notes = Vec::new();
    if s.gpu_count > 1 {
        cmd.push_str(&format!(" --tensor-parallel-size {}", s.gpu_count));
    }
    match s.kv_dtype {
        KvDtype::Bf16 | KvDtype::F16 => {}
        KvDtype::Fp8 => cmd.push_str(" --kv-cache-dtype fp8"),
        other => notes.push(unsupported("vLLM", other)),
    }
    if s.has_sliding {
        notes.push(
            "vLLM's hybrid KV cache manager frees sliding-window blocks outside the window but \
             pads layer groups, so its usage falls between the per-layer and uniform figures"
                .into(),
        );
    }
    EngineCommand {
        engine: "vllm",
        command: cmd,
        notes,
    }
}

pub fn sglang(s: &LaunchSpec) -> EngineCommand {
    let mut cmd = format!(
        "python -m sglang.launch_server --model-path {} --context-length {} \
         --max-running-requests {} --mem-fraction-static {:.2}",
        s.model, s.seq_len, s.users, s.utilisation
    );
    let mut notes = Vec::new();
    match s.kv_dtype {
        KvDtype::Bf16 | KvDtype::F16 => {}
        KvDtype::Fp8 => cmd.push_str(" --kv-cache-dtype fp8_e4m3"),
        other => notes.push(unsupported("SGLang", other)),
    }
    if s.gpu_count > 1 {
        notes.push(format!(
            "set tensor parallelism for {} GPUs yourself; KVScope has not checked that flag",
            s.gpu_count
        ));
    }
    notes.push(
        "--mem-fraction-static covers weights and the KV pool, like the planner's utilisation"
            .into(),
    );
    EngineCommand {
        engine: "sglang",
        command: cmd,
        notes,
    }
}

pub fn llama_cpp(s: &LaunchSpec) -> EngineCommand {
    let total_ctx = s.seq_len.saturating_mul(s.users);
    let mut cmd = format!(
        "llama-server -m {}.gguf -c {} -np {} -ngl all -fa on",
        gguf_name(&s.model),
        total_ctx,
        s.users
    );
    let mut notes = vec![format!(
        "-c is the context of all {} slots together ({} x {}); per the server docs the unified \
         KV buffer is only on by default when -np is auto",
        s.users, s.seq_len, s.users
    )];
    let kv = match s.kv_dtype {
        KvDtype::F16 => None,
        KvDtype::Bf16 => Some("bf16"),
        KvDtype::F32 => Some("f32"),
        KvDtype::Fp8 => Some("f8_e4m3"),
        KvDtype::Q8_0 => Some("q8_0"),
        KvDtype::Q4_0 => Some("q4_0"),
    };
    if let Some(t) = kv {
        cmd.push_str(&format!(" -ctk {t} -ctv {t}"));
    }
    if s.gpu_count > 1 {
        notes.push("llama.cpp splits layers across all visible GPUs by default".into());
    }
    notes.push("needs a GGUF file of the model; weight memory depends on its quantisation".into());
    EngineCommand {
        engine: "llama.cpp",
        command: cmd,
        notes,
    }
}

pub fn ollama(s: &LaunchSpec) -> EngineCommand {
    let mut env = format!(
        "OLLAMA_CONTEXT_LENGTH={} OLLAMA_NUM_PARALLEL={}",
        s.seq_len, s.users
    );
    let mut notes = vec![
        "Ollama's context length is per request; parallel requests multiply the KV memory".into(),
    ];
    match s.kv_dtype {
        KvDtype::F16 | KvDtype::Bf16 => {}
        KvDtype::Q8_0 => env.push_str(" OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1"),
        KvDtype::Q4_0 => env.push_str(" OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_FLASH_ATTENTION=1"),
        other => notes.push(format!(
            "Ollama's KV cache types are f16, q8_0 and q4_0; {other} is not available, so it \
             would use f16, which may need more memory than this plan assumes"
        )),
    }
    EngineCommand {
        engine: "ollama",
        command: format!("{env} ollama serve"),
        notes,
    }
}

pub fn all(s: &LaunchSpec) -> Vec<EngineCommand> {
    vec![vllm(s), sglang(s), llama_cpp(s), ollama(s)]
}

fn unsupported(engine: &str, dtype: KvDtype) -> String {
    format!(
        "{engine} has no {dtype} KV cache; left at its default, which may need more memory \
         than this plan assumes"
    )
}

fn gguf_name(model: &str) -> String {
    model.rsplit('/').next().unwrap_or(model).to_string()
}

/// A Slurm batch script that starts vLLM with the plan's settings.
/// `summary` is placed in a comment at the top.
pub fn sbatch_vllm(s: &LaunchSpec, summary: &str) -> String {
    let vllm = vllm(s);
    let summary: String = summary.lines().map(|l| format!("# {l}\n")).collect();
    format!(
        "#!/bin/bash\n\
         # Generated by `kvscope plan --sbatch`. This is a template: set the partition,\n\
         # account, modules and GPU type for your cluster.\n\
         {summary}\
         # Many sites need a GPU type in --gres, e.g. gpu:h100:{gpus}. The CPU, host\n\
         # memory and time limits below are template defaults, not part of the plan.\n\
         #SBATCH --job-name=kvscope-vllm\n\
         #SBATCH --nodes=1\n\
         #SBATCH --gres=gpu:{gpus}\n\
         #SBATCH --cpus-per-task=8\n\
         #SBATCH --mem=64G\n\
         #SBATCH --time=02:00:00\n\
         \n\
         set -euo pipefail\n\
         # module load cuda               # site-specific\n\
         \n\
         {cmd} --port 8000\n",
        gpus = s.gpu_count,
        cmd = vllm.command,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dtype: KvDtype, gpus: u32) -> LaunchSpec {
        LaunchSpec {
            model: "openai/gpt-oss-120b".into(),
            seq_len: 32768,
            users: 16,
            gpu_count: gpus,
            kv_dtype: dtype,
            utilisation: 0.9,
            has_sliding: true,
        }
    }

    #[test]
    fn vllm_flags() {
        let c = vllm(&spec(KvDtype::Bf16, 1));
        assert_eq!(
            c.command,
            "vllm serve openai/gpt-oss-120b --max-model-len 32768 --max-num-seqs 16 \
             --gpu-memory-utilization 0.90"
        );
        let c = vllm(&spec(KvDtype::Fp8, 4));
        assert!(c
            .command
            .ends_with("--tensor-parallel-size 4 --kv-cache-dtype fp8"));
        assert!(vllm(&spec(KvDtype::Q4_0, 1)).notes[0].contains("no q4_0"));
        let no_sliding = LaunchSpec {
            has_sliding: false,
            ..spec(KvDtype::Bf16, 1)
        };
        assert!(vllm(&no_sliding).notes.is_empty());
    }

    #[test]
    fn llama_cpp_context_is_total_across_slots() {
        let c = llama_cpp(&spec(KvDtype::Q8_0, 1));
        assert_eq!(
            c.command,
            "llama-server -m gpt-oss-120b.gguf -c 524288 -np 16 -ngl all -fa on -ctk q8_0 -ctv q8_0"
        );
    }

    #[test]
    fn ollama_env() {
        let c = ollama(&spec(KvDtype::Q8_0, 1));
        assert_eq!(
            c.command,
            "OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=16 OLLAMA_KV_CACHE_TYPE=q8_0 \
             OLLAMA_FLASH_ATTENTION=1 ollama serve"
        );
        assert!(ollama(&spec(KvDtype::Fp8, 1)).notes[1].contains("not available"));
    }

    #[test]
    fn sglang_flags() {
        let c = sglang(&spec(KvDtype::Fp8, 1));
        assert!(c
            .command
            .contains("--context-length 32768 --max-running-requests 16"));
        assert!(c.command.ends_with("--kv-cache-dtype fp8_e4m3"));
    }

    #[test]
    fn sbatch_has_directives_and_command() {
        let script = sbatch_vllm(&spec(KvDtype::Bf16, 2), "line one\nline two");
        assert!(script.starts_with("#!/bin/bash\n"));
        assert!(script.contains("# line one\n# line two\n"));
        assert!(script.contains("#SBATCH --gres=gpu:2\n"));
        // No trailing comments on #SBATCH lines: put comments on their own lines.
        assert!(script
            .lines()
            .filter(|l| l.starts_with("#SBATCH"))
            .all(|l| !l.contains(" #")));
        assert!(script.contains("--tensor-parallel-size 2 --port 8000\n"));
        // Every #SBATCH line must come before the first command for Slurm to read it.
        let first_cmd = script.find("set -euo").unwrap();
        assert!(script.rfind("#SBATCH").unwrap() < first_cmd);
    }
}
