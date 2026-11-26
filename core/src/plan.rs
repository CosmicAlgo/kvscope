//! `kvscope plan`: KV memory for a workload, and how many requests fit on a GPU.
//!
//! Every number carries its basis:
//! * `calculated`: derived here from config.json and the inputs;
//! * `measured`: read from a KVScope run (GPU, date and source file given);
//! * `assumed`: a default the user did not set (e.g. a nominal GPU size);
//! * `input`: given by the user.

use crate::config::{LayerKind, ModelLayout};
use crate::engines::{self, EngineCommand, LaunchSpec};
use crate::gpu::Gpu;
use crate::kv::{density_bytes_per_token_per_layer, sequence_kv, KvOptions};
use crate::measured::{self, Measurement};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Basis {
    Calculated,
    Measured,
    Assumed,
    Input,
}

/// A value together with where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Labelled<T> {
    pub value: T,
    pub basis: Basis,
}

impl<T> Labelled<T> {
    pub fn new(value: T, basis: Basis) -> Self {
        Labelled { value, basis }
    }
}

/// The same quantity under both allocation policies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ByPolicy<T> {
    pub per_layer: T,
    pub uniform: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanRequest {
    pub model_id: Option<String>,
    pub seq_len: u64,
    pub users: u64,
    pub gpu: Gpu,
    pub gpu_count: u32,
    pub kv: KvOptions,
    pub kv_dtype_basis: Basis,
    pub weights_bytes: Option<f64>,
    pub overhead_bytes_per_gpu: Labelled<f64>,
    pub utilisation: Labelled<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerSummary {
    pub total: usize,
    pub full: usize,
    pub sliding: usize,
    pub latent: usize,
    pub no_kv: usize,
    /// Distinct sliding-window sizes, in tokens.
    pub windows: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fit {
    pub weights_bytes: Labelled<f64>,
    pub overhead_bytes: Labelled<f64>,
    pub utilisation: Labelled<f64>,
    /// GPU memory x utilisation - weights - overhead.
    pub kv_budget_bytes: Labelled<f64>,
    pub max_requests: ByPolicy<u64>,
    pub users_fit: ByPolicy<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Plan {
    pub model_id: Option<String>,
    pub model_type: String,
    pub layers: LayerSummary,
    pub seq_len: u64,
    pub users: u64,
    pub kv_dtype: Labelled<String>,
    pub mla_storage: String,
    /// Bytes per token per KV-holding layer, ignoring window caps.
    pub density_bytes_per_token_per_layer: Labelled<f64>,
    pub per_request_bytes: ByPolicy<f64>,
    pub all_users_bytes: ByPolicy<f64>,
    /// 1 - per_layer / uniform.
    pub uniform_waste_fraction: Labelled<f64>,
    pub gpu: Gpu,
    pub gpu_count: u32,
    pub gpu_memory_bytes: Labelled<f64>,
    pub fit: Option<Fit>,
    pub measured: Vec<Measurement>,
    /// Settings that run this plan on each engine.
    pub engines: Vec<EngineCommand>,
    pub notes: Vec<String>,
}

pub fn summarize_layers(layout: &ModelLayout) -> LayerSummary {
    let mut s = LayerSummary {
        total: layout.layers.len(),
        full: 0,
        sliding: 0,
        latent: 0,
        no_kv: 0,
        windows: Vec::new(),
    };
    for layer in &layout.layers {
        match layer {
            LayerKind::Full(_) => s.full += 1,
            LayerKind::Sliding { window, .. } => {
                s.sliding += 1;
                if !s.windows.contains(window) {
                    s.windows.push(*window);
                }
            }
            LayerKind::Latent { .. } => s.latent += 1,
            LayerKind::NoKv(_) => s.no_kv += 1,
        }
    }
    s
}

pub fn plan(layout: &ModelLayout, req: &PlanRequest) -> Plan {
    let calc = |v| Labelled::new(v, Basis::Calculated);
    let one = sequence_kv(layout, req.seq_len, req.kv);
    let per_request = ByPolicy {
        per_layer: one.per_layer_total,
        uniform: one.uniform_total,
    };
    let users = req.users as f64;
    let all_users = ByPolicy {
        per_layer: per_request.per_layer * users,
        uniform: per_request.uniform * users,
    };

    let gpu_basis = if req.gpu.source.starts_with("measured") {
        Basis::Measured
    } else {
        Basis::Assumed
    };
    let gpu_memory = req.gpu.memory_bytes * f64::from(req.gpu_count);

    let fit = req.weights_bytes.map(|weights| {
        let overhead = req.overhead_bytes_per_gpu.value * f64::from(req.gpu_count);
        let budget = gpu_memory * req.utilisation.value - weights - overhead;
        let max = |per: f64| {
            if budget > 0.0 && per > 0.0 {
                (budget / per).floor() as u64
            } else {
                0
            }
        };
        let max_requests = ByPolicy {
            per_layer: max(per_request.per_layer),
            uniform: max(per_request.uniform),
        };
        Fit {
            weights_bytes: Labelled::new(weights, Basis::Input),
            overhead_bytes: Labelled::new(overhead, req.overhead_bytes_per_gpu.basis),
            utilisation: req.utilisation,
            kv_budget_bytes: calc(budget),
            users_fit: ByPolicy {
                per_layer: max_requests.per_layer >= req.users,
                uniform: max_requests.uniform >= req.users,
            },
            max_requests,
        }
    });

    let mut notes = layout.notes.clone();
    if layout
        .layers
        .iter()
        .any(|l| matches!(l, LayerKind::Sliding { .. }))
    {
        notes.push(
            "per-layer allocation gives a sliding-window layer one window of tokens. Whether an \
             engine does this varies: in the measured runs Hugging Face transformers kept 127 \
             tokens in gpt-oss's 128-token sliding layers but all tokens in Gemma 4's"
                .into(),
        );
    }

    Plan {
        model_id: req.model_id.clone(),
        model_type: layout.model_type.clone(),
        layers: summarize_layers(layout),
        seq_len: req.seq_len,
        users: req.users,
        kv_dtype: Labelled::new(req.kv.dtype.to_string(), req.kv_dtype_basis),
        mla_storage: format!("{:?}", req.kv.mla).to_lowercase(),
        density_bytes_per_token_per_layer: calc(density_bytes_per_token_per_layer(layout, req.kv)),
        per_request_bytes: per_request,
        all_users_bytes: all_users,
        uniform_waste_fraction: calc(one.waste_fraction()),
        gpu: req.gpu,
        gpu_count: req.gpu_count,
        gpu_memory_bytes: Labelled::new(gpu_memory, gpu_basis),
        fit,
        measured: req
            .model_id
            .as_deref()
            .map(measured::for_model)
            .unwrap_or_default(),
        engines: engines::all(&launch_spec(layout, req)),
        notes,
    }
}

/// Engine settings for a request.
pub fn launch_spec(layout: &ModelLayout, req: &PlanRequest) -> LaunchSpec {
    LaunchSpec {
        model: req.model_id.clone().unwrap_or_else(|| "<model>".into()),
        seq_len: req.seq_len,
        users: req.users,
        gpu_count: req.gpu_count,
        kv_dtype: req.kv.dtype,
        utilisation: req.utilisation.value,
        has_sliding: layout
            .layers
            .iter()
            .any(|l| matches!(l, LayerKind::Sliding { .. })),
    }
}

/// Parse a context length such as `32768`, `32k` or `1m` (k = 1024, m = 1024²).
pub fn parse_tokens(s: &str) -> Result<u64, String> {
    let lower = s.trim().to_ascii_lowercase();
    let (digits, scale) = match lower.strip_suffix('k') {
        Some(d) => (d, 1024),
        None => match lower.strip_suffix('m') {
            Some(d) => (d, 1024 * 1024),
            None => (lower.as_str(), 1),
        },
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("'{s}' is not a token count (examples: 4096, 32k, 1m)"))?;
    n.checked_mul(scale)
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("'{s}' is not a usable token count"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts() {
        assert_eq!(parse_tokens("4096"), Ok(4096));
        assert_eq!(parse_tokens("32k"), Ok(32 * 1024));
        assert_eq!(parse_tokens("1M"), Ok(1024 * 1024));
        assert!(parse_tokens("0").is_err());
        assert!(parse_tokens("lots").is_err());
    }
}
