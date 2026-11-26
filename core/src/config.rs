//! Parse a Hugging Face `config.json` into a per-layer description of what
//! each layer keeps in the KV cache.
//!
//! Supported layouts, by the config fields that describe them:
//! * dense attention: `num_key_value_heads`, `head_dim` (MHA when the KV head
//!   count is absent, GQA when it is smaller than `num_attention_heads`);
//! * per-layer types: `layer_types` with `full_attention`,
//!   `sliding_attention` (needs `sliding_window`) or `linear_attention`
//!   (e.g. Gated DeltaNet, which keeps no KV cache);
//! * Gemma-style global layers with their own shape: `global_head_dim`,
//!   `num_global_key_value_heads`;
//! * MLA: `kv_lora_rank` + `qk_rope_head_dim`;
//! * Nemotron-H style hybrids: `hybrid_override_pattern`
//!   (`M` = Mamba, `*` = attention, `-` = MLP);
//! * `num_kv_shared_layers`: trailing layers that reuse earlier layers' KV.
//!
//! Anything else is reported as an error rather than guessed.

use crate::dtype::KvDtype;
use serde_json::{Map, Value};
use std::fmt;

/// Shape of the K and V tensors one attention layer stores per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub kv_heads: u32,
    pub k_head_dim: u32,
    pub v_head_dim: u32,
}

impl KvShape {
    /// Number of values (K plus V) stored per token.
    pub fn values_per_token(&self) -> u64 {
        u64::from(self.kv_heads) * (u64::from(self.k_head_dim) + u64::from(self.v_head_dim))
    }
}

/// Why a layer keeps no KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoKvReason {
    /// Linear attention (e.g. Gated DeltaNet): fixed-size recurrent state.
    LinearAttention,
    /// Mamba / SSM block: fixed-size recurrent state.
    Mamba,
    /// Feed-forward block with no attention.
    Mlp,
    /// Reuses the KV cache of an earlier layer.
    SharedKv,
}

/// What one layer stores in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Attention over the whole sequence.
    Full(KvShape),
    /// Attention over the last `window` tokens only.
    Sliding { shape: KvShape, window: u64 },
    /// Multi-head latent attention. An MLA-aware engine caches `latent_dim`
    /// values per token (latent + RoPE key part); an engine that expands it
    /// caches full K/V with shape `expanded`.
    Latent { latent_dim: u32, expanded: KvShape },
    /// No KV cache.
    NoKv(NoKvReason),
}

impl LayerKind {
    pub fn holds_kv(&self) -> bool {
        !matches!(self, LayerKind::NoKv(_))
    }
}

/// A model as the planner sees it: one entry per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLayout {
    pub model_type: String,
    pub layers: Vec<LayerKind>,
    /// KV dtype implied by the config (`torch_dtype` / `dtype`).
    pub config_dtype: KvDtype,
    /// Things the planner had to assume or that a reader should know.
    pub notes: Vec<String>,
}

impl ModelLayout {
    pub fn kv_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.holds_kv()).count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Json(String),
    NotAnObject,
    Missing(&'static str),
    Invalid { field: &'static str, reason: String },
    Unsupported(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Json(e) => write!(f, "config is not valid JSON: {e}"),
            ConfigError::NotAnObject => write!(f, "config must be a JSON object"),
            ConfigError::Missing(field) => write!(f, "config has no '{field}'"),
            ConfigError::Invalid { field, reason } => write!(f, "invalid '{field}': {reason}"),
            ConfigError::Unsupported(what) => write!(f, "unsupported layout: {what}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Parse the text of a `config.json`.
pub fn parse_config(json: &str) -> Result<ModelLayout, ConfigError> {
    let root: Value = serde_json::from_str(json).map_err(|e| ConfigError::Json(e.to_string()))?;
    let root = root.as_object().ok_or(ConfigError::NotAnObject)?;
    layout_from(&merged(root))
}

/// Multimodal configs keep the language model's fields under `text_config`.
/// Those take precedence over top-level fields.
fn merged(root: &Map<String, Value>) -> Map<String, Value> {
    let mut cfg = root.clone();
    if let Some(Value::Object(text)) = root.get("text_config") {
        for (k, v) in text {
            cfg.insert(k.clone(), v.clone());
        }
    }
    cfg
}

/// A non-negative integer field; `None` if absent or `null`.
fn get_u64(cfg: &Map<String, Value>, field: &'static str) -> Result<Option<u64>, ConfigError> {
    match cfg.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| ConfigError::Invalid {
            field,
            reason: format!("expected a non-negative integer, got {v}"),
        }),
    }
}

fn req_u64(cfg: &Map<String, Value>, field: &'static str) -> Result<u64, ConfigError> {
    get_u64(cfg, field)?.ok_or(ConfigError::Missing(field))
}

fn to_u32(value: u64, field: &'static str) -> Result<u32, ConfigError> {
    u32::try_from(value).map_err(|_| ConfigError::Invalid {
        field,
        reason: format!("{value} is too large"),
    })
}

fn layout_from(cfg: &Map<String, Value>) -> Result<ModelLayout, ConfigError> {
    let mut notes = Vec::new();
    let model_type = cfg
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let n_layers =
        usize::try_from(req_u64(cfg, "num_hidden_layers")?).map_err(|_| ConfigError::Invalid {
            field: "num_hidden_layers",
            reason: "too large".into(),
        })?;
    let n_heads = to_u32(req_u64(cfg, "num_attention_heads")?, "num_attention_heads")?;
    if n_heads == 0 {
        return Err(ConfigError::Invalid {
            field: "num_attention_heads",
            reason: "is 0".into(),
        });
    }
    let kv_heads = match get_u64(cfg, "num_key_value_heads")? {
        Some(h) => to_u32(h, "num_key_value_heads")?,
        None => n_heads, // MHA: one KV head per query head
    };
    let mla_rank = get_u64(cfg, "kv_lora_rank")?;
    // MLA layers take their shape from the MLA fields (see `to_latent`), and
    // hidden_size / num_attention_heads need not be a whole number for them.
    let head_dim = if mla_rank.is_some() {
        0
    } else {
        head_dim(cfg, n_heads)?
    };
    let base = KvShape {
        kv_heads,
        k_head_dim: head_dim,
        v_head_dim: head_dim,
    };

    // Gemma-style full-attention ("global") layers can have their own shape.
    let global_dim = get_u64(cfg, "global_head_dim")?;
    let global_kv = get_u64(cfg, "num_global_key_value_heads")?;
    let full_shape = if global_dim.is_some() || global_kv.is_some() {
        let d = match global_dim {
            Some(d) => to_u32(d, "global_head_dim")?,
            None => head_dim,
        };
        let h = match global_kv {
            Some(h) => to_u32(h, "num_global_key_value_heads")?,
            None => kv_heads,
        };
        KvShape {
            kv_heads: h,
            k_head_dim: d,
            v_head_dim: d,
        }
    } else {
        base
    };
    if cfg.get("attention_k_eq_v").and_then(Value::as_bool) == Some(true) {
        notes.push(
            "config sets attention_k_eq_v: K and V are counted separately, as Hugging Face \
             transformers stored them in the measured Gemma 4 runs; an engine that stores \
             them once would need half for those layers"
                .into(),
        );
    }

    let sliding_window = get_u64(cfg, "sliding_window")?;
    let mut layers = if let Some(pattern) = cfg.get("hybrid_override_pattern") {
        let pattern = pattern.as_str().ok_or(ConfigError::Invalid {
            field: "hybrid_override_pattern",
            reason: "expected a string".into(),
        })?;
        from_pattern(pattern, full_shape)?
    } else if let Some(types) = cfg.get("layer_types") {
        from_layer_types(types, base, full_shape, sliding_window)?
    } else if let (Some(window), false) = (
        sliding_window,
        cfg.get("use_sliding_window") == Some(&Value::Bool(false)),
    ) {
        notes.push(format!(
            "no layer_types: every layer treated as sliding-window ({window} tokens)"
        ));
        vec![
            LayerKind::Sliding {
                shape: base,
                window
            };
            n_layers
        ]
    } else {
        vec![LayerKind::Full(full_shape); n_layers]
    };

    if layers.len() != n_layers {
        return Err(ConfigError::Invalid {
            field: "num_hidden_layers",
            reason: format!(
                "is {n_layers}, but the per-layer description has {}",
                layers.len()
            ),
        });
    }

    if let Some(rank) = mla_rank {
        layers = to_latent(cfg, layers, rank, n_heads)?;
        notes.push(
            "MLA: an MLA-aware engine caches the latent plus the RoPE key part; Hugging Face \
             transformers cached expanded K/V in the measured GLM-4.7-Flash run"
                .into(),
        );
    }

    if let Some(shared) = get_u64(cfg, "num_kv_shared_layers")? {
        let shared = usize::try_from(shared).unwrap_or(usize::MAX).min(n_layers);
        for layer in layers.iter_mut().skip(n_layers - shared) {
            *layer = LayerKind::NoKv(NoKvReason::SharedKv);
        }
    }

    let dtype_name = cfg
        .get("torch_dtype")
        .or_else(|| cfg.get("dtype"))
        .and_then(Value::as_str);
    let config_dtype = match dtype_name.and_then(KvDtype::from_config_name) {
        Some(d) => d,
        None => {
            notes.push("config gives no dtype: assuming a bf16 KV cache".into());
            KvDtype::Bf16
        }
    };

    if layers.iter().any(|l| {
        matches!(
            l,
            LayerKind::NoKv(NoKvReason::LinearAttention | NoKvReason::Mamba)
        )
    }) {
        notes.push(
            "linear-attention / Mamba layers keep a fixed-size recurrent state per sequence; \
             it is not KV cache and is not counted"
                .into(),
        );
    }

    Ok(ModelLayout {
        model_type,
        layers,
        config_dtype,
        notes,
    })
}

fn head_dim(cfg: &Map<String, Value>, n_heads: u32) -> Result<u32, ConfigError> {
    if let Some(d) = get_u64(cfg, "head_dim")? {
        return to_u32(d, "head_dim");
    }
    if let Some(d) = get_u64(cfg, "attention_head_dim")? {
        return to_u32(d, "attention_head_dim");
    }
    let hidden = req_u64(cfg, "hidden_size")?;
    if hidden % u64::from(n_heads) != 0 {
        return Err(ConfigError::Invalid {
            field: "hidden_size",
            reason: format!("{hidden} is not divisible by num_attention_heads ({n_heads})"),
        });
    }
    to_u32(hidden / u64::from(n_heads), "hidden_size")
}

fn from_layer_types(
    types: &Value,
    sliding_shape: KvShape,
    full_shape: KvShape,
    sliding_window: Option<u64>,
) -> Result<Vec<LayerKind>, ConfigError> {
    let invalid = |reason: String| ConfigError::Invalid {
        field: "layer_types",
        reason,
    };
    let types = types
        .as_array()
        .ok_or_else(|| invalid("expected a list".into()))?;
    types
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => Ok(LayerKind::Full(full_shape)),
            Some("sliding_attention") => {
                let window = sliding_window.ok_or(ConfigError::Missing("sliding_window"))?;
                Ok(LayerKind::Sliding {
                    shape: sliding_shape,
                    window,
                })
            }
            Some("linear_attention") => Ok(LayerKind::NoKv(NoKvReason::LinearAttention)),
            Some(other) => Err(ConfigError::Unsupported(format!("layer type '{other}'"))),
            None => Err(invalid(format!("expected strings, got {t}"))),
        })
        .collect()
}

fn from_pattern(pattern: &str, attn_shape: KvShape) -> Result<Vec<LayerKind>, ConfigError> {
    pattern
        .chars()
        .map(|c| match c {
            '*' => Ok(LayerKind::Full(attn_shape)),
            'M' => Ok(LayerKind::NoKv(NoKvReason::Mamba)),
            '-' => Ok(LayerKind::NoKv(NoKvReason::Mlp)),
            other => Err(ConfigError::Unsupported(format!(
                "hybrid_override_pattern symbol '{other}'"
            ))),
        })
        .collect()
}

fn to_latent(
    cfg: &Map<String, Value>,
    layers: Vec<LayerKind>,
    rank: u64,
    n_heads: u32,
) -> Result<Vec<LayerKind>, ConfigError> {
    let rope = get_u64(cfg, "qk_rope_head_dim")?.unwrap_or(0);
    let nope = get_u64(cfg, "qk_nope_head_dim")?.ok_or(ConfigError::Missing("qk_nope_head_dim"))?;
    let v_dim = get_u64(cfg, "v_head_dim")?.ok_or(ConfigError::Missing("v_head_dim"))?;
    let latent_dim = to_u32(rank + rope, "kv_lora_rank")?;
    let expanded = KvShape {
        kv_heads: n_heads,
        k_head_dim: to_u32(nope + rope, "qk_nope_head_dim")?,
        v_head_dim: to_u32(v_dim, "v_head_dim")?,
    };
    layers
        .into_iter()
        .map(|l| match l {
            LayerKind::Full(_) => Ok(LayerKind::Latent {
                latent_dim,
                expanded,
            }),
            LayerKind::NoKv(r) => Ok(LayerKind::NoKv(r)),
            _ => Err(ConfigError::Unsupported(
                "MLA combined with sliding-window layers".into(),
            )),
        })
        .collect()
}
