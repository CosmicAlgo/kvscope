//! KV-cache size arithmetic for one sequence.
//!
//! Two allocation policies are compared:
//! * **per-layer**: each layer holds exactly what it needs; a sliding-window
//!   layer holds `min(seq_len, window)` tokens;
//! * **uniform**: every KV-holding layer is given the size of the largest
//!   layer at `seq_len` tokens, as an allocator with one slot size for all
//!   layers would.
//!
//! Waste = 1 - per_layer_total / uniform_total. For a model with equal
//! numbers of full and sliding layers of the same shape (gpt-oss), this is
//! 1/2 x (1 - window / seq_len).

use crate::config::{LayerKind, ModelLayout};
use crate::dtype::KvDtype;

/// What an MLA layer stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaStorage {
    /// The latent plus the RoPE key part, as MLA-aware engines store it.
    Latent,
    /// Full per-head K and V, as Hugging Face transformers stored it in the
    /// measured GLM-4.7-Flash run.
    Expanded,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvOptions {
    pub dtype: KvDtype,
    pub mla: MlaStorage,
}

/// Values one layer stores per token, ignoring any window cap.
pub fn values_per_token(layer: &LayerKind, mla: MlaStorage) -> u64 {
    match layer {
        LayerKind::Full(shape) | LayerKind::Sliding { shape, .. } => shape.values_per_token(),
        LayerKind::Latent {
            latent_dim,
            expanded,
        } => match mla {
            MlaStorage::Latent => u64::from(*latent_dim),
            MlaStorage::Expanded => expanded.values_per_token(),
        },
        LayerKind::NoKv(_) => 0,
    }
}

/// Tokens a layer holds for a sequence of `seq_len` tokens under per-layer
/// allocation.
pub fn tokens_held(layer: &LayerKind, seq_len: u64) -> u64 {
    match layer {
        LayerKind::Sliding { window, .. } => seq_len.min(*window),
        LayerKind::NoKv(_) => 0,
        _ => seq_len,
    }
}

/// Bytes a layer holds for one sequence under per-layer allocation.
pub fn layer_bytes(layer: &LayerKind, seq_len: u64, opts: KvOptions) -> f64 {
    tokens_held(layer, seq_len) as f64
        * values_per_token(layer, opts.mla) as f64
        * opts.dtype.bytes_per_value()
}

/// KV cache of one sequence, under both allocation policies.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceKv {
    pub seq_len: u64,
    /// Bytes per layer under per-layer allocation (0 for layers without KV).
    pub per_layer: Vec<f64>,
    pub per_layer_total: f64,
    /// Every KV-holding layer sized like the largest one at `seq_len`.
    pub uniform_total: f64,
    pub kv_layers: usize,
}

impl SequenceKv {
    /// Fraction of the uniform allocation that per-layer allocation does not need.
    pub fn waste_fraction(&self) -> f64 {
        if self.uniform_total > 0.0 {
            1.0 - self.per_layer_total / self.uniform_total
        } else {
            0.0
        }
    }
}

pub fn sequence_kv(layout: &ModelLayout, seq_len: u64, opts: KvOptions) -> SequenceKv {
    let per_layer: Vec<f64> = layout
        .layers
        .iter()
        .map(|l| layer_bytes(l, seq_len, opts))
        .collect();
    let per_layer_total = per_layer.iter().sum();
    let largest_per_token = layout
        .layers
        .iter()
        .map(|l| values_per_token(l, opts.mla))
        .max()
        .unwrap_or(0);
    let kv_layers = layout.kv_layer_count();
    let uniform_total =
        kv_layers as f64 * seq_len as f64 * largest_per_token as f64 * opts.dtype.bytes_per_value();
    SequenceKv {
        seq_len,
        per_layer,
        per_layer_total,
        uniform_total,
        kv_layers,
    }
}

/// Average bytes per token per KV-holding layer, with no window cap.
///
/// This is the quantity the Python tracer reports as "KB/token/layer" while
/// the sequence is shorter than every window, and what it reported for
/// Gemma 4, whose sliding layers Hugging Face did not trim.
pub fn density_bytes_per_token_per_layer(layout: &ModelLayout, opts: KvOptions) -> f64 {
    let kv_layers = layout.kv_layer_count();
    if kv_layers == 0 {
        return 0.0;
    }
    let total: u64 = layout
        .layers
        .iter()
        .map(|l| values_per_token(l, opts.mla))
        .sum();
    total as f64 * opts.dtype.bytes_per_value() / kv_layers as f64
}
