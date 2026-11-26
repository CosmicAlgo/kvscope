//! KVScope core: KV-cache capacity planning from a model's `config.json`.
//!
//! The library does no file or network I/O, so it can also be compiled to
//! WebAssembly. Callers pass in the text of a config file.

pub mod config;
pub mod dtype;
pub mod kv;
pub mod units;

pub use config::{parse_config, ConfigError, KvShape, LayerKind, ModelLayout, NoKvReason};
pub use dtype::KvDtype;
pub use kv::{density_bytes_per_token_per_layer, sequence_kv, KvOptions, MlaStorage, SequenceKv};
