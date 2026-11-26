//! KVScope core: KV-cache capacity planning from a model's `config.json`.
//!
//! The library does no file or network I/O, so it can also be compiled to
//! WebAssembly. Callers pass in the text of a config file.

pub mod config;
pub mod dtype;
pub mod units;

pub use config::{parse_config, ConfigError, KvShape, LayerKind, ModelLayout, NoKvReason};
pub use dtype::KvDtype;
