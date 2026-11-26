//! Values measured by the KVScope Python tracer, embedded from
//! `data/measurements.json`. Each names its GPU, date, engine and source file.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Measurement {
    pub models: Vec<String>,
    pub checkpoint: String,
    pub quantity: String,
    pub value: f64,
    pub gpu: String,
    pub date: String,
    pub engine: String,
    pub source: String,
    pub path: String,
}

#[derive(Deserialize)]
struct Table {
    measurements: Vec<Measurement>,
}

/// All embedded measurements.
pub fn all() -> Vec<Measurement> {
    let table: Table = serde_json::from_str(include_str!("../data/measurements.json"))
        .expect("data/measurements.json is checked by the test suite");
    table.measurements
}

/// Measurements for one Hugging Face model id (case-insensitive).
pub fn for_model(model_id: &str) -> Vec<Measurement> {
    all()
        .into_iter()
        .filter(|m| m.models.iter().any(|id| id.eq_ignore_ascii_case(model_id)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_parses_and_lookup_works() {
        assert!(all().len() >= 8);
        assert_eq!(for_model("openai/GPT-OSS-120B").len(), 3);
        assert!(for_model("nobody/nothing").is_empty());
    }
}
