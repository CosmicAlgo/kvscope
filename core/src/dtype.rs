//! Data types the KV cache can be stored in, and their size per value.

use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    Bf16,
    F16,
    F32,
    Fp8,
    /// llama.cpp block format: 32 values in 34 bytes (int8 values + fp16 scale).
    Q8_0,
    /// llama.cpp block format: 32 values in 18 bytes (4-bit values + fp16 scale).
    Q4_0,
}

impl KvDtype {
    /// Bytes per stored value. Fractional for block-quantised formats.
    pub fn bytes_per_value(self) -> f64 {
        match self {
            KvDtype::Bf16 | KvDtype::F16 => 2.0,
            KvDtype::F32 => 4.0,
            KvDtype::Fp8 => 1.0,
            KvDtype::Q8_0 => 34.0 / 32.0,
            KvDtype::Q4_0 => 18.0 / 32.0,
        }
    }

    /// Map a config.json `torch_dtype` / `dtype` string to a KV dtype.
    pub fn from_config_name(name: &str) -> Option<KvDtype> {
        match name {
            "bfloat16" => Some(KvDtype::Bf16),
            "float16" => Some(KvDtype::F16),
            "float32" => Some(KvDtype::F32),
            _ => None,
        }
    }
}

impl fmt::Display for KvDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            KvDtype::Bf16 => "bf16",
            KvDtype::F16 => "fp16",
            KvDtype::F32 => "fp32",
            KvDtype::Fp8 => "fp8",
            KvDtype::Q8_0 => "q8_0",
            KvDtype::Q4_0 => "q4_0",
        };
        f.write_str(s)
    }
}

impl FromStr for KvDtype {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(KvDtype::Bf16),
            "fp16" | "f16" | "float16" => Ok(KvDtype::F16),
            "fp32" | "f32" | "float32" => Ok(KvDtype::F32),
            "fp8" => Ok(KvDtype::Fp8),
            "q8_0" => Ok(KvDtype::Q8_0),
            "q4_0" => Ok(KvDtype::Q4_0),
            other => Err(format!(
                "unknown KV dtype '{other}' (expected bf16, fp16, fp32, fp8, q8_0 or q4_0)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(KvDtype::Bf16.bytes_per_value(), 2.0);
        assert_eq!(KvDtype::Q8_0.bytes_per_value(), 1.0625);
        assert_eq!(KvDtype::Q4_0.bytes_per_value(), 0.5625);
    }

    #[test]
    fn parse_round_trip() {
        for d in [
            KvDtype::Bf16,
            KvDtype::F16,
            KvDtype::F32,
            KvDtype::Fp8,
            KvDtype::Q8_0,
            KvDtype::Q4_0,
        ] {
            assert_eq!(d.to_string().parse::<KvDtype>(), Ok(d));
        }
        assert!("int3".parse::<KvDtype>().is_err());
    }
}
