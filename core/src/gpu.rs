//! GPU memory sizes. Each entry says where its number comes from: measured
//! (nvidia-smi output saved with a KVScope run) or nominal (the spec-sheet
//! size taken as GiB; usable memory is usually somewhat lower).

use crate::units::MIB;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Gpu {
    pub name: &'static str,
    pub memory_bytes: f64,
    /// "measured: <file>" or "nominal".
    pub source: &'static str,
}

const fn mib(n: f64) -> f64 {
    n * MIB
}

pub const CATALOG: &[Gpu] = &[
    Gpu {
        name: "H100-80GB",
        memory_bytes: mib(81559.0),
        source: "measured: nvidia-smi, results/h100_v1/logs/env_20260427T051317Z.txt",
    },
    Gpu {
        name: "H200-141GB",
        memory_bytes: mib(143771.0),
        source: "measured: nvidia-smi, results/h200_v2/nvidia_smi_full.txt",
    },
    Gpu {
        name: "A100-80GB",
        memory_bytes: mib(80.0 * 1024.0),
        source: "nominal",
    },
    Gpu {
        name: "A100-40GB",
        memory_bytes: mib(40.0 * 1024.0),
        source: "nominal",
    },
    Gpu {
        name: "L4-24GB",
        memory_bytes: mib(24.0 * 1024.0),
        source: "nominal",
    },
    Gpu {
        name: "T4-16GB",
        memory_bytes: mib(16.0 * 1024.0),
        source: "nominal",
    },
    Gpu {
        name: "P100-16GB",
        memory_bytes: mib(16.0 * 1024.0),
        source: "nominal",
    },
    Gpu {
        name: "RTX4060-8GB",
        memory_bytes: mib(8.0 * 1024.0),
        source: "nominal",
    },
];

/// Look up a GPU by name, ignoring case and `-`/`_`/space differences.
pub fn find(name: &str) -> Option<Gpu> {
    let key = normalise(name);
    CATALOG.iter().copied().find(|g| normalise(g.name) == key)
}

fn normalise(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_ignores_case_and_separators() {
        assert_eq!(find("h100-80gb").unwrap().name, "H100-80GB");
        assert_eq!(find("H100_80GB").unwrap().name, "H100-80GB");
        assert!(find("H100").is_none());
    }

    #[test]
    fn measured_sizes() {
        assert_eq!(find("H100-80GB").unwrap().memory_bytes, 81559.0 * MIB);
        assert_eq!(find("H200-141GB").unwrap().memory_bytes, 143771.0 * MIB);
    }
}
