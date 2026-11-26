//! Byte units. KVScope uses 1024-based units throughout, matching the Python
//! tracer (whose "KB"/"MB" are 1024 and 1024² bytes). Output labels them
//! KiB / MiB / GiB so there is no ambiguity.

pub const KIB: f64 = 1024.0;
pub const MIB: f64 = 1024.0 * 1024.0;
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Format a byte count with the largest unit that keeps the number >= 1.
pub fn fmt_bytes(bytes: f64) -> String {
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_largest_unit() {
        assert_eq!(fmt_bytes(512.0), "512 B");
        assert_eq!(fmt_bytes(8192.0), "8.00 KiB");
        assert_eq!(fmt_bytes(4.211 * MIB), "4.21 MiB");
        assert_eq!(fmt_bytes(80.0 * GIB), "80.00 GiB");
    }
}
