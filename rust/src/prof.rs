//! Profiler internal ringan, opt-in lewat `ASCA_PROF=1`.
//!
//! Tanpa flag ini biayanya satu pembacaan atomik per scope; dengan flag ini setiap
//! komponen mengakumulasi waktu dinding supaya keputusan optimasi berangkat dari
//! pengukuran, bukan dugaan.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const NAMES: [&str; 10] = [
    "delta.qkv",
    "delta.scan",
    "delta.out",
    "delta.bwd",
    "moe.fwd",
    "moe.bwd",
    "norm",
    "head",
    "embed",
    "other",
];

pub static ENABLED: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 10] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub fn init() {
    ENABLED.store(
        std::env::var("ASCA_PROF").ok().as_deref() == Some("1"),
        Ordering::Relaxed,
    );
}

#[inline]
pub fn add(idx: usize, nanos: u64) {
    if ENABLED.load(Ordering::Relaxed) {
        NS[idx].fetch_add(nanos, Ordering::Relaxed);
    }
}

/// Ukur satu scope ke slot `idx`.
#[inline]
pub fn time<R>(idx: usize, f: impl FnOnce() -> R) -> R {
    if !ENABLED.load(Ordering::Relaxed) {
        return f();
    }
    let t0 = std::time::Instant::now();
    let out = f();
    add(idx, t0.elapsed().as_nanos() as u64);
    out
}

pub fn report() -> String {
    let total: u64 = NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
    let mut lines = vec![format!("profil (total {:.1} ms)", total as f64 / 1e6)];
    for (i, name) in NAMES.iter().enumerate() {
        let ns = NS[i].load(Ordering::Relaxed);
        if ns == 0 {
            continue;
        }
        lines.push(format!(
            "  {name:<12} {:>9.1} ms  {:>5.1}%",
            ns as f64 / 1e6,
            100.0 * ns as f64 / total.max(1) as f64
        ));
    }
    lines.join("\n")
}
