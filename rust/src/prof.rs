//! Profiler internal ringan, opt-in lewat `ASCA_PROF=1`.
//!
//! Tanpa flag ini biayanya satu pembacaan atomik per scope; dengan flag ini setiap
//! komponen mengakumulasi waktu dinding supaya keputusan optimasi berangkat dari
//! pengukuran, bukan dugaan.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const NAMES: [&str; 15] = [
    "delta.qkv",
    "delta.scan",
    "delta.out",
    "delta.bwd",
    "moe.fwd",
    "moe.bwd",
    "norm",
    "head",
    "embed",
    "head.bwd",
    "dense.attn",
    "dense.attn.bwd",
    "dense.mlp",
    "dense.mlp.bwd",
    "adamw",
];

pub static ENABLED: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 15] = [
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
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Waktu dinding sejak `init`, untuk menghitung berapa persen yang TIDAK masuk scope mana
/// pun. Tanpa ini persentase di laporan hanya berarti "pangsa antar scope terukur", bukan
/// pangsa dari waktu nyata - mudah disalahartikan.
pub static WALL_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn wall_fraction() -> f64 {
    let Some(t0) = WALL_T0.get() else {
        return 0.0;
    };
    let wall = t0.elapsed().as_nanos() as u64;
    let total: u64 = NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
    100.0 * (1.0 - total as f64 / wall.max(1) as f64)
}

pub fn init() {
    ENABLED.store(
        std::env::var("ASCA_PROF").ok().as_deref() == Some("1"),
        Ordering::Relaxed,
    );
    let _ = WALL_T0.set(std::time::Instant::now());
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
    let mut lines = vec![format!(
        "profil (scope terukur {:.1} ms; sisa = {:.1}% dari dinding)",
        total as f64 / 1e6,
        wall_fraction()
    )];
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
