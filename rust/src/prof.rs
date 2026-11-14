//! Profiler internal ringan, opt-in lewat `ASCA_PROF=1`.
//!
//! Tanpa flag ini biayanya satu pembacaan atomik per scope; dengan flag ini setiap
//! komponen mengakumulasi waktu dinding supaya keputusan optimasi berangkat dari
//! pengukuran, bukan dugaan.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const NAMES: [&str; 19] = [
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
    "zero.grad",
    "batch",
    "fwd.total",
    "bwd.total",
];

pub static ENABLED: AtomicBool = AtomicBool::new(false);

// Scope yang sedang berjalan di thread ini. Dipakai untuk membebankan setiap byte
// H2D/D2H ke komponen yang menyebabkannya — tanpa itu, "743 MB per langkah" hanya angka
// total dan tidak memberi tahu mana yang harus diperbaiki.
thread_local! {
    static CUR: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[inline]
pub fn current() -> usize {
    CUR.with(|c| c.get())
}

/// Byte per scope: [0..N] untuk tiap komponen, indeks N untuk di luar scope mana pun.
static TRAFFIC_IN: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];
static TRAFFIC_OUT: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];

/// Catat lalu lintas device ke scope yang sedang aktif.
#[inline]
pub fn add_traffic(bytes: u64, outbound: bool) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let i = current().min(NAMES.len());
    if outbound {
        TRAFFIC_OUT[i].fetch_add(bytes, Ordering::Relaxed);
    } else {
        TRAFFIC_IN[i].fetch_add(bytes, Ordering::Relaxed);
    }
}
static NS: [AtomicU64; 19] = [
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
    let prev = CUR.with(|c| c.replace(idx));
    let out = f();
    CUR.with(|c| c.set(prev));
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
        let tin = TRAFFIC_IN[i].load(Ordering::Relaxed) as f64 / 1e6;
        let tout = TRAFFIC_OUT[i].load(Ordering::Relaxed) as f64 / 1e6;
        lines.push(format!(
            "  {name:<12} {:>9.1} ms  {:>5.1}%   {:>7.1} MB naik, {:>7.1} MB turun",
            ns as f64 / 1e6,
            100.0 * ns as f64 / total.max(1) as f64,
            tin,
            tout
        ));
    }
    let oi = NAMES.len();
    let tin = TRAFFIC_IN[oi].load(Ordering::Relaxed) as f64 / 1e6;
    let tout = TRAFFIC_OUT[oi].load(Ordering::Relaxed) as f64 / 1e6;
    if tin + tout > 0.0 {
        lines.push(format!(
            "  {:<12} {:>9}      {:>7.1} MB naik, {:>7.1} MB turun",
            "(luar scope)", "", tin, tout
        ));
    }
    lines.join("\n")
}
