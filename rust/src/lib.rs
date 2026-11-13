//! ASCA v2 — arsitektur tanpa attention, state konstan, ditulis dari nol di Rust.
//!
//! `dense` adalah baseline pembanding yang dijalankan di engine yang sama
//! (Mat, Param, Rng, RMSNorm identik) supaya perbandingan efisiensi sah.

pub mod bench;
pub mod delta;
pub mod dense;
pub mod model;
pub mod tensor;
pub mod train;
