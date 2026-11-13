//! Kompilasi `kernels/asca.cu` ke PTX dengan `nvcc`, lalu embed hasilnya.
//!
//! Jalur GPU bersifat opsional: tanpa `nvcc` crate tetap bisa dibangun dan
//! `cuda::available()` melaporkan tidak ada backend, sehingga jalur CPU tetap jalan
//! di mesin tanpa toolkit CUDA. Identitas build (arch, hash PTX, daftar kernel)
//! ikut ditulis karena setiap angka benchmark harus bisa menyebut artifact-nya.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_ARCH: &str = "sm_89";

fn main() {
    println!("cargo:rerun-if-changed=kernels/asca.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ASCA_CUDA_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let source = Path::new("kernels/asca.cu");
    let arch = env::var("ASCA_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.to_string());
    let ptx = out_dir.join("asca.ptx");

    let mut nvcc = "unavailable".to_string();
    let mut kernels: Vec<String> = Vec::new();
    let mut ok = false;

    if let Ok(output) = Command::new("nvcc")
        .args([
            "-ptx",
            "-O3",
            "-std=c++17",
            &format!("-arch={arch}"),
            "-o",
            &ptx.to_string_lossy(),
            &source.to_string_lossy(),
        ])
        .output()
    {
        if output.status.success() {
            ok = true;
            if let Ok(ver) = Command::new("nvcc").arg("--version").output() {
                nvcc = String::from_utf8_lossy(&ver.stdout)
                    .lines()
                    .find(|l| l.contains("release"))
                    .unwrap_or("unknown")
                    .trim()
                    .to_string();
            }
            if let Ok(text) = std::fs::read_to_string(&ptx) {
                for line in text.lines() {
                    if let Some(rest) = line.strip_prefix(".visible .entry ") {
                        let name = rest.split('(').next().unwrap_or("").trim().to_string();
                        if !name.is_empty() {
                            kernels.push(name);
                        }
                    }
                }
            }
        } else {
            println!(
                "cargo:warning=GPU backend disabled: nvcc gagal: {}",
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
            );
        }
    } else {
        println!("cargo:warning=GPU backend disabled: nvcc tidak ditemukan");
    }

    let ptx_binding = if ok {
        format!("Some(include_str!({:?}))", ptx.to_string_lossy())
    } else {
        "None".to_string()
    };
    let generated = format!(
        "pub const PTX: Option<&str> = {ptx_binding};\n\
         pub const ARCH: &str = {arch:?};\n\
         pub const NVCC: &str = {nvcc:?};\n\
         pub const KERNELS: &[&str] = &{kernels:?};\n"
    );
    std::fs::write(out_dir.join("gpu_build.rs"), generated).expect("tulis gpu_build.rs");
}
