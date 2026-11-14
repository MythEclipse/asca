//! Binding minimal CUDA driver API. Hanya yang dipakai: satu context, buffer device,
//! satu modul PTX, dan peluncuran kernel sinkron. Tanpa crate eksternal.

use std::ffi::{c_char, c_void};

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Device(pub i32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Context(pub *mut c_void);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Module(pub *mut c_void);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Function(pub *mut c_void);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct DevicePtr(pub u64);

pub const SUCCESS: i32 = 0;

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
pub const ATTRIBUTE_MULTIPROCESSORS: i32 = 16;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
pub const FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED: i32 = 8;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> i32;
    pub fn cuDeviceGet(device: *mut Device, ordinal: i32) -> i32;
    pub fn cuDeviceGetName(name: *mut c_char, len: i32, device: Device) -> i32;
    pub fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: Device) -> i32;
    pub fn cuCtxCreate_v2(context: *mut Context, flags: u32, device: Device) -> i32;
    pub fn cuCtxDestroy_v2(context: Context) -> i32;
    pub fn cuCtxSetCurrent(context: Context) -> i32;
    pub fn cuCtxSynchronize() -> i32;
    pub fn cuMemAlloc_v2(pointer: *mut DevicePtr, bytes: usize) -> i32;
    pub fn cuMemFree_v2(pointer: DevicePtr) -> i32;
    pub fn cuMemcpyHtoD_v2(dst: DevicePtr, src: *const c_void, bytes: usize) -> i32;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: DevicePtr, bytes: usize) -> i32;
    pub fn cuMemsetD32_v2(dst: DevicePtr, value: u32, n: usize) -> i32;
    pub fn cuModuleLoadData(module: *mut Module, image: *const c_void) -> i32;
    pub fn cuModuleGetFunction(function: *mut Function, module: Module, name: *const c_char)
    -> i32;
    pub fn cuFuncSetAttribute(function: Function, attribute: i32, value: i32) -> i32;
    pub fn cuLaunchKernel(
        function: Function,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

/// Pesan galat dari kode hasil driver (hanya kode numerik; nama butuh cuGetErrorName).
pub fn err(code: i32) -> String {
    format!("CUDA error {code}")
}
