# GPU native libraries: what each backend loads, and when

Inventory for #377 (item 4 of #1265). It lists the native GPU libraries each AiDotNet.Tensors backend can load, whether the backend needs each one, when it is loaded, and what happens without it.

The rule is the same everywhere: vendor *math* libraries are optional accelerators. A backend needs only its driver or runtime (and, on CUDA and HIP, the runtime compiler that builds the library's generated kernels). Every vendor math library is loaded lazily, and its absence is normal: no exception and no warning. A generated or built-in kernel serves the operation instead.

## CUDA (`CudaBackend`)

| Library (Windows / Linux) | Needed? | Loaded | Without it |
|---|---|---|---|
| `nvcuda` / `libcuda.so.1` (driver) | **Required** | When the backend is constructed (`CudaBackend.IsCudaAvailable`) | The backend is unavailable (`IsAvailable == false`) |
| `nvrtc64_*` / `libnvrtc.so.*` (runtime compiler; 13.0, 12.x and 11.x names are probed) | **Required** | When the backend is constructed | The backend is unavailable |
| `cublas64_12` / `libcublas.so.12` | Optional | Lazily, on the first GEMM (`IsCublasAvailable` creates this thread's handle) | **Every GEMM runs on the generated kernels** (#1265). Covers FP32 in all transpose forms, strided-batched, FP16 / BF16 in with FP32 out, FP16 in and out, and FP64. See below. |
| `cublasLt` | Optional | Lazily (`TryGetCuBlasLt`) | Callers take their non-Lt path |
| `cudnn64_9.dll` (falls back to `cudnn64_8.dll`) / `libcudnn` | Optional | Lazily, on the first convolution / batch norm / RNN that would use it (`EnsureCudnnConv`, `EnsureCudnnBatchNorm`) | `CudaDispatchPolicy.UseCudnnForConv` / `UseCudnnForBatchNorm` are false and the generic kernels run |
| `curand64_10` / `libcurand` | Optional | Lazily, on the first device RNG draw | Random numbers are generated another way (`CuRand.LastUniformUsedDevice` reports which) |
| `nccl` / `libnccl` | Optional | Only when a multi-GPU communicator (`NcclComm`) is created | Multi-GPU collectives are unavailable; single-GPU work is unaffected |

### GEMM without cuBLAS

`CudaBackend` routes every cuBLAS GEMM through helpers (`CudaBackend.BlasDispatch.cs`) that keep cuBLAS's column-major calling convention:

- **With cuBLAS:** they call cuBLAS.
- **Without it:** they run the generated GEMM (`PtxSgemmGenerator`) on the row-major equivalent problem.
  - FP32 uses the first tuned seed that admits the actual leading dimensions and strides. Every other precision, and any problem no seed admits, uses a scalar kernel that admits any shape.
  - The fallback always computes FP32 strictly (no TF32), so it is never less accurate than cuBLAS. It matches cuBLAS in strict mode to about 4e-7 relative.
  - With FP16 output the fallback accumulates in FP32, where cuBLAS `Hgemm` accumulates in FP16.
- **The only constraint:** the first fallback GEMM of a given configuration must run outside CUDA graph capture, because its module cannot be loaded inside one.

Three ways to switch cuBLAS off:

| Switch | Scope |
|---|---|
| No cuBLAS library on the machine | Process |
| `AIDOTNET_DISABLE_CUBLAS=1` | Process, read at startup |
| `TensorCodecOptions.UseCublas = false` | Thread, read at every GEMM |

### Tuned routing

Independent of cuBLAS's presence, `Gemm` routes a shape to a tuned generated kernel only where it measured faster than cuBLAS on this device (#1309, `CudaBackend.TuneGemmRouteAsync`). cuBLAS calls stay wherever cuBLAS still wins, so they are kept and made optional rather than removed.

## HIP (`HipBackend`)

| Library | Needed? | Loaded | Without it |
|---|---|---|---|
| `amdhip64_6` (runtime) | **Required** | When the backend is constructed (`HipBackend.IsHipAvailable`) | The backend is unavailable |
| `hiprtc0604` (runtime compiler) | **Required** for generated kernels | When the backend is constructed | The backend is unavailable |
| `hipblas` | Optional | At backend initialization, only if its code objects cover the GPU's architecture (`TryConfigureForArchitecture`) | GEMMs use the direct HIP kernels |
| `rocblas` | Optional | With hipBLAS, and for rocSOLVER-backed linear algebra | As hipBLAS |
| `rocsparse` | Optional | Lazily (`HipSparseBackend.IsAvailable`) | The custom sparse kernels serve |
| `MIOpen` | Optional | Lazily, on the first convolution that would use it | The generic kernels run |

## OpenCL (`OpenClBackend`)

| Library | Needed? | Loaded | Without it |
|---|---|---|---|
| `OpenCL.dll` / `libOpenCL.so` | **Required** | When the backend is constructed | The backend is unavailable |
| `clblast` | Optional | Lazily (`ClBlastNative.IsAvailable`) | GEMMs use the generated or evolution-tuned OpenCL kernels (`OpenClGemmDispatchPath` reports which) |

## Vulkan (`VulkanBackend`)

| Library | Needed? | Loaded | Without it |
|---|---|---|---|
| `vulkan-1.dll` / `libvulkan.so.1` / `libvulkan.1.dylib` | **Required** | When the backend is constructed | The backend is unavailable |

The Vulkan backend has no vendor math libraries; every kernel is its own GLSL / SPIR-V.

## Metal (`MetalBackend`)

| Library | Needed? | Loaded | Without it |
|---|---|---|---|
| `Metal.framework` via `/usr/lib/libobjc.dylib` | **Required** (macOS only) | When the backend is constructed | The backend is unavailable |

## CPU libraries

These are listed for completeness; none is GPU-related.

| Library | Needed? | Loaded | Without it |
|---|---|---|---|
| `libopenblas` | Optional, opt-in with `AIDOTNET_USE_BLAS=1` | On first BLAS use | The managed GEMM (`BlasManaged`) runs |
