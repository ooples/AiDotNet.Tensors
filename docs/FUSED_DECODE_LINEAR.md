# Fused decode-linear routes

The decode-linear routes compute a linear projection, a bias add and tanh-GELU in one device pass, for the
decode-token shapes. They are declared by `IFusedDecodeLinearBackend` and implemented by all six GPU backends.

## The routes

Weights are canonical output-major `[N, K]`: row `n` holds output `n`'s `K` weights. Activations are contiguous;
the sixteen-row route is row-major `[16, K]`. `K` is the input feature count and `N` the output feature count.
GELU is `0.5 x (1 + tanh(0.7978845608 (x + 0.044715 x^3)))` on every backend.

| Route | Operands | Accumulation | Output |
|---|---|---|---|
| `FusedLinearGELUTransposedM1` | FP32 input `[K]`, FP32 weights | FP32 | FP32 `[N]` |
| `FusedLinearGELUFp16TransposedM1` | FP16 input `[K]`, FP16 weights, FP32 bias | FP32 | FP32 `[N]` |
| `FusedLinearGELUFp16TransposedM16` | FP16 input `[16, K]`, FP16 weights, FP32 bias | FP32 | FP32 `[16, N]` |
| `FusedLinearGELUW8A8TransposedM1` | int8 input `[K]`, int8 weights, one FP32 activation scale, FP32 weight scales `[N]`, FP32 bias | int32, exact | FP32 `[N]` |
| `W8A8DequantBiasGelu` | int32 accumulator `[N]`, the same scales and bias | none (epilogue only) | FP32 `[N]` |

The quantized routes are symmetric: zero-points are exactly zero. Dequantization is
`accumulator * activationScale[0] * weightScales[n] + bias[n]`, in FP32.

## Backend status

| Backend | Implementation | Verified |
|---|---|---|
| CUDA | Direct-PTX specializations where admitted; cuBLAS or resident NVRTC kernels otherwise. The FP16 routes need compute capability 5.0 or later and throw `NotSupportedException` below it. | RTX 3080 |
| Vulkan | The shared `dl_p_` kernels, translated to GLSL. | RTX 3080 |
| OpenCL | The shared `dl_p_` kernels, translated to OpenCL C. | RTX 3080 (NVIDIA OpenCL) |
| WebGPU | Hand-written WGSL ports of the `dl_p_` kernels. | Compiled and executed on Dawn (D3D12) through the test harness; not through the browser backend. |
| HIP | The shared `dl_p_` kernels, compiled as they are. | Not run: no AMD ROCm device was available. |
| Metal | The shared `dl_p_` kernels, translated to MSL. | Not run: no macOS device was available. |

The shared kernels live in `BatchExecutionCudaKernels.DecodeLinearSource` and are launched by
`PortableFusedDecodeLinear`: one dispatch per route, one thread per output element.

## Limits and behaviour to know

- **Quantized input length.** Each int8 product is at most 128 x 128 in magnitude, so an int32 accumulator cannot
  overflow for `K` up to 131,071. The shared-kernel backends reject a longer `K` with `ArgumentOutOfRangeException`.
- **Operand padding.** The shared kernels read FP16 and int8 operands through 32-bit words. A byte buffer whose
  size is not a whole number of words (an odd count of FP16 values, or an int8 count that is not a multiple of
  four) is copied into a padded scratch buffer on every call, and that copy goes through the host. It is counted
  by the residency probe like any other readback. To keep such an operand resident, allocate it with its byte
  size rounded up to a multiple of four. Vulkan byte buffers are always whole words. CUDA is not affected.
- **Tolerance.** CUDA's direct-PTX paths use `tanh.approx.f32` and are documented to 5e-4. The shared kernels
  differ from a double-precision oracle only by FP32 accumulation and the device's `tanh`; the parity test holds
  them to 2e-5 relative.
- **No engine route yet.** The routes are backend entry points; nothing in the tensor engine dispatches to them
  on any backend.
