namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Self-contained FP16 kernels (#558 layer 7) that do NOT depend on &lt;cuda_fp16.h&gt;. The toolkit
/// header is absent on driver-only machines (the bundled nvrtc 12.0 doesn't ship it), which silently
/// disabled the entire existing FP16 path (CudaFp16Kernels: "could not open source file cuda_fp16.h").
/// These kernels convert half↔float with the standard IEEE-754 bit algorithm in-kernel, so they compile
/// with the driver alone — confirming the FP16-native approach works on this hardware. (The __half2
/// packed path in CudaFp16Kernels remains the faster variant for toolkit machines.)
/// </summary>
public static class CudaFp16NativeKernels
{
    public static string GetSource()
    {
        return @"
// ── IEEE-754 half <-> float, no cuda_fp16.h ──────────────────────────────
__device__ __forceinline__ float h2f(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int f;
    if (exp == 0u) {
        if (mant == 0u) { f = sign; }
        else {
            int e = 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = sign | ((unsigned int)(e + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return __int_as_float((int)f);
}

__device__ __forceinline__ unsigned short f2h(float x)
{
    unsigned int f = (unsigned int)__float_as_int(x);
    unsigned int sign = (f >> 16) & 0x8000u;
    int exp = (int)((f >> 23) & 0xFFu) - 127 + 15;
    unsigned int mant = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFFu) == 0xFFu) {                 // inf / nan
        return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) { return (unsigned short)(sign | 0x7C00u); }   // overflow -> inf
    if (exp <= 0) {                                     // subnormal / underflow
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        unsigned int h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) h++;           // round to nearest
        return (unsigned short)(sign | h);
    }
    unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1u) h++;                         // round to nearest
    return (unsigned short)(sign | h);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp32_to_fp16_native(
    const float* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = f2h(input[idx]);
}

extern ""C"" __global__ __launch_bounds__(256) void convert_fp16_to_fp32_native(
    const unsigned short* __restrict__ input, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = h2f(input[idx]);
}

// FP16-NATIVE GELU: reads FP16 directly, computes in FP32 in-register, writes FP16. No FP32 buffer is
// materialized — half the bandwidth + half-resident activation, the property the GPU memory/util win needs.
extern ""C"" __global__ __launch_bounds__(256) void fp16_gelu_native(
    const unsigned short* __restrict__ input, unsigned short* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = h2f(input[idx]);
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float r = 0.5f * x * (1.0f + tanhf(inner));
    output[idx] = f2h(r);
}
";
    }

    public static string[] GetKernelNames() => new[]
    {
        "convert_fp32_to_fp16_native",
        "convert_fp16_to_fp32_native",
        "fp16_gelu_native"
    };
}
