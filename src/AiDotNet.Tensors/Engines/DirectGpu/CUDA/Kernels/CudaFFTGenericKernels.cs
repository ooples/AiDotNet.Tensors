// Copyright (c) AiDotNet. All rights reserved.
// Generic-precision, arbitrary-length CUDA FFT kernels.
//
// This complements CudaFFTKernels, which ships float32 radix-2 only and requires callers with a non-power-of-two
// length to fall back to the CPU Bluestein path. Two consequences of that fallback motivated this file:
//
//   1. A device-resident tensor whose transform length is not a power of two is copied to the host, transformed,
//      and copied back. For a spectral neural operator running per layer, that round trip costs far more than
//      the transform. Qwen2.5-0.5B has width 896 = 2^7 * 7, so the model this library is most often pointed at
//      lands on exactly that path.
//   2. Every buffer is float32, so a caller running a bf16 or fp16 model must widen before the transform and
//      narrow after - two full passes over the activation tensor, on kernels that are memory-bound to begin
//      with.
//
// Both are addressed here: Bluestein runs on the device for arbitrary n, and the buffers are generic over
// element type while the arithmetic stays in float32.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// Emits CUDA source for the generic FFT path, specialized per <see cref="FftElementType"/>.
    /// </summary>
    internal static class CudaFFTGenericKernels
    {
        /// <summary>
        /// Entry-point base names, before the element-type suffix is appended. Kernels whose buffers are always
        /// float32 (the Bluestein workspace and the twiddle/chirp builders) are listed in
        /// <see cref="TypeInvariantKernelNames"/> instead and are emitted once.
        /// </summary>
        private static readonly string[] TypedKernelBaseNames =
        {
            "fftg_batched_bit_reverse",
            "fftg_batched_butterfly",
            "fftg_scale",
            "fftg_bluestein_premul",
            "fftg_bluestein_postmul",
        };

        /// <summary>Kernels that operate purely on float32 scratch and are emitted once, without a suffix.</summary>
        private static readonly string[] TypeInvariantKernelNames =
        {
            "fftg_build_twiddles",
            "fftg_build_chirp",
            "fftg_build_chirp_padded",
            "fftg_f32_bit_reverse",
            "fftg_f32_butterfly",
            "fftg_f32_scale",
            "fftg_bluestein_pointwise",
        };

        /// <summary>All entry points present in the module emitted by <see cref="GetSource"/>.</summary>
        public static string[] GetKernelNames(FftElementType type)
        {
            string suffix = type.KernelSuffix();
            var names = new string[TypedKernelBaseNames.Length + TypeInvariantKernelNames.Length];
            for (int i = 0; i < TypedKernelBaseNames.Length; i++)
            {
                names[i] = TypedKernelBaseNames[i] + suffix;
            }

            System.Array.Copy(TypeInvariantKernelNames, 0, names, TypedKernelBaseNames.Length, TypeInvariantKernelNames.Length);
            return names;
        }

        /// <summary>
        /// Smallest power of two greater than or equal to <c>2*n - 1</c>: the convolution length Bluestein needs
        /// so that the cyclic convolution it performs equals the linear one it requires.
        /// </summary>
        public static int BluesteinLength(int n)
        {
            if (n <= 1)
            {
                return 1;
            }

            int target = (2 * n) - 1;
            int m = 1;
            while (m < target)
            {
                m <<= 1;
            }

            return m;
        }

        /// <summary>True when <paramref name="n"/> can use the direct radix-2 path rather than Bluestein.</summary>
        public static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

        /// <summary>
        /// CUDA source for one element-type specialization. The narrow types appear only in the load and store
        /// helpers; every arithmetic expression below operates on <c>float</c>.
        /// </summary>
        public static string GetSource(FftElementType type)
        {
            string suffix = type.KernelSuffix();
            string storeType;
            string loadExpr;
            string storeExpr;
            string includes;

            switch (type)
            {
                case FftElementType.Float32:
                    includes = string.Empty;
                    storeType = "float";
                    loadExpr = "(v)";
                    storeExpr = "(v)";
                    break;
                case FftElementType.Float16:
                    includes = "#include <cuda_fp16.h>\n";
                    storeType = "__half";
                    loadExpr = "__half2float(v)";
                    storeExpr = "__float2half_rn(v)";
                    break;
                case FftElementType.BFloat16:
                    includes = "#include <cuda_bf16.h>\n";
                    storeType = "__nv_bfloat16";
                    loadExpr = "__bfloat162float(v)";
                    storeExpr = "__float2bfloat16(v)";
                    break;
                default:
                    throw new System.ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type.");
            }

            // The typed kernels are written once with STORE_T / LD / ST macros and emitted per element type.
            // Keeping one body avoids the classic failure mode of hand-maintained per-precision copies, where a
            // bug fixed in the float32 variant quietly survives in the others.
            string src = includes + @"
#include <math.h>

#define AIDN_PI 3.14159265358979323846f
#define STORE_T " + storeType + @"
__device__ __forceinline__ float aidn_ld(STORE_T v) { return " + loadExpr + @"; }
__device__ __forceinline__ STORE_T aidn_st(float v) { return " + storeExpr + @"; }

// ---------------------------------------------------------------------------
// Shared float32 scratch kernels (emitted once; no element-type suffix).
// ---------------------------------------------------------------------------
#ifndef AIDN_FFTG_SHARED
#define AIDN_FFTG_SHARED

// Twiddle table for a length-n radix-2 transform: w[j] = exp(-2*pi*i*j/n) for j < n/2.
// The existing radix-2 butterfly calls cosf/sinf per element per stage; tabulating once removes
// log2(n) transcendental pairs per element and makes the butterfly purely load/store bound.
extern ""C"" __global__ __launch_bounds__(256) void fftg_build_twiddles(
    float* twRe, float* twIm, int half, int n, int inverse)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    float sign = inverse ? 2.0f : -2.0f;
    float angle = sign * AIDN_PI * (float)j / (float)n;
    twRe[j] = __cosf(angle);
    twIm[j] = __sinf(angle);
}

// Bluestein chirp c[k] = exp(+i*pi*k^2/n) (forward; conjugated when inverse), length n.
// k*k overflows int for n beyond ~46341, so the squaring is done in 64-bit and reduced mod 2n
// BEFORE the float conversion - reducing after conversion loses the low bits that carry the phase.
extern ""C"" __global__ __launch_bounds__(256) void fftg_build_chirp(
    float* chRe, float* chIm, int n, int inverse)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    long long kk = (long long)k * (long long)k;
    long long twoN = 2LL * (long long)n;
    long long r = kk % twoN;                 // exp(i*pi*k^2/n) has period 2n in k^2
    float sign = inverse ? -1.0f : 1.0f;
    float angle = sign * AIDN_PI * (float)r / (float)n;
    chRe[k] = cosf(angle);
    chIm[k] = sinf(angle);
}

// Bluestein convolution kernel, length m, SYMMETRICALLY EXTENDED:
//     b[j]     = c[j]   for 0 <= j < n
//     b[m - j] = c[j]   for 1 <= j < n
//     b[.]     = 0      elsewhere
// The extension is not cosmetic. Bluestein needs a LINEAR convolution; an m-point FFT computes a CYCLIC one,
// and the two agree only when the kernel is mirrored into the upper tail. Filling just [0, n) leaves the
// transform correct at k = 0 and wrong everywhere else - a reference implementation of this exact omission
// measures 0.71 relative error at n = 896 (hre_port_tests/fftref.py, negative control), i.e. it is the kind of
// bug that passes a smoke test on DC and fails on everything anyone cares about.
extern ""C"" __global__ __launch_bounds__(256) void fftg_build_chirp_padded(
    float* bRe, float* bIm, int n, int m, int inverse)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;

    // Map the padded index back to the chirp index it mirrors, or mark it as zero padding.
    int src = -1;
    if (j < n) { src = j; }
    else if (j >= m - n + 1) { src = m - j; }

    if (src < 0) { bRe[j] = 0.0f; bIm[j] = 0.0f; return; }

    long long kk = (long long)src * (long long)src;
    long long twoN = 2LL * (long long)n;
    long long r = kk % twoN;
    float sign = inverse ? -1.0f : 1.0f;
    float angle = sign * AIDN_PI * (float)r / (float)n;
    bRe[j] = cosf(angle);
    bIm[j] = sinf(angle);
}

extern ""C"" __global__ __launch_bounds__(256) void fftg_f32_bit_reverse(
    const float* srcRe, const float* srcIm, float* dstRe, float* dstIm, int batch, int n, int log2n)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= n) return;
    int rev = 0, t = idx;
    for (int i = 0; i < log2n; i++) { rev = (rev << 1) | (t & 1); t >>= 1; }
    int base = b * n;
    dstRe[base + idx] = srcRe[base + rev];
    dstIm[base + idx] = srcIm[base + rev];
}

extern ""C"" __global__ __launch_bounds__(256) void fftg_f32_butterfly(
    float* re, float* im, const float* twRe, const float* twIm,
    int batch, int n, int stride)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride >> 1;
    int numBf = n / stride;
    int bf = idx / halfStride;
    int wing = idx - bf * halfStride;
    if (b >= batch || bf >= numBf) return;

    // Table is built for the full length n; stage `stride` samples it every n/stride entries.
    int step = n / stride;
    float twR = twRe[wing * step];
    float twI = twIm[wing * step];

    int top = b * n + bf * stride + wing;
    int bot = top + halfStride;
    float tR = re[top], tI = im[top];
    float bR = re[bot], bI = im[bot];
    float xR = bR * twR - bI * twI;
    float xI = bR * twI + bI * twR;
    re[top] = tR + xR; im[top] = tI + xI;
    re[bot] = tR - xR; im[bot] = tI - xI;
}

extern ""C"" __global__ __launch_bounds__(256) void fftg_f32_scale(float* re, float* im, int count, float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    re[idx] *= s; im[idx] *= s;
}

// Pointwise product of the padded signal spectrum with the pre-transformed chirp spectrum.
extern ""C"" __global__ __launch_bounds__(256) void fftg_bluestein_pointwise(
    float* re, float* im, const float* kRe, const float* kIm, int batch, int m)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= m) return;
    int o = b * m + idx;
    float aR = re[o], aI = im[o];
    float bR = kRe[idx], bI = kIm[idx];
    re[o] = aR * bR - aI * bI;
    im[o] = aR * bI + aI * bR;
}

#endif // AIDN_FFTG_SHARED

// ---------------------------------------------------------------------------
// Element-type specialized kernels. Narrow load, float32 compute, narrow store.
// ---------------------------------------------------------------------------

extern ""C"" __global__ __launch_bounds__(256) void fftg_batched_bit_reverse" + suffix + @"(
    const STORE_T* srcRe, const STORE_T* srcIm, STORE_T* dstRe, STORE_T* dstIm, int batch, int n, int log2n)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= n) return;
    int rev = 0, t = idx;
    for (int i = 0; i < log2n; i++) { rev = (rev << 1) | (t & 1); t >>= 1; }
    int base = b * n;
    dstRe[base + idx] = srcRe[base + rev];
    dstIm[base + idx] = srcIm[base + rev];
}

extern ""C"" __global__ __launch_bounds__(256) void fftg_batched_butterfly" + suffix + @"(
    STORE_T* re, STORE_T* im, const float* twRe, const float* twIm,
    int batch, int n, int stride)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride >> 1;
    int numBf = n / stride;
    int bf = idx / halfStride;
    int wing = idx - bf * halfStride;
    if (b >= batch || bf >= numBf) return;

    int step = n / stride;
    float twR = twRe[wing * step];
    float twI = twIm[wing * step];

    int top = b * n + bf * stride + wing;
    int bot = top + halfStride;

    // Widen on load; every operation below is float32. Storing narrow is what halves the traffic.
    float tR = aidn_ld(re[top]), tI = aidn_ld(im[top]);
    float bR = aidn_ld(re[bot]), bI = aidn_ld(im[bot]);
    float xR = bR * twR - bI * twI;
    float xI = bR * twI + bI * twR;
    re[top] = aidn_st(tR + xR); im[top] = aidn_st(tI + xI);
    re[bot] = aidn_st(tR - xR); im[bot] = aidn_st(tI - xI);
}

extern ""C"" __global__ __launch_bounds__(256) void fftg_scale" + suffix + @"(
    STORE_T* re, STORE_T* im, int count, float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    re[idx] = aidn_st(aidn_ld(re[idx]) * s);
    im[idx] = aidn_st(aidn_ld(im[idx]) * s);
}

// Bluestein step 1: x[k] * conj(chirp[k]) into a zero-padded float32 workspace of length m.
// The workspace is float32 regardless of the caller's element type: it is transient scratch, so narrowing it
// would compound rounding through two extra transforms while saving no traffic the caller ever sees.
extern ""C"" __global__ __launch_bounds__(256) void fftg_bluestein_premul" + suffix + @"(
    const STORE_T* xRe, const STORE_T* xIm, const float* chRe, const float* chIm,
    float* wRe, float* wIm, int batch, int n, int m)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= m) return;
    int o = b * m + idx;
    if (idx >= n) { wRe[o] = 0.0f; wIm[o] = 0.0f; return; }
    int s = b * n + idx;
    float aR = aidn_ld(xRe[s]), aI = aidn_ld(xIm[s]);
    float cR = chRe[idx], cI = -chIm[idx];        // conjugate
    wRe[o] = aR * cR - aI * cI;
    wIm[o] = aR * cI + aI * cR;
}

// Bluestein step 3: multiply the first n outputs by conj(chirp) and write back in the caller's element type.
extern ""C"" __global__ __launch_bounds__(256) void fftg_bluestein_postmul" + suffix + @"(
    const float* wRe, const float* wIm, const float* chRe, const float* chIm,
    STORE_T* yRe, STORE_T* yIm, int batch, int n, int m, float scale)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= n) return;
    int o = b * m + idx;
    float aR = wRe[o] * scale, aI = wIm[o] * scale;
    float cR = chRe[idx], cI = -chIm[idx];        // conjugate
    yRe[b * n + idx] = aidn_st(aR * cR - aI * cI);
    yIm[b * n + idx] = aidn_st(aR * cI + aI * cR);
}
";
            return src;
        }
    }
}
