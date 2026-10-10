// Copyright (c) AiDotNet. All rights reserved.
// Device kernels for PyTorch-parity elementwise ops that computed on the host (#1112).
//
// Each kernel mirrors the CPU reference lambda in CpuEngine.TorchMisc / TorchSpecial exactly, including the NaN
// behaviour of the .NET Math calls it replaces: Math.Max / Math.Min PROPAGATE a NaN, whereas CUDA's fmaxf / fminf
// return the non-NaN operand, so the clamps below test for NaN explicitly. Compiled WITHOUT --use_fast_math: erff,
// log1pf, expf and sinpif must be the accurate libm versions to agree with the CPU's double-precision reference.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>CUDA source for the PyTorch-parity elementwise kernels.</summary>
    internal static class CudaTorchParityElementwiseKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

#define PARITY_INDEX int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;

// celu: x > 0 ? x : alpha * (exp(x / alpha) - 1)
extern ""C"" __global__ __launch_bounds__(256) void parity_celu(const float* __restrict__ x, float* __restrict__ y, int n, float alpha, float unused)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v > 0.0f ? v : alpha * (expf(v / alpha) - 1.0f);
}

// hardtanh: min(max(x, lo), hi), NaN propagating (Math.Max / Math.Min semantics)
extern ""C"" __global__ __launch_bounds__(256) void parity_hardtanh(const float* __restrict__ x, float* __restrict__ y, int n, float lo, float hi)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = isnan(v) ? v : (v < lo ? lo : (v > hi ? hi : v));
}

// logsigmoid: min(x, 0) - log(1 + exp(-|x|)) (the stable form the CPU uses), NaN propagating
extern ""C"" __global__ __launch_bounds__(256) void parity_logsigmoid(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = isnan(v) ? v : (v < 0.0f ? v : 0.0f) - log1pf(expf(-fabsf(v)));
}

// softsign: x / (1 + |x|)
extern ""C"" __global__ __launch_bounds__(256) void parity_softsign(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v / (1.0f + fabsf(v));
}

// logit: c = min(max(x, lo), hi) (lo/hi are -inf/+inf without eps), log(c / (1 - c)), NaN propagating
extern ""C"" __global__ __launch_bounds__(256) void parity_logit(const float* __restrict__ x, float* __restrict__ y, int n, float lo, float hi)
{
    PARITY_INDEX
    float v = x[i];
    float c = isnan(v) ? v : (v < lo ? lo : (v > hi ? hi : v));
    y[i] = logf(c / (1.0f - c));
}

// sinc: x == 0 ? 1 : isinf(x) ? 0 : sin(pi x) / (pi x)
extern ""C"" __global__ __launch_bounds__(256) void parity_sinc(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = v == 0.0f ? 1.0f : (isinf(v) ? 0.0f : sinpif(v) / (3.14159265358979323846f * v));
}

// erf
extern ""C"" __global__ __launch_bounds__(256) void parity_erf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    y[i] = erff(x[i]);
}

// fmax / fmin: NaN-ignoring max/min (isnan(a) ? b : isnan(b) ? a : max) -- exactly CUDA's fmaxf / fminf
extern ""C"" __global__ __launch_bounds__(256) void parity_fmax(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = fmaxf(a[i], b[i]);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_fmin(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = fminf(a[i], b[i]);
}

// heaviside: isnan(x) ? x : x < 0 ? 0 : x > 0 ? 1 : values
extern ""C"" __global__ __launch_bounds__(256) void parity_heaviside(const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ y, int n)
{
    PARITY_INDEX
    float t = x[i];
    y[i] = isnan(t) ? t : (t < 0.0f ? 0.0f : (t > 0.0f ? 1.0f : v[i]));
}

// greater_equal / less_equal: 1 or 0 (a comparison with NaN is false, as in .NET)
extern ""C"" __global__ __launch_bounds__(256) void parity_greater_equal(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = a[i] >= b[i] ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_less_equal(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n)
{
    PARITY_INDEX
    y[i] = a[i] <= b[i] ? 1.0f : 0.0f;
}

// Special functions and indicators (#1112 tier 4): CUDA libm's j0f / j1f / y0f / y1f / cyl_bessel_i0f / i1f /
// erfcxf / normcdff / normcdfinvf share the CPU references' edge values (Y_n: NaN for x < 0, -inf at 0, 0 at +inf;
// J_n: 0 at +-inf; ndtri: NaN outside [0, 1], -+inf at 0 / 1). entr mirrors its CPU lambda.
extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_j0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = j0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_j1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = j1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_y0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = y0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_y1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = y1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_i0(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = cyl_bessel_i0f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_bessel_i1(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = cyl_bessel_i1f(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_erfcx(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = erfcxf(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_ndtr(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = normcdff(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_ndtri(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = normcdfinvf(v);
}

extern ""C"" __global__ __launch_bounds__(256) void parity_entr(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = isnan(v) ? v : (v > 0.0f ? -v * logf(v) : (v == 0.0f ? 0.0f : -INFINITY));
}

extern ""C"" __global__ __launch_bounds__(256) void parity_signbit(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = signbit(v) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_isposinf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = (isinf(v) && v > 0.0f) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void parity_isneginf(const float* __restrict__ x, float* __restrict__ y, int n, float unused0, float unused1)
{
    PARITY_INDEX
    float v = x[i];
    y[i] = (isinf(v) && v < 0.0f) ? 1.0f : 0.0f;
}
";
        }

        public static string[] GetKernelNames() => new[]
        {
            "parity_celu", "parity_hardtanh", "parity_logsigmoid", "parity_softsign", "parity_logit", "parity_sinc",
            "parity_erf", "parity_fmax", "parity_fmin", "parity_heaviside", "parity_greater_equal", "parity_less_equal",
            "parity_bessel_j0", "parity_bessel_j1", "parity_bessel_y0", "parity_bessel_y1", "parity_bessel_i0", "parity_bessel_i1", "parity_erfcx", "parity_ndtr", "parity_ndtri", "parity_entr", "parity_signbit", "parity_isposinf", "parity_isneginf",
        };
    }
}
