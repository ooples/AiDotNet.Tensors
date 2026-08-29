// Copyright (c) AiDotNet. All rights reserved.
// Metal Shading Language (MSL) kernel for categorical sampling — the device twin of
// CpuEngine.TensorCategoricalSample and of the OpenCL/CUDA/HIP/Vulkan kernels.
//
// WHY THIS IS NOT A LINE-FOR-LINE PORT OF THE OTHERS.
// The CPU reference, and every other backend, accumulates the probability sum and the cumulative
// walk in `double`. MSL HAS NO `double` TYPE — it is not a capability the device may or may not
// advertise, it is absent from the language. Accumulating in `float` instead is not an option: the
// running total drifts, a target sitting near a bucket edge selects the neighbouring category, and
// the exact-parity test treats a single differing category as a hard mismatch.
//
// So the accumulation is carried in a TWO-FLOAT (double-single) expansion: a hi/lo pair whose
// combined significand is roughly 48 bits against double's 53. Knuth's two_sum is exact under
// round-to-nearest, so summing N float inputs this way is exact whenever the true sum needs no more
// than ~48 bits — which covers any realistic class count, since each addend carries only 24. That
// makes this kernel agree with the double reference rather than merely approximate it.
//
// REQUIRES IEEE SEMANTICS, AND FAILS CLOSED WITHOUT THEM. two_sum and the fma-based product are
// exact only if the compiler does not reassociate them, and Metal compiles with fast math ENABLED
// by default. This library is therefore compiled with MTLCompileOptions requesting strict floating
// point (setMathMode: / setFastMathEnabled:, whichever this runtime implements — see
// MetalDevice.TryCreateStrictFloatingPointOptions), and if NEITHER can be established the library
// is not compiled at all and the route is not advertised. The pragma below is belt-and-braces, not
// the mechanism: an older compiler treats an unknown pragma as a warning and keeps fast math, so
// the pragma alone is not evidence of anything.
//
// Failing closed is deliberate. The alternative — advertising the route and hoping — degrades to
// plain float accumulation with no diagnostic, which still produces a valid one-hot and so passes
// every smoke test while disagreeing with the CPU reference at bucket edges.
//
// NOT EXECUTED IN VERIFICATION: Metal runs only on macOS and this was written on Windows, so this
// kernel is unverified against the CPU oracle. The algorithm mirrors the OpenCL kernel, which IS
// verified, with the accumulation type as the single deliberate difference.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    /// <summary>
    /// MSL implementation of categorical sampling, matching the managed reference's category
    /// selection via compensated two-float accumulation (MSL has no <c>double</c>).
    /// </summary>
    internal static class MetalCategoricalKernels
    {
        public static string[] GetKernelNames() => new[] { "categorical_sample" };

        public const string Source = @"
#include <metal_stdlib>
using namespace metal;

// Compensated summation below is only exact under non-reassociating FP math.
#pragma METAL fp math_mode(safe)

// Knuth two_sum: exact under round-to-nearest. Returns (sum, error) with sum + error == a + b.
inline float2 two_sum(float a, float b)
{
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return float2(s, err);
}

// Add a float to a two-float accumulator, renormalising so the pair stays non-overlapping.
inline float2 df_add(float2 acc, float v)
{
    float2 t = two_sum(acc.x, v);
    return two_sum(t.x, t.y + acc.y);
}

// Multiply a two-float by a float. fma recovers the exact rounding error of the head product.
inline float2 df_mul(float2 a, float b)
{
    float p = a.x * b;
    float e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return two_sum(p, e);
}

// Lexicographic compare of two non-overlapping pairs is a compare of the exact values.
inline bool df_less(float2 a, float2 b)
{
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

// StatelessRandom.Uniform01(seed, index) — identical constants to the managed helper and to every
// other backend's kernel. Keying on the ROW index lets a thread compute its own uniform without
// replaying the rows before it.
inline float stateless_uniform01(uint seed32, uint index)
{
    uint state = index * 747796405u + seed32 + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    uint draw = (word >> 22) ^ word;
    return float(draw >> 8) * (1.0f / 16777216.0f);
}

kernel void categorical_sample(
    device const float* probabilities [[buffer(0)]],
    device float* oneHot             [[buffer(1)]],
    constant int& rows               [[buffer(2)]],
    constant int& classes            [[buffer(3)]],
    constant uint& seedLo            [[buffer(4)]],
    constant uint& seedHi            [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]])
{
    int row = int(gid);
    if (row >= rows) return;

    uint seed32 = seedLo ^ seedHi;
    float u = stateless_uniform01(seed32, uint(row));

    int offset = row * classes;

    float2 sum = float2(0.0f, 0.0f);
    for (int c = 0; c < classes; c++) sum = df_add(sum, probabilities[offset + c]);

    float2 target = df_mul(sum, u);
    float2 cumulative = float2(0.0f, 0.0f);
    int selected = classes - 1;
    for (int c = 0; c < classes; c++)
    {
        cumulative = df_add(cumulative, probabilities[offset + c]);
        if (df_less(target, cumulative)) { selected = c; break; }
    }

    for (int c = 0; c < classes; c++) oneHot[offset + c] = 0.0f;
    oneHot[offset + selected] = 1.0f;
}
";
    }
}
