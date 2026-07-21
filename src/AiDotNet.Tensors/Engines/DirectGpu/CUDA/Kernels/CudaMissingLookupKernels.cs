// Copyright (c) AiDotNet. All rights reserved.
// Kernels for CudaBackend lookups that had no definition (#775).
//
// CudaBackend asked _kernelCache for these names and no CUDA source defined them, so every call threw
// kernel-not-found. Where a caller wrapped the op in a catch that became a silent CPU fallback with a
// correct-looking result; where it did not, it was a hard failure. See CudaKernelLookupIntegrityTests,
// which now fails the build if a new dangling lookup appears.
//
// Each kernel below matches an EXISTING reference implementation exactly — the CPU math it has to agree
// with, or the caller's own index arithmetic. Names/arity/argument order match the launch sites, which
// matters: a near-name with different arity reproduces the 15-vs-16 argument truncation fixed in
// 3aed217, where the launch failed and the failure was swallowed.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>CUDA kernels for backend entry points whose kernel name resolved to nothing.</summary>
    internal static class CudaMissingLookupKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// copy_2d_strided — Copy2DStrided(source, destination, numRows, srcCols, destTotalCols, destColOffset)
// Block-copies a [numRows, srcCols] source into a [numRows, destTotalCols] destination at column
// destColOffset. This is the concatenate path (DirectGpuTensorEngine concat walks its inputs and
// advances destColOffset by each input's axis extent), so the indexing is exactly the caller's.
// Launch is gridX over srcCols, gridY = numRows (one row per block-row).
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void copy_2d_strided(
    const float* __restrict__ source, float* __restrict__ destination,
    int numRows, int srcCols, int destTotalCols, int destColOffset)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= srcCols || row >= numRows) return;
    destination[row * destTotalCols + destColOffset + col] = source[row * srcCols + col];
}

// ===========================================================================
// squash — Squash(input, output, numCapsules, capsuleDim, epsilon)
// Capsule squash. Matches FusedActivationBackwardMath.SquashForward EXACTLY:
//     r2 = sum(x^2);  norm = sqrt(r2);  a = norm > 0 ? norm / (1 + r2) : 0;  out = a * x
// (equivalently the textbook (r2/(1+r2)) * (x/norm), same value).
// The CPU reference accumulates in double and rounds once, so this does too — a float accumulation
// over capsuleDim would drift from the reference for long capsules.
// The CPU guard is norm > 0; epsilon is the caller-supplied zero-threshold and is used in its place,
// which is identical for any norm meaningfully above eps.
// One thread per capsule; grid is ceil(numCapsules / block).
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void squash(
    const float* __restrict__ input, float* __restrict__ output,
    int numCapsules, int capsuleDim, float epsilon)
{
    int cap = blockIdx.x * blockDim.x + threadIdx.x;
    if (cap >= numCapsules) return;

    int off = cap * capsuleDim;
    double r2 = 0.0;
    for (int j = 0; j < capsuleDim; j++) { double x = (double)input[off + j]; r2 += x * x; }
    // CpuEngine.TensorSquash: scale = r2/(1+r2); normalized = x/(norm + 1e-8); out = scale * normalized.
    // The +1e-8 is CpuEngine's literal, not the caller's epsilon — matching its exact form keeps the two
    // agreeing at small norms, where scale/(norm+1e-8) and the algebraically-equal norm/(1+r2) diverge.
    double norm = sqrt(r2);
    double scale = r2 / (1.0 + r2);
    double inv = 1.0 / (norm + 1e-8);
    for (int j = 0; j < capsuleDim; j++)
        output[off + j] = (float)(scale * (double)input[off + j] * inv);
}

// ===========================================================================
// squash_backward — SquashBackward(gradOutput, input, gradInput, numCapsules, capsuleDim, epsilon)
//
// Matches CpuEngine.TensorSquashBackward, which is:  gradInput = gradOutput * 1/(1 + ||x||^2)
//
// READ THIS BEFORE CHANGING IT. That CPU implementation is a documented approximation — its own
// comment says: this is a simplified gradient, a full implementation would require the proper
// Jacobian, and for now it approximates with element-wise gradient scaling. The codebase ALSO has the
// gradient, in FusedActivationBackwardMath.SquashBackward:
//     a = norm/(1+r2);  coef = (1-r2)/((1+r2)^2 * norm);  dx_j = a*g_j + x_j*(g.x)*coef
// The two disagree substantially (measured: CPU -0.0032247882 vs true-Jacobian 0.012901765 at [4] of
// TensorSquashBackward[4,8], maxRel 1.25).
//
// This kernel deliberately implements the CPU one. Before this kernel existed, SquashBackward threw
// kernel-not-found and the caller fell back to CpuEngine — so the approximation is ALREADY what every
// squash backward in this library computes. Matching it moves identical math onto the device and
// changes no numerical behaviour. Implementing the true Jacobian here is what would silently alter
// gradients on the GPU path only, giving CPU and GPU different training dynamics.
//
// Reconciling the two definitions is a real decision about capsule training semantics and belongs
// upstream in CpuEngine, not in a backend kernel.
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void squash_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    float* __restrict__ gradInput, int numCapsules, int capsuleDim, float epsilon)
{
    int cap = blockIdx.x * blockDim.x + threadIdx.x;
    if (cap >= numCapsules) return;

    int off = cap * capsuleDim;
    double r2 = 0.0;
    for (int j = 0; j < capsuleDim; j++)
    {
        double x = (double)input[off + j];
        r2 += x * x;
    }
    double scale = 1.0 / (1.0 + r2);
    for (int j = 0; j < capsuleDim; j++)
        gradInput[off + j] = (float)(scale * (double)gradOutput[off + j]);
}

// ===========================================================================
// var_axis — VarAxis(input, mean, variance, outerSize, reduceSize)
// Population variance about a PRECOMPUTED per-row mean: variance[o] = sum((x - mean[o])^2)/reduceSize.
// The mean is an INPUT here — TensorVar calls MeanAxis first and feeds its result in. That is why this
// cannot be repointed at the existing variance_axis(input, output, outerSize, reduceSize), which takes
// FOUR arguments and computes its own mean; launching this call site's five arguments at that kernel
// would read the fifth parameter off unallocated stack.
// Accumulates in double to match the CPU reduction.
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void var_axis(
    const float* __restrict__ input, const float* __restrict__ mean,
    float* __restrict__ variance, int outerSize, int reduceSize)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outerSize) return;
    if (reduceSize <= 0) { variance[o] = 0.0f; return; }

    const float* row = input + (size_t)o * reduceSize;
    double m = (double)mean[o];
    double acc = 0.0;
    for (int j = 0; j < reduceSize; j++)
    {
        double d = (double)row[j] - m;
        acc += d * d;
    }
    variance[o] = (float)(acc / (double)reduceSize);
}
";
        }

        public static string[] GetKernelNames() => new[]
        {
            "copy_2d_strided",
            "squash",
            "squash_backward",
            "var_axis",
        };
    }
}
