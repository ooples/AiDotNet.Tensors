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
// Regularized capsule squash used by CpuEngine.TensorSquash:
//     r2 = sum(x^2);  norm = sqrt(r2);  a = r2 / ((1 + r2) * (norm + epsilon));  out = a * x.
// The CPU reference accumulates in double and rounds once, so this does too — a float accumulation
// over capsuleDim would drift from the reference for long capsules.
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
    // TensorSquash uses the engine default (1e-8); SquashGpu exposes epsilon for callers that need a
    // different stability term, so the backend must preserve the value supplied by its caller.
    double norm = sqrt(r2);
    double denominator = (1.0 + r2) * (norm + (double)epsilon);
    double scale = denominator != 0.0 ? r2 / denominator : 0.0;
    for (int j = 0; j < capsuleDim; j++)
        output[off + j] = (float)(scale * (double)input[off + j]);
}

// ===========================================================================
// squash_backward — SquashBackward(gradOutput, input, gradInput, numCapsules, capsuleDim, epsilon)
//
// For y = f(r)x and f(r) = r^2 / ((1+r^2)(r+epsilon)), the exact vector-Jacobian product is
//     dx = f(r)g + (f'(r)/r)x dot(g,x)
// where f'(r)/r = (r + 2*epsilon - r^3) / ((1+r^2)^2(r+epsilon)^2).
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void squash_backward(
    const float* __restrict__ gradOutput, const float* __restrict__ input,
    float* __restrict__ gradInput, int numCapsules, int capsuleDim, float epsilon)
{
    int cap = blockIdx.x * blockDim.x + threadIdx.x;
    if (cap >= numCapsules) return;

    int off = cap * capsuleDim;
    double r2 = 0.0;
    double dot = 0.0;
    for (int j = 0; j < capsuleDim; j++)
    {
        double x = (double)input[off + j];
        r2 += x * x;
        dot += x * (double)gradOutput[off + j];
    }
    double norm = sqrt(r2);
    double normPlusEpsilon = norm + (double)epsilon;
    double onePlusR2 = 1.0 + r2;
    double denominator = onePlusR2 * normPlusEpsilon;
    double scale = denominator != 0.0 ? r2 / denominator : 0.0;
    double coefficient = denominator != 0.0
        ? (norm + 2.0 * (double)epsilon - r2 * norm) / (denominator * denominator)
        : 0.0;
    for (int j = 0; j < capsuleDim; j++)
        gradInput[off + j] = (float)(scale * (double)gradOutput[off + j]
            + coefficient * (double)input[off + j] * dot);
}

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

// ===========================================================================
// csr_segmented_{max,min,stddev} — CsrSegmented*(colIndices, rowPointers, input, output, M, K, N[, eps])
//
// PORTED FROM THE WebGPU REFERENCE (WebGpuKernels csr_segmented_*), not invented. Segmented reduction
// over a CSR row's gathered rows: for output element (row, j), walk the row's nonzero column indices
// and reduce input[col * N + j].
//
// The EMPTY-ROW result is 0.0, which is the reference's explicit contract
// (`if (start >= end) { output[idx] = 0.0; return; }`) — NOT the -INFINITY that OpenCL's scatter_max
// uses for empty groups. These two conventions differ; this follows the one its own siblings use.
//
// CudaBackend launches these 2D as gridX = M rows, gridY = ceil(N / block), so row comes from
// blockIdx.x and j from the y-dimension, unlike the reference's flat 1D index. The col/row-pointer
// buffers are float-typed on the device and hold integral values, matching the other CSR kernels.
// ===========================================================================
extern ""C"" __global__ __launch_bounds__(256) void csr_segmented_max(
    const float* __restrict__ colIndices, const float* __restrict__ rowPointers,
    const float* __restrict__ input, float* __restrict__ output, int M, int K, int N)
{
    int row = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= M || j >= N) return;

    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    if (start >= end) { output[idx] = 0.0f; return; }

    float maxVal = -3.402823e+38f;
    for (int i = start; i < end; i++)
    {
        int col = (int)colIndices[i];
        maxVal = fmaxf(maxVal, input[col * N + j]);
    }
    output[idx] = maxVal;
}

extern ""C"" __global__ __launch_bounds__(256) void csr_segmented_min(
    const float* __restrict__ colIndices, const float* __restrict__ rowPointers,
    const float* __restrict__ input, float* __restrict__ output, int M, int K, int N)
{
    int row = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= M || j >= N) return;

    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    if (start >= end) { output[idx] = 0.0f; return; }

    float minVal = 3.402823e+38f;
    for (int i = start; i < end; i++)
    {
        int col = (int)colIndices[i];
        minVal = fminf(minVal, input[col * N + j]);
    }
    output[idx] = minVal;
}

extern ""C"" __global__ __launch_bounds__(256) void csr_segmented_stddev(
    const float* __restrict__ colIndices, const float* __restrict__ rowPointers,
    const float* __restrict__ input, float* __restrict__ output, int M, int K, int N, float epsilon)
{
    int row = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= M || j >= N) return;

    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    int count = end - start;
    if (count <= 0) { output[idx] = 0.0f; return; }

    // Two-pass mean then variance, exactly as the reference does (not a sum-of-squares shortcut,
    // which would differ in rounding).
    float mean = 0.0f;
    for (int i = start; i < end; i++)
    {
        int col = (int)colIndices[i];
        mean += input[col * N + j];
    }
    mean /= (float)count;

    float varAcc = 0.0f;
    for (int i = start; i < end; i++)
    {
        int col = (int)colIndices[i];
        float d = input[col * N + j] - mean;
        varAcc += d * d;
    }
    output[idx] = sqrtf(varAcc / (float)count + epsilon);
}
";
        }

        public static string[] GetKernelNames() => new[]
        {
            "copy_2d_strided",
            "squash",
            "squash_backward",
            "var_axis",
            "csr_segmented_max",
            "csr_segmented_min",
            "csr_segmented_stddev",
        };
    }
}
