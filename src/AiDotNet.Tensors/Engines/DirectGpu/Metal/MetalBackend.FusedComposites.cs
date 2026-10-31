// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner (metal.yml) -- the dev box has no Apple GPU.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// One kernel each for members that were compositions of other backend members (TENS0008, #1638): the base-2/base-10
// exponentials and logarithms, expm1 and log1p (each a scale or offset launch plus exp/log), the MSE and cross-entropy
// gradients (a subtract then a scale), the softplus gradient (a sigmoid into the output, then a multiply -- wrong when
// the output aliased the incoming gradient) and the CSR SpMM with its bias. Compiled with fast math off: expm1 and log1p
// use Kahan's correction, which needs (1 + x) - 1 evaluated as written; exp(x) - 1 and log(1 + x) lose every digit of
// a small x.
public sealed partial class MetalBackend
{
    private const string FusedCompositesLibraryName = "FusedComposites";

    /// <summary>Selector of the unary_math kernel.</summary>
    private enum UnaryMathOp : uint { Exp2 = 0, Exp10 = 1, ExpM1 = 2, Log2 = 3, Log10 = 4, Log1P = 5 }

    internal const string FusedCompositesSource = @"
#include <metal_stdlib>
using namespace metal;

inline float fc_expm1(float x)
{
    float u = exp(x);
    if (u == 1.0f) return x;
    if (isinf(u)) return u;
    float um1 = u - 1.0f;
    if (um1 == -1.0f) return -1.0f;
    return um1 * x / log(u);
}

inline float fc_log1p(float x)
{
    float u = 1.0f + x;
    if (u == 1.0f) return x;
    if (isinf(u)) return u;
    return log(u) * x / (u - 1.0f);
}

kernel void unary_math(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = A[gid];
    float y;
    switch (op) {
        case 0u: y = exp2(x); break;
        case 1u: y = exp10(x); break;
        case 2u: y = fc_expm1(x); break;
        case 3u: y = log2(x); break;
        case 4u: y = log10(x); break;
        default: y = fc_log1p(x); break;
    }
    B[gid] = y;
}

// G = (P - T) * scale: the MSE gradient (scale 2 / n) and the softmax cross-entropy gradient (scale 1 / batch).
kernel void scaled_difference(
    device const float* P [[buffer(0)]],
    device const float* T [[buffer(1)]],
    device float* G [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) G[gid] = (P[gid] - T[gid]) * scale;
}

// dX = dY * sigmoid(X), the activation library's numerically stable sigmoid. dX may be dY.
kernel void softplus_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    float s;
    if (x >= 0.0f) {
        s = 1.0f / (1.0f + exp(-x));
    } else {
        float e = exp(x);
        s = e / (1.0f + e);
    }
    gradInput[gid] = gradOutput[gid] * s;
}

// sparse_csr_spmm with the bias of the output column added before the store.
kernel void csr_spmm_bias(
    device const float* csrValues       [[buffer(0)]],
    device const int*   csrColIndices   [[buffer(1)]],
    device const int*   csrRowPointers  [[buffer(2)]],
    device const float* denseB          [[buffer(3)]],
    device const float* bias            [[buffer(4)]],
    device float*       outC            [[buffer(5)]],
    constant int&       M               [[buffer(6)]],
    constant int&       N               [[buffer(7)]],
    uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= M * N) return;
    int row = (int)gid / N;
    int col = (int)gid - row * N;
    float sum = 0.0f;
    for (int i = csrRowPointers[row]; i < csrRowPointers[row + 1]; i++)
        sum += csrValues[i] * denseB[csrColIndices[i] * N + col];
    outC[gid] = sum + bias[col];
}
";

    private IntPtr _fusedCompositesLibrary;

    private MetalPipelineState FusedCompositesPipeline(string kernelName)
    {
        if (_fusedCompositesLibrary == IntPtr.Zero)
        {
            lock (_tunedSlotsLock)
            {
                if (_fusedCompositesLibrary == IntPtr.Zero)
                    _fusedCompositesLibrary = _shaderLibrary.CompileLibrary(FusedCompositesLibraryName, FusedCompositesSource,
                        requireStrictFloatingPoint: true);
            }
        }
        return GetPipeline(FusedCompositesLibraryName, _fusedCompositesLibrary, kernelName);
    }

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.Exp2);

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.Exp10);

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.ExpM1);

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.Log2);

    public void Log10(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.Log10);

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size) => UnaryMath(A, B, size, UnaryMathOp.Log1P);

    private void UnaryMath(IGpuBuffer A, IGpuBuffer B, int size, UnaryMathOp op)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        var pipeline = FusedCompositesPipeline("unary_math");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)A, 0);
        encoder.SetBuffer((MetalGpuBuffer)B, 1);
        encoder.SetBytes((uint)size, 2);
        encoder.SetBytes((uint)op, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => ScaledDifference(predictions, targets, gradInput, size, 2.0f / size);

    public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
        => ScaledDifference(predictions, targets, gradInput, batchSize * numClasses, 1.0f / batchSize);

    private void ScaledDifference(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float scale)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        var pipeline = FusedCompositesPipeline("scaled_difference");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)predictions, 0);
        encoder.SetBuffer((MetalGpuBuffer)targets, 1);
        encoder.SetBuffer((MetalGpuBuffer)gradInput, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.SetBytes(scale, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        var pipeline = FusedCompositesPipeline("softplus_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)gradOutput, 0);
        encoder.SetBuffer((MetalGpuBuffer)input, 1);
        encoder.SetBuffer((MetalGpuBuffer)gradInput, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB,
        IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        ThrowIfDisposed();
        if (M <= 0 || N <= 0) return;
        long total = (long)M * N;
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(M), $"CSR SpMM output too large for a single Metal dispatch (M*N = {total}).");
        var pipeline = FusedCompositesPipeline("csr_spmm_bias");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch((int)total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)csrValues, 0);
        encoder.SetBuffer((MetalGpuBuffer)csrColIndices, 1);
        encoder.SetBuffer((MetalGpuBuffer)csrRowPointers, 2);
        encoder.SetBuffer((MetalGpuBuffer)denseB, 3);
        encoder.SetBuffer((MetalGpuBuffer)bias, 4);
        encoder.SetBuffer((MetalGpuBuffer)output, 5);
        encoder.SetBytes(M, 6);
        encoder.SetBytes(N, 7);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
