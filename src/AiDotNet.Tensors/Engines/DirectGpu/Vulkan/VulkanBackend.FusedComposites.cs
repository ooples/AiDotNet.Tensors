// Copyright (c) CrowdTrainer LLC. All rights reserved.
// One kernel each for members that were compositions of other backend members (TENS0008, #1638): the split-complex
// element-wise family (multiply was six launches through four scratch buffers), the batched dot product (a multiply into
// a temporary, a full synchronize, then a row sum) and the CSR SpMM with its bias added in the same pass.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    private const string FusedCompositeHeader = @"#version 450
layout(local_size_x = 256) in;
";

    /// <summary>Selector of <see cref="SplitComplexBinarySource"/>.</summary>
    private enum SplitComplexBinaryOp : uint { Multiply = 0, Add = 1, CrossSpectral = 2 }

    /// <summary>Selector of <see cref="SplitComplexUnarySource"/>.</summary>
    private enum SplitComplexUnaryOp : uint { Conjugate = 0, Scale = 1 }

    // (aR, aI) op (bR, bI) -> (oR, oI). CrossSpectral is a * conj(b). The products are summed in the order the composed
    // launches summed them.
    private const string SplitComplexBinarySource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer AR { float aR[]; };
layout(std430, binding = 1) readonly buffer AI { float aI[]; };
layout(std430, binding = 2) readonly buffer BR { float bR[]; };
layout(std430, binding = 3) readonly buffer BI { float bI[]; };
layout(std430, binding = 4) writeonly buffer OR { float oR[]; };
layout(std430, binding = 5) writeonly buffer OI { float oI[]; };
layout(push_constant) uniform Params { uint n; uint op; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    float xr = aR[i], xi = aI[i], yr = bR[i], yi = bI[i];
    if (op == 0u) { oR[i] = xr * yr - xi * yi; oI[i] = xr * yi + xi * yr; }
    else if (op == 1u) { oR[i] = xr + yr; oI[i] = xi + yi; }
    else { oR[i] = xr * yr + xi * yi; oI[i] = xi * yr - xr * yi; }
}";

    // (iR, iI) -> (oR, oI): conjugate, or both parts times s.
    private const string SplitComplexUnarySource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer IR { float iR[]; };
layout(std430, binding = 1) readonly buffer II { float iI[]; };
layout(std430, binding = 2) writeonly buffer OR { float oR[]; };
layout(std430, binding = 3) writeonly buffer OI { float oI[]; };
layout(push_constant) uniform Params { uint n; uint op; float s; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    if (op == 0u) { oR[i] = iR[i]; oI[i] = -iI[i]; }
    else { oR[i] = iR[i] * s; oI[i] = iI[i] * s; }
}";

    private const string SplitComplexMagnitudeSquaredSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer IR { float iR[]; };
layout(std430, binding = 1) readonly buffer II { float iI[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    o[i] = iR[i] * iR[i] + iI[i] * iI[i];
}";

    /// <summary>Threads per row of <see cref="BatchedDotProductSource"/> (its local size).</summary>
    private const int BatchedDotGroupSize = 256;

    // One workgroup per row: strided partial dot products, then a shared-memory tree.
    private const string BatchedDotProductSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer R { float r[]; };
layout(push_constant) uniform Params { uint batchSize; uint vecSize; };
shared float partial[256];
void main() {
    uint row = gl_WorkGroupID.x;
    uint t = gl_LocalInvocationID.x;
    float s = 0.0;
    if (row < batchSize) {
        uint base = row * vecSize;
        for (uint i = t; i < vecSize; i += 256u) s += a[base + i] * b[base + i];
    }
    partial[t] = s;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride) partial[t] += partial[t + stride];
        barrier();
    }
    if (t == 0u && row < batchSize) r[row] = partial[0];
}";

    // -sum(target * log_softmax(row)) per row.
    private const string DenseCrossEntropyRowsSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer P { float logits[]; };
layout(std430, binding = 1) readonly buffer T { float targets[]; };
layout(std430, binding = 2) writeonly buffer L { float rowLoss[]; };
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    uint row = b * numClasses;
    float maxVal = -3.402823466e+38;
    for (uint i = 0u; i < numClasses; i++) maxVal = max(maxVal, logits[row + i]);
    float sumExp = 0.0;
    for (uint i = 0u; i < numClasses; i++) sumExp += exp(logits[row + i] - maxVal);
    float logSum = maxVal + log(sumExp);
    float loss = 0.0;
    for (uint i = 0u; i < numClasses; i++) loss -= targets[row + i] * (logits[row + i] - logSum);
    rowLoss[b] = loss;
}";

    // (softmax(logits) - target) / batch per element, the row's max and sum recomputed by each element (as CUDA and OpenCL).
    private const string DenseCrossEntropyBackwardSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer P { float logits[]; };
layout(std430, binding = 1) readonly buffer T { float targets[]; };
layout(std430, binding = 2) writeonly buffer G { float grad[]; };
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize * numClasses) return;
    uint row = (idx / numClasses) * numClasses;
    float maxVal = -3.402823466e+38;
    for (uint i = 0u; i < numClasses; i++) maxVal = max(maxVal, logits[row + i]);
    float sumExp = 0.0;
    for (uint i = 0u; i < numClasses; i++) sumExp += exp(logits[row + i] - maxVal);
    grad[idx] = (exp(logits[idx] - maxVal) / sumExp - targets[idx]) / float(batchSize);
}";

    // VulkanSparseKernels.CsrSpMM with the bias of the output column added before the store.
    private const string CsrSpMMBiasSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer CsrValues { float csrValues[]; };
layout(std430, binding = 1) readonly buffer CsrColIdx { int csrColIndices[]; };
layout(std430, binding = 2) readonly buffer CsrRowPtr { int csrRowPointers[]; };
layout(std430, binding = 3) readonly buffer DenseB    { float denseB[]; };
layout(std430, binding = 4) readonly buffer Bias      { float bias[]; };
layout(std430, binding = 5) writeonly buffer Output   { float outC[]; };
layout(push_constant) uniform PC { uint M; uint K; uint N; uint nnz; } pc;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.M * pc.N) return;
    uint row = gid / pc.N;
    uint col = gid - row * pc.N;
    float sum = 0.0;
    for (int i = csrRowPointers[row]; i < csrRowPointers[row + 1u]; i++)
        sum += csrValues[i] * denseB[uint(csrColIndices[i]) * pc.N + col];
    outC[gid] = sum + bias[col];
}";

    public void SplitComplexMultiply(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(aR, aI, bR, bI, oR, oI, n, SplitComplexBinaryOp.Multiply);

    public void SplitComplexAdd(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(aR, aI, bR, bI, oR, oI, n, SplitComplexBinaryOp.Add);

    public void SplitComplexCrossSpectral(IGpuBuffer xR, IGpuBuffer xI, IGpuBuffer yR, IGpuBuffer yI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexBinary(xR, xI, yR, yI, oR, oI, n, SplitComplexBinaryOp.CrossSpectral);

    public void SplitComplexConjugate(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n)
        => SplitComplexUnary(iR, iI, oR, oI, n, SplitComplexUnaryOp.Conjugate, 0f);

    public void SplitComplexScale(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, float s, int n)
        => SplitComplexUnary(iR, iI, oR, oI, n, SplitComplexUnaryOp.Scale, s);

    public void SplitComplexMagnitudeSquared(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n)
    {
        if (n <= 0) return;
        GlslNaryOp(SplitComplexMagnitudeSquaredSource, new[] { iR, iI, o }, n, new[] { (uint)n });
    }

    private void SplitComplexBinary(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI,
        int n, SplitComplexBinaryOp op)
    {
        if (n <= 0) return;
        GlslNaryOp(SplitComplexBinarySource, new[] { aR, aI, bR, bI, oR, oI }, n, new[] { (uint)n, (uint)op });
    }

    private void SplitComplexUnary(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n, SplitComplexUnaryOp op, float s)
    {
        if (n <= 0) return;
        GlslNaryOp(SplitComplexUnarySource, new[] { iR, iI, oR, oI }, n, new[] { (uint)n, (uint)op, FloatBits(s) });
    }

    public void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer result, int batchSize, int vecSize)
    {
        EnsureInitialized();
        if (batchSize <= 0 || vecSize <= 0) return;
        if ((long)batchSize * vecSize > a.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({(long)batchSize * vecSize}) exceeds buffer A length ({a.Size}).");
        if ((long)batchSize * vecSize > b.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize*vecSize ({(long)batchSize * vecSize}) exceeds buffer B length ({b.Size}).");
        if (batchSize > result.Size) throw new ArgumentOutOfRangeException(nameof(batchSize), $"batchSize ({batchSize}) exceeds result length ({result.Size}).");
        GlslNaryOp(BatchedDotProductSource, new[] { a, b, result }, checked(batchSize * BatchedDotGroupSize),
            new[] { (uint)batchSize, (uint)vecSize }, BatchedDotGroupSize);
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB,
        IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        if (M <= 0 || N <= 0) return;
        long total = (long)M * N;
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(M),
                $"CSR SpMM output too large for a single Vulkan dispatch (M*N = {total}). Tile the problem or route to a vendor SpMM backend.");
        GlslNaryOp(CsrSpMMBiasSource, new[] { csrValues, csrColIndices, csrRowPointers, denseB, bias, output }, (int)total,
            new[] { (uint)M, (uint)K, (uint)N, (uint)nnz });
    }
}
