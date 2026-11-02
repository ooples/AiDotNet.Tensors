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
    // -sum(t * (logit - max)) + log(sumExp) * sum(t): adding maxVal back first rounds the normalizer away for large
    // logits, and summing the two terms apart keeps a compiler from folding them back together.
    float shifted = 0.0, targetSum = 0.0;
    for (uint i = 0u; i < numClasses; i++) { shifted += targets[row + i] * (logits[row + i] - maxVal); targetSum += targets[row + i]; }
    rowLoss[b] = log(sumExp) * targetSum - shifted;
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

    // C = alpha * A * B + beta * C with A in the 2:4 layout of VulkanResidentKernels.Decompress2x4: two kept values per
    // block of four, and one byte per block whose low two bit pairs are the kept offsets. Only the kept values are
    // multiplied; C is not read when beta is zero, as in a BLAS GEMM.
    private const string Sparse24GemmSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer Values  { float values[]; };
layout(std430, binding = 1) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding = 2) readonly buffer DenseB  { float denseB[]; };
layout(std430, binding = 3) buffer Output           { float outC[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; uint alphaBits; uint betaBits; } pc;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.M * pc.N) return;
    uint row = gid / pc.N;
    uint col = gid - row * pc.N;
    uint blocks = pc.K / 4u;
    float acc = 0.0;
    for (uint blk = 0u; blk < blocks; blk++) {
        uint group = row * blocks + blk;
        uint packed = (offsets[group / 4u] >> ((group % 4u) * 8u)) & 255u;
        acc += values[group * 2u] * denseB[(blk * 4u + (packed & 3u)) * pc.N + col];
        acc += values[group * 2u + 1u] * denseB[(blk * 4u + ((packed >> 2u) & 3u)) * pc.N + col];
    }
    float alpha = uintBitsToFloat(pc.alphaBits), beta = uintBitsToFloat(pc.betaBits);
    outC[gid] = beta == 0.0 ? alpha * acc : alpha * acc + beta * outC[gid];
}";

    // max(0, A * B + bias) with A in the same 2:4 layout.
    private const string Sparse24GemmBiasReluSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer Values  { float values[]; };
layout(std430, binding = 1) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding = 2) readonly buffer DenseB  { float denseB[]; };
layout(std430, binding = 3) readonly buffer Bias    { float bias[]; };
layout(std430, binding = 4) writeonly buffer Output { float outC[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.M * pc.N) return;
    uint row = gid / pc.N;
    uint col = gid - row * pc.N;
    uint blocks = pc.K / 4u;
    float acc = 0.0;
    for (uint blk = 0u; blk < blocks; blk++) {
        uint group = row * blocks + blk;
        uint packed = (offsets[group / 4u] >> ((group % 4u) * 8u)) & 255u;
        acc += values[group * 2u] * denseB[(blk * 4u + (packed & 3u)) * pc.N + col];
        acc += values[group * 2u + 1u] * denseB[(blk * 4u + ((packed >> 2u) & 3u)) * pc.N + col];
    }
    outC[gid] = max(0.0, acc + bias[col]);
}";

    // The n / 2 + 1 non-negative-frequency bins of a real input (the interface's RFFT contract; the composed version wrote
    // all n bins). The twiddle index k * j is kept modulo n as it advances, so it neither overflows nor loses precision.
    private const string RealDftSource = FusedCompositeHeader + @"
layout(std430, binding = 0) readonly buffer Input   { float inputData[]; };
layout(std430, binding = 1) writeonly buffer OutRe  { float outRe[]; };
layout(std430, binding = 2) writeonly buffer OutIm  { float outIm[]; };
layout(push_constant) uniform PC { uint n; } pc;
void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k > pc.n / 2u) return;
    float step = -6.283185307179586 / float(pc.n);
    float re = 0.0, im = 0.0;
    uint phase = 0u;
    for (uint j = 0u; j < pc.n; j++) {
        float a = step * float(phase);
        re += inputData[j] * cos(a);
        im += inputData[j] * sin(a);
        phase += k;
        if (phase >= pc.n) phase -= pc.n;
    }
    outRe[k] = re;
    outIm[k] = im;
}";

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        ValidateSparse24(M, N, K);
        GlslNaryOp(Sparse24GemmSource, new[] { sparseAValues, sparseAIndices, B, C }, checked(M * N),
            new[] { (uint)M, (uint)N, (uint)K, FloatBits(alpha), FloatBits(beta) });
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias,
        int M, int N, int K)
    {
        EnsureInitialized();
        ValidateSparse24(M, N, K);
        var output = AllocateBuffer(checked(M * N));
        GlslNaryOp(Sparse24GemmBiasReluSource, new[] { sparseAValues, sparseAIndices, B, bias, output }, M * N,
            new[] { (uint)M, (uint)N, (uint)K });
        return output;
    }

    private static void ValidateSparse24(int M, int N, int K)
    {
        if (M <= 0) throw new ArgumentOutOfRangeException(nameof(M), "M must be positive.");
        if (N <= 0) throw new ArgumentOutOfRangeException(nameof(N), "N must be positive.");
        if (K <= 0 || K % 4 != 0) throw new ArgumentException($"K ({K}) must be a positive multiple of 4 for 2:4 sparsity.", nameof(K));
    }

    public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        EnsureInitialized();
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "n must be positive.");
        GlslNaryOp(RealDftSource, new[] { input, outputReal, outputImag }, n / 2 + 1, new[] { (uint)n });
    }
}
