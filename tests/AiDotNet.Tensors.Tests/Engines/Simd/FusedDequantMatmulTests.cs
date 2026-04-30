// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Issue #276 sub-feature 3 (continued): fused dequant-matmul kernel
/// numerical correctness. Compare against explicit-dequant-then-float-
/// matmul reference. The fused path must agree to within FP32 round-off
/// across the K dimension.
/// </summary>
public class FusedDequantMatmulTests
{
    [Fact]
    public void Q8MatMul_AgainstDequantThenMatmul_MatchesWithinRoundoff()
    {
        const int M = 4, K = 64, N = 8;
        var rng = new Random(7);
        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2 - 1);

        // Quantize a column-major weight (K rows × N cols).
        var wFloat = new Tensor<float>(new[] { K, N });
        var wSpan = wFloat.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        // Per-tensor scale (groupSize = total) so the fused-kernel
        // weightsScale.Scales[0] applies uniformly to every (k, j) lane.
        // Per-column / per-(k-group, col) scale is the future work.
        var qt = QuantizedTensor<sbyte>.FromFloatInt8(wFloat, groupSize: K * N);
        var raw = new sbyte[K * N];
        for (int i = 0; i < raw.Length; i++) raw[i] = (sbyte)qt.Payload[i];

        var fused = new float[M * N];
        FusedDequantMatmulKernels.Q8MatMul(act, raw, qt.Scale, fused, M, K, N);

        // Reference: dequantize then float matmul.
        var dequant = qt.Dequantize().AsSpan();
        var refOut = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += act[i * K + k] * dequant[k * N + j];
                refOut[i * N + j] = acc;
            }

        for (int i = 0; i < fused.Length; i++)
        {
            float relErr = Math.Abs(fused[i] - refOut[i]) / Math.Max(1e-6f, Math.Abs(refOut[i]));
            Assert.True(relErr < 1e-4f, $"Fused-Q8 vs ref relative error {relErr} at i={i}");
        }
    }

    [Fact]
    public void Q8MatMul_PerGroupScale_MatchesDequantThenMatmul()
    {
        // Per-group (not per-tensor) scale: groupSize=32 over the flattened
        // [K,N] row-major weight span, so scale[g] covers flat indices
        // [g*32, (g+1)*32). The kernel must look up scales by
        // (kk*N + j) / groupSize, NOT by (k-group, column). Earlier
        // versions of this kernel assumed the latter and silently produced
        // wrong results for any n > 1 with groupSize < k*n.
        const int M = 2, K = 64, N = 4;
        var rng = new Random(101);
        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2 - 1);

        var wFloat = new Tensor<float>(new[] { K, N });
        var wSpan = wFloat.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = (float)(rng.NextDouble() * 2 - 1);

        var qt = QuantizedTensor<sbyte>.FromFloatInt8(wFloat, groupSize: 32);
        Assert.Equal(8, qt.Scale.Scales.Length); // K*N=256, /32 = 8 groups
        var raw = new sbyte[K * N];
        for (int i = 0; i < raw.Length; i++) raw[i] = (sbyte)qt.Payload[i];

        var fused = new float[M * N];
        FusedDequantMatmulKernels.Q8MatMul(act, raw, qt.Scale, fused, M, K, N);

        var dequant = qt.Dequantize().AsSpan();
        var refOut = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += act[i * K + k] * dequant[k * N + j];
                refOut[i * N + j] = acc;
            }

        for (int i = 0; i < fused.Length; i++)
        {
            float relErr = Math.Abs(fused[i] - refOut[i]) / Math.Max(1e-6f, Math.Abs(refOut[i]));
            Assert.True(relErr < 1e-4f, $"Per-group Fused-Q8 vs ref relative error {relErr} at i={i}");
        }
    }

    [Fact]
    public void Q4MatMul_AgainstDequantThenMatmul_MatchesWithinRoundoff()
    {
        const int M = 2, K = 64, N = 4;
        var rng = new Random(13);
        var act = new float[M * K];
        for (int i = 0; i < act.Length; i++) act[i] = (float)(rng.NextDouble() * 2 - 1);

        var wFloat = new Tensor<float>(new[] { K, N });
        for (int i = 0; i < wFloat.Length; i++) wFloat.AsWritableSpan()[i] = (float)(rng.NextDouble() * 2 - 1);

        var qt = QuantizedTensor<PackedInt4>.FromFloatInt4(wFloat, groupSize: K * N);
        var packed = new PackedInt4[qt.Payload.Length];
        for (int i = 0; i < packed.Length; i++) packed[i] = new PackedInt4(qt.Payload[i]);

        var fused = new float[M * N];
        FusedDequantMatmulKernels.Q4MatMul(act, packed, qt.Scale, fused, M, K, N);

        var dequant = qt.Dequantize().AsSpan();
        var refOut = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0;
                for (int k = 0; k < K; k++) acc += act[i * K + k] * dequant[k * N + j];
                refOut[i * N + j] = acc;
            }

        for (int i = 0; i < fused.Length; i++)
        {
            float relErr = Math.Abs(fused[i] - refOut[i]) / Math.Max(1e-6f, Math.Abs(refOut[i]));
            Assert.True(relErr < 1e-4f, $"Fused-Q4 vs ref relative error {relErr} at i={i}");
        }
    }
}
#endif
