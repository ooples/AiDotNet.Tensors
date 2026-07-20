// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the two GPU-resident decoder primitives added for LLM serving:
//   * rope_interleaved            — fused interleaved (GPT-NeoX / LLaMA / GGML) RoPE
//   * scaled_dot_product_attention with numKVHeads — Grouped-Query Attention (broadcast K/V)
// Each OpenCL kernel is compared against a scalar CPU reference implementing the exact contract.
// Skips when no OpenCL device is available, unless AIDOTNET_REQUIRE_GPU_TESTS=1.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class RopeGqaOpenClTests : IDisposable
{
    private readonly OpenClBackend? _backend;
    private readonly bool _ready;
    private readonly Exception? _initException;

    public RopeGqaOpenClTests()
    {
        try
        {
            _backend = new OpenClBackend();
            _ready = _backend.IsAvailable;
        }
        catch (Exception ex)
        {
            _initException = ex;
            _ready = false;
        }
    }

    public void Dispose() => _backend?.Dispose();

    private bool EnsureReady()
    {
        if (_ready) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests required but the OpenCL backend was unavailable.", _initException);
        return false;
    }

    // ---- RoPE ----------------------------------------------------------------

    // Builds interleaved cos/sin caches [maxSeq, headDim/2] using LLaMA-style theta.
    private static (float[] cos, float[] sin) BuildRopeCache(int maxSeq, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[maxSeq * half];
        var sin = new float[maxSeq * half];
        for (int pos = 0; pos < maxSeq; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(theta, (2.0 * i) / headDim);
                double angle = pos * freq;
                cos[pos * half + i] = (float)Math.Cos(angle);
                sin[pos * half + i] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static float[] RopeReference(float[] input, float[] cos, float[] sin,
        int rows, int headDim, int seqLen, int startPosition)
    {
        int half = headDim / 2;
        var outp = new float[input.Length];
        for (int row = 0; row < rows; row++)
        {
            int s = row % seqLen;
            int pos = startPosition + s;
            int baseIdx = row * headDim;
            for (int i = 0; i < half; i++)
            {
                float c = cos[pos * half + i];
                float sn = sin[pos * half + i];
                float xe = input[baseIdx + 2 * i];
                float xo = input[baseIdx + 2 * i + 1];
                outp[baseIdx + 2 * i] = xe * c - xo * sn;
                outp[baseIdx + 2 * i + 1] = xe * sn + xo * c;
            }
        }
        return outp;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]   // decode with a non-zero absolute position offset
    public void RopeInterleaved_MatchesCpuReference(int startPosition)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        const int heads = 3, seqLen = 6, headDim = 8, maxSeq = 32;
        int rows = heads * seqLen; // single batch: leading (heads) * seqLen
        var rng = new Random(7);
        var input = new float[rows * headDim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var (cos, sin) = BuildRopeCache(maxSeq, headDim, 10000f);

        var expected = RopeReference(input, cos, sin, rows, headDim, seqLen, startPosition);

        float[] actual;
        IGpuBuffer? inBuf = null, cosBuf = null, sinBuf = null, outBuf = null;
        try
        {
            inBuf = backend.AllocateBuffer(input);
            cosBuf = backend.AllocateBuffer(cos);
            sinBuf = backend.AllocateBuffer(sin);
            outBuf = backend.AllocateBuffer(new float[input.Length]);
            backend.RopeInterleaved(inBuf, cosBuf, sinBuf, outBuf, rows, headDim, seqLen, startPosition);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            inBuf?.Dispose();
            cosBuf?.Dispose();
            sinBuf?.Dispose();
            outBuf?.Dispose();
        }

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f,
                $"RoPE mismatch at {i}: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>
    /// The device-agnostic <c>IEngine.ApplyRoPEInterleaved</c> CPU path must use the SAME interleaving
    /// convention as the GPU kernel (validated against the same reference above), so a decoder's RoPE stays
    /// numerically identical whether it runs on CPU or GPU. Runs anywhere (no GPU required).
    /// </summary>
    [Fact]
    public void ApplyRoPEInterleaved_CpuEngine_MatchesReference()
    {
        const int heads = 3, seqLen = 6, headDim = 8, maxSeq = 32, startPosition = 5;
        int rows = heads * seqLen;
        var rng = new Random(11);
        var input = new float[rows * headDim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var (cos, sin) = BuildRopeCache(maxSeq, headDim, 10000f);

        var expected = RopeReference(input, cos, sin, rows, headDim, seqLen, startPosition);

        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var inT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(input, new[] { heads, seqLen, headDim });
        var cosT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(cos, new[] { maxSeq, headDim / 2 });
        var sinT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(sin, new[] { maxSeq, headDim / 2 });

        var actual = ((AiDotNet.Tensors.Engines.IEngine)engine)
            .ApplyRoPEInterleaved(inT, cosT, sinT, startPosition).AsSpan().ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-5f,
                $"CPU RoPE mismatch at {i}: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>
    /// RoPE's backward (ApplyRoPEInterleavedBackward) recovers the input gradient by re-applying the
    /// interleaved rotation with the sine negated — i.e. R(-θ)·R(θ) = I. This verifies that exact operation:
    /// rotating forward then rotating by the negated angle returns the original tensor. If this holds, the
    /// recorded backward (which computes gradInput = ApplyRoPEInterleaved(gradOut, cos, -sin)) is correct.
    /// </summary>
    [Fact]
    public void ApplyRoPEInterleaved_InverseRotation_RecoversInput()
    {
        const int heads = 3, seqLen = 6, headDim = 8, maxSeq = 32, startPosition = 5;
        int rows = heads * seqLen;
        var rng = new Random(29);
        var input = new float[rows * headDim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var (cos, sin) = BuildRopeCache(maxSeq, headDim, 10000f);
        var negSin = new float[sin.Length];
        for (int i = 0; i < sin.Length; i++) negSin[i] = -sin[i];

        var engine = (AiDotNet.Tensors.Engines.IEngine)new AiDotNet.Tensors.Engines.CpuEngine();
        var inT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(input, new[] { heads, seqLen, headDim });
        var cosT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(cos, new[] { maxSeq, headDim / 2 });
        var sinT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(sin, new[] { maxSeq, headDim / 2 });
        var negSinT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(negSin, new[] { maxSeq, headDim / 2 });

        var rotated = engine.ApplyRoPEInterleaved(inT, cosT, sinT, startPosition);
        var recovered = engine.ApplyRoPEInterleaved(rotated, cosT, negSinT, startPosition).AsSpan().ToArray();

        Assert.Equal(input.Length, recovered.Length);
        for (int i = 0; i < input.Length; i++)
            Assert.True(Math.Abs(input[i] - recovered[i]) < 1e-5f,
                $"Inverse RoPE did not recover input at {i}: expected {input[i]}, got {recovered[i]}");
    }

    // ---- Grouped-Query Attention --------------------------------------------

    private static float[] GqaReference(float[] q, float[] k, float[] v,
        int batch, int qHeads, int kvHeads, int seqQ, int seqK, int headDim, float scale, bool causal)
    {
        int group = qHeads / kvHeads;
        var outp = new float[batch * qHeads * seqQ * headDim];
        var scores = new float[seqK];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < qHeads; h++)
            {
                int kvh = h / group;
                for (int qi = 0; qi < seqQ; qi++)
                {
                    int qOff = ((b * qHeads + h) * seqQ + qi) * headDim;
                    int kvBase = (b * kvHeads + kvh) * seqK * headDim;
                    float max = float.NegativeInfinity;
                    for (int ki = 0; ki < seqK; ki++)
                    {
                        if (causal && ki > qi) { scores[ki] = float.NegativeInfinity; continue; }
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += q[qOff + d] * k[kvBase + ki * headDim + d];
                        scores[ki] = dot * scale;
                        if (scores[ki] > max) max = scores[ki];
                    }
                    float sum = 0f;
                    for (int ki = 0; ki < seqK; ki++)
                    {
                        if (causal && ki > qi) { scores[ki] = 0f; continue; }
                        scores[ki] = (float)Math.Exp(scores[ki] - max);
                        sum += scores[ki];
                    }
                    float inv = sum > 0f ? 1f / sum : 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int ki = 0; ki < seqK; ki++)
                            acc += scores[ki] * inv * v[kvBase + ki * headDim + d];
                        outp[qOff + d] = acc;
                    }
                }
            }
        }
        return outp;
    }

    /// <summary>
    /// The device-agnostic <c>IEngine.ScaledDotProductAttentionGqa</c> CPU path (broadcast KV heads + attend)
    /// must match the same reference the fused GPU GQA kernel passes, so a decoder's attention is numerically
    /// identical whether it runs on CPU or the recordable GPU kernel — with UNEXPANDED K/V (no ExpandKVHeads).
    /// Runs anywhere (no GPU required).
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScaledDotProductAttentionGqa_CpuEngine_MatchesReference(bool causal)
    {
        const int batch = 1, qHeads = 6, kvHeads = 2, seqQ = 5, seqK = 5, headDim = 8;
        float scale = 1f / (float)Math.Sqrt(headDim);
        var rng = new Random(23);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var q = Rand(batch * qHeads * seqQ * headDim);
        var k = Rand(batch * kvHeads * seqK * headDim);
        var v = Rand(batch * kvHeads * seqK * headDim);

        var expected = GqaReference(q, k, v, batch, qHeads, kvHeads, seqQ, seqK, headDim, scale, causal);

        var engine = new AiDotNet.Tensors.Engines.CpuEngine();
        var qT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(q, new[] { batch, qHeads, seqQ, headDim });
        var kT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(k, new[] { batch, kvHeads, seqK, headDim });
        var vT = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(v, new[] { batch, kvHeads, seqK, headDim });

        var actual = ((AiDotNet.Tensors.Engines.IEngine)engine)
            .ScaledDotProductAttentionGqa(qT, kT, vT, scale, causal).AsSpan().ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f,
                $"CPU GQA-SDPA mismatch at {i}: expected {expected[i]}, got {actual[i]}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GqaScaledDotProductAttention_MatchesCpuReference(bool causal)
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        // SmolLM2-style ratio: 9 query heads share 3 KV heads (group of 3).
        const int batch = 1, qHeads = 6, kvHeads = 2, seqQ = 5, seqK = 5, headDim = 8;
        float scale = 1f / (float)Math.Sqrt(headDim);
        var rng = new Random(11);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var q = Rand(batch * qHeads * seqQ * headDim);
        var k = Rand(batch * kvHeads * seqK * headDim);
        var v = Rand(batch * kvHeads * seqK * headDim);

        var expected = GqaReference(q, k, v, batch, qHeads, kvHeads, seqQ, seqK, headDim, scale, causal);

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(k);
            vBuf = backend.AllocateBuffer(v);
            outBuf = backend.AllocateBuffer(new float[q.Length]);
            backend.ScaledDotProductAttention(qBuf, kBuf, vBuf, outBuf, null, null,
                batch, qHeads, seqQ, seqK, headDim, scale, causal, softcap: 0f, numKVHeads: kvHeads);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose();
            kBuf?.Dispose();
            vBuf?.Dispose();
            outBuf?.Dispose();
        }

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f,
                $"GQA attention mismatch at {i}: expected {expected[i]}, got {actual[i]}");
    }

    // Guards the MHA collapse: numKVHeads == qHeads must equal a plain full-head reference.
    [Fact]
    public void GqaWithEqualHeads_EqualsStandardAttention()
    {
        if (!EnsureReady()) return;
        var backend = _backend!;

        const int batch = 1, heads = 4, seqQ = 4, seqK = 4, headDim = 8;
        float scale = 1f / (float)Math.Sqrt(headDim);
        var rng = new Random(3);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        var q = Rand(batch * heads * seqQ * headDim);
        var k = Rand(batch * heads * seqK * headDim);
        var v = Rand(batch * heads * seqK * headDim);

        var expected = GqaReference(q, k, v, batch, heads, heads, seqQ, seqK, headDim, scale, false);

        float[] actual;
        IGpuBuffer? qBuf = null, kBuf = null, vBuf = null, outBuf = null;
        try
        {
            qBuf = backend.AllocateBuffer(q);
            kBuf = backend.AllocateBuffer(k);
            vBuf = backend.AllocateBuffer(v);
            outBuf = backend.AllocateBuffer(new float[q.Length]);
            // numKVHeads == 0 selects the MHA path inside the kernel.
            backend.ScaledDotProductAttention(qBuf, kBuf, vBuf, outBuf, null, null,
                batch, heads, seqQ, seqK, headDim, scale, false);
            actual = backend.DownloadBuffer(outBuf);
        }
        finally
        {
            qBuf?.Dispose();
            kBuf?.Dispose();
            vBuf?.Dispose();
            outBuf?.Dispose();
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f,
                $"MHA mismatch at {i}: expected {expected[i]}, got {actual[i]}");
    }
}

#endif
