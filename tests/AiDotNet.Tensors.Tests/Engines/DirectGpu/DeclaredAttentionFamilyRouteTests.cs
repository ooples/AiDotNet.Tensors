using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The rest of the attention family through declared programs (#1440) against the hand-written kernels on the device:
/// grouped-query attention forward (with and without stored weights, causal at equal and unequal lengths) and backward,
/// and FlashDecode. The attention routes build a shape's program on its second sighting, so each declared call runs
/// twice and the second must launch the declared program.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredAttentionFamilyRouteTests
{
    private static CudaBackend OpenOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = CudaBackend.CreateOrThrow();
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is off for this process or this architecture has no released codegen kernels.");
        }
        return backend;
    }

    private static float[] Values(Random random, int count, float magnitude = 1f)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((random.NextDouble() * 2 - 1) * magnitude);
        return v;
    }

    private static void AssertClose(float[] expected, float[] actual, string what, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
    }

    // Runs call hand-written, then declared twice (the second sighting builds the program), asserting it launched.
    private static void Declared(CudaBackend backend, Action call)
    {
        backend.DeclaredKernelsSwitch = true;
        call();
        long before = backend.DeclaredProgramLaunches;
        call();
        Assert.True(backend.DeclaredProgramLaunches > before, "The declared route declined on a device that supports it.");
    }

    [SkippableTheory]
    [InlineData(2, 8, 2, 16, 16, 32, false, true)]
    [InlineData(1, 6, 3, 24, 24, 48, true, true)]
    [InlineData(2, 4, 1, 1, 40, 64, true, false)]     // decode-shaped: one query against a 40-key cache, causal
    [InlineData(1, 8, 4, 12, 20, 16, true, true)]     // causal across unequal lengths (a KV-cache offset)
    public void GroupedQueryAttention_ForwardAndBackward_MatchTheHandWrittenKernels(int batch, int heads, int kvHeads,
        int seqQ, int seqK, int headDim, bool causal, bool withBackward)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1440 + heads * seqK);
        float scale = 1f / MathF.Sqrt(headDim);
        int qSize = batch * heads * seqQ * headDim, kvSize = batch * kvHeads * seqK * headDim, pSize = batch * heads * seqQ * seqK;
        using var q = backend.AllocateBuffer(Values(random, qSize));
        using var k = backend.AllocateBuffer(Values(random, kvSize));
        using var v = backend.AllocateBuffer(Values(random, kvSize));

        float[] Run(bool declared, bool weights, out float[] w)
        {
            using var output = backend.AllocateBuffer(qSize);
            using var wBuf = backend.AllocateBuffer(pSize);
            void Call() => backend.GroupedQueryAttention(q, k, v, output, weights ? wBuf : null, batch, heads, kvHeads,
                seqQ, seqK, headDim, scale, causal);
            if (declared) Declared(backend, Call);
            else { backend.DeclaredKernelsSwitch = false; Call(); }
            w = backend.DownloadBuffer(wBuf);
            return backend.DownloadBuffer(output);
        }

        var outHand = Run(false, true, out var wHand);
        var outDecl = Run(true, true, out var wDecl);
        AssertClose(outHand, outDecl, "gqa output", 2e-4f);
        AssertClose(wHand, wDecl, "gqa weights", 2e-4f);
        var noWeightsDecl = Run(true, false, out _);
        AssertClose(outHand, noWeightsDecl, "gqa output without weights", 2e-4f);

        if (!withBackward) return;
        using var dout = backend.AllocateBuffer(Values(random, qSize));
        using var weightsBuf = backend.AllocateBuffer(wHand);
        float[][] Backward(bool declared)
        {
            using var dq = backend.AllocateBuffer(qSize);
            using var dk = backend.AllocateBuffer(kvSize);
            using var dv = backend.AllocateBuffer(kvSize);
            void Call() => backend.GroupedQueryAttentionBackward(dout, q, k, v, weightsBuf, dq, dk, dv, batch, heads, kvHeads,
                seqQ, seqK, headDim, scale, heads / kvHeads);
            if (declared) Declared(backend, Call);
            else { backend.DeclaredKernelsSwitch = false; Call(); }
            return new[] { backend.DownloadBuffer(dq), backend.DownloadBuffer(dk), backend.DownloadBuffer(dv) };
        }
        var hand = Backward(false);
        var decl = Backward(true);
        AssertClose(hand[0], decl[0], "gqa dq", 5e-4f);
        AssertClose(hand[1], decl[1], "gqa dk", 5e-4f);
        AssertClose(hand[2], decl[2], "gqa dv", 5e-4f);
    }

    [SkippableTheory]
    [InlineData(8, 2, 32, 100)]
    [InlineData(12, 12, 80, 37)]
    [InlineData(16, 4, 128, 513)]
    public void FlashDecode_MatchesTheHandWrittenKernel(int heads, int kvHeads, int headDim, int seqLen)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1441 + seqLen);
        float scale = 1f / MathF.Sqrt(headDim);
        using var q = backend.AllocateBuffer(Values(random, heads * headDim));
        using var k = backend.AllocateBuffer(Values(random, seqLen * kvHeads * headDim));
        using var v = backend.AllocateBuffer(Values(random, seqLen * kvHeads * headDim));
        backend.DeclaredKernelsSwitch = false;
        using var hand = backend.FlashDecode(q, k, v, heads, kvHeads, headDim, seqLen, scale);
        var expected = backend.DownloadBuffer(hand);
        backend.DeclaredKernelsSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        using var declared = backend.FlashDecode(q, k, v, heads, kvHeads, headDim, seqLen, scale);
        Assert.True(backend.DeclaredProgramLaunches > before, "The declared flash decode declined on a device that supports it.");
        AssertClose(expected, backend.DownloadBuffer(declared), "flash decode", 2e-4f);
    }
}