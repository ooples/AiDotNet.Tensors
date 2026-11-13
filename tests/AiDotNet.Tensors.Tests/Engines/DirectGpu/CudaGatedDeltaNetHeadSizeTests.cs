using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The CUDA Gated DeltaNet scan runs a warp per state row with ceil(headDim / 32) entries per lane, compiled for 1, 2,
/// 4 and 8: every variant, partial last lanes included (33, 100, 200), must match the managed forward and backward in
/// both reduction modes, and deterministic mode must repeat bit for bit.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaGatedDeltaNetHeadSizeTests
{
    private const int Batch = 2, SeqLen = 9, NumHeads = 2;

    private static float[] Values(int count, int seed, bool gate)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var v = new float[count];
        for (int i = 0; i < count; i++)
            v[i] = gate ? (float)(0.2 + 0.75 * rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return v;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        double scale = Math.Max(expected.Max(Math.Abs), 1e-6);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-4 * scale, $"{what}[{i}]: cpu {expected[i]}, gpu {actual[i]} (scale {scale})");
    }

    [SkippableTheory]
    [InlineData(8)]
    [InlineData(33)]
    [InlineData(64)]
    [InlineData(100)]
    [InlineData(128)]
    [InlineData(200)]
    [InlineData(256)]
    public void ForwardAndBackward_MatchTheCpu_InBothModes(int headDim)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        int modelDim = NumHeads * headDim;
        int n = Batch * SeqLen * modelDim, ng = Batch * SeqLen * NumHeads;
        int[] shape = { Batch, SeqLen, modelDim }, gateShape = { Batch, SeqLen, NumHeads };
        var host = new[] { Values(n, 1, false), Values(n, 2, false), Values(n, 3, false), Values(ng, 4, true), Values(ng, 5, true) };
        var dOut = Values(n, 6, false);

        var cpu = new CpuEngine();
        var inputs = host.Select((h, i) => new Tensor<float>(h, i < 3 ? shape : gateShape)).ToArray();
        float[] expectedOut;
        var expected = new float[5][];
        using (var tape = new GradientTape<float>())
        {
            var output = cpu.GatedDeltaNetScanForward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], NumHeads);
            expectedOut = output.ToArray();
            var loss = cpu.ReduceSum(cpu.TensorMultiply(output, new Tensor<float>(dOut, shape)), null);
            var grads = tape.ComputeGradients(loss, inputs);
            for (int i = 0; i < 5; i++) expected[i] = grads[inputs[i]].ToArray();
        }

        var bufs = host.Select(backend.AllocateBuffer).ToArray();
        using var dOutB = backend.AllocateBuffer(dOut);
        using var outB = backend.AllocateBuffer(n);
        try
        {
            backend.GatedDeltaNetScanForward(bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], outB, Batch, SeqLen, modelDim, NumHeads, headDim);
            AssertClose(expectedOut, backend.DownloadBuffer(outB), "output");

            var saved = BlasProvider.GetThreadLocalDeterministicMode();
            try
            {
                float[][]? previous = null;
                foreach (bool det in new[] { false, true, true })
                {
                    BlasProvider.SetThreadLocalDeterministicMode(det);
                    var gradBufs = new[] { n, n, n, ng, ng }.Select(backend.AllocateBuffer).ToArray();
                    try
                    {
                        ((IGatedDeltaNetBackwardBackend)backend).GatedDeltaNetScanBackward(dOutB, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4],
                            gradBufs[0], gradBufs[1], gradBufs[2], gradBufs[3], gradBufs[4], Batch, SeqLen, modelDim, NumHeads, headDim);
                        string[] names = { "dQ", "dK", "dV", "dAlpha", "dBeta" };
                        var results = gradBufs.Select(backend.DownloadBuffer).ToArray();
                        for (int g = 0; g < 5; g++) AssertClose(expected[g], results[g], $"{names[g]} (det {det})");
                        if (det && previous is not null)
                            for (int g = 0; g < 5; g++)
                                for (int i = 0; i < results[g].Length; i++)
                                    Assert.True(BitConverter.SingleToInt32Bits(previous[g][i]) == BitConverter.SingleToInt32Bits(results[g][i]),
                                        $"deterministic {names[g]}[{i}]: {previous[g][i]:R} then {results[g][i]:R}");
                        if (det) previous = results;
                    }
                    finally
                    {
                        foreach (var b in gradBufs) b.Dispose();
                    }
                }
            }
            finally
            {
                BlasProvider.SetThreadLocalDeterministicMode(saved);
            }
        }
        finally
        {
            foreach (var b in bufs) b.Dispose();
        }
    }
}
