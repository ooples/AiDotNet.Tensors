using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The eager-tape forms of FusedConv2D (one tape entry for conv + channel bias + ReLU) and of an unpadded float
/// MaxPool2D (no saved indices; the backward re-scans the input). Each must give the gradients the decomposed /
/// indexed form gives.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ConvPoolTapeTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tolerance, string what)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected.GetFlat(i), a = actual.GetFlat(i);
            Assert.True(Math.Abs(e - a) <= tolerance * Math.Max(1f, Math.Abs(e)),
                $"{what}[{i}]: expected {e}, got {a}");
        }
    }

    [Theory]
    [InlineData(FusedActivationType.ReLU)]
    [InlineData(FusedActivationType.None)]
    public void FusedConv2D_Float_RecordsOneEntry_AndMatchesTheDecomposedGradients(FusedActivationType activation)
    {
        var input = Random(1, 2, 3, 9, 9);
        var kernel = Random(2, 4, 3, 3, 3);
        var bias = Random(3, 4);
        var target = Random(4, 2, 4, 5, 5);

        Tensor<float> fusedOut;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> fused;
        using (var tape = new GradientTape<float>())
        {
            fusedOut = _engine.FusedConv2D(input, kernel, bias, 2, 2, 1, 1, 1, 1, activation);
            Assert.Equal(1, tape.EntryCount);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(fusedOut, target), new[] { 0, 1, 2, 3 }, keepDims: false);
            fused = tape.ComputeGradients(loss, new[] { input, kernel, bias });
        }

        Tensor<float> referenceOut;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> reference;
        using (var tape = new GradientTape<float>())
        {
            var z = _engine.TensorChannelBiasAdd(
                _engine.Conv2D(input, kernel, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 }), bias);
            referenceOut = activation == FusedActivationType.ReLU ? _engine.ReLU(z) : z;
            var loss = _engine.ReduceSum(_engine.TensorMultiply(referenceOut, target), new[] { 0, 1, 2, 3 }, keepDims: false);
            reference = tape.ComputeGradients(loss, new[] { input, kernel, bias });
        }

        AssertClose(referenceOut, fusedOut, 0f, "output");
        AssertClose(reference[input], fused[input], 1e-5f, "input gradient");
        AssertClose(reference[kernel], fused[kernel], 1e-5f, "kernel gradient");
        AssertClose(reference[bias], fused[bias], 1e-5f, "bias gradient");
    }

    [Fact]
    public void FusedConv2D_Float_ReLU_UnderTape_KeepsNaNFromBias()
    {
        // The eager FusedConv2D contract (FusedConv2DDoublePerfTests.FusedConv2D_Float_ReLU_PreservesNaNFromBias):
        // ReLU keeps a NaN, as torch.relu does. The tape form must not switch to an epilogue that stores +0.
        var input = Random(5, 1, 2, 8, 8);
        var kernel = Random(6, 3, 2, 1, 1);
        var bias = new Tensor<float>(new[] { 0.1f, float.NaN, -0.1f }, new[] { 3 });

        using var tape = new GradientTape<float>();
        var output = _engine.FusedConv2D(input, kernel, bias, 1, 1, 0, 0, 1, 1, FusedActivationType.ReLU);

        for (int i = 0; i < 64; i++)
            Assert.True(float.IsNaN(output[0, 1, i / 8, i % 8]), $"channel 1 cell {i} lost its NaN");
    }

    [Theory]
    [InlineData(2, 2, 8)]
    [InlineData(2, 1, 7)]
    [InlineData(3, 2, 9)]
    public void MaxPool2D_Float_UnderTape_MatchesTheIndexedBackward(int pool, int stride, int size)
    {
        // Ties included: every value is rounded to a quarter, so most windows hold a repeated maximum and the
        // winner rule (first maximum in row-major window order) decides where the gradient lands.
        var input = Random(7, 2, 3, size, size);
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, MathF.Round(input.GetFlat(i) * 2f) / 4f);
        int outSize = (size - pool) / stride + 1;
        var upstream = Random(8, 2, 3, outSize, outSize);

        Tensor<float> output, gradient;
        using (var tape = new GradientTape<float>())
        {
            output = _engine.MaxPool2D(input, pool, stride);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(output, upstream), new[] { 0, 1, 2, 3 }, keepDims: false);
            gradient = tape.ComputeGradients(loss, new[] { input })[input];
        }

        var indexedOutput = _engine.MaxPool2DWithIndices(input, new[] { pool, pool }, new[] { stride, stride }, out var indices);
        var indexedGradient = _engine.MaxPool2DBackward(upstream, indices, input.Shape.ToArray(), new[] { pool, pool }, new[] { stride, stride });

        AssertClose(indexedOutput, output, 0f, "output");
        AssertClose(indexedGradient, gradient, 0f, "input gradient");
    }

    [Theory]
    [InlineData(4, 6, 5, 9, 3, 2, 1)]   // strided 3x3: the whole-batch lowered GEMM
    [InlineData(3, 4, 8, 4, 3, 1, 1)]   // small-plane stride 1
    [InlineData(2, 5, 7, 8, 1, 2, 0)]   // strided 1x1 projection
    public void Conv2D_KernelGradient_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad)
    {
        var input = Random(11, batch, inC, size, size);
        var kernel = Random(12, outC, inC, k, k);
        int outSize = (size + 2 * pad - k) / stride + 1;
        var upstream = Random(13, batch, outC, outSize, outSize);

        Tensor<float> gradient;
        using (var tape = new GradientTape<float>())
        {
            var y = _engine.Conv2D(input, kernel, new[] { stride, stride }, new[] { pad, pad }, new[] { 1, 1 });
            var loss = _engine.ReduceSum(_engine.TensorMultiply(y, upstream), new[] { 0, 1, 2, 3 }, keepDims: false);
            gradient = tape.ComputeGradients(loss, new[] { kernel })[kernel];
        }

        // dK[o, c, i, j] = sum over b, oh, ow of dY[b, o, oh, ow] * x[b, c, oh*s + i - p, ow*s + j - p], in double.
        var expected = new Tensor<float>(new[] { outC, inC, k, k });
        for (int o = 0; o < outC; o++)
        for (int c = 0; c < inC; c++)
        for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
        {
            double acc = 0;
            for (int b = 0; b < batch; b++)
            for (int oh = 0; oh < outSize; oh++)
            for (int ow = 0; ow < outSize; ow++)
            {
                int ih = oh * stride + i - pad, iw = ow * stride + j - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)upstream[b, o, oh, ow] * input[b, c, ih, iw];
            }
            expected[o, c, i, j] = (float)acc;
        }

        AssertClose(expected, gradient, 1e-4f, "kernel gradient");
    }

    [Theory]
    [InlineData(4, 6, 5, 9, 3, 2, 1)]    // strided small plane: the whole-batch lowered GEMM
    [InlineData(3, 16, 12, 8, 3, 1, 1)]  // stride-1 8x8 plane: the whole-batch lowered GEMM
    [InlineData(2, 5, 7, 8, 1, 2, 0)]    // strided 1x1 projection
    [InlineData(1, 6, 5, 9, 3, 2, 1)]    // a single image keeps the per-image route
    public void Conv2D_Forward_MatchesANaiveReference(int batch, int inC, int outC, int size, int k, int stride, int pad)
    {
        var input = Random(21, batch, inC, size, size);
        var kernel = Random(22, outC, inC, k, k);
        int outSize = (size + 2 * pad - k) / stride + 1;

        var actual = _engine.Conv2D(input, kernel, new[] { stride, stride }, new[] { pad, pad }, new[] { 1, 1 });

        var expected = new Tensor<float>(new[] { batch, outC, outSize, outSize });
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < outC; o++)
        for (int oh = 0; oh < outSize; oh++)
        for (int ow = 0; ow < outSize; ow++)
        {
            double acc = 0;
            for (int c = 0; c < inC; c++)
            for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                int ih = oh * stride + i - pad, iw = ow * stride + j - pad;
                if (ih < 0 || ih >= size || iw < 0 || iw >= size) continue;
                acc += (double)kernel[o, c, i, j] * input[b, c, ih, iw];
            }
            expected[b, o, oh, ow] = (float)acc;
        }

        AssertClose(expected, actual, 1e-4f, "output");
    }

    [Fact]
    public void FusedConv2D_Float_UnderTape_OverwritesAReusedArenaBuffer()
    {
        // The fused tape entry rents its output uninitialized. Inside an arena, the previous step's buffer comes back
        // with its old contents: fill one with NaN, release it with the tape, and the conv must overwrite every cell.
        var input = Random(31, 2, 3, 9, 9);
        var kernel = Random(32, 4, 3, 3, 3);
        var bias = Random(33, 4);
        var expected = _engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);

        using var arena = TensorArena.Create();
        using (new GradientTape<float>())
        {
            var stale = TensorAllocator.RentUninitialized<float>(new[] { 2, 4, 9, 9 });
            for (int i = 0; i < stale.Length; i++) stale.SetFlat(i, float.NaN);
        }
        using (new GradientTape<float>())
        {
            var actual = _engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
            AssertClose(expected, actual, 0f, "output");
        }
    }
}
