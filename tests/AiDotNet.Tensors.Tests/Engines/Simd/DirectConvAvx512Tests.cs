// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET9_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// The direct convolution's AVX-512 kernel (<c>ForwardSextetRun512</c>) against its AVX2 kernel: every output's FMA
/// chain is the same, so forward and input-gradient results must be bit-identical, over plain and strided grids, run
/// tails that are not a multiple of six, several input-channel passes, and batches whose runs span images.
/// On hosts without AVX-512 the kernel runs on the software Vector512 fallback (the same arithmetic), so it is tested
/// everywhere; the hosted CI runners are often AVX-512 Xeons or Zen 4/5, where it runs natively.
/// </summary>
[Collection("ConvPerfSerial")]
public class DirectConvAvx512Tests
{
    public static TheoryData<int, int, int, int, int, int> Shapes => new()
    {
        // batch, inChannels, outChannels, size, stride, padding
        { 2, 64, 64, 9, 1, 1 },
        { 3, 32, 64, 7, 2, 1 },
        { 2, 128, 128, 4, 1, 1 },
        { 2, 64, 128, 8, 2, 1 },
        { 4, 32, 32, 5, 1, 1 },
        { 8, 512, 64, 4, 1, 1 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void TheAvx512KernelMatchesTheAvx2KernelBitForBit(int batch, int inChannels, int outChannels, int size, int stride, int padding)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "The AVX2 side of the comparison needs AVX2 and FMA.");
        var rng = new Random(batch * 131 + inChannels);
        int o = (size + 2 * padding - 3) / stride + 1;
        var x = Random(rng, batch, inChannels, size, size);
        var w = Random(rng, outChannels, inChannels, 3, 3);
        var dy = Random(rng, batch, outChannels, o, o);
        bool previous = DirectConvAvx2.s_useAvx512, previousForce = DirectConvAvx2.s_forceAvx512ForTest;
        try
        {
            DirectConvAvx2.s_useAvx512 = true;
            DirectConvAvx2.s_forceAvx512ForTest = !System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated;
            var (y512, dx512) = Run(x, w, dy, batch, inChannels, outChannels, size, stride, padding, o);
            DirectConvAvx2.s_useAvx512 = false;
            var (y256, dx256) = Run(x, w, dy, batch, inChannels, outChannels, size, stride, padding, o);
            Assert.Equal(y256, y512);
            Assert.Equal(dx256, dx512);
        }
        finally
        {
            DirectConvAvx2.s_useAvx512 = previous;
            DirectConvAvx2.s_forceAvx512ForTest = previousForce;
        }
    }

    // Straight through the direct kernel (no route choice), so the comparison cannot land on another implementation.
    private static (float[] Y, float[] Dx) Run(Tensor<float> x, Tensor<float> w, Tensor<float> dy,
        int batch, int inChannels, int outChannels, int size, int stride, int padding, int o)
    {
        var y = new float[batch * outChannels * o * o];
        DirectConvAvx2.Forward(x.ToArray(), 0, w.ToArray(), 0, y, 0, accumulate: false, batch, inChannels, size, size,
            outChannels, 3, 3, stride, stride, padding, padding, 1, 1, o, o);
        var dx = new float[batch * inChannels * size * size];
        DirectConvAvx2.BackwardInput(dy.ToArray(), 0, w.ToArray(), 0, dx, 0, accumulate: false, batch, inChannels, size, size,
            outChannels, 3, 3, stride, stride, padding, padding, o, o);
        return (y, dx);
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
#endif
