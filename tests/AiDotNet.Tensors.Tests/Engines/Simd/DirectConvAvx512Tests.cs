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
/// chain is the same, so forward, input-gradient and weight-gradient results must be bit-identical, over plain and strided grids, run
/// tails that are not a multiple of six, several input-channel passes, and batches whose runs span images.
/// On hosts without AVX-512 the kernel runs on the software Vector512 fallback (the same arithmetic), so it is tested
/// everywhere; the hosted CI runners are often AVX-512 Xeons or Zen 4/5, where it runs natively.
/// </summary>
[Collection("ConvPerfSerial")]
public class DirectConvAvx512Tests
{
    public static TheoryData<int, int, int, int, int, int, int, int> Shapes => new()
    {
        // batch, inChannels, outChannels, size, stride, padding, kernel, dilation
        { 2, 64, 64, 9, 1, 1, 3, 1 },
        { 3, 32, 64, 7, 2, 1, 3, 1 },
        { 2, 128, 128, 4, 1, 1, 3, 1 },
        { 2, 64, 128, 8, 2, 1, 3, 1 },
        { 4, 32, 32, 5, 1, 1, 3, 1 },
        { 8, 512, 64, 4, 1, 1, 3, 1 },
        { 16, 32, 64, 6, 1, 1, 3, 1 },   // weight gradient split into partials over the batch
        // The weight-gradient tile's other column cases: one column (no zero-fill branch), five columns (a full group
        // of three and a partial last group of two), and dilation 2 (the column step is dilationW * Block).
        { 2, 64, 64, 9, 1, 0, 1, 1 },
        { 2, 64, 64, 9, 1, 2, 5, 1 },
        { 3, 32, 64, 11, 2, 2, 5, 1 },
        { 2, 64, 64, 9, 1, 2, 3, 2 },
        // Rows of twelve or more positions take the row kernel (twelve positions from one pointer): rows of 30 (two
        // row runs and a six-position remainder), a dilated row, and 256 input channels (several seeded passes).
        { 2, 64, 64, 30, 1, 1, 3, 1 },
        { 1, 32, 32, 17, 1, 2, 3, 2 },
        { 1, 256, 32, 14, 1, 1, 3, 1 },
        // Groups of four consecutive positions that wrap rows: 8-wide planes (8 + 4), 4-wide (4 + 4 + 4), runs spanning
        // images, and a 6-wide plane where only some twelves split into aligned groups.
        { 8, 64, 128, 8, 1, 1, 3, 1 },
        { 24, 32, 128, 4, 1, 1, 3, 1 },
        { 2, 32, 64, 6, 1, 1, 3, 1 },
        // Stride 2: groups of four positions 2 Block apart, on 8-wide (16 -> 8) and 12-wide (24 -> 12) outputs. Enough
        // positions per forward task (the default 128-task split) for a task to hold twelve.
        { 6, 32, 128, 16, 2, 1, 3, 1 },
        { 4, 64, 128, 24, 2, 1, 3, 1 },
        // #1445: the strided input gradient's phases packed tile-interleaved, over several 32-channel tiles (64 and 128
        // input channels), a 1x1 stride-2 conv whose odd phases receive no tap, and stride 3 (nine phases, uneven taps).
        // The weight gradient packs its input sixteen channels wide for the 512-bit tile; a 13-wide plane exercises the
        // pack's scalar tail (widths not a multiple of eight).
        { 2, 64, 64, 16, 2, 1, 3, 1 },
        { 2, 128, 64, 12, 2, 1, 3, 1 },
        { 2, 64, 32, 8, 2, 0, 1, 1 },
        { 2, 32, 64, 10, 3, 1, 3, 1 },
        { 3, 64, 32, 13, 1, 1, 3, 1 },
    };

    // The sixteen-wide weight-gradient packing on its own: 48 input channels (three wide blocks, not a multiple of the
    // 32-channel tiles the forward and input gradient need), so only the weight gradient is compared.
    [SkippableFact]
    public void WideWeightGradientPacking_MatchesTheAvx2Tile_At48InputChannels()
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "The AVX2 side of the comparison needs AVX2 and FMA.");
        const int batch = 2, inChannels = 48, outChannels = 32, size = 11, kernel = 3, padding = 1, stride = 1;
        int o = (size + 2 * padding - kernel) / stride + 1;
        var rng = new Random(1445);
        var x = Random(rng, batch, inChannels, size, size);
        var dy = Random(rng, batch, outChannels, o, o);
        float[] Dw()
        {
            var dw = new float[outChannels * inChannels * kernel * kernel];
            DirectConvAvx2.BackwardKernel(x.ToArray(), 0, dy.ToArray(), 0, dw, 0, accumulate: false, batch, inChannels, size,
                size, outChannels, kernel, kernel, stride, stride, padding, padding, 1, 1, o, o);
            return dw;
        }
        bool previous = DirectConvAvx2.s_useAvx512, previousForce = DirectConvAvx2.s_forceAvx512ForTest;
        try
        {
            DirectConvAvx2.s_useAvx512 = true;
            DirectConvAvx2.s_forceAvx512ForTest = !System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated;
            var wide = Dw();
            DirectConvAvx2.s_useAvx512 = false;
            var avx2 = Dw();
            Assert.Equal(avx2, wide);
        }
        finally
        {
            DirectConvAvx2.s_useAvx512 = previous;
            DirectConvAvx2.s_forceAvx512ForTest = previousForce;
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void TheAvx512KernelMatchesTheAvx2KernelBitForBit(int batch, int inChannels, int outChannels, int size, int stride,
        int padding, int kernel, int dilation)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
            "The AVX2 side of the comparison needs AVX2 and FMA.");
        var rng = new Random(batch * 131 + inChannels + kernel * 7 + dilation);
        int o = (size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
        var x = Random(rng, batch, inChannels, size, size);
        var w = Random(rng, outChannels, inChannels, kernel, kernel);
        var dy = Random(rng, batch, outChannels, o, o);
        bool previous = DirectConvAvx2.s_useAvx512, previousForce = DirectConvAvx2.s_forceAvx512ForTest;
        bool previousInterleave = DirectConvAvx2.s_interleave512;
        try
        {
            DirectConvAvx2.s_useAvx512 = true;
            DirectConvAvx2.s_forceAvx512ForTest = !System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated;
            DirectConvAvx2.s_interleave512 = true;
            var (y512, dx512, dw512) = Run(x, w, dy, batch, inChannels, outChannels, size, stride, padding, kernel, dilation, o);
            // The 512-bit tile over the AVX2 weight layout (each 16-lane vector joined from two 8-lane blocks).
            DirectConvAvx2.s_interleave512 = false;
            var (y512b, dx512b, _) = Run(x, w, dy, batch, inChannels, outChannels, size, stride, padding, kernel, dilation, o);
            DirectConvAvx2.s_useAvx512 = false;
            var (y256, dx256, dw256) = Run(x, w, dy, batch, inChannels, outChannels, size, stride, padding, kernel, dilation, o);
            Assert.Equal(y256, y512);
            Assert.Equal(dx256, dx512);
            Assert.Equal(dw256, dw512);
            Assert.Equal(y256, y512b);
            Assert.Equal(dx256, dx512b);
        }
        finally
        {
            DirectConvAvx2.s_useAvx512 = previous;
            DirectConvAvx2.s_forceAvx512ForTest = previousForce;
            DirectConvAvx2.s_interleave512 = previousInterleave;
        }
    }

    // Straight through the direct kernel (no route choice), so the comparison cannot land on another implementation.
    // BackwardInput has no dilation parameter (its callers route dilated convolutions elsewhere), so a dilated shape
    // compares an empty input gradient on both sides and is checked through the forward and the weight gradient.
    private static (float[] Y, float[] Dx, float[] Dw) Run(Tensor<float> x, Tensor<float> w, Tensor<float> dy,
        int batch, int inChannels, int outChannels, int size, int stride, int padding, int kernel, int dilation, int o)
    {
        var y = new float[batch * outChannels * o * o];
        DirectConvAvx2.Forward(x.ToArray(), 0, w.ToArray(), 0, y, 0, accumulate: false, batch, inChannels, size, size,
            outChannels, kernel, kernel, stride, stride, padding, padding, dilation, dilation, o, o);
        var dx = new float[dilation == 1 ? batch * inChannels * size * size : 0];
        if (dilation == 1)
            DirectConvAvx2.BackwardInput(dy.ToArray(), 0, w.ToArray(), 0, dx, 0, accumulate: false, batch, inChannels, size, size,
                outChannels, kernel, kernel, stride, stride, padding, padding, o, o);
        // The weight gradient accumulated onto a seeded buffer (the += path), with a target task count that splits the
        // batch into partial gradients.
        var dw = new float[outChannels * inChannels * kernel * kernel];
        for (int i = 0; i < dw.Length; i++) dw[i] = (i % 7) * 0.125f;
        DirectConvAvx2.BackwardKernel(x.ToArray(), 0, dy.ToArray(), 0, dw, 0, accumulate: true, batch, inChannels, size, size,
            outChannels, kernel, kernel, stride, stride, padding, padding, dilation, dilation, o, o, targetTasks: 4 * outChannels);
        return (y, dx, dw);
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
#endif
