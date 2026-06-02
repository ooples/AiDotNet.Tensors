// Copyright (c) AiDotNet. All rights reserved.
// #403 small-batch channel-parallel conv-backward contract. At batch=1 (the CNN
// model-family training shape [1,3,224,224]) Conv2DBackwardInputInto /
// Conv2DBackwardKernelInto can no longer parallelize over the batch axis, so the
// im2col build (Im2ColStridedSingleChannelRange) and the col2im scatter
// (Col2ImAccumulateChannelRange) parallelize over INPUT CHANNELS instead — each
// channel owns a disjoint c*H*W output slab / kH*kW im2col row block, so the
// scatter is race-free AND each output element's += reduction stays on a single
// thread in fixed order. This pins two properties the parallelization must keep:
//   1. Correctness: the Into result matches the allocating reference AND a scalar
//      reference, at a shape large enough that the channel loop actually runs
//      multi-threaded (work above the grain gate) and receptive fields overlap
//      (stride=1, k=3, pad=1 → every interior input pixel accumulates 9 columns).
//   2. Determinism: under deterministic mode (the training harness setting) the
//      same backward run at MaxDegreeOfParallelism 1 / 2 / 4 / 16 — which changes
//      the channel-partition boundaries — is BIT-IDENTICAL. deterministicSafe=true
//      on the channel loops keeps them parallel in deterministic mode, so this is
//      the executable guard that the disjoint-write/fixed-order-reduction contract
//      holds (same shape as DeterministicParallelGemmContractTests, for conv).

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// Toggles process-wide deterministic mode + MaxDegreeOfParallelism; serialize
// against the other determinism-sensitive tests.
[Collection("BlasManaged-Stats-Serial")]
public class Conv2DBackwardChannelParallelTests
{
    // batch=1 forces the channel-parallel path. inC=16, H=W=32, k=3, p=1, s=1 →
    // colW = 32*32 = 1024, channel work = inC*kH*kW*colW = 16*9*1024 = 147456,
    // well above the 32K serial grain gate, so the channel loop runs in parallel.
    private const int Batch = 1, InC = 16, H = 32, W = 32, OutC = 16, KH = 3, KW = 3;
    private const int PadH = 1, PadW = 1;
    private const int OH = H + 2 * PadH - KH + 1; // = H
    private const int OW = W + 2 * PadW - KW + 1; // = W

    private static readonly int[] ThreadCounts = { 1, 2, 4, 16 };

    [Fact]
    public void BackwardInput_FLOAT_ChannelParallel_MatchesAllocatingAndScalarReference()
    {
        var engine = new CpuEngine();
        var rng = new Random(2024);
        var gradOut = new Tensor<float>(new[] { Batch, OutC, OH, OW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var kernel = new Tensor<float>(new[] { OutC, InC, KH, KW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() - 0.5);

        int beforeMax = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);

            var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
                new[] { Batch, InC, H, W }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 });
            var dest = new Tensor<float>(new[] { Batch, InC, H, W });
            engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
                new[] { Batch, InC, H, W }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 },
                accumulate: false);

            // Scalar reference straight from the backward-input definition.
            var expected = new float[Batch * InC * H * W];
            for (int ic = 0; ic < InC; ic++)
                for (int ih = 0; ih < H; ih++)
                    for (int iw = 0; iw < W; iw++)
                    {
                        double sum = 0.0;
                        for (int oc = 0; oc < OutC; oc++)
                            for (int kh = 0; kh < KH; kh++)
                                for (int kw = 0; kw < KW; kw++)
                                {
                                    int oh = ih + PadH - kh;
                                    int ow = iw + PadW - kw;
                                    if (oh < 0 || oh >= OH || ow < 0 || ow >= OW) continue;
                                    sum += (double)kernel[oc, ic, kh, kw] * gradOut[0, oc, oh, ow];
                                }
                        expected[(ic * H + ih) * W + iw] = (float)sum;
                    }

            for (int i = 0; i < dest.Length; i++)
            {
                Assert.True(Math.Abs(allocating[i] - dest[i]) < 1e-4f,
                    $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
                Assert.True(Math.Abs(expected[i] - dest[i]) < 1e-3f,
                    $"[{i}] ref={expected[i]:F6} into={dest[i]:F6}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = beforeMax; }
    }

    [Fact]
    public void BackwardKernel_FLOAT_ChannelParallel_MatchesAllocatingAndScalarReference()
    {
        var engine = new CpuEngine();
        var rng = new Random(2025);
        var gradOut = new Tensor<float>(new[] { Batch, OutC, OH, OW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var input = new Tensor<float>(new[] { Batch, InC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        int beforeMax = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);

            var allocating = engine.Conv2DBackwardKernel(gradOut, input,
                new[] { OutC, InC, KH, KW }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 });
            var dest = new Tensor<float>(new[] { OutC, InC, KH, KW });
            engine.Conv2DBackwardKernelInto(dest, gradOut, input,
                new[] { OutC, InC, KH, KW }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 },
                accumulate: false);

            // Scalar reference straight from the backward-kernel definition.
            var expected = new float[OutC * InC * KH * KW];
            for (int oc = 0; oc < OutC; oc++)
                for (int ic = 0; ic < InC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                        {
                            double sum = 0.0;
                            for (int oh = 0; oh < OH; oh++)
                                for (int ow = 0; ow < OW; ow++)
                                {
                                    int ih = oh + kh - PadH;
                                    int iw = ow + kw - PadW;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    sum += (double)gradOut[0, oc, oh, ow] * input[0, ic, ih, iw];
                                }
                            expected[((oc * InC + ic) * KH + kh) * KW + kw] = (float)sum;
                        }

            for (int i = 0; i < dest.Length; i++)
            {
                Assert.True(Math.Abs(allocating[i] - dest[i]) < 1e-3f,
                    $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
                Assert.True(Math.Abs(expected[i] - dest[i]) < 1e-2f,
                    $"[{i}] ref={expected[i]:F6} into={dest[i]:F6}");
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = beforeMax; }
    }

    [Fact]
    public void BackwardInput_DOUBLE_BitIdentical_AcrossThreadCounts()
        => AssertBackwardInputBitIdenticalAcrossThreadCounts();

    [Fact]
    public void BackwardKernel_DOUBLE_BitIdentical_AcrossThreadCounts()
        => AssertBackwardKernelBitIdenticalAcrossThreadCounts();

    private static void AssertBackwardInputBitIdenticalAcrossThreadCounts()
    {
        var engine = new CpuEngine();
        var rng = new Random(31);
        var gradOut = new Tensor<double>(new[] { Batch, OutC, OH, OW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { OutC, InC, KH, KW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        bool? beforeThreadDet = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeReductions = CpuParallelSettings.DeterministicReductions;
        int beforeMax = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            CpuParallelSettings.DeterministicReductions = true;

            double[]? reference = null;
            int referenceThreads = 0;
            foreach (int threads in ThreadCounts)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var dest = new Tensor<double>(new[] { Batch, InC, H, W });
                engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
                    new[] { Batch, InC, H, W }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 },
                    accumulate: false);
                var result = new double[dest.Length];
                for (int i = 0; i < dest.Length; i++) result[i] = dest[i];

                if (reference is null) { reference = result; referenceThreads = threads; }
                else
                    for (int i = 0; i < reference.Length; i++)
                        Assert.True(reference[i] == result[i],
                            $"backward-input not bit-identical: threads={referenceThreads} vs {threads} at [{i}]: " +
                            $"{reference[i]:R} vs {result[i]:R}");
            }
        }
        finally
        {
            BlasProvider.SetDeterministicMode(beforeDet);
            if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(beforeThreadDet);
            CpuParallelSettings.DeterministicReductions = beforeReductions;
            CpuParallelSettings.MaxDegreeOfParallelism = beforeMax;
        }
    }

    private static void AssertBackwardKernelBitIdenticalAcrossThreadCounts()
    {
        var engine = new CpuEngine();
        var rng = new Random(32);
        var gradOut = new Tensor<double>(new[] { Batch, OutC, OH, OW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { Batch, InC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        bool? beforeThreadDet = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        bool beforeDet = BlasProvider.IsDeterministicMode;
        bool beforeReductions = CpuParallelSettings.DeterministicReductions;
        int beforeMax = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            BlasProvider.SetDeterministicMode(true);
            CpuParallelSettings.DeterministicReductions = true;

            double[]? reference = null;
            int referenceThreads = 0;
            foreach (int threads in ThreadCounts)
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                var dest = new Tensor<double>(new[] { OutC, InC, KH, KW });
                engine.Conv2DBackwardKernelInto(dest, gradOut, input,
                    new[] { OutC, InC, KH, KW }, new[] { 1, 1 }, new[] { PadH, PadW }, new[] { 1, 1 },
                    accumulate: false);
                var result = new double[dest.Length];
                for (int i = 0; i < dest.Length; i++) result[i] = dest[i];

                if (reference is null) { reference = result; referenceThreads = threads; }
                else
                    for (int i = 0; i < reference.Length; i++)
                        Assert.True(reference[i] == result[i],
                            $"backward-kernel not bit-identical: threads={referenceThreads} vs {threads} at [{i}]: " +
                            $"{reference[i]:R} vs {result[i]:R}");
            }
        }
        finally
        {
            BlasProvider.SetDeterministicMode(beforeDet);
            if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(beforeThreadDet);
            CpuParallelSettings.DeterministicReductions = beforeReductions;
            CpuParallelSettings.MaxDegreeOfParallelism = beforeMax;
        }
    }
}
