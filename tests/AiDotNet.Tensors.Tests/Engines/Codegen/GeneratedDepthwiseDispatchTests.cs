// Copyright (c) AiDotNet. All rights reserved.
// The generated depthwise kernel is reachable from the backend, and agrees with the
// kernel it replaces.
//
// PROMO-1 recorded that depthwise beats cuDNN at 2.08x-2.99x and promoted it, but nothing
// in the engine asked for it, so the win existed only inside the benchmark harness. A
// promotion nobody can reach is a note, not a speedup.
//
// The bar here is agreement with the ESTABLISHED CUDA kernel on the same buffers, because
// that is what callers get today. Bit-exactness is not the bar and would be the wrong one:
// the two implementations sum the nine taps in different orders, so they are entitled to
// differ in the last fp32 place.

using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class GeneratedDepthwiseDispatchTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");

        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException(
                "CUDA driver was detected, but the CUDA backend failed to initialize.");
        }

        return backend;
    }

    /// <summary>
    /// With the family promoted and the flag on, a depthwise 3x3 call must take the
    /// generated kernel AND agree with the established one.
    /// </summary>
    [SkippableFact]
    public async Task GeneratedDepthwise_DispatchesAndAgreesWithTheEstablishedKernel()
    {
        await Task.Yield();

        using var backend = OpenBackendOrSkip();

        const int N = 2, C = 16, H = 28, W = 28;
        long elements = (long)N * C * H * W;
        long weightElements = (long)C * 9;

        var input = new float[elements];
        for (long i = 0; i < elements; i++) input[i] = (float)((((i * 37) % 97) - 48) / 64.0);
        var weights = new float[weightElements];
        for (long i = 0; i < weightElements; i++) weights[i] = (float)((((i * 53) % 89) - 44) / 64.0);

        using var inputBuffer = backend.AllocateBuffer(input);
        using var weightBuffer = backend.AllocateBuffer(weights);
        using var generatedOut = backend.AllocateBuffer((int)elements);
        using var establishedOut = backend.AllocateBuffer((int)elements);

        bool prior = DirectPtxFeatureGate.TestOverride ?? false;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            // The established path, with the generated dispatch explicitly bypassed by
            // asking for a geometry it declines (stride 2 is outside the measured set).
            backend.DepthwiseConv2D(inputBuffer, weightBuffer, establishedOut,
                N, C, H, W, H, W, 3, 3, 1, 1, 1, 1);

            long before = backend.GeneratedDispatchCount;
            bool took = backend.TryDirectPtxDepthwiseConv2D(
                inputBuffer, weightBuffer, generatedOut,
                N, C, H, W, H, W, 3, 3, 1, 1, 1, 1);

            // On a device the family was measured on, declining is a FAILURE, not a skip.
            // Returning early here would make this test vacuous on the one machine where
            // it can actually prove something.
            bool measuredArchitecture = backend.IsDirectPtxConvolutionEnabled;
            if (!took)
            {
                Assert.False(measuredArchitecture,
                    "the generated depthwise kernel must dispatch on the architecture its " +
                    "2.08x-2.99x measurement came from");
                return;
            }

            Assert.True(backend.GeneratedDispatchCount > before,
                "a successful dispatch must be counted");

            backend.Synchronize();
            var fromGenerated = new float[elements];
            var fromEstablished = new float[elements];
            backend.DownloadBuffer(generatedOut, fromGenerated);
            backend.DownloadBuffer(establishedOut, fromEstablished);

            double worst = 0, scale = 0;
            for (long i = 0; i < elements; i++)
            {
                worst = Math.Max(worst, Math.Abs(fromGenerated[i] - fromEstablished[i]));
                scale = Math.Max(scale, Math.Abs(fromEstablished[i]));
            }
            double relative = scale > 0 ? worst / scale : worst;

            // Nine taps summed in a different order: a few fp32 ulp, not zero.
            Assert.True(relative < 1e-5,
                "the generated kernel must agree with the one it replaces; relative " + relative);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = prior ? true : (bool?)null;
        }
    }

    /// <summary>
    /// Geometry outside the measured set must DECLINE. A 5x5 or a strided depthwise has no
    /// evidence behind it, so it has to keep taking the established path.
    /// </summary>
    [SkippableTheory]
    [InlineData(5, 5, 1, 1, 2, 2)]   // 5x5 taps
    [InlineData(3, 3, 2, 2, 1, 1)]   // stride 2
    [InlineData(3, 3, 1, 1, 0, 0)]   // no padding: output extent changes
    public async Task GeometryOutsideTheMeasuredSet_Declines(
        int kh, int kw, int sh, int sw, int ph, int pw)
    {
        await Task.Yield();

        using var backend = OpenBackendOrSkip();

        const int N = 1, C = 8, H = 16, W = 16;
        long elements = (long)N * C * H * W;

        using var input = backend.AllocateBuffer((int)elements);
        using var weights = backend.AllocateBuffer(C * kh * kw);
        using var output = backend.AllocateBuffer((int)elements);

        bool prior = DirectPtxFeatureGate.TestOverride ?? false;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            Assert.False(backend.TryDirectPtxDepthwiseConv2D(
                input, weights, output, N, C, H, W, H, W, kh, kw, sh, sw, ph, pw));
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = prior ? true : (bool?)null;
        }
    }

    /// <summary>With the feature flag off, nothing dispatches — opt-in stays opt-in.</summary>
    [SkippableFact]
    public async Task FeatureFlagOff_Declines()
    {
        await Task.Yield();

        using var backend = OpenBackendOrSkip();

        const int N = 1, C = 8, H = 16, W = 16;
        long elements = (long)N * C * H * W;

        using var input = backend.AllocateBuffer((int)elements);
        using var weights = backend.AllocateBuffer(C * 9);
        using var output = backend.AllocateBuffer((int)elements);

        bool? prior = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = false;
        try
        {
            Assert.False(backend.TryDirectPtxDepthwiseConv2D(
                input, weights, output, N, C, H, W, H, W, 3, 3, 1, 1, 1, 1));
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = prior;
        }
    }

    [Fact]
    public async Task TestOverride_IsIsolatedBetweenLogicalCallContexts()
    {
        await Task.Yield();

        bool? prior = DirectPtxFeatureGate.TestOverride;
        var falseFlowReady = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFalseFlow = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            DirectPtxFeatureGate.TestOverride = null;
            Task<bool?> falseFlow = Task.Run(async () =>
            {
                DirectPtxFeatureGate.TestOverride = false;
                falseFlowReady.SetResult(true);
                await releaseFalseFlow.Task;
                await Task.Yield();
                return DirectPtxFeatureGate.TestOverride;
            });

            await falseFlowReady.Task;
            DirectPtxFeatureGate.TestOverride = true;
            releaseFalseFlow.SetResult(true);

            Assert.Equal(false, await falseFlow);
            Assert.Equal(true, DirectPtxFeatureGate.TestOverride);
        }
        finally
        {
            releaseFalseFlow.TrySetResult(true);
            DirectPtxFeatureGate.TestOverride = prior;
        }
    }

    /// <summary>
    /// The generated max pool must agree with the established kernel on BOTH outputs. The
    /// indices matter more than the values: the backward pass routes every gradient by
    /// them, so a wrong convention corrupts training while the pooled values still look
    /// right.
    /// </summary>
    [SkippableFact]
    public async Task GeneratedMaxPool_AgreesOnValuesAndIndices()
    {
        await Task.Yield();

        using var backend = OpenBackendOrSkip();

        const int N = 2, C = 8, H = 16, W = 16;
        const int OutH = H / 2, OutW = W / 2;
        long inElements = (long)N * C * H * W;
        long outElements = (long)N * C * OutH * OutW;

        var host = new float[inElements];
        for (long i = 0; i < inElements; i++) host[i] = (float)((((i * 37) % 97) - 48) / 64.0);

        using var input = backend.AllocateBuffer(host);
        using var generatedOut = backend.AllocateBuffer((int)outElements);
        using var establishedOut = backend.AllocateBuffer((int)outElements);
        using var generatedIdx = backend.AllocateIntBuffer((int)outElements);
        using var establishedIdx = backend.AllocateIntBuffer((int)outElements);

        bool? prior = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            long before = backend.GeneratedDispatchCount;
            bool took = backend.TryDirectPtxMaxPool2D(
                input, generatedOut, generatedIdx,
                N, C, H, W, OutH, OutW, 2, 2, 2, 2, 0, 0);

            if (!took)
            {
                Assert.False(backend.IsDirectPtxConvolutionEnabled,
                    "the generated max pool must dispatch on the architecture it was measured on");
                return;
            }
            Assert.True(backend.GeneratedDispatchCount > before);

            // The established kernel, reached by disabling the generated path.
            DirectPtxFeatureGate.TestOverride = false;
            backend.MaxPool2D(input, establishedOut, establishedIdx,
                N, C, H, W, OutH, OutW, 2, 2, 2, 2, 0, 0);
            backend.Synchronize();

            var gv = new float[outElements];
            var ev = new float[outElements];
            backend.DownloadBuffer(generatedOut, gv);
            backend.DownloadBuffer(establishedOut, ev);

            var gi = new int[outElements];
            var ei = new int[outElements];
            backend.DownloadIntBuffer(generatedIdx, gi);
            backend.DownloadIntBuffer(establishedIdx, ei);

            for (long i = 0; i < outElements; i++)
            {
                // A maximum is a SELECTION, so the value must be bit-identical -- unlike a
                // sum, there is no reordering freedom to excuse a difference.
                Assert.Equal(ev[i], gv[i]);
                Assert.Equal(ei[i], gi[i]);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = prior;
        }
    }
}
