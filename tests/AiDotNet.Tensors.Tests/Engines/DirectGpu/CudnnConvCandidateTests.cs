// Copyright (c) AiDotNet. All rights reserved.
// cuDNN as candidates of the tuned convolution slots: with cuDNN on, each pass's vendor slot measures every cuDNN
// algorithm and backend route against the established path, and the entry point runs the winner.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class CudnnConvCandidateTests
{
    private static CudaBackend RequireCudnnBackend()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        if (!CuDnnConvolution.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "cuDNN is not available.");
        }
        return backend;
    }

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 97.0);
        return values;
    }

    // Direct fp64 convolution, NCHW / KCRS.
    private static double[] Forward(float[] x, float[] w, int n, int c, int k, int h, int wd, int r, int stride, int pad,
        int oh, int ow)
    {
        var y = new double[n * k * oh * ow];
        for (int b = 0; b < n; b++)
            for (int o = 0; o < k; o++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        double s = 0;
                        for (int ch = 0; ch < c; ch++)
                            for (int u = 0; u < r; u++)
                                for (int v = 0; v < r; v++)
                                {
                                    int hi = i * stride - pad + u, wi = j * stride - pad + v;
                                    if (hi < 0 || hi >= h || wi < 0 || wi >= wd) continue;
                                    s += (double)x[((b * c + ch) * h + hi) * wd + wi] * w[((o * c + ch) * r + u) * r + v];
                                }
                        y[((b * k + o) * oh + i) * ow + j] = s;
                    }
        return y;
    }

    /// <summary>
    /// With cuDNN on, Conv2D, Conv2DBackwardInput and Conv2DBackwardKernel each put the shape through their vendor slot
    /// (a decision is recorded), the forward agrees with fp64, and under determinism the slot never picks an
    /// atomic-accumulating cuDNN algorithm.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 16, 32, 20, 3, 1, 1)]
    [InlineData(2, 8, 16, 17, 3, 2, 1)]
    [InlineData(4, 32, 16, 14, 1, 1, 0)]
    public void TheEntryPointsRouteThroughTheVendorSlots(int n, int c, int k, int h, int r, int stride, int pad)
    {
        using var backend = RequireCudnnBackend();
        TunedKernelMode? previousMode = TunedKernelPolicy.ModeOverride;
        bool previousPersist = TunedKernelPolicy.PersistDecisions;
        double previousBudget = TunedKernelPolicy.BudgetMilliseconds;
        bool previousDeterminism = AiDotNetEngine.DeterministicMode;
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.BudgetMilliseconds = 120_000;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            int oh = (h + 2 * pad - r) / stride + 1, ow = oh;
            float[] x = Fill(n * c * h * h, 1), w = Fill(k * c * r * r, 2), dy = Fill(n * k * oh * ow, 3);
            using var bx = backend.AllocateBuffer(x);
            using var bw = backend.AllocateBuffer(w);
            using var bdy = backend.AllocateBuffer(dy);
            using var by = backend.AllocateBuffer(n * k * oh * ow);
            using var bdx = backend.AllocateBuffer(n * c * h * h);
            using var bdw = backend.AllocateBuffer(k * c * r * r);
            backend.Conv2D(bx, bw, by, n, c, h, h, k, oh, ow, r, r, stride, stride, pad, pad, 1, 1);
            backend.Conv2DBackwardInput(bdy, bw, bdx, n, c, h, h, k, oh, ow, r, r, stride, stride, pad, pad, 1, 1);
            backend.Conv2DBackwardKernel(bx, bdy, bdw, n, c, h, h, k, oh, ow, r, r, stride, stride, pad, pad, 1, 1);
            backend.Synchronize();

            double[] expected = Forward(x, w, n, c, k, h, h, r, stride, pad, oh, ow);
            float[] got = backend.DownloadBuffer(by);
            double largest = expected.Max(Math.Abs), worst = 0;
            for (int i = 0; i < got.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]));
            Assert.True(worst <= 1e-4 * Math.Max(1, largest), $"forward error {worst:E3} of {largest:E3}");

            var slots = new ITunedKernelSlotInfo[]
            {
                backend.Conv2DForwardVendorSlot, backend.Conv2DBackwardVendorSlot(false), backend.Conv2DBackwardVendorSlot(true),
            };
            string[] atomic = { "cudnn.conv2d_dx.algo0", "cudnn.conv2d_dw.algo0", "cudnn.conv2d_dw.algo3" };
            foreach (var slot in slots)
            {
                var decision = Assert.Single(slot.CachedDecisions);
                Assert.DoesNotContain(decision.CandidateId, atomic);
            }
        }
        finally
        {
            TunedKernelPolicy.ModeOverride = previousMode;
            TunedKernelPolicy.PersistDecisions = previousPersist;
            TunedKernelPolicy.BudgetMilliseconds = previousBudget;
            AiDotNetEngine.SetDeterministicMode(previousDeterminism);
        }
    }
}
