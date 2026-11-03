// Copyright (c) CrowdTrainer LLC. All rights reserved.
// cuDNN's forward default: with no algorithm given, Conv2DForwardGpu runs cudnnGetConvolutionForwardAlgorithm_v7's
// pick for the shape instead of a fixed implicit-precomp GEMM. That default is the convolution slots' reference and the
// route served once the tuning budget is spent.

using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class CudnnForwardHeuristicTests
{
    [SkippableFact]
    public void DistinctShapesDoNotGrowTheHeuristicCacheWithoutLimit()
    {
        using var backend = RequireCudnnBackend();
        using var conv = new CuDnnConvolution();
        bool previous = CuDnnConvolution.HeuristicForwardEnabled;
        try
        {
            CuDnnConvolution.HeuristicForwardEnabled = true;
            // These are descriptor-only queries; no device tensor or workspace is allocated.
            for (int batch = 1; batch <= 4097; batch++)
                conv.DefaultForwardAlgorithm(batch, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                    CuDnnNative.CudnnDataType.Float, CuDnnNative.CudnnMathType.Fma);

            var cache = (IDictionary)typeof(CuDnnConvolution)
                .GetField("_heuristicForward", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(conv)!;
            Assert.InRange(cache.Count, 1, 4096);
        }
        finally
        {
            CuDnnConvolution.HeuristicForwardEnabled = previous;
        }
    }

    /// <summary>cudnnConvolutionFwdAlgoPerf_t is 48 bytes with the size_t workspace at offset 16 on 64-bit hosts.</summary>
    [SkippableFact]
    public void AlgoPerfMatchesTheNativeLayout()
    {
        Skip.IfNot(Environment.Is64BitProcess);
        Assert.Equal(48, Marshal.SizeOf<CuDnnNative.CudnnConvolutionFwdAlgoPerf>());
        Assert.Equal(16, (int)Marshal.OffsetOf<CuDnnNative.CudnnConvolutionFwdAlgoPerf>(nameof(CuDnnNative.CudnnConvolutionFwdAlgoPerf.Memory)));
        Assert.Equal(28, (int)Marshal.OffsetOf<CuDnnNative.CudnnConvolutionFwdAlgoPerf>(nameof(CuDnnNative.CudnnConvolutionFwdAlgoPerf.MathType)));
    }

    private static CudaBackend RequireCudnnBackend()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA and NVRTC were detected, but the CUDA backend failed to initialize.");
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

    /// <summary>
    /// The default forward agrees with an explicit implicit-precomp GEMM (the previous fixed default) on shapes where the
    /// heuristic picks something else, and the kill switch returns the fixed default.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 64, 64, 16, 3, 1, 1)]
    [InlineData(1, 64, 64, 56, 3, 1, 1)]
    [InlineData(2, 128, 128, 28, 3, 1, 1)]
    [InlineData(2, 16, 32, 17, 3, 2, 1)]
    [InlineData(2, 64, 256, 14, 1, 1, 0)]
    public void TheHeuristicDefaultAgreesWithTheFixedDefault(int n, int c, int k, int h, int r, int stride, int pad)
    {
        using var backend = RequireCudnnBackend();
        bool previous = CuDnnConvolution.HeuristicForwardEnabled;
        using var conv = new CuDnnConvolution();
        try
        {
            int oh = (h + 2 * pad - r) / stride + 1;
            const CuDnnNative.CudnnMathType fma = CuDnnNative.CudnnMathType.Fma;
            CuDnnConvolution.HeuristicForwardEnabled = true;
            var chosen = conv.DefaultForwardAlgorithm(n, c, h, h, k, r, r, pad, pad, stride, stride, 1, 1,
                CuDnnNative.CudnnDataType.Float, fma);
            Assert.True(chosen >= 0 && chosen < CuDnnNative.CudnnConvolutionFwdAlgo.Count, $"algo {chosen}");
            // Cached per shape: the same answer twice.
            Assert.Equal(chosen, conv.DefaultForwardAlgorithm(n, c, h, h, k, r, r, pad, pad, stride, stride, 1, 1,
                CuDnnNative.CudnnDataType.Float, fma));

            float[] x = Fill(n * c * h * h, 1), w = Fill(k * c * r * r, 2);
            using var bx = backend.AllocateBuffer(x);
            using var bw = backend.AllocateBuffer(w);
            using var byDefault = backend.AllocateBuffer(n * k * oh * oh);
            using var byFixed = backend.AllocateBuffer(n * k * oh * oh);
            // Through the backend's own cuDNN handle (bound to its context and stream) via the slot's candidates.
            var slot = backend.Conv2DForwardVendorSlot;
            slot.Candidate("cudnn.conv2d.default")!.Execute(new CudaConv2DArgs(bx, bw, byDefault, n, c, h, h, k, oh, oh, r,
                r, stride, stride, pad, pad, 1, 1));
            slot.Candidate("cudnn.conv2d.algo1")!.Execute(new CudaConv2DArgs(bx, bw, byFixed, n, c, h, h, k, oh, oh, r, r,
                stride, stride, pad, pad, 1, 1));
            backend.Synchronize();
            float[] got = backend.DownloadBuffer(byDefault), expected = backend.DownloadBuffer(byFixed);
            double largest = expected.Max(v => Math.Abs((double)v)), worst = 0;
            for (int i = 0; i < got.Length; i++) worst = Math.Max(worst, Math.Abs((double)got[i] - expected[i]));
            Assert.True(largest > 0, "the fixed default wrote nothing");
            Assert.True(worst <= 1e-4 * Math.Max(1, largest), $"algo {chosen}: error {worst:E3} of {largest:E3}");

            CuDnnConvolution.HeuristicForwardEnabled = false;
            Assert.Equal(CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitPrecompGemm, conv.DefaultForwardAlgorithm(n, c, h, h,
                k, r, r, pad, pad, stride, stride, 1, 1, CuDnnNative.CudnnDataType.Float, fma));
        }
        finally
        {
            CuDnnConvolution.HeuristicForwardEnabled = previous;
        }
    }
}
