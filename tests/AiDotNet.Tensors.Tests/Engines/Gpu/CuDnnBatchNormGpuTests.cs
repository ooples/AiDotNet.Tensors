#if !NET462
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: smoke tests for CuDnnBatchNorm's GPU-pointer helpers
/// with the new optional CudnnDataType parameter.
/// </summary>
public class CuDnnBatchNormGpuTests
{
    private static bool CuDnnAvailable => CuDnnContext.IsAvailable;

    [Fact]
    public void ForwardInferenceGpu_DefaultDtype_AcceptsFloat()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var bn = new CuDnnBatchNorm(ctx))
        {
            const int n = 1, c = 4, h = 4, w = 4;
            int elems = n * c * h * w;

            using var input = ctx.Allocate<float>(elems);
            using var output = ctx.Allocate<float>(elems);
            using var scale = ctx.Allocate<float>(c);
            using var bias = ctx.Allocate<float>(c);
            using var mean = ctx.Allocate<float>(c);
            using var variance = ctx.Allocate<float>(c);

            // Need running variance > 0 to avoid division-by-zero in cuDNN.
            var varHost = new float[c];
            for (int i = 0; i < c; i++) varHost[i] = 1.0f;
            ctx.CopyToDevice(variance, varHost);
            ctx.CopyToDevice(input, new float[elems]);

            var ex = Record.Exception(() =>
                bn.ForwardInferenceGpu(
                    input.DevicePtr, output.DevicePtr,
                    scale.DevicePtr, bias.DevicePtr,
                    mean.DevicePtr, variance.DevicePtr,
                    n, c, h, w));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void ForwardInferenceGpu_HalfDtype_AcceptsHalfInput()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var bn = new CuDnnBatchNorm(ctx))
        {
            const int n = 1, c = 4, h = 4, w = 4;
            int elems = n * c * h * w;

            // Input/output are fp16; scale/bias/mean/var remain fp32 per
            // cuDNN's mixed-precision convention.
            using var input = ctx.Allocate<ushort>(elems);
            using var output = ctx.Allocate<ushort>(elems);
            using var scale = ctx.Allocate<float>(c);
            using var bias = ctx.Allocate<float>(c);
            using var mean = ctx.Allocate<float>(c);
            using var variance = ctx.Allocate<float>(c);

            var varHost = new float[c];
            for (int i = 0; i < c; i++) varHost[i] = 1.0f;
            ctx.CopyToDevice(variance, varHost);
            ctx.CopyToDevice(input, new ushort[elems]);

            var ex = Record.Exception(() =>
                bn.ForwardInferenceGpu(
                    input.DevicePtr, output.DevicePtr,
                    scale.DevicePtr, bias.DevicePtr,
                    mean.DevicePtr, variance.DevicePtr,
                    n, c, h, w,
                    epsilon: 1e-5,
                    dataType: CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }
}
#endif
