#if !NET462
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: smoke tests for CuDnnPooling's new GPU-pointer helpers.
/// Forward + Backward both accept an optional CudnnDataType parameter.
/// Tests skip on hosts without cuDNN.
/// </summary>
public class CuDnnPoolingGpuTests
{
    private static bool CuDnnAvailable => CuDnnContext.IsAvailable;

    [Fact]
    public void Pool2DForwardGpu_MaxPool_AcceptsFloatDtype()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var pool = new CuDnnPooling(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int windowH = 2, windowW = 2;
            const int strideH = 2, strideW = 2;
            int outH = h / strideH;
            int outW = w / strideW;

            using var input = ctx.Allocate<float>(n * c * h * w);
            using var output = ctx.Allocate<float>(n * c * outH * outW);
            ctx.CopyToDevice(input, new float[n * c * h * w]);

            var ex = Record.Exception(() =>
                pool.ForwardGpu(
                    input.DevicePtr, output.DevicePtr,
                    n, c, h, w, outH, outW,
                    windowH, windowW, 0, 0, strideH, strideW,
                    CuDnnNative.CudnnPoolingMode.Max));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void Pool2DForwardGpu_AvgPool_AcceptsHalfDtype()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var pool = new CuDnnPooling(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int windowH = 2, windowW = 2;
            const int strideH = 2, strideW = 2;
            int outH = h / strideH;
            int outW = w / strideW;

            // fp16 = 2 bytes per element. Allocate at half-count (with
            // tail rounding) since AllocateBuffer is float-sized.
            using var input = ctx.Allocate<ushort>(n * c * h * w);
            using var output = ctx.Allocate<ushort>(n * c * outH * outW);
            ctx.CopyToDevice(input, new ushort[n * c * h * w]);

            var ex = Record.Exception(() =>
                pool.ForwardGpu(
                    input.DevicePtr, output.DevicePtr,
                    n, c, h, w, outH, outW,
                    windowH, windowW, 0, 0, strideH, strideW,
                    CuDnnNative.CudnnPoolingMode.AverageCountExcludePadding,
                    CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }
}
#endif
