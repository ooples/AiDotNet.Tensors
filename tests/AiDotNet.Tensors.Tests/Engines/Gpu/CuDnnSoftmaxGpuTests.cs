#if !NET462
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: smoke tests for CuDnnSoftmax's new GPU-pointer helpers.
/// Forward + Backward both accept an optional CudnnDataType parameter.
/// </summary>
public class CuDnnSoftmaxGpuTests
{
    private static bool CuDnnAvailable => CuDnnContext.IsAvailable;

    [Fact]
    public void SoftmaxForwardGpu_DefaultDtype_AcceptsFloat()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var softmax = new CuDnnSoftmax(ctx))
        {
            const int n = 2, c = 16, h = 1, w = 1;
            int elems = n * c * h * w;
            using var input = ctx.Allocate<float>(elems);
            using var output = ctx.Allocate<float>(elems);
            ctx.CopyToDevice(input, new float[elems]);

            var ex = Record.Exception(() =>
                softmax.ForwardGpu(input.DevicePtr, output.DevicePtr, n, c, h, w));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void SoftmaxBackwardGpu_DefaultDtype_AcceptsFloat()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var softmax = new CuDnnSoftmax(ctx))
        {
            const int n = 2, c = 16, h = 1, w = 1;
            int elems = n * c * h * w;
            using var output = ctx.Allocate<float>(elems);
            using var gradOutput = ctx.Allocate<float>(elems);
            using var gradInput = ctx.Allocate<float>(elems);
            ctx.CopyToDevice(output, new float[elems]);

            var ex = Record.Exception(() =>
                softmax.BackwardGpu(
                    output.DevicePtr, gradOutput.DevicePtr, gradInput.DevicePtr,
                    n, c, h, w));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void SoftmaxForwardGpu_HalfDtype_AcceptsHalf()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var softmax = new CuDnnSoftmax(ctx))
        {
            const int n = 1, c = 8, h = 1, w = 1;
            int elems = n * c * h * w;
            using var input = ctx.Allocate<ushort>(elems);
            using var output = ctx.Allocate<ushort>(elems);
            ctx.CopyToDevice(input, new ushort[elems]);

            var ex = Record.Exception(() =>
                softmax.ForwardGpu(input.DevicePtr, output.DevicePtr,
                    n, c, h, w, CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }
}
#endif
