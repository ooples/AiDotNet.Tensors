#if !NET462
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: smoke tests for the new CuDnnActivation class. Forward
/// + Backward both accept an optional CudnnDataType parameter.
/// </summary>
public class CuDnnActivationGpuTests
{
    private static bool CuDnnAvailable => CuDnnContext.IsAvailable;

    [Fact]
    public void ActivationForwardGpu_ReLU_AcceptsFloatDtype()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var act = new CuDnnActivation(ctx))
        {
            const int n = 1, c = 4, h = 4, w = 4;
            int elems = n * c * h * w;
            using var input = ctx.Allocate<float>(elems);
            using var output = ctx.Allocate<float>(elems);
            ctx.CopyToDevice(input, new float[elems]);

            var ex = Record.Exception(() =>
                act.ForwardGpu(
                    input.DevicePtr, output.DevicePtr,
                    n, c, h, w,
                    CuDnnNative.CudnnActivationMode.ReLU));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void ActivationForwardGpu_Tanh_AcceptsHalfDtype()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var act = new CuDnnActivation(ctx))
        {
            const int n = 1, c = 4, h = 4, w = 4;
            int elems = n * c * h * w;
            using var input = ctx.Allocate<ushort>(elems);
            using var output = ctx.Allocate<ushort>(elems);
            ctx.CopyToDevice(input, new ushort[elems]);

            var ex = Record.Exception(() =>
                act.ForwardGpu(
                    input.DevicePtr, output.DevicePtr,
                    n, c, h, w,
                    CuDnnNative.CudnnActivationMode.Tanh,
                    coef: 0.0,
                    dataType: CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }

    [Fact]
    public void ActivationBackwardGpu_Sigmoid_AcceptsFloatDtype()
    {
        if (!CuDnnAvailable) return;
        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); } catch { return; }

        using (ctx)
        using (var act = new CuDnnActivation(ctx))
        {
            const int n = 1, c = 4, h = 4, w = 4;
            int elems = n * c * h * w;
            using var input = ctx.Allocate<float>(elems);
            using var output = ctx.Allocate<float>(elems);
            using var gradOutput = ctx.Allocate<float>(elems);
            using var gradInput = ctx.Allocate<float>(elems);
            ctx.CopyToDevice(input, new float[elems]);

            var ex = Record.Exception(() =>
                act.BackwardGpu(
                    output.DevicePtr, gradOutput.DevicePtr,
                    input.DevicePtr, gradInput.DevicePtr,
                    n, c, h, w,
                    CuDnnNative.CudnnActivationMode.Sigmoid));
            Assert.Null(ex);
        }
    }
}
#endif
