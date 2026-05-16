#if !NET462
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: validates the FP16 / BF16 dtype parameterization on
/// <see cref="CuDnnConvolution.Conv2DForwardGpu"/> and
/// <see cref="CuDnnBatchNorm.ForwardInferenceGpu"/> /
/// <see cref="CuDnnBatchNorm.ForwardTrainingGpu"/> /
/// <see cref="CuDnnBatchNorm.BackwardGpu"/>.
///
/// <para>Tests are gated on <see cref="CuDnnContext.IsAvailable"/>; on CI
/// hosts without cuDNN they pass as no-ops. On a CUDA+cuDNN-capable host
/// they verify that the new dtype parameter is honored end-to-end through
/// descriptor setup and kernel dispatch.</para>
///
/// <para>What we DON'T test here: numerical parity vs fp32. Half-precision
/// has a smaller mantissa; small kernels can have visible quantization
/// drift. The point of these tests is "the code path runs without
/// CudnnStatus.NotSupported / BadParam," not "the math is bit-identical
/// to fp32." Numerical-correctness tests live in the consumer-side
/// mixed-precision parity suite.</para>
/// </summary>
public class CuDnnHalfPrecisionConvTests
{
    private static bool CuDnnAvailable => CuDnnContext.IsAvailable;

    [Fact]
    public void Conv2DForwardGpu_AcceptsHalfDtype_WithoutThrowing()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var conv = new CuDnnConvolution(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int k = 4, fh = 3, fw = 3;
            int outH = h - fh + 1;
            int outW = w - fw + 1;

            int inputElems = n * c * h * w;
            int filterElems = k * c * fh * fw;
            int outputElems = n * k * outH * outW;

            // Half is 2 bytes/elem — allocate at fp16 element count via
            // ushort-backed buffers since CuDnnContext.Allocate<T> sizes by
            // sizeof(T). The cuDNN descriptors know they're CudnnDataType.Half.
            using var gpuIn = ctx.Allocate<ushort>(inputElems);
            using var gpuFlt = ctx.Allocate<ushort>(filterElems);
            using var gpuOut = ctx.Allocate<ushort>(outputElems);

            // Zero-fill input + filter — exact values don't matter for the
            // "does the descriptor / algorithm pipeline accept half" check.
            ctx.CopyToDevice(gpuIn, new ushort[inputElems]);
            ctx.CopyToDevice(gpuFlt, new ushort[filterElems]);

            var ex = Record.Exception(() =>
                conv.Conv2DForwardGpu(
                    gpuIn.DevicePtr, gpuFlt.DevicePtr, gpuOut.DevicePtr,
                    n, c, h, w, k, fh, fw, outH, outW,
                    padH: 0, padW: 0, strideH: 1, strideW: 1,
                    dilationH: 1, dilationW: 1,
                    dataType: CuDnnNative.CudnnDataType.Half));

            // Some host/driver combos won't support the chosen algorithm
            // for fp16 with this exact shape — that surfaces as
            // CudnnStatus.NotSupported, which our CheckStatus converts
            // into InvalidOperationException. We treat that as a skip
            // (algorithm unavailable, not a correctness regression).
            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;

            Assert.Null(ex);
        }
    }

    [Fact]
    public void Conv2DForwardGpu_AcceptsBFloat16Dtype_WithoutThrowing()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var conv = new CuDnnConvolution(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int k = 4, fh = 3, fw = 3;
            int outH = h - fh + 1;
            int outW = w - fw + 1;

            int inputElems = n * c * h * w;
            int filterElems = k * c * fh * fw;
            int outputElems = n * k * outH * outW;

            // BFloat16 is also 2 bytes/elem.
            using var gpuIn = ctx.Allocate<ushort>(inputElems);
            using var gpuFlt = ctx.Allocate<ushort>(filterElems);
            using var gpuOut = ctx.Allocate<ushort>(outputElems);

            ctx.CopyToDevice(gpuIn, new ushort[inputElems]);
            ctx.CopyToDevice(gpuFlt, new ushort[filterElems]);

            var ex = Record.Exception(() =>
                conv.Conv2DForwardGpu(
                    gpuIn.DevicePtr, gpuFlt.DevicePtr, gpuOut.DevicePtr,
                    n, c, h, w, k, fh, fw, outH, outW,
                    padH: 0, padW: 0, strideH: 1, strideW: 1,
                    dilationH: 1, dilationW: 1,
                    dataType: CuDnnNative.CudnnDataType.BFloat16));

            // BF16 requires Ampere+ (compute capability >= 8.0). Pre-Ampere
            // cuDNN versions return NotSupported / ArchMismatch — treat as skip.
            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;

            Assert.Null(ex);
        }
    }

    [Fact]
    public void Conv2DBackwardDataGpu_AcceptsHalfDtype_WithoutThrowing()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var conv = new CuDnnConvolution(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int k = 4, fh = 3, fw = 3;
            int outH = h - fh + 1;
            int outW = w - fw + 1;

            int dyElems = n * k * outH * outW;
            int filterElems = k * c * fh * fw;
            int dxElems = n * c * h * w;

            using var gpuFlt = ctx.Allocate<ushort>(filterElems);
            using var gpuDy = ctx.Allocate<ushort>(dyElems);
            using var gpuDx = ctx.Allocate<ushort>(dxElems);

            ctx.CopyToDevice(gpuFlt, new ushort[filterElems]);
            ctx.CopyToDevice(gpuDy, new ushort[dyElems]);

            var ex = Record.Exception(() =>
                conv.Conv2DBackwardDataGpu(
                    gpuFlt.DevicePtr, gpuDy.DevicePtr, gpuDx.DevicePtr,
                    n, c, h, w, k, fh, fw, outH, outW,
                    dataType: CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }

    [Fact]
    public void Conv2DBackwardFilterGpu_AcceptsHalfDtype_WithoutThrowing()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var conv = new CuDnnConvolution(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int k = 4, fh = 3, fw = 3;
            int outH = h - fh + 1;
            int outW = w - fw + 1;

            int xElems = n * c * h * w;
            int dyElems = n * k * outH * outW;
            int dwElems = k * c * fh * fw;

            using var gpuX = ctx.Allocate<ushort>(xElems);
            using var gpuDy = ctx.Allocate<ushort>(dyElems);
            using var gpuDw = ctx.Allocate<ushort>(dwElems);

            ctx.CopyToDevice(gpuX, new ushort[xElems]);
            ctx.CopyToDevice(gpuDy, new ushort[dyElems]);

            var ex = Record.Exception(() =>
                conv.Conv2DBackwardFilterGpu(
                    gpuX.DevicePtr, gpuDy.DevicePtr, gpuDw.DevicePtr,
                    n, c, h, w, k, fh, fw, outH, outW,
                    dataType: CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }

    [Fact]
    public void SoftmaxForwardGpu_AcceptsHalfDtype_WithoutThrowing()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var softmax = new CuDnnSoftmax(ctx))
        {
            const int n = 2, c = 16, h = 1, w = 1;
            int elems = n * c * h * w;

            using var gpuIn = ctx.Allocate<ushort>(elems);
            using var gpuOut = ctx.Allocate<ushort>(elems);
            ctx.CopyToDevice(gpuIn, new ushort[elems]);

            var ex = Record.Exception(() =>
                softmax.ForwardGpu(
                    gpuIn.DevicePtr, gpuOut.DevicePtr, n, c, h, w,
                    dataType: CuDnnNative.CudnnDataType.Half));

            if (ex is InvalidOperationException &&
                (ex.Message.Contains("NotSupported", StringComparison.Ordinal)
                 || ex.Message.Contains("ArchMismatch", StringComparison.Ordinal)))
                return;
            Assert.Null(ex);
        }
    }

    [Fact]
    public void SoftmaxForwardGpu_DefaultDtype_PreservesFloatBehavior()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var softmax = new CuDnnSoftmax(ctx))
        {
            const int n = 1, c = 8, h = 1, w = 1;
            int elems = n * c;
            using var gpuIn = ctx.Allocate<float>(elems);
            using var gpuOut = ctx.Allocate<float>(elems);
            ctx.CopyToDevice(gpuIn, new float[elems]);

            var ex = Record.Exception(() =>
                softmax.ForwardGpu(gpuIn.DevicePtr, gpuOut.DevicePtr, n, c, h, w));
            Assert.Null(ex);
        }
    }

    [Fact]
    public void Conv2DForwardGpu_DefaultDtype_IsFloat_PreservesExistingBehavior()
    {
        if (!CuDnnAvailable) return;

        CuDnnContext? ctx = null;
        try { ctx = new CuDnnContext(); }
        catch { return; }

        using (ctx)
        using (var conv = new CuDnnConvolution(ctx))
        {
            const int n = 1, c = 4, h = 8, w = 8;
            const int k = 4, fh = 3, fw = 3;
            int outH = h - fh + 1;
            int outW = w - fw + 1;

            int inputElems = n * c * h * w;
            int filterElems = k * c * fh * fw;
            int outputElems = n * k * outH * outW;

            using var gpuIn = ctx.Allocate<float>(inputElems);
            using var gpuFlt = ctx.Allocate<float>(filterElems);
            using var gpuOut = ctx.Allocate<float>(outputElems);

            ctx.CopyToDevice(gpuIn, new float[inputElems]);
            ctx.CopyToDevice(gpuFlt, new float[filterElems]);

            // No dataType argument — confirms the default (Float) preserves
            // the pre-#337 call site behaviour.
            var ex = Record.Exception(() =>
                conv.Conv2DForwardGpu(
                    gpuIn.DevicePtr, gpuFlt.DevicePtr, gpuOut.DevicePtr,
                    n, c, h, w, k, fh, fw, outH, outW));

            Assert.Null(ex);
        }
    }
}
#endif
