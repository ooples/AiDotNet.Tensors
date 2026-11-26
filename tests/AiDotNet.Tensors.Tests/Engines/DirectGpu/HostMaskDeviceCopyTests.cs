using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A host mask's device copy (made once so a captured step reads a stable buffer) is freed with the engine that made
/// it. It used to live until the mask was collected, on a backend the engine may share with others.
/// </summary>
public sealed class HostMaskDeviceCopyTests
{
    [SkippableFact]
    public void A_masks_device_copy_is_freed_when_the_engine_is_disposed()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        var x = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, i);
        var mask = new Tensor<bool>(new[] { 2, 3 });
        mask.SetFlat(1, true);
        mask.SetFlat(4, true);
        var y = gpu.TensorMaskedFill(x, mask, -1f);
        Assert.Equal(-1f, y.GetFlat(1));
        Assert.Equal(1, gpu.LiveHostMaskDeviceCopiesForTests);
        gpu.Dispose();
        Assert.Equal(0, gpu.LiveHostMaskDeviceCopiesForTests);
        GC.KeepAlive(mask);
    }

    /// <summary>
    /// A mask whose device copy was made on a backend that has since closed is refreshed on another engine, and every
    /// later use of the mask still works. A regression guard for the release order (the copy is cleared before its
    /// stale buffer is released, and a closed backend is never asked to queue the free); it passed on the code before
    /// that change as well, since the CUDA queued free did not throw there.
    /// </summary>
    [SkippableFact]
    public void A_masks_copy_from_a_closed_backend_is_replaced_on_another_engine()
    {
        AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend? first;
        try { first = AiDotNet.Tensors.Engines.DirectGpu.DirectGpuBackendFactory.CreateNew(); }
        catch { first = null; }
        Skip.If(first is null, "No GPU backend");
        var x = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, i);
        var mask = new Tensor<bool>(new[] { 2, 3 });
        mask.SetFlat(1, true);
        var a = new DirectGpuTensorEngine(new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(first!, borrowsBackend: true));
        Assert.Equal(-1f, a.TensorMaskedFill(x, mask, -1f).GetFlat(1));
        first!.Dispose();   // the backend closes under the engine that made the copy

        DirectGpuTensorEngine b;
        try { b = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No second GPU engine"); return; }
        using (b)
        {
            Assert.Equal(-1f, b.TensorMaskedFill(x, mask, -1f).GetFlat(1));
            Assert.Equal(-1f, b.TensorMaskedFill(x, mask, -1f).GetFlat(1));
        }
        a.Dispose();
        GC.KeepAlive(mask);
    }
}
