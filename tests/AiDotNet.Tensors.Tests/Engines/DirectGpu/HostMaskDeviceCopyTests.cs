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
}
