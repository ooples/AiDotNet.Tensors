// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The device roi_align kernels follow torchvision on an aligned RoI of zero or negative size: an empty sampling grid,
/// so every bin is 0. They clamped the grid to one sample and returned the image's value at the box edge.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class RoIAlignDegenerateGpuTests
{
    [SkippableFact]
    public void Aligned_degenerate_rois_average_to_zero_on_the_device()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        bool saved = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            var input = new Tensor<float>(Enumerable.Range(0, 2 * 3 * 7 * 9).Select(i => 1f + (float)Math.Sin(i * 0.37)).ToArray(),
                new[] { 2, 3, 7, 9 });
            // A zero-width box, an inverted box, and an ordinary box beside them.
            var boxes = new Tensor<float>(new float[]
            {
                1, 4.0f, 3.0f, 4.0f, 5.0f,
                0, 6.0f, 2.0f, 5.5f, 1.5f,
                0, 0.5f, 0.25f, 6.0f, 5.5f,
            }, new[] { 3, 5 });

            GpuLaunchProbe.Reset();
            var device = gpu.RoIAlign(input, boxes, 3, 2, 1.0f, 0, true).ToArray();
            Assert.True(GpuLaunchProbe.Count > 0, "The roi_align kernel did not run.");
            var host = new CpuEngine().RoIAlign(input, boxes, 3, 2, 1.0f, 0, true).ToArray();

            int perBox = 3 * 3 * 2;
            for (int i = 0; i < 2 * perBox; i++) Assert.Equal(0f, device[i]);
            for (int i = 0; i < host.Length; i++)
                Assert.True(Math.Abs(host[i] - device[i]) <= 1e-5f * (1 + Math.Abs(host[i])), $"[{i}] cpu {host[i]} vs gpu {device[i]}");
            Assert.Contains(device.Skip(2 * perBox), v => v != 0f);
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = saved;
            gpu.Dispose();
        }
    }
}
