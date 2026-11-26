// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The bf16 device copy an engine makes of a bf16 streaming weight is freed when the engine is disposed. Before, the
/// copy lived as long as the weight (it was only reachable from a weak table keyed on it), so on a backend the engine
/// shared with others it held device memory after the engine was gone, until the weight was collected and the buffer's
/// finalizer ran.
/// </summary>
[Collection("WeightRegistry")]
public class Bf16DeviceWeightDisposeTests
{
    [SkippableFact]
    public void Disposing_the_engine_frees_the_bf16_copy_of_a_weight_that_is_still_alive()
    {
        DirectGpuEngine direct;
        try { direct = new DirectGpuEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        try
        {
            Skip.IfNot(direct.IsAvailable && direct.Backend is CudaBackend, "Needs the CUDA backend (reads the buffer's release state).");
            var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-dispose-{Guid.NewGuid():N}");
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingBackingStorePath = dir,
                StreamingPoolMaxResidentBytes = 1024L * 1024,
                StreamingStoreDtype = StreamingStoreDtype.Bf16,
            });
            WeightRegistry.SetStreamingExecutionTraining(false);
            try
            {
                var w = new Tensor<float>(new[] { 8, 5 });
                for (int i = 0; i < w.Length; i++) w.SetFlat(i, (i % 9 - 4) / 16f);
                w.Lifetime = WeightLifetime.Streaming;
                WeightRegistry.RegisterWeight(w);
                var x = new Tensor<float>(new[] { 3, 8 });
                for (int i = 0; i < x.Length; i++) x.SetFlat(i, (i % 5 - 2) / 8f);

                var gpu = new DirectGpuTensorEngine(direct);
                gpu.FusedLinear(x, w, null, FusedActivationType.None);
                var copy = gpu.Bf16DeviceCopyOf(w) as CudaBackend.CudaGpuBuffer;
                Assert.NotNull(copy);
                Assert.False(copy!.IsReleased);

                gpu.Dispose();

                Assert.True(copy.IsReleased, "the bf16 device copy outlived the engine");
                GC.KeepAlive(w);
            }
            finally
            {
                WeightRegistry.SetStreamingExecutionTraining(null);
                WeightRegistry.Reset();
                if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
            }
        }
        finally { direct.Dispose(); }
    }
}
