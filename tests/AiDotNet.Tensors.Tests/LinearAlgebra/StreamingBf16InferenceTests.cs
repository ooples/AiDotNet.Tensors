// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Inference over weights held in the streaming pool's bf16 store (read-only by design). The CPU engine read a bf16
/// weight's bias through the write-intent accessor, so a linear layer over bf16 weights threw "Cannot mutate a
/// streaming tensor"; the GPU engine uploaded such a weight at fp32 size. The values here are exact in bf16, so every
/// path must reproduce the plain fp32 reference.
/// </summary>
[Collection("WeightRegistry")]
public class StreamingBf16InferenceTests
{
    private const int M = 3, K = 8, N = 5;

    private static float Value(int i, int salt) => ((i * 7 + salt) % 17 - 8) / 64f;   // exact in bf16

    private static Tensor<T> Filled<T>(int[] shape, int salt, Func<float, T> cast)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, cast(Value(i, salt)));
        return t;
    }

    private static string Configure()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-inf-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        WeightRegistry.SetStreamingExecutionTraining(null);
        WeightRegistry.Reset();
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    private static Tensor<T> Stream<T>(Tensor<T> t)
    {
        t.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal((byte)1, t.StreamingStoreEncoding);   // bf16
        return t;
    }

    private static void Linear<T>(IEngine engine, Func<float, T> cast, Func<T, double> back)
    {
        var x = Filled(new[] { M, K }, 1, cast);
        // Reference: the same values in ordinary (non-streamed) tensors.
        var expected = new CpuEngine().FusedLinear(x, Filled(new[] { K, N }, 2, cast), Filled(new[] { N }, 3, cast), FusedActivationType.None);
        var dir = Configure();
        try
        {
            var w = Stream(Filled(new[] { K, N }, 2, cast));
            var b = Stream(Filled(new[] { N }, 3, cast));
            var y = engine.FusedLinear(x, w, b, FusedActivationType.None);
            for (int i = 0; i < M * N; i++)
                Assert.True(Math.Abs(back(y.GetFlat(i)) - back(expected.GetFlat(i))) <= 1e-5,
                    $"[{i}] {back(y.GetFlat(i))} vs {back(expected.GetFlat(i))}");
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void Cpu_linear_layer_reads_bf16_streamed_weights_and_bias_float() => Linear(new CpuEngine(), v => v, v => v);

    [Fact]
    public void Cpu_linear_layer_reads_bf16_streamed_weights_and_bias_double() => Linear(new CpuEngine(), v => (double)v, v => v);

    [SkippableFact]
    public void Gpu_keeps_bf16_streamed_weights_bf16_on_the_device_and_matches_the_reference()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        try
        {
            long before = gpu.Bf16WeightBytesUploaded;
            Linear<float>(gpu, v => v, v => v);
            // The weight went up as bf16 (2 bytes per element), not through the fp32 host view.
            Assert.True(gpu.Bf16WeightBytesUploaded - before >= 2L * K * N,
                $"bf16 weight bytes uploaded {gpu.Bf16WeightBytesUploaded - before}, expected at least {2L * K * N}");
        }
        finally { gpu.Dispose(); }
    }

    /// <summary>
    /// A device byte budget on the bf16 weight copies: three layers whose bf16 copies need more than the budget run with
    /// the least recently used copies evicted and re-uploaded on reuse, and every output equals the unbounded result.
    /// Without the budget the engine kept every copy (device bytes above the budget, no evictions), so a model whose bf16
    /// weights exceed the card could not run.
    /// </summary>
    [SkippableFact]
    public void Gpu_bf16_weight_budget_evicts_least_recently_used_copies_and_keeps_results()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true }) { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies"); return; }
        var cpu = new CpuEngine();
        var x = Filled(new[] { M, K }, 1, v => v);
        var dir = Configure();
        try
        {
            const int layers = 3;
            var w = new Tensor<float>[layers];
            var b = new Tensor<float>[layers];
            var expected = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++)
            {
                expected[l] = cpu.FusedLinear(x, Filled(new[] { K, N }, 2 + l, v => v), Filled(new[] { N }, 7 + l, v => v), FusedActivationType.None);
                w[l] = Stream(Filled(new[] { K, N }, 2 + l, v => v));
                b[l] = Stream(Filled(new[] { N }, 7 + l, v => v));
            }
            long pair = 4L * ((2 * K * N + 3) / 4) + 4L * ((2 * N + 3) / 4);   // one layer's device copies
            gpu.Bf16WeightDeviceBudgetBytes = 2 * pair;
            long evictions0 = gpu.Bf16WeightEvictions, uploaded0 = gpu.Bf16WeightBytesUploaded;
            foreach (int l in new[] { 0, 1, 2, 0 })
            {
                var y = gpu.FusedLinear(x, w[l], b[l], FusedActivationType.None);
                for (int i = 0; i < M * N; i++)
                    Assert.True(Math.Abs(y.GetFlat(i) - expected[l].GetFlat(i)) <= 1e-5,
                        $"layer {l} [{i}] {y.GetFlat(i)} vs {expected[l].GetFlat(i)}");
                Assert.True(gpu.Bf16WeightDeviceBytes <= 2 * pair,
                    $"after layer {l}: {gpu.Bf16WeightDeviceBytes} device bytes held, budget {2 * pair}");
            }
            // Layer 2 evicts layer 0's pair; layer 0 again evicts layer 1's pair and is uploaded a second time.
            Assert.Equal(4, gpu.Bf16WeightEvictions - evictions0);
            Assert.Equal(2L * (4 * (K * N + N)), gpu.Bf16WeightBytesUploaded - uploaded0);
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    /// <summary>
    /// A bf16 copy whose weight has been collected is released by the next upload, on an unbounded engine too. The copies
    /// are held strongly (the eviction list), so without that release a dropped model's copies stayed on the device until
    /// the engine was disposed.
    /// </summary>
    [SkippableFact]
    public void Gpu_bf16_copy_of_a_collected_weight_is_released_by_the_next_upload()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true }) { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies"); return; }
        var x = Filled(new[] { M, K }, 1, v => v);
        var dir = Configure();
        try
        {
            long pair = 4L * ((2 * K * N + 3) / 4) + 4L * ((2 * N + 3) / 4);   // one layer's device copies
            long before = gpu.Bf16WeightDeviceBytes;
            RunDroppedLayer(gpu, x);
            Assert.Equal(before + pair, gpu.Bf16WeightDeviceBytes);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var w = Stream(Filled(new[] { K, N }, 5, v => v));
            var b = Stream(Filled(new[] { N }, 6, v => v));
            gpu.FusedLinear(x, w, b, FusedActivationType.None);
            Assert.Equal(before + pair, gpu.Bf16WeightDeviceBytes);   // the dropped layer's copies were released
            Assert.Equal(0, gpu.Bf16WeightEvictions);                  // released as dead, not evicted: no budget
            GC.KeepAlive(w);
            GC.KeepAlive(b);
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    // A layer whose weights are unreachable once this returns.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunDroppedLayer(DirectGpuTensorEngine gpu, Tensor<float> x)
    {
        var w = Stream(Filled(new[] { K, N }, 2, v => v));
        var b = Stream(Filled(new[] { N }, 3, v => v));
        gpu.FusedLinear(x, w, b, FusedActivationType.None);
    }

    /// <summary>
    /// A bf16 streaming weight trained after its device copy was made: switching to training promotes the store to a
    /// writable encoding, and each of several steps writes new values into the weight (as an optimizer step does) and
    /// runs the op on the GPU and the CPU. The GPU must read the written values, bit for bit with the CPU, and the bf16
    /// device copies made before the promotion must be released, not held until the weights are collected.
    /// </summary>
    [SkippableFact]
    public void Gpu_bf16_weight_written_after_its_device_copy_is_read_with_the_written_values()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        var cpu = new CpuEngine();
        var x = Filled(new[] { M, K }, 1, v => v);
        var dir = Configure();
        try
        {
            var w = Stream(Filled(new[] { K, N }, 2, v => v));
            var b = Stream(Filled(new[] { N }, 3, v => v));
            gpu.FusedLinear(x, w, b, FusedActivationType.None);   // makes the device copy from the stored bytes
            WeightRegistry.SetStreamingExecutionTraining(true);
            for (int step = 1; step <= 3; step++)
            {
                WeightRegistry.Materialize(w);
                for (int i = 0; i < w.Length; i++) w.SetFlat(i, Value(i, 2 + 5 * step));   // exact in bf16
                var expected = cpu.FusedLinear(x, w, b, FusedActivationType.None);
                var actual = gpu.FusedLinear(x, w, b, FusedActivationType.None);
                for (int i = 0; i < M * N; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected.GetFlat(i)) == TestHelpers.MathCompat.SingleToInt32Bits(actual.GetFlat(i)),
                        $"step {step} [{i}]: GPU {actual.GetFlat(i)} vs CPU {expected.GetFlat(i)}");
            }
            // The copies made from the bf16 store are dead once it is promoted, and their device memory is released.
            Assert.Equal(0L, gpu.Bf16WeightDeviceBytes);
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    /// <summary>
    /// A streamed bf16 load whose layers ran a GPU forward before their weights were streamed (a lazy layer's
    /// materializing forward, as an MoE builder runs for every expert): each forward cached the weight's fp32 device
    /// copy under its host array, and streaming the weight dropped the array but not the copy, so device memory grew by
    /// every layer's full-precision weights, outside the bf16 budget. The peak must stay within the budget plus one
    /// layer's working set.
    /// </summary>
    [SkippableFact]
    public void A_streamed_load_after_gpu_forwards_stays_within_the_budget_plus_one_layer()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable || gpu.GetBackend() is not { } backend) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        const int Layers = 12, In = 256, Out = 512;
        long layerBytes = 4L * In * Out;
        var dir = Configure();
        try
        {
            gpu.Bf16WeightDeviceBudgetBytes = 2 * 2L * In * Out;   // two layers' bf16 copies
            var x = Filled(new[] { 1, In }, 1, v => v);
            var warm = Filled(new[] { In, Out }, 2, v => v);
            gpu.FusedLinear(x, warm, null, FusedActivationType.None);   // the engine's own scratch, before the baseline
            long baseline = GpuMemoryStats.Snapshot(backend).AllocatedBytes;
            int cachedBefore = gpu.CachedTensorCount;
            long peak = 0;
            var kept = new Tensor<float>[Layers];
            for (int l = 0; l < Layers; l++)
            {
                var w = Filled(new[] { In, Out }, 10 + l, v => v);
                gpu.FusedLinear(x, w, null, FusedActivationType.None);   // the materializing forward
                kept[l] = Stream(w);
                gpu.FusedLinear(x, w, null, FusedActivationType.None);   // inference on the streamed weight
                peak = Math.Max(peak, GpuMemoryStats.Snapshot(backend).AllocatedBytes - baseline);
            }
            Assert.Equal(cachedBefore, gpu.CachedTensorCount);
            Assert.True(peak <= gpu.Bf16WeightDeviceBudgetBytes + 2 * layerBytes,
                $"device memory grew by {peak} bytes over {Layers} layers; budget {gpu.Bf16WeightDeviceBudgetBytes} + one layer's {layerBytes} (x2 for its output and scratch)");
            GC.KeepAlive(kept);
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }
}
