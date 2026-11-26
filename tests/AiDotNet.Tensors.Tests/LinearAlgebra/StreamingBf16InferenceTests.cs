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

#if NET5_0_OR_GREATER
    /// <summary>
    /// Evicting and re-uploading bf16 copies under a device budget makes no host copy of the weight. Each upload used to
    /// allocate two arrays of the weight's size (the store snapshot and a packed float[] of it); a model whose bf16
    /// weights exceed the budget re-uploads on every pass, and its host memory grew about two bytes per byte uploaded
    /// faster than the collector reclaimed it (a 7 GiB budget during one evaluation forward: 14 GB of host growth).
    /// </summary>
    /// <remarks>Both with the weights resident in the pool and paged out of it: a paged-out weight was read back into a
    /// fresh resident array on every upload (two for a compressed store), evicting another whose array became garbage.
    /// </remarks>
    [SkippableTheory]
    [InlineData(64L * 1024 * 1024, false)]
    [InlineData(64L * 1024, false)]
    [InlineData(64L * 1024, true)]
    public void Gpu_bf16_evict_and_reupload_cycles_make_no_host_copy_of_the_weight(long poolResidentBytes, bool compressed)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true }) { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies"); return; }
        const int k = 512, n = 1024;   // 1 MiB of bf16 per weight
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-churn-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            // 64 MiB keeps both weights resident in the pool; 64 KiB pages them out, so every re-upload reads the disk.
            StreamingPoolMaxResidentBytes = poolResidentBytes,
            EnableCompression = compressed,
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        try
        {
            var x = Filled(new[] { 1, k }, 1, v => v);
            var w = new[] { Stream(Filled(new[] { k, n }, 2, v => v)), Stream(Filled(new[] { k, n }, 3, v => v)) };
            var b = new[] { Stream(Filled(new[] { n }, 4, v => v)), Stream(Filled(new[] { n }, 5, v => v)) };
            long pair = 4L * ((2L * k * n + 3) / 4) + 4L * ((2L * n + 3) / 4);
            gpu.Bf16WeightDeviceBudgetBytes = pair;   // one layer: alternating layers evict and re-upload every time
            for (int l = 0; l < 2; l++) _ = gpu.FusedLinear(x, w[l], b[l], FusedActivationType.None).GetFlat(0);   // warm-up

            const int cycles = 20;
            long uploaded0 = gpu.Bf16WeightBytesUploaded, evictions0 = gpu.Bf16WeightEvictions;
            long allocated0 = GC.GetAllocatedBytesForCurrentThread();
            for (int c = 0; c < cycles; c++) _ = gpu.FusedLinear(x, w[c % 2], b[c % 2], FusedActivationType.None);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - allocated0;
            long uploaded = gpu.Bf16WeightBytesUploaded - uploaded0;

            Assert.True(gpu.Bf16WeightEvictions - evictions0 >= 2 * (cycles - 1), "the budget did not force re-uploads");
            Assert.True(uploaded >= (long)cycles * 2 * k * n, $"uploaded {uploaded} bytes");
            if (poolResidentBytes < 2L * k * n)
                Assert.True(WeightRegistry.StreamingPool.GetReport().DiskReadBytes > 0, "the weights never paged out");
            // A few small host results and bookkeeping per call; one weight-sized array per upload would be 1 MiB each.
            Assert.True(allocated < uploaded / 8,
                $"{allocated} host bytes allocated for {uploaded} bf16 bytes uploaded over {cycles} re-uploads");
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }
#endif

    /// <summary>
    /// Device memory over many evict and re-upload cycles stays within the bf16 budget plus one layer's working set (its
    /// fp32 widened weight and output): the widened temporaries and evicted copies are reused or freed, not accumulated.
    /// </summary>
    [SkippableFact]
    public void Gpu_bf16_evict_and_reupload_cycles_keep_device_memory_within_the_budget_plus_one_layer()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true } || gpu.GetBackend() is not IDeviceMemoryPressure memory
            || !memory.TryGetDeviceMemory(out _, out _))
        { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies or reports no device memory"); return; }
        const int k = 2048, n = 4096, layers = 4;   // 16 MiB of bf16, 32 MiB widened, per weight
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-dev-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 256L * 1024 * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        try
        {
            var x = Filled(new[] { 8, k }, 1, v => v);
            var w = new Tensor<float>[layers];
            var b = new Tensor<float>[layers];
            for (int l = 0; l < layers; l++) { w[l] = Stream(Filled(new[] { k, n }, 2 + l, v => v)); b[l] = Stream(Filled(new[] { n }, 9 + l, v => v)); }
            long pair = 4L * ((2L * k * n + 3) / 4) + 4L * ((2L * n + 3) / 4);
            gpu.Bf16WeightDeviceBudgetBytes = 2 * pair;
            for (int l = 0; l < layers; l++) _ = gpu.FusedLinear(x, w[l], b[l], FusedActivationType.None).GetFlat(0);
            memory.TryGetDeviceMemory(out ulong free0, out _);
            long peakGrowth = 0;
            for (int c = 0; c < 6 * layers; c++)
            {
                _ = gpu.FusedLinear(x, w[c % layers], b[c % layers], FusedActivationType.None).GetFlat(0);
                memory.TryGetDeviceMemory(out ulong free, out _);
                peakGrowth = Math.Max(peakGrowth, (long)free0 - (long)free);
            }
            long layerWorkingSet = 4L * k * n + 4L * n + 4L * 8 * n;
            Assert.True(peakGrowth <= gpu.Bf16WeightDeviceBudgetBytes + layerWorkingSet,
                $"device use grew {peakGrowth / 1048576.0:F1} MiB over the warm state; budget {gpu.Bf16WeightDeviceBudgetBytes / 1048576.0:F1} MiB, one layer {layerWorkingSet / 1048576.0:F1} MiB");
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    /// <summary>
    /// A layer-by-layer GPU forward over bf16-stored weights (each weight made ready before its op, as a layer's
    /// forward does) holds no fp32 host copy of them: the ops read the bf16 device copies. A Materialize before each op
    /// restored an fp32 host copy that stayed referenced up to the owner budget (half the pool's), so a mixture of
    /// experts grew the host by two bytes per bf16 byte its forward reached. A host read still sees the values.
    /// </summary>
    [SkippableFact]
    public void Gpu_forward_over_bf16_weights_holds_no_fp32_host_copy_of_them()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true }) { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies"); return; }
        const int k = 512, n = 1024, experts = 16;   // 2 MiB of fp32 per expert weight
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-experts-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 256L * 1024 * 1024,   // all resident; the owner budget is 128 MiB
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        try
        {
            var x = Filled(new[] { 4, k }, 1, v => v);
            var w = new Tensor<float>[experts];
            var b = new Tensor<float>[experts];
            for (int e = 0; e < experts; e++)
            {
                w[e] = Stream(Filled(new[] { k, n }, 10 + e, v => v));
                b[e] = Stream(Filled(new[] { n }, 100 + e, v => v));
            }
            var cpu = new CpuEngine();
            var expected = new float[experts][];
            for (int e = 0; e < experts; e++)
                expected[e] = cpu.FusedLinear(x, w[e], b[e], FusedActivationType.None).ToArray();
            for (int e = 0; e < experts; e++) { WeightRegistry.ReleaseToPool(w[e]); WeightRegistry.ReleaseToPool(b[e]); }

            long before = GC.GetTotalMemory(forceFullCollection: true);
            var outputs = new Tensor<float>[experts];
            using (AiDotNetEngine.UseInCurrentContext(gpu))
                for (int e = 0; e < experts; e++)
                {
                    // A layer makes its weights ready before its op takes a lease: the query must need none.
                    gpu.StrictGpuLeases = true;
                    try
                    {
                        WeightRegistry.MaterializeForForward(w[e], gpu);
                        WeightRegistry.MaterializeForForward(b[e], gpu);
                    }
                    finally { gpu.StrictGpuLeases = false; }
                    outputs[e] = gpu.FusedLinear(x, w[e], b[e], FusedActivationType.None);
                    _ = outputs[e].GetFlat(0);
                }
            long held = GC.GetTotalMemory(forceFullCollection: true) - before;
            GC.KeepAlive(outputs);

            long oneWeight = 4L * k * n;
            Assert.True(held < oneWeight,
                $"{held} host bytes held after a forward over {experts} bf16 weights of {oneWeight} fp32 bytes each");
            // A view of a weight the forward left on the device pages it back in (a view shares the storage, and the
            // constructor used to find it empty): here the transposed weight's product on the GPU.
            WeightRegistry.MaterializeForForward(w[5], gpu);
            Assert.True(w[5].DataVector.Length != w[5].Length, "the forward restored a host copy of the weight");
            var viaView = gpu.TensorMatMul(x, w[5].Transpose(new[] { 1, 0 }).Transpose(new[] { 1, 0 })).ToArray();
            var direct = cpu.TensorMatMul(x, w[5]).ToArray();
            for (int i = 0; i < direct.Length; i++)
                Assert.True(Math.Abs(viaView[i] - direct[i]) <= 2e-2f * (1f + Math.Abs(direct[i])), $"view [{i}]: {viaView[i]} vs {direct[i]}");
            // The engine defaults to the current one.
            using (AiDotNetEngine.UseInCurrentContext(gpu)) WeightRegistry.MaterializeForForward(w[6]);
            Assert.True(w[6].DataVector.Length != w[6].Length, "the forward restored a host copy of the weight");
            var reshaped = w[6].Reshape(k * n).Reshape(k, n);   // built while the weight is dropped
            Assert.Equal(cpu.TensorMatMul(x, w[6]).ToArray(), cpu.TensorMatMul(x, reshaped).ToArray());
            // So does an in-place strided view of its storage.
            WeightRegistry.MaterializeForForward(w[7], gpu);
            Assert.True(w[7].DataVector.Length != w[7].Length, "the forward restored a host copy of the weight");
            var strided = new Tensor<float>(new[] { 1 });
            cpu.TensorSetInPlace(strided, w[7], 0, new[] { k, n }, new[] { n, 1 });
            Assert.Equal(cpu.TensorMatMul(x, w[7]).ToArray(), cpu.TensorMatMul(x, strided).ToArray());
            // A host read of a weight the forward left on the device pages it back in.
            Assert.Equal(new CpuEngine().FusedLinear(x, w[3], b[3], FusedActivationType.None).ToArray(), expected[3]);
            for (int e = 0; e < experts; e++)
            {
                var got = outputs[e].ToArray();
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - expected[e][i]) <= 2e-2f * (1f + Math.Abs(expected[e][i])),
                        $"expert {e} [{i}]: {got[i]} vs {expected[e][i]}");
            }
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    /// <summary>
    /// A view of a streaming weight that is already resident runs no owner-budget shed pass: it paged nothing in. A pass
    /// per view (a bias reshaped every forward) retried, under the registry lock, the write-back of a cold owner a live
    /// view pinned. A view that pages the weight in still runs one.
    /// </summary>
    [Fact]
    public void A_view_of_a_resident_streaming_weight_runs_no_shed_pass()
    {
        var dir = Configure();
        try
        {
            var w = Stream(Filled(new[] { K, N }, 2, v => v));
            WeightRegistry.Materialize(w);
            Assert.True(w.DataVector.Length == w.Length);
            long before = WeightRegistry.OwnerShedPassesOnThisThread;
            for (int i = 0; i < 20; i++) _ = w.Reshape(K * N);
            Assert.Equal(before, WeightRegistry.OwnerShedPassesOnThisThread);
            // The views above share w's storage, so w cannot be dropped now; a weight registration dropped pages in.
            var cold = Stream(Filled(new[] { K, N }, 2, v => v));
            Assert.True(cold.DataVector.Length != cold.Length);
            long beforePageIn = WeightRegistry.OwnerShedPassesOnThisThread;
            var paged = cold.Reshape(K * N);
            // Two passes: the page-in's own (Materialize), and the view's when it ends its use.
            Assert.True(WeightRegistry.OwnerShedPassesOnThisThread >= beforePageIn + 2, "a view that paged the weight in ran no shed pass");
            Assert.Equal(Filled(new[] { K, N }, 2, v => v).ToArray(), paged.ToArray());
        }
        finally { Cleanup(dir); }
    }

    /// <summary>
    /// An in-place strided view of a no-upcast int8 weight's storage dequantizes it first, as a host read does; it built
    /// a view over the empty fp32 storage. Re-striding a registered streaming weight itself is refused: it made the weight
    /// a view, which the pool never writes back or sheds.
    /// </summary>
    [Fact]
    public void In_place_strided_views_of_streaming_weights_page_in_or_are_refused()
    {
        const int r = 16, c = 32;   // the int8 store's per-output layout, as StreamingInt8NoUpcastTests uses
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8-inplace-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 1024L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        try
        {
            var w = Filled(new[] { r, c }, 2, v => v);
            w.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(w);
            WeightRegistry.Materialize(w);
            Assert.NotNull(w.StreamingInt8);
            Assert.Equal(0, w.DataVector.Length);
            var cpu = new CpuEngine();
            var strided = new Tensor<float>(new[] { 1 });
            cpu.TensorSetInPlace(strided, w, 0, new[] { r, c }, new[] { c, 1 });
            Assert.Equal(w.ToArray(), strided.ToArray());
            Assert.Throws<InvalidOperationException>(() => cpu.TensorAsStridedInPlace(w, new[] { c, r }, new[] { 1, c }));
        }
        finally { Cleanup(dir); }
    }

    /// <summary>
    /// A host copy that is resident when a forward reaches its weight on the GPU stays the owner budget's most recent
    /// use, as it is after Materialize: the next copy made resident sheds an older owner, not the one the forward is
    /// using. The forward used to leave it at the budget's oldest place.
    /// </summary>
    [SkippableFact]
    public void Gpu_forward_over_a_resident_bf16_weight_keeps_it_most_recent_in_the_owner_budget()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        if (gpu.GetBackend() is not IBf16WidenBackend { CanWidenBf16: true }) { gpu.Dispose(); Skip.If(true, "The backend keeps no bf16 weight copies"); return; }
        const int k = 64, n = 64;   // 16 KiB of fp32 per weight
        long oneWeight = 4L * k * n;
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-bf16-owners-{Guid.NewGuid():N}");
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 4 * oneWeight,   // the owner budget (half) holds two fp32 copies
            StreamingStoreDtype = StreamingStoreDtype.Bf16,
        });
        WeightRegistry.SetStreamingExecutionTraining(false);
        try
        {
            var a = Stream(Filled(new[] { k, n }, 1, v => v));
            var b = Stream(Filled(new[] { k, n }, 2, v => v));
            var c = Stream(Filled(new[] { k, n }, 3, v => v));
            WeightRegistry.ReleaseToPool(a); WeightRegistry.ReleaseToPool(b); WeightRegistry.ReleaseToPool(c);
            WeightRegistry.Materialize(a);
            WeightRegistry.Materialize(b);
            WeightRegistry.MaterializeForForward(a, gpu);   // a is in use again: the most recent
            WeightRegistry.Materialize(c);                  // over budget: the oldest, b, is shed
            Assert.True(a.DataVector.Length == a.Length, "the owner the forward used was shed");
            Assert.True(b.DataVector.Length != b.Length, "the oldest owner was kept");
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

    /// <summary>
    /// Dropping a streamed weight releases a device copy cached under its array only when nothing else can still read
    /// the array. Another tensor over the same data keeps it: freeing its device copy then would be a use-after-free.
    /// </summary>
    [SkippableFact]
    public void A_streaming_drop_keeps_the_device_copy_of_an_array_another_tensor_still_reads()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        const int In = 64, Out = 32;
        var dir = Configure();
        try
        {
            var x = Filled(new[] { 1, In }, 1, v => v);
            var shared = Filled(new[] { In, Out }, 3, v => v);
            var peer = new Tensor<float>(new[] { In, Out }, shared.DataVector);   // a second storage over the same data
            Skip.IfNot(ReferenceEquals(peer.DataVector, shared.DataVector), "The constructor copied the vector.");
            var expected = gpu.FusedLinear(x, peer, null, FusedActivationType.None).ToArray();
            int cached = gpu.CachedTensorCount;
            Stream(shared);
            Assert.Equal(cached, gpu.CachedTensorCount);
            Assert.Equal(expected, gpu.FusedLinear(x, peer, null, FusedActivationType.None).ToArray());
        }
        finally { gpu.Dispose(); Cleanup(dir); }
    }

    /// <summary>
    /// A recycled array's device copy is released from whatever thread recycled it, with no GPU op lease held there.
    /// The release used to borrow the backend through the leased path, which throws under strict leases after the
    /// cache entry was already removed, so the buffer was never freed and the exception escaped into the recycler.
    /// </summary>
    [SkippableFact]
    public void Recycling_an_array_releases_its_device_copy_without_a_lease()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        try
        {
            var x = Filled(new[] { 1, 64 }, 1, v => v);
            var w = Filled(new[] { 64, 32 }, 2, v => v);
            gpu.FusedLinear(x, w, null, FusedActivationType.None);
            int cached = gpu.CachedTensorCount;
            var array = w.GetBackingArrayForCacheLookupUnsafe();
            Assert.NotNull(array);
            gpu.StrictGpuLeases = true;
            AiDotNet.Tensors.Helpers.PooledArrayRecycling.Notify(array!);
            Assert.Equal(cached - 1, gpu.CachedTensorCount);
        }
        finally { gpu.StrictGpuLeases = false; gpu.Dispose(); }
    }
}
