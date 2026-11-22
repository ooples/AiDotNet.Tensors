using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// IEngine.KvCacheAppend: a fixed-capacity decode cache written at a position held in a tensor, with the position
/// incremented in place, eagerly and replayed by a compiled (and captured) plan with no host-side position.
/// </summary>
[Collection("DirectGpuSerial")]
public class KvCacheAppendTests
{
    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private const int Batch = 3, Capacity = 6, Width = 5;

    private static float[] Row(int t) => Enumerable.Range(0, Batch * Width).Select(i => 100f * t + i + 1).ToArray();

    // After appending positions 0..t: cache[b, s, j] = Row(s)[b * Width + j] for s <= t, zero after.
    private static void CheckCache(float[] cache, int t)
    {
        for (int b = 0; b < Batch; b++)
            for (int s = 0; s < Capacity; s++)
                for (int j = 0; j < Width; j++)
                    Assert.Equal(s <= t ? Row(s)[b * Width + j] : 0f, cache[(b * Capacity + s) * Width + j]);
    }

    private static void CheckMask(float[] mask, int t)
    {
        for (int s = 0; s < Capacity; s++)
            Assert.Equal(s <= t ? 0f : float.NegativeInfinity, mask[s]);
    }

    private static void RunEager(IEngine engine)
    {
        var cache = new Tensor<float>([Batch, Capacity, Width]);
        var position = new Tensor<float>([1]);
        using (new EngineScope(engine))
        {
            for (int t = 0; t < Capacity; t++)
            {
                var mask = engine.KvCacheAppend(cache, position, new Tensor<float>([Batch, Width], new Vector<float>(Row(t))));
                CheckMask(mask.ToArray(), t);
                CheckCache(cache.ToArray(), t);
                Assert.Equal(t + 1, position.ToArray()[0]);
            }
            // Every backend refuses a full cache, leaving the cache and the position as they were.
            var extra = new Tensor<float>([Batch, Width], new Vector<float>(Row(Capacity)));
            var full = Assert.Throws<InvalidOperationException>(() => engine.KvCacheAppend(cache, position, extra));
            Assert.Contains("full", full.Message);
            CheckCache(cache.ToArray(), Capacity - 1);
            Assert.Equal(Capacity, position.ToArray()[0]);
        }
    }

    // A position that is not a non-negative integer is refused by every backend before anything is written: slot 0
    // keeps its row (a NaN converted to 0 would overwrite it), nothing before the cache is written (a negative
    // position), and a fractional position is not truncated onto a slot.
    private static void RunBadPositions(IEngine engine)
    {
        using (new EngineScope(engine))
        {
            foreach (float bad in new[] { -1f, -0.5f, float.NaN, 1.5f, float.PositiveInfinity, float.NegativeInfinity, 1e9f })
            {
                var cache = new Tensor<float>([Batch, Capacity, Width]);
                var position = new Tensor<float>([1]);
                engine.KvCacheAppend(cache, position, new Tensor<float>([Batch, Width], new Vector<float>(Row(0))));
                SetPosition(position, bad);
                var extra = new Tensor<float>([Batch, Width], new Vector<float>(Row(1)));
                var ex = Assert.Throws<InvalidOperationException>(() => engine.KvCacheAppend(cache, position, extra));
                Assert.Contains(bad == 1e9f ? "full" : "non-negative integer", ex.Message);
                CheckCache(cache.ToArray(), 0);
                float after = position.ToArray()[0];
                Assert.True(after.Equals(bad), $"position {bad} became {after}");
                // The next append from a valid position still works.
                SetPosition(position, 1f);
                var mask = engine.KvCacheAppend(cache, position, extra);
                CheckMask(mask.ToArray(), 1);
                CheckCache(cache.ToArray(), 1);
                Assert.Equal(2f, position.ToArray()[0]);
            }
        }
    }

    private static void SetPosition(Tensor<float> position, float value)
    {
        position[0] = value;
        position.IncrementVersion();
    }

    // StrictGpuLeases asserts every backend lookup holds a GPU op lease, the test's own lookups included.
    private static AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend? BackendOf(DirectGpuTensorEngine gpu)
    {
        using var op = gpu.EnterGpuOp();
        return gpu.GetBackend();
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (gpu is not null && !gpu.IsGpuAvailable) { gpu.Dispose(); gpu = null; }
        if (gpu is not null) gpu.StrictGpuLeases = true;
        return gpu;
    }

    [Fact]
    public void The_cpu_append_writes_the_row_advances_the_position_and_masks_later_slots()
        => RunEager(new CpuEngine());

    [SkippableFact]
    public void The_gpu_append_writes_the_row_advances_the_position_and_masks_later_slots()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu) RunEager(gpu!);
    }

    [Fact]
    public void The_cpu_append_refuses_a_negative_nan_fractional_or_huge_position()
        => RunBadPositions(new CpuEngine());

    [SkippableFact]
    public void The_gpu_append_refuses_a_negative_nan_fractional_or_huge_position()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu) RunBadPositions(gpu!);
    }

    // The position is counted in T: a type that cannot hold every position up to capacity exactly, or the mask's
    // negative infinity, is refused on every engine rather than rounding positions onto neighbouring slots.
    [Fact]
    public void An_element_type_that_cannot_count_every_position_is_refused()
    {
        var engines = new List<IEngine> { new CpuEngine() };
        var gpu = TryGpu();
        if (gpu is not null) engines.Add(gpu);
        try
        {
            foreach (var engine in engines)
            {
                using (new EngineScope(engine))
                {
                    Assert.Throws<NotSupportedException>(() => engine.KvCacheAppend(
                        new Tensor<Half>([1, 2049, 1]), new Tensor<Half>([1]), new Tensor<Half>([1, 1])));
                    var ok = engine.KvCacheAppend(new Tensor<Half>([1, 2048, 1]), new Tensor<Half>([1]), new Tensor<Half>([1, 1]));
                    Assert.Equal(0.0, (double)ok.ToArray()[0]);
                    Assert.True(Half.IsNegativeInfinity(ok.ToArray()[1]));
                    Assert.Throws<NotSupportedException>(() => engine.KvCacheAppend(
                        new Tensor<float>([1, (1 << 24) + 1, 1]), new Tensor<float>([1]), new Tensor<float>([1, 1])));
                    Assert.Throws<NotSupportedException>(() => engine.KvCacheAppend(
                        new Tensor<int>([1, 4, 1]), new Tensor<int>([1]), new Tensor<int>([1, 1])));
                }
            }
        }
        finally { gpu?.Dispose(); }
    }

    [Fact]
    public void Strict_gpu_leases_cover_kv_cache_append_and_state_residency_helpers()
    {
        var backend = DirectGpu.MockDirectGpuBackend.Create(new DirectGpu.MockBackendState());
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct) { StrictGpuLeases = true };

        var cache = new Tensor<float>([Batch, Capacity, Width]);
        var position = new Tensor<float>([1]);
        var row0 = new Tensor<float>([Batch, Width], new Vector<float>(Row(0)));
        var mask = engine.KvCacheAppend(cache, position, row0);
        CheckMask(mask.ToArray(), 0);
        CheckCache(cache.ToArray(), 0);
        Assert.Equal(1f, position.ToArray()[0]);

        var state0 = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var state1 = new Tensor<float>(new[] { 3f, 4f }, new[] { 2 });
        var leaf = new Tensor<float>(new[] { 5f, 6f }, new[] { 2 });
        var next0 = new Tensor<float>(new[] { 10f, 20f }, new[] { 2 });
        var next1 = new Tensor<float>(new[] { 30f, 40f }, new[] { 2 });

        var pinnedState = engine.PinInputResident(state0);
        var pinnedLeaf = engine.PinLeafResident(leaf);
        Assert.NotNull(pinnedState);
        Assert.NotNull(pinnedLeaf);

        engine.AssignStatesInPlace(new[] { state0, state1 }, new[] { next0, next1 });
        Assert.Equal(new[] { 10f, 20f }, state0.ToArray());
        Assert.Equal(new[] { 30f, 40f }, state1.ToArray());

        if (pinnedState is not null)
        {
            engine.RearmStateAfterReplay(state0, pinnedState);
        }
    }

    // A compiled plan appending each step's input: the position advances on the device once per execution (the capture
    // call included), the mask follows it, and the captured graph copies nothing.
    // With realizeInTrace an RMSNorm downstream of the append computes its graph-mode value eagerly, which runs the
    // append during the trace; compiling must put the cache and position back, so the first execution writes slot 0.
    [SkippableTheory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    [InlineData(false, true)]
    public void A_compiled_plan_appends_once_per_execution_from_a_device_position(bool capture, bool realizeInTrace)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU backend did not resolve.");
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                var cache = new Tensor<float>([Batch, Capacity, Width]);
                var position = new Tensor<float>([1]);
                var x = new Tensor<float>([Batch, Width]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var weights = gpu!.Softmax(gpu.KvCacheAppend(cache, position, x), -1);   // uniform over the filled slots
                    if (realizeInTrace)
                        weights = gpu.TensorAdd(weights, gpu.TensorMultiplyScalar(gpu.RMSNorm(gpu.Reshape(weights, [1, Capacity]),
                            new Tensor<float>([Capacity], new Vector<float>(Enumerable.Repeat(1f, Capacity).ToArray())), 1e-6, out _), 0f));
                    plan = scope.CompileInference(weights, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
                }
                using (plan)
                {
                    for (int t = 0; t < Capacity; t++)
                    {
                        plan.SetInputs(new[] { new Tensor<float>([Batch, Width], new Vector<float>(Row(t))) });
                        var w = plan.Execute().ToArray();
                        for (int s = 0; s < Capacity; s++)
                            Assert.True(Math.Abs(w[s] - (s <= t ? 1f / (t + 1) : 0f)) < 1e-6, $"t{t} weight[{s}] = {w[s]}");
                        CheckCache(cache.ToArray(), t);
                        Assert.Equal(t + 1, position.ToArray()[0]);
                    }
                    if (capture && BackendOf(gpu!) is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cuda)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                        Assert.Equal(0, cuda.LastCaptureMemcpyNodeCount);
                    }
                    // One execution past capacity: the plan, captured or not, throws as the eager append does, and the
                    // cache and the position are left as they were.
                    plan.SetInputs(new[] { new Tensor<float>([Batch, Width], new Vector<float>(Row(Capacity))) });
                    var full = Assert.Throws<InvalidOperationException>(() => plan.Execute());
                    Assert.Contains("full", full.Message);
                    CheckCache(cache.ToArray(), Capacity - 1);
                    Assert.Equal(Capacity, position.ToArray()[0]);
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }
}
