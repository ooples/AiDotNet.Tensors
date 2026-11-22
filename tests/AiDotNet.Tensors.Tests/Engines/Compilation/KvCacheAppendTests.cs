using System;
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

    private static void RunEager(IEngine engine, bool hostOverflowThrows)
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
            var extra = new Tensor<float>([Batch, Width], new Vector<float>(Row(Capacity)));
            if (hostOverflowThrows)
                Assert.Throws<InvalidOperationException>(() => engine.KvCacheAppend(cache, position, extra));
            else
            {
                // A device kernel past capacity writes nothing, keeps the position and masks nothing.
                var mask = engine.KvCacheAppend(cache, position, extra);
                Assert.All(mask.ToArray(), m => Assert.Equal(0f, m));
                CheckCache(cache.ToArray(), Capacity - 1);
                Assert.Equal(Capacity, position.ToArray()[0]);
            }
        }
    }

    [Fact]
    public void The_cpu_append_writes_the_row_advances_the_position_and_masks_later_slots()
        => RunEager(new CpuEngine(), hostOverflowThrows: true);

    [SkippableFact]
    public void The_gpu_append_writes_the_row_advances_the_position_and_masks_later_slots()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            bool isCuda = gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend;
            gpu.StrictGpuLeases = true;
            RunEager(gpu, hostOverflowThrows: !isCuda);
        }
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

    [SkippableTheory]
    [InlineData(-1f)]
    [InlineData(1.5f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Invalid_positions_do_not_corrupt_the_cache_or_advance_the_position(float badPosition)
    {
        var cpu = new CpuEngine();
        var cpuCache = new Tensor<float>([Batch, Capacity, Width]);
        var cpuPosition = new Tensor<float>(new[] { badPosition }, new[] { 1 });
        var value = new Tensor<float>([Batch, Width], new Vector<float>(Row(0)));
        Assert.Throws<InvalidOperationException>(() => cpu.KvCacheAppend(cpuCache, cpuPosition, value));

        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            bool isCuda = gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend;
            gpu.StrictGpuLeases = true;
            var sentinel = Enumerable.Repeat(7f, Batch * Capacity * Width).ToArray();
            var cache = new Tensor<float>([Batch, Capacity, Width], new Vector<float>(sentinel));
            var position = new Tensor<float>(new[] { badPosition }, new[] { 1 });
            if (isCuda)
            {
                var mask = gpu.KvCacheAppend(cache, position, value);
                Assert.All(mask.ToArray(), m => Assert.True(float.IsNegativeInfinity(m)));
                Assert.All(cache.ToArray(), c => Assert.Equal(7f, c));
                float observedPosition = position.ToArray()[0];
                if (float.IsNaN(badPosition))
                {
                    Assert.True(float.IsNaN(observedPosition));
                }
                else
                {
                    Assert.Equal(badPosition, observedPosition);
                }
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => gpu.KvCacheAppend(cache, position, value));
            }
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
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
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
                    if (capture && gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cuda)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                        Assert.Equal(0, cuda.LastCaptureMemcpyNodeCount);
                    }
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }
}
