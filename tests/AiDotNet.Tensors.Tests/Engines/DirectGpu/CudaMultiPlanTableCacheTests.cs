using System.Collections;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CudaMultiPlanTableCacheTests
{
    private static object Field(object owner, string name) => owner.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.GetValue(owner)!;

    private sealed class Buffers : IDisposable
    {
        public readonly int[] Sizes;
        public readonly IGpuBuffer[] Parameters, Gradients, First, Second, ReferenceParameters, ReferenceFirst, ReferenceSecond;
        public readonly double ExpectedSum;
        private readonly IDirectGpuBackend _backend;

        public Buffers(IDirectGpuBackend backend, int salt, int count, int firstSize = 5)
        {
            _backend = backend;
            Sizes = Enumerable.Range(0, count).Select(i => firstSize + 13 * i).ToArray();
            IGpuBuffer[] Make(float offset) => Sizes.Select(n => backend.AllocateBuffer(
                Enumerable.Range(0, n).Select(i => offset + i * .002f).ToArray())).ToArray();
            Parameters = Make(salt * .1f); Gradients = Make((salt % 2 == 0 ? 1f : -1f) * .25f);
            First = Make(0f); Second = Make(.125f);
            ReferenceParameters = Make(salt * .1f); ReferenceFirst = Make(0f); ReferenceSecond = Make(.125f);
            ExpectedSum = Gradients.Sum(g => backend.DownloadBuffer(g).Sum(v => (double)v * v));
        }

        public void Step(int step)
        {
            ((IMultiTensorGpuOptimizerBackend)_backend).AdamWMultiTensorUpdate(Parameters, Gradients, First, Second,
                Sizes, .01f, .9f, .999f, 1e-8f, .02f, step);
            for (int i = 0; i < Sizes.Length; i++)
                _backend.AdamWUpdate(ReferenceParameters[i], Gradients[i], ReferenceFirst[i], ReferenceSecond[i],
                    .01f, .9f, .999f, 1e-8f, .02f, step, Sizes[i]);
        }

        public void Check()
        {
            for (int t = 0; t < Sizes.Length; t++)
                foreach (var pair in new[] { (Parameters[t], ReferenceParameters[t]), (First[t], ReferenceFirst[t]), (Second[t], ReferenceSecond[t]) })
                {
                    var actual = _backend.DownloadBuffer(pair.Item1);
                    var expected = _backend.DownloadBuffer(pair.Item2);
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-6f * (1f + Math.Abs(expected[i])));
                }
        }

        public void Dispose()
        {
            foreach (var group in new[] { Parameters, Gradients, First, Second, ReferenceParameters, ReferenceFirst, ReferenceSecond })
                foreach (var buffer in group) buffer.Dispose();
        }
    }

    [SkippableFact]
    public void Alternating_plans_reuse_their_own_tables_and_preserve_optimizer_values()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "Actual CUDA backend required.");
        var backend = engine.GetBackend()!;
        var multi = (IMultiTensorKernels)backend;
        using var sum = backend.AllocateBuffer(new float[AiDotNet.Tensors.Engines.Gpu.DeterministicGradientNorm.RecordWords]);
        using var a = new Buffers(backend, 1, 2);
        using var b = new Buffers(backend, 2, 2); // Same sizes, different operand addresses.
        using var c = new Buffers(backend, 3, 3);
        var sets = new[] { a, b, c };
        var optimizerTables = new object[sets.Length]; var gradientTables = new object[sets.Length];
        for (int step = 1; step <= 6; step++)
            for (int i = 0; i < sets.Length; i++)
            {
                var set = sets[i]; set.Step(step);
                multi.MultiTensorSumOfSquares(set.Gradients, set.Sizes, sum);
                Assert.True(Math.Abs(multi.ReadSumOfSquares(sum) - set.ExpectedSum) <= 1e-12 * Math.Max(1.0, set.ExpectedSum));
                if (step == 1)
                {
                    optimizerTables[i] = Field(backend, "_adamTable");
                    gradientTables[i] = Field(backend, "_gradientTable");
                }
                else
                {
                    Assert.Same(optimizerTables[i], Field(backend, "_adamTable"));
                    Assert.Same(gradientTables[i], Field(backend, "_gradientTable"));
                }
                set.Check();
            }
    }

    // The bound, read from the backend so the test follows it.
    private static int TableCapacity() => (int)(typeof(CudaBackend)
        .GetField("MultiTensorTableCacheCapacity", BindingFlags.Static | BindingFlags.NonPublic)?.GetRawConstantValue()
        ?? throw new InvalidOperationException("CudaBackend.MultiTensorTableCacheCapacity was renamed."));

    [SkippableFact]
    public void Descriptor_caches_are_bounded_and_dispose_evicted_tables()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "Actual CUDA backend required.");
        var backend = engine.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        int capacity = TableCapacity();
        using var sum = backend.AllocateBuffer(new float[AiDotNet.Tensors.Engines.Gpu.DeterministicGradientNorm.RecordWords]);
        // Distinct sizes per plan: a plan whose sizes match a retained table refreshes that table's pointers in place
        // instead of building one, so only plans of different shapes grow the cache. Two more than the bound.
        var sets = Enumerable.Range(0, capacity + 2).Select(i => new Buffers(backend, i + 1, 2, firstSize: 5 + i)).ToArray();
        try
        {
            IGpuBuffer? firstAdam = null, firstGradient = null;
            for (int i = 0; i < sets.Length; i++)
            {
                sets[i].Step(1);
                ((IMultiTensorKernels)backend).MultiTensorSumOfSquares(sets[i].Gradients, sets[i].Sizes, sum);
                if (i == 0)
                {
                    firstAdam = (IGpuBuffer)Field(Field(backend, "_adamTable"), "Pointers");
                    firstGradient = (IGpuBuffer)Field(Field(backend, "_gradientTable"), "Pointers");
                }
                Assert.Equal(Math.Min(i + 1, capacity), ((ICollection)Field(backend, "_retainedAdamTables")).Count);
                Assert.Equal(Math.Min(i + 1, capacity), ((ICollection)Field(backend, "_retainedGradientTables")).Count);
                sets[i].Check();
            }
            // The oldest table was evicted, and eviction disposed it.
            Assert.NotNull(firstAdam);
            Assert.NotNull(firstGradient);
            Assert.Equal(IntPtr.Zero, firstAdam.Handle);
            Assert.Equal(IntPtr.Zero, firstGradient.Handle);
        }
        finally { foreach (var set in sets) set.Dispose(); }
    }

    [SkippableFact]
    public void Plans_of_the_same_sizes_share_one_table_refreshed_in_place()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "Actual CUDA backend required.");
        var backend = engine.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var multi = (IMultiTensorKernels)backend;
        using var sum = backend.AllocateBuffer(new float[AiDotNet.Tensors.Engines.Gpu.DeterministicGradientNorm.RecordWords]);
        var sets = Enumerable.Range(0, 10).Select(i => new Buffers(backend, i + 1, 2)).ToArray();
        try
        {
            object? adamTable = null, gradientTable = null;
            for (int i = 0; i < sets.Length; i++)
            {
                sets[i].Step(1);
                multi.MultiTensorSumOfSquares(sets[i].Gradients, sets[i].Sizes, sum);
                // The refreshed table points at THIS plan's operands: the sum and the update are this plan's.
                Assert.True(Math.Abs(multi.ReadSumOfSquares(sum) - sets[i].ExpectedSum) <= 1e-12 * Math.Max(1.0, sets[i].ExpectedSum));
                sets[i].Check();
                if (i == 0)
                {
                    adamTable = Field(backend, "_adamTable");
                    gradientTable = Field(backend, "_gradientTable");
                }
                Assert.Same(adamTable, Field(backend, "_adamTable"));
                Assert.Same(gradientTable, Field(backend, "_gradientTable"));
                Assert.Single((ICollection)Field(backend, "_retainedAdamTables"));
                Assert.Single((ICollection)Field(backend, "_retainedGradientTables"));
            }
        }
        finally { foreach (var set in sets) set.Dispose(); }
    }
}
