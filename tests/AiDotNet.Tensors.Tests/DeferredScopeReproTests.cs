#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests;

/// <summary>
/// Localizes the Phase-2 transparent-fusion gap: does a SINGLE op materialize its result when run inside a
/// deferred scope (capture -> optimize -> replay)? If a trivial matmul comes back all-zero, the deferred
/// path is broken at the basic output-materialization level (not decoder-specific). Skips without a GPU.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeferredScopeReproTests
{
    [SkippableFact]
    public void DeferredScope_SingleMatMul_MaterializesSameAsEager()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

            // Eager reference on the same GPU engine.
            var eager = gpu.TensorMatMul(a, b).Contiguous().AsSpan().ToArray();

            var scope = gpu.BeginDeferredScope();
            if (scope is null) { Skip.If(true, "Backend has no deferred execution"); return; }

            float[] deferred;
            using (scope)
            {
                var r = gpu.TensorMatMul(a, b); // records into the graph
                scope.Execute();                // optimize + replay
                deferred = r.Contiguous().AsSpan().ToArray();
            }

            bool allZero = true;
            for (int i = 0; i < deferred.Length; i++) if (Math.Abs(deferred[i]) > 1e-9) { allZero = false; break; }
            Assert.False(allZero, "Deferred replay produced all-zero output — output buffer not materialized.");

            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - deferred[i]) < 1e-3f,
                    $"[{i}] eager {eager[i]} vs deferred {deferred[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}

#endif
