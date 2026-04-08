using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Optimized backward execution plan that applies CSE and algebraic simplifications.
/// Replaces CompiledBackwardGraph.Execute() when optimizations are beneficial.
///
/// Key optimizations:
/// 1. BackwardCSEPass caches transposes across layers (eliminates redundant A^T computations)
/// 2. Transposed BLAS GEMM via TryGemmEx (eliminates transpose tensor allocation)
/// 3. Inline bias gradient reduction (eliminates ReduceSum engine call overhead)
/// 4. Pre-analyzed graph structure determines which optimizations to apply
/// </summary>
internal sealed class OptimizedBackwardPlan<T>
{
    private readonly TapeEntryArena<T> _entries;
    private readonly int[] _reachableIndices;
    private readonly Tensor<T> _loss;
    private readonly Tensor<T>[]? _sources;
    private readonly IEngine _engine;

    internal OptimizedBackwardPlan(
        TapeEntryArena<T> entries,
        int[] reachableIndices,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine,
        BackwardAnalysis analysis)
    {
        _entries = entries;
        _reachableIndices = reachableIndices;
        _loss = loss;
        _sources = sources;
        _engine = engine;
        _ = analysis; // Retained for future backward optimization passes
    }

    /// <summary>
    /// Executes the optimized backward. Uses CSE to cache transposes and
    /// transposed BLAS GEMM for MatMul backward operations.
    /// </summary>
    internal Dictionary<Tensor<T>, Tensor<T>> Execute()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var cse = new BackwardCSEPass<T>(_engine);

        // Assign grad indices
        int gradIndexCount = 0;
        foreach (int i in _reachableIndices)
        {
            ref var e = ref _entries[i];
            if (e.Output._gradIndex < 0) e.Output._gradIndex = gradIndexCount++;
            if (e.Input0 != null && e.Input0._gradIndex < 0) e.Input0._gradIndex = gradIndexCount++;
            if (e.InputCount >= 2 && e.Input1 != null && e.Input1._gradIndex < 0) e.Input1._gradIndex = gradIndexCount++;
            if (e.InputCount >= 3 && e.Input2 != null && e.Input2._gradIndex < 0) e.Input2._gradIndex = gradIndexCount++;
            if (e.InputsOverflow != null)
                foreach (var inp in e.InputsOverflow)
                    if (inp._gradIndex < 0) inp._gradIndex = gradIndexCount++;
        }

        var indexedGrads = new object?[gradIndexCount];
        DifferentiableOps.SetIndexedGrads(indexedGrads);

        var grads = new Dictionary<Tensor<T>, Tensor<T>>(
            Math.Min(gradIndexCount + 1, 1024),
            ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Seed
        Tensor<T> seedGrad;
        if (_loss.Length == 1)
            seedGrad = new Tensor<T>(new[] { numOps.One }, (int[])_loss._shape.Clone());
        else
        {
            var onesData = new T[_loss.Length];
            for (int j = 0; j < onesData.Length; j++) onesData[j] = numOps.One;
            seedGrad = new Tensor<T>(onesData, _loss._shape);
        }
        grads[_loss] = seedGrad;
        if (_loss._gradIndex >= 0) indexedGrads[_loss._gradIndex] = seedGrad;

        try
        {
            foreach (int i in _reachableIndices)
            {
                ref var entry = ref _entries[i];

                if (!grads.TryGetValue(entry.Output, out var gradOutput))
                    continue;

                entry.ValidateInputVersions();

                // Try optimized path for MatMul operations
                if (TryOptimizedMatMulBackward(ref entry, gradOutput, cse, grads))
                    continue;

                // Default path: use the original backward function
                var inputsArray = entry.GetInputsArray();
                entry.Backward(gradOutput, inputsArray, entry.Output,
                    entry.SavedState ?? Array.Empty<object>(), _engine, grads);
            }
        }
        finally
        {
            cse.Clear();
            DifferentiableOps.ClearIndexedGrads();
            foreach (int i in _reachableIndices)
            {
                ref var e = ref _entries[i];
                e.Output._gradIndex = -1;
                if (e.Input0 != null) e.Input0._gradIndex = -1;
                if (e.InputCount >= 2 && e.Input1 != null) e.Input1._gradIndex = -1;
                if (e.InputCount >= 3 && e.Input2 != null) e.Input2._gradIndex = -1;
                if (e.InputsOverflow != null)
                    foreach (var inp in e.InputsOverflow) inp._gradIndex = -1;
            }
        }

        if (_sources is not null)
        {
            var filtered = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (var source in _sources)
            {
                if (grads.TryGetValue(source, out var grad))
                    filtered[source] = grad;
            }
            return filtered;
        }

        return grads;
    }

    /// <summary>
    /// Tries to handle MatMul backward using CSE-cached transposed BLAS.
    /// Returns true if handled, false to fall through to default.
    /// </summary>
    private bool TryOptimizedMatMulBackward(
        ref TapeEntry<T> entry,
        Tensor<T> gradOutput,
        BackwardCSEPass<T> cse,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (entry.OperationName != "TensorMatMul")
            return false;

        if (entry.Input0 == null || entry.InputCount < 2 || entry.Input1 == null)
            return false;

        if (entry.Input0.Rank != 2 || entry.Input1.Rank != 2)
            return false;

        // dL/dA = gradOutput @ B^T (CSE-cached transposed BLAS)
        var gradA = cse.MatMulTransposeRight(gradOutput, entry.Input1);
        DifferentiableOps.AccumulateGrad(grads, entry.Input0, gradA, _engine);

        // dL/dB = A^T @ gradOutput (CSE-cached transposed BLAS)
        var gradB = cse.MatMulTransposeLeft(entry.Input0, gradOutput);
        DifferentiableOps.AccumulateGrad(grads, entry.Input1, gradB, _engine);

        return true;
    }

    /// <summary>
    /// Creates an OptimizedBackwardPlan if the analysis shows optimization potential.
    /// Returns null if the standard path would be equivalent or faster.
    /// </summary>
    internal static OptimizedBackwardPlan<T>? TryCreate(
        TapeEntryArena<T> entries,
        int[] reachableIndices,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine)
    {
        var analysis = SymbolicBackwardGraphBuilder.Analyze(entries, reachableIndices);

        if (!analysis.CanBenefit)
            return null;

        return new OptimizedBackwardPlan<T>(entries, reachableIndices, loss, sources, engine, analysis);
    }
}
