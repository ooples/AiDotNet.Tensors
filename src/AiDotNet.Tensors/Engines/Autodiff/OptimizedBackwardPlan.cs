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

    /// <summary>
    /// Optional reference to the owning tape's RetainGrad set. Same parity rule
    /// as <see cref="CompiledBackwardGraph{T}"/> — when a user marks a tensor
    /// with <see cref="GradientTape{T}.RetainGrad"/>, its <c>.Grad</c> must
    /// survive the cleanup phase below.
    /// </summary>
    private readonly HashSet<Tensor<T>>? _retainGrad;

    internal OptimizedBackwardPlan(
        TapeEntryArena<T> entries,
        int[] reachableIndices,
        Tensor<T> loss,
        Tensor<T>[]? sources,
        IEngine engine,
        BackwardAnalysis analysis,
        HashSet<Tensor<T>>? retainGrad = null)
    {
        _entries = entries;
        _reachableIndices = reachableIndices;
        _loss = loss;
        _sources = sources;
        _engine = engine;
        _retainGrad = retainGrad;
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

            // Parity with CompiledBackwardGraph.Execute's finally block:
            // a consumer-side cache that retains a forward intermediate
            // (e.g. a layer's `_lastInput`) would otherwise pin one full
            // backward's worth of gradient tensors across successive
            // Execute calls when this optimized path is taken (the default
            // when TensorCodecOptions.EnableAlgebraicBackward is on).
            // Same RetainGrad / sourceSet preservation rule used by the
            // non-optimized path so behavior is consistent.
            HashSet<Tensor<T>>? sourceSet = null;
            if (_sources is not null)
            {
                sourceSet = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
                foreach (var s in _sources) sourceSet.Add(s);
            }

            // Pool-return gating: createGraph=true keeps the recorded backward
            // ops alive past this cleanup; skipping pool-return on that path
            // matches ComputeGradientsViaGraphCore's behaviour.
            bool canPoolNodes = !DifferentiableOps._isBackwardCreateGraph;

            // Issue #283 parity: pre-collect intermediates owned by this plan.
            // Used by CleanupInput below to decide whether an input is an
            // intermediate (clear GradFn/Grad) or a leaf (preserve .Grad).
            var intermediates = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
            foreach (int j in _reachableIndices)
            {
                var o = _entries[j].Output;
                if (o is not null) intermediates.Add(o);
            }

            foreach (int i in _reachableIndices)
            {
                ref var e = ref _entries[i];
                e.Output._gradIndex = -1;
                e.Output._pinnedByTape = false;
                if (e.Input0 != null)
                {
                    e.Input0._gradIndex = -1;
                    e.Input0._pinnedByTape = false;
                }
                if (e.InputCount >= 2 && e.Input1 != null)
                {
                    e.Input1._gradIndex = -1;
                    e.Input1._pinnedByTape = false;
                }
                if (e.InputCount >= 3 && e.Input2 != null)
                {
                    e.Input2._gradIndex = -1;
                    e.Input2._pinnedByTape = false;
                }
                if (e.InputsOverflow != null)
                {
                    foreach (var inp in e.InputsOverflow)
                    {
                        inp._gradIndex = -1;
                        inp._pinnedByTape = false;
                    }
                }

                // e.Output is always a graph intermediate; inputs may be
                // leaves (parameters) so we don't touch their .Grad here.
                // GradFn cleanup parity with the recording path — see the
                // matching comment in CompiledBackwardGraph.Execute.
                bool keepGrad =
                    (sourceSet?.Contains(e.Output) == true)
                    || (_retainGrad is not null && _retainGrad.Contains(e.Output));
                // Issue #283: capture the GradNode before nulling so it can
                // be pool-returned. Parity with ComputeGradientsViaGraphCore's
                // GradNodePool<T>.Return call.
                var node = e.Output.GradFn;
                e.Output.GradFn = null;
                if (!keepGrad) e.Output.Grad = null;
                if (canPoolNodes && node is not null)
                {
                    GradNodePool<T>.Return(node);
                }

                // Issue #283 parity: clear input-side back-pointers too.
                CleanupInput(e.Input0, intermediates, sourceSet, canPoolNodes);
                if (e.InputCount >= 2) CleanupInput(e.Input1, intermediates, sourceSet, canPoolNodes);
                if (e.InputCount >= 3) CleanupInput(e.Input2, intermediates, sourceSet, canPoolNodes);
                if (e.InputsOverflow is not null)
                {
                    foreach (var inp in e.InputsOverflow)
                        CleanupInput(inp, intermediates, sourceSet, canPoolNodes);
                }
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

    private void CleanupInput(
        Tensor<T>? t,
        HashSet<Tensor<T>> intermediates,
        HashSet<Tensor<T>>? sourceSet,
        bool canPoolNodes)
    {
        if (t is null) return;
        if (!intermediates.Contains(t)) return;
        var inNode = t.GradFn;
        t.GradFn = null;
        if (canPoolNodes && inNode is not null) GradNodePool<T>.Return(inNode);
        bool keep = (sourceSet?.Contains(t) == true)
                    || (_retainGrad is not null && _retainGrad.Contains(t));
        if (!keep) t.Grad = null;
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
        IEngine engine,
        HashSet<Tensor<T>>? retainGrad = null)
    {
        var analysis = SymbolicBackwardGraphBuilder.Analyze(entries, reachableIndices);

        if (!analysis.CanBenefit)
            return null;

        return new OptimizedBackwardPlan<T>(entries, reachableIndices, loss, sources, engine, analysis, retainGrad);
    }
}
