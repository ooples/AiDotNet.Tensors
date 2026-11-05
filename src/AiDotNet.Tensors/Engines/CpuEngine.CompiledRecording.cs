using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Records an op as one node of a compiled training (or compatibility) graph, for ops whose eager path records a
    /// tape edge but have no graph-native node of their own.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without a node, such an op ran once at trace time on the trace's values and was frozen into the plan as a
    /// constant: every input feeding it then got a zero gradient from every compiled step, while the eager tape was
    /// correct (#1680; <c>DifferentiableOpsGradCheckSweep.EveryDifferentiableOp_CompiledTrainingGradientMatchesTheTape</c>
    /// found 62 such ops).
    /// </para>
    /// <para>
    /// The op runs once here, with graph mode suspended, to give the node its exact output shape (a copy, so a view
    /// result can never alias an input). Each replay re-runs <paramref name="forward"/> into that output, and the
    /// backward re-runs it under a tape (<see cref="BackwardFunctions{T}.ReplayUnderTape"/>), so the compiled
    /// gradient is by construction the gradient the eager tape computes for the same op.
    /// </para>
    /// <para>
    /// Inference traces keep their own behaviour (an inference plan needs no backward), so this declines them.
    /// </para>
    /// </remarks>
    /// <param name="opName">The op's name, for the graph and diagnostics.</param>
    /// <param name="inputs">Every tensor operand of element type <typeparamref name="T"/>, differentiable or not; nulls are ignored.</param>
    /// <param name="forward">The op itself on a given engine, closing over all of its arguments.</param>
    /// <param name="result">The recorded node's output when this returns true.</param>
    internal bool TryRecordCompiledOp<T>(string opName, Tensor<T>?[] inputs, Func<IEngine, Tensor<T>> forward,
        out Tensor<T> result)
    {
        result = null!;
        if (!GraphMode.IsActive || GraphMode.IsInferenceTrace) return false;
        var scope = GraphMode.Current;
        if (scope is null || !scope.BindEngine(this)) return false;

        var operands = new List<Tensor<T>>(inputs.Length);
        foreach (var input in inputs)
            if (input is not null && !operands.Contains(input)) operands.Add(input);
        if (operands.Count == 0) return false;

        Tensor<T> traced;
        GraphMode.SetCurrent(null);
        try
        {
            using (new NoGradScope<T>())
                traced = forward(this).Clone();
        }
        finally
        {
            GraphMode.SetCurrent(scope);
        }

        result = scope.RecordMaterializedVariadic(LazyNodeType.Custom, opName, operands.ToArray(), traced,
            (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, CheckReplayShape(opName, forward(eng), output), output),
            BackwardFunctions<T>.ReplayUnderTape(forward));
        return true;
    }

    // A plan's buffers have the trace's shapes. An op whose output shape depends on data (MaskedSelect counts its
    // mask) is recorded with the trace's shape; a replay whose data changed that shape fails here, loudly.
    private static Tensor<T> CheckReplayShape<T>(string opName, Tensor<T> fresh, Tensor<T> output)
    {
        if (!fresh._shape.AsSpan().SequenceEqual(output._shape))
            throw new InvalidOperationException(
                $"{opName} produced shape [{string.Join(", ", fresh._shape)}] in a compiled replay where the trace " +
                $"produced [{string.Join(", ", output._shape)}]: its output shape depends on data that changed " +
                "between steps, which a compiled plan cannot follow. Run this op outside the compiled step.");
        return fresh;
    }

    /// <summary>
    /// <see cref="TryRecordCompiledOp{T}(string, Tensor{T}?[], Func{IEngine, Tensor{T}}, out Tensor{T})"/> for an op
    /// that also returns a non-differentiable auxiliary output through an <c>out</c> parameter (argmax indices).
    /// </summary>
    /// <remarks>
    /// The caller receives the trace's auxiliary output. Each replay recomputes it and <paramref name="refreshAux"/>
    /// copies the fresh value into that same object in place, so a caller holding it reads the current step's
    /// indices, as AiDotNet#1331 does for ScatterMean's counts.
    /// </remarks>
    internal bool TryRecordCompiledOp<T, TAux>(string opName, Tensor<T>?[] inputs,
        Func<IEngine, (Tensor<T> Result, TAux Aux)> forward, Action<IEngine, TAux, TAux> refreshAux,
        out Tensor<T> result, out TAux aux)
    {
        result = null!;
        aux = default!;
        if (!GraphMode.IsActive || GraphMode.IsInferenceTrace) return false;
        var scope = GraphMode.Current;
        if (scope is null || !scope.BindEngine(this)) return false;

        var operands = new List<Tensor<T>>(inputs.Length);
        foreach (var input in inputs)
            if (input is not null && !operands.Contains(input)) operands.Add(input);
        if (operands.Count == 0) return false;

        Tensor<T> traced;
        TAux tracedAux;
        GraphMode.SetCurrent(null);
        try
        {
            using (new NoGradScope<T>())
            {
                var (r, a) = forward(this);
                traced = r.Clone();
                tracedAux = a;
            }
        }
        finally
        {
            GraphMode.SetCurrent(scope);
        }

        aux = tracedAux;
        result = scope.RecordMaterializedVariadic(LazyNodeType.Custom, opName, operands.ToArray(), traced,
            (eng, output) =>
            {
                var (r, a) = forward(eng);
                DirectGpuTensorEngine.CopyResultInto(eng, CheckReplayShape(opName, r, output), output);
                refreshAux(eng, a, tracedAux);
            },
            BackwardFunctions<T>.ReplayUnderTape(eng => forward(eng).Result));
        return true;
    }

    // Copies a replay's fresh index tensor into the one the caller holds.
    private static void RefreshIndices(IEngine engine, Tensor<int>? fresh, Tensor<int>? held)
    {
        if (fresh is null || held is null || ReferenceEquals(fresh, held)) return;
        if (fresh.Length != held.Length)
            throw new InvalidOperationException(
                $"A compiled replay produced {fresh.Length} indices where the trace produced {held.Length}.");
        DirectGpuTensorEngine.CopyResultInto(engine, fresh, held);
    }

    // The operands of a variadic op, flattened for TryRecordCompiledOp.
    private protected static Tensor<T>?[] Combine<T>(params IEnumerable<Tensor<T>>[] groups)
    {
        var all = new List<Tensor<T>?>();
        foreach (var group in groups)
            if (group is not null) all.AddRange(group);
        return all.ToArray();
    }
}
