using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
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
    /// <param name="backwardForward">
    /// The forward the backward re-runs under a tape, when it must differ from <paramref name="forward"/>: a random op
    /// whose forward draws a fresh seed each step re-runs, for its backward, with the seed that step drew.
    /// </param>
    /// <param name="replaysHostState">
    /// True when <paramref name="forward"/> updates host state every replay (such a seed). The plan then never captures
    /// its step as a CUDA graph, whose replays would keep the capture-time state.
    /// </param>
    internal bool TryRecordCompiledOp<T>(string opName, Tensor<T>?[] inputs, Func<IEngine, Tensor<T>> forward,
        [NotNullWhen(true)] out Tensor<T>? result, Func<IEngine, Tensor<T>>? backwardForward = null,
        bool replaysHostState = false)
    {
        result = null;
        if (BeginCompiledRecord(inputs, out var operands) is not { } scope) return false;
        BindCaptures(scope, forward, operands);
        if (backwardForward is not null) BindCaptures(scope, backwardForward, operands);

        Tensor<T> traced;
        GraphMode.SetCurrent(null);
        try
        {
            using (new NoGradScope<T>())
                traced = OwnedOutput(forward(this));
        }
        finally
        {
            GraphMode.SetCurrent(scope);
        }

        result = scope.RecordMaterializedVariadic(LazyNodeType.Custom, opName, operands.ToArray(), traced,
            (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, CheckReplayShape(opName, forward(eng), output), output),
            BackwardFunctions<T>.ReplayUnderTape(backwardForward ?? forward));
        if (replaysHostState && result.LazySource is LazyNode<T> node) node.ReplaysHostState = true;
        return true;
    }

    // The node's output buffer must own its storage. Clone() is a copy-on-write share of the trace-time eager result
    // (a pooled tensor), so the plan's buffer for this node - and the views consumers took of it while tracing, such as
    // the Reshape a Flatten records - stayed tied to storage the plan does not own. Measured (#1705): a compiled
    // Conv/MaxPool/Flatten/Dense CPU step whose loss was 1.5e13 or 2.3e35 (ln 10 = 2.3026, all-zero logits, on a fresh
    // process) where the eager step's was 3.1733, with two identical compiled steps disagreeing; with an owned copy the
    // compiled loss is 3.1733 and the step matches eager to 3e-8.
    private static Tensor<T> OwnedOutput<T>(Tensor<T> eager) => (Tensor<T>)eager.CloneDeepCopy();

    // The recording scope of a training trace that binds this engine, and the distinct non-null operands in order;
    // null when the op runs eagerly (no trace, an inference trace, another engine's trace, or no tensor operand).
    private LazyTensorScope? BeginCompiledRecord<T>(Tensor<T>?[] inputs, out List<Tensor<T>> operands)
    {
        operands = new List<Tensor<T>>(inputs.Length);
        if (!GraphMode.IsActive || GraphMode.IsInferenceTrace) return null;
        var current = GraphMode.Current;
        if (current is null || !current.BindEngine(this)) return null;
        for (int i = 0; i < inputs.Length; i++)
        {
            var input = inputs[i];
            if (input is not null && !operands.Contains(input)) operands.Add(input);
        }

        return operands.Count == 0 ? null : current;
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
        [NotNullWhen(true)] out Tensor<T>? result, [MaybeNullWhen(false)] out TAux aux)
    {
        result = null;
        aux = default;
        if (BeginCompiledRecord(inputs, out var operands) is not { } scope) return false;
        BindCaptures(scope, forward, operands);

        Tensor<T> traced;
        TAux tracedAux;
        GraphMode.SetCurrent(null);
        try
        {
            using (new NoGradScope<T>())
            {
                var (r, a) = forward(this);
                traced = OwnedOutput(r);
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
        if (tracedAux is not null) scope.RegisterAuxiliaryOutput(tracedAux, result);
        return true;
    }

    /// <summary>
    /// Fixes what a recorded op's closure captured to its value at trace time, and makes the node depend on the
    /// producers of the auxiliary outputs it reads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each replay re-runs the closure, so an array argument (a pad, axes, a pool size, the operand list of a variadic
    /// op) that the caller changed after tracing would change the op without changing the recorded graph. Every
    /// captured array is replaced by a copy, as an eager call would have used the value it was given.
    /// </para>
    /// <para>
    /// A captured object that is another node's auxiliary output (MaxPool's argmax, ReduceMax's indices) is not copied:
    /// that node refreshes it in place on each replay. Its primary output joins the operands instead, so every
    /// schedule runs the producer first. Without the edge a consumer of only the indices (ScatterAdd, Ldexp,
    /// MaxUnpool) could run before the refresh and read the previous step's winners.
    /// </para>
    /// </remarks>
    private static void BindCaptures<T>(LazyTensorScope scope, Delegate closure, List<Tensor<T>> operands)
    {
        if (closure.Target is { } target
            && target.GetType().IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute), false))
            BindCaptures(scope, target, operands, depth: 0);
    }

    private static void BindCaptures<T>(LazyTensorScope scope, object closure, List<Tensor<T>> operands, int depth)
    {
        const System.Reflection.BindingFlags fields =
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        foreach (var field in closure.GetType().GetFields(fields))
        {
            if (field.FieldType.IsValueType || field.GetValue(closure) is not { } value) continue;
            if (DependOnProducer(scope, value, operands)) continue;
            if (value is Array array)
            {
                if (array.GetType().GetElementType() is { IsValueType: false })
                    foreach (var element in array.OfType<object>())
                        DependOnProducer(scope, element, operands);
                field.SetValue(closure, array.Clone());
            }
            else if (depth < 4 && field.FieldType.IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute), false))
            {
                // A closure over variables of an enclosing scope holds that scope's closure object.
                BindCaptures(scope, value, operands, depth + 1);
            }
        }
    }

    private static bool DependOnProducer<T>(LazyTensorScope scope, object value, List<Tensor<T>> operands)
    {
        if (!scope.TryGetAuxiliaryProducer(value, out var producer)) return false;
        if (producer is Tensor<T> output && !operands.Contains(output)) operands.Add(output);
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
        foreach (var group in groups.Where(g => g is not null))
            all.AddRange(group);
        return all.ToArray();
    }
}
