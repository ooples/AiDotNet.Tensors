// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Registry-derived graph-capture contract. Every required case is compiled once, its exact leaf
/// tensors are changed, and replay is compared with an eager invocation whose <see cref="OpInput"/>
/// values receive the same deterministic change. An eager computation hidden behind a later lazy
/// projection therefore fails as stale rather than passing a superficial LazySource assertion.
/// </summary>
[Collection("OpParity")]
public sealed class GraphCaptureParityTests
{
    private static readonly IReadOnlyDictionary<string, OpCase> Cases =
        OpParityRegistry.All().ToDictionary(o => o.Name);

    public static IEnumerable<object[]> ForwardCases =>
        OpParityRegistry.All()
            .Where(o => o.GraphCaptureExpectation != GraphCaptureExpectation.BackwardKernel)
            .Where(o => ParityShard.Include(o.Name))
            .Select(o => new object[] { o.Name });

    public static IEnumerable<object[]> MultiOutputCases =>
        OpParityRegistry.All()
            .Where(o => o.HasMultipleOutputs &&
                        o.GraphCaptureExpectation == GraphCaptureExpectation.Required)
            .Where(o => ParityShard.Include(o.Name))
            .Select(o => new object[] { o.Name });

    [SkippableTheory]
    [MemberData(nameof(ForwardCases))]
    public void Forward_CompiledReplayReflectsLiveInputs(string opName)
    {
        var op = Cases[opName];
        var engine = new CpuEngine();

        if (op.GraphCaptureExpectation == GraphCaptureExpectation.NonDeterministic)
        {
            VerifyNonDeterministicCaptureIsSafe(op, engine);
            return;
        }

        if (op.GraphCaptureExpectation == GraphCaptureExpectation.InputIndependent)
        {
            VerifyInputIndependentContract(op, engine);
            return;
        }

        if (op.GraphCaptureSignatureConstraint != GraphCaptureSignatureConstraint.None ||
            op.GraphCaptureExpectation is GraphCaptureExpectation.DataDependentOutputShape or
            GraphCaptureExpectation.HeterogeneousInput or
            GraphCaptureExpectation.HeterogeneousOutput or
            GraphCaptureExpectation.MixedElementTypes or
            GraphCaptureExpectation.HostBoundary or
            GraphCaptureExpectation.Stateful)
        {
            VerifyCaptureIsRejected(op, engine);
            return;
        }

        GraphMutationProfile mutation = SelectMutation(op, engine, out float[] eagerChanged);

        var snapshots = new List<FloatInputSnapshot>();
        CompiledInferencePlan<float> plan;
        using (var graphScope = GraphMode.EnableInference())
        using (OpInput.CaptureFloatInputSnapshots(snapshots))
        {
            var graphOutput = op.RunFloat(engine);
            Assert.NotNull(graphOutput.LazySource);
            Assert.NotEmpty(snapshots);
            Assert.Contains(snapshots, snapshot => snapshot.Role == GraphInputRole.MutableValue);
            plan = graphScope.CompileInference(
                graphOutput,
                snapshots.Select(snapshot => snapshot.Tensor).ToArray());
        }

        using (plan)
        {
            foreach (var snapshot in snapshots)
            {
                var destination = snapshot.Tensor.AsWritableSpan();
                if (snapshot.Role == GraphInputRole.MutableValue)
                {
                    var mutated = (float[])snapshot.InitialValues.Clone();
                    OpInput.ApplyGraphMutation(
                        mutated, snapshot.InitialValues, mutation, snapshot.MutableOrdinal);
                    mutated.AsSpan().CopyTo(destination);
                }
                else
                {
                    snapshot.InitialValues.AsSpan().CopyTo(destination);
                }
            }

            float[] compiledChanged = plan.Execute().ToArray();
            Assert.Equal(eagerChanged.Length, compiledChanged.Length);
            Assert.True(
                ParityMath.Within(compiledChanged, eagerChanged, op.Fwd, out var delta),
                $"{op.Name}: compiled replay did not reflect live inputs. {delta.Describe()}");
        }
    }

    [Theory]
    [MemberData(nameof(MultiOutputCases))]
    public void MultipleOutputs_CompiledReplayReflectsLiveInputs(string opName)
    {
        var op = Cases[opName];
        var engine = new CpuEngine();

        Tensor<float>[] eagerOriginal = op.RunFloatOutputs!(engine);
        GraphMutationProfile mutation = SelectOutputMutation(op, engine, eagerOriginal, out Tensor<float>[] eagerChanged);

        Assert.True(eagerOriginal.Length > 1, $"{op.Name}: a multi-output contract must expose at least two outputs.");
        Assert.Equal(eagerOriginal.Length, eagerChanged.Length);

        for (int outputIndex = 0; outputIndex < eagerOriginal.Length; outputIndex++)
        {
            Assert.Equal(eagerOriginal[outputIndex].Shape.ToArray(), eagerChanged[outputIndex].Shape.ToArray());

            var snapshots = new List<FloatInputSnapshot>();
            CompiledInferencePlan<float> plan;
            using (var graphScope = GraphMode.EnableInference())
            using (OpInput.CaptureFloatInputSnapshots(snapshots))
            {
                Tensor<float>[] graphOutputs = op.RunFloatOutputs!(engine);
                Assert.Equal(eagerOriginal.Length, graphOutputs.Length);
                Assert.All(graphOutputs, output => Assert.NotNull(output.LazySource));
                Assert.NotEmpty(snapshots);
                Assert.Contains(snapshots, snapshot => snapshot.Role == GraphInputRole.MutableValue);
                plan = graphScope.CompileInference(
                    graphOutputs[outputIndex],
                    snapshots.Select(snapshot => snapshot.Tensor).ToArray());
            }

            using (plan)
            {
                foreach (var snapshot in snapshots)
                {
                    var destination = snapshot.Tensor.AsWritableSpan();
                    if (snapshot.Role == GraphInputRole.MutableValue)
                    {
                        var mutated = (float[])snapshot.InitialValues.Clone();
                        OpInput.ApplyGraphMutation(
                            mutated, snapshot.InitialValues, mutation, snapshot.MutableOrdinal);
                        mutated.AsSpan().CopyTo(destination);
                    }
                    else
                    {
                        snapshot.InitialValues.AsSpan().CopyTo(destination);
                    }
                }

                Tensor<float> compiledChanged = plan.Execute();
                Assert.Equal(eagerChanged[outputIndex].Shape.ToArray(), compiledChanged.Shape.ToArray());
                Assert.True(
                    ParityMath.Within(
                        compiledChanged.ToArray(), eagerChanged[outputIndex].ToArray(), op.Fwd, out var delta),
                    $"{op.Name} output[{outputIndex}]: compiled replay did not reflect live inputs. {delta.Describe()}");
            }
        }
    }

    private static readonly GraphMutationProfile[] MutationProfiles =
    {
        GraphMutationProfile.RotateValues,
        GraphMutationProfile.AlternatingScale,
        GraphMutationProfile.Affine,
        GraphMutationProfile.ContractTowardZero,
        GraphMutationProfile.FirstMutableInput,
        GraphMutationProfile.SentinelPattern
    };

    private static GraphMutationProfile SelectMutation(
        OpCase op,
        IEngine engine,
        out float[] eagerChanged)
    {
        using Tensor<float> originalTensor = op.RunFloat(engine);
        float[] eagerOriginal = originalTensor.ToArray();
        foreach (GraphMutationProfile candidate in MutationProfiles)
        {
            try
            {
                Tensor<float> changedTensor;
                using (OpInput.UseGraphMutation(candidate))
                    changedTensor = op.RunFloat(engine);
                using (changedTensor)
                {
                    eagerChanged = changedTensor.ToArray();
                }

                if (eagerOriginal.Length == eagerChanged.Length &&
                    PreservesFiniteDomain(eagerOriginal, eagerChanged) &&
                    !ParityMath.BitExact(eagerOriginal, eagerChanged, out _))
                    return candidate;
            }
            catch (ArgumentException)
            {
                // This mutation violated an operation-specific input domain. Try the next
                // deterministic profile; the chosen profile still has to replay exactly.
            }
        }

        eagerChanged = eagerOriginal;
        throw new Xunit.Sdk.XunitException(
            $"{op.Name}: none of the typed graph mutations changed the eager result. " +
            "Mark true constants/metadata with a typed GraphCaptureExpectation, or make numeric " +
            "OpInput.From data explicitly mutable with OpInput.MutableFrom.");
    }

    private static GraphMutationProfile SelectOutputMutation(
        OpCase op,
        IEngine engine,
        Tensor<float>[] eagerOriginal,
        out Tensor<float>[] eagerChanged)
    {
        foreach (GraphMutationProfile candidate in MutationProfiles)
        {
            Tensor<float>[]? candidateOutputs = null;
            bool selected = false;
            try
            {
                using (OpInput.UseGraphMutation(candidate))
                    candidateOutputs = op.RunFloatOutputs!(engine);

                bool everyOutputSatisfied = eagerOriginal.Length == candidateOutputs.Length;
                for (int outputIndex = 0;
                     everyOutputSatisfied && outputIndex < eagerOriginal.Length;
                     outputIndex++)
                {
                    float[] before = eagerOriginal[outputIndex].ToArray();
                    float[] after = candidateOutputs[outputIndex].ToArray();
                    bool bitExact = ParityMath.BitExact(before, after, out _);
                    everyOutputSatisfied = before.Length == after.Length &&
                        PreservesFiniteDomain(before, after) &&
                        (OutputDependency(op, outputIndex) == GraphOutputDependency.MutableInput
                            ? !bitExact
                            : bitExact);
                }

                if (everyOutputSatisfied)
                {
                    selected = true;
                    eagerChanged = candidateOutputs;
                    return candidate;
                }
            }
            catch (ArgumentException)
            {
                // See SelectMutation: domain-invalid probes are discarded, never accepted.
            }
            finally
            {
                if (!selected && candidateOutputs is not null)
                    foreach (Tensor<float>? output in candidateOutputs)
                        output?.Dispose();
            }
        }

        eagerChanged = eagerOriginal;
        throw new Xunit.Sdk.XunitException(
            $"{op.Name}: no typed graph mutation satisfied every homogeneous output dependency.");
    }

    private static GraphOutputDependency OutputDependency(OpCase op, int outputIndex)
    {
        GraphOutputDependency[]? dependencies = op.GraphOutputDependencies;
        return dependencies is not null && outputIndex < dependencies.Length
            ? dependencies[outputIndex]
            : GraphOutputDependency.MutableInput;
    }

    private static bool PreservesFiniteDomain(float[] before, float[] after)
    {
        for (int i = 0; i < before.Length; i++)
            if (float.IsFinite(before[i]) && !float.IsFinite(after[i]))
                return false;
        return true;
    }

    private static void VerifyCaptureIsRejected(OpCase op, IEngine engine)
    {
        var snapshots = new List<FloatInputSnapshot>();
        using var scope = GraphMode.EnableInference();
        using var capture = OpInput.CaptureFloatInputSnapshots(snapshots);
        Tensor<float> output;
        try
        {
            output = op.RunFloat(engine);
        }
        catch (GraphCaptureNotSupportedException)
        {
            return;
        }
        using (output)
        using (CompiledInferencePlan<float> unexpected = scope.CompileInference(
            output,
            snapshots.Select(snapshot => snapshot.Tensor).ToArray()))
        {
        }

        string captureContract = op.GraphCaptureSignatureConstraint != GraphCaptureSignatureConstraint.None
            ? $"signature constraint {op.GraphCaptureSignatureConstraint}"
            : $"capture expectation {op.GraphCaptureExpectation}";
        throw new Xunit.Sdk.XunitException(
            $"{op.Name}: {captureContract} must fail closed during inference capture.");
    }

    private static void VerifyInputIndependentContract(OpCase op, IEngine engine)
    {
        float[] baseline = op.RunFloat(engine).ToArray();
        foreach (GraphMutationProfile mutation in MutationProfiles)
        {
            float[] candidate;
            using (OpInput.UseGraphMutation(mutation))
                candidate = op.RunFloat(engine).ToArray();
            Assert.True(
                ParityMath.BitExact(baseline, candidate, out _),
                $"{op.Name}: InputIndependent result changed under {mutation}.");
        }

        var snapshots = new List<FloatInputSnapshot>();
        using var scope = GraphMode.EnableInference();
        using var capture = OpInput.CaptureFloatInputSnapshots(snapshots);
        _ = op.RunFloat(engine);
        scope.MarkCompiled();
    }

    private static void VerifyNonDeterministicCaptureIsSafe(OpCase op, IEngine engine)
    {
        var snapshots = new List<FloatInputSnapshot>();
        using var scope = GraphMode.EnableInference();
        using var capture = OpInput.CaptureFloatInputSnapshots(snapshots);
        _ = op.RunFloat(engine);
        scope.MarkCompiled();
    }
}

/// <summary>
/// Generated invariant over the complete parity registry: whenever an operation elects to record
/// a graph node, the placeholder shape must already equal its eager result shape. This catches
/// incorrect capture metadata without requiring a hand-maintained list of graph-aware operations.
/// </summary>
[Collection("OpParity")]
public sealed class GraphRecordedOutputShapeTests
{
    private static readonly IReadOnlyDictionary<string, OpCase> Cases =
        OpParityRegistry.All().ToDictionary(o => o.Name);

    public static IEnumerable<object[]> CasesInShard =>
        OpParityRegistry.All()
            .Where(o => ParityShard.Include(o.Name))
            .Select(o => new object[] { o.Name });

    [Theory]
    [MemberData(nameof(CasesInShard))]
    public void RecordedOutputShape_MatchesEagerOutput(string opName)
    {
        var op = Cases[opName];
        var engine = new CpuEngine();
        int[] eagerShape = op.RunFloat(engine).Shape.ToArray();

        Tensor<float> graphOutput;
        try
        {
            using var scope = GraphMode.EnableInference();
            graphOutput = op.RunFloat(engine);

            // This invariant deliberately inspects capture metadata before execution. Marking the
            // scope complete prevents a malformed, undersized placeholder from throwing during
            // Dispose and replacing the more useful shape assertion below.
            scope.MarkCompiled();
        }
        catch (NotSupportedException)
        {
            return;
        }

        if (graphOutput.LazySource is null)
            return;

        Assert.Equal(eagerShape, graphOutput.Shape.ToArray());
    }
}

#endif
