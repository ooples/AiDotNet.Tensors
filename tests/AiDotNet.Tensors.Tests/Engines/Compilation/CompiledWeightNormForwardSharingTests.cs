using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledWeightNormForwardSharingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Repeated_forward_keeps_both_derivatives_and_five_Adam_updates(bool secondNoGrad)
    {
        var baseline = WeightNormSharingTestGraph.Run(new CpuEngine(), false, secondNoGrad);
        var shared = WeightNormSharingTestGraph.Run(new CpuEngine(), true, secondNoGrad);
        WeightNormSharingTestGraph.AssertEquivalent(baseline, shared, expectSharing: true);
    }

    [Fact]
    public void Declared_forward_outputs_remain_independent()
    {
        var baseline = WeightNormSharingTestGraph.Run(new CpuEngine(), false, false, retainProducers: true);
        var guarded = WeightNormSharingTestGraph.Run(new CpuEngine(), true, false, retainProducers: true);
        WeightNormSharingTestGraph.AssertEquivalent(baseline, guarded, expectSharing: false);
    }

    [Fact]
    public void Custom_cpu_engine_keeps_separate_storage()
    {
        var baseline = WeightNormSharingTestGraph.Run(new CustomCpuEngine(), false, false);
        var guarded = WeightNormSharingTestGraph.Run(new CustomCpuEngine(), true, false);
        WeightNormSharingTestGraph.AssertEquivalent(baseline, guarded, expectSharing: false);
    }

    private sealed class CustomCpuEngine : CpuEngine { }
}

internal static class WeightNormSharingTestGraph
{
    private const string Switch = "AIDOTNET_COMPILED_WEIGHTNORM_FORWARD_SHARE";
    private const double Epsilon = .03125;
    private static readonly float[] InitialDirection = { .5f, -.75f, .125f, .375f, -.625f, -.5f, .25f, .625f, -.125f, .375f };
    private static readonly float[] InitialGain = { .75f, -1.25f };
    private static readonly float[] SeedA = { .125f, -.25f, .5f, .375f, -.625f, -.5f, .25f, .625f, -.125f, .375f };
    private static readonly float[] SeedB = { -.25f, .125f, -.375f, .25f, .0625f, .5f, -.375f, .125f, .25f, -.625f };

    internal sealed class Result
    {
        internal readonly List<float[]> Values = new();
        internal long DistinctForwardBackingBytes;
        internal int DistinctForwardArrays;
        internal bool ObservedManagedBacking;
        internal int DistinctForwardStorages;
        internal int DistinctForwardVectors;
        internal long DistinctForwardVectorCapacityBytes;
        internal int DifferentiableWeightNormNodes;
        internal int NoGradWeightNormNodes;
    }

    internal static Result Run(IEngine engine, bool sharing, bool secondNoGrad, bool retainProducers = false)
    {
        string? oldSwitch = Environment.GetEnvironmentVariable(Switch);
        IEngine oldEngine = AiDotNetEngine.Current;
        Environment.SetEnvironmentVariable(Switch, sharing ? null : "0");
        AiDotNetEngine.Current = engine;
        try
        {
            var direction = new Tensor<float>((float[])InitialDirection.Clone(), new[] { 2, 5 });
            var gain = new Tensor<float>((float[])InitialGain.Clone(), new[] { 2 });
            var seedA = new Tensor<float>((float[])SeedA.Clone(), new[] { 2, 5 });
            var seedB = new Tensor<float>((float[])SeedB.Clone(), new[] { 2, 5 });
            var parameters = new[] { direction, gain };
            if (engine is DirectGpuTensorEngine gpu)
                foreach (var input in new[] { direction, gain, seedA, seedB }) gpu.EnsureResidentInput(input);
            Tensor<float> first;
            Tensor<float> second;
            Tensor<float> observed;
            ICompiledTrainingPlan<float> compiled;
            using (var graph = GraphMode.EnableTraining(parameters))
            {
                first = engine.TensorWeightNorm(direction, gain, Epsilon);
                if (secondNoGrad)
                    using (new NoGradScope<float>()) second = engine.TensorWeightNorm(direction, gain, Epsilon);
                else second = engine.TensorWeightNorm(direction, gain, Epsilon);
                // The cross term reads both normalized values during backward. Sharing must preserve
                // both graph edges, including the case in which only the second WN is under NoGrad.
                var linear = engine.TensorAdd(engine.TensorMultiply(first, seedA), engine.TensorMultiply(second, seedB));
                observed = engine.TensorAdd(linear, engine.TensorMultiplyScalar(engine.TensorMultiply(first, second), .125f));
                graph.RetainOutput(observed);
                if (retainProducers) { graph.RetainOutput(first); graph.RetainOutput(second); }
                compiled = graph.CompileTraining(parameters, engine.ReduceSum(observed));
            }
            using (compiled)
            {
                var plan = Assert.IsType<CompiledTrainingPlan<float>>(compiled);
                plan.DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, .003, .9, .999, 1e-8);
                var weightNorm = ((CompiledStep<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(plan)!)
                    .Where(step => step.OpName == "TensorWeightNorm").ToArray();
                Assert.Equal(2, weightNorm.Length);
                var result = new Result
                {
                    DifferentiableWeightNormNodes = weightNorm.Count(step => step.BackwardFn is not null && !step.RecordedUnderNoGrad),
                    NoGradWeightNormNodes = weightNorm.Count(step => step.BackwardFn is null && step.RecordedUnderNoGrad),
                };
                Assert.Equal(secondNoGrad ? 1 : 2, result.DifferentiableWeightNormNodes);
                Assert.Equal(secondNoGrad ? 1 : 0, result.NoGradWeightNormNodes);
                var expectedParameters = new[] { (float[])InitialDirection.Clone(), (float[])InitialGain.Clone() };
                var moment1 = new[] { new double[10], new double[2] };
                var moment2 = new[] { new double[10], new double[2] };
                for (int iteration = 1; iteration <= 5; ++iteration)
                {
                    var expected = Oracle(expectedParameters[0], expectedParameters[1], secondNoGrad);
                    var loss = plan.Step().ToArray();
                    var actualOutput = observed.ToArray();
                    result.Values.Add(loss);
                    result.Values.Add(actualOutput);
                    Assert.InRange(Math.Abs(loss[0] - expected.Loss), 0, 2e-5);
                    for (int i = 0; i < actualOutput.Length; ++i)
                        Assert.InRange(Math.Abs(actualOutput[i] - expected.Output[i]), 0, 2e-5);
                    for (int p = 0; p < parameters.Length; ++p)
                    {
                        float[] actualGradient = plan.Gradients[p].ToArray();
                        result.Values.Add(actualGradient);
                        for (int i = 0; i < actualGradient.Length; ++i)
                        {
                            Assert.InRange(Math.Abs(actualGradient[i] - expected.Gradients[p][i]), 0, 2e-5);
                            double gradient = expected.Gradients[p][i];
                            moment1[p][i] = .9 * moment1[p][i] + .1 * gradient;
                            moment2[p][i] = .999 * moment2[p][i] + .001 * gradient * gradient;
                            expectedParameters[p][i] -= (float)(.003 * moment1[p][i] / (1 - Math.Pow(.9, iteration))
                                / (Math.Sqrt(moment2[p][i] / (1 - Math.Pow(.999, iteration))) + 1e-8));
                        }
                        float[] actualParameter = parameters[p].ToArray();
                        result.Values.Add(actualParameter);
                        for (int i = 0; i < actualParameter.Length; ++i)
                            Assert.InRange(Math.Abs(actualParameter[i] - expectedParameters[p][i]), 0, 3e-5);
                    }
                    if (retainProducers)
                    {
                        // An explicitly published output must not expose a writable alias to its peer.
                        float peer = second[0];
                        first[0] += 7f;
                        Assert.Equal(peer, second[0]);
                    }
                    var storages = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    var vectors = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    long vectorCapacityBytes = 0;
                    foreach (var step in weightNorm)
                    {
                        storages.Add(step.OutputBuffer._storage);
                        var vector = (Vector<float>)step.OutputBuffer.GraphVectorIdentity;
                        Assert.Equal(step.OutputBuffer.Length, vector.Length);
                        if (vectors.Add(vector)) vectorCapacityBytes += (long)vector.Length * sizeof(float);
                    }
                    long bytes = 0;
                    int arrayCount = 0;
                    bool observeManagedBacking = engine is not DirectGpuTensorEngine;
                    if (observeManagedBacking)
                    {
                        var arrays = weightNorm.Select(step => step.OutputBuffer.GetBackingArrayForCacheLookupUnsafe()).ToArray();
                        Assert.All(arrays, array => Assert.NotNull(array));
                        var distinct = new HashSet<float[]>(ReferenceEqualityComparer.Instance);
                        foreach (float[] array in arrays.Cast<float[]>())
                            if (distinct.Add(array)) bytes += (long)array.Length * sizeof(float);
                        arrayCount = distinct.Count;
                    }
                    if (engine is DirectGpuTensorEngine native)
                    {
                        Assert.Equal(SeedA, seedA.ToArray());
                        Assert.Equal(SeedB, seedB.ToArray());
                        // Deferred device storage need not have a managed array. The direct CUDA fixture
                        // separately proves simultaneous native alias ownership and surviving-view lifetime.
                        native.ReleaseUnusedDeviceMemory();
                    }
                    if (iteration == 1)
                    {
                        result.ObservedManagedBacking = observeManagedBacking;
                        result.DistinctForwardBackingBytes = bytes;
                        result.DistinctForwardArrays = arrayCount;
                        result.DistinctForwardStorages = storages.Count;
                        result.DistinctForwardVectors = vectors.Count;
                        result.DistinctForwardVectorCapacityBytes = vectorCapacityBytes;
                    }
                    else
                    {
                        Assert.Equal(result.DistinctForwardBackingBytes, bytes);
                        Assert.Equal(result.DistinctForwardArrays, arrayCount);
                        Assert.Equal(result.DistinctForwardStorages, storages.Count);
                        Assert.Equal(result.DistinctForwardVectors, vectors.Count);
                        Assert.Equal(result.DistinctForwardVectorCapacityBytes, vectorCapacityBytes);
                    }
                }
                Assert.Contains(result.Values[2], value => value != 0f);
                return result;
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(Switch, oldSwitch);
            AiDotNetEngine.Current = oldEngine;
        }
    }

    internal static void AssertEquivalent(Result baseline, Result candidate, bool expectSharing)
    {
        Assert.Equal(2, baseline.DistinctForwardStorages);
        Assert.Equal(2, baseline.DistinctForwardVectors);
        Assert.Equal(expectSharing ? 1 : 2, candidate.DistinctForwardStorages);
        Assert.Equal(expectSharing ? 1 : 2, candidate.DistinctForwardVectors);
        Assert.Equal(baseline.DistinctForwardVectorCapacityBytes,
            candidate.DistinctForwardVectorCapacityBytes * (expectSharing ? 2 : 1));
        Assert.Equal(baseline.ObservedManagedBacking, candidate.ObservedManagedBacking);
        if (baseline.ObservedManagedBacking)
        {
            Assert.Equal(2, baseline.DistinctForwardArrays);
            Assert.Equal(expectSharing ? 1 : 2, candidate.DistinctForwardArrays);
            Assert.Equal(baseline.DistinctForwardBackingBytes,
                candidate.DistinctForwardBackingBytes * (expectSharing ? 2 : 1));
        }
        Assert.Equal(baseline.DifferentiableWeightNormNodes, candidate.DifferentiableWeightNormNodes);
        Assert.Equal(baseline.NoGradWeightNormNodes, candidate.NoGradWeightNormNodes);
        Assert.Equal(baseline.Values.Count, candidate.Values.Count);
        for (int row = 0; row < baseline.Values.Count; ++row)
            Assert.Equal(baseline.Values[row].Select(BitConverter.SingleToInt32Bits),
                candidate.Values[row].Select(BitConverter.SingleToInt32Bits));
    }

    private static (double Loss, double[] Output, double[][] Gradients) Oracle(float[] direction, float[] gain, bool secondNoGrad)
    {
        var output = new double[direction.Length];
        var gradients = new[] { new double[direction.Length], new double[gain.Length] };
        double loss = 0;
        for (int row = 0; row < gain.Length; ++row)
        {
            double squares = 0;
            for (int column = 0; column < 5; ++column)
            {
                double value = direction[row * 5 + column];
                squares += value * value;
            }
            double inverse = 1 / Math.Sqrt(squares + Epsilon);
            double dot = 0;
            var adjoint = new double[5];
            for (int column = 0; column < 5; ++column)
            {
                int i = row * 5 + column;
                double weight = gain[row] * direction[i] * inverse;
                output[i] = (SeedA[i] + SeedB[i]) * weight + .125 * weight * weight;
                loss += output[i];
                adjoint[column] = SeedA[i] + .125 * weight;
                if (!secondNoGrad) adjoint[column] += SeedB[i] + .125 * weight;
                dot += adjoint[column] * direction[i];
            }
            gradients[1][row] = dot * inverse;
            for (int column = 0; column < 5; ++column)
            {
                int i = row * 5 + column;
                gradients[0][i] = gain[row] * inverse * (adjoint[column] - direction[i] * dot * inverse * inverse);
            }
        }
        return (loss, output, gradients);
    }
}
