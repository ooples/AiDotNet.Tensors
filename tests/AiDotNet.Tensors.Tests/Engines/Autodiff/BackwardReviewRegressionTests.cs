using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class BackwardReviewRegressionTests
{
    public enum ExecutionPath { Tape, Compiled, Optimized }

    [Fact]
    public void MaxPool1DBackward_AcceptsNonContiguousGradientWithoutMutatingIt()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1, 2, 4 });
        var output = new Tensor<float>(new[] { 1, 2, 2 });
        var gradient = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 2, 2 })
            .Transpose(new[] { 0, 2, 1 });
        Assert.False(gradient.IsContiguous);
        float[] before = gradient.ToArray();
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();

        BackwardFunctions<float>.MaxPool1DBackward(gradient, new[] { input }, output,
            new object[] { new[] { 1, 3, 5, 7 } }, engine, gradients);

        Assert.Equal(new float[] { 0, 1, 0, 3, 0, 2, 0, 4 }, gradients[input].ToArray());
        Assert.Equal(before, gradient.ToArray());
        Assert.False(gradient.IsContiguous);
    }

    [Theory]
    [InlineData(ExecutionPath.Tape)]
    [InlineData(ExecutionPath.Compiled)]
    [InlineData(ExecutionPath.Optimized)]
    public void Float32Policy_ReachesFanOutAdditionButNotBackwardKernels(ExecutionPath execution)
        => VerifyAccumulationPolicy(execution, GradientAccumulationPrecision.Float32);

    [Theory]
    [InlineData(ExecutionPath.Tape)]
    [InlineData(ExecutionPath.Compiled)]
    [InlineData(ExecutionPath.Optimized)]
    public void InheritedPolicy_PreservesLowerPrecisionAndOverridesAnOuterPolicy(ExecutionPath execution)
        => VerifyAccumulationPolicy(execution, GradientAccumulationPrecision.InheritBackwardPrecision);

    private static void VerifyAccumulationPolicy(ExecutionPath execution, GradientAccumulationPrecision precision)
    {
        var engine = new PrecisionObservingCpuEngine();
        IEngine previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var autocast = new AutocastScope(PrecisionMode.Float16);
            using var tape = new GradientTape<float>(new GradientTapeOptions
            {
                Persistent = true,
                GradientAccumulationPrecision = precision,
            });
            var input = new Tensor<float>(new[] { 1f }, new[] { 1 });
            var first = engine.TensorMultiplyScalar(input, 1f);
            var second = engine.TensorMultiplyScalar(input, 0.0005f);
            var loss = engine.ReduceSum(engine.TensorAdd(first, second), new[] { 0 }, false);
            engine.Observe = true;
            GradientAccumulationPrecision outerPrecision = precision == GradientAccumulationPrecision.Float32
                ? GradientAccumulationPrecision.InheritBackwardPrecision
                : GradientAccumulationPrecision.Float32;
            using var outerPolicy = new GradientAccumulationPrecisionScope(outerPrecision);

            Dictionary<Tensor<float>, Tensor<float>> gradients;
            if (execution == ExecutionPath.Tape)
                gradients = tape.ComputeGradients(loss, new[] { input }, createGraph: true);
            else if (execution == ExecutionPath.Compiled)
                gradients = tape.CompileBackward(loss, new[] { input }).Execute();
            else
            {
                int[] indices = Enumerable.Range(0, tape.Entries.Count).Reverse().ToArray();
                var plan = new OptimizedBackwardPlan<float>(tape.Entries, indices, loss,
                    new[] { input }, engine, new BackwardAnalysis(), accumulationPrecision: precision);
                gradients = plan.Execute();
            }

            Assert.InRange(Math.Abs(gradients[input][0] - 1.0005f), 0, 1e-6f);
            Assert.NotEmpty(engine.AdditionPrecisions);
            PrecisionMode expectedAddition = precision == GradientAccumulationPrecision.Float32
                ? PrecisionMode.Float32 : PrecisionMode.Float16;
            Assert.All(engine.AdditionPrecisions, observed => Assert.Equal(expectedAddition, observed));
            Assert.NotEmpty(engine.ScalarPrecisions);
            Assert.All(engine.ScalarPrecisions, precision => Assert.Equal(PrecisionMode.Float16, precision));
            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
            using var restoredOuterAddition = GradientAccumulationPrecisionScope.EnterFloat32AutocastForAddition();
            Assert.Equal(outerPrecision == GradientAccumulationPrecision.Float32 ? PrecisionMode.Float32 : PrecisionMode.Float16,
                AutocastScope.ActivePrecision);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private sealed class PrecisionObservingCpuEngine : CpuEngine
    {
        internal bool Observe { get; set; }
        internal List<PrecisionMode> AdditionPrecisions { get; } = new();
        internal List<PrecisionMode> ScalarPrecisions { get; } = new();

        public override Tensor<T> TensorAdd<T>(Tensor<T> left, Tensor<T> right)
        {
            if (Observe) AdditionPrecisions.Add(AutocastScope.ActivePrecision);
            return base.TensorAdd(left, right);
        }

        public override void TensorAddInPlace<T>(Tensor<T> left, Tensor<T> right)
        {
            if (Observe) AdditionPrecisions.Add(AutocastScope.ActivePrecision);
            base.TensorAddInPlace(left, right);
        }

        public override Tensor<T> TensorMultiplyScalar<T>(Tensor<T> input, T scalar)
        {
            if (Observe) ScalarPrecisions.Add(AutocastScope.ActivePrecision);
            return base.TensorMultiplyScalar(input, scalar);
        }
    }
}
