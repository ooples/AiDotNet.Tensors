using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public class CompiledGraphRegressionTests
{
    public enum ComparisonKind
    {
        Equal,
        NotEqual,
        GreaterThan,
        LessThan
    }

    public enum OperandKind
    {
        Tensor,
        Scalar
    }

    [Theory]
    [InlineData(ComparisonKind.Equal, OperandKind.Tensor)]
    [InlineData(ComparisonKind.Equal, OperandKind.Scalar)]
    [InlineData(ComparisonKind.NotEqual, OperandKind.Tensor)]
    [InlineData(ComparisonKind.NotEqual, OperandKind.Scalar)]
    [InlineData(ComparisonKind.GreaterThan, OperandKind.Tensor)]
    [InlineData(ComparisonKind.GreaterThan, OperandKind.Scalar)]
    [InlineData(ComparisonKind.LessThan, OperandKind.Tensor)]
    [InlineData(ComparisonKind.LessThan, OperandKind.Scalar)]
    public void TrainingPlan_RecomputesComparisonMaskOnReplay(
        ComparisonKind comparisonKind,
        OperandKind operandKind)
    {
        var engine = new CpuEngine();
        float traceValue = comparisonKind is ComparisonKind.NotEqual or ComparisonKind.LessThan ? 1f : 0f;
        float replayValue = comparisonKind == ComparisonKind.LessThan ? 0f :
            comparisonKind == ComparisonKind.Equal ? 1f : 2f;
        using var parameter = new Tensor<float>(new[] { traceValue }, new[] { 1 });
        using var right = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var whenTrue = new Tensor<float>(new[] { 11f }, new[] { 1 });
        using var whenFalse = new Tensor<float>(new[] { 22f }, new[] { 1 });
        using var cache = new CompiledModelCache<float>();

        Tensor<float> ForwardAndLoss()
        {
            Tensor<float> condition = (comparisonKind, operandKind) switch
            {
                (ComparisonKind.Equal, OperandKind.Tensor) => engine.TensorEquals(parameter, right),
                (ComparisonKind.Equal, OperandKind.Scalar) => engine.TensorEquals(parameter, 1f),
                (ComparisonKind.NotEqual, OperandKind.Tensor) => engine.TensorNotEquals(parameter, right),
                (ComparisonKind.NotEqual, OperandKind.Scalar) => engine.TensorNotEquals(parameter, 1f),
                (ComparisonKind.GreaterThan, OperandKind.Tensor) => engine.TensorGreaterThan(parameter, right),
                (ComparisonKind.GreaterThan, OperandKind.Scalar) => engine.TensorGreaterThan(parameter, 1f),
                (ComparisonKind.LessThan, OperandKind.Tensor) => engine.TensorLessThan(parameter, right),
                (ComparisonKind.LessThan, OperandKind.Scalar) => engine.TensorLessThan(parameter, 1f),
                _ => throw new ArgumentOutOfRangeException()
            };
            return engine.ReduceSum(engine.TensorWhere(condition, whenTrue, whenFalse), null);
        }

        var plan = cache.GetOrCompileTraining(parameter._shape, ForwardAndLoss, new[] { parameter });
        parameter[0] = replayValue;

        float loss = plan.Step()[0];

        Assert.Equal(11f, loss);
    }

    [Fact]
    public void TrainingPlan_TensorWhereRoutesGradientUsingReplayMask()
    {
        var engine = new CpuEngine();
        using var parameter = new Tensor<float>(new[] { 0.5f }, new[] { 1 });
        using var one = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var cache = new CompiledModelCache<float>();

        Tensor<float> ForwardAndLoss()
        {
            var reciprocal = engine.TensorDivide(one, parameter);
            var condition = engine.TensorGreaterThan(parameter, one);
            var selected = engine.TensorWhere(condition, reciprocal, parameter);
            return engine.ReduceSum(selected, null);
        }

        var plan = cache.GetOrCompileTraining(parameter._shape, ForwardAndLoss, new[] { parameter });
        parameter[0] = 2f;

        float loss = plan.Step()[0];

        Assert.Equal(0.5f, loss, precision: 6);
        Assert.Equal(-0.25f, plan.Gradients[0][0], precision: 6);
    }

    [Theory]
    [InlineData(ComparisonKind.Equal, OperandKind.Tensor)]
    [InlineData(ComparisonKind.Equal, OperandKind.Scalar)]
    [InlineData(ComparisonKind.NotEqual, OperandKind.Tensor)]
    [InlineData(ComparisonKind.NotEqual, OperandKind.Scalar)]
    public void DirectGpuTrace_CapturesEqualityBeforeKernelDispatch(
        ComparisonKind comparisonKind,
        OperandKind operandKind)
    {
        using var directGpu = CreateMockDirectGpu(out var state);
        using var engine = new DirectGpuTensorEngine(directGpu);
        IEngine dispatch = engine;
        using var parameter = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var right = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Tensor<float>? condition = null;

        try
        {
            using (GraphMode.EnableTraining(new[] { parameter }))
            {
                condition = (comparisonKind, operandKind) switch
                {
                    (ComparisonKind.Equal, OperandKind.Tensor) => dispatch.TensorEquals(parameter, right),
                    (ComparisonKind.Equal, OperandKind.Scalar) => dispatch.TensorEquals(parameter, 1f),
                    (ComparisonKind.NotEqual, OperandKind.Tensor) => dispatch.TensorNotEquals(parameter, right),
                    (ComparisonKind.NotEqual, OperandKind.Scalar) => dispatch.TensorNotEquals(parameter, 1f),
                    _ => throw new ArgumentOutOfRangeException()
                };

                Assert.NotNull(condition.LazySource);
                Assert.Equal(0, state.EqualsCalls);
                Assert.Equal(0, state.NotEqualsCalls);
            }
        }
        finally
        {
            condition?.Dispose();
        }
    }

    [Fact]
    public void DirectGpuTrace_CapturesWhereBeforeKernelDispatch()
    {
        using var directGpu = CreateMockDirectGpu(out var state);
        using var engine = new DirectGpuTensorEngine(directGpu);
        IEngine dispatch = engine;
        using var parameter = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var condition = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var whenTrue = new Tensor<float>(new[] { 11f }, new[] { 1 });
        using var whenFalse = new Tensor<float>(new[] { 22f }, new[] { 1 });
        Tensor<float>? selected = null;

        try
        {
            using (GraphMode.EnableTraining(new[] { parameter }))
            {
                selected = dispatch.TensorWhere(condition, whenTrue, whenFalse);

                Assert.NotNull(selected.LazySource);
                Assert.Equal(0, state.WhereCalls);
            }
        }
        finally
        {
            selected?.Dispose();
        }
    }

    [Fact]
    public void FailedExplicitInferenceTrace_DoesNotExecuteRecordedCallbacksOnDispose()
    {
        using var input = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Tensor<float>? output = null;
        int callbackCount = 0;

        var error = Assert.Throws<InvalidOperationException>((Action)(() =>
        {
            using var scope = GraphMode.EnableInference();
            output = scope.RecordUnary(
                LazyNodeType.Custom,
                "SideEffectProbe",
                input,
                input._shape,
                (_, _) => callbackCount++);
            throw new InvalidOperationException("forced trace failure");
        }));

        Assert.Equal("forced trace failure", error.Message);
        Assert.Equal(0, callbackCount);
        Assert.False(GraphMode.IsActive);
        output?.Dispose();
    }

    [Fact]
    public void FailedExplicitTrainingTrace_DoesNotExecuteRecordedCallbacksOnDispose()
    {
        using var parameter = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Tensor<float>? output = null;
        int callbackCount = 0;

        var error = Assert.Throws<InvalidOperationException>((Action)(() =>
        {
            using var scope = GraphMode.EnableTraining(new[] { parameter });
            output = scope.RecordUnary(
                LazyNodeType.Custom,
                "SideEffectProbe",
                parameter,
                parameter._shape,
                (_, _) => callbackCount++);
            throw new InvalidOperationException("forced trace failure");
        }));

        Assert.Equal("forced trace failure", error.Message);
        Assert.Equal(0, callbackCount);
        Assert.False(GraphMode.IsActive);
        output?.Dispose();
    }

    [Fact]
    public void CompatibilityTrace_StillExecutesRecordedCallbacksOnDispose()
    {
        using var input = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Tensor<float>? output = null;
        int callbackCount = 0;

        using (var scope = GraphMode.Enable())
        {
            output = scope.RecordUnary(
                LazyNodeType.Custom,
                "CompatibilityProbe",
                input,
                input._shape,
                (_, destination) =>
                {
                    callbackCount++;
                    destination[0] = 7f;
                });
        }

        Assert.Equal(1, callbackCount);
        Assert.Equal(7f, output![0]);
        output.Dispose();
    }

    [Fact]
    public void UnrootedInferenceOutput_UsesTypedCaptureLimitation()
    {
        var engine = new CpuEngine();
        using var input = new Tensor<float>(new[] { 1f }, new[] { 1 });
        using var cache = new CompiledModelCache<float>();

        Tensor<float>? unrooted = null;
        var error = Assert.Throws<GraphCaptureNotSupportedException>(() =>
            cache.GetOrCompileInference(input, () =>
            {
                _ = engine.TensorAdd(input, input);
                unrooted = new Tensor<float>(new[] { 2f }, new[] { 1 });
                return unrooted;
            }));

        Assert.Equal(GraphCaptureLimitation.UnrootedOutput, error.Limitation);
        unrooted?.Dispose();
    }

    private static void SetField(object target, string name, object value)
    {
        var field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field not found: {name}");
        field.SetValue(target, value);
    }

    private static DirectGpuEngine CreateMockDirectGpu(out MockBackendState state)
    {
        string? priorBackends = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        DirectGpuEngine directGpu;
        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "none");
            directGpu = new DirectGpuEngine();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", priorBackends);
        }

        state = new MockBackendState();
        SetField(directGpu, "_backend", MockDirectGpuBackend.Create(state));
        SetField(directGpu, "_isAvailable", true);
        return directGpu;
    }
}
