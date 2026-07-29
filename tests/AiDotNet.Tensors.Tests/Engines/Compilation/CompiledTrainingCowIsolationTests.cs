using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompiledTrainingCowIsolationTests
{
    [Fact]
    public void TrainingTrace_DetachesCowParameterBeforeCapturingDerivedView()
    {
        var engine = new CpuEngine();
        var peer = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var trainable = (Tensor<double>)peer.CloneShared();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });

        using var plan = CompileSquaredError(engine, input, target, trainable, new[] { trainable });
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.1f, 0.9f, 0.999f, 1e-8f, 0f);

        var loss = plan.Step()[0];

        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
        Assert.True(loss > 0.0);
        Assert.NotEqual(1.0, trainable[0]);
        Assert.Equal(1.0, peer[0]);
    }

    [Fact]
    public void SequentialCowPeers_CompileAndTrainIndependently()
    {
        var engine = new CpuEngine();
        var first = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var second = (Tensor<double>)first.CloneShared();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });

        using (var firstPlan = CompileSquaredError(engine, input, target, first, new[] { first }))
        {
            firstPlan.ConfigureOptimizer(OptimizerType.SGD, 0.1f, 0.9f, 0.999f, 1e-8f, 0f);
            firstPlan.Step();
        }

        double firstAfterOwnStep = first[0];
        Assert.Equal(1.0, second[0]);

        using (var secondPlan = CompileSquaredError(engine, input, target, second, new[] { second }))
        {
            secondPlan.ConfigureOptimizer(OptimizerType.SGD, 0.05f, 0.9f, 0.999f, 1e-8f, 0f);
            secondPlan.Step();
        }

        Assert.Equal(firstAfterOwnStep, first[0]);
        Assert.NotEqual(1.0, second[0]);
    }

    [Fact]
    public void CachedTrainingPlan_DoesNotRetraceOrRedetachOnCacheHit()
    {
        var engine = new CpuEngine();
        var peer = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });
        var trainable = (Tensor<double>)peer.CloneShared();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
        int traceCount = 0;

        using var cache = new CompiledModelCache<double>();
        Tensor<double> Trace()
        {
            traceCount++;
            var prediction = engine.TensorMatMul(input, trainable);
            var error = engine.TensorSubtract(prediction, target);
            return engine.ReduceSum(engine.TensorMultiply(error, error), null);
        }

        var firstPlan = cache.GetOrCompileTraining(input._shape, Trace, new[] { trainable });
        var cachedPlan = cache.GetOrCompileTraining(input._shape, Trace, new[] { trainable });

        Assert.Same(firstPlan, cachedPlan);
        Assert.Equal(1, traceCount);
        Assert.False(trainable.IsCowShared);
        Assert.Equal(1.0, peer[0]);
    }

    [Fact]
    public void FailedTrainingTrace_LeavesPreparedParameterIsolated()
    {
        var peer = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var trainable = (Tensor<double>)peer.CloneShared();
        var parameters = new[] { trainable };

        Assert.Throws<InvalidOperationException>((Action)(() =>
        {
            using var scope = GraphMode.EnableTraining(parameters);
            throw new InvalidOperationException("synthetic trace failure");
        }));

        trainable[0] = 7.0;
        Assert.Equal(1.0, peer[0]);
    }

    [Fact]
    public void PlainGraphScope_RejectsUnpreparedCowTrainingParameter()
    {
        var engine = new CpuEngine();
        var peer = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });
        var trainable = (Tensor<double>)peer.CloneShared();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var scope = GraphMode.Enable();
        try
        {
            var output = engine.TensorMatMul(input, trainable);
            var loss = engine.ReduceSum(output, null);

            var error = Assert.Throws<InvalidOperationException>(
                () => scope.CompileTraining(new[] { trainable }, loss));
            Assert.Contains("EnableTraining", error.Message);
        }
        finally
        {
            scope.MarkCompiled();
            scope.Dispose();
        }
    }

    [Fact]
    public void DuplicateTiedParameter_IsUpdatedExactlyOnce()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
        var single = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var tied = new Tensor<double>(new[] { 1.0 }, new[] { 1 });

        using var singlePlan = CompileSquaredError(engine, input, target, single, new[] { single });
        using var tiedPlan = CompileSquaredError(engine, input, target, tied, new[] { tied, tied });
        singlePlan.ConfigureOptimizer(OptimizerType.SGD, 0.1f, 0.9f, 0.999f, 1e-8f, 0f);
        tiedPlan.ConfigureOptimizer(OptimizerType.SGD, 0.1f, 0.9f, 0.999f, 1e-8f, 0f);

        singlePlan.Step();
        tiedPlan.Step();

        Assert.Equal(single[0], tied[0], precision: 12);
    }

    [Fact]
    public void DuplicateTiedParameter_GroupedOptimizerAcceptsOriginalRegistrationMap()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
        var single = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var tied = new Tensor<double>(new[] { 1.0 }, new[] { 1 });

        using var singlePlan = CompileSquaredError(engine, input, target, single, new[] { single });
        using var tiedPlan = CompileSquaredError(engine, input, target, tied, new[] { tied, tied });
        var schedules = new[] { LrSchedule.Constant(0.1) };
        singlePlan.ConfigureOptimizerGrouped(OptimizerType.SGD, schedules, new[] { 0 });
        tiedPlan.ConfigureOptimizerGrouped(OptimizerType.SGD, schedules, new[] { 0, 0 });

        singlePlan.Step();
        tiedPlan.Step();

        Assert.Equal(single[0], tied[0], precision: 12);
    }

    [Fact]
    public void DuplicateTiedParameter_GroupedOptimizerRejectsConflictingGroups()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
        var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
        var tied = new Tensor<double>(new[] { 1.0 }, new[] { 1 });

        using var plan = CompileSquaredError(engine, input, target, tied, new[] { tied, tied });
        var error = Assert.Throws<ArgumentException>(() =>
            plan.ConfigureOptimizerGrouped(
                OptimizerType.SGD,
                new[] { LrSchedule.Constant(0.1), LrSchedule.Constant(0.01) },
                new[] { 0, 1 }));

        Assert.Contains("conflicting optimizer groups", error.Message);
    }

    [Fact]
    public async Task ConcurrentCowPeers_CompileAndTrainWithoutCrossMutation()
    {
        var untouched = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var first = (Tensor<double>)untouched.CloneShared();
        var second = (Tensor<double>)untouched.CloneShared();

        async Task TrainAsync(Tensor<double> parameter, double learningRate)
        {
            await Task.Yield();
            var engine = new CpuEngine();
            var input = new Tensor<double>(new[] { 2.0 }, new[] { 1, 1 });
            var target = new Tensor<double>(new[] { 0.0 }, new[] { 1, 1 });
            using var plan = CompileSquaredError(engine, input, target, parameter, new[] { parameter });
            plan.ConfigureOptimizer(OptimizerType.SGD, (float)learningRate, 0.9f, 0.999f, 1e-8f, 0f);
            plan.Step();
        }

        await Task.WhenAll(TrainAsync(first, 0.1), TrainAsync(second, 0.05));

        Assert.Equal(1.0, untouched[0]);
        Assert.NotEqual(1.0, first[0]);
        Assert.NotEqual(1.0, second[0]);
        Assert.NotEqual(first[0], second[0]);
    }

    [Fact]
    public void MetadataViewChain_CompiledBackwardUpdatesRegisteredCowParameter()
    {
        var engine = new CpuEngine();
        var peer = new Tensor<double>(new[] { 2.0, 3.0 }, new[] { 1, 2 });
        var parameter = (Tensor<double>)peer.CloneShared();
        var parameters = new[] { parameter };

        using var scope = GraphMode.EnableTraining(parameters);
        var expanded = parameter.ExpandDims(0);
        var squeezed = expanded.Squeeze(0);
        var permuted = squeezed.Transpose(new[] { 1, 0 });
        var loss = engine.ReduceSum(engine.TensorMultiply(permuted, permuted), null);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f);

        plan.Step();

        Assert.NotEqual(2.0, parameter[0]);
        Assert.NotEqual(3.0, parameter[1]);
        Assert.Equal(new[] { 2.0, 3.0 }, peer.ToArray());
    }

    [Fact]
    public void SliceViewChain_CompiledBackwardUpdatesOnlySelectedCowParameterRegion()
    {
        var engine = new CpuEngine();
        var peer = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 2 });
        var parameter = (Tensor<double>)peer.CloneShared();
        var parameters = new[] { parameter };

        using var scope = GraphMode.EnableTraining(parameters);
        var row = parameter.SubTensor(1);
        var selected = row.Slice(axis: 0, start: 0, end: 1);
        var loss = engine.ReduceSum(engine.TensorMultiply(selected, selected), null);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.01f, 0.9f, 0.999f, 1e-8f, 0f);

        plan.Step();

        Assert.Same(scope, GraphMode.Current);
        var compiled = Assert.IsType<CompiledTrainingPlan<double>>(plan);
        Assert.Equal(new[] { 0.0, 0.0, 6.0, 0.0 }, compiled.Gradients[0].ToArray());
        Assert.Equal(1.0, parameter[0]);
        Assert.Equal(2.0, parameter[1]);
        Assert.NotEqual(3.0, parameter[2]);
        Assert.Equal(4.0, parameter[3]);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, peer.ToArray());
    }

    private static ICompiledTrainingPlan<double> CompileSquaredError(
        CpuEngine engine,
        Tensor<double> input,
        Tensor<double> target,
        Tensor<double> parameter,
        Tensor<double>[] parameters)
    {
        using var scope = GraphMode.EnableTraining(parameters);
        var parameterView = parameter.Reshape(1, 1);
        var prediction = engine.TensorMatMul(input, parameterView);
        var error = engine.TensorSubtract(prediction, target);
        var loss = engine.ReduceSum(engine.TensorMultiply(error, error), null);
        return scope.CompileTraining(parameters, loss);
    }
}
