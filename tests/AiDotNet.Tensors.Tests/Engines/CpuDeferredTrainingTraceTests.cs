using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Perf-Serial")]
public class CpuDeferredTrainingTraceTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Compiled_replay_preserves_views_gradients_and_live_parameter_updates(bool defer)
    {
        bool previous = LazyTensorScope.DeferCpuTrainingOutputs;
        var engine = AiDotNetEngine.Current;
        try
        {
            LazyTensorScope.DeferCpuTrainingOutputs = defer;
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            using var parameter = new Tensor<float>(new[] { 1f, 2f, -3f }, new[] { 1, 3 });
            parameter.RequiresGrad = true;
            using var coefficients = new Tensor<float>(new[] { .5f, -1f, 2f }, new[] { 3, 1 });
            using var scope = GraphMode.EnableTraining(new[] { parameter });
            var scaled = cpu.TensorMultiplyScalar(parameter, 2f);
            var view = cpu.Reshape(cpu.Reshape(scaled, new[] { 3 }), new[] { 1, 3 });
            var loss = cpu.TensorMatMul(view, coefficients);
            using var plan = scope.CompileTraining(new[] { parameter }, loss);
            Assert.Equal(-15f, plan.Step().ToArray()[0]);
            Assert.NotNull(parameter.Grad);
            Assert.Equal(new[] { 1f, -2f, 4f }, parameter.Grad!.ToArray());
            parameter[0, 0] = 5f;
            Assert.Equal(-11f, plan.Step().ToArray()[0]);
            Assert.Equal(new[] { 1f, -2f, 4f }, parameter.Grad!.ToArray());
        }
        finally
        {
            LazyTensorScope.DeferCpuTrainingOutputs = previous;
            AiDotNetEngine.Current = engine;
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Host_read_realizes_the_recorded_training_value_and_preserves_its_input(bool defer)
    {
        bool previous = LazyTensorScope.DeferCpuTrainingOutputs;
        var engine = AiDotNetEngine.Current;
        try
        {
            LazyTensorScope.DeferCpuTrainingOutputs = defer;
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            using var input = new Tensor<float>(new[] { 1, 65536 });
            input[0, 0] = 3f;
            input[0, 65535] = -7f;
            using var scope = GraphMode.EnableTraining(new[] { input });
            using var output = cpu.TensorMultiplyScalar(input, 2f);
            Assert.Equal(defer, output.HasDeferredHostStorage);
            Assert.Equal(6f, output[0, 0]);
            Assert.Equal(-14f, output[0, 65535]);
            Assert.False(output.HasDeferredHostStorage);
            Assert.Equal(3f, input[0, 0]);
            Assert.Equal(-7f, input[0, 65535]);
        }
        finally
        {
            LazyTensorScope.DeferCpuTrainingOutputs = previous;
            AiDotNetEngine.Current = engine;
        }
    }

    [Fact]
    public void Compatibility_trace_keeps_its_existing_allocation_contract()
    {
        bool previous = LazyTensorScope.DeferCpuTrainingOutputs;
        var engine = AiDotNetEngine.Current;
        try
        {
            LazyTensorScope.DeferCpuTrainingOutputs = true;
            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            using var input = new Tensor<double>(new[] { 32 });
            input[0] = 5d;
            using var scope = GraphMode.Enable();
            using var output = cpu.TensorMultiplyScalar(input, 3d);
            Assert.False(output.HasDeferredHostStorage);
            Assert.Equal(15d, output[0]);
        }
        finally
        {
            LazyTensorScope.DeferCpuTrainingOutputs = previous;
            AiDotNetEngine.Current = engine;
        }
    }
}
