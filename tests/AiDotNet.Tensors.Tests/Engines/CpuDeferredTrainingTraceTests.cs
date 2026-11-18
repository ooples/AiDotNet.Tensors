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
