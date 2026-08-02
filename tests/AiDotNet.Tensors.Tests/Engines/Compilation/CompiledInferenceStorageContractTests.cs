using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class CompiledInferenceStorageContractTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledInferenceStorageContractTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    [Fact]
    public void Transpose_OffsetInputAndPaddedOutput_UsesLiveLogicalStorageAcrossReplay()
    {
        var engine = new CpuEngine();
        var inputBacking = new[] { 777f, 1f, 2f, 3f, 4f, 5f, 6f, 888f };
        var input = Tensor<float>.FromMemory(
            new Memory<float>(inputBacking, 1, 6), new[] { 2, 3 });

        var outputBacking = new float[12];
        for (int i = 0; i < outputBacking.Length; i++)
            outputBacking[i] = 999f;
        var outputStorage = Tensor<float>.FromMemory(
            new Memory<float>(outputBacking, 2, 6), new[] { 3, 2 });

        ICompiledPlan<float> plan;
        Tensor<float> capturedOutput;
        using (var scope = GraphMode.Enable())
        {
            capturedOutput = engine.TensorTranspose(input);
            // Force the plan's final output onto an offset, padded backing before
            // specialization. Legacy GetDataArray() wrote a detached six-element
            // copy and left this storage unchanged.
            capturedOutput.RebindStorageFrom(outputStorage);
            plan = scope.CompileInference<float>();
        }

        using (plan)
        {
            var first = plan.Execute();
            Assert.Equal(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, first.AsSpan().ToArray());
            AssertSentinels(outputBacking);

            input[0] = 10f;
            input[1] = 20f;
            input[2] = 30f;
            input[3] = 40f;
            input[4] = 50f;
            input[5] = 60f;

            var second = plan.Execute();
            Assert.Equal(new[] { 10f, 40f, 20f, 50f, 30f, 60f }, second.AsSpan().ToArray());
            AssertSentinels(outputBacking);
        }
    }

    private static void AssertSentinels(float[] backing)
    {
        Assert.Equal(999f, backing[0]);
        Assert.Equal(999f, backing[1]);
        for (int i = 8; i < backing.Length; i++)
            Assert.Equal(999f, backing[i]);
    }
}
