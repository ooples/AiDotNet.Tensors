using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// ForwardCSEPass runs after memory planning. A recomputation it replaces with a copy of an earlier step's output reads
/// that output later than any original reader did, after memory planning may have handed its buffer to a step in
/// between: the copy then returned that step's result.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledCseAfterMemoryPlanningTests
{
    private static Tensor<float> Forward(IEngine e, Tensor<float> x, Tensor<float> w)
    {
        var first = e.TensorMatMul(x, w);          // dies after the tanh: its buffer is free for the chain below
        var h = e.Tanh(first);
        for (int i = 0; i < 6; i++) h = e.Sigmoid(e.TensorMultiplyScalar(h, 1.5f));
        var again = e.TensorMatMul(x, w);          // the same matmul: CSE replaces it with a copy of `first`
        return e.TensorAdd(h, again);
    }

    [Fact]
    public void A_recomputation_is_not_copied_from_a_buffer_memory_planning_reused()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>([1, 8]);
        for (int i = 0; i < 8; i++) x[0, i] = i * 0.1f - 0.3f;
        var w = new Tensor<float>([8, 8]);
        for (int i = 0; i < w.Length; i++) w[i / 8, i % 8] = ((i * 7) % 11 - 5) * 0.2f;

        var expected = Forward(engine, x, w).ToArray();
        using var cache = new CompiledModelCache<float>();
        var plan = cache.GetOrCompileInference(x, () => Forward(engine, x, w));
        for (int run = 0; run < 2; run++)
        {
            var got = plan.Execute().ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(System.Math.Abs(got[i] - expected[i]) <= 1e-5f, $"run {run} [{i}] compiled {got[i]} vs eager {expected[i]}");
        }
    }
}
