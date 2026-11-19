using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// PointwiseFusionPass fuses a chain of unary steps into one. It must stop at an intermediate that another step also
/// reads: fused through it, that reader saw a tensor nothing wrote any more.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledPointwiseFusionReadersTests
{
    private static Tensor<float> Forward(IEngine e, Tensor<float> x, Tensor<float> w)
    {
        var h = e.Tanh(e.TensorMatMul(e.Tanh(e.TensorMatMul(x, w)), w));
        var y = e.Sigmoid(h);                 // read by the Tanh below AND by the matmul after it
        var z = e.Tanh(y);
        var u = e.TensorMatMul(y, w);
        return e.TensorAdd(u, z);
    }

    [Fact]
    public void A_unary_chain_is_not_fused_through_an_intermediate_with_other_readers()
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
