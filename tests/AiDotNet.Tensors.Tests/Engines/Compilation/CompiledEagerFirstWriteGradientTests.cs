using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The CPU eager compiled step does not zero a gradient buffer whose every writer is a generic
/// (AccumulateGrad) backward: its first contribution of the step is copied in under a
/// GradWriteGeneration. With the learning rate at zero the parameters never move, so every step must
/// reproduce step 0's gradients bit for bit — a skipped zeroing not covered by a first-write copy adds
/// onto the previous step's gradient and shows up from step 1 on. Step 0 (which zeroes everything) is
/// also checked against the eager tape.
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledEagerFirstWriteGradientTests
{
    private const int Batch = 4, Features = 6;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    /// <summary>Generic backwards (permute, layer norm, tanh, multiply, reduce) plus a residual
    /// (multi-consumer) tensor: MatMul → Permute → Permute → LayerNorm → + residual → Tanh → Σ s².</summary>
    private static Tensor<float> Forward(IEngine engine, Tensor<float> x, Tensor<float> w, Tensor<float> gamma, Tensor<float> beta)
    {
        var h = engine.TensorMatMul(x, w);
        var p = engine.TensorPermute(engine.TensorPermute(h, new[] { 1, 0 }), new[] { 1, 0 });
        var ln = engine.LayerNorm(p, gamma, beta, 1e-5, out _, out _);
        var r = engine.TensorAdd(ln, h);
        var s = engine.Tanh(r);
        return engine.ReduceSum(engine.TensorMultiply(s, s), null);
    }

    [Fact]
    public void GenericBackward_EveryStepReproducesStepZeroGradients_BitExact()
    {
        var engine = new CpuEngine();
        var x = Filled(new[] { Batch, Features }, 1, 2f);
        var w = Filled(new[] { Features, Features }, 2, 1f);
        var gamma = Filled(new[] { Features }, 3, 1f);
        var beta = Filled(new[] { Features }, 4, 1f);

        Tensor<float> eagerW, eagerGamma, eagerBeta;
        using (var tape = new GradientTape<float>())
        {
            var loss = Forward(engine, x, w, gamma, beta);
            var grads = tape.ComputeGradients(loss, sources: new[] { w, gamma, beta });
            eagerW = grads[w];
            eagerGamma = grads[gamma];
            eagerBeta = grads[beta];
        }

        var wF = Filled(new[] { Features, Features }, 2, 1f);
        var gammaF = Filled(new[] { Features }, 3, 1f);
        var betaF = Filled(new[] { Features }, 4, 1f);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(engine, x, wF, gammaF, betaF);
            plan = scope.CompileTraining(new[] { wF, gammaF, betaF });
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            // The graph must actually exercise the un-zeroed path, or this test proves nothing.
            Assert.True(concrete.EagerFirstWriteCandidateCount > 0,
                "graph has no generic-only gradient buffer; the first-write path is not exercised");

            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            var step0W = Snapshot(wF.Grad, "W");
            var step0Gamma = Snapshot(gammaF.Grad, "gamma");
            var step0Beta = Snapshot(betaF.Grad, "beta");
            AssertClose(eagerW, step0W, "W step 0 vs eager");
            AssertClose(eagerGamma, step0Gamma, "gamma step 0 vs eager");
            AssertClose(eagerBeta, step0Beta, "beta step 0 vs eager");

            for (int step = 1; step < 5; step++)
            {
                plan.Step();
                AssertBitEqual(step0W, wF.Grad, $"W step {step}");
                AssertBitEqual(step0Gamma, gammaF.Grad, $"gamma step {step}");
                AssertBitEqual(step0Beta, betaF.Grad, $"beta step {step}");
            }
            Assert.True(concrete.EagerFirstWriteCandidateCount > 0,
                "verification run dropped every candidate; later steps never skipped a zeroing");
        }
    }

    private static float[] Snapshot(Tensor<float>? grad, string what)
    {
        Assert.True(grad is not null, $"{what}.Grad is null");
        return grad is null ? new float[0] : grad.ToArray();
    }

    private static void AssertClose(Tensor<float> expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (System.Math.Abs(e - a) > 1e-4f * (1f + System.Math.Abs(e)))
                Assert.Fail($"{what}: element {i} eager={e:R} compiled={a:R}");
        }
    }

    private static void AssertBitEqual(float[] expected, Tensor<float>? actual, string what)
    {
        Assert.True(actual is not null, $"{what}: gradient is null");
        if (actual is null) return;
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int e = System.BitConverter.ToInt32(System.BitConverter.GetBytes(expected[i]), 0);
            int a = System.BitConverter.ToInt32(System.BitConverter.GetBytes(actual[i]), 0);
            if (e != a) Assert.Fail($"{what}: element {i} step0={expected[i]:R} now={actual[i]:R}");
        }
    }
}
