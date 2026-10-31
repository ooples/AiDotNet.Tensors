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

    /// <summary>A pre-norm attention-like block: h = x·W0 feeds both the norm and the residual add, and the norm's output
    /// n feeds three projections (q, k, v) -- the two multi-consumer gradients whose writers include specialized
    /// delegates (the residual add, the shared-operand matmuls).</summary>
    private static Tensor<float> ResidualForward(IEngine engine, Tensor<float> x, Tensor<float> w0, Tensor<float> gamma,
        Tensor<float> beta, Tensor<float> wq, Tensor<float> wk, Tensor<float> wv)
    {
        var h = engine.TensorMatMul(x, w0);
        var n = engine.LayerNorm(h, gamma, beta, 1e-5, out _, out _);
        var q = engine.TensorMatMul(n, wq);
        var k = engine.TensorMatMul(n, wk);
        var v = engine.TensorMatMul(n, wv);
        var a = engine.TensorAdd(engine.TensorMultiply(q, k), v);
        var s = engine.Tanh(engine.TensorAdd(h, a));
        return engine.ReduceSum(engine.TensorMultiply(s, s), null);
    }

    /// <summary>
    /// The residual add, the shared-operand matmuls and the multiply (s·s reads s twice) claim the first write of a
    /// multi-consumer gradient (overwrite, later writers add), so the step zeroes none of those gradients. Every
    /// step must still reproduce step 0's gradients bit for bit, and the zeroing pass must clear nothing.
    /// </summary>
    [Fact]
    public void ClaimingSpecializedWriters_SkipZeroingAndReproduceStepZeroGradients_BitExact()
    {
        const int Rows = 8, Dim = 6;
        var engine = new CpuEngine();
        var x = Filled(new[] { Rows, Dim }, 11, 2f);
        Tensor<float>[] MakeParams() => new[]
        {
            Filled(new[] { Dim, Dim }, 12, 1f), Filled(new[] { Dim }, 13, 1f), Filled(new[] { Dim }, 14, 1f),
            Filled(new[] { Dim, Dim }, 15, 1f), Filled(new[] { Dim, Dim }, 16, 1f), Filled(new[] { Dim, Dim }, 17, 1f),
        };
        string[] names = { "W0", "gamma", "beta", "Wq", "Wk", "Wv" };

        var eagerParams = MakeParams();
        var eager = new Tensor<float>[eagerParams.Length];
        using (var tape = new GradientTape<float>())
        {
            var p = eagerParams;
            var loss = ResidualForward(engine, x, p[0], p[1], p[2], p[3], p[4], p[5]);
            var grads = tape.ComputeGradients(loss, sources: p);
            for (int i = 0; i < p.Length; i++) eager[i] = grads[p[i]];
        }

        var ps = MakeParams();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            ResidualForward(engine, x, ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]);
            plan = scope.CompileTraining(ps);
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            // The claiming delegates must actually be in the plan, or this test proves nothing.
            Assert.Contains("specialized:TensorMatMul", concrete.BackwardStepKinds);
            Assert.Contains("specialized:TensorAdd", concrete.BackwardStepKinds);

            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            var step0 = new float[ps.Length][];
            for (int i = 0; i < ps.Length; i++)
            {
                step0[i] = Snapshot(ps[i].Grad, names[i]);
                AssertClose(eager[i], step0[i], $"{names[i]} step 0 vs eager");
            }

            for (int step = 1; step < 5; step++)
            {
                plan.Step();
                for (int i = 0; i < ps.Length; i++)
                    AssertBitEqual(step0[i], ps[i].Grad, $"{names[i]} step {step}");
                if (step >= 2)
                    Assert.Equal(0L, concrete.LastClearedGradElements);
            }
        }
    }

    /// <summary>
    /// DenseNet's pattern: a training BatchNorm whose input a later concatenation also reads, so the BatchNorm's
    /// specialized backward and a generic backward both write that input's gradient. The BatchNorm backward claims its
    /// first write, and the BatchNorm's batch-statistic nodes (no backward) do not count as writers, so after the
    /// verification step nothing is zeroed and every step still reproduces step 0's gradients bit for bit. A tanh follows
    /// the BatchNorm so it is not fused with a ReLU.
    /// </summary>
    [Fact]
    public void TrainingBatchNorm_SharedInput_NeedsNoZeroing_AndStaysBitExact()
        => AssertSharedInputBatchNorm(useRelu: false, "specialized:BatchNorm");

    /// <summary>
    /// The same graph with a ReLU after the BatchNorm: on the CPU the pair fuses (BatchNormReluFusionPattern), so the
    /// BatchNorm output and its gradient have no buffer, and the fused backward masks the gradient inside the batch-norm
    /// kernel. Its gradients must match the eager tape, need no zeroing, and repeat bit for bit.
    /// </summary>
    [Fact]
    public void FusedTrainingBatchNormRelu_SharedInput_MatchesEager_NeedsNoZeroing()
        => AssertSharedInputBatchNorm(useRelu: true, "specialized:BatchNormReLU");

    private static void AssertSharedInputBatchNorm(bool useRelu, string expectedKind)
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 4, 3, 3 }, 21, 2f);
        Tensor<float>[] MakeParams() => new[]
        {
            Filled(new[] { 2, 4, 3, 3 }, 22, 1f), Filled(new[] { 4 }, 23, 1f), Filled(new[] { 4 }, 24, 1f),
        };
        string[] names = { "scale", "gamma", "beta" };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            var x = engine.TensorMultiply(input, p[0]);
            var y = engine.BatchNorm(x, p[1], p[2], 1e-5, out _, out _);
            var a = useRelu ? engine.ReLU(y) : engine.Tanh(y);
            var z = engine.TensorConcatenate(new[] { a, x }, 1);   // x: read by the BatchNorm and this concat
            return engine.ReduceSum(engine.TensorMultiply(z, z), null);
        }

        var eagerParams = MakeParams();
        var eager = new Tensor<float>[eagerParams.Length];
        using (var tape = new GradientTape<float>())
        {
            var grads = tape.ComputeGradients(Forward(eagerParams), sources: eagerParams);
            for (int i = 0; i < eagerParams.Length; i++) eager[i] = grads[eagerParams[i]];
        }

        var ps = MakeParams();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(ps);
            plan = scope.CompileTraining(ps);
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            Assert.Contains(expectedKind, concrete.BackwardStepKinds);

            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            var step0 = new float[ps.Length][];
            for (int i = 0; i < ps.Length; i++)
            {
                step0[i] = Snapshot(ps[i].Grad, names[i]);
                AssertClose(eager[i], step0[i], $"{names[i]} step 0 vs eager");
            }

            for (int step = 1; step < 5; step++)
            {
                plan.Step();
                for (int i = 0; i < ps.Length; i++)
                    AssertBitEqual(step0[i], ps[i].Grad, $"{names[i]} step {step}");
                if (step >= 2)
                    Assert.Equal(0L, concrete.LastClearedGradElements);
            }
        }
    }
    /// <summary>
    /// A training BatchNorm followed by a pool: the pool is the sole writer of the BatchNorm output's gradient, but the
    /// BatchNorm's batch-statistic nodes also read that output. The pool's specialized backward claims the gradient's
    /// first write, so it must overwrite it: counting the statistic readers as consumers made it add into a buffer nobody
    /// zeroed, and the gradients grew every step.
    /// </summary>
    [Theory]
    [InlineData("MaxPool2D")]
    [InlineData("AvgPool2D")]
    public void PoolAfterTrainingBatchNorm_OverwritesItsClaimedGradient_BitExact(string pool)
    {
        var engine = new CpuEngine();
        var input = Filled(new[] { 2, 4, 4, 4 }, 31, 2f);
        Tensor<float>[] MakeParams() => new[]
        {
            Filled(new[] { 2, 4, 4, 4 }, 32, 1f), Filled(new[] { 4 }, 33, 1f), Filled(new[] { 4 }, 34, 1f),
        };
        string[] names = { "scale", "gamma", "beta" };
        Tensor<float> Forward(Tensor<float>[] p)
        {
            var x = engine.TensorMultiply(input, p[0]);
            var y = engine.BatchNorm(x, p[1], p[2], 1e-5, out _, out _);
            var z = pool == "MaxPool2D" ? engine.MaxPool2D(y, 2, 2) : engine.AvgPool2D(y, 2, 2);
            return engine.ReduceSum(engine.TensorMultiply(z, z), null);
        }

        var eagerParams = MakeParams();
        var eager = new Tensor<float>[eagerParams.Length];
        using (var tape = new GradientTape<float>())
        {
            var grads = tape.ComputeGradients(Forward(eagerParams), sources: eagerParams);
            for (int i = 0; i < eagerParams.Length; i++) eager[i] = grads[eagerParams[i]];
        }

        var ps = MakeParams();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(ps);
            plan = scope.CompileTraining(ps);
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            Assert.Contains("specialized:" + pool, concrete.BackwardStepKinds);

            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            var step0 = new float[ps.Length][];
            for (int i = 0; i < ps.Length; i++)
            {
                step0[i] = Snapshot(ps[i].Grad, names[i]);
                AssertClose(eager[i], step0[i], $"{names[i]} step 0 vs eager");
            }

            for (int step = 1; step < 5; step++)
            {
                plan.Step();
                for (int i = 0; i < ps.Length; i++)
                    AssertBitEqual(step0[i], ps[i].Grad, $"{names[i]} step {step}");
                if (step >= 2)
                    Assert.Equal(0L, concrete.LastClearedGradElements);
            }
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
