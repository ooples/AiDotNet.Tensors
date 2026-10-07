using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Compiled training of CpuEngine.LstmSequenceForward (float, single output) records one fused node: the fused
/// training forward into the plan buffer and the fused BPTT backward, sharing one saved-state workspace across
/// steps. It runs exactly the arithmetic of the eager tape node, so loss and every gradient must equal the eager
/// tape bit for bit, on every step (a stale workspace or a skipped re-seed would show from step 1 on).
/// </summary>
[Collection("CompilationGlobalState")]
public class CompiledFusedLstmTrainingTests
{
    private const int Batch = 3, Seq = 5, In = 4, Hidden = 6;

    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private sealed class Inputs
    {
        public Tensor<float> X = Filled(new[] { Batch, Seq, In }, 1, 2f);
        public Tensor<float> WIh = Filled(new[] { 4 * Hidden, In }, 2, 1f);
        public Tensor<float> WHh = Filled(new[] { 4 * Hidden, Hidden }, 3, 1f);
        public Tensor<float> BIh = Filled(new[] { 4 * Hidden }, 4, 0.5f);
        public Tensor<float> BHh = Filled(new[] { 4 * Hidden }, 5, 0.5f);
        public Tensor<float> H0 = Filled(new[] { Batch, Hidden }, 6, 1f);
        public Tensor<float> C0 = Filled(new[] { Batch, Hidden }, 7, 1f);

        public Tensor<float>[] Sources(bool withState) => withState
            ? new[] { X, WIh, WHh, BIh, BHh, H0, C0 }
            : new[] { X, WIh, WHh, BIh };
    }

    private static Tensor<float> Forward(CpuEngine engine, Inputs p, bool returnSequences, bool withState)
    {
        var y = withState
            ? engine.LstmSequenceForward(p.X, p.H0, p.C0, p.WIh, p.WHh, p.BIh, p.BHh, returnSequences)
            : engine.LstmSequenceForward(p.X, null, null, p.WIh, p.WHh, p.BIh, null, returnSequences);
        var z = engine.Tanh(y);   // a consumer after the node, so its gradOutput is not just the seed
        return engine.ReduceSum(engine.TensorMultiply(z, z), null);
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void CompiledFusedLstm_EveryStepMatchesEagerTape_BitExact(bool returnSequences, bool withState)
    {
        var engine = new CpuEngine();
        var eager = new Inputs();
        var eagerSources = eager.Sources(withState);
        float eagerLoss;
        var eagerGrads = new float[eagerSources.Length][];
        using (var tape = new GradientTape<float>())
        {
            var loss = Forward(engine, eager, returnSequences, withState);
            eagerLoss = loss[0];
            var grads = tape.ComputeGradients(loss, sources: eagerSources);
            for (int i = 0; i < eagerSources.Length; i++) eagerGrads[i] = grads[eagerSources[i]].ToArray();
        }

        var compiled = new Inputs();
        var sources = compiled.Sources(withState);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(engine, compiled, returnSequences, withState);
            plan = scope.CompileTraining(sources);
        }
        using (plan)
        {
            var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
            // One fused node, not the per-timestep decomposition (which has dozens of forward steps per timestep).
            Assert.True(concrete.ForwardStepCount < 10,
                $"expected the fused LSTM node, got {concrete.ForwardStepCount} forward steps");

            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 4; step++)
            {
                var loss = plan.Step();
                AssertBitEqual(new[] { eagerLoss }, new[] { loss[0] }, $"loss step {step}");
                for (int i = 0; i < sources.Length; i++)
                {
                    var g = sources[i].Grad;
                    Assert.True(g is not null, $"source {i} has no gradient at step {step}");
                    if (g is not null) AssertBitEqual(eagerGrads[i], g.ToArray(), $"source {i} grad step {step}");
                }
            }
        }
    }

    private static int Bits(float v) => System.BitConverter.ToInt32(System.BitConverter.GetBytes(v), 0);

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (Bits(expected[i]) != Bits(actual[i]))
                Assert.Fail($"{what}: element {i} eager={expected[i]:R} compiled={actual[i]:R}");
    }
}