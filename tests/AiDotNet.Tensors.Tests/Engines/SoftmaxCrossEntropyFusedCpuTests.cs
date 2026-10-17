using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The host float32 TensorSoftmaxCrossEntropy is one fused op (CpuEngine.SoftmaxCrossEntropy.cs): its loss and both
/// gradients must equal the composed objective -- -(sum of t * log_softmax(z)) / max(1, rows with target mass), the
/// count detached -- computed here independently in double precision, on the eager tape and in a compiled plan.
/// </summary>
[Collection("CompilationGlobalState")]
public class SoftmaxCrossEntropyFusedCpuTests
{
    private const int Rows = 7, Cols = 37;   // 37 = four 8-lane vectors and a 5-lane tail

    private static float[] Logits(int seed)
    {
        var rng = new Random(seed);
        var z = new float[Rows * Cols];
        for (int i = 0; i < z.Length; i++) z[i] = (float)((rng.NextDouble() * 2 - 1) * 6);
        return z;
    }

    /// <summary>Row 0 one-hot, row 1 ignored (all zero), row 2 a weighted soft target with a negative entry, row 3
    /// signed weights that cancel to zero mass (still supervised: absolute mass is non-zero), the rest soft.</summary>
    private static float[] Targets(int seed)
    {
        var rng = new Random(seed);
        var t = new float[Rows * Cols];
        t[3] = 1f;
        for (int c = 0; c < Cols; c++) t[2 * Cols + c] = (float)(rng.NextDouble() * 0.5) - (c == 4 ? 0.3f : 0f);
        t[3 * Cols + 1] = 0.75f;
        t[3 * Cols + 9] = -0.75f;
        for (int r = 4; r < Rows; r++)
            for (int c = 0; c < Cols; c++) t[r * Cols + c] = (float)rng.NextDouble();
        return t;
    }

    private static (double Loss, double[] DZ, double[] DT) Reference(float[] z, float[] t)
    {
        double total = 0;
        int count = 0;
        var lse = new double[Rows];
        for (int r = 0; r < Rows; r++)
        {
            double max = double.NegativeInfinity, abs = 0;
            for (int c = 0; c < Cols; c++) max = Math.Max(max, z[r * Cols + c]);
            double sum = 0;
            for (int c = 0; c < Cols; c++) sum += Math.Exp(z[r * Cols + c] - max);
            lse[r] = max + Math.Log(sum);
            for (int c = 0; c < Cols; c++)
            {
                total += t[r * Cols + c] * (z[r * Cols + c] - lse[r]);
                abs += Math.Abs(t[r * Cols + c]);
            }
            if (abs > 0) count++;
        }
        double n = Math.Max(1, count);
        var dz = new double[Rows * Cols];
        var dt = new double[Rows * Cols];
        for (int r = 0; r < Rows; r++)
        {
            double mass = 0;
            for (int c = 0; c < Cols; c++) mass += t[r * Cols + c];
            for (int c = 0; c < Cols; c++)
            {
                int i = r * Cols + c;
                dz[i] = (mass * Math.Exp(z[i] - lse[r]) - t[i]) / n;
                dt[i] = -(z[i] - lse[r]) / n;
            }
        }
        return (-total / n, dz, dt);
    }

    private static void AssertClose(double[] expected, Tensor<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 2e-6 + 2e-5 * Math.Abs(expected[i]),
                $"{what}[{i}] = {actual[i]:G9}, expected {expected[i]:G9}");
    }

    [Fact]
    public void EagerTape_LossAndBothGradients_MatchTheComposedObjective()
    {
        var engine = new CpuEngine();
        float[] zData = Logits(1), tData = Targets(2);
        var (loss, dz, dt) = Reference(zData, tData);
        var z = new Tensor<float>(zData, new[] { Rows, Cols });
        var t = new Tensor<float>(tData, new[] { Rows, Cols });

        using var tape = new GradientTape<float>();
        var l = engine.TensorSoftmaxCrossEntropy(z, t);
        var grads = tape.ComputeGradients(l, new[] { z, t });

        Assert.Equal(0, l.Rank);
        Assert.True(Math.Abs(l[0] - loss) <= 1e-5 * Math.Abs(loss), $"loss {l[0]:G9}, expected {loss:G9}");
        AssertClose(dz, grads[z], "dL/dz");
        AssertClose(dt, grads[t], "dL/dt");
    }

    [Fact]
    public void EagerTape_UpstreamGradientScalesTheLogitsGradient()
    {
        // loss * 3: the backward must read gradOutput, not assume 1.
        var engine = new CpuEngine();
        float[] zData = Logits(3), tData = Targets(4);
        var (_, dz, _) = Reference(zData, tData);
        var z = new Tensor<float>(zData, new[] { Rows, Cols });
        var t = new Tensor<float>(tData, new[] { Rows, Cols });

        using var tape = new GradientTape<float>();
        var l = engine.TensorMultiplyScalar(engine.TensorSoftmaxCrossEntropy(z, t), 3f);
        var grads = tape.ComputeGradients(l, new[] { z });

        for (int i = 0; i < dz.Length; i++) dz[i] *= 3;
        AssertClose(dz, grads[z], "dL/dz with upstream 3");
    }

    [Fact]
    public void AllRowsIgnored_LossIsZeroAndGradientIsZero()
    {
        var engine = new CpuEngine();
        var z = new Tensor<float>(Logits(5), new[] { Rows, Cols });
        var t = new Tensor<float>(new[] { Rows, Cols });

        using var tape = new GradientTape<float>();
        var l = engine.TensorSoftmaxCrossEntropy(z, t);
        var grads = tape.ComputeGradients(l, new[] { z });

        Assert.Equal(0f, l[0]);
        Assert.All(grads[z].ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void NaNTargetInOneRow_MakesTheCountNaN_SoEveryRowsGradientIsNaN()
    {
        // The composed count is sum(sign(sum|t|)) and sign(NaN) is NaN, so one NaN target poisons the divisor and with
        // it every row's gradient -- not only the NaN row's. A count that skipped the NaN row would leave row 0 finite.
        var engine = new CpuEngine();
        var tData = Targets(6);
        tData[5 * Cols + 2] = float.NaN;
        var z = new Tensor<float>(Logits(7), new[] { Rows, Cols });
        var t = new Tensor<float>(tData, new[] { Rows, Cols });

        using var tape = new GradientTape<float>();
        var l = engine.TensorSoftmaxCrossEntropy(z, t);
        var grads = tape.ComputeGradients(l, new[] { z });

        Assert.True(float.IsNaN(l[0]), $"loss {l[0]}");
        for (int c = 0; c < Cols; c++) Assert.True(float.IsNaN(grads[z][c]), $"dL/dz[0,{c}] = {grads[z][c]}");
    }

    [Fact]
    public void CompiledPlan_EveryReplayMatchesTheComposedObjectiveForThatStepsBatch()
    {
        // A compiled step reuses one forward state across replays; each step feeds a new batch into the same input
        // buffers, so the loss and gradient must follow the batch, not the trace's.
        var engine = new CpuEngine();
        var z = new Tensor<float>(Logits(10), new[] { Rows, Cols });
        var t = new Tensor<float>(Targets(11), new[] { Rows, Cols });
        var scale = new Tensor<float>(new[] { 1f }, new[] { 1 });

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorSoftmaxCrossEntropy(engine.TensorMultiply(z, engine.TensorBroadcastTo(scale, new[] { Rows, Cols })), t);
            plan = scope.CompileTraining(new[] { scale });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 3; step++)
            {
                float[] zData = Logits(20 + step), tData = Targets(30 + step);
                zData.AsSpan().CopyTo(z.AsWritableSpan());
                tData.AsSpan().CopyTo(t.AsWritableSpan());
                var (loss, dz, _) = Reference(zData, tData);
                float got = plan.Step()[0];
                Assert.True(Math.Abs(got - loss) <= 1e-5 * Math.Abs(loss), $"step {step}: loss {got:G9}, expected {loss:G9}");
                // dL/dscale = sum_i z_i * dL/d(scaled z)_i at scale 1.
                double expected = 0;
                for (int i = 0; i < dz.Length; i++) expected += zData[i] * dz[i];
                var g = scale.Grad ?? throw new InvalidOperationException("scale has no gradient");
                Assert.True(Math.Abs(g[0] - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"step {step}: dL/dscale {g[0]:G9}, expected {expected:G9}");
            }
        }
    }
}
