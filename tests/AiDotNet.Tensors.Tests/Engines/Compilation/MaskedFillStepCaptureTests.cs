using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A boolean-masked fill (the causal mask of an attention layer) inside a compiled training step must not stop the
/// step from capturing as a CUDA graph, and the replayed steps must train exactly as the eager steps do -- including
/// after the caller rewrites the mask between steps (the graph reads the mask's device copy through a baked pointer).
/// Before the fix the GPU masked fill uploaded the host mask on every call: a synchronous copy inside the capture, which
/// failed it ("cuStreamSynchronize(weight upload) failed", CUDA error 900), so the step never replayed.
/// </summary>
[Collection("DirectGpuSerial")]
public class MaskedFillStepCaptureTests
{
    private const int Rows = 8, Cols = 8, Steps = 10, StepsAfterRewrite = 3;

    [SkippableFact]
    public void ACausalMaskedFillInTheStep_CapturesAndReplays_AndMatchesTheEagerTrajectory()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "No CUDA device.");
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
            var xs = Enumerable.Range(0, Rows * Cols).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
            var w0 = Enumerable.Range(0, Cols * Cols).Select(_ => (float)(rng.NextDouble() - 0.5) * 0.5f).ToArray();
            // Upper triangle (j > i) masked, as a causal attention mask.
            var mask = new Tensor<bool>(new[] { Rows, Cols });
            for (int i = 0; i < Rows; i++) for (int j = 0; j < Cols; j++) mask[i, j] = j > i;

            var x = new Tensor<float>(new[] { Rows, Cols }, new Vector<float>(xs));
            var w = new Tensor<float>(new[] { Cols, Cols }, new Vector<float>(w0));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var scores = gpu.TensorMaskedFill(gpu.TensorMatMul(x, w), mask, -4f);
                var p = gpu.Softmax(scores, -1);
                gpu.ReduceSum(gpu.TensorMultiply(p, p), null);
                plan = scope.CompileTraining(new[] { w });
            }
            var replaysProperty = typeof(CompiledTrainingPlan<float>).GetField("_planGraphReplayCount",
                BindingFlags.NonPublic | BindingFlags.Instance) ?? throw new InvalidOperationException("no replay counter");
            const int total = Steps + StepsAfterRewrite;
            var losses = new float[total];
            long replays;
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.5);
                for (int s = 0; s < Steps; s++) losses[s] = plan.Step().ToArray()[0];
                mask[Rows - 1, 0] = true;   // the caller rewrites the mask; the next replays must see it
                for (int s = Steps; s < total; s++) losses[s] = plan.Step().ToArray()[0];
                replays = (long)TestRequire.NotNull(replaysProperty.GetValue(plan), "the replay count");
            }

            // Reference: the same SGD steps on the host, in double.
            var wr = w0.Select(v => (double)v).ToArray();
            var reference = new double[total];
            for (int s = 0; s < total; s++)
            {
                bool Masked(int i, int j) => j > i || (s >= Steps && i == Rows - 1 && j == 0);
                double loss = 0;
                var g = new double[Cols * Cols];
                for (int i = 0; i < Rows; i++)
                {
                    var z = new double[Cols];
                    for (int j = 0; j < Cols; j++)
                    {
                        double v = 0;
                        for (int k = 0; k < Cols; k++) v += xs[i * Cols + k] * wr[k * Cols + j];
                        z[j] = Masked(i, j) ? -4.0 : v;
                    }
                    double m = z.Max(), sum = z.Sum(v => Math.Exp(v - m));
                    var p = z.Select(v => Math.Exp(v - m) / sum).ToArray();
                    loss += p.Sum(v => v * v);
                    // d(sum p^2)/dz_j = 2 p_j (p_j - sum_k p_k^2); masked entries get no gradient.
                    double pp = p.Sum(v => v * v);
                    for (int j = 0; j < Cols; j++)
                    {
                        if (Masked(i, j)) continue;
                        double dz = 2 * p[j] * (p[j] - pp);
                        for (int k = 0; k < Cols; k++) g[k * Cols + j] += xs[i * Cols + k] * dz;
                    }
                }
                reference[s] = loss;
                for (int q = 0; q < wr.Length; q++) wr[q] -= 0.5 * g[q];
            }
            double worst = losses.Zip(reference, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"replays={replays} losses={string.Join(",", losses.Select(v => v.ToString("G6")))} worst={worst:G3}");
            // Measured 5.8e-7 (fp32 against the double reference). A backward that kept the traced mask after the rewrite
            // measured 3.5e-5, so the bound separates the two.
            Assert.True(worst < 5e-6, $"the compiled step diverged from the host reference by {worst}");
            // 3 eager warm-up steps and the capture step; every later step replays the graph.
            Assert.True(replays == total - 4, $"the step captured and replayed {replays} times of {total}");
        }
    }
}
