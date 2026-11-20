using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// IEngine.GatedDeltaNetStep: one decode position of the gated delta rule with the state updated in place, against a
/// float64 implementation of the rule's definition, eagerly on the CPU and GPU engines and replayed by a captured plan.
/// </summary>
public class GatedDeltaNetStepTests
{
    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private sealed class Inputs
    {
        public float[] Q = Array.Empty<float>(), K = Array.Empty<float>(), V = Array.Empty<float>(), A = Array.Empty<float>(), B = Array.Empty<float>();
    }

    private static Inputs Draw(Random rng, int rows, int dv, int dk)
    {
        float[] R(int n, double lo, double hi) => Enumerable.Range(0, n).Select(_ => (float)(lo + rng.NextDouble() * (hi - lo))).ToArray();
        return new Inputs { Q = R(rows * dk, -1, 1), K = R(rows * dk, -0.5, 0.5), V = R(rows * dv, -1, 1), A = R(rows, 0.6, 1.0), B = R(rows, 0.1, 0.9) };
    }

    // The rule's definition in float64: pred = S k, delta = beta (v - pred), S = alpha S + delta k^T, o = S q.
    private static double[] Reference(double[] s, Inputs x, int rows, int dv, int dk)
    {
        var o = new double[rows * dv];
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < dv; i++)
            {
                int row = (r * dv + i) * dk;
                double pred = 0;
                for (int j = 0; j < dk; j++) pred += s[row + j] * x.K[r * dk + j];
                double delta = x.B[r] * (x.V[r * dv + i] - pred), acc = 0;
                for (int j = 0; j < dk; j++)
                {
                    s[row + j] = x.A[r] * s[row + j] + delta * x.K[r * dk + j];
                    acc += s[row + j] * x.Q[r * dk + j];
                }
                o[r * dv + i] = acc;
            }
        return o;
    }

    private static void Close(double[] expect, float[] got, string what)
    {
        Assert.Equal(expect.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-4 * (1 + Math.Abs(expect[i])), $"{what}[{i}]: {got[i]} vs {expect[i]}");
    }

    private static void RunEager(IEngine engine, int rows, int dv, int dk)
    {
        var rng = new Random(rows * 131 + dk);
        var state = new Tensor<float>([rows, dv, dk]);
        var s = new double[rows * dv * dk];
        using (new EngineScope(engine))
            for (int t = 0; t < 4; t++)
            {
                var x = Draw(rng, rows, dv, dk);
                var o = engine.GatedDeltaNetStep(state, new Tensor<float>([rows, dk], new Vector<float>(x.Q)), new Tensor<float>([rows, dk], new Vector<float>(x.K)),
                    new Tensor<float>([rows, dv], new Vector<float>(x.V)), new Tensor<float>([rows], new Vector<float>(x.A)), new Tensor<float>([rows], new Vector<float>(x.B)));
                var expect = Reference(s, x, rows, dv, dk);
                Assert.Equal(new[] { rows, dv }, o.Shape.ToArray());
                Close(expect, o.ToArray(), $"step {t} o");
                Close(s, state.ToArray(), $"step {t} state");
            }
    }

    [Theory]
    [InlineData(3, 4, 5)]
    [InlineData(8, 64, 64)]
    [InlineData(2, 7, 70)]
    public void The_cpu_step_matches_the_rule_and_updates_the_state_in_place(int rows, int dv, int dk)
        => RunEager(new CpuEngine(), rows, dv, dk);

    [SkippableTheory]
    [InlineData(3, 4, 5)]
    [InlineData(8, 64, 64)]
    [InlineData(2, 7, 70)]
    public void The_gpu_step_matches_the_rule_and_updates_the_state_in_place(int rows, int dv, int dk)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) RunEager(gpu!, rows, dv, dk);
    }

    // A compiled plan whose traced step calls GatedDeltaNetStep on a persistent state (no declared state update): every
    // execution, the capture call included, advances the state once, and the captured graph copies no state.
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_compiled_plan_advances_the_state_once_per_execution(bool capture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                const int rows = 4, dv = 16, dk = 16;
                var rng = new Random(9);
                var state = new Tensor<float>([rows, dv, dk]);
                var q = new Tensor<float>([rows, dk]); var k = new Tensor<float>([rows, dk]); var v = new Tensor<float>([rows, dv]);
                var a = new Tensor<float>([rows]); var b = new Tensor<float>([rows]);
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var o = gpu!.GatedDeltaNetStep(state, q, k, v, a, b);
                    var y = gpu.TensorMultiplyScalar(o, 2f);
                    plan = scope.CompileInference(y, new[] { q, k, v, a, b }, Array.Empty<(Tensor<float>, Tensor<float>)>());
                }
                using (plan)
                {
                    var s = new double[rows * dv * dk];
                    for (int t = 0; t < 6; t++)
                    {
                        var x = Draw(rng, rows, dv, dk);
                        plan.SetInputs(new[]
                        {
                            new Tensor<float>([rows, dk], new Vector<float>(x.Q)), new Tensor<float>([rows, dk], new Vector<float>(x.K)),
                            new Tensor<float>([rows, dv], new Vector<float>(x.V)), new Tensor<float>([rows], new Vector<float>(x.A)),
                            new Tensor<float>([rows], new Vector<float>(x.B)),
                        });
                        var y = plan.Execute().ToArray();
                        var expect = Reference(s, x, rows, dv, dk).Select(e => 2 * e).ToArray();
                        Close(expect, y, $"step {t} y");
                        Close(s, state.ToArray(), $"step {t} state");
                    }
                    if (capture && gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cuda)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                        Assert.Equal(0, cuda.LastCaptureMemcpyNodeCount);
                    }
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }
}
