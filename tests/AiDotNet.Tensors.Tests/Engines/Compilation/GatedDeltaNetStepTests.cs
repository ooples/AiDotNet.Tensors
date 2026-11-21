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
[Collection("DirectGpuSerial")]
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

    // The rule's definition in float64, with k' = c k: pred = d S k', delta = beta (v - pred), S = alpha S + delta k'^T,
    // o = S q; Standard (the paper) has c = 1, d = alpha, ScaledKeys c = 1/sqrt(dk), d = 1.
    private static double[] Reference(double[] s, Inputs x, int rows, int dv, int dk, GatedDeltaRule rule)
    {
        double c = rule == GatedDeltaRule.Standard ? 1.0 : 1.0 / Math.Sqrt(dk);
        var o = new double[rows * dv];
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < dv; i++)
            {
                int row = (r * dv + i) * dk;
                double pred = 0;
                for (int j = 0; j < dk; j++) pred += s[row + j] * c * x.K[r * dk + j];
                if (rule == GatedDeltaRule.Standard) pred *= x.A[r];
                double delta = x.B[r] * (x.V[r * dv + i] - pred), acc = 0;
                for (int j = 0; j < dk; j++)
                {
                    s[row + j] = x.A[r] * s[row + j] + delta * c * x.K[r * dk + j];
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

    private static void RunEager(IEngine engine, int rows, int dv, int dk, GatedDeltaRule rule)
    {
        var rng = new Random(rows * 131 + dk);
        var state = new Tensor<float>([rows, dv, dk]);
        var s = new double[rows * dv * dk];
        using (new EngineScope(engine))
            for (int t = 0; t < 4; t++)
            {
                var x = Draw(rng, rows, dv, dk);
                var o = engine.GatedDeltaNetStep(state, new Tensor<float>([rows, dk], new Vector<float>(x.Q)), new Tensor<float>([rows, dk], new Vector<float>(x.K)),
                    new Tensor<float>([rows, dv], new Vector<float>(x.V)), new Tensor<float>([rows], new Vector<float>(x.A)), new Tensor<float>([rows], new Vector<float>(x.B)),
                    rule);
                var expect = Reference(s, x, rows, dv, dk, rule);
                Assert.Equal(new[] { rows, dv }, o.Shape.ToArray());
                Close(expect, o.ToArray(), $"step {t} o");
                Close(s, state.ToArray(), $"step {t} state");
            }
    }

    [Theory]
    [InlineData(3, 4, 5, GatedDeltaRule.Standard)]
    [InlineData(8, 64, 64, GatedDeltaRule.Standard)]
    [InlineData(2, 7, 70, GatedDeltaRule.Standard)]
    [InlineData(3, 4, 5, GatedDeltaRule.ScaledKeys)]
    [InlineData(8, 64, 64, GatedDeltaRule.ScaledKeys)]
    public void The_cpu_step_matches_the_rule_and_updates_the_state_in_place(int rows, int dv, int dk, GatedDeltaRule rule)
        => RunEager(new CpuEngine(), rows, dv, dk, rule);

    [SkippableTheory]
    [InlineData(3, 4, 5, GatedDeltaRule.Standard)]
    [InlineData(8, 64, 64, GatedDeltaRule.Standard)]
    [InlineData(2, 7, 70, GatedDeltaRule.Standard)]
    [InlineData(3, 4, 5, GatedDeltaRule.ScaledKeys)]
    [InlineData(8, 64, 64, GatedDeltaRule.ScaledKeys)]
    public void The_gpu_step_matches_the_rule_and_updates_the_state_in_place(int rows, int dv, int dk, GatedDeltaRule rule)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) RunEager(gpu!, rows, dv, dk, rule);
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
                    var o = gpu!.GatedDeltaNetStep(state, q, k, v, a, b, GatedDeltaRule.Standard);
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
                        var expect = Reference(s, x, rows, dv, dk, GatedDeltaRule.Standard).Select(e => 2 * e).ToArray();
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


    // A decode under a rule reproduces that rule's scan: T steps from a zero state give the scan's T outputs, per head.
    [Theory]
    [InlineData(GatedDeltaRule.Standard)]
    [InlineData(GatedDeltaRule.ScaledKeys)]
    public void Steps_reproduce_the_scan_under_the_same_rule(GatedDeltaRule rule)
    {
        const int batch = 2, seqLen = 7, heads = 3, d = 8, model = heads * d;
        var engine = new CpuEngine();
        var rng = new Random(77);
        float[] R(int n, double lo, double hi) => Enumerable.Range(0, n).Select(_ => (float)(lo + rng.NextDouble() * (hi - lo))).ToArray();
        float[] q = R(batch * seqLen * model, -1, 1), k = R(batch * seqLen * model, -0.5, 0.5), v = R(batch * seqLen * model, -1, 1);
        float[] a = R(batch * seqLen * heads, 0.6, 1.0), b = R(batch * seqLen * heads, 0.1, 0.9);
        var scan = engine.GatedDeltaNetScanForward(new Tensor<float>(q, [batch, seqLen, model]), new Tensor<float>(k, [batch, seqLen, model]),
            new Tensor<float>(v, [batch, seqLen, model]), new Tensor<float>(a, [batch, seqLen, heads]), new Tensor<float>(b, [batch, seqLen, heads]),
            heads, rule).ToArray();
        // rows = batch * heads, each row a [d, d] state; step t takes position t of every (batch, head).
        var state = new Tensor<float>([batch * heads, d, d]);
        for (int t = 0; t < seqLen; t++)
        {
            float[] Pos(float[] x, int width) => Enumerable.Range(0, batch * heads * width)
                .Select(i => { int row = i / width, j = i % width, bb = row / heads, h = row % heads; return x[(bb * seqLen + t) * heads * width + h * width + j]; })
                .ToArray();
            float[] Gate(float[] x) => Enumerable.Range(0, batch * heads).Select(row => x[(row / heads * seqLen + t) * heads + row % heads]).ToArray();
            var o = engine.GatedDeltaNetStep(state, new Tensor<float>(Pos(q, d), [batch * heads, d]), new Tensor<float>(Pos(k, d), [batch * heads, d]),
                new Tensor<float>(Pos(v, d), [batch * heads, d]), new Tensor<float>(Gate(a), [batch * heads]), new Tensor<float>(Gate(b), [batch * heads]),
                rule).ToArray();
            for (int row = 0; row < batch * heads; row++)
                for (int j = 0; j < d; j++)
                {
                    int bb = row / heads, h = row % heads;
                    float expect = scan[(bb * seqLen + t) * model + h * d + j], got = o[row * d + j];
                    Assert.True(Math.Abs(got - expect) <= 1e-5f * (1 + Math.Abs(expect)), $"{rule} t={t} row={row} j={j}: step {got} vs scan {expect}");
                }
        }
    }
}
