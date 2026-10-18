using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The compiled (graph-mode) float LayerNorm node normalizes into the plan's buffer and writes dX, dGamma and dBeta
/// straight into the gradient accumulators. Checked against a double-precision transcription, on every step of a
/// compiled plan (the buffers are reused, so a backward that adds onto the previous step's gradient shows on step 2).
/// </summary>
public class LayerNormCompiledTests
{
    public static TheoryData<int[], int[]> Shapes => new()
    {
        // input shape, gamma shape
        { new[] { 64, 32, 64 }, new[] { 64 } },
        { new[] { 3, 7, 37 }, new[] { 37 } },        // features not a multiple of the vector width, few rows
        { new[] { 1000, 24 }, new[] { 24 } },        // rows not divisible by the chunk count
        { new[] { 6, 2, 5 }, new[] { 2, 5 } },       // two normalized dimensions
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public async Task CompiledPlan_ForwardAndGradients_MatchReference_OnEveryStep(int[] inputShape, int[] gammaShape)
    {
        await Task.Yield();
        var rng = new Random(11);
        Tensor<float> Make(int[] shape, double offset)
        {
            int n = 1; foreach (var d in shape) n *= d;
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(offset + rng.NextDouble() * 2 - 1);
            return new Tensor<float>(a, shape);
        }
        var input = Make(inputShape, 0.3);
        var gamma = Make(gammaShape, 1.0);
        var beta = Make(gammaShape, 0.0);
        var upstream = Make(inputShape, 0.0);
        const double eps = 1e-5;
        var engine = new CpuEngine();

        Reference(input.ToArray(), gamma.ToArray(), beta.ToArray(), upstream.ToArray(), eps,
            out var y, out var dx, out var dg, out var db);

        ICompiledTrainingPlan<float> plan;
        Tensor<float> output;
        using (var scope = GraphMode.Enable())
        {
            output = engine.LayerNorm(input, gamma, beta, eps, out _, out _);
            engine.ReduceSum(engine.TensorMultiply(output, upstream), null);
            plan = scope.CompileTraining(new[] { input, gamma, beta });
        }

        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                AssertClose(y, output.ToArray(), $"y step {step}");
                AssertClose(dx, plan.Gradients[0].ToArray(), $"dX step {step}");
                AssertClose(dg, plan.Gradients[1].ToArray(), $"dGamma step {step}");
                AssertClose(db, plan.Gradients[2].ToArray(), $"dBeta step {step}");
            }
        }
    }

    /// <summary>
    /// A pre-LN block's input feeds both the norm and the residual add, so its gradient is the sum of two contributions:
    /// the norm's backward must ADD its dX onto the residual's (and, the buffer being reused, start from
    /// zero again on step 2). 300 rows span three row blocks with a partial last one; 20 features leave a scalar tail.
    /// </summary>
    [Theory]
    [InlineData(new[] { 300, 20 })]
    [InlineData(new[] { 4, 32, 64 })]
    public async Task CompiledPlan_ResidualInput_AddsTheNormGradientToTheResidualOne(int[] inputShape)
    {
        await Task.Yield();
        var rng = new Random(23);
        Tensor<float> Make(int[] shape, double offset)
        {
            int n = 1; foreach (var d in shape) n *= d;
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(offset + rng.NextDouble() * 2 - 1);
            return new Tensor<float>(a, shape);
        }
        int fs = inputShape[inputShape.Length - 1];
        var input = Make(inputShape, 0.3);
        var gamma = Make(new[] { fs }, 1.0);
        var beta = Make(new[] { fs }, 0.0);
        var upstream = Make(inputShape, 0.0);
        const double eps = 1e-5;
        var engine = new CpuEngine();

        Reference(input.ToArray(), gamma.ToArray(), beta.ToArray(), upstream.ToArray(), eps,
            out _, out var dxNorm, out var dg, out var db);
        var dy = upstream.ToArray();
        var dx = new double[dxNorm.Length];
        for (int i = 0; i < dx.Length; i++) dx[i] = dxNorm[i] + dy[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var normed = engine.LayerNorm(input, gamma, beta, eps, out _, out _);
            engine.ReduceSum(engine.TensorMultiply(engine.TensorAdd(input, normed), upstream), null);
            plan = scope.CompileTraining(new[] { input, gamma, beta });
        }

        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                AssertClose(dx, plan.Gradients[0].ToArray(), $"dX step {step}");
                AssertClose(dg, plan.Gradients[1].ToArray(), $"dGamma step {step}");
                AssertClose(db, plan.Gradients[2].ToArray(), $"dBeta step {step}");
            }
        }
    }

    private static void Reference(float[] x, float[] g, float[] b, float[] dy, double eps,
        out double[] y, out double[] dx, out double[] dg, out double[] db)
    {
        int fs = g.Length, rows = x.Length / fs;
        y = new double[x.Length]; dx = new double[x.Length]; dg = new double[fs]; db = new double[fs];
        for (int r = 0; r < rows; r++)
        {
            int o = r * fs;
            double m = 0; for (int f = 0; f < fs; f++) m += x[o + f]; m /= fs;
            double v = 0; for (int f = 0; f < fs; f++) v += (x[o + f] - m) * (x[o + f] - m); v /= fs;
            double inv = 1.0 / Math.Sqrt(v + eps);
            double sumG = 0, sumGX = 0;
            for (int f = 0; f < fs; f++)
            {
                double xh = (x[o + f] - m) * inv;
                y[o + f] = xh * g[f] + b[f];
                dg[f] += dy[o + f] * xh;
                db[f] += dy[o + f];
                sumG += g[f] * dy[o + f];
                sumGX += g[f] * dy[o + f] * xh;
            }
            for (int f = 0; f < fs; f++)
            {
                double xh = (x[o + f] - m) * inv;
                dx[o + f] = inv * (g[f] * dy[o + f] - sumG / fs - xh * sumGX / fs);
            }
        }
    }

    private static void AssertClose(double[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0, maxRef = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]));
            maxRef = Math.Max(maxRef, Math.Abs(expected[i]));
        }
        Assert.True(maxErr <= 1e-5 + 2e-5 * maxRef, $"{what}: max |error| {maxErr:G4}, max |reference| {maxRef:G4}");
    }
}
