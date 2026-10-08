using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="SimdGemm.TryGemmSmallJit"/> (the small-K panel routes for a linear layer's forward and dX = dY·Wᵀ)
/// against a double-precision product, and the compiled FusedLinear+ReLU backward that uses them.
/// </summary>
public class SmallJitGemmTests
{
    public static TheoryData<int, int, int, bool, bool> Shapes => new()
    {
        // m, k, n, transA, transB -- every shape is in a route's band (work >= the panel's minimum)
        { 2048, 64, 64, false, false },
        { 2048, 128, 64, false, true },     // dX = dY[2048,128] . W[64,128]^T (over the work ceiling, under the traffic one)
        { 2048, 64, 128, false, false },    // the FFN's first GEMM: admitted by the B-traffic test
        { 64, 2048, 128, true, false },     // dW = X^T[64,2048] . dY[2048,128]: split-k, 16 chunks, m padded to 66
        { 70, 1900, 50, true, false },      // ragged last chunk; m and n off the panel grid
        { 2000, 96, 40, false, true },      // m and n off the 6x16 panel grid: managed edge strips
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public async Task MatchesDoubleProduct(int m, int k, int n, bool transA, bool transB)
    {
        await Task.Yield();
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(m * 31 + k * 7 + n);
        // A is stored [k, m] when transposed, else [m, k]; B is stored [n, k] when transposed, else [k, n].
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = float.NaN;   // must be fully overwritten

        bool ran = SimdGemm.TryGemmSmallJit(a, transA ? m : k, transA, b, transB ? k : n, transB, c, m, k, n);
#if !NET471
        // Every shape here is inside a route's band: where the panel kernel exists, the route must engage (a silent
        // decline would pass the comparison below vacuously).
        // Each route has its own switch (AIDOTNET_JIT_SMALLK, and AIDOTNET_JIT_SPLITK for A transposed); a disabled
        // route declines by design, so the assertion holds only where every switch on the shape's route is on.
        bool routeEnabled = SimdGemm.JitSmallKEnabled && (!transA || SimdGemm.SplitKTransAEnabled);
        if (JitGemmAvx2.Available && routeEnabled) Assert.True(ran, "the small-K route declined an in-band shape");
#endif
        if (!ran)
        {
            // The route is optional (kernel unavailable on this CPU / framework); nothing may have been written.
            Assert.All(c, v => Assert.True(float.IsNaN(v)));
            return;
        }

        double maxErr = 0, maxRef = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++)
                {
                    double av = transA ? a[p * m + i] : a[i * k + p];
                    double bv = transB ? b[j * k + p] : b[p * n + j];
                    s += av * bv;
                }
                maxErr = Math.Max(maxErr, Math.Abs(s - c[i * n + j]));
                maxRef = Math.Max(maxRef, Math.Abs(s));
            }
        Assert.True(maxErr <= 1e-4 * Math.Max(1, maxRef), $"max |error| {maxErr:G4} (max |reference| {maxRef:G4})");
    }

    /// <summary>The compiled FusedLinear+ReLU backward on every step of a plan, against a double transcription.</summary>
    [Fact]
    public async Task CompiledFusedLinearRelu_Gradients_MatchReference_OnEveryStep()
    {
        await Task.Yield();
        const int rows = 2048, inF = 64, outF = 128;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        Tensor<float> Make(params int[] shape)
        {
            int len = 1; foreach (var d in shape) len *= d;
            var arr = new float[len];
            for (int i = 0; i < len; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            return new Tensor<float>(arr, shape);
        }
        var x = Make(rows, inF); var w = Make(inF, outF); var bias = Make(outF); var g = Make(rows, outF);
        var engine = new CpuEngine();

        float[] xa = x.ToArray(), wa = w.ToArray(), ba = bias.ToArray(), ga = g.ToArray();
        var dz = new double[rows * outF];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < outF; j++)
            {
                double z = ba[j];
                for (int p = 0; p < inF; p++) z += (double)xa[r * inF + p] * wa[p * outF + j];
                dz[r * outF + j] = z > 0 ? ga[r * outF + j] : 0;
            }
        var dx = new double[rows * inF]; var dw = new double[inF * outF]; var db = new double[outF];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < outF; j++)
            {
                double d = dz[r * outF + j];
                if (d == 0) continue;
                db[j] += d;
                for (int p = 0; p < inF; p++) { dx[r * inF + p] += d * wa[p * outF + j]; dw[p * outF + j] += d * xa[r * inF + p]; }
            }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var y = engine.FusedLinear(x, w, bias, FusedActivationType.ReLU);
            engine.ReduceSum(engine.TensorMultiply(y, g), null);
            plan = scope.CompileTraining(new[] { x, w, bias });
        }
        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                AssertClose(dx, plan.Gradients[0].ToArray(), $"dX step {step}");
                AssertClose(dw, plan.Gradients[1].ToArray(), $"dW step {step}");
                AssertClose(db, plan.Gradients[2].ToArray(), $"db step {step}");
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
        Assert.True(maxErr <= 1e-4 * Math.Max(1, maxRef), $"{what}: max |error| {maxErr:G4}, max |reference| {maxRef:G4}");
    }
}
