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
[Collection("CompilationGlobalState")]
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
#if NETCOREAPP
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

#if NETCOREAPP
    /// <summary>
    /// The rows below the last full 6-row panel and the columns right of the last full 16-column panel come from the
    /// managed edge pass, which sums a[i,k]·b[k,j] for k in order from zero, a multiply then an add -- the scalar loop it
    /// replaced, bit for bit. Widths cover a full 8-lane vector plus a scalar tail (n = 16·q + 11) and edge rows of
    /// 1 to 5 (m = 6·p + r).
    /// </summary>
    [Theory]
    [InlineData(64, 128, 512)]   // 4 edge rows, no edge columns
    [InlineData(67, 40, 539)]    // 1 edge row; an 11-wide right strip (one vector, three scalars)
    [InlineData(23, 9, 27)]      // 5 edge rows; tiny k
    public async Task EdgeRowsAndColumns_EqualTheOrderedScalarSum(int m, int k, int n)
    {
        await Task.Yield();
        if (!JitGemmAvx2.Available) return;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(m * 131 + k * 17 + n);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[m * n];
        for (int i = 0; i < c.Length; i++) c[i] = float.NaN;

        JitGemmAvx2.RunJit(a, b, c, m, n, k);

        int mFull = m - m % 6, nFull = n - n % 16;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                if (i < mFull && j < nFull) continue;   // panel kernel output (FMA): covered by MatchesDoubleProduct
                float acc = 0f;
                for (int p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
                float got = c[i * n + j];
                if (BitConverter.SingleToInt32Bits(got) != BitConverter.SingleToInt32Bits(acc))
                    Assert.Fail($"c[{i},{j}] = {got:R}, the ordered scalar sum is {acc:R}");
            }
    }

    /// <summary>
    /// SimdGemm.TrySgemmTransADeepK, the weight-gradient route C = Aᵀ·B over thousands of rows: full 6x16 tiles, edge
    /// rows (m % 6), an edge column block with an 8-wide vector and scalars (n % 16 &gt; 8) and one under 16 columns,
    /// against a double product; then the same call at one thread and at the machine's width must agree bit for bit
    /// (fixed row slices, each output one FMA chain).
    /// </summary>
    [Theory]
    [InlineData(64, 2048, 128)]
    [InlineData(70, 1900, 59)]
    [InlineData(5, 600, 12)]
    public async Task TransADeepK_MatchesDoubleProduct_AndIsThreadCountIndependent(int m, int k, int n)
    {
        await Task.Yield();
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) return;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(m * 17 + k + n * 5);
        var a = new float[k * m];   // stored [k, m]
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var c = new float[m * n];
        Array.Fill(c, float.NaN);
        Assert.True(SimdGemm.TrySgemmTransADeepK(a, m, b, n, c, m, k, n), "the deep-K route declined an in-band shape");

        double maxErr = 0, maxRef = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int p = 0; p < k; p++) s += (double)a[p * m + i] * b[p * n + j];
                maxErr = Math.Max(maxErr, Math.Abs(s - c[i * n + j]));
                maxRef = Math.Max(maxRef, Math.Abs(s));
            }
        Assert.True(maxErr <= 1e-4 * Math.Max(1, maxRef), $"max |error| {maxErr:G4} (max |reference| {maxRef:G4})");

        int priorDop = AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism;
        var serial = new float[m * n];
        try
        {
            AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
            Assert.True(SimdGemm.TrySgemmTransADeepK(a, m, b, n, serial, m, k, n));
        }
        finally { AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = priorDop; }
        for (int i = 0; i < c.Length; i++)
            if (BitConverter.SingleToInt32Bits(c[i]) != BitConverter.SingleToInt32Bits(serial[i]))
                Assert.Fail($"c[{i}] = {c[i]:R} in parallel, {serial[i]:R} on one thread");
    }

    /// <summary>
    /// An output column's value must not depend on where the width puts it: columns 8-11 are the scalar tail of a
    /// 12-wide C and vector lanes of a full 6x16 tile of a 16-wide C over the same data, and agree bit for bit (both
    /// fused multiply-add chains over the same row slices).
    /// </summary>
    [Fact]
    public async Task TransADeepK_ScalarEdgeColumns_EqualTheVectorTile()
    {
        await Task.Yield();
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) return;
        const int m = 6, k = 1024;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4242);
        var a = new float[k * m];
        var b16 = new float[k * 16];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b16.Length; i++) b16[i] = (float)(rng.NextDouble() * 2 - 1);
        var b12 = new float[k * 12];
        for (int r = 0; r < k; r++) Array.Copy(b16, r * 16, b12, r * 12, 12);
        var c16 = new float[m * 16];
        var c12 = new float[m * 12];
        Assert.True(SimdGemm.TrySgemmTransADeepK(a, m, b16, 16, c16, m, k, 16));
        Assert.True(SimdGemm.TrySgemmTransADeepK(a, m, b12, 12, c12, m, k, 12));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < 12; j++)
                if (BitConverter.SingleToInt32Bits(c12[i * 12 + j]) != BitConverter.SingleToInt32Bits(c16[i * 16 + j]))
                    Assert.Fail($"C[{i},{j}]: {c12[i * 12 + j]:R} at width 12, {c16[i * 16 + j]:R} at width 16");
    }

    /// <summary>
    /// Inside the thin-M box BlasManaged's direct-parallel kernel serves, the small-K routes stand aside: a 64-row
    /// batch is ~11 of the panel's 6-row blocks, and the transposed-B route adds a serial transpose of B, so they ran
    /// 2-8x slower there ([64,128]x[128,512]: 72 vs 36 µs; dY[64,128]·W[512,128]ᵀ: 204 vs 34 µs). A decline writes nothing.
    /// </summary>
    [Theory]
    [InlineData(64, 128, 512, false)]   // a linear layer's forward at batch 64
    [InlineData(64, 128, 512, true)]    // its input gradient dY·Wᵀ
    [InlineData(512, 128, 128, false)]
    public async Task DeclinesTheThinMBoxBlasManagedServes(int m, int k, int n, bool transB)
    {
        await Task.Yield();
        Assert.True(AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.ThinMDirectServesShape(m, n, k, false, transB)
            || !System.Runtime.Intrinsics.X86.Fma.IsSupported, "the shape is meant to be inside the thin-M box");
        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * n];
        Array.Fill(c, float.NaN);
        Assert.False(SimdGemm.TryGemmSmallJit(a, k, false, b, transB ? k : n, transB, c, m, k, n));
        Assert.All(c, v => Assert.True(float.IsNaN(v)));
    }
#endif

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
