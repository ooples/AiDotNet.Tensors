using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 1 drill-down on the bottleneck identified by
/// <see cref="OpLevelPerfHarness"/>: three BERT MatMul shapes account for
/// ~96% of our total time on the sampled hot ops. Our
/// <c>[1,256,3072]×[3072,768]</c> MatMul takes ~1.12 s vs ORT's ~2.3 ms
/// (496× slower), implying we're running at ~0.18% of AVX-512 peak.
///
/// <para>This harness times FOUR dispatch paths on the same BERT FFN
/// shapes so we can localise WHERE the time is lost:</para>
///
/// <list type="number">
/// <item><b>SimdGemm.Sgemm direct</b> — raw kernel with a flat
/// <c>float[]</c>. Zero plumbing; if this is fast, the slowdown is in
/// tensor / plan / importer layers above it.</item>
/// <item><b>MatrixMultiplyHelper.TryGemm (ReadOnlyMemory)</b> — one layer
/// up; the wrapper our CpuEngine dispatches into.</item>
/// <item><b>CpuEngine.TensorMatMul</b> — the public engine API, reached
/// via the 2D × 2D dispatch path.</item>
/// <item><b>CpuEngine.TensorMatMul batched (ND × 2D)</b> — what ONNX
/// MatMul actually hits when the graph feeds <c>[1,256,K]×[K,N]</c>.
/// This is <c>TensorMatMulBatched</c> with its "collapse to single GEMM"
/// path.</item>
/// </list>
///
/// <para>Prints wall-clock µs per call + derived GFLOPS for each path.
/// The comparison tells us which layer is swallowing the time.</para>
/// </summary>
public class MatMulRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public MatMulRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 5;
    private const int Iters  = 30;

    [SkippableFact]
    public void LocaliseBertFfnBottleneck()
    {
        // Evidence harness for the TensorBroadcastAdd(x, zeros) root-cause
        // investigation. Gated so CI runs fast; set the env var to re-run.
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        _output.WriteLine($"Avx512Sgemm.CanUse = {Avx512Sgemm.CanUse}");
        _output.WriteLine($"CPU cores = {Environment.ProcessorCount}");
        _output.WriteLine("");

        // The three BERT shapes that dominate.
        (int m, int k, int n, string label)[] cases =
        {
            (256, 3072, 768,  "FFN down  [1,256,3072]×[3072,768]"),
            (256,  768, 3072, "FFN up    [1,256, 768]×[ 768,3072]"),
            (256,  768,  768, "QKV proj  [1,256, 768]×[ 768, 768]"),
        };

        foreach (var (m, k, n, label) in cases)
        {
            _output.WriteLine($"=== {label}  — {m}×{k}×{n} = {((long)m * k * n * 2):N0} FMAs ===");
            double gflopsDenom = 2.0 * m * k * n / 1e9;

            // (1) Raw SimdGemm.Sgemm
            double t1 = TimeSimdGemmDirect(m, k, n);
            _output.WriteLine($"  [1] SimdGemm.Sgemm direct    : {t1 * 1000:F1} µs  →  {gflopsDenom / t1:F1} GFLOPS");

            // (2) MatrixMultiplyHelper.TryGemm
            double t2 = TimeMatrixMultiplyHelper(m, k, n);
            _output.WriteLine($"  [2] MatrixMultiplyHelper.Try : {t2 * 1000:F1} µs  →  {gflopsDenom / t2:F1} GFLOPS");

            // (3) CpuEngine.TensorMatMul 2D
            double t3 = TimeCpuEngineMatMul2D(m, k, n);
            _output.WriteLine($"  [3] CpuEngine.TensorMatMul 2D: {t3 * 1000:F1} µs  →  {gflopsDenom / t3:F1} GFLOPS");

            // (4) CpuEngine.TensorMatMul ND × 2D (batched broadcast)
            double t4 = TimeCpuEngineMatMul3Dx2D(m, k, n);
            _output.WriteLine($"  [4] CpuEngine ND×2D matmul    : {t4 * 1000:F1} µs  →  {gflopsDenom / t4:F1} GFLOPS");

            // (5) Full ONNX import → CompiledPlan.Execute — what the
            // op-level harness measured at ~1.12 s/call. If THIS is 200×
            // [4], the gap lives in CompiledPlan.Execute wrap / fusion
            // passes / memory planner / Add-zero-wrap output bookkeeping.
            double t5 = TimeOnnxCompiledPlan(m, k, n, out int stepCount);
            _output.WriteLine($"  [5] OnnxImport → Plan.Execute : {t5 * 1000:F1} µs  →  {gflopsDenom / t5:F1} GFLOPS  (StepCount={stepCount})");

            // Break the plan down step-by-step so we can see WHICH of the
            // 6 compiled steps eat the 200× overhead.
            var perStep = ProfilePerStepOf(m, k, n);
            _output.WriteLine("      Per-step breakdown:");
            for (int s = 0; s < perStep.Length; s++)
                _output.WriteLine($"        step[{s}] {perStep[s].OpName,-30} {perStep[s].AvgMs * 1000:F1} µs");

            _output.WriteLine($"  Ratio [4]/[1] = {t4 / t1:F1}x  (tensor-layer overhead)");
            _output.WriteLine($"  Ratio [5]/[4] = {t5 / t4:F1}x  (ONNX/plan-layer overhead)");
            _output.WriteLine($"  Ratio [5]/[1] = {t5 / t1:F1}x  (TOTAL gap vs kernel-only)");
            _output.WriteLine("");
        }
    }

    // ─── timing paths ───────────────────────────────────────────────────────

    private static double TimeSimdGemmDirect(int m, int k, int n)
    {
        var a = Rand(0xDD01, m * k);
        var b = Rand(0xDD02, k * n);
        var c = new float[m * n];

        for (int i = 0; i < Warmup; i++) SimdGemm.Sgemm(a, b, c, m, k, n);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) SimdGemm.Sgemm(a, b, c, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeMatrixMultiplyHelper(int m, int k, int n)
    {
        // TryGemm<T> takes ReadOnlyMemory<T> + offsets. This is the layer
        // CpuEngine.TensorMatMul2D calls immediately after any setup.
        var a = Rand(0xDD11, m * k);
        var b = Rand(0xDD12, k * n);
        var c = new float[m * n];

        for (int i = 0; i < Warmup; i++)
            Helpers.MatrixMultiplyHelper.TryGemm<float>(a.AsMemory(), 0, b.AsMemory(), 0, c.AsMemory(), 0, m, k, n);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            Helpers.MatrixMultiplyHelper.TryGemm<float>(a.AsMemory(), 0, b.AsMemory(), 0, c.AsMemory(), 0, m, k, n);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeCpuEngineMatMul2D(int m, int k, int n)
    {
        var engine = new CpuEngine();
        var aT = new Tensor<float>(new[] { m, k });
        var bT = new Tensor<float>(new[] { k, n });
        Rand(0xDD21, m * k).AsSpan().CopyTo(aT.AsWritableSpan());
        Rand(0xDD22, k * n).AsSpan().CopyTo(bT.AsWritableSpan());

        for (int i = 0; i < Warmup; i++) _ = engine.TensorMatMul(aT, bT);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) _ = engine.TensorMatMul(aT, bT);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeCpuEngineMatMul3Dx2D(int m, int k, int n)
    {
        // The exact shape our ONNX MatMul hits for BERT FFN.
        var engine = new CpuEngine();
        var aT = new Tensor<float>(new[] { 1, m, k });
        var bT = new Tensor<float>(new[] { k, n });
        Rand(0xDD31, m * k).AsSpan().CopyTo(aT.AsWritableSpan());
        Rand(0xDD32, k * n).AsSpan().CopyTo(bT.AsWritableSpan());

        for (int i = 0; i < Warmup; i++) _ = engine.TensorMatMul(aT, bT);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) _ = engine.TensorMatMul(aT, bT);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static (string OpName, double AvgMs)[] ProfilePerStepOf(int m, int k, int n)
    {
        var A = Rand(0xDD41, 1 * m * k);
        var B = Rand(0xDD42, k * n);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] { (name: "A", shape: new[] { 1, m, k }, elemType: OnnxTestHelpers.FLOAT) },
            output: (name: "C", shape: new[] { 1, m, n }, elemType: OnnxTestHelpers.FLOAT),
            initializers: new[] { (name: "B", shape: new[] { k, n }, data: B) });
        var bytes = OnnxTestGraphBuilder.Serialize(model);

        var engine = new CpuEngine();
        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        A.AsSpan().CopyTo(result.Inputs["A"].AsWritableSpan());
        var plan = (CompiledInferencePlan<float>)result.Plan!;
        return plan.ProfilePerStep(Warmup, Iters);
    }

    private static double TimeOnnxCompiledPlan(int m, int k, int n, out int stepCount)
    {
        // Build the same 1-op ONNX MatMul graph the op-level harness used,
        // import via OnnxImporter, and time Execute(). This isolates whether
        // the slowness lives in CompiledPlan.Execute or not.
        var A = Rand(0xDD41, 1 * m * k);
        var B = Rand(0xDD42, k * n);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] { (name: "A", shape: new[] { 1, m, k }, elemType: OnnxTestHelpers.FLOAT) },
            output: (name: "C", shape: new[] { 1, m, n }, elemType: OnnxTestHelpers.FLOAT),
            initializers: new[] { (name: "B", shape: new[] { k, n }, data: B) });
        var bytes = OnnxTestGraphBuilder.Serialize(model);

        var engine = new CpuEngine();
        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        A.AsSpan().CopyTo(result.Inputs["A"].AsWritableSpan());
        stepCount = result.Plan!.StepCount;

        for (int i = 0; i < Warmup; i++) result.Plan.Execute();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) result.Plan.Execute();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
