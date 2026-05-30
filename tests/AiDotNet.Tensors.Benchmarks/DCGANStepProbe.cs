// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #403 Phase A — per-component microbench at DCGAN-typical shapes.
///
/// <para>The DCGAN training step is dominated by Conv2D / ConvTranspose2D
/// on a small set of (batch, channels, spatial, kernel) shapes. Per the
/// issue's profile, ~95% of wall time per step lives in this repo with
/// the heaviest substep being the generator-adversarial gradient pass.
/// This probe isolates the underlying tensor ops so Phase B-E can target
/// the right hotspot.</para>
///
/// <para>Substeps measured (mirrors the issue's Phase A.2 list):</para>
/// <list type="bullet">
///   <item><b>Im2Col</b>: <see cref="Im2ColHelper.Im2Col(ReadOnlySpan{float}, Span{float}, int, int, int, int, int, int, int, int, int, int, int, int)"/>
///         at the DCGAN-decoder shapes (batch=2, kernel 4×4, 8×8 → 16×16).</item>
///   <item><b>Conv2D forward</b>: <see cref="CpuEngine.Conv2D{T}(Tensor{T}, Tensor{T}, int, int, int)"/>
///         (the entry point that wraps Im2Col + GEMM).</item>
///   <item><b>Conv2D backward input</b>: <see cref="CpuEngine.Conv2DBackwardInput{T}"/>.</item>
///   <item><b>Conv2D backward kernel</b>: <see cref="CpuEngine.Conv2DBackwardKernel{T}"/>.</item>
///   <item><b>RecordBinary overhead</b>: tape-recorded TensorAdd inside an
///         active tape vs the same op with no tape.</item>
///   <item><b>BatchNorm forward / backward</b>:
///         <see cref="CpuEngine.BatchNorm{T}"/> and
///         <see cref="CpuEngine.BatchNormBackward{T}"/>.</item>
/// </list>
///
/// <para>Shape choice mirrors a late-DCGAN-decoder layer (batch=2,
/// in=64, out=64, spatial=8×8, kernel=4×4 with stride/pad as appropriate).
/// FP64 is the primary mode because the issue's failing test
/// <c>DCGANTests.MoreData_ShouldNotDegrade</c> instantiates the model with
/// <c>double</c> as <c>T</c>. An FP32 sibling benchmark is included for
/// the Phase F escape-hatch comparison.</para>
///
/// <para>Run via <c>dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks
/// -- --filter *DCGANStepProbe*</c>. Pair the output with a
/// <see cref="ShapeInstrumenter"/> scope over a real DCGAN step to identify
/// which substep dominates wall time for the production model.</para>
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
public class DCGANStepProbe
{
    // ─── DCGAN-typical shapes (late decoder layer, batch=2) ───────────────
    private const int Batch = 2;
    private const int InChannels = 64;
    private const int OutChannels = 64;
    private const int SpatialIn = 8;
    private const int KernelHW = 4;
    private const int Stride = 1;
    private const int Padding = 1;  // (8 + 2*1 - 4)/1 + 1 = 7
    private const int SpatialOut = (SpatialIn + 2 * Padding - KernelHW) / Stride + 1;

    // Im2Col working buffer dims
    private const int ColH = InChannels * KernelHW * KernelHW;
    private const int ColW = SpatialOut * SpatialOut;

    // PR #412 CodeRabbit fix: hoist the shape / stride / padding / dilation
    // int[] arrays to static readonly fields so the backward probe methods
    // don't allocate three fresh arrays per call. The allocation profile is
    // measuring op-internal allocations; the harness overhead of allocating
    // these argument arrays per call contaminates the per-substep ranking
    // (especially Conv2DBackward* which fire ~1 KB of GC-heap traffic per
    // call from just the int[] args before any engine code runs).
    private static readonly int[] InputShape =
        { Batch, InChannels, SpatialIn, SpatialIn };
    private static readonly int[] KernelShape =
        { OutChannels, InChannels, KernelHW, KernelHW };
    private static readonly int[] StrideArr = { Stride, Stride };
    private static readonly int[] PaddingArr = { Padding, Padding };
    private static readonly int[] DilationArr = { 1, 1 };

    private CpuEngine _engine = null!;

    // FP64 (matches DCGANTests.MoreData_ShouldNotDegrade)
    private Tensor<double> _input64 = null!;
    private Tensor<double> _kernel64 = null!;
    private Tensor<double> _gradOutput64 = null!;
    private Tensor<double> _bnGamma64 = null!;
    private Tensor<double> _bnBeta64 = null!;
    private Tensor<double> _bnIn64 = null!;
    private Tensor<double> _bnGradOut64 = null!;
    // PR #412 CodeRabbit fix: precomputed mean/var so BatchNormBackward_Fp64
    // only measures the backward step. Pre-fix the benchmark called BatchNorm
    // (forward) inside the measured method, so the reported "backward" time
    // and allocation profile actually included the full forward pass.
    private Tensor<double> _bnMean64 = null!;
    private Tensor<double> _bnVar64 = null!;
    private Tensor<double> _addA64 = null!;
    private Tensor<double> _addB64 = null!;

    // FP32 (Phase F escape-hatch — 8-lane AVX2 vs FP64's 4-lane)
    private Tensor<float> _input32 = null!;
    private Tensor<float> _kernel32 = null!;
    private Tensor<float> _gradOutput32 = null!;

    // Im2Col scratch (FP32, since the public helper is float-only)
    private float[] _im2colIn = null!;
    private float[] _im2colOut = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();
        var rng = new Random(42);

        _input64 = Tensor<double>.CreateRandom(rng, Batch, InChannels, SpatialIn, SpatialIn);
        _kernel64 = Tensor<double>.CreateRandom(rng, OutChannels, InChannels, KernelHW, KernelHW);
        _gradOutput64 = Tensor<double>.CreateRandom(rng, Batch, OutChannels, SpatialOut, SpatialOut);

        _bnIn64 = Tensor<double>.CreateRandom(rng, Batch, InChannels, SpatialIn, SpatialIn);
        _bnGamma64 = Tensor<double>.CreateRandom(rng, InChannels);
        _bnBeta64 = Tensor<double>.CreateRandom(rng, InChannels);
        _bnGradOut64 = Tensor<double>.CreateRandom(rng, Batch, InChannels, SpatialIn, SpatialIn);

        // PR #412 CodeRabbit fix: run BN forward ONCE during setup so the
        // backward probe can reuse mean/var without paying forward cost on
        // each measured iteration.
        _ = _engine.BatchNorm(_bnIn64, _bnGamma64, _bnBeta64, 1e-5, out _bnMean64, out _bnVar64);

        int addLen = Batch * OutChannels * SpatialOut * SpatialOut;
        _addA64 = Tensor<double>.CreateRandom(rng, addLen);
        _addB64 = Tensor<double>.CreateRandom(rng, addLen);

        _input32 = Tensor<float>.CreateRandom(rng, Batch, InChannels, SpatialIn, SpatialIn);
        _kernel32 = Tensor<float>.CreateRandom(rng, OutChannels, InChannels, KernelHW, KernelHW);
        _gradOutput32 = Tensor<float>.CreateRandom(rng, Batch, OutChannels, SpatialOut, SpatialOut);

        _im2colIn = new float[Batch * InChannels * SpatialIn * SpatialIn];
        for (int i = 0; i < _im2colIn.Length; i++) _im2colIn[i] = (float)rng.NextDouble();
        _im2colOut = new float[Batch * ColH * ColW];
    }

    // ─── Im2Col (raw helper, FP32 — issue lists this as a top allocator) ──

    [Benchmark, BenchmarkCategory("Im2Col")]
    public void Im2Col_Fp32_DcganShape()
    {
        Im2ColHelper.Im2Col(
            _im2colIn, _im2colOut,
            Batch, InChannels, SpatialIn, SpatialIn,
            KernelHW, KernelHW,
            Stride, Stride,
            Padding, Padding,
            1, 1);
    }

    // ─── Conv2D forward (entry point that wraps Im2Col + GEMM) ────────────

    [Benchmark, BenchmarkCategory("Conv2DForward")]
    public Tensor<double> Conv2DForward_Fp64()
        => _engine.Conv2D(_input64, _kernel64, Stride, Padding, 1);

    [Benchmark, BenchmarkCategory("Conv2DForward")]
    public Tensor<float> Conv2DForward_Fp32()
        => _engine.Conv2D(_input32, _kernel32, Stride, Padding, 1);

    // ─── Conv2D backward (the heavier half of the per-step profile) ───────

    [Benchmark, BenchmarkCategory("Conv2DBackward")]
    public Tensor<double> Conv2DBackwardInput_Fp64()
        => _engine.Conv2DBackwardInput(_gradOutput64, _kernel64,
            InputShape, StrideArr, PaddingArr, DilationArr);

    [Benchmark, BenchmarkCategory("Conv2DBackward")]
    public Tensor<double> Conv2DBackwardKernel_Fp64()
        => _engine.Conv2DBackwardKernel(_gradOutput64, _input64,
            KernelShape, StrideArr, PaddingArr, DilationArr);

    // ─── RecordBinary overhead (tape inactive vs tape active) ─────────────

    [Benchmark(Baseline = true), BenchmarkCategory("RecordBinary")]
    public Tensor<double> TensorAdd_NoTape()
        => _engine.TensorAdd(_addA64, _addB64);

    [Benchmark, BenchmarkCategory("RecordBinary")]
    public Tensor<double> TensorAdd_WithTape()
    {
        using var tape = new GradientTape<double>();
        return _engine.TensorAdd(_addA64, _addB64);
    }

    // ─── BatchNorm forward / backward (issue Phase A.2 list item) ─────────

    [Benchmark, BenchmarkCategory("BatchNorm")]
    public Tensor<double> BatchNormForward_Fp64()
        => _engine.BatchNorm(_bnIn64, _bnGamma64, _bnBeta64, 1e-5, out _, out _);

    [Benchmark, BenchmarkCategory("BatchNorm")]
    public Tensor<double> BatchNormBackward_Fp64()
        // Use the precomputed mean/var from Setup so this measures backward-only.
        => _engine.BatchNormBackward(
            _bnGradOut64, _bnIn64, _bnGamma64, _bnMean64, _bnVar64, 1e-5,
            out _, out _);

    // ─── Phase A.3 allocation profile (callable outside BDN) ──────────────

    /// <summary>
    /// Walks every Phase A.2 substep once with <see cref="GC.GetAllocatedBytesForCurrentThread"/>
    /// bracketing, returning bytes-allocated per substep. Independent of
    /// BenchmarkDotNet — designed to be called from an xunit fact or the
    /// benchmark runner's Program.cs so reviewers can compare alloc shape
    /// against the issue's hypothesis (GradNode + closures + per-call im2col).
    /// </summary>
    /// <remarks>
    /// Per-thread counter ignores cross-thread allocations from worker
    /// pools, which is correct for serial single-call profiling. For a
    /// process-wide view use BDN's <c>[MemoryDiagnoser]</c> output above.
    /// </remarks>
    public static AllocationProfile RunAllocationProfile()
    {
        var probe = new DCGANStepProbe();
        probe.Setup();
        var p = new AllocationProfile();

        // Warm up every measured path once before allocation sampling.
        // PR #412 CodeRabbit fix: pre-fix only Conv2DForward_Fp64 was warmed,
        // so the other 8 substeps still incurred first-call/JIT-tier-up alloc
        // overhead in their MeasureAlloc result — skewing the Phase A
        // per-substep ranking. Warming all paths gives steady-state numbers.
        probe.Im2Col_Fp32_DcganShape();
        _ = probe.Conv2DForward_Fp64();
        _ = probe.Conv2DForward_Fp32();
        _ = probe.Conv2DBackwardInput_Fp64();
        _ = probe.Conv2DBackwardKernel_Fp64();
        _ = probe.TensorAdd_NoTape();
        _ = probe.TensorAdd_WithTape();
        _ = probe.BatchNormForward_Fp64();
        _ = probe.BatchNormBackward_Fp64();

        p.Im2Col_Fp32 = MeasureAlloc(() => probe.Im2Col_Fp32_DcganShape());
        p.Conv2DForward_Fp64 = MeasureAlloc(() => probe.Conv2DForward_Fp64());
        p.Conv2DForward_Fp32 = MeasureAlloc(() => probe.Conv2DForward_Fp32());
        p.Conv2DBackwardInput_Fp64 = MeasureAlloc(() => probe.Conv2DBackwardInput_Fp64());
        p.Conv2DBackwardKernel_Fp64 = MeasureAlloc(() => probe.Conv2DBackwardKernel_Fp64());
        p.TensorAdd_NoTape = MeasureAlloc(() => probe.TensorAdd_NoTape());
        p.TensorAdd_WithTape = MeasureAlloc(() => probe.TensorAdd_WithTape());
        p.BatchNormForward_Fp64 = MeasureAlloc(() => probe.BatchNormForward_Fp64());
        p.BatchNormBackward_Fp64 = MeasureAlloc(() => probe.BatchNormBackward_Fp64());

        return p;
    }

    /// <summary>
    /// Issue #403 Phase A.2 — per-substep wall-clock ranking via min-of-many.
    /// The allocation profile shows the conv backward paths are already
    /// ArrayPool-backed, so the 519ms DCGAN&lt;double&gt; step is compute-bound,
    /// not allocation-bound. This method ranks substeps by time so Phase B-E
    /// target the right GEMM. Min-of-many (industry-standard on a noisy box):
    /// each substep is run <paramref name="innerReps"/> times per sample, the
    /// per-call time is the sample, and the reported figure is the minimum
    /// across <paramref name="outerSamples"/> samples (least-perturbed run).
    /// </summary>
    public static WallClockProfile RunWallClockProfile(int innerReps = 50, int outerSamples = 25)
    {
        var probe = new DCGANStepProbe();
        probe.Setup();

        // Warm all paths (JIT tier-up + pool fill) before timing.
        probe.Im2Col_Fp32_DcganShape();
        _ = probe.Conv2DForward_Fp64();
        _ = probe.Conv2DForward_Fp32();
        _ = probe.Conv2DBackwardInput_Fp64();
        _ = probe.Conv2DBackwardKernel_Fp64();
        _ = probe.TensorAdd_NoTape();
        _ = probe.TensorAdd_WithTape();
        _ = probe.BatchNormForward_Fp64();
        _ = probe.BatchNormBackward_Fp64();

        return new WallClockProfile
        {
            Im2Col_Fp32 = MinTime(() => probe.Im2Col_Fp32_DcganShape(), innerReps, outerSamples),
            Conv2DForward_Fp64 = MinTime(() => probe.Conv2DForward_Fp64(), innerReps, outerSamples),
            Conv2DForward_Fp32 = MinTime(() => probe.Conv2DForward_Fp32(), innerReps, outerSamples),
            Conv2DBackwardInput_Fp64 = MinTime(() => probe.Conv2DBackwardInput_Fp64(), innerReps, outerSamples),
            Conv2DBackwardKernel_Fp64 = MinTime(() => probe.Conv2DBackwardKernel_Fp64(), innerReps, outerSamples),
            TensorAdd_NoTape = MinTime(() => probe.TensorAdd_NoTape(), innerReps, outerSamples),
            TensorAdd_WithTape = MinTime(() => probe.TensorAdd_WithTape(), innerReps, outerSamples),
            BatchNormForward_Fp64 = MinTime(() => probe.BatchNormForward_Fp64(), innerReps, outerSamples),
            BatchNormBackward_Fp64 = MinTime(() => probe.BatchNormBackward_Fp64(), innerReps, outerSamples),
        };
    }

    /// <summary>
    /// Issue #403 Phase A.2b — bare <see cref="Engines.BlasManaged.BlasManaged.Gemm{T}"/>
    /// timing at the three catalog shapes, with NO conv wrapper (no im2col, no
    /// batch-parallel fan-out). Isolates whether the BackwardKernel pathology is
    /// the transposed tiny-K/wide-N GEMM shape itself, vs the conv wrapper's
    /// nested batch parallelism. Each shape runs single-call (the conv path
    /// already dispatches one GEMM per batch), so this measures the inner GEMM
    /// the way the conv path invokes it.
    /// </summary>
    public static string RunBareGemmProbe(int innerReps = 50, int outerSamples = 25)
    {
        // Shapes straight from the catalog (single batch slice each):
        //   Forward:        C[64,49]   = A[64,1024]   · B[1024,49]            (NN)
        //   BackwardInput:  C[1024,49] = A[64,1024]^T · B[64,49]             (tA)
        //   BackwardKernel: C[64,1024] = A[64,49]     · B[1024,49]^T (transB) (tB)
        var rng = new Random(7);
        double[] Rand(int n) { var x = new double[n]; for (int i = 0; i < n; i++) x[i] = rng.NextDouble(); return x; }

        // Forward: M=64, N=49, K=1024 (NN)
        double[] fwdA = Rand(64 * 1024), fwdB = Rand(1024 * 49), fwdC = new double[64 * 49];
        // BackwardInput: M=1024, N=49, K=64 (transA): A stored [64,1024], B [64,49]
        double[] biA = Rand(64 * 1024), biB = Rand(64 * 49), biC = new double[1024 * 49];
        // BackwardKernel: M=64, N=1024, K=49 (transB): A [64,49], B stored [1024,49]
        double[] bkA = Rand(64 * 49), bkB = Rand(1024 * 49), bkC = new double[64 * 1024];

        void Fwd() => Engines.BlasManaged.BlasManaged.Gemm<double>(fwdA, 1024, false, fwdB, 49, false, fwdC, 49, 64, 49, 1024);
        void BwdInput() => Engines.BlasManaged.BlasManaged.Gemm<double>(biA, 1024, true, biB, 49, false, biC, 49, 1024, 49, 64);
        void BwdKernel() => Engines.BlasManaged.BlasManaged.Gemm<double>(bkA, 49, false, bkB, 49, true, bkC, 1024, 64, 1024, 49);

        Fwd(); BwdInput(); BwdKernel();  // warm
        double fwd = MinTime(Fwd, innerReps, outerSamples);
        double bi = MinTime(BwdInput, innerReps, outerSamples);
        double bk = MinTime(BwdKernel, innerReps, outerSamples);

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Bare BlasManaged.Gemm<double> (min-of-many, µs/call) — no conv wrapper:");
        sb.AppendLine($"  {fwd / 1000.0,10:N3} µs  Forward        M=64  N=49   K=1024 NN");
        sb.AppendLine($"  {bi / 1000.0,10:N3} µs  BackwardInput  M=1024 N=49  K=64   tA");
        sb.AppendLine($"  {bk / 1000.0,10:N3} µs  BackwardKernel M=64  N=1024 K=49   tB  <-- catalog hot shape");
        return sb.ToString();
    }

    /// <summary>
    /// Issue #403 Phase C — Conv2DBackwardInput sub-step breakdown at the DCGAN
    /// shape. Phase A's bare-GEMM probe showed the BackwardInput GEMM is only
    /// ~134µs but the wrapper is ~751µs; this splits the wrapper into its three
    /// pieces (kernel transpose, GEMM, col2im scatter) so we know which to
    /// optimize. Single-batch cost (the wrapper runs one of each per batch).
    /// </summary>
    public static string RunBackwardInputBreakdown(int innerReps = 50, int outerSamples = 25)
    {
        int colH = InChannels * KernelHW * KernelHW; // 1024
        int colW = SpatialOut * SpatialOut;          // 49
        var rng = new Random(11);
        double[] R(int n) { var x = new double[n]; for (int i = 0; i < n; i++) x[i] = rng.NextDouble(); return x; }

        double[] kernel = R(OutChannels * colH);   // [outC, colH]
        double[] kernelTD = new double[colH * OutChannels];
        double[] gradOut = R(OutChannels * colW);  // one batch slice [outC, colW]
        double[] colBuf = R(colH * colW);          // pre-filled (col2im reads it)
        double[] gradInput = new double[InChannels * SpatialIn * SpatialIn];

        void Transpose()
        {
            for (int r = 0; r < OutChannels; r++)
                for (int c = 0; c < colH; c++)
                    kernelTD[c * OutChannels + r] = kernel[r * colH + c];
        }
        void Gemm() => Engines.BlasManaged.BlasManaged.Gemm<double>(
            kernelTD, OutChannels, false, gradOut, colW, false, colBuf, colW, colH, colW, OutChannels);
        void Col2Im() => Im2ColHelper.Col2ImAccumulate(
            colBuf, gradInput, InChannels, SpatialIn, SpatialIn,
            KernelHW, KernelHW, Stride, Stride, Padding, Padding, 1, 1, SpatialOut, SpatialOut);

        Transpose(); Gemm(); Col2Im();
        double t = MinTime(Transpose, innerReps, outerSamples);
        double g = MinTime(Gemm, innerReps, outerSamples);
        double c = MinTime(Col2Im, innerReps, outerSamples);

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Conv2DBackwardInput sub-step breakdown (min-of-many, µs, single batch slice):");
        sb.AppendLine($"  {t / 1000.0,10:N3} µs  kernel transpose ([outC,colH] -> [colH,outC], once/call)");
        sb.AppendLine($"  {g / 1000.0,10:N3} µs  GEMM colH×colW×outC (full width)");
        sb.AppendLine($"  {c / 1000.0,10:N3} µs  Col2ImAccumulate scatter (per batch)");
        return sb.ToString();
    }

    /// <summary>
    /// Issue #403 Phase D — Conv2DForward sub-step breakdown at the DCGAN shape.
    /// Forward lowers to im2col + GEMM per batch (no kernel transpose — the
    /// kernel is already [outC, colH]). Splits the wrapper into im2col vs GEMM
    /// so we know which (if either) is the lever for the remaining ~790µs.
    /// Single-batch cost.
    /// </summary>
    public static string RunForwardBreakdown(int innerReps = 50, int outerSamples = 25)
    {
        int colH = InChannels * KernelHW * KernelHW; // K = 1024
        int colW = SpatialOut * SpatialOut;          // N = 49
        var rng = new Random(13);
        double[] R(int n) { var x = new double[n]; for (int i = 0; i < n; i++) x[i] = rng.NextDouble(); return x; }

        double[] inputSlice = R(InChannels * SpatialIn * SpatialIn); // one image
        double[] kernel = R(OutChannels * colH);                     // [outC, colH]
        double[] col = new double[colH * colW];
        double[] output = new double[OutChannels * colW];

        void Im2ColD() => Im2ColHelper.Im2Col(
            new ReadOnlySpan<double>(inputSlice), new Span<double>(col),
            1, InChannels, SpatialIn, SpatialIn, KernelHW, KernelHW,
            Stride, Stride, Padding, Padding, 1, 1);
        void Gemm() => Engines.BlasManaged.BlasManaged.Gemm<double>(
            kernel, colH, false, col, colW, false, output, colW, OutChannels, colW, colH);

        Im2ColD(); Gemm();
        double i2c = MinTime(Im2ColD, innerReps, outerSamples);
        double g = MinTime(Gemm, innerReps, outerSamples);

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Conv2DForward sub-step breakdown (min-of-many, µs, single batch slice):");
        sb.AppendLine($"  {i2c / 1000.0,10:N3} µs  Im2Col (double, per batch)");
        sb.AppendLine($"  {g / 1000.0,10:N3} µs  GEMM outC×colW×colH (M=64,N=49,K=1024, full width)");
        return sb.ToString();
    }

    /// <summary>
    /// Issue #403 Phase D — strategy sweep for the three conv GEMM shapes.
    /// Each shape is timed under Auto (the deterministic-mode static heuristic)
    /// vs forced Streaming / PackAOnly / PackBoth, so we can see whether the
    /// heuristic's PackBoth choice is wrong for these small-output/large-K
    /// shapes (where pack overhead dwarfs the ~6.4 MFLOP compute).
    /// </summary>
    public static string RunConvGemmStrategySweep(int innerReps = 50, int outerSamples = 25)
    {
        int colH = InChannels * KernelHW * KernelHW; // 1024
        int colW = SpatialOut * SpatialOut;          // 49
        var rng = new Random(17);
        double[] R(int n) { var x = new double[n]; for (int i = 0; i < n; i++) x[i] = rng.NextDouble(); return x; }

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Conv GEMM strategy sweep (min-of-many, µs/call):");

        // Forward: C[outC,colW] = kernel[outC,colH] @ im2col[colH,colW]  (NN)
        {
            double[] kern = R(OutChannels * colH), col = R(colH * colW), o = new double[OutChannels * colW];
            double Run(Engines.BlasManaged.PackingMode pm) => MinTime(() =>
                Engines.BlasManaged.BlasManaged.Gemm<double>(kern, colH, false, col, colW, false, o, colW,
                    OutChannels, colW, colH, new Engines.BlasManaged.BlasOptions<double> { PackingMode = pm }),
                innerReps, outerSamples);
            sb.AppendLine($"  Forward       M=64  N=49   K=1024 NN : Auto {Run(Engines.BlasManaged.PackingMode.Auto)/1000.0,8:N2}  Stream {Run(Engines.BlasManaged.PackingMode.ForceStreaming)/1000.0,8:N2}  PackA {Run(Engines.BlasManaged.PackingMode.ForcePackAOnly)/1000.0,8:N2}  PackBoth {Run(Engines.BlasManaged.PackingMode.ForcePackBoth)/1000.0,8:N2}");
        }
        // BackwardInput: C[colH,colW] = kernel^T[colH,outC] @ gradOut[outC,colW]  (transA)
        {
            double[] kern = R(OutChannels * colH), go = R(OutChannels * colW), o = new double[colH * colW];
            double Run(Engines.BlasManaged.PackingMode pm) => MinTime(() =>
                Engines.BlasManaged.BlasManaged.Gemm<double>(kern, colH, true, go, colW, false, o, colW,
                    colH, colW, OutChannels, new Engines.BlasManaged.BlasOptions<double> { PackingMode = pm }),
                innerReps, outerSamples);
            sb.AppendLine($"  BackwardInput M=1024 N=49  K=64   tA : Auto {Run(Engines.BlasManaged.PackingMode.Auto)/1000.0,8:N2}  Stream {Run(Engines.BlasManaged.PackingMode.ForceStreaming)/1000.0,8:N2}  PackA {Run(Engines.BlasManaged.PackingMode.ForcePackAOnly)/1000.0,8:N2}  PackBoth {Run(Engines.BlasManaged.PackingMode.ForcePackBoth)/1000.0,8:N2}");
        }
        // BackwardKernel: C[outC,colH] = gradOut[outC,colW] @ im2col[colH,colW]^T  (transB)
        {
            double[] go = R(OutChannels * colW), col = R(colH * colW), o = new double[OutChannels * colH];
            double Run(Engines.BlasManaged.PackingMode pm) => MinTime(() =>
                Engines.BlasManaged.BlasManaged.Gemm<double>(go, colW, false, col, colW, true, o, colH,
                    OutChannels, colH, colW, new Engines.BlasManaged.BlasOptions<double> { PackingMode = pm }),
                innerReps, outerSamples);
            sb.AppendLine($"  BackwardKern  M=64  N=1024 K=49   tB : Auto {Run(Engines.BlasManaged.PackingMode.Auto)/1000.0,8:N2}  Stream {Run(Engines.BlasManaged.PackingMode.ForceStreaming)/1000.0,8:N2}  PackA {Run(Engines.BlasManaged.PackingMode.ForcePackAOnly)/1000.0,8:N2}  PackBoth {Run(Engines.BlasManaged.PackingMode.ForcePackBoth)/1000.0,8:N2}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Issue #403 Phase D — machine-code-vs-managed crossover for NN GEMMs.
    /// Forward routes through the #409 machine-code fast path (Auto, NN) which
    /// is ~5× slower than managed for small outputs. Sweep (M,N) at K=1024 to
    /// find where machine-code (Auto) catches up to PackBoth, so a min-output
    /// guard on the machine-code gate can exclude only the loss region.
    /// </summary>
    public static string RunMachineKernelCrossover(int innerReps = 30, int outerSamples = 20)
    {
        var rng = new Random(19);
        double[] R(int n) { var x = new double[n]; for (int i = 0; i < n; i++) x[i] = rng.NextDouble(); return x; }

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Machine-code (Auto,NN) vs PackBoth crossover, K=1024 (min-of-many, µs/call):");
        (int m, int n)[] shapes = { (64, 49), (64, 256), (128, 128), (256, 256), (256, 49), (512, 512), (1024, 49) };
        const int K = 1024;
        foreach (var (m, n) in shapes)
        {
            double[] a = R(m * K), bb = R(K * n), c = new double[m * n];
            double auto = MinTime(() => Engines.BlasManaged.BlasManaged.Gemm<double>(
                a, K, false, bb, n, false, c, n, m, n, K), innerReps, outerSamples);
            double packBoth = MinTime(() => Engines.BlasManaged.BlasManaged.Gemm<double>(
                a, K, false, bb, n, false, c, n, m, n, K,
                new Engines.BlasManaged.BlasOptions<double> { PackingMode = Engines.BlasManaged.PackingMode.ForcePackBoth }),
                innerReps, outerSamples);
            string winner = auto <= packBoth * 1.05 ? "machine-code OK" : "MANAGED WINS";
            sb.AppendLine($"  M={m,5} N={n,5} (M·N={m * n,8}) : Auto {auto / 1000.0,8:N2}  PackBoth {packBoth / 1000.0,8:N2}  -> {winner}");
        }
        return sb.ToString();
    }

    /// <summary>Minimum per-call nanoseconds across <paramref name="outerSamples"/>
    /// samples, each averaging <paramref name="innerReps"/> back-to-back calls.</summary>
    private static double MinTime(Action action, int innerReps, int outerSamples)
    {
        double nsPerTick = 1_000_000_000.0 / System.Diagnostics.Stopwatch.Frequency;
        double best = double.MaxValue;
        for (int s = 0; s < outerSamples; s++)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int r = 0; r < innerReps; r++) action();
            long end = System.Diagnostics.Stopwatch.GetTimestamp();
            double perCallNs = (end - start) * nsPerTick / innerReps;
            if (perCallNs < best) best = perCallNs;
        }
        return best;
    }

    private static long MeasureAlloc(Action action)
    {
        // GC.Collect before the measurement to keep allocator state
        // deterministic across substeps — without it, lazy LOH compaction
        // and pinned-pool free-list state from earlier substeps shows up
        // as noise in later measurements.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long before = GC.GetAllocatedBytesForCurrentThread();
        action();
        long after = GC.GetAllocatedBytesForCurrentThread();
        return after - before;
    }

    /// <summary>Per-substep min-of-many wall-clock (nanoseconds per call).</summary>
    public sealed class WallClockProfile
    {
        public double Im2Col_Fp32;
        public double Conv2DForward_Fp64;
        public double Conv2DForward_Fp32;
        public double Conv2DBackwardInput_Fp64;
        public double Conv2DBackwardKernel_Fp64;
        public double TensorAdd_NoTape;
        public double TensorAdd_WithTape;
        public double BatchNormForward_Fp64;
        public double BatchNormBackward_Fp64;

        public string Format()
        {
            var entries = new (string name, double ns)[]
            {
                ("Conv2DBackwardInput_Fp64",  Conv2DBackwardInput_Fp64),
                ("Conv2DBackwardKernel_Fp64", Conv2DBackwardKernel_Fp64),
                ("Conv2DForward_Fp64",        Conv2DForward_Fp64),
                ("Conv2DForward_Fp32",        Conv2DForward_Fp32),
                ("BatchNormBackward_Fp64",    BatchNormBackward_Fp64),
                ("BatchNormForward_Fp64",     BatchNormForward_Fp64),
                ("Im2Col_Fp32",               Im2Col_Fp32),
                ("TensorAdd_WithTape",        TensorAdd_WithTape),
                ("TensorAdd_NoTape",          TensorAdd_NoTape),
            };
            Array.Sort(entries, (a, b) => b.ns.CompareTo(a.ns));

            // FP64 substeps that make up one generator/discriminator conv layer's
            // forward+backward (the #403 hot path). Sum to contextualize the ranking.
            double convStepNs = Conv2DForward_Fp64 + Conv2DBackwardInput_Fp64 + Conv2DBackwardKernel_Fp64;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Per-substep wall-clock (min-of-many, µs/call, descending):");
            foreach (var (name, ns) in entries)
                sb.AppendLine($"  {ns / 1000.0,12:N3} µs  {name}");
            sb.AppendLine($"  ----");
            sb.AppendLine($"  {convStepNs / 1000.0,12:N3} µs  [Conv2D fwd + bwdInput + bwdKernel, FP64]");
            return sb.ToString();
        }
    }

    /// <summary>Per-substep bytes-allocated (one-call sample, current thread).</summary>
    public sealed class AllocationProfile
    {
        public long Im2Col_Fp32;
        public long Conv2DForward_Fp64;
        public long Conv2DForward_Fp32;
        public long Conv2DBackwardInput_Fp64;
        public long Conv2DBackwardKernel_Fp64;
        public long TensorAdd_NoTape;
        public long TensorAdd_WithTape;
        public long BatchNormForward_Fp64;
        public long BatchNormBackward_Fp64;

        public string Format()
        {
            var entries = new (string name, long bytes)[]
            {
                ("Conv2DBackwardInput_Fp64",  Conv2DBackwardInput_Fp64),
                ("Conv2DBackwardKernel_Fp64", Conv2DBackwardKernel_Fp64),
                ("Conv2DForward_Fp64",        Conv2DForward_Fp64),
                ("Conv2DForward_Fp32",        Conv2DForward_Fp32),
                ("BatchNormBackward_Fp64",    BatchNormBackward_Fp64),
                ("BatchNormForward_Fp64",     BatchNormForward_Fp64),
                ("Im2Col_Fp32",               Im2Col_Fp32),
                ("TensorAdd_WithTape",        TensorAdd_WithTape),
                ("TensorAdd_NoTape",          TensorAdd_NoTape),
            };
            Array.Sort(entries, (a, b) => b.bytes.CompareTo(a.bytes));

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Per-substep allocation profile (bytes, current-thread, descending):");
            foreach (var (name, bytes) in entries)
                sb.AppendLine($"  {bytes,12:N0}  {name}");
            return sb.ToString();
        }
    }
}
