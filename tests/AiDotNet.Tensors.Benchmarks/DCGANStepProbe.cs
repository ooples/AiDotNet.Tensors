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
