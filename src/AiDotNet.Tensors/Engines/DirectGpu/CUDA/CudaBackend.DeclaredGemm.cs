// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Plain GEMMs on declared programs (#890 phase 4c): Gemm, MatMulTransposed, MatMulTransposedA and BatchedGemm at
// alpha 1, beta 0 lower CodegenBackendPrograms.Gemm under one of a fixed set of schedules -- the planner's own
// lowering or the implicit-GEMM emitter's tiles, optionally split along the reduction. (The tiled-contraction emitter
// does not apply: it streams one operand past a rank-2 matrix, and a GEMM's two rank-2 operands are ambiguous to it.)
// Each schedule is a Generated candidate of the GEMM tuned slot, whose reference is the entry point's own path (the
// tuned route, the narrow kernels skinny_gemm / small_gemm_* / wgrad_tile_tn / skinny_gemm_tn, cuBLAS): a schedule
// serves a shape only after the slot's evidence gate measured it faster there, and the decision persists.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

internal enum DeclaredGemmLowering : byte
{
    /// <summary>The program planner's own lowering (its split or unsplit affine kernel).</summary>
    Planner,
    /// <summary>The implicit-GEMM emitter at a tile, optionally split along the reduction.</summary>
    ImplicitGemm,
}

internal sealed class DeclaredGemmSchedule
{
    internal DeclaredGemmSchedule(DeclaredGemmLowering lowering, int tileM = 0, int tileN = 0, int splitFactor = 0,
        bool vectorCombine = false)
    {
        Lowering = lowering;
        TileM = tileM;
        TileN = tileN;
        SplitFactor = splitFactor;
        VectorCombine = vectorCombine;
        Name = lowering == DeclaredGemmLowering.Planner
            ? "planner"
            : "implicit-gemm." + (splitFactor > 1 ? "split" + splitFactor + (vectorCombine ? "v" : "") + "." : "") +
              "m" + tileM + "n" + tileN;
    }

    internal DeclaredGemmLowering Lowering { get; }
    internal int TileM { get; }
    internal int TileN { get; }
    internal int SplitFactor { get; }

    /// <summary>A split's combine reads four outputs' partials with one vector load (PtxAffineEmitter.VectorTileReductions).</summary>
    internal bool VectorCombine { get; }

    /// <summary>The schedule's name, which is also the declared key's name (one program per shape and schedule).</summary>
    internal string Name { get; }
}

/// <summary>A row-major GEMM at alpha 1, beta 0, as a GEMM entry point received it.</summary>
internal readonly struct CudaGemmArgs
{
    internal CudaGemmArgs(DeclaredFamily family, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int m, int n, int k)
    {
        Family = family; A = a; B = b; C = c; Batch = batch; M = m; N = n; K = k;
    }

    internal DeclaredFamily Family { get; }
    internal IGpuBuffer A { get; }
    internal IGpuBuffer B { get; }
    internal IGpuBuffer C { get; }
    internal int Batch { get; }
    internal int M { get; }
    internal int N { get; }
    internal int K { get; }

    internal int OutputCount => Batch * M * N;

    /// <summary>Shape class: layout (0 plain, 1 transposed A, 2 transposed B), batch, M, N, K.</summary>
    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[5];
        d[0] = Family == DeclaredFamily.GemmTransposedA ? 1 : Family == DeclaredFamily.GemmTransposedB ? 2 : 0;
        d[1] = Batch; d[2] = M; d[3] = N; d[4] = K;
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 5));
    }

    internal static DeclaredFamily FamilyOf(in TunedShape shape) => shape[0] switch
    {
        1 => DeclaredFamily.GemmTransposedA,
        2 => DeclaredFamily.GemmTransposedB,
        _ => DeclaredFamily.Gemm,
    };
}

public sealed partial class CudaBackend
{
    private static readonly string[] s_gemmNames = { "a", "b", "c" };

    private static readonly DeclaredGemmSchedule[] s_declaredGemmSchedules = BuildDeclaredGemmSchedules();

    // The schedules that won a shape in the GTX 1660 Ti sweep (`--kernel-routes --family gemm`); every candidate a shape
    // admits is compiled and timed against the incumbent out of the process's shared tuning budget, so ones that never
    // won are not offered (a 4-way split, the tiled-contraction geometries). The splits' combines read four outputs'
    // partials per vector load: the same speed as one float per load where a split wins (128x128x2048 transposed-A:
    // 74.6 against 74.7 us; 256x256x8192: 798.0 against 798.1), and every global load of the program 128-bit.
    private static DeclaredGemmSchedule[] BuildDeclaredGemmSchedules()
    {
        return new DeclaredGemmSchedule[]
        {
            new(DeclaredGemmLowering.Planner),
            new(DeclaredGemmLowering.ImplicitGemm, 64, 64),
            new(DeclaredGemmLowering.ImplicitGemm, 128, 64),
            new(DeclaredGemmLowering.ImplicitGemm, 64, 64, 8, vectorCombine: true),
            new(DeclaredGemmLowering.ImplicitGemm, 64, 64, 16, vectorCombine: true),
        };
    }

    /// <summary>Every schedule a declared GEMM can lower under, in a fixed order (the index is the schedule's id).</summary>
    internal static IReadOnlyList<DeclaredGemmSchedule> DeclaredGemmSchedules => s_declaredGemmSchedules;

    /// <summary>
    /// The key of a row-major GEMM of <paramref name="family"/>'s layout under schedule <paramref name="schedule"/>;
    /// batch 1 is an unbatched GEMM.
    /// </summary>
    internal static DeclaredKey GemmKey(DeclaredFamily family, int batch, int m, int n, int k, int schedule) =>
        family is DeclaredFamily.Gemm or DeclaredFamily.GemmTransposedA or DeclaredFamily.GemmTransposedB
            ? new DeclaredKey(family, batch, m, n, k, 0, 0, s_declaredGemmSchedules[schedule].Name)
            : throw new ArgumentOutOfRangeException(nameof(family), family, "Not a GEMM family.");

    // The GEMM entry points and the narrow kernels agree to a few ulps of the output scale over K-long sums.
    private const double GemmTolerance = 1e-4;

    private TunedKernelSlot<CudaGemmArgs>? _gemmSlot;

    private const string GemmReferenceId = "backend.gemm.entry";

    internal TunedKernelSlot<CudaGemmArgs> GemmSlot
    {
        get
        {
            if (_gemmSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_gemmSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaGemmArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaGemmArgs>(GemmReferenceId, TunedKernelOrigin.Builtin, true,
                            _ => true, (in CudaGemmArgs a) => RunGemmReference(a)),
                    };
                    for (int i = 0; i < s_declaredGemmSchedules.Length; i++)
                    {
                        int schedule = i;
                        candidates.Add(new CudaTunedKernelCandidate<CudaGemmArgs>(
                            GemmCandidateId(i), TunedKernelOrigin.Generated, true,
                            shape => DeclaredGemmAdmits(shape, schedule),
                            (in CudaGemmArgs a) =>
                            {
                                // Not yet built under a capture (a persisted decision met first there): the entry's own path.
                                if (TryDeclaredGemmSchedule(a.Family, a.A, a.B, a.C, a.Batch, a.M, a.N, a.K, schedule))
                                    System.Threading.Volatile.Write(ref _lastGemmDispatchPath,
                                        (int)CudaGemmDispatchPath.DeclaredProgram);
                                else
                                    RunGemmReference(a);
                            }));
                    }
                    _gemmSlot = new TunedKernelSlot<CudaGemmArgs>(TunedKernelOp.Gemm, TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaGemmArgs>(this, GemmTolerance,
                            a => new[] { (a.C, a.OutputCount) },
                            a => a.C.Handle == a.A.Handle || a.C.Handle == a.B.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_gemmSlot);
            }
        }
    }

    private static readonly string[] s_gemmCandidateIds = BuildGemmCandidateIds();

    private static string[] BuildGemmCandidateIds()
    {
        var ids = new string[s_declaredGemmSchedules.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = "ptx.gemm." + s_declaredGemmSchedules[i].Name;
        return ids;
    }

    private static string GemmCandidateId(int schedule) => s_gemmCandidateIds[schedule];

    // The schedule of a declared candidate, or -1 for the reference.
    private static int GemmScheduleOf(ITunedKernelCandidate<CudaGemmArgs> candidate)
    {
        for (int i = 0; i < s_gemmCandidateIds.Length; i++)
            if (string.Equals(candidate.Id, s_gemmCandidateIds[i], StringComparison.Ordinal)) return i;
        return -1;
    }

    private static readonly string[] s_gemmBiasActivationNames = { "x", "w", "bias", "y" };

    /// <summary>
    /// <c>C = act(A[M,K] W[K,N] + bias)</c> when the GEMM slot serves this shape with a declared schedule: that
    /// schedule's fused-linear program, the bias and ReLU in its epilogue, so the fused call stays bit-identical to
    /// Gemm, BiasAdd and Relu run separately (IGemmBiasActivationBackend's contract). Null when the slot keeps the
    /// entry's own path for the shape (or the route is off): the narrow kernels' own epilogue then matches it. False
    /// when the declared schedule cannot run the fused program here (first met under capture, or declined): the caller
    /// then runs the GEMM and the epilogue separately.
    /// </summary>
    private bool? TryDeclaredGemmBiasActivation(IGpuBuffer a, IGpuBuffer w, IGpuBuffer bias, IGpuBuffer c, int m, int n,
        int k, bool relu)
    {
        if (!s_declaredKernels || !DeclaredKernelsEnabled || m <= 0 || n <= 0 || k <= 0) return null;
        var args = new CudaGemmArgs(DeclaredFamily.Gemm, a, w, c, 1, m, n, k);
        var chosen = GemmSlot.Resolve(args.Shape(), args);
        int schedule = chosen is null ? -1 : GemmScheduleOf(chosen);
        if (schedule < 0) return null;
        var activation = relu ? CodegenActivationKind.ReLU : CodegenActivationKind.None;
        var key = new DeclaredKey(DeclaredFamily.FusedLinear, m, k, n, (int)activation, 0, 0,
            s_declaredGemmSchedules[schedule].Name);
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredPrograms.TryGetValue(key, out program))
            {
                if (IsStreamCapturing()) return false;
                program = BuildUnderGemmSchedule(key, CodegenBackendPrograms.FusedLinear(m, k, n, activation),
                    s_gemmBiasActivationNames, s_declaredGemmSchedules[schedule]);
                _declaredPrograms[key] = program;
            }
        }
        var bound = new DeclaredBindings(a, w, bias, c);
        if (program is null || !AlignedFor(program, s_gemmBiasActivationNames, bound)) return false;
        RunDeclaredProgram(program, s_gemmBiasActivationNames, bound);
        return true;
    }

    // The entry point's own path after the slot (its narrow kernels and cuBLAS) for these arguments.
    private void RunGemmReference(in CudaGemmArgs a)
    {
        if (a.Batch > 1) BatchedGemmIncumbent(a.A, a.B, a.C, a.M, a.N, a.K, a.Batch, 1f, 0f);
        else if (a.Family == DeclaredFamily.GemmTransposedA) MatMulTransposedAIncumbent(a.A, a.B, a.C, a.M, a.N, a.K, 1f, 0f);
        else if (a.Family == DeclaredFamily.GemmTransposedB) MatMulTransposedIncumbent(a.A, a.B, a.C, a.M, a.N, a.K, 1f, 0f);
        else GemmIncumbent(a.A, a.B, a.C, a.M, a.N, a.K, 1f, 0f);
    }

    // Whether a schedule can lower the shape, decided from the spec alone (nothing is compiled). Split schedules only
    // where the output tiles leave SMs idle and each split keeps a long reduction; the planner's lowering only for a
    // few thousand outputs over a long reduction. The gate times candidates as captured-graph replays, where the
    // planner's two launches (a split and its combine) cost nothing extra; eagerly they lost on short reductions
    // (32x10x64: 18.4 us against the narrow kernel's 13.9) and won only on long ones (16x10x4096: 25.1 against 37.3).
    private bool DeclaredGemmAdmits(in TunedShape shape, int schedule)
    {
        if (!DeclaredKernelsEnabled) return false;
        var family = CudaGemmArgs.FamilyOf(shape);
        int batch = shape[1], m = shape[2], n = shape[3], k = shape[4];
        var sc = s_declaredGemmSchedules[schedule];
        long outputs = (long)batch * m * n;
        long tiles = (long)batch * ((m + 63) / 64) * ((n + 63) / 64);
        int units = Math.Max(1, ComputeUnits);
        try
        {
            var spec = CodegenBackendPrograms.Gemm(batch, m, n, k, family == DeclaredFamily.GemmTransposedA,
                family == DeclaredFamily.GemmTransposedB).Passes[0];
            if (sc.Lowering == DeclaredGemmLowering.Planner) return outputs <= 4096 && k >= 1024;
            if (sc.SplitFactor > 1)
                return tiles < units && k >= 64 * sc.SplitFactor &&
                    CodegenSplitReduction.TryPlanChunkedForTiles(spec, sc.SplitFactor) is { } split &&
                    PtxAffineEmitter.PlanImplicitGemm(split.Partial, out _) is not null;
            return PtxAffineEmitter.PlanImplicitGemm(spec, out _) is not null;
        }
        catch (Exception ex) when (ex is NotSupportedException or ArgumentException)
        {
            return false;
        }
    }

    private bool TryDeclaredGemm(DeclaredFamily family, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch, int m, int n,
        int k, float alpha, float beta)
    {
        if (alpha != 1f || beta != 0f || batch <= 0 || m <= 0 || n <= 0 || k <= 0 ||
            !s_declaredKernels || !DeclaredKernelsEnabled)
            return false;
        var args = new CudaGemmArgs(family, a, b, c, batch, m, n, k);
        return GemmSlot.TryExecute(args.Shape(), args);
    }

    /// <summary>
    /// Runs the declared GEMM of this shape under schedule <paramref name="schedule"/>. False -- nothing launched -- when
    /// the route is off, the schedule cannot lower the shape (<see cref="DeclaredDeclineReason"/> says why), or the
    /// program is first met inside a stream capture.
    /// </summary>
    internal bool TryDeclaredGemmSchedule(DeclaredFamily family, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch,
        int m, int n, int k, int schedule)
    {
        if (!DeclaredKernelsEnabled) return false;
        var key = GemmKey(family, batch, m, n, k, schedule);
        if (EnsureDeclaredGemm(key, schedule) is not { } program) return false;
        var bound = new DeclaredBindings(a, b, c);
        if (!AlignedFor(program, s_gemmNames, bound)) return false;
        RunDeclaredProgram(program, s_gemmNames, bound);
        return true;
    }

    /// <summary>
    /// The GEMM as a resident-program stage under schedule <paramref name="schedule"/>, in the routed call's contract
    /// (a [M,K] or transposed, b [K,N] or transposed, c [M,N], each with a leading batch above 1). The stage's program is
    /// the one the routed call runs for the same key.
    /// </summary>
    internal DeclaredResidentStage GemmStage(DeclaredFamily family, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int batch,
        int m, int n, int k, int schedule)
    {
        var key = GemmKey(family, batch, m, n, k, schedule);
        if (!IsStreamCapturing()) EnsureDeclaredGemm(key, schedule);
        var sc = s_declaredGemmSchedules[schedule];
        Func<CodegenKernelSpec, PtxPassLowering?>? lower = sc.Lowering == DeclaredGemmLowering.Planner
            ? null
            : pass => LowerDeclaredConv(pass, new DeclaredConvSchedule(sc.TileM, sc.TileN, sc.SplitFactor), minBlocks: 2,
                sc.VectorCombine);
        return new DeclaredResidentStage(key, s_buildGemm, s_gemmNames, new DeclaredBindings(a, b, c), lower);
    }

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildGemm = key => CodegenBackendPrograms.Gemm(
        key.D0, key.D1, key.D2, key.D3, transA: key.Family == DeclaredFamily.GemmTransposedA,
        transB: key.Family == DeclaredFamily.GemmTransposedB);

    // The key's program, built on first use (never under a capture); null when the schedule cannot lower the shape.
    private DeclaredProgram? EnsureDeclaredGemm(DeclaredKey key, int schedule)
    {
        lock (_directPtxLock)
        {
            if (_declaredPrograms.TryGetValue(key, out var program)) return program;
            if (IsStreamCapturing()) return null;
            program = BuildDeclaredGemm(key, s_declaredGemmSchedules[schedule]);
            _declaredPrograms[key] = program;
            return program;
        }
    }

    private DeclaredProgram? BuildDeclaredGemm(DeclaredKey key, DeclaredGemmSchedule schedule) =>
        BuildUnderGemmSchedule(key, s_buildGemm(key), s_gemmNames, schedule);

    // A one-contraction program lowered under a GEMM schedule; null (with the reason recorded) when it cannot be.
    private DeclaredProgram? BuildUnderGemmSchedule(DeclaredKey key, CodegenProgram source, string[] names,
        DeclaredGemmSchedule schedule)
    {
        string? decline;
        DeclaredProgram? program;
        if (schedule.Lowering == DeclaredGemmLowering.Planner)
            program = BuildDeclaredProgram(source, key.ToString(), names, null, out decline);
        else
        {
            var tiles = new DeclaredConvSchedule(schedule.TileM, schedule.TileN, schedule.SplitFactor);
            program = BuildDeclaredProgram(source, key.ToString(), names,
                pass => LowerDeclaredConv(pass, tiles, minBlocks: 2, schedule.VectorCombine), out decline);
            // As for the convolutions: a ragged shape's guards can spill under the two-blocks-a-SM bound.
            if (program is null && decline is not null && decline.IndexOf("local bytes", StringComparison.Ordinal) >= 0)
                program = BuildDeclaredProgram(source, key.ToString(), names,
                    pass => LowerDeclaredConv(pass, tiles, minBlocks: 0, schedule.VectorCombine), out decline);
        }
        if (decline is not null) _declaredDeclines[key] = decline;
        return program;
    }
}
