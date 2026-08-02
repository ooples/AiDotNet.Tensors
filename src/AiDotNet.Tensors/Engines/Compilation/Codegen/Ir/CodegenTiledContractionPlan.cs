// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>One exact, replayable SIMT contraction tile in the measured search space.</summary>
public sealed class CodegenTiledContractionSchedule
{
    private static readonly IReadOnlyList<CodegenTiledContractionSchedule> _searchSpace =
        BuildSearchSpace();

    private static IReadOnlyList<CodegenTiledContractionSchedule> BuildSearchSpace()
    {
        var geometries = new[]
        {
            // The model-selected tile is retained separately as "tiled-contraction".
            // These schedules test the two missing architectural levers identified by
            // counters: more output reuse per CTA and fewer synchronization points in K.
            new CodegenTiledContractionSchedule(64, 56, 8, 8, 2),
            new CodegenTiledContractionSchedule(32, 112, 8, 4, 4),
            new CodegenTiledContractionSchedule(64, 112, 8, 8, 4),
            new CodegenTiledContractionSchedule(32, 56, 16, 4, 2),
            new CodegenTiledContractionSchedule(64, 56, 16, 8, 2),
            new CodegenTiledContractionSchedule(32, 112, 16, 4, 4),
            new CodegenTiledContractionSchedule(64, 112, 16, 8, 4),
            new CodegenTiledContractionSchedule(32, 56, 32, 4, 2),
            new CodegenTiledContractionSchedule(64, 56, 32, 8, 2),
        };
        var schedules = new List<CodegenTiledContractionSchedule>(geometries.Length * 2);
        schedules.AddRange(geometries);
        foreach (CodegenTiledContractionSchedule geometry in geometries)
        {
            schedules.Add(new CodegenTiledContractionSchedule(
                geometry.TileM, geometry.TileN, geometry.TileK,
                geometry.ThreadTileM, geometry.ThreadTileN,
                registerPrefetch: true));
        }
        return schedules.AsReadOnly();
    }

    public CodegenTiledContractionSchedule(
        int tileM, int tileN, int tileK, int threadTileM, int threadTileN,
        bool registerPrefetch = false)
    {
        TileM = tileM;
        TileN = tileN;
        TileK = tileK;
        ThreadTileM = threadTileM;
        ThreadTileN = threadTileN;
        RegisterPrefetch = registerPrefetch;
    }

    public int TileM { get; }
    public int TileN { get; }
    public int TileK { get; }
    public int ThreadTileM { get; }
    public int ThreadTileN { get; }
    public bool RegisterPrefetch { get; }
    public string WinnerName => FormattableString.Invariant(
        $"tiled-contraction:m{TileM}n{TileN}k{TileK}tm{ThreadTileM}tn{ThreadTileN}{(RegisterPrefetch ? ":rp" : string.Empty)}");
    public static IReadOnlyList<CodegenTiledContractionSchedule> SearchSpace => _searchSpace;

    public static CodegenTiledContractionSchedule? Find(string? winner)
    {
        if (string.IsNullOrWhiteSpace(winner)) return null;
        foreach (CodegenTiledContractionSchedule schedule in _searchSpace)
            if (string.Equals(schedule.WinnerName, winner, StringComparison.Ordinal))
                return schedule;
        return null;
    }
}

/// <summary>
/// A SIMT FP32 output tile recovered from a semantic contraction, independently of its name.
/// </summary>
/// <remarks>
/// The first accepted form is the pointwise-convolution family
/// <c>out[batch,m,n] = sum(k) matrix[m,k] * stream[batch,k,n]</c>, also accepting
/// the transposed physical matrix layout. The batch and
/// spatial coordinates may each be several physical dimensions; they are flattened only
/// when all participating bindings preserve the same row-major order. Keeping this as an IR
/// plan, instead of recognizing catalog names in the PTX emitter, lets forward and adjoint
/// specs reach the same schedule without maintaining a second description of the operator.
/// An optional exact bias over M and ReLU remain fused in the output epilogue.
/// </remarks>
public sealed class CodegenTiledContractionPlan
{
    private const int DefaultMaximumTileM = 32;
    private const int ReductionMajorMaximumTileM = 64;
    private const int WideMThreadTile = 8;

    private CodegenTiledContractionPlan(
        int matrixInput, int streamInput, int? biasInput, int? scaleInput,
        int mAxis, int reductionAxis,
        int batch, int m, int n, int k, bool matrixReductionMajor,
        int tileM, int tileN, int tileK, int threadTileM, int threadTileN,
        int stages, bool registerPrefetch)
    {
        MatrixInput = matrixInput;
        StreamInput = streamInput;
        BiasInput = biasInput;
        ScaleInput = scaleInput;
        MAxis = mAxis;
        ReductionAxis = reductionAxis;
        Batch = batch;
        M = m;
        N = n;
        K = k;
        MatrixReductionMajor = matrixReductionMajor;
        TileM = tileM;
        TileN = tileN;
        TileK = tileK;
        ThreadTileM = threadTileM;
        ThreadTileN = threadTileN;
        Stages = stages;
        RegisterPrefetch = registerPrefetch;
    }

    /// <summary>Product operand containing only the M and K axes.</summary>
    public int MatrixInput { get; }

    /// <summary>Product operand containing batch, K and the flattened N axes.</summary>
    public int StreamInput { get; }

    /// <summary>Optional one-dimensional fp32 bias broadcast over M.</summary>
    public int? BiasInput { get; }

    /// <summary>Optional one-dimensional fp32 post-bias scale broadcast over M.</summary>
    public int? ScaleInput { get; }

    /// <summary>The spec axis represented by M.</summary>
    public int MAxis { get; }

    /// <summary>The single contracted spec axis.</summary>
    public int ReductionAxis { get; }

    /// <summary>Product of output dimensions physically before M.</summary>
    public int Batch { get; }

    /// <summary>Rows of the per-batch output matrix.</summary>
    public int M { get; }

    /// <summary>Product of output dimensions physically after M.</summary>
    public int N { get; }

    /// <summary>Contracted extent.</summary>
    public int K { get; }

    /// <summary>Whether the matrix is physically [K,M], making each staged M row contiguous.</summary>
    public bool MatrixReductionMajor { get; }

    /// <summary>CTA output rows.</summary>
    public int TileM { get; }

    /// <summary>CTA output columns.</summary>
    public int TileN { get; }

    /// <summary>Contraction depth staged per step.</summary>
    public int TileK { get; }

    /// <summary>Output rows accumulated by one thread.</summary>
    public int ThreadTileM { get; }

    /// <summary>Output columns accumulated by one thread.</summary>
    public int ThreadTileN { get; }

    /// <summary>Shared-memory buffers used by the contraction pipeline.</summary>
    public int Stages { get; }

    /// <summary>Whether the next shared-memory K fragment is prefetched into registers.</summary>
    public bool RegisterPrefetch { get; }

    /// <summary>Threads along the M side of the CTA tile.</summary>
    public int ThreadsM => TileM / ThreadTileM;

    /// <summary>Threads along the N side of the CTA tile.</summary>
    public int ThreadsN => TileN / ThreadTileN;

    /// <summary>Threads launched per CTA.</summary>
    public int BlockThreads => ThreadsM * ThreadsN;

    /// <summary>Number of contraction steps.</summary>
    public int Steps => K / TileK;

    /// <summary>CTAs launched by this plan.</summary>
    public int Blocks => Batch * (M / TileM) * (N / TileN);

    /// <summary>FP32 bytes in one shared-memory stage.</summary>
    public int StageBytes => TileK * (TileM + TileN) * sizeof(float);

    /// <summary>Total dynamic-free shared-memory requirement.</summary>
    public int SharedMemoryBytes => Stages * StageBytes;

    /// <summary>
    /// Recovers a tiled pointwise contraction from index maps and chooses a whole-tile
    /// schedule. A refusal explains the first violated semantic or layout invariant.
    /// </summary>
    public static bool TryCreate(
        CodegenKernelSpec spec, out CodegenTiledContractionPlan? plan, out string reason)
        => TryCreate(spec, null, out plan, out reason);

    /// <summary>Recovers the contraction and applies an exact measured schedule.</summary>
    public static bool TryCreate(
        CodegenKernelSpec spec, CodegenTiledContractionSchedule? schedule,
        out CodegenTiledContractionPlan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a tiled contraction needs an untransformed sum of exactly two operands";
            return false;
        }

        if (spec.PreBiasInput.HasValue || spec.ReduceScale != 1.0 ||
            (spec.Activation != CodegenActivationKind.None &&
             spec.Activation != CodegenActivationKind.ReLU) ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the tiled path accepts only optional M bias/scale and ReLU epilogue";
            return false;
        }

        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the SIMT contraction accumulates and stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the SIMT contraction currently stages fp32 operands";
                return false;
            }

        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 1)
        {
            reason = "the pointwise tiled form has one contraction axis; this spec has " +
                reductions.Length;
            return false;
        }
        int reduction = reductions[0];

        if (!TryIdentityOutput(spec, out int[] outputAxes))
        {
            reason = "the tiled store requires an identity, row-major output map";
            return false;
        }

        int matrixInput = -1, mAxis = -1;
        bool reductionMajor = false;
        foreach (int input in spec.ProductInputs)
        {
            if (!TryMatrix(spec, spec.Inputs[input], reduction, out int candidateM,
                    out bool candidateReductionMajor))
                continue;
            if (matrixInput >= 0)
            {
                reason = "both operands look like the rank-2 matrix; the streamed operand is ambiguous";
                return false;
            }
            matrixInput = input;
            mAxis = candidateM;
            reductionMajor = candidateReductionMajor;
        }
        if (matrixInput < 0)
        {
            reason = "no product operand is a plain rank-2 matrix over M and K";
            return false;
        }

        int mDimension = Array.IndexOf(outputAxes, mAxis);
        if (mDimension < 0 || mDimension == outputAxes.Length - 1)
        {
            reason = "M must be an output dimension with at least one contiguous N dimension after it";
            return false;
        }

        int streamInput = spec.ProductInputs[0] == matrixInput
            ? spec.ProductInputs[1]
            : spec.ProductInputs[0];
        if (!IsStreamedBinding(
                spec, spec.Inputs[streamInput], outputAxes, mDimension, reduction))
        {
            reason = "the other operand is not the output layout with M replaced by K";
            return false;
        }
        if (spec.BiasInput.HasValue &&
            !IsMBias(spec.Inputs[spec.BiasInput.Value], mAxis, spec.Space.Axes[mAxis].Extent))
        {
            reason = "the tiled epilogue bias must be a one-dimensional fp32 broadcast over M";
            return false;
        }
        if (spec.ScaleInput.HasValue &&
            !IsMBias(spec.Inputs[spec.ScaleInput.Value], mAxis, spec.Space.Axes[mAxis].Extent))
        {
            reason = "the tiled epilogue scale must be a one-dimensional fp32 broadcast over M";
            return false;
        }

        int batch = 1, n = 1;
        for (int d = 0; d < mDimension; d++) batch = checked(batch * spec.Output.Shape[d]);
        for (int d = mDimension + 1; d < spec.Output.Shape.Count; d++)
            n = checked(n * spec.Output.Shape[d]);
        int m = spec.Space.Axes[mAxis].Extent;
        int k = spec.Space.Axes[reduction].Extent;

        // A [K,M] matrix exposes a contiguous M row to each async copy. Owning the full
        // 64-wide row halves the CTA count and amortizes each streamed value across twice
        // as many outputs; [M,K] retains the smaller tile because its copies run along K.
        int tileM = schedule?.TileM ?? LargestDivisorAtMost(m,
            reductionMajor ? ReductionMajorMaximumTileM : DefaultMaximumTileM, 4);
        int tileN = schedule?.TileN ?? LargestDivisorAtMost(n, 64, 4);
        // A physically [M,K] matrix is copied along K, so each async copy needs four
        // adjacent values.  [K,M] copies along M and has no corresponding K constraint.
        int tileK = schedule?.TileK ??
            LargestDivisorAtMost(k, 8, reductionMajor ? 1 : 4);
        bool matrixCopyAligned = reductionMajor ? tileM % 4 == 0 : tileK % 4 == 0;
        if (tileM <= 0 || tileN <= 0 || tileK <= 0 ||
            m % tileM != 0 || n % tileN != 0 || k % tileK != 0 ||
            tileN % 4 != 0 || !matrixCopyAligned)
        {
            reason = schedule is null
                ? "the output or contraction extent has no supported whole tile"
                : "the requested schedule is not a whole, 16-byte-aligned tile";
            return false;
        }

        int threadTileM = schedule?.ThreadTileM ??
            (tileM >= ReductionMajorMaximumTileM
                ? WideMThreadTile
                : tileM >= 16 ? 4 : tileM >= 4 ? 2 : 1);
        int threadTileN = schedule?.ThreadTileN ?? (tileN >= 8 ? 2 : 1);
        if (threadTileM <= 0 || threadTileN <= 0 ||
            tileM % threadTileM != 0 || tileN % threadTileN != 0)
        {
            reason = "the requested thread tile does not divide the CTA tile";
            return false;
        }
        int threads = (tileM / threadTileM) * (tileN / threadTileN);
        if (threads < 32 || threads > 256)
        {
            reason = "the selected whole tile needs " + threads + " threads, outside [32,256]";
            return false;
        }
        int stages = CodegenSharedMemoryBudget.DoubleBufferStages;
        long sharedBytes = stages * tileK * ((long)tileM + tileN) * sizeof(float);
        if (!CodegenSharedMemoryBudget.Fits(sharedBytes, out reason))
            return false;

        plan = new CodegenTiledContractionPlan(
            matrixInput, streamInput, spec.BiasInput, spec.ScaleInput,
            mAxis, reduction, batch, m, n, k,
            reductionMajor, tileM, tileN, tileK, threadTileM, threadTileN,
            stages, schedule?.RegisterPrefetch ?? false);
        reason = "eligible";
        return true;
    }

    private static bool TryIdentityOutput(CodegenKernelSpec spec, out int[] axes)
    {
        axes = new int[spec.Output.Map.Count];
        if (spec.Output.Shape.Count != spec.Output.Map.Count) return false;
        var seen = new HashSet<int>();
        for (int d = 0; d < axes.Length; d++)
        {
            if (!TryPlainAxis(spec.Output.Map[d], out axes[d])) return false;
            if (spec.Space.Axes[axes[d]].IsReduction || !seen.Add(axes[d])) return false;
            if (spec.Output.Shape[d] != spec.Space.Axes[axes[d]].Extent) return false;
        }
        return seen.Count == spec.Space.ParallelAxes.Length;
    }

    private static bool TryMatrix(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int reduction, out int mAxis,
        out bool reductionMajor)
    {
        mAxis = -1;
        reductionMajor = false;
        if (binding.Shape.Count != 2 || binding.Map.Count != 2) return false;
        if (!TryPlainAxis(binding.Map[0], out int first) ||
            !TryPlainAxis(binding.Map[1], out int second)) return false;
        if (binding.Shape[0] != spec.Space.Axes[first].Extent ||
            binding.Shape[1] != spec.Space.Axes[second].Extent)
            return false;
        if (first == reduction && second != reduction)
        {
            mAxis = second;
            reductionMajor = true;
            return true;
        }
        if (second == reduction && first != reduction)
        {
            mAxis = first;
            reductionMajor = false;
            return true;
        }
        return false;
    }

    private static bool IsStreamedBinding(
        CodegenKernelSpec spec, CodegenTensorBinding binding,
        int[] outputAxes, int mDimension, int reduction)
    {
        if (binding.Shape.Count != outputAxes.Length || binding.Map.Count != outputAxes.Length)
            return false;
        for (int d = 0; d < outputAxes.Length; d++)
        {
            int expected = d == mDimension ? reduction : outputAxes[d];
            if (!TryPlainAxis(binding.Map[d], out int actual) || actual != expected ||
                binding.Shape[d] != spec.Space.Axes[actual].Extent)
                return false;
        }
        return true;
    }

    private static bool IsMBias(CodegenTensorBinding binding, int mAxis, int m)
    {
        return binding.ElementType == CodegenElementType.Float32 &&
            binding.Shape.Count == 1 && binding.Map.Count == 1 &&
            binding.Shape[0] == m && TryPlainAxis(binding.Map[0], out int axis) &&
            axis == mAxis;
    }

    private static bool TryPlainAxis(CodegenAffineExpr expression, out int axis)
    {
        axis = -1;
        if (expression.Terms.Count != 1 || expression.Terms[0].Coefficient != 1 ||
            expression.Constant != 0 || expression.Divisor != 1 ||
            expression.RequiresExactDivision)
            return false;
        axis = expression.Terms[0].Axis;
        return true;
    }

    private static int LargestDivisorAtMost(int extent, int maximum, int quantum)
    {
        for (int candidate = Math.Min(extent, maximum); candidate >= quantum; candidate--)
            if (candidate % quantum == 0 && extent % candidate == 0) return candidate;
        return 0;
    }
}
