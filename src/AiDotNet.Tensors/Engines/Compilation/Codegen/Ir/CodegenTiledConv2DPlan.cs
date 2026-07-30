// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>One exact, replayable tile geometry in the dense 3x3 search space.</summary>
public sealed class CodegenTiledConv2DSchedule
{
    private static readonly IReadOnlyList<CodegenTiledConv2DSchedule> _searchSpace =
        Array.AsReadOnly(new[]
        {
            // Row seven wins backward-data; row fourteen with a wider register tile
            // wins forward. Tested slower geometries are not retained as default work.
            new CodegenTiledConv2DSchedule(16, 7, 8, 4),
            new CodegenTiledConv2DSchedule(16, 14, 8, 8),
            new CodegenTiledConv2DSchedule(16, 4, 8, 8),
            new CodegenTiledConv2DSchedule(16, 7, 8, 8),
            new CodegenTiledConv2DSchedule(16, 4, 8, 4),
            new CodegenTiledConv2DSchedule(32, 2, 8, 8),
            new CodegenTiledConv2DSchedule(32, 4, 8, 8),
            new CodegenTiledConv2DSchedule(8, 14, 8, 4),
            new CodegenTiledConv2DSchedule(8, 7, 8, 4),
            new CodegenTiledConv2DSchedule(16, 4, 4, 8),
            new CodegenTiledConv2DSchedule(16, 4, 16, 8),
            new CodegenTiledConv2DSchedule(16, 7, 8, 4, warpHalo: true),
            new CodegenTiledConv2DSchedule(16, 7, 8, 4, directStream: true),
            new CodegenTiledConv2DSchedule(
                16, 7, 8, 4, warpHalo: true, directStream: true),
            new CodegenTiledConv2DSchedule(16, 7, 8, 4, asymmetricStages: true),
        });

    public CodegenTiledConv2DSchedule(
        int tileM, int tileRows, int tileChannels, int threadTileM,
        bool warpHalo = false, bool directStream = false,
        bool asymmetricStages = false)
    {
        TileM = tileM;
        TileRows = tileRows;
        TileChannels = tileChannels;
        ThreadTileM = threadTileM;
        WarpHalo = warpHalo;
        DirectStream = directStream;
        AsymmetricStages = asymmetricStages;
    }

    public int TileM { get; }
    public int TileRows { get; }
    public int TileChannels { get; }
    public int ThreadTileM { get; }
    public bool WarpHalo { get; }
    public bool DirectStream { get; }
    public bool AsymmetricStages { get; }
    public int ThreadTileWidth => 4;

    /// <summary>Stable name stored in autotune evidence and resolved by the conveyor.</summary>
    public string WinnerName => FormattableString.Invariant(
        $"tiled-conv2d:m{TileM}r{TileRows}c{TileChannels}tm{ThreadTileM}{(WarpHalo ? ":wh" : string.Empty)}{(DirectStream ? ":ds" : string.Empty)}{(AsymmetricStages ? ":asym" : string.Empty)}");

    /// <summary>Finite schedule set shared by identity, measurement, and replay.</summary>
    public static IReadOnlyList<CodegenTiledConv2DSchedule> SearchSpace => _searchSpace;

    public static CodegenTiledConv2DSchedule? Find(string? winner)
    {
        if (string.IsNullOrWhiteSpace(winner)) return null;
        foreach (CodegenTiledConv2DSchedule schedule in _searchSpace)
            if (string.Equals(schedule.WinnerName, winner, StringComparison.Ordinal))
                return schedule;
        return null;
    }
}

/// <summary>
/// One exact composition of a dense-convolution tile and a deterministic reduction split.
/// </summary>
/// <remarks>
/// Cooperative tiles can expose much less device parallelism than the scalar iteration
/// space suggests. Keeping the split beside the exact tile makes that hidden schedule
/// property visible to autotuning, identity, and replay instead of baking a catalog-shape
/// exception into any of them.
/// </remarks>
public sealed class CodegenTiledConv2DSplitSchedule
{
    private static readonly IReadOnlyList<CodegenTiledConv2DSplitSchedule> _searchSpace =
        BuildSearchSpace();

    public CodegenTiledConv2DSplitSchedule(
        CodegenTiledConv2DSchedule tile, int splitFactor)
    {
        Tile = tile ?? throw new ArgumentNullException(nameof(tile));
        if (splitFactor <= 1)
            throw new ArgumentOutOfRangeException(nameof(splitFactor));
        SplitFactor = splitFactor;
    }

    public CodegenTiledConv2DSchedule Tile { get; }
    public int SplitFactor { get; }
    public string WinnerName => Tile.WinnerName + FormattableString.Invariant($":sk{SplitFactor}");
    public static IReadOnlyList<CodegenTiledConv2DSplitSchedule> SearchSpace => _searchSpace;

    public static CodegenTiledConv2DSplitSchedule? Find(string? winner)
    {
        if (string.IsNullOrWhiteSpace(winner)) return null;
        foreach (CodegenTiledConv2DSplitSchedule schedule in _searchSpace)
            if (string.Equals(schedule.WinnerName, winner, StringComparison.Ordinal))
                return schedule;
        return null;
    }

    private static IReadOnlyList<CodegenTiledConv2DSplitSchedule> BuildSearchSpace()
    {
        var schedules = new List<CodegenTiledConv2DSplitSchedule>();
        int[] factors = { 2, 4 };
        foreach (CodegenTiledConv2DSchedule tile in CodegenTiledConv2DSchedule.SearchSpace)
            foreach (int factor in factors)
                schedules.Add(new CodegenTiledConv2DSplitSchedule(tile, factor));
        return schedules.AsReadOnly();
    }
}

/// <summary>A cooperative same-row tile recovered from a dense 3x3 convolution spec.</summary>
/// <remarks>
/// One CTA owns <c>[TileM, outputWidth]</c> for a fixed batch and output row. Each
/// reduction step stages several channels of all three input rows and all nine weights,
/// so input values are reused across output channels and adjacent taps while weights are
/// reused across the complete output row. Both direct and adjoint windows are recognized
/// from their affine maps.
/// </remarks>
public sealed class CodegenTiledConv2DPlan
{
    private CodegenTiledConv2DPlan(
        int matrixInput, int streamInput, int? biasInput,
        int batchAxis, int mAxis, int rowAxis, int columnAxis,
        int reductionChannelAxis, int tapRowAxis, int tapColumnAxis,
        bool matrixReductionMajor, int tapSign, int windowConstant,
        int batch, int m, int outputHeight, int outputWidth,
        int reductionChannels, int physicalReductionChannels,
        int splitFactor, int chunkAxis,
        int inputHeight, int inputWidth,
        int tileM, int tileRows, int tileChannels,
        int threadTileM, int threadTileWidth,
        int stages, bool warpHalo, bool directStream, bool asymmetricStages)
    {
        MatrixInput = matrixInput;
        StreamInput = streamInput;
        BiasInput = biasInput;
        BatchAxis = batchAxis;
        MAxis = mAxis;
        RowAxis = rowAxis;
        ColumnAxis = columnAxis;
        ReductionChannelAxis = reductionChannelAxis;
        TapRowAxis = tapRowAxis;
        TapColumnAxis = tapColumnAxis;
        MatrixReductionMajor = matrixReductionMajor;
        TapSign = tapSign;
        WindowConstant = windowConstant;
        Batch = batch;
        M = m;
        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        ReductionChannels = reductionChannels;
        PhysicalReductionChannels = physicalReductionChannels;
        SplitFactor = splitFactor;
        ChunkAxis = chunkAxis;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        TileM = tileM;
        TileRows = tileRows;
        TileChannels = tileChannels;
        ThreadTileM = threadTileM;
        ThreadTileWidth = threadTileWidth;
        Stages = stages;
        WarpHalo = warpHalo;
        DirectStream = directStream;
        AsymmetricStages = asymmetricStages;
    }

    public int MatrixInput { get; }
    public int StreamInput { get; }
    public int? BiasInput { get; }
    public int BatchAxis { get; }
    public int MAxis { get; }
    public int RowAxis { get; }
    public int ColumnAxis { get; }
    public int ReductionChannelAxis { get; }
    public int TapRowAxis { get; }
    public int TapColumnAxis { get; }
    public bool MatrixReductionMajor { get; }
    public int TapSign { get; }
    public int WindowConstant { get; }
    public int Batch { get; }
    public int M { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int ReductionChannels { get; }
    /// <summary>Physical channel stride, including all deterministic split chunks.</summary>
    public int PhysicalReductionChannels { get; }
    public int SplitFactor { get; }
    public int ChunkAxis { get; }
    public bool IsChunkedPartial => SplitFactor > 1;
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int TileM { get; }
    public int TileRows { get; }
    public int TileChannels { get; }
    public int ThreadTileM { get; }
    public int ThreadTileWidth { get; }
    public int Stages { get; }
    public bool WarpHalo { get; }
    public bool DirectStream { get; }
    public bool AsymmetricStages { get; }
    public int TapRows => 3;
    public int TapColumns => 3;
    public int WindowRows => TileRows + TapRows - 1;
    public int ThreadsM => TileM / ThreadTileM;
    public int ThreadsWidth => (OutputWidth + ThreadTileWidth - 1) / ThreadTileWidth;
    public int ScheduledThreadsWidth => WarpHalo ? ThreadsWidth + 1 : ThreadsWidth;
    public int ThreadsSpatialLogical => TileRows * ThreadsWidth;
    public int ThreadsSpatial
    {
        get
        {
            if (WarpHalo) return TileRows * ScheduledThreadsWidth;
            int logicalThreads = ThreadsM * ThreadsSpatialLogical;
            int scheduledThreads = Math.Max(32, ((logicalThreads + 31) / 32) * 32);
            return (scheduledThreads + ThreadsM - 1) / ThreadsM;
        }
    }
    public int BlockThreads => ThreadsM * ThreadsSpatial;
    public int Steps => ReductionChannels / TileChannels;
    public int Blocks => SplitFactor * Batch * (OutputHeight / TileRows) * (M / TileM);
    public int MatrixStageElements => TileM * TileChannels * TapRows * TapColumns;
    public int StreamStageElements => DirectStream ? 0 : TileChannels * WindowRows * InputWidth;
    public int MatrixStageBytes => MatrixStageElements * sizeof(float);
    public int StreamStageBytes => StreamStageElements * sizeof(float);
    public int StageBytes => (MatrixStageElements + StreamStageElements) * sizeof(float);
    public int BufferToggleBytes => AsymmetricStages ? StreamStageBytes : StageBytes;
    public int SharedMemoryBytes => AsymmetricStages
        ? MatrixStageBytes + Stages * StreamStageBytes
        : Stages * StageBytes;

    public static bool TryCreate(
        CodegenKernelSpec spec, out CodegenTiledConv2DPlan? plan, out string reason)
        => TryCreateCore(spec, null, out plan, out reason);

    /// <summary>Recovers the convolution contract with an exact measured-candidate schedule.</summary>
    public static bool TryCreate(
        CodegenKernelSpec spec, CodegenTiledConv2DSchedule schedule,
        out CodegenTiledConv2DPlan? plan, out string reason)
    {
        if (schedule is null) throw new ArgumentNullException(nameof(schedule));
        return TryCreateCore(spec, schedule, out plan, out reason);
    }

    private static bool TryCreateCore(
        CodegenKernelSpec spec, CodegenTiledConv2DSchedule? schedule,
        out CodegenTiledConv2DPlan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a tiled dense convolution needs an untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.ScaleInput.HasValue ||
            (spec.Activation != CodegenActivationKind.None &&
             spec.Activation != CodegenActivationKind.ReLU) ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the tiled dense convolution accepts only an optional M bias and ReLU";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the tiled dense convolution accumulates and stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the tiled dense convolution currently stages fp32 operands";
                return false;
            }

        if (!TryIdentityOutput(spec, out int[] outputAxes) ||
            outputAxes.Length is not 4 and not 5)
        {
            reason = "the tiled dense-convolution output must be identity [batch,M,H,W] " +
                "with an optional trailing reduction chunk";
            return false;
        }
        int splitFactor = outputAxes.Length == 5 ? spec.Output.Shape[4] : 1;
        int chunkAxis = outputAxes.Length == 5 ? outputAxes[4] : -1;
        if (splitFactor < 1 ||
            (chunkAxis >= 0 && spec.Space.Axes[chunkAxis].IsReduction))
        {
            reason = "the optional trailing chunk must be a non-empty parallel axis";
            return false;
        }
        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 3)
        {
            reason = "a dense 3x3 tile needs channel and two tap reductions";
            return false;
        }

        int matrixInput = -1, reductionChannel = -1, tapRow = -1, tapColumn = -1;
        int physicalReductionChannels = -1;
        bool reductionMajor = false;
        foreach (int input in spec.ProductInputs)
        {
            if (!TryMatrix(spec, spec.Inputs[input], outputAxes[1], reductions,
                    chunkAxis, splitFactor,
                    out int candidateReduction, out int candidateTapRow,
                    out int candidateTapColumn, out bool candidateReductionMajor,
                    out int candidatePhysicalReductionChannels))
                continue;
            if (matrixInput >= 0)
            {
                reason = "both product operands look like the dense 3x3 weight matrix";
                return false;
            }
            matrixInput = input;
            reductionChannel = candidateReduction;
            tapRow = candidateTapRow;
            tapColumn = candidateTapColumn;
            reductionMajor = candidateReductionMajor;
            physicalReductionChannels = candidatePhysicalReductionChannels;
        }
        if (matrixInput < 0)
        {
            reason = "no product operand is [M,C,3,3] or [C,M,3,3]";
            return false;
        }

        int streamInput = spec.ProductInputs[0] == matrixInput
            ? spec.ProductInputs[1]
            : spec.ProductInputs[0];
        if (!TryStream(spec, spec.Inputs[streamInput], outputAxes,
                reductionChannel, tapRow, tapColumn, chunkAxis,
                spec.Space.Axes[reductionChannel].Extent,
                physicalReductionChannels,
                out int tapSign, out int windowConstant))
        {
            reason = "the activation operand is not [batch,C,window(H),window(W)]";
            return false;
        }

        int m = spec.Output.Shape[1];
        if (spec.BiasInput.HasValue &&
            !IsMBias(spec, spec.Inputs[spec.BiasInput.Value], outputAxes[1], m))
        {
            reason = "the tiled dense-convolution bias must be a one-dimensional fp32 M broadcast";
            return false;
        }

        int inputHeight = spec.Inputs[streamInput].Shape[2];
        int inputWidth = spec.Inputs[streamInput].Shape[3];
        int outputHeight = spec.Output.Shape[2];
        int outputWidth = spec.Output.Shape[3];
        if (inputHeight != outputHeight || inputWidth != outputWidth ||
            !((tapSign == 1 && windowConstant == -1) ||
              (tapSign == -1 && windowConstant == 1)))
        {
            reason = "the row tile currently requires same-size padding-one direct or adjoint windows";
            return false;
        }
        if (inputWidth % 4 != 0 || outputWidth % 4 != 0)
        {
            reason = "input and output rows must vectorize by four";
            return false;
        }
        int channels = spec.Space.Axes[reductionChannel].Extent;
        if (physicalReductionChannels != checked(channels * splitFactor))
        {
            reason = "the physical reduction channels must equal chunk channels times chunks";
            return false;
        }
        int tileM = schedule?.TileM ??
            LargestDivisorAtMost(m, reductionMajor ? 16 : 64, 4);
        int tileRows = schedule?.TileRows ??
            LargestDivisorAtMost(outputHeight, 4, 1);
        int tileChannels = schedule?.TileChannels ??
            LargestDivisorAtMost(channels, reductionMajor ? 16 : 8, 4);
        if (tileM == 0 || tileRows == 0 || tileChannels == 0)
        {
            reason = "M, output rows, and reduction channels need supported whole tiles";
            return false;
        }
        int threadTileM = schedule?.ThreadTileM ??
            (tileM >= 32 ? 8 : tileM >= 16 ? 4 : 1);
        const int threadTileWidth = 4;
        if (tileM < 4 || tileRows < 1 || tileChannels < 4 || threadTileM < 1 ||
            m % tileM != 0 || outputHeight % tileRows != 0 ||
            channels % tileChannels != 0 || tileM % threadTileM != 0)
        {
            reason = "the exact schedule must divide M, rows, channels, and its thread tile";
            return false;
        }
        int stages = CodegenSharedMemoryBudget.DoubleBufferStages;
        bool directStream = schedule?.DirectStream ?? false;
        bool asymmetricStages = schedule?.AsymmetricStages ?? false;
        long streamStageElements = directStream
            ? 0
            : tileChannels * (long)(tileRows + 2) * inputWidth;
        long matrixStageElements = tileM * (long)tileChannels * 9;
        long sharedBytes = (asymmetricStages
            ? matrixStageElements + stages * streamStageElements
            : stages * (matrixStageElements + streamStageElements)) * sizeof(float);
        if (!CodegenSharedMemoryBudget.Fits(sharedBytes, out reason))
            return false;

        bool warpHalo = schedule?.WarpHalo ?? false;
        int logicalSpatialThreads = tileRows *
            ((outputWidth + threadTileWidth - 1) / threadTileWidth);
        int threadsM = tileM / threadTileM;
        int scheduledSpatialThreads;
        if (warpHalo)
        {
            int threadsWidth = outputWidth / threadTileWidth;
            int threadsPerScheduledRow = threadsM * (threadsWidth + 1);
            if (threadsPerScheduledRow > 32 || 32 % threadsPerScheduledRow != 0)
            {
                reason = "a warp-halo row must divide one warp exactly";
                return false;
            }
            scheduledSpatialThreads = tileRows * (threadsWidth + 1);
        }
        else
        {
            int logicalThreads = threadsM * logicalSpatialThreads;
            int roundedThreads = Math.Max(32, ((logicalThreads + 31) / 32) * 32);
            scheduledSpatialThreads =
                (roundedThreads + threadsM - 1) / threadsM;
        }
        int threads = threadsM * scheduledSpatialThreads;
        if (threads < 32 || threads > 256 || (warpHalo && threads % 32 != 0))
        {
            reason = "the selected row tile needs " + threads + " threads, outside [32,256]";
            return false;
        }
        plan = new CodegenTiledConv2DPlan(
            matrixInput, streamInput, spec.BiasInput,
            outputAxes[0], outputAxes[1], outputAxes[2], outputAxes[3],
            reductionChannel, tapRow, tapColumn,
            reductionMajor, tapSign, windowConstant,
            spec.Output.Shape[0], m, outputHeight, outputWidth,
            channels, physicalReductionChannels, splitFactor, chunkAxis,
            inputHeight, inputWidth,
            tileM, tileRows, tileChannels, threadTileM, threadTileWidth,
            stages, warpHalo, directStream, asymmetricStages);
        reason = "eligible";
        return true;
    }

    private static bool TryMatrix(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int mAxis, int[] reductions,
        int chunkAxis, int splitFactor,
        out int reductionChannel, out int tapRow, out int tapColumn,
        out bool reductionMajor, out int physicalReductionChannels)
    {
        reductionChannel = tapRow = tapColumn = -1;
        reductionMajor = false;
        physicalReductionChannels = -1;
        if (binding.Shape.Count != 4 || binding.Map.Count != 4) return false;
        if (!TryPlainAxis(binding.Map[2], out int physicalTapRow) ||
            !TryPlainAxis(binding.Map[3], out int physicalTapColumn))
            return false;

        bool firstIsPlain = TryPlainAxis(binding.Map[0], out int firstPlain);
        bool secondIsPlain = TryPlainAxis(binding.Map[1], out int secondPlain);
        bool firstIsReduction = TryReductionAxis(
            spec, binding.Map[0], reductions, chunkAxis, splitFactor, out int firstReduction);
        bool secondIsReduction = TryReductionAxis(
            spec, binding.Map[1], reductions, chunkAxis, splitFactor, out int secondReduction);
        if (firstIsPlain && firstPlain == mAxis && secondIsReduction)
        {
            reductionChannel = secondReduction;
            reductionMajor = false;
        }
        else if (firstIsReduction && secondIsPlain && secondPlain == mAxis)
        {
            reductionChannel = firstReduction;
            reductionMajor = true;
        }
        else return false;

        if (!Contains(reductions, physicalTapRow) ||
            !Contains(reductions, physicalTapColumn) ||
            physicalTapRow == reductionChannel || physicalTapColumn == reductionChannel ||
            physicalTapRow == physicalTapColumn ||
            binding.Shape[2] != 3 || binding.Shape[3] != 3)
            return false;
        int reductionDimension = reductionMajor ? 0 : 1;
        int mDimension = reductionMajor ? 1 : 0;
        int expectedPhysicalChannels = checked(
            spec.Space.Axes[reductionChannel].Extent * splitFactor);
        if (binding.Shape[reductionDimension] != expectedPhysicalChannels ||
            binding.Shape[mDimension] != spec.Space.Axes[mAxis].Extent ||
            binding.Shape[2] != spec.Space.Axes[physicalTapRow].Extent ||
            binding.Shape[3] != spec.Space.Axes[physicalTapColumn].Extent)
            return false;
        tapRow = physicalTapRow;
        tapColumn = physicalTapColumn;
        physicalReductionChannels = binding.Shape[reductionDimension];
        return true;
    }

    private static bool TryStream(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int[] outputAxes,
        int reductionChannel, int tapRow, int tapColumn, int chunkAxis,
        int chunkChannels, int physicalReductionChannels,
        out int tapSign, out int windowConstant)
    {
        tapSign = 0;
        windowConstant = 0;
        if (binding.Shape.Count != 4 || binding.Map.Count != 4) return false;
        if (!TryPlainAxis(binding.Map[0], out int batch) || batch != outputAxes[0] ||
            !TryReductionAxis(spec, binding.Map[1], new[] { reductionChannel },
                chunkAxis, physicalReductionChannels / chunkChannels, out int channel) ||
            channel != reductionChannel)
            return false;
        if (!TryUnitWindow(binding.Map[2], outputAxes[2], tapRow,
                out int rowSign, out int rowConstant) ||
            !TryUnitWindow(binding.Map[3], outputAxes[3], tapColumn,
                out int columnSign, out int columnConstant) ||
            rowSign != columnSign || rowConstant != columnConstant)
            return false;
        if (binding.Shape[0] != spec.Space.Axes[outputAxes[0]].Extent ||
            binding.Shape[1] != physicalReductionChannels)
            return false;
        tapSign = rowSign;
        windowConstant = rowConstant;
        return true;
    }

    private static bool TryReductionAxis(
        CodegenKernelSpec spec, CodegenAffineExpr expression, int[] reductions,
        int chunkAxis, int splitFactor, out int reductionAxis)
    {
        reductionAxis = -1;
        if (expression.Constant != 0 || expression.Divisor != 1 ||
            expression.RequiresExactDivision)
            return false;

        int chunkCoefficient = 0;
        foreach (CodegenAffineTerm term in expression.Terms)
        {
            if (term.Axis == chunkAxis)
            {
                if (chunkAxis < 0 || chunkCoefficient != 0) return false;
                chunkCoefficient = term.Coefficient;
                continue;
            }
            if (term.Coefficient != 1 || !Contains(reductions, term.Axis) ||
                reductionAxis >= 0)
                return false;
            reductionAxis = term.Axis;
        }
        if (reductionAxis < 0) return false;
        if (chunkAxis < 0)
            return expression.Terms.Count == 1 && chunkCoefficient == 0;

        int chunkExtent = spec.Space.Axes[reductionAxis].Extent;
        return expression.Terms.Count == 2 && splitFactor > 1 &&
            chunkCoefficient == chunkExtent;
    }

    private static bool TryUnitWindow(
        CodegenAffineExpr expression, int spatialAxis, int tapAxis,
        out int tapSign, out int constant)
    {
        tapSign = 0;
        constant = 0;
        if (expression.Terms.Count != 2 || expression.Divisor != 1 ||
            expression.RequiresExactDivision)
            return false;
        int spatialCoefficient = 0;
        foreach (var term in expression.Terms)
        {
            if (term.Axis == spatialAxis) spatialCoefficient = term.Coefficient;
            else if (term.Axis == tapAxis) tapSign = term.Coefficient;
            else return false;
        }
        if (spatialCoefficient != 1 || (tapSign != 1 && tapSign != -1)) return false;
        constant = expression.Constant;
        return true;
    }

    private static bool IsMBias(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int mAxis, int m)
    {
        return binding.ElementType == CodegenElementType.Float32 &&
            binding.Shape.Count == 1 && binding.Map.Count == 1 &&
            binding.Shape[0] == m && binding.Shape[0] == spec.Space.Axes[mAxis].Extent &&
            TryPlainAxis(binding.Map[0], out int axis) && axis == mAxis;
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

    private static bool Contains(int[] values, int value)
    {
        for (int i = 0; i < values.Length; i++) if (values[i] == value) return true;
        return false;
    }

    private static int LargestDivisorAtMost(int extent, int maximum, int quantum)
    {
        for (int candidate = Math.Min(extent, maximum); candidate >= quantum; candidate--)
            if (candidate % quantum == 0 && extent % candidate == 0) return candidate;
        return 0;
    }
}

/// <summary>Exact deterministic two-pass program for a split dense-convolution tile.</summary>
public sealed class CodegenTiledConv2DSplitPlan
{
    private CodegenTiledConv2DSplitPlan(
        CodegenTiledConv2DSplitSchedule schedule,
        CodegenSplitPlan split,
        CodegenTiledConv2DPlan partialPlan)
    {
        Schedule = schedule;
        Split = split;
        PartialPlan = partialPlan;
    }

    public CodegenTiledConv2DSplitSchedule Schedule { get; }
    public CodegenSplitPlan Split { get; }
    public CodegenTiledConv2DPlan PartialPlan { get; }

    public static bool TryCreate(
        CodegenKernelSpec spec, CodegenTiledConv2DSplitSchedule schedule,
        out CodegenTiledConv2DSplitPlan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (schedule is null) throw new ArgumentNullException(nameof(schedule));
        plan = null;

        if (!CodegenTiledConv2DPlan.TryCreate(
                spec, schedule.Tile, out CodegenTiledConv2DPlan? original, out reason))
            return false;
        if (original!.ReductionChannels % schedule.SplitFactor != 0)
        {
            reason = "the reduction channels must divide the exact split factor";
            return false;
        }
        int chunkChannels = original.ReductionChannels / schedule.SplitFactor;
        if (chunkChannels < schedule.Tile.TileChannels ||
            chunkChannels % schedule.Tile.TileChannels != 0)
        {
            reason = "each split chunk must contain whole channel tiles";
            return false;
        }

        var (partial, combine) = CodegenSplitReduction.SplitChunked(
            spec, original.ReductionChannelAxis, schedule.SplitFactor);
        if (!CodegenTiledConv2DPlan.TryCreate(
                partial, schedule.Tile, out CodegenTiledConv2DPlan? partialPlan, out reason))
            return false;

        var split = new CodegenSplitPlan(
            partial, combine, partial.Output.ElementCount,
            new[] { original.ReductionChannelAxis });
        plan = new CodegenTiledConv2DSplitPlan(schedule, split, partialPlan!);
        reason = "eligible";
        return true;
    }
}
