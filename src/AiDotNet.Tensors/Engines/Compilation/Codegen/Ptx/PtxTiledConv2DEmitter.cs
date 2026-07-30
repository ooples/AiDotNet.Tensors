// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>Emits a cooperative, double-buffered, true-FP32 dense 3x3 row tile.</summary>
public sealed class PtxTiledConv2DEmitter
{
    private const int FixedR = 16;
    private const int FixedRd = 8;
    private const int FixedP = 4;
    private readonly CodegenTiledConv2DSchedule? _schedule;
    private readonly StringBuilder _body = new();
    private int _r, _rd, _p, _f;

    public CodegenTiledConv2DPlan? Plan { get; private set; }
    public uint LaunchBlocks => checked((uint)(Plan?.Blocks ?? 0));
    public int LaunchBlockThreads => Plan?.BlockThreads ?? 0;
    public int SharedMemoryBytes => Plan?.SharedMemoryBytes ?? 0;

    public PtxTiledConv2DEmitter()
    {
    }

    public PtxTiledConv2DEmitter(CodegenTiledConv2DSchedule schedule) =>
        _schedule = schedule ?? throw new ArgumentNullException(nameof(schedule));

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (computeMajor < 8)
            throw new NotSupportedException("The double-buffered dense-convolution tile requires cp.async on sm_80+.");
        CodegenTiledConv2DPlan? possible;
        string reason;
        bool eligible = _schedule is null
            ? CodegenTiledConv2DPlan.TryCreate(spec, out possible, out reason)
            : CodegenTiledConv2DPlan.TryCreate(
                spec, _schedule, out possible, out reason);
        if (!eligible)
            throw new NotSupportedException("This spec cannot use the tiled dense convolution: " + reason);
        var plan = possible!;

        Plan = plan;
        _body.Clear();
        _r = FixedR;
        _rd = FixedRd;
        _p = FixedP;
        _f = 0;

        int matrixParam = spec.Inputs[plan.MatrixInput].ParameterIndex;
        int streamParam = spec.Inputs[plan.StreamInput].ParameterIndex;
        int outputParam = spec.Output.ParameterIndex;
        int tilesM = plan.M / plan.TileM;
        int rowTiles = plan.OutputHeight / plan.TileRows;

        L($"ld.param.u64 %rd0, [p{I(matrixParam)}];");
        L($"ld.param.u64 %rd1, [p{I(streamParam)}];");
        L($"ld.param.u64 %rd2, [p{I(outputParam)}];");
        L("mov.u64 %rd3, stage;");
        if (plan.BiasInput.HasValue)
        {
            int biasParam = spec.Inputs[plan.BiasInput.Value].ParameterIndex;
            L($"ld.param.u64 %rd4, [p{I(biasParam)}];");
        }
        L("mov.u32 %r0, %ctaid.x;");
        L("mov.u32 %r1, %tid.x;");
        L($"rem.u32 %r2, %r0, {I(tilesM)};             // M tile");
        L($"div.u32 %r3, %r0, {I(tilesM)};");
        L($"rem.u32 %r4, %r3, {I(rowTiles)};           // output-row tile");
        L($"div.u32 %r5, %r3, {I(rowTiles)};");
        if (plan.IsChunkedPartial)
        {
            L($"div.u32 %r13, %r5, {I(plan.Batch)};    // reduction chunk");
            L($"rem.u32 %r5, %r5, {I(plan.Batch)};     // batch");
        }
        L($"mul.lo.u32 %r4, %r4, {I(plan.TileRows)};   // output-row origin");
        if (plan.WarpHalo)
        {
            // One warp owns one output row. M varies fastest, so equal activation
            // fragments form native shared broadcasts and adjacent fragments of the
            // same M lane sit exactly ThreadsM lanes apart for halo shuffles.
            L($"rem.u32 %r7, %r1, {I(plan.ThreadsM)};  // thread M");
            L($"div.u32 %r6, %r1, {I(plan.ThreadsM)};");
            L($"div.u32 %r14, %r6, {I(plan.ScheduledThreadsWidth)}; // local output row");
            L($"rem.u32 %r6, %r6, {I(plan.ScheduledThreadsWidth)};  // thread column");
            L($"setp.lt.u32 %p2, %r6, {I(plan.ThreadsWidth)};");
            L("selp.u32 %r6, %r6, 0, %p2;                 // safe padding lane");
        }
        else
        {
            // Interleave M lanes so every warp sees one activation fragment as a
            // native shared broadcast across ThreadsM consumers. The same mapping
            // also gives each M lane its distinct, conflict-free weight fragment.
            L($"rem.u32 %r7, %r1, {I(plan.ThreadsM)};  // thread M");
            L($"div.u32 %r6, %r1, {I(plan.ThreadsM)};");
            L($"setp.lt.u32 %p2, %r6, {I(plan.ThreadsSpatialLogical)};");
            L("selp.u32 %r6, %r6, 0, %p2;                 // safe padding lane");
            L($"div.u32 %r14, %r6, {I(plan.ThreadsWidth)}; // local output row");
            L($"rem.u32 %r6, %r6, {I(plan.ThreadsWidth)};  // thread column");
        }
        L("add.u32 %r15, %r4, %r14;                    // output row");
        L($"mul.lo.u32 %r8, %r2, {I(plan.TileM)};      // M origin");

        var accumulators = new string[plan.ThreadTileM, plan.ThreadTileWidth];
        for (int i = 0; i < plan.ThreadTileM; i++)
            for (int j = 0; j < plan.ThreadTileWidth; j++)
            {
                accumulators[i, j] = NextF();
                L($"mov.f32 {accumulators[i, j]}, 0f00000000;");
            }

        // r9 is the current channel step, r10 its shared-buffer byte offset.
        // r11/r12 hold the next step/buffer, and p0 is a uniform has-next predicate.
        L("mov.u32 %r9, 0;");
        L("mov.u32 %r10, 0;");
        EmitAsyncStage(plan, "%r9", "%r10");
        L("cp.async.commit_group;");
        L("cp.async.wait_group 0;");
        L("bar.sync 0;");
        L("TILED_CONV_REDUCE:");
        L("add.u32 %r11, %r9, 1;");
        L($"setp.lt.u32 %p0, %r11, {I(plan.Steps)};");
        L($"xor.b32 %r12, %r10, {I(plan.BufferToggleBytes)};");
        L("@!%p0 bra TILED_CONV_COMPUTE;");
        EmitAsyncStage(plan, "%r11", "%r12",
            includeMatrix: !plan.AsymmetricStages, includeStream: true);
        L("cp.async.commit_group;");
        L("TILED_CONV_COMPUTE:");
        EmitCompute(plan, "%r10", "%r9", accumulators);
        L("@!%p0 bra TILED_CONV_DONE;");
        if (plan.AsymmetricStages)
        {
            // The current weight slab can be overwritten only after its consumers have
            // finished. The larger activation slab was already copied concurrently.
            L("bar.sync 0;");
            EmitAsyncStage(plan, "%r11", "%r12",
                includeMatrix: true, includeStream: false);
            L("cp.async.commit_group;");
        }
        L("cp.async.wait_group 0;");
        L("bar.sync 0;");
        L("mov.u32 %r9, %r11;");
        L("mov.u32 %r10, %r12;");
        L("bra TILED_CONV_REDUCE;");
        L("TILED_CONV_DONE:");

        EmitStores(plan, spec.Activation, accumulators);
        L("ret;");

        var text = new StringBuilder();
        // The ignore-source predicate on cp.async was added in PTX ISA 7.5.
        // Ampere targets otherwise inherit the affine emitter's historical 7.1 floor.
        string isaVersion = computeMajor == 8 && computeMinor < 9
            ? "7.5"
            : PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor);
        text.Append(".version ").Append(isaVersion).Append('\n')
            .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
            .Append(".address_size 64\n\n")
            .Append("// generated by PtxTiledConv2DEmitter -- true-fp32 SIMT\n")
            .Append("// row tile ").Append(I(plan.TileM)).Append('x')
            .Append(I(plan.OutputWidth)).Append(", channels ").Append(I(plan.TileChannels))
            .Append(", stages ").Append(I(plan.Stages)).Append('\n')
            .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
            .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            text.Append("    .param .u64 p").Append(I(i))
                .Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        text.Append(")\n{\n")
            .Append("    .shared .align 16 .b8 stage[").Append(I(plan.SharedMemoryBytes))
            .Append("];\n")
            .Append("    .reg .pred %p<").Append(I(_p + 2)).Append(">;\n")
            .Append("    .reg .b32 %r<").Append(I(_r + 2)).Append(">;\n")
            .Append("    .reg .b64 %rd<").Append(I(_rd + 2)).Append(">;\n")
            .Append("    .reg .f32 %f<").Append(I(_f + 2)).Append(">;\n\n")
            .Append(_body)
            .Append("}\n");
        return text.ToString();
    }

    private void EmitAsyncStage(
        CodegenTiledConv2DPlan plan, string step, string bufferBase,
        bool includeMatrix = true, bool includeStream = true)
    {
        const int ValuesPerCopy = 4;
        int streamBase = plan.MatrixStageElements * sizeof(float);

        // The physical matrices are [M,C,tap] or [C,M,tap], neither of which places
        // adjacent M values together. Stage a transposed [C,tap,M] tile with scalar
        // asynchronous copies. The extra copy instructions are paid once per CTA and
        // let every compute thread replace two hot shared loads with one aligned v2 load.
        int matrixElements = plan.MatrixStageElements;
        int matrixPasses = (matrixElements + plan.BlockThreads - 1) / plan.BlockThreads;
        if (includeMatrix)
        {
            for (int pass = 0; pass < matrixPasses; pass++)
            {
                string index = NextR(), localChannel = NextR(), inner = NextR();
                string tap = NextR(), localM = NextR(), globalChannel = NextR();
                string globalM = NextR(), element = NextR(), destinationElement = NextR();
                string bytes = NextRd(), source = NextRd(), sharedBytes = NextRd(), destination = NextRd();
                string valid = NextP();
                L($"add.u32 {index}, %r1, {I(pass * plan.BlockThreads)};");
                L($"setp.lt.u32 {valid}, {index}, {I(matrixElements)};");
                // Assign lanes in physical source order so each warp reads contiguous
                // weights, then transpose only the shared destination to [C,tap,M].
                if (plan.MatrixReductionMajor)
                {
                    L($"div.u32 {localChannel}, {index}, {I(9 * plan.TileM)};");
                    L($"rem.u32 {inner}, {index}, {I(9 * plan.TileM)};");
                    L($"div.u32 {localM}, {inner}, 9;");
                    L($"rem.u32 {tap}, {inner}, 9;");
                }
                else
                {
                    L($"div.u32 {localM}, {index}, {I(9 * plan.TileChannels)};");
                    L($"rem.u32 {inner}, {index}, {I(9 * plan.TileChannels)};");
                    L($"div.u32 {localChannel}, {inner}, 9;");
                    L($"rem.u32 {tap}, {inner}, 9;");
                }
                L($"mad.lo.u32 {globalChannel}, {step}, {I(plan.TileChannels)}, {localChannel};");
                if (plan.IsChunkedPartial)
                    L($"mad.lo.u32 {globalChannel}, %r13, {I(plan.ReductionChannels)}, {globalChannel};");
                L($"add.u32 {globalM}, {localM}, %r8;");
                if (plan.MatrixReductionMajor)
                {
                    L($"mad.lo.u32 {element}, {globalChannel}, {I(plan.M)}, {globalM};");
                }
                else
                {
                    L($"mad.lo.u32 {element}, {globalM}, {I(plan.PhysicalReductionChannels)}, {globalChannel};");
                }
                L($"mad.lo.u32 {element}, {element}, 9, {tap};");
                L($"mul.wide.u32 {bytes}, {element}, 4;");
                L($"add.u64 {source}, %rd0, {bytes};");
                L($"mad.lo.u32 {destinationElement}, {localChannel}, 9, {tap};");
                L($"mad.lo.u32 {destinationElement}, {destinationElement}, {I(plan.TileM)}, {localM};");
                L($"mul.wide.u32 {sharedBytes}, {destinationElement}, 4;");
                L($"add.u64 {destination}, %rd3, {sharedBytes};");
                if (!plan.AsymmetricStages)
                {
                    L($"cvt.u64.u32 {sharedBytes}, {bufferBase};");
                    L($"add.u64 {destination}, {destination}, {sharedBytes};");
                }
                L($"@{valid} cp.async.ca.shared.global [{destination}], [{source}], 4;");
            }
        }

        if (includeStream && !plan.DirectStream)
        {
            int chunksPerInputRow = plan.InputWidth / ValuesPerCopy;
            int inputChunks = plan.StreamStageElements / ValuesPerCopy;
            int inputPasses = (inputChunks + plan.BlockThreads - 1) / plan.BlockThreads;
            for (int pass = 0; pass < inputPasses; pass++)
            {
                string index = NextR(), row = NextR(), column = NextR();
                string localChannel = NextR(), windowRow = NextR(), globalChannel = NextR();
                string sourceRow = NextR(), safeRow = NextR(), element = NextR();
                string bytes = NextRd(), source = NextRd(), sharedBytes = NextRd(), destination = NextRd();
                string valid = NextP(), below = NextP(), above = NextP(), ignore = NextP();
                L($"add.u32 {index}, %r1, {I(pass * plan.BlockThreads)};");
                L($"setp.lt.u32 {valid}, {index}, {I(inputChunks)};");
                L($"div.u32 {row}, {index}, {I(chunksPerInputRow)};");
                L($"rem.u32 {column}, {index}, {I(chunksPerInputRow)};");
                L($"div.u32 {localChannel}, {row}, {I(plan.WindowRows)};");
                L($"rem.u32 {windowRow}, {row}, {I(plan.WindowRows)};");
                L($"mad.lo.u32 {globalChannel}, {step}, {I(plan.TileChannels)}, {localChannel};");
                if (plan.IsChunkedPartial)
                    L($"mad.lo.u32 {globalChannel}, %r13, {I(plan.ReductionChannels)}, {globalChannel};");
                L($"add.s32 {sourceRow}, %r4, {windowRow};");
                L($"add.s32 {sourceRow}, {sourceRow}, -1;");
                L($"setp.lt.s32 {below}, {sourceRow}, 0;");
                L($"setp.ge.s32 {above}, {sourceRow}, {I(plan.InputHeight)};");
                L($"or.pred {ignore}, {below}, {above};");
                L($"max.s32 {safeRow}, {sourceRow}, 0;");
                L($"min.s32 {safeRow}, {safeRow}, {I(plan.InputHeight - 1)};");
                L($"mad.lo.u32 {element}, %r5, {I(plan.PhysicalReductionChannels)}, {globalChannel};");
                L($"mad.lo.u32 {element}, {element}, {I(plan.InputHeight)}, {safeRow};");
                L($"mul.lo.u32 {element}, {element}, {I(plan.InputWidth)};");
                L($"mad.lo.u32 {element}, {column}, {I(ValuesPerCopy)}, {element};");
                L($"mul.wide.u32 {bytes}, {element}, 4;");
                L($"add.u64 {source}, %rd1, {bytes};");
                L($"mul.wide.u32 {sharedBytes}, {index}, 16;");
                L($"add.u64 {destination}, %rd3, {sharedBytes};");
                L($"cvt.u64.u32 {sharedBytes}, {bufferBase};");
                L($"add.u64 {destination}, {destination}, {sharedBytes};");
                L($"@{valid} cp.async.ca.shared.global [{destination}+{I(streamBase)}], [{source}], 16, {ignore};");
            }
        }
    }

    private void EmitCompute(
        CodegenTiledConv2DPlan plan, string bufferBase, string step,
        string[,] accumulators)
    {
        int streamBase = plan.MatrixStageElements * sizeof(float);

        // These thread-local positions do not change during a channel/tap step. Build
        // their shared addresses once so the unrolled reduction uses immediate offsets
        // instead of replaying a mad/cvt/add chain for every operand load.
        string matrixOffset = NextR(), streamRowOffset = NextR();
        string matrixOffset64 = NextRd(), streamRowOffset64 = NextRd();
        string streamColumnBytes = NextRd();
        string matrixBase = NextRd(), streamRowBase = NextRd(), streamThreadBase = NextRd();
        if (plan.AsymmetricStages)
            L($"mul.lo.u32 {matrixOffset}, %r7, " +
              $"{I(plan.ThreadTileM * sizeof(float))};");
        else
            L($"mad.lo.u32 {matrixOffset}, %r7, " +
              $"{I(plan.ThreadTileM * sizeof(float))}, {bufferBase};");
        L($"mad.lo.u32 {streamRowOffset}, %r14, " +
          $"{I(plan.InputWidth * sizeof(float))}, {bufferBase};");
        L($"add.u32 {streamRowOffset}, {streamRowOffset}, {I(streamBase)};");
        L($"cvt.u64.u32 {matrixOffset64}, {matrixOffset};");
        L($"cvt.u64.u32 {streamRowOffset64}, {streamRowOffset};");
        L($"mul.wide.u32 {streamColumnBytes}, %r6, " +
          $"{I(plan.ThreadTileWidth * sizeof(float))};");
        L($"add.u64 {matrixBase}, %rd3, {matrixOffset64};");
        L($"add.u64 {streamRowBase}, %rd3, {streamRowOffset64};");
        L($"add.u64 {streamThreadBase}, {streamRowBase}, {streamColumnBytes};");

        string? sharedHasLeft = null, sharedHasRight = null;
        string? sharedLeftBase = null, sharedRightBase = null;
        if (!plan.DirectStream && !plan.WarpHalo)
        {
            sharedHasLeft = NextP();
            sharedHasRight = NextP();
            sharedLeftBase = NextRd();
            sharedRightBase = NextRd();
            L($"setp.gt.u32 {sharedHasLeft}, %r6, 0;");
            L($"setp.lt.u32 {sharedHasRight}, %r6, {I(plan.ThreadsWidth - 1)};");
            L($"sub.u64 {sharedLeftBase}, {streamThreadBase}, 4;");
            L($"add.u64 {sharedRightBase}, {streamThreadBase}, " +
              $"{I(plan.ThreadTileWidth * sizeof(float))};");
        }

        for (int c = 0; c < plan.TileChannels; c++)
            for (int kh = 0; kh < plan.TapRows; kh++)
            {
                // Four adjacent outputs across all three column taps need only the six
                // activation values [ow-1, ow..ow+3, ow+4]. Load that window once per
                // channel/input-row and reuse overlapping values for every tap.
                int windowTap = plan.TapSign == 1
                    ? kh
                    : plan.TapRows - 1 - kh;
                int rowBase = (c * plan.WindowRows + windowTap) * plan.InputWidth;
                string[] streamWindow;
                if (plan.DirectStream)
                {
                    streamWindow = EmitDirectStreamWindow(
                        plan, step, c, windowTap);
                }
                else
                {
                    streamWindow = new string[plan.ThreadTileWidth + 2];
                    for (int j = 0; j < plan.ThreadTileWidth; j++)
                    {
                        streamWindow[j + 1] = NextF();
                    }
                    L($"ld.shared.v4.f32 " +
                      $"{{{streamWindow[1]}, {streamWindow[2]}, " +
                      $"{streamWindow[3]}, {streamWindow[4]}}}, " +
                      $"[{streamThreadBase}+{I(rowBase * sizeof(float))}];");
                    if (plan.ThreadTileWidth != 4)
                        throw new InvalidOperationException(
                            "The immediate halo path requires the proven four-wide thread tile.");
                    if (!plan.WarpHalo)
                    {
                        streamWindow[0] = NextF();
                        streamWindow[^1] = NextF();
                        L($"mov.f32 {streamWindow[0]}, 0f00000000;");
                        L($"mov.f32 {streamWindow[^1]}, 0f00000000;");
                        int byteOffset = rowBase * sizeof(float);
                        L($"@{sharedHasLeft} ld.shared.f32 {streamWindow[0]}, " +
                          $"[{sharedLeftBase}+{I(byteOffset)}];");
                        L($"@{sharedHasRight} ld.shared.f32 {streamWindow[^1]}, " +
                          $"[{sharedRightBase}+{I(byteOffset)}];");
                    }
                    if (plan.WarpHalo)
                    {
                        string hasLeft = NextP(), hasRight = NextP();
                        streamWindow[0] = NextF();
                        streamWindow[^1] = NextF();
                        L($"shfl.sync.up.b32 {streamWindow[0]}, " +
                          $"{streamWindow[plan.ThreadTileWidth]}, " +
                          $"{I(plan.ThreadsM)}, 0, 0xffffffff;");
                        L($"shfl.sync.down.b32 {streamWindow[^1]}, {streamWindow[1]}, " +
                          $"{I(plan.ThreadsM)}, 31, 0xffffffff;");
                        L($"setp.gt.u32 {hasLeft}, %r6, 0;");
                        L($"setp.lt.u32 {hasRight}, %r6, {I(plan.ThreadsWidth - 1)};");
                        L($"selp.f32 {streamWindow[0]}, {streamWindow[0]}, 0f00000000, {hasLeft};");
                        L($"selp.f32 {streamWindow[^1]}, {streamWindow[^1]}, 0f00000000, {hasRight};");
                    }
                }

                for (int kw = 0; kw < plan.TapColumns; kw++)
                {
                    var matrix = new string[plan.ThreadTileM];
                    int tap = kh * plan.TapColumns + kw;
                    if (matrix.Length == 2 || matrix.Length % 4 == 0)
                    {
                        for (int i = 0; i < matrix.Length; i++) matrix[i] = NextF();
                        int constant = (c * 9 + tap) * plan.TileM * sizeof(float);
                        if (matrix.Length == 2)
                        {
                            L($"ld.shared.v2.f32 {{{matrix[0]}, {matrix[1]}}}, " +
                              $"[{matrixBase}+{I(constant)}];");
                        }
                        else
                            for (int i = 0; i < matrix.Length; i += 4)
                            {
                                L($"ld.shared.v4.f32 {{{matrix[i]}, {matrix[i + 1]}, " +
                                  $"{matrix[i + 2]}, {matrix[i + 3]}}}, " +
                                  $"[{matrixBase}+{I(constant + i * sizeof(float))}];");
                            }
                    }
                    else
                    {
                        for (int i = 0; i < matrix.Length; i++)
                        {
                            matrix[i] = NextF();
                            int constant =
                                ((c * 9 + tap) * plan.TileM + i) * sizeof(float);
                            L($"ld.shared.f32 {matrix[i]}, " +
                              $"[{matrixBase}+{I(constant)}];");
                        }
                    }
                    for (int i = 0; i < matrix.Length; i++)
                        for (int j = 0; j < plan.ThreadTileWidth; j++)
                        {
                            int windowIndex = j + (plan.TapSign == 1
                                ? kw
                                : plan.TapColumns - 1 - kw);
                            L($"fma.rn.f32 {accumulators[i, j]}, {matrix[i]}, " +
                              $"{streamWindow[windowIndex]}, {accumulators[i, j]};");
                        }
                }
            }
    }

    private string EmitScalarStream(
        CodegenTiledConv2DPlan plan, string streamRowBase,
        int rowBase, int columnConstant)
    {
        string sourceColumn = NextR(), safeColumn = NextR();
        string columnBytes = NextRd(), address = NextRd();
        string inRangeLow = NextP(), inRangeHigh = NextP(), inRange = NextP();
        string loaded = NextF(), result = NextF();
        L($"mad.lo.s32 {sourceColumn}, %r6, {I(plan.ThreadTileWidth)}, {I(columnConstant)};");
        L($"setp.ge.s32 {inRangeLow}, {sourceColumn}, 0;");
        L($"setp.lt.s32 {inRangeHigh}, {sourceColumn}, {I(plan.InputWidth)};");
        L($"and.pred {inRange}, {inRangeLow}, {inRangeHigh};");
        L($"max.s32 {safeColumn}, {sourceColumn}, 0;");
        L($"min.s32 {safeColumn}, {safeColumn}, {I(plan.InputWidth - 1)};");
        L($"mul.wide.u32 {columnBytes}, {safeColumn}, 4;");
        L($"add.u64 {address}, {streamRowBase}, {columnBytes};");
        L($"ld.shared.f32 {loaded}, [{address}+{I(rowBase * sizeof(float))}];");
        L($"selp.f32 {result}, {loaded}, 0f00000000, {inRange};");
        return result;
    }

    private string[] EmitDirectStreamWindow(
        CodegenTiledConv2DPlan plan, string step, int localChannel, int windowTap)
    {
        var window = new string[plan.ThreadTileWidth + 2];
        string globalChannel = NextR(), sourceRow = NextR(), safeRow = NextR();
        string rowElement = NextR(), column = NextR();
        string rowBytes = NextRd(), rowBase = NextRd(), columnBytes = NextRd();
        string threadBase = NextRd();
        string below = NextP(), above = NextP(), rowValid = NextP();

        L($"mad.lo.u32 {globalChannel}, {step}, {I(plan.TileChannels)}, {I(localChannel)};");
        if (plan.IsChunkedPartial)
            L($"mad.lo.u32 {globalChannel}, %r13, {I(plan.ReductionChannels)}, {globalChannel};");
        L($"add.s32 {sourceRow}, %r15, {I(windowTap - 1)};");
        L($"setp.ge.s32 {below}, {sourceRow}, 0;");
        L($"setp.lt.s32 {above}, {sourceRow}, {I(plan.InputHeight)};");
        L($"and.pred {rowValid}, {below}, {above};");
        L($"max.s32 {safeRow}, {sourceRow}, 0;");
        L($"min.s32 {safeRow}, {safeRow}, {I(plan.InputHeight - 1)};");
        L($"mad.lo.u32 {rowElement}, %r5, {I(plan.PhysicalReductionChannels)}, {globalChannel};");
        L($"mad.lo.u32 {rowElement}, {rowElement}, {I(plan.InputHeight)}, {safeRow};");
        L($"mul.lo.u32 {rowElement}, {rowElement}, {I(plan.InputWidth)};");
        L($"mul.wide.u32 {rowBytes}, {rowElement}, 4;");
        L($"add.u64 {rowBase}, %rd1, {rowBytes};");
        L($"mul.lo.u32 {column}, %r6, {I(plan.ThreadTileWidth)};");
        L($"mul.wide.u32 {columnBytes}, {column}, 4;");
        L($"add.u64 {threadBase}, {rowBase}, {columnBytes};");

        for (int j = 0; j < plan.ThreadTileWidth; j++)
        {
            window[j + 1] = NextF();
            L($"mov.f32 {window[j + 1]}, 0f00000000;");
        }
        if (plan.WarpHalo)
        {
            string isLoader = NextP(), loadValid = NextP();
            string sourceLane = NextR();
            L($"setp.eq.u32 {isLoader}, %r7, 0;");
            L($"and.pred {loadValid}, {rowValid}, {isLoader};");
            L($"@{loadValid} ld.global.ca.v4.f32 " +
              $"{{{window[1]}, {window[2]}, {window[3]}, {window[4]}}}, [{threadBase}];");
            L($"mul.lo.u32 {sourceLane}, %r6, {I(plan.ThreadsM)};");
            // Multiple scheduled rows can share one warp. shfl.idx names an absolute
            // lane within that warp, so include the packed row's warp-local base before
            // broadcasting the one loader's fragment across its M consumers.
            L($"mad.lo.u32 {sourceLane}, %r14, " +
              $"{I(plan.ThreadsM * plan.ScheduledThreadsWidth)}, {sourceLane};");
            L($"and.b32 {sourceLane}, {sourceLane}, 31;");
            for (int j = 1; j <= plan.ThreadTileWidth; j++)
                L($"shfl.sync.idx.b32 {window[j]}, {window[j]}, {sourceLane}, 31, 0xffffffff;");

            string hasLeft = NextP(), hasRight = NextP();
            window[0] = NextF();
            window[^1] = NextF();
            L($"shfl.sync.up.b32 {window[0]}, {window[plan.ThreadTileWidth]}, " +
              $"{I(plan.ThreadsM)}, 0, 0xffffffff;");
            L($"shfl.sync.down.b32 {window[^1]}, {window[1]}, " +
              $"{I(plan.ThreadsM)}, 31, 0xffffffff;");
            L($"setp.gt.u32 {hasLeft}, %r6, 0;");
            L($"setp.lt.u32 {hasRight}, %r6, {I(plan.ThreadsWidth - 1)};");
            L($"selp.f32 {window[0]}, {window[0]}, 0f00000000, {hasLeft};");
            L($"selp.f32 {window[^1]}, {window[^1]}, 0f00000000, {hasRight};");
        }
        else
        {
            L($"@{rowValid} ld.global.ca.v4.f32 " +
              $"{{{window[1]}, {window[2]}, {window[3]}, {window[4]}}}, [{threadBase}];");
            window[0] = EmitDirectScalarStream(
                plan, rowBase, rowValid, columnConstant: -1);
            window[^1] = EmitDirectScalarStream(
                plan, rowBase, rowValid, columnConstant: plan.ThreadTileWidth);
        }
        return window;
    }

    private string EmitDirectScalarStream(
        CodegenTiledConv2DPlan plan, string rowBase, string rowValid,
        int columnConstant)
    {
        string sourceColumn = NextR(), safeColumn = NextR();
        string columnBytes = NextRd(), address = NextRd();
        string low = NextP(), high = NextP(), columnValid = NextP(), valid = NextP();
        string loaded = NextF(), result = NextF();
        L($"mad.lo.s32 {sourceColumn}, %r6, {I(plan.ThreadTileWidth)}, {I(columnConstant)};");
        L($"setp.ge.s32 {low}, {sourceColumn}, 0;");
        L($"setp.lt.s32 {high}, {sourceColumn}, {I(plan.InputWidth)};");
        L($"and.pred {columnValid}, {low}, {high};");
        L($"and.pred {valid}, {rowValid}, {columnValid};");
        L($"max.s32 {safeColumn}, {sourceColumn}, 0;");
        L($"min.s32 {safeColumn}, {safeColumn}, {I(plan.InputWidth - 1)};");
        L($"mul.wide.u32 {columnBytes}, {safeColumn}, 4;");
        L($"add.u64 {address}, {rowBase}, {columnBytes};");
        L($"ld.global.ca.f32 {loaded}, [{address}];");
        L($"selp.f32 {result}, {loaded}, 0f00000000, {valid};");
        return result;
    }

    private void EmitStores(
        CodegenTiledConv2DPlan plan, CodegenActivationKind activation,
        string[,] accumulators)
    {
        // Every live spatial thread owns one aligned four-column fragment, and the plan
        // admits only widths divisible by four. Form its N/K/H/W base once; channel and
        // column positions within the fragment are compile-time byte offsets.
        string localM = NextR(), globalM = NextR(), column = NextR(), element = NextR();
        string bytes = NextRd(), outputBase = NextRd();
        L($"mul.lo.u32 {localM}, %r7, {I(plan.ThreadTileM)};");
        L($"add.u32 {globalM}, %r8, {localM};");
        L($"mul.lo.u32 {column}, %r6, {I(plan.ThreadTileWidth)};");
        L($"mad.lo.u32 {element}, %r5, {I(plan.M)}, {globalM};");
        L($"mad.lo.u32 {element}, {element}, {I(plan.OutputHeight)}, %r15;");
        L($"mad.lo.u32 {element}, {element}, {I(plan.OutputWidth)}, {column};");
        if (plan.IsChunkedPartial)
            L($"mad.lo.u32 {element}, {element}, {I(plan.SplitFactor)}, %r13;");
        L($"mul.wide.u32 {bytes}, {element}, 4;");
        L($"add.u64 {outputBase}, %rd2, {bytes};");

        if (plan.IsChunkedPartial)
        {
            // The generic deterministic combine wants the chunk as the trailing,
            // contiguous reduction dimension. Partial columns are therefore strided by
            // SplitFactor; scalar stores preserve that layout without a transpose pass.
            for (int i = 0; i < plan.ThreadTileM; i++)
                for (int j = 0; j < plan.ThreadTileWidth; j++)
                {
                    int byteOffset =
                        (i * plan.OutputHeight * plan.OutputWidth + j) *
                        plan.SplitFactor * sizeof(float);
                    L($"@%p2 st.global.f32 [{outputBase}+{I(byteOffset)}], " +
                      $"{accumulators[i, j]};");
                }
            return;
        }

        if (plan.BiasInput.HasValue)
        {
            EmitScalarStores(plan, activation, globalM, outputBase, accumulators);
            return;
        }

        // A bias-free tile owns four naturally aligned adjacent columns. Collapse the
        // four address chains, predicates, and scalar stores into one vector store.
        // Biased epilogues keep the scalar form below: carrying the vector address state
        // across their adds/ReLUs raises register pressure on the dense forward shape.
        for (int i = 0; i < plan.ThreadTileM; i++)
        {
            for (int j = 0; j < plan.ThreadTileWidth; j++)
            {
                if (activation == CodegenActivationKind.ReLU)
                    L($"max.f32 {accumulators[i, j]}, {accumulators[i, j]}, 0f00000000;");
            }
            L($"@%p2 st.global.v4.f32 [{outputBase}+" +
              $"{I(i * plan.OutputHeight * plan.OutputWidth * sizeof(float))}], " +
              $"{{{accumulators[i, 0]}, {accumulators[i, 1]}, " +
              $"{accumulators[i, 2]}, {accumulators[i, 3]}}};");
        }
    }

    private void EmitScalarStores(
        CodegenTiledConv2DPlan plan, CodegenActivationKind activation,
        string globalM, string outputBase, string[,] accumulators)
    {
        string biasBytes = NextRd(), biasBase = NextRd();
        L($"mul.wide.u32 {biasBytes}, {globalM}, 4;");
        L($"add.u64 {biasBase}, %rd4, {biasBytes};");
        for (int i = 0; i < plan.ThreadTileM; i++)
        {
            string bias = NextF();
            L($"ld.global.f32 {bias}, [{biasBase}+{I(i * sizeof(float))}];");
            for (int j = 0; j < plan.ThreadTileWidth; j++)
            {
                L($"add.rn.f32 {accumulators[i, j]}, {accumulators[i, j]}, {bias};");
                if (activation == CodegenActivationKind.ReLU)
                    L($"max.f32 {accumulators[i, j]}, {accumulators[i, j]}, 0f00000000;");
                int byteOffset =
                    (i * plan.OutputHeight * plan.OutputWidth + j) * sizeof(float);
                L($"@%p2 st.global.f32 [{outputBase}+{I(byteOffset)}], {accumulators[i, j]};");
            }
        }
    }

    private string NextR() => "%r" + I(_r++);
    private string NextRd() => "%rd" + I(_rd++);
    private string NextP() => "%p" + I(_p++);
    private string NextF() => "%f" + I(_f++);
    private void L(string line) => _body.Append("    ").Append(line).Append('\n');
    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);

}
