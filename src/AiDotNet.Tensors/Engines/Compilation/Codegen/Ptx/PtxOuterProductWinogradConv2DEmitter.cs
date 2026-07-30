// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// True-fp32 Winograd F(2,3) main pass with an 8x8 register outer product.
/// Each thread owns one transform component and one 8x8 (M,tile) microtile;
/// the CTA transposes completed components through shared memory before the
/// inverse transform. This gives every staged operand four-way register reuse.
/// </summary>
public sealed class PtxOuterProductWinogradConv2DEmitter
{
    private const int BlockM = 32;
    private const int BlockChannels = 8;
    private const int BlockTiles = 32;
    private const int TilePartitions = 64 / BlockTiles;
    private const int TransformElements = 16;
    private const int MicroM = 8;
    private const int MicroTiles = 8;
    private const int PhysicalMicroTiles = 7;
    private const int ReciprocalShift = 10;
    private const int PhysicalMicroTilesReciprocal =
        ((1 << ReciprocalShift) + PhysicalMicroTiles - 1) / PhysicalMicroTiles;
    private const int AccumulatorElements = MicroM * PhysicalMicroTiles;
    private const int MGroups = BlockM / MicroM;
    private const int TileGroups = BlockTiles / MicroTiles;
    private const int MicroGroups = MGroups * TileGroups;
    private const int UStride = 36;
    private const int VStride = 36;
    private const int BlockThreads = 256;
    private const int UElements = BlockChannels * TransformElements * UStride;
    private const int VElements = BlockChannels * TransformElements * VStride;
    private const int VByteOffset = UElements * sizeof(float);
    private const int BufferBytes = (UElements + VElements) * sizeof(float);
    private const int DoubleBufferBytes = BufferBytes * 2;
    private const int WorkspaceGroups = MicroGroups;
    private const int WorkspaceElementStride = 17;
    private const int WorkspaceGroupStride = AccumulatorElements * WorkspaceElementStride + 10;
    private const int WorkspaceBytes = WorkspaceGroups * WorkspaceGroupStride * sizeof(float);
    private const int SharedBytes = DoubleBufferBytes > WorkspaceBytes
        ? DoubleBufferBytes
        : WorkspaceBytes;

    private readonly StringBuilder _body = new(131072);

    static PtxOuterProductWinogradConv2DEmitter()
    {
        // The reciprocal is applied after pair/MicroM, so this is its complete
        // emitted input range. Fail at type initialization if changed geometry
        // makes the strength reduction inexact.
        int maximum = (WorkspaceGroups * AccumulatorElements - 1) / MicroM;
        for (int value = 0; value <= maximum; value++)
            if (((value * PhysicalMicroTilesReciprocal) >> ReciprocalShift) !=
                value / PhysicalMicroTiles)
                throw new InvalidOperationException(
                    "The Winograd reciprocal quotient is not exact for the emitted range.");
    }

    public CodegenTiledConv2DPlan? Plan { get; private set; }
    public string? EntryPoint { get; private set; }
    public uint LaunchBlocks { get; private set; }
    public int LaunchBlockThreads => BlockThreads;
    public int SharedMemoryBytes => SharedBytes;

    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (computeMajor < 8)
            throw new NotSupportedException("The outer-product Winograd pass requires sm_80+.");
        if (!CodegenTiledConv2DPlan.TryCreate(spec, out var possible, out string reason))
            throw new NotSupportedException("This spec cannot use Winograd: " + reason);
        CodegenTiledConv2DPlan plan = possible!;
        int tileRows = plan.OutputHeight / 2;
        int tileColumns = plan.OutputWidth / 2;
        if (plan.WindowConstant != 1 || plan.InputHeight != plan.OutputHeight ||
            plan.InputWidth != plan.OutputWidth ||
            (plan.OutputHeight & 1) != 0 || (plan.OutputWidth & 1) != 0 ||
            plan.M != BlockM || plan.ReductionChannels % BlockChannels != 0 ||
            tileRows % PhysicalMicroTiles != 0 ||
            tileColumns % PhysicalMicroTiles != 0 ||
            plan.BiasInput.HasValue || spec.Activation != CodegenActivationKind.None)
            throw new NotSupportedException(
                "The outer-product Winograd pass requires bias-free M32, C%8, " +
                "even same-shaped 3x3 geometry, and a 7x7-divisible tile grid.");

        Plan = plan;
        // Keep the semantic spec's entry point so this lowering can be replayed by
        // the same conveyor path as every other autotune winner.
        EntryPoint = spec.Name;
        int quadrantRows = tileRows / PhysicalMicroTiles;
        int quadrantColumns = tileColumns / PhysicalMicroTiles;
        int quadrants = quadrantRows * quadrantColumns;
        LaunchBlocks = checked((uint)(plan.Batch * quadrants * TilePartitions));
        _body.Clear();

        int matrixParam = spec.Inputs[plan.MatrixInput].ParameterIndex;
        int streamParam = spec.Inputs[plan.StreamInput].ParameterIndex;
        int outputParam = spec.Output.ParameterIndex;
        L($"ld.param.u64 %rd0, [p{I(matrixParam)}];     // reduction-major raw 3x3 filters");
        L($"ld.param.u64 %rd1, [p{I(streamParam)}];");
        L($"ld.param.u64 %rd2, [p{I(outputParam)}];");
        L("mov.u32 %r0, %tid.x;");
        L("mov.u32 %r1, %ctaid.x;");
        EmitRemainder("%r2", "%r1", TilePartitions, "tile partition");
        EmitQuotient("%r3", "%r1", TilePartitions, null);
        EmitRemainder("%r4", "%r3", quadrants, "quadrant");
        EmitQuotient("%r5", "%r3", quadrants, "batch");
        L("shr.u32 %r6, %r0, 4;                       // transform component");
        L("and.b32 %r7, %r0, 15;                      // 8x8 microtile group");
        EmitRemainder("%r8", "%r7", MGroups, "M group");
        EmitQuotient("%r9", "%r7", MGroups, "tile group");
        EmitQuotient("%r10", "%r4", quadrantColumns, null);
        L($"mul.lo.u32 %r10, %r10, {I(PhysicalMicroTiles)};" +
          "                 // quadrant tile row");
        EmitRemainder("%r11", "%r4", quadrantColumns, null);
        L($"mul.lo.u32 %r11, %r11, {I(PhysicalMicroTiles)};" +
          "                 // quadrant tile column");
        EmitVGeometry(plan);
        EmitUCoordinates(plan);
        L("mov.u32 %r12, 0;                           // C stage");
        L($"mov.u32 %r36, {I(plan.ReductionChannels / BlockChannels - 1)}; // final pipelined stage");
        L("mov.u64 %rd10, stage;                      // current stage buffer");
        L("mov.u64 %rd11, stage;");
        L($"add.u64 %rd11, %rd11, {I(BufferBytes)};   // next stage buffer");
        for (int f = 0; f < AccumulatorElements; f++)
            L($"mov.f32 %f{I(f)}, 0f00000000;");

        EmitStagePreloads(plan, "%r12", "OUTER_INITIAL");
        EmitStageStores("%rd10", "OUTER_INITIAL");
        L("bar.sync 0;");

        L("OUTER_WINO_PIPELINE:");
        L("add.u32 %r33, %r12, 1;                    // stage being prefetched");
        EmitStagePreloads(plan, "%r33", "OUTER_NEXT");
        EmitOuterProducts("%rd10");
        EmitStageStores("%rd11", "OUTER_NEXT");
        L("bar.sync 0;");
        L("mov.u64 %rd7, %rd10;");
        L("mov.u64 %rd10, %rd11;");
        L("mov.u64 %rd11, %rd7;");
        L("add.u32 %r12, %r12, 1;");
        L("setp.lt.u32 %p0, %r12, %r36;");
        L("@%p0 bra OUTER_WINO_PIPELINE;");
        EmitOuterProducts("%rd10");
        L("bar.sync 0;");
        EmitOutput(plan);
        L("ret;");

        var ptx = new StringBuilder(_body.Length + 2048);
        ptx.Append(".version ")
            .Append(PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
            .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
            .Append(".address_size 64\n\n")
            .Append("// generated by PtxOuterProductWinogradConv2DEmitter -- true-fp32 SIMT\n")
            .Append("// M32 x 32 tiles x F(2,3), component-owned 8x8 outer product\n")
            .Append(".extern .shared .align 16 .b8 stage[];\n\n")
            .Append(".visible .entry ").Append(EntryPoint).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            ptx.Append("    .param .u64 p").Append(I(i))
                .Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        ptx.Append(")\n{\n")
            .Append("    .reg .pred %p<16>;\n")
            .Append("    .reg .b32 %r<40>;\n")
            .Append("    .reg .b64 %rd<16>;\n")
            .Append("    .reg .f32 %f<136>;\n\n")
            .Append(_body)
            .Append("}\n");
        return ptx.ToString();
    }

    private void EmitStagePreloads(
        CodegenTiledConv2DPlan plan, string stageRegister, string labelPrefix)
    {
        int hw = plan.InputHeight * plan.InputWidth;
        L($"@!%p14 bra {labelPrefix}_V_LOAD_DONE;");
        L($"mad.lo.u32 %r18, {stageRegister}, {I(BlockChannels)}, %r26;");
        L($"mad.lo.u32 %r19, %r5, {I(plan.ReductionChannels)}, %r18;");
        L($"mul.lo.u32 %r19, %r19, {I(hw)};");
        L("mul.wide.u32 %rd4, %r19, 4;");
        L("add.u64 %rd4, %rd1, %rd4;");
        L("mul.wide.s32 %rd5, %r31, 4;");
        L("add.u64 %rd5, %rd4, %rd5;");
        for (int di = 0; di < 4; di++)
        {
            int f = 96 + di * 4;
            for (int dj = 0; dj < 4; dj++)
                L($"mov.f32 %f{I(f + dj)}, 0f00000000;");

            L($"and.pred %p2, %p{I(5 + di)}, %p9;");
            int rowOffset = di * plan.InputWidth * sizeof(float);
            L($"@%p2 ld.global.ca.f32 %f{I(f)}, [%rd5+{I(rowOffset)}];");

            // The two center columns are in bounds for every even-sized F(2,3)
            // output tile, and patchLeft + 1 is even. One naturally aligned v2
            // load therefore replaces two independently issued scalar loads.
            int middleOffset = (di * plan.InputWidth + 1) * sizeof(float);
            L($"@%p{I(5 + di)} ld.global.ca.v2.f32 " +
              $"{{%f{I(f + 1)}, %f{I(f + 2)}}}, [%rd5+{I(middleOffset)}];");

            L($"and.pred %p2, %p{I(5 + di)}, %p12;");
            int rightOffset = (di * plan.InputWidth + 3) * sizeof(float);
            L($"@%p2 ld.global.ca.f32 %f{I(f + 3)}, [%rd5+{I(rightOffset)}];");
        }
        L(labelPrefix + "_V_LOAD_DONE:");
        L("and.b32 %r13, %r0, 31;                    // filter M");
        L("shr.u32 %r14, %r0, 5;                     // filter local C");
        L($"mad.lo.u32 %r14, {stageRegister}, {I(BlockChannels)}, %r14;");
        L($"mad.lo.u32 %r19, %r14, {I(plan.M)}, %r13;");
        L("mul.wide.u32 %rd4, %r19, 36;");
        L("add.u64 %rd4, %rd0, %rd4;");
        for (int gi = 0; gi < 3; gi++)
            for (int gj = 0; gj < 3; gj++)
            {
                int source = (2 - gi) * 3 + (2 - gj);
                L($"ld.global.nc.f32 %f{I(112 + gi * 3 + gj)}, [%rd4+{I(source * 4)}];");
            }
    }

    private void EmitStageStores(string bufferBase, string labelPrefix)
    {
        L($"@!%p14 bra {labelPrefix}_V_STORE_DONE;");
        for (int column = 0; column < 4; column++)
        {
            int d0 = 96 + column, d1 = 100 + column;
            int d2 = 104 + column, d3 = 108 + column;
            L($"add.rn.f32 %f128, %f{I(d1)}, %f{I(d2)};");
            L($"sub.rn.f32 %f{I(d0)}, %f{I(d0)}, %f{I(d2)};");
            L($"sub.rn.f32 %f{I(d3)}, %f{I(d1)}, %f{I(d3)};");
            L($"sub.rn.f32 %f{I(d2)}, %f{I(d2)}, %f{I(d1)};");
            L($"mov.f32 %f{I(d1)}, %f128;");
        }
        L($"mov.u64 %rd6, {bufferBase};");
        L($"add.u64 %rd6, %rd6, {I(VByteOffset)};");
        for (int row = 0; row < 4; row++)
        {
            int x0 = 96 + row * 4, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
            L($"sub.rn.f32 %f129, %f{I(x0)}, %f{I(x2)};");
            L($"add.rn.f32 %f130, %f{I(x1)}, %f{I(x2)};");
            L($"sub.rn.f32 %f131, %f{I(x2)}, %f{I(x1)};");
            L($"sub.rn.f32 %f132, %f{I(x1)}, %f{I(x3)};");
            L($"mad.lo.u32 %r19, %r26, 16, {I(row * 4)};");
            L($"mad.lo.u32 %r19, %r19, {I(VStride)}, %r27;");
            L("mul.wide.u32 %rd5, %r19, 4;");
            L("add.u64 %rd5, %rd6, %rd5;");
            for (int component = 0; component < 4; component++)
                L($"st.shared.f32 [%rd5+{I(component * VStride * sizeof(float))}], " +
                  $"%f{I(129 + component)};");
        }
        L(labelPrefix + "_V_STORE_DONE:");
        EmitInlineUTransformStores(bufferBase);
    }

    private void EmitInlineUTransformStores(string bufferBase)
    {
        for (int column = 0; column < 3; column++)
        {
            int g0 = 112 + column;
            int g1 = 115 + column;
            int g2 = 118 + column;
            L($"mov.f32 %f{I(121 + column)}, %f{I(g0)};");
            L($"add.rn.f32 %f133, %f{I(g0)}, %f{I(g1)};");
            L($"add.rn.f32 %f133, %f133, %f{I(g2)};");
            L($"mul.rn.f32 %f{I(124 + column)}, %f133, 0f3F000000;");
            L($"sub.rn.f32 %f133, %f{I(g0)}, %f{I(g1)};");
            L($"add.rn.f32 %f133, %f133, %f{I(g2)};");
            L($"mul.rn.f32 %f{I(127 + column)}, %f133, 0f3F000000;");
            L($"mov.f32 %f{I(130 + column)}, %f{I(g2)};");
        }
        L("and.b32 %r13, %r0, 31;                    // filter M");
        L("shr.u32 %r14, %r0, 5;                     // filter local C");
        for (int row = 0; row < 4; row++)
        {
            int a = 121 + row * 3;
            L($"mov.f32 %f96, %f{I(a)};");
            L($"add.rn.f32 %f133, %f{I(a)}, %f{I(a + 1)};");
            L($"add.rn.f32 %f133, %f133, %f{I(a + 2)};");
            L("mul.rn.f32 %f97, %f133, 0f3F000000;");
            L($"sub.rn.f32 %f133, %f{I(a)}, %f{I(a + 1)};");
            L($"add.rn.f32 %f133, %f133, %f{I(a + 2)};");
            L("mul.rn.f32 %f98, %f133, 0f3F000000;");
            L($"mov.f32 %f99, %f{I(a + 2)};");
            for (int column = 0; column < 4; column++)
            {
                int component = row * 4 + column;
                L($"shl.b32 %r16, {I(component >> 2)}, 3;");
                L("xor.b32 %r16, %r13, %r16;");
                L($"mad.lo.u32 %r19, %r14, {I(TransformElements)}, {I(component)};");
                L($"mad.lo.u32 %r19, %r19, {I(UStride)}, %r16;");
                L("mul.wide.u32 %rd5, %r19, 4;");
                L($"mov.u64 %rd6, {bufferBase};");
                L("add.u64 %rd5, %rd6, %rd5;");
                L($"st.shared.f32 [%rd5], %f{I(96 + column)};");
            }
        }
    }

    private void EmitVGeometry(CodegenTiledConv2DPlan plan)
    {
        L($"setp.lt.u32 %p14, %r0, {I(BlockChannels * BlockTiles)};");
        L("shr.u32 %r26, %r0, 5;                      // V producer C");
        L($"and.b32 %r27, %r0, {I(BlockTiles - 1)};    // V producer tile");
        L($"mad.lo.u32 %r28, %r2, {I(BlockTiles)}, %r27;");
        L($"shr.u32 %r29, %r28, {I(PowerOfTwoShift(MicroTiles))};");
        L($"and.b32 %r30, %r28, {I(MicroTiles - 1)};");
        L($"setp.lt.u32 %p4, %r29, {I(PhysicalMicroTiles)};");
        L($"setp.lt.u32 %p13, %r30, {I(PhysicalMicroTiles)};");
        L("and.pred %p4, %p4, %p13;                  // physical tile");
        L("add.u32 %r29, %r10, %r29;");
        L("add.u32 %r30, %r11, %r30;");
        L("mul.lo.u32 %r29, %r29, 2;");
        L("add.s32 %r29, %r29, -1;                   // patch top");
        L("mul.lo.u32 %r30, %r30, 2;");
        L("add.s32 %r30, %r30, -1;                   // patch left");
        L($"mad.lo.s32 %r31, %r29, {I(plan.InputWidth)}, %r30;");
        for (int row = 0; row < 4; row++)
        {
            int predicate = 5 + row;
            L($"add.s32 %r32, %r29, {I(row)};");
            L($"setp.ge.s32 %p{I(predicate)}, %r32, 0;");
            L($"setp.lt.s32 %p13, %r32, {I(plan.InputHeight)};");
            L($"and.pred %p{I(predicate)}, %p{I(predicate)}, %p13;");
            L($"and.pred %p{I(predicate)}, %p{I(predicate)}, %p4;");
        }
        for (int column = 0; column < 4; column++)
        {
            int predicate = 9 + column;
            L($"add.s32 %r32, %r30, {I(column)};");
            L($"setp.ge.s32 %p{I(predicate)}, %r32, 0;");
            L($"setp.lt.s32 %p13, %r32, {I(plan.InputWidth)};");
            L($"and.pred %p{I(predicate)}, %p{I(predicate)}, %p13;");
        }
    }

    private void EmitUCoordinates(CodegenTiledConv2DPlan plan)
    {
        // Every thread loads four consecutive transform components. Hoist the
        // stage-invariant source and destination offsets once, then XOR-skew
        // each component group's eight-wide M chunk. The skew keeps the U
        // consumer's v4 loads aligned while assigning all 32 producer lanes
        // to distinct shared-memory banks.
        L("and.b32 %r13, %r0, 3;                     // U component group");
        L("shr.u32 %r14, %r0, 2;");
        L($"and.b32 %r14, %r14, {I(BlockM - 1)};      // U logical M");
        L($"shr.u32 %r15, %r0, {I(PowerOfTwoShift(BlockM * 4))}; // U local C");
        L($"mad.lo.u32 %r34, %r14, {I(plan.ReductionChannels)}, %r15;");
        L("mul.lo.u32 %r34, %r34, 16;");
        L("mad.lo.u32 %r34, %r13, 4, %r34;          // invariant U source word");
        L("shl.b32 %r16, %r13, 3;");
        L("xor.b32 %r16, %r14, %r16;                // conflict-free U transpose slot");
        L("mad.lo.u32 %r17, %r15, 4, %r13;");
        L("shl.b32 %r17, %r17, 2;                   // first component row");
        L($"mad.lo.u32 %r35, %r17, {I(UStride)}, %r16; // invariant U shared word");
    }

    private void EmitOuterProducts(string bufferBase)
    {
        L($"mul.lo.u32 %r13, %r8, {I(MicroM)};");
        L("shr.u32 %r14, %r6, 2;");
        L("shl.b32 %r14, %r14, 3;");
        L("xor.b32 %r13, %r13, %r14;                // recover transposed U chunk");
        L($"mad.lo.u32 %r13, %r6, {I(UStride)}, %r13;");
        L("mul.wide.u32 %rd4, %r13, 4;");
        L($"mov.u64 %rd5, {bufferBase};");
        L("add.u64 %rd4, %rd5, %rd4;                 // U row");
        L($"mad.lo.u32 %r14, %r6, {I(VStride)}, 0;");
        L("mad.lo.u32 %r14, %r9, 8, %r14;");
        L("mul.wide.u32 %rd6, %r14, 4;");
        L($"add.u64 %rd5, %rd5, {I(VByteOffset)};");
        L("add.u64 %rd6, %rd5, %rd6;                 // V row");
        int operandFragments = MicroM / 4 + (MicroTiles + 3) / 4;
        for (int fragment = 0; fragment < operandFragments; fragment++)
            EmitOuterFragmentLoad(0, 64, fragment);

        for (int c = 0; c < BlockChannels; c++)
        {
            int current = (c & 1) == 0 ? 64 : 80;
            int next = (c & 1) == 0 ? 80 : 64;
            int issued = 0;
            int nextFragment = 0;
            for (int tile = 0; tile < PhysicalMicroTiles; tile++)
                for (int m = 0; m < MicroM; m++)
                {
                    if (c + 1 < BlockChannels &&
                        nextFragment < operandFragments &&
                        issued == nextFragment * AccumulatorElements / operandFragments)
                    {
                        EmitOuterFragmentLoad(c + 1, next, nextFragment);
                        nextFragment++;
                    }
                    int accumulator = m * PhysicalMicroTiles + tile;
                    L($"fma.rn.f32 %f{I(accumulator)}, %f{I(current + m)}, " +
                      $"%f{I(current + MicroM + tile)}, %f{I(accumulator)};");
                    issued++;
                }
        }
    }

    private void EmitOuterFragmentLoad(int c, int registerBase, int fragment)
    {
        int uFragments = MicroM / 4;
        bool u = fragment < uFragments;
        int stageOffset = c * TransformElements * (u ? UStride : VStride) * sizeof(float);
        int operandFragment = u ? fragment : fragment - uFragments;
        int vectorOffset = stageOffset + operandFragment * 16;
        string address = u ? "%rd4" : "%rd6";
        int firstRegister = registerBase + fragment * 4;
        L($"ld.shared.v4.f32 {{%f{I(firstRegister)}, %f{I(firstRegister + 1)}, " +
          $"%f{I(firstRegister + 2)}, %f{I(firstRegister + 3)}}}, " +
          $"[{address}+{I(vectorOffset)}];");
    }

    private void EmitOutput(CodegenTiledConv2DPlan plan)
    {
        int hw = plan.OutputHeight * plan.OutputWidth;
        L($"mad.lo.u32 %r13, %r7, {I(WorkspaceGroupStride)}, %r6;");
        L("mul.wide.u32 %rd4, %r13, 4;");
        L("mov.u64 %rd5, stage;");
        L("add.u64 %rd4, %rd5, %rd4;");
        for (int element = 0; element < AccumulatorElements; element++)
            L($"st.shared.f32 [%rd4+{I(element * WorkspaceElementStride * sizeof(float))}], " +
              $"%f{I(element)};");
        L("bar.sync 0;");

        int outputPairs = WorkspaceGroups * AccumulatorElements;
        int pairWaves = (outputPairs + BlockThreads - 1) / BlockThreads;
        for (int pairWave = 0; pairWave < pairWaves; pairWave++)
        {
            L($"add.u32 %r13, %r0, {I(pairWave * BlockThreads)};");
            bool partialWave = (pairWave + 1) * BlockThreads > outputPairs;
            if (partialWave)
            {
                L($"setp.lt.u32 %p3, %r13, {I(outputPairs)};");
                L($"@!%p3 bra OUTER_OUTPUT_PAIR_{I(pairWave)}_DONE;");
            }
            L($"shr.u32 %r14, %r13, {I(PowerOfTwoShift(MicroM))};");
            L($"mul.lo.u32 %r14, %r14, {I(PhysicalMicroTilesReciprocal)};");
            L($"shr.u32 %r14, %r14, {I(ReciprocalShift)};" +
              $"           // group = pair / {I(AccumulatorElements)}");
            L($"mad.lo.s32 %r15, %r14, -{I(AccumulatorElements)}, %r13;");
            L($"mul.lo.u32 %r17, %r14, {I(WorkspaceGroupStride)};");
            L($"mad.lo.u32 %r16, %r15, {I(WorkspaceElementStride)}, %r17;");
            L("mul.wide.u32 %rd4, %r16, 4;");
            L("mov.u64 %rd5, stage;");
            L("add.u64 %rd4, %rd5, %rd4;");
            for (int component = 0; component < TransformElements; component++)
                L($"ld.shared.f32 %f{I(64 + component)}, [%rd4+{I(component * sizeof(float))}];");
            for (int column = 0; column < 4; column++)
            {
                L($"add.rn.f32 %f{I(80 + column)}, %f{I(64 + column)}, %f{I(68 + column)};");
                L($"add.rn.f32 %f{I(80 + column)}, %f{I(80 + column)}, %f{I(72 + column)};");
                L($"sub.rn.f32 %f{I(84 + column)}, %f{I(68 + column)}, %f{I(72 + column)};");
                L($"sub.rn.f32 %f{I(84 + column)}, %f{I(84 + column)}, %f{I(76 + column)};");
            }
            L("add.rn.f32 %f64, %f80, %f81;");
            L("add.rn.f32 %f64, %f64, %f82;");
            L("sub.rn.f32 %f65, %f81, %f82;");
            L("sub.rn.f32 %f65, %f65, %f83;");
            L("add.rn.f32 %f66, %f84, %f85;");
            L("add.rn.f32 %f66, %f66, %f86;");
            L("sub.rn.f32 %f67, %f85, %f86;");
            L("sub.rn.f32 %f67, %f67, %f87;");

            L($"and.b32 %r16, %r14, {I(MGroups - 1)}; // M group");
            L($"shr.u32 %r17, %r14, {I(PowerOfTwoShift(MGroups))};" +
              "             // tile group");
            L($"mul.lo.u32 %r18, %r15, {I(PhysicalMicroTilesReciprocal)};");
            L($"shr.u32 %r18, %r18, {I(ReciprocalShift)};" +
              "            // M within group");
            L($"mad.lo.s32 %r19, %r18, -{I(PhysicalMicroTiles)}, %r15;");
            L($"mad.lo.u32 %r16, %r16, {I(MicroM)}, %r18;");
            L($"mad.lo.u32 %r17, %r17, {I(MicroTiles)}, %r19;");
            L($"mad.lo.u32 %r17, %r2, {I(BlockTiles)}, %r17;");
            L($"shr.u32 %r18, %r17, {I(PowerOfTwoShift(MicroTiles))};");
            L($"and.b32 %r19, %r17, {I(MicroTiles - 1)};");
            L($"setp.lt.u32 %p1, %r18, {I(PhysicalMicroTiles)};");
            L($"setp.lt.u32 %p2, %r19, {I(PhysicalMicroTiles)};");
            L("and.pred %p1, %p1, %p2;");
            L("add.u32 %r18, %r10, %r18;");
            L("add.u32 %r19, %r11, %r19;");
            L("mul.lo.u32 %r18, %r18, 2;");
            L("mul.lo.u32 %r19, %r19, 2;");
            L($"mad.lo.u32 %r20, %r5, {I(BlockM)}, %r16;");
            L($"mul.lo.u32 %r20, %r20, {I(hw)};");
            L($"mad.lo.u32 %r20, %r18, {I(plan.OutputWidth)}, %r20;");
            L("add.u32 %r20, %r20, %r19;");
            L("mul.wide.u32 %rd4, %r20, 4;");
            L("add.u64 %rd4, %rd2, %rd4;");
            L("@%p1 st.global.v2.f32 [%rd4], {%f64, %f65};");
            L($"@%p1 st.global.v2.f32 [%rd4+{I(plan.OutputWidth * sizeof(float))}], " +
              "{%f66, %f67};");
            if (partialWave)
                L($"OUTER_OUTPUT_PAIR_{I(pairWave)}_DONE:");
        }
    }

    private void EmitQuotient(string destination, string source, int divisor, string? comment)
    {
        int shift = PowerOfTwoShift(divisor);
        string suffix = comment is null ? string.Empty : " // " + comment;
        if (shift >= 0)
            L($"shr.u32 {destination}, {source}, {I(shift)};{suffix}");
        else
            L($"div.u32 {destination}, {source}, {I(divisor)};{suffix}");
    }

    private void EmitRemainder(string destination, string source, int divisor, string? comment)
    {
        int shift = PowerOfTwoShift(divisor);
        string suffix = comment is null ? string.Empty : " // " + comment;
        if (shift >= 0)
            L($"and.b32 {destination}, {source}, {I(divisor - 1)};{suffix}");
        else
            L($"rem.u32 {destination}, {source}, {I(divisor)};{suffix}");
    }

    private static int PowerOfTwoShift(int value)
    {
        if (value <= 0 || (value & (value - 1)) != 0) return -1;
        int shift = 0;
        while ((1 << shift) != value) shift++;
        return shift;
    }

    private void L(string line) => _body.Append("    ").Append(line).Append('\n');
    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);
}
