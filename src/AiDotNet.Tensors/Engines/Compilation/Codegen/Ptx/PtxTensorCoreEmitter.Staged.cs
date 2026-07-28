// Copyright (c) AiDotNet. All rights reserved.
//
// Shared-memory staged tensor-core GEMM, single- and double-buffered.
//
// The naive lowering gives one warp one 16x16 output tile and lets it stream its own operand
// bands straight from global. That measured 11.8 TFLOP/s at 2048^3 and COLLAPSED to 3.0 at
// 4096^3, against cuBLAS holding 57.6. Total operand traffic there is O(M*N*K), because
// nothing is shared between the warps that need the same bands; once the reused bands outgrow
// L2 every warp goes to DRAM.
//
// Staging: a BLOCK of four warps owns a 64x64 output tile and stages the 64x16 slab of A and
// the 16x64 slab of B into shared memory once per K step. Each element of A is fetched from
// global once per block rather than once per warp:
//
//   naive    per 16x16 tile: 16K halves of A + 16K of B  ->  M*N*K/8   halves total
//   staged   per 64x64 tile: 64K halves of A + 64K of B  ->  M*N*K/32  halves total
//
// Double buffering: single-buffered staging needs TWO barriers per step -- one after the
// shared store, one after the reads, to stop a fast warp overwriting a slab a slow one is
// still reading. Between them the global load and the arithmetic cannot overlap at all: every
// warp waits at the first barrier for the copy, then computes, then waits again. With two
// slabs the copy for step k+1 targets the buffer nobody is reading, so it is ISSUED BEFORE
// the mma work for step k and its latency hides behind that arithmetic. One barrier per step.
//
// NOTE ON THE LEVER ITSELF. Shared-memory staging was tried on dense 3x3 convolution earlier
// in this campaign and REFUTED -- it raised L1 from 64.08% to 77.45%, because on NVIDIA
// hardware shared memory IS L1TEX, so ld.shared is counted by the very metric it was meant to
// relieve. That lever died because `mio_throttle` sat at 3.03%: the load pipe was never that
// kernel's bottleneck. The justification here is different in kind and does not rest on a
// throughput percentage at all -- throughput FELL FOURFOLD when the working set outgrew L2,
// which is a locality statement no stall counter is needed to read.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

public sealed partial class PtxTensorCoreEmitter
{
    /// <summary>Output rows a block computes.</summary>
    public const int BlockTileM = 64;

    /// <summary>Output columns a block computes.</summary>
    public const int BlockTileN = 64;

    /// <summary>Contraction depth staged per step.</summary>
    public const int BlockTileK = 16;

    /// <summary>Warps per staged block. Four, arranged 2x2 over the block tile.</summary>
    public const int StagedWarps = 4;

    /// <summary>wmma tiles a warp owns along each axis (its quadrant is 32x32).</summary>
    private const int WarpTiles = 2;

    /// <summary>Bytes one staging buffer occupies: the A slab followed by the B slab.</summary>
    public const int StageBufferBytes = (BlockTileM * BlockTileK + BlockTileK * BlockTileN) * 2;

    /// <summary>Byte offset of the B slab inside a buffer.</summary>
    private const int BSlabOffset = BlockTileM * BlockTileK * 2;

    /// <summary>Whether the last emission used the shared-memory staged lowering.</summary>
    public bool Staged { get; private set; }

    /// <summary>Whether the last emission double-buffered the staging slabs.</summary>
    public bool DoubleBuffered { get; private set; }

    /// <summary>Shared-memory bytes the last emission requires.</summary>
    public int SharedMemoryBytes { get; private set; }

    /// <summary><c>bar.sync</c> instructions the last emission placed inside the K loop.</summary>
    public int LoopBarriers { get; private set; }

    /// <summary>
    /// Whether to double-buffer the staging slabs when the shape allows it.
    /// </summary>
    /// <remarks>
    /// Settable so the two forms can be measured against each other on the same shape.
    /// Turning it off is a measurement device, not a supported configuration.
    /// </remarks>
    public bool EnableDoubleBuffering { get; set; } = true;

    /// <summary>
    /// Whether a plan can take the staged lowering, and if not, why not.
    /// </summary>
    /// <remarks>
    /// The restrictions are all about the cooperative load being a clean, aligned, whole-slab
    /// copy. A partial block tile would need per-thread bounds tests inside the staging loop,
    /// which is a different kernel rather than a tweak to this one; a transposed B would need
    /// its slab gathered column-wise instead of copied. Both fall back to the naive path,
    /// which is correct at every shape.
    /// </remarks>
    public static bool CanStage(Plan plan, out string reason)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));

        if (!plan.ARowMajor || !plan.BRowMajor)
        {
            reason = "staging copies whole row-major slabs; a transposed operand would have " +
                "to be gathered column-wise";
            return false;
        }

        if (plan.M % BlockTileM != 0 || plan.N % BlockTileN != 0)
        {
            reason = "the staged block tile is " + BlockTileM + "x" + BlockTileN +
                " and this output is " + plan.M + "x" + plan.N +
                ", which is not a whole number of them";
            return false;
        }

        if (plan.K % BlockTileK != 0)
        {
            reason = "the staged step is " + BlockTileK + " deep and K is " + plan.K;
            return false;
        }

        reason = "eligible";
        return true;
    }

    /// <summary>
    /// Whether a staged plan can double-buffer, and if not, why not.
    /// </summary>
    /// <remarks>
    /// The loop body is emitted TWICE, once per buffer, so the buffer index is a compile-time
    /// constant rather than a register the shared address has to be computed from. That costs
    /// an even step count. An odd one falls back to single-buffered staging, which is correct
    /// and merely slower -- padding K to make it even would change the operator.
    /// </remarks>
    public static bool CanDoubleBuffer(Plan plan, out string reason)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));

        if (!CanStage(plan, out reason)) return false;

        int steps = plan.K / BlockTileK;
        if (steps < 2)
        {
            reason = "double buffering needs at least two K steps; this shape has " + steps;
            return false;
        }

        if (steps % 2 != 0)
        {
            reason = "the loop body is emitted once per buffer, so the step count must be " +
                "even; K = " + plan.K + " gives " + steps + " steps";
            return false;
        }

        reason = "eligible";
        return true;
    }

    /// <summary>Blocks a staged plan launches.</summary>
    public static int StagedBlockCount(Plan plan) =>
        plan is null ? throw new ArgumentNullException(nameof(plan))
                     : (plan.M / BlockTileM) * (plan.N / BlockTileN);

    // Shared slab geometry, in u32 words, which is what the cooperative copy moves.
    private const int AWords = BlockTileM * BlockTileK / 2;      // 512
    private const int BWords = BlockTileK * BlockTileN / 2;      // 512
    private const int StageThreads = StagedWarps * 32;           // 128
    private const int AWordsPerThread = AWords / StageThreads;   // 4
    private const int BWordsPerThread = BWords / StageThreads;   // 4
    private const int AWordsPerRow = BlockTileK / 2;             // 8
    private const int BWordsPerRow = BlockTileN / 2;             // 32

    /// <summary>Emits the staged kernel, double-buffered where the shape allows it.</summary>
    private string EmitStaged(CodegenKernelSpec spec, Plan plan, int computeMajor, int computeMinor)
    {
        int aIndex = spec.ProductInputs[0], bIndex = spec.ProductInputs[1];
        int outIndex = spec.Inputs.Count;

        bool doubleBuffer = EnableDoubleBuffering && CanDoubleBuffer(plan, out _);

        Staged = true;
        DoubleBuffered = doubleBuffer;
        SharedMemoryBytes = doubleBuffer ? StageBufferBytes * 2 : StageBufferBytes;
        MmaInstructions = 0;
        LoopBarriers = 0;

        int tilesN = plan.N / BlockTileN;
        int steps = plan.K / BlockTileK;

        var header = new System.Text.StringBuilder();
        header.Append(".version 7.1\n")
              .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
              .Append(".address_size 64\n\n")
              .Append("// generated by PtxTensorCoreEmitter -- STAGED wmma, ")
              .Append(I(BlockTileM)).Append('x').Append(I(BlockTileN)).Append('x')
              .Append(I(BlockTileK)).Append(" block tile, ").Append(I(StagedWarps)).Append(" warps, ")
              .Append(doubleBuffer ? "double-buffered" : "single-buffered").Append('\n')
              .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
              .Append(".visible .entry ").Append(spec.Name).Append("(\n");

        int paramCount = spec.ParameterCount;
        for (int i = 0; i < paramCount; i++)
            header.Append("    .param .u64 p").Append(I(i)).Append(i == paramCount - 1 ? "\n" : ",\n");
        header.Append(")\n{\n");

        // THE BODY IS BUILT FIRST and the register declarations sized from it, because a
        // literal bound is a silent ceiling rather than a generous one: the affine emitter
        // shipped %p<256> until coarsening pushed one past it, and ptxas reported that as
        // "Arguments mismatch for instruction 'setp'" -- an undeclared register, not a
        // malformed instruction.
        _sb.Clear();
        _reg = FixedRegisters;
        _reg64 = FixedRegisters64;

        L($"ld.param.u64 %rd0, [p{I(aIndex)}];");
        L($"ld.param.u64 %rd1, [p{I(bIndex)}];");
        L($"ld.param.u64 %rd2, [p{I(outIndex)}];");

        L("mov.u32 %r0, %ctaid.x;");
        L("mov.u32 %r1, %tid.x;");
        L($"div.u32 %r2, %r0, {I(tilesN)};        // block row");
        L($"rem.u32 %r3, %r0, {I(tilesN)};        // block column");
        L("shr.u32 %r4, %r1, 5;                   // warp id 0..3");
        L("shr.u32 %r5, %r4, 1;                   // warp row 0..1");
        L("and.b32 %r6, %r4, 1;                   // warp column 0..1");

        L("mov.u64 %rd3, stage;");

        // Global slab origins for this block, in ELEMENTS. These advance as the K walk
        // proceeds and always point at the step about to be PREFETCHED.
        L($"mul.lo.u32 %r7, %r2, {I(BlockTileM * plan.K)};    // A slab origin");
        L($"mul.lo.u32 %r8, %r3, {I(BlockTileN)};             // B slab origin");
        L("mul.wide.u32 %rd5, %r7, 2;");
        L("add.u64 %rd6, %rd0, %rd5;              // A slab pointer");
        L("mul.wide.u32 %rd7, %r8, 2;");
        L("add.u64 %rd8, %rd1, %rd7;              // B slab pointer");

        // The cooperative copy's per-thread word indices are loop-invariant, so they are
        // computed once rather than per K step.
        var aWordIndex = new string[AWordsPerThread];
        var bWordIndex = new string[BWordsPerThread];
        for (int w = 0; w < AWordsPerThread; w++)
        {
            aWordIndex[w] = NextR();
            L($"add.u32 {aWordIndex[w]}, %r1, {I(w * StageThreads)};");
        }
        for (int w = 0; w < BWordsPerThread; w++)
        {
            bWordIndex[w] = NextR();
            L($"add.u32 {bWordIndex[w]}, %r1, {I(w * StageThreads)};");
        }

        // Global byte offsets for this thread's share of each slab, also loop-invariant: only
        // the slab POINTERS move as K advances.
        var aGlobalOffset = new string[AWordsPerThread];
        var bGlobalOffset = new string[BWordsPerThread];
        for (int w = 0; w < AWordsPerThread; w++)
        {
            string row = NextR(), col = NextR(), off = NextR();
            L($"div.u32 {row}, {aWordIndex[w]}, {I(AWordsPerRow)};");
            L($"rem.u32 {col}, {aWordIndex[w]}, {I(AWordsPerRow)};");
            L($"mad.lo.u32 {off}, {row}, {I(plan.K / 2)}, {col};");
            aGlobalOffset[w] = NextRd();
            L($"mul.wide.u32 {aGlobalOffset[w]}, {off}, 4;");
        }
        for (int w = 0; w < BWordsPerThread; w++)
        {
            string row = NextR(), col = NextR(), off = NextR();
            L($"div.u32 {row}, {bWordIndex[w]}, {I(BWordsPerRow)};");
            L($"rem.u32 {col}, {bWordIndex[w]}, {I(BWordsPerRow)};");
            L($"mad.lo.u32 {off}, {row}, {I(plan.N / 2)}, {col};");
            bGlobalOffset[w] = NextRd();
            L($"mul.wide.u32 {bGlobalOffset[w]}, {off}, 4;");
        }

        // Shared byte offsets for this thread's share, and for this warp's fragments. Both are
        // relative to a buffer's base, so the buffer index stays a compile-time addend.
        var aSharedOffset = new string[AWordsPerThread];
        var bSharedOffset = new string[BWordsPerThread];
        for (int w = 0; w < AWordsPerThread; w++)
        {
            aSharedOffset[w] = NextRd();
            L($"mul.wide.u32 {aSharedOffset[w]}, {aWordIndex[w]}, 4;");
        }
        for (int w = 0; w < BWordsPerThread; w++)
        {
            bSharedOffset[w] = NextRd();
            L($"mul.wide.u32 {bSharedOffset[w]}, {bWordIndex[w]}, 4;");
        }

        string warpAOffset = NextRd(), warpBOffset = NextRd();
        {
            string aBytes = NextR(), bBytes = NextR();
            L($"mul.lo.u32 {aBytes}, %r5, {I(WarpTiles * 16 * BlockTileK * 2)};");
            L($"mul.wide.u32 {warpAOffset}, {aBytes}, 1;");
            L($"mul.lo.u32 {bBytes}, %r6, {I(WarpTiles * 16 * 2)};");
            L($"mul.wide.u32 {warpBOffset}, {bBytes}, 1;");
        }

        // Accumulators: one fragment per (i, j) quadrant tile.
        for (int t = 0; t < WarpTiles * WarpTiles; t++)
            for (int f = 0; f < FragmentRegisters; f++)
                L($"mov.f32 %fc{I(t * FragmentRegisters + f)}, 0f00000000;");

        var prefetch = new List<string>();

        if (!doubleBuffer)
        {
            // ---- single-buffered: copy, fence, compute, fence -----------------------------
            L("mov.u32 %r9, 0;                        // k step");
            _sb.Append("KLOOP:\n");

            var regs = EmitSlabLoad(aGlobalOffset, bGlobalOffset, null);
            EmitSlabStore(regs, aSharedOffset, bSharedOffset, buffer: 0);
            L("bar.sync 0;");
            LoopBarriers++;

            EmitComputeStep(spec, plan, warpAOffset, warpBOffset, buffer: 0);

            // THE SECOND BARRIER IS NOT OPTIONAL HERE. Without it a fast warp begins
            // overwriting the slab for step k+1 while a slow one is still reading step k out
            // of it -- which produces plausible magnitudes and a different answer per run.
            // Double buffering removes the need for it by writing elsewhere, which is exactly
            // where the overlap comes from.
            L("bar.sync 0;");
            LoopBarriers++;

            EmitAdvance(plan);
            L("add.u32 %r9, %r9, 1;");
            L($"setp.lt.u32 %p1, %r9, {I(steps)};");
            L("@%p1 bra KLOOP;");
        }
        else
        {
            // ---- double-buffered ----------------------------------------------------------
            //
            // Prologue stages step 0 into buffer 0. Thereafter each body consumes one buffer
            // while filling the other, so the global load for step k+1 is ISSUED BEFORE the
            // mma work for step k and its latency hides behind that arithmetic.
            var first = EmitSlabLoad(aGlobalOffset, bGlobalOffset, null);
            EmitSlabStore(first, aSharedOffset, bSharedOffset, buffer: 0);
            EmitAdvance(plan);
            L("bar.sync 0;");

            L("mov.u32 %r9, 0;                        // pair index");
            L($"mov.u32 %r10, 1;                      // step being prefetched");
            _sb.Append("KLOOP:\n");

            // Two bodies per iteration so the buffer index is a compile-time constant and the
            // shared addresses stay immediate offsets rather than computed ones.
            for (int half = 0; half < 2; half++)
            {
                int current = half, next = 1 - half;

                // The final body's prefetch targets step `steps`, which is past the end of A's
                // row. Predicating it keeps the read inside the allocation; the stale register
                // it then stores is written to a buffer the loop never reads again.
                string guard = NextP();
                L($"setp.lt.u32 {guard}, %r10, {I(steps)};");

                prefetch.Clear();
                prefetch.AddRange(EmitSlabLoad(aGlobalOffset, bGlobalOffset, guard));
                EmitAdvance(plan);
                L("add.u32 %r10, %r10, 1;");

                EmitComputeStep(spec, plan, warpAOffset, warpBOffset, current);

                EmitSlabStore(prefetch, aSharedOffset, bSharedOffset, next);
                L("bar.sync 0;");
                LoopBarriers++;
            }

            L("add.u32 %r9, %r9, 1;");
            L($"setp.lt.u32 %p1, %r9, {I(steps / 2)};");
            L("@%p1 bra KLOOP;");
        }

        // ---- epilogue and store ----------------------------------------------------------
        if (spec.Activation != CodegenActivationKind.None)
            for (int t = 0; t < WarpTiles * WarpTiles; t++)
                for (int f = 0; f < FragmentRegisters; f++)
                    EmitActivation(spec.Activation, $"%fc{I(t * FragmentRegisters + f)}");

        if (spec.ReduceScale != 1.0)
            for (int t = 0; t < WarpTiles * WarpTiles; t++)
                for (int f = 0; f < FragmentRegisters; f++)
                {
                    string reg = $"%fc{I(t * FragmentRegisters + f)}";
                    L($"mul.f32 {reg}, {reg}, {F32(spec.ReduceScale)};");
                }

        for (int i = 0; i < WarpTiles; i++)
            for (int j = 0; j < WarpTiles; j++)
            {
                int t = i * WarpTiles + j;
                string row = NextR(), col = NextR(), off = NextR();

                L($"mul.lo.u32 {row}, %r2, {I(BlockTileM)};");
                L($"mad.lo.u32 {row}, %r5, {I(WarpTiles * 16)}, {row};");
                L($"add.u32 {row}, {row}, {I(i * 16)};");

                L($"mul.lo.u32 {col}, %r3, {I(BlockTileN)};");
                L($"mad.lo.u32 {col}, %r6, {I(WarpTiles * 16)}, {col};");
                L($"add.u32 {col}, {col}, {I(j * 16)};");

                L($"mad.lo.u32 {off}, {row}, {I(plan.N)}, {col};");

                string bytes = NextRd(), addr = NextRd();
                L($"mul.wide.u32 {bytes}, {off}, 4;");
                L($"add.u64 {addr}, %rd2, {bytes};");
                L($"wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [{addr}], " +
                  $"{Fragment("%fc", t * FragmentRegisters)}, {I(plan.N)};");
            }

        _sb.Append("END:\n    ret;\n}\n");

        header.Append("    .shared .align 16 .b8 stage[").Append(I(SharedMemoryBytes)).Append("];\n")
              .Append("    .reg .pred   %p<").Append(I(_pred + 8)).Append(">;\n")
              .Append("    .reg .b32    %r<").Append(I(_reg + 8)).Append(">;\n")
              .Append("    .reg .f32    %f<40>;\n")
              .Append("    .reg .b64    %rd<").Append(I(_reg64 + 8)).Append(">;\n")
              .Append("    .reg .b32    %fa<").Append(I(WarpTiles * FragmentRegisters)).Append(">;\n")
              .Append("    .reg .b32    %fb<").Append(I(WarpTiles * FragmentRegisters)).Append(">;\n")
              .Append("    .reg .f32    %fc<")
              .Append(I(WarpTiles * WarpTiles * FragmentRegisters)).Append(">;\n\n")
              .Append(_sb);

        return header.ToString();
    }

    /// <summary>
    /// Issues the global reads for one K step into registers, optionally predicated.
    /// </summary>
    /// <remarks>
    /// Loading into registers and storing to shared LATER is what the whole double-buffered
    /// form turns on: the reads are issued before the arithmetic they overlap with, and only
    /// the store has to wait.
    /// </remarks>
    private string[] EmitSlabLoad(string[] aGlobalOffset, string[] bGlobalOffset, string? guard)
    {
        var regs = new string[AWordsPerThread + BWordsPerThread];

        for (int w = 0; w < AWordsPerThread; w++)
        {
            string addr = NextRd();
            regs[w] = NextR();
            L($"add.u64 {addr}, %rd6, {aGlobalOffset[w]};");
            if (guard is null) L($"ld.global.nc.u32 {regs[w]}, [{addr}];");
            else L($"@{guard} ld.global.nc.u32 {regs[w]}, [{addr}];");
        }

        for (int w = 0; w < BWordsPerThread; w++)
        {
            string addr = NextRd();
            regs[AWordsPerThread + w] = NextR();
            L($"add.u64 {addr}, %rd8, {bGlobalOffset[w]};");
            if (guard is null) L($"ld.global.nc.u32 {regs[AWordsPerThread + w]}, [{addr}];");
            else L($"@{guard} ld.global.nc.u32 {regs[AWordsPerThread + w]}, [{addr}];");
        }

        return regs;
    }

    /// <summary>Writes staged registers into one of the shared buffers.</summary>
    private void EmitSlabStore(
        IReadOnlyList<string> regs, string[] aSharedOffset, string[] bSharedOffset, int buffer)
    {
        int bufferBase = buffer * StageBufferBytes;

        for (int w = 0; w < AWordsPerThread; w++)
        {
            string addr = NextRd();
            L($"add.u64 {addr}, %rd3, {aSharedOffset[w]};");
            L($"st.shared.u32 [{addr}+{I(bufferBase)}], {regs[w]};");
        }

        for (int w = 0; w < BWordsPerThread; w++)
        {
            string addr = NextRd();
            L($"add.u64 {addr}, %rd3, {bSharedOffset[w]};");
            L($"st.shared.u32 [{addr}+{I(bufferBase + BSlabOffset)}], {regs[AWordsPerThread + w]};");
        }
    }

    /// <summary>Loads this warp's fragments from a buffer and issues its four mma instructions.</summary>
    private void EmitComputeStep(
        CodegenKernelSpec spec, Plan plan, string warpAOffset, string warpBOffset, int buffer)
    {
        int bufferBase = buffer * StageBufferBytes;

        for (int i = 0; i < WarpTiles; i++)
        {
            string addr = NextRd();
            L($"add.u64 {addr}, %rd3, {warpAOffset};");
            L($"wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 " +
              $"{Fragment("%fa", i * FragmentRegisters)}, " +
              $"[{addr}+{I(bufferBase + i * 16 * BlockTileK * 2)}], {I(BlockTileK)};");
        }

        for (int j = 0; j < WarpTiles; j++)
        {
            string addr = NextRd();
            L($"add.u64 {addr}, %rd3, {warpBOffset};");
            L($"wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 " +
              $"{Fragment("%fb", j * FragmentRegisters)}, " +
              $"[{addr}+{I(bufferBase + BSlabOffset + j * 16 * 2)}], {I(BlockTileN)};");
        }

        for (int i = 0; i < WarpTiles; i++)
            for (int j = 0; j < WarpTiles; j++)
            {
                int t = i * WarpTiles + j;
                L($"wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 " +
                  $"{Fragment("%fc", t * FragmentRegisters)}, {Fragment("%fa", i * FragmentRegisters)}, " +
                  $"{Fragment("%fb", j * FragmentRegisters)}, {Fragment("%fc", t * FragmentRegisters)};");
                MmaInstructions++;
            }
    }

    /// <summary>Advances the global slab pointers by one K step.</summary>
    private void EmitAdvance(Plan plan)
    {
        L($"add.u64 %rd6, %rd6, {I(BlockTileK * 2)};            // A advances 16 columns");
        L($"add.u64 %rd8, %rd8, {I(BlockTileK * plan.N * 2)};   // B advances 16 rows");
    }

    /// <summary>A fragment register list starting at <paramref name="start"/>.</summary>
    private static string Fragment(string prefix, int start)
    {
        var sb = new System.Text.StringBuilder("{");
        for (int f = 0; f < FragmentRegisters; f++)
        {
            if (f > 0) sb.Append(", ");
            sb.Append(prefix).Append((start + f).ToString(CultureInfo.InvariantCulture));
        }
        return sb.Append('}').ToString();
    }

    /// <summary>First freely allocatable %r; below this the registers have fixed roles.</summary>
    private const int FixedRegisters = 20;

    /// <summary>First freely allocatable %rd.</summary>
    private const int FixedRegisters64 = 10;

    /// <summary>First freely allocatable %p.</summary>
    private const int FixedPredicates = 4;

    private int _reg = FixedRegisters;
    private int _reg64 = FixedRegisters64;
    private int _pred = FixedPredicates;

    private string NextR() => "%r" + (_reg++).ToString(CultureInfo.InvariantCulture);

    private string NextRd() => "%rd" + (_reg64++).ToString(CultureInfo.InvariantCulture);

    private string NextP() => "%p" + (_pred++).ToString(CultureInfo.InvariantCulture);
}
