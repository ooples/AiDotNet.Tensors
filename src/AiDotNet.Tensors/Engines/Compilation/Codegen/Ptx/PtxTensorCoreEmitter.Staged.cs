// Copyright (c) AiDotNet. All rights reserved.
//
// Shared-memory staged tensor-core GEMM.
//
// The naive lowering gives one warp one 16x16 output tile and lets it stream its own operand
// bands straight from global. That measured 11.8 TFLOP/s at 2048^3 and then COLLAPSED to 3.0
// at 4096^3, against cuBLAS holding 57.6 -- and the collapse is the whole argument for this
// file. Total operand traffic there is O(M*N*K), because nothing is shared between the warps
// that need the same bands; once the reused bands outgrow L2 every warp goes to DRAM.
//
// Here a BLOCK of four warps owns a 64x64 output tile and stages the 64x16 slab of A and the
// 16x64 slab of B into shared memory once per K step. Each element of A is then fetched from
// global once per block rather than once per warp:
//
//   naive    per 16x16 tile: 16K halves of A + 16K of B  ->  M*N*K/8   halves total
//   staged   per 64x64 tile: 64K halves of A + 64K of B  ->  M*N*K/32  halves total
//
// A factor of four, and the same factor again in fragment loads: a warp loads two A fragments
// and two B fragments per K step and issues FOUR mma instructions from them, instead of two
// loads per mma.
//
// NOTE ON THE LEVER ITSELF. Shared-memory staging was tried on dense 3x3 convolution earlier
// in this campaign and REFUTED -- it raised L1 from 64.08% to 77.45%, because on NVIDIA
// hardware shared memory IS L1TEX, so ld.shared is counted by the very metric it was meant to
// relieve. That lever died because `mio_throttle` sat at 3.03%: the load pipe was never that
// kernel's bottleneck. The justification here is different in kind and does not rest on a
// throughput percentage at all -- throughput FALLS FOURFOLD when the working set outgrows L2,
// which is a locality statement no stall counter is needed to read.

using System;
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

    /// <summary>Bytes of shared memory a staged block reserves.</summary>
    public const int StagedSharedBytes = (BlockTileM * BlockTileK + BlockTileK * BlockTileN) * 2;

    /// <summary>Whether the last emission used the staged lowering.</summary>
    public bool Staged { get; private set; }

    /// <summary>Shared-memory bytes the last emission requires.</summary>
    public int SharedMemoryBytes { get; private set; }

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

    /// <summary>Blocks a staged plan launches.</summary>
    public static int StagedBlockCount(Plan plan) =>
        plan is null ? throw new ArgumentNullException(nameof(plan))
                     : (plan.M / BlockTileM) * (plan.N / BlockTileN);

    /// <summary>Emits the staged kernel.</summary>
    private string EmitStaged(CodegenKernelSpec spec, Plan plan, int computeMajor, int computeMinor)
    {
        int aIndex = spec.ProductInputs[0], bIndex = spec.ProductInputs[1];
        int outIndex = spec.Inputs.Count;

        Staged = true;
        SharedMemoryBytes = StagedSharedBytes;
        MmaInstructions = 0;

        int tilesN = plan.N / BlockTileN;
        int steps = plan.K / BlockTileK;

        // Shared slab geometry, in u32 words, which is what the cooperative copy moves.
        const int AWords = BlockTileM * BlockTileK / 2;      // 512
        const int BWords = BlockTileK * BlockTileN / 2;      // 512
        const int Threads = StagedWarps * 32;                // 128
        const int AWordsPerThread = AWords / Threads;        // 4
        const int BWordsPerThread = BWords / Threads;        // 4
        const int AWordsPerRow = BlockTileK / 2;             // 8
        const int BWordsPerRow = BlockTileN / 2;             // 32

        var header = new System.Text.StringBuilder();
        header.Append(".version 7.1\n")
              .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
              .Append(".address_size 64\n\n")
              .Append("// generated by PtxTensorCoreEmitter -- STAGED wmma, ")
              .Append(I(BlockTileM)).Append('x').Append(I(BlockTileN)).Append('x')
              .Append(I(BlockTileK)).Append(" block tile, ").Append(I(StagedWarps)).Append(" warps\n")
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

        // Shared base addresses. sB follows sA.
        L("mov.u64 %rd3, stage;");
        L($"add.u64 %rd4, %rd3, {I(AWords * 4)};   // sB");

        // Global slab origins for this block, in ELEMENTS, before the K walk advances them.
        L($"mul.lo.u32 %r7, %r2, {I(BlockTileM * plan.K)};    // A slab origin");
        L($"mul.lo.u32 %r8, %r3, {I(BlockTileN)};             // B slab origin");
        L("mul.wide.u32 %rd5, %r7, 2;");
        L("add.u64 %rd6, %rd0, %rd5;              // A slab pointer");
        L("mul.wide.u32 %rd7, %r8, 2;");
        L("add.u64 %rd8, %rd1, %rd7;              // B slab pointer");

        // THE COOPERATIVE COPY'S PER-THREAD ADDRESSES ARE LOOP-INVARIANT, so they are computed
        // once here rather than per K step. Each thread moves AWordsPerThread words of A and
        // BWordsPerThread of B, strided by the block size so the accesses stay coalesced.
        for (int w = 0; w < AWordsPerThread; w++)
        {
            int word = w * Threads;
            string wordIndex = $"%r{I(10 + w)}";
            L($"add.u32 {wordIndex}, %r1, {I(word)};");
        }
        for (int w = 0; w < BWordsPerThread; w++)
        {
            int word = w * Threads;
            string wordIndex = $"%r{I(14 + w)}";
            L($"add.u32 {wordIndex}, %r1, {I(word)};");
        }

        // Accumulators: one fragment per (i, j) quadrant tile.
        for (int t = 0; t < WarpTiles * WarpTiles; t++)
            for (int f = 0; f < FragmentRegisters; f++)
                L($"mov.f32 %fc{I(t * FragmentRegisters + f)}, 0f00000000;");

        L("mov.u32 %r9, 0;                        // k step");
        _sb.Append("KLOOP:\n");

        // ---- stage A: 64 rows x 16 halves, row r at global (slabRow + r)*K + k0 -----------
        for (int w = 0; w < AWordsPerThread; w++)
        {
            string wordIndex = $"%r{I(10 + w)}";
            string row = NextR(), col = NextR(), off = NextR();
            L($"div.u32 {row}, {wordIndex}, {I(AWordsPerRow)};");
            L($"rem.u32 {col}, {wordIndex}, {I(AWordsPerRow)};");
            L($"mul.lo.u32 {off}, {row}, {I(plan.K / 2)};   // row stride in words");
            L($"add.u32 {off}, {off}, {col};");

            string bytes = NextRd(), addr = NextRd(), data = NextR();
            L($"mul.wide.u32 {bytes}, {off}, 4;");
            L($"add.u64 {addr}, %rd6, {bytes};");
            L($"ld.global.nc.u32 {data}, [{addr}];");

            string sBytes = NextRd(), sAddr = NextRd();
            L($"mul.wide.u32 {sBytes}, {wordIndex}, 4;");
            L($"add.u64 {sAddr}, %rd3, {sBytes};");
            L($"st.shared.u32 [{sAddr}], {data};");
        }

        // ---- stage B: 16 rows x 64 halves, row r at global (k0 + r)*N + slabCol ----------
        for (int w = 0; w < BWordsPerThread; w++)
        {
            string wordIndex = $"%r{I(14 + w)}";
            string row = NextR(), col = NextR(), off = NextR();
            L($"div.u32 {row}, {wordIndex}, {I(BWordsPerRow)};");
            L($"rem.u32 {col}, {wordIndex}, {I(BWordsPerRow)};");
            L($"mul.lo.u32 {off}, {row}, {I(plan.N / 2)};");
            L($"add.u32 {off}, {off}, {col};");

            string bytes = NextRd(), addr = NextRd(), data = NextR();
            L($"mul.wide.u32 {bytes}, {off}, 4;");
            L($"add.u64 {addr}, %rd8, {bytes};");
            L($"ld.global.nc.u32 {data}, [{addr}];");

            string sBytes = NextRd(), sAddr = NextRd();
            L($"mul.wide.u32 {sBytes}, {wordIndex}, 4;");
            L($"add.u64 {sAddr}, %rd4, {sBytes};");
            L($"st.shared.u32 [{sAddr}], {data};");
        }

        L("bar.sync 0;");

        // ---- the warp's four tiles, from shared ------------------------------------------
        //
        // Two A fragments and two B fragments feed FOUR mma instructions. That ratio is the
        // second half of the win: the naive lowering issues two fragment loads per mma.
        for (int i = 0; i < WarpTiles; i++)
        {
            string rowOffset = NextR(), bytes = NextRd(), addr = NextRd();
            L($"mad.lo.u32 {rowOffset}, %r5, {I(WarpTiles * 16 * BlockTileK)}, {I(i * 16 * BlockTileK)};");
            L($"mul.wide.u32 {bytes}, {rowOffset}, 2;");
            L($"add.u64 {addr}, %rd3, {bytes};");
            L($"wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {Fragment("%fa", i * FragmentRegisters)}, [{addr}], {I(BlockTileK)};");
        }

        for (int j = 0; j < WarpTiles; j++)
        {
            string colOffset = NextR(), bytes = NextRd(), addr = NextRd();
            L($"mad.lo.u32 {colOffset}, %r6, {I(WarpTiles * 16)}, {I(j * 16)};");
            L($"mul.wide.u32 {bytes}, {colOffset}, 2;");
            L($"add.u64 {addr}, %rd4, {bytes};");
            L($"wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 {Fragment("%fb", j * FragmentRegisters)}, [{addr}], {I(BlockTileN)};");
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

        // THE SECOND BARRIER IS NOT OPTIONAL. Without it a fast warp begins overwriting the
        // slabs for step k+1 while a slow one is still reading step k out of them -- which
        // produces plausible magnitudes and a different answer per run.
        L("bar.sync 0;");

        L($"add.u64 %rd6, %rd6, {I(BlockTileK * 2)};        // A advances 16 columns");
        L($"add.u64 %rd8, %rd8, {I(BlockTileK * plan.N * 2)};   // B advances 16 rows");
        L("add.u32 %r9, %r9, 1;");
        L($"setp.lt.u32 %p1, %r9, {I(steps)};");
        L("@%p1 bra KLOOP;");

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

                L($"mad.lo.u32 {row}, %r2, {I(BlockTileM)}, 0;");
                L($"mad.lo.u32 {row}, %r5, {I(WarpTiles * 16)}, {row};");
                L($"add.u32 {row}, {row}, {I(i * 16)};");

                L($"mad.lo.u32 {col}, %r3, {I(BlockTileN)}, 0;");
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

        header.Append("    .shared .align 16 .b8 stage[").Append(I(StagedSharedBytes)).Append("];\n")
              .Append("    .reg .pred   %p<8>;\n")
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

    private int _reg = FixedRegisters;
    private int _reg64 = FixedRegisters64;

    private string NextR() => "%r" + (_reg++).ToString(CultureInfo.InvariantCulture);

    private string NextRd() => "%rd" + (_reg64++).ToString(CultureInfo.InvariantCulture);
}
