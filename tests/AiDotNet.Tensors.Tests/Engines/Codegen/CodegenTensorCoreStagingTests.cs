// Copyright (c) AiDotNet. All rights reserved.
// Shared-memory staging for the tensor-core GEMM.
//
// The naive lowering gives one warp one 16x16 tile and lets it stream its own operand bands
// from global, so nothing is shared between the warps that need the same bands. That measured
// 11.8 TFLOP/s at 2048^3 and COLLAPSED to 3.0 at 4096^3 once the reused bands outgrew L2.
//
// Staging makes a block of four warps own a 64x64 tile and copy the operand slabs into shared
// memory once per K step, cutting global operand traffic fourfold and fragment loads fourfold
// again. Measured: 32.7 TFLOP/s at 4096^3, and throughput now RISES with size instead of
// falling off.
//
// The dangerous failure is the second barrier. Without it a fast warp starts overwriting the
// slabs for step k+1 while a slow one is still reading step k, which produces plausible
// magnitudes and a different answer per run.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenTensorCoreStagingTests
{
    private const int Sm86Major = 8, Sm86Minor = 6;

    private static CodegenKernelSpec MatMul(
        int m, int k, int n,
        CodegenActivationKind activation = CodegenActivationKind.None,
        bool transposeB = false)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n),
            CodegenAxis.Reduce("k", k));

        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: CodegenElementType.Float16);

        var b = transposeB
            ? new CodegenTensorBinding(1, "b", new[] { n, k },
                new[] { CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(2) },
                elementType: CodegenElementType.Float16)
            : new CodegenTensorBinding(1, "b", new[] { k, n },
                new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
                elementType: CodegenElementType.Float16);

        var output = new CodegenTensorBinding(2, "out", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec("tc", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum, activation: activation);
    }

    private static PtxTensorCoreEmitter.Plan PlanFor(CodegenKernelSpec spec)
    {
        Assert.True(PtxTensorCoreEmitter.TryPlan(
            spec, Sm86Major, Sm86Minor, out var plan, out string reason), reason);
        return plan!;
    }

    /// <summary>A whole number of 64x64 block tiles is what staging needs.</summary>
    [Theory]
    [InlineData(64, 64, 64)]
    [InlineData(512, 512, 512)]
    [InlineData(1024, 4096, 1024)]
    public void WholeBlockTiles_AreEligible(int m, int k, int n)
    {
        Assert.True(PtxTensorCoreEmitter.CanStage(PlanFor(MatMul(m, k, n)), out string reason),
            reason);
    }

    /// <summary>
    /// A partial block tile falls back rather than being handled with per-thread bounds tests
    /// inside the staging loop, which would be a different kernel.
    /// </summary>
    [Theory]
    [InlineData(32, 64, 64)]        // M is a whole wmma tile but not a whole block tile
    [InlineData(64, 64, 48)]
    public void PartialBlockTiles_FallBack(int m, int k, int n)
    {
        Assert.False(PtxTensorCoreEmitter.CanStage(PlanFor(MatMul(m, k, n)), out string reason));
        Assert.Contains("block tile", reason, StringComparison.Ordinal);

        // ...and the naive path still handles them, exactly as before.
        var emitter = new PtxTensorCoreEmitter();
        emitter.Emit(MatMul(m, k, n), Sm86Major, Sm86Minor);
        Assert.False(emitter.Staged);
    }

    /// <summary>A transposed operand cannot be copied as a row-major slab.</summary>
    [Fact]
    public void TransposedB_FallsBack()
    {
        Assert.False(PtxTensorCoreEmitter.CanStage(
            PlanFor(MatMul(128, 128, 128, transposeB: true)), out string reason));
        Assert.Contains("column-wise", reason, StringComparison.Ordinal);
    }

    /// <summary>The staged kernel must load its fragments from SHARED, not global.</summary>
    [Fact]
    public void StagedKernel_LoadsFragmentsFromShared()
    {
        var emitter = new PtxTensorCoreEmitter();
        string ptx = emitter.Emit(MatMul(512, 512, 512), Sm86Major, Sm86Minor);

        Assert.True(emitter.Staged);
        Assert.Contains("wmma.load.a.sync.aligned.row.m16n16k16.shared.f16", ptx, StringComparison.Ordinal);
        Assert.Contains("wmma.load.b.sync.aligned.row.m16n16k16.shared.f16", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("wmma.load.a.sync.aligned.row.m16n16k16.global.f16", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// FOUR mma instructions per K step from FOUR fragment loads. That two-to-one ratio is
    /// half the win: the naive lowering issues two fragment loads per single mma.
    /// </summary>
    [Fact]
    public void StagedKernel_IssuesFourMmaPerStep()
    {
        var emitter = new PtxTensorCoreEmitter();
        string ptx = emitter.Emit(MatMul(512, 512, 512), Sm86Major, Sm86Minor);

        Assert.Equal(4, emitter.MmaInstructions);
        Assert.Equal(2, CountOccurrences(ptx, "wmma.load.a.sync"));
        Assert.Equal(2, CountOccurrences(ptx, "wmma.load.b.sync"));
    }

    /// <summary>
    /// TWO barriers per K step, and the second is not optional: without it a fast warp
    /// overwrites the slabs for step k+1 while a slow one is still reading step k out of
    /// them. That races silently -- plausible magnitudes, a different answer per run.
    /// </summary>
    [Fact]
    public void StagedKernel_BarriersOnBothSidesOfTheComputation()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(MatMul(512, 512, 512), Sm86Major, Sm86Minor);

        Assert.Equal(2, CountOccurrences(ptx, "bar.sync 0;"));

        int firstBarrier = ptx.IndexOf("bar.sync 0;", StringComparison.Ordinal);
        int firstMma = ptx.IndexOf("wmma.mma.sync", StringComparison.Ordinal);
        int lastBarrier = ptx.LastIndexOf("bar.sync 0;", StringComparison.Ordinal);

        Assert.True(firstBarrier < firstMma, "the staging store must be fenced before any mma");
        Assert.True(lastBarrier > firstMma, "the slabs must be fenced before being overwritten");
    }

    /// <summary>The shared slabs must be reserved, and sized to hold both.</summary>
    [Fact]
    public void StagedKernel_ReservesBothSlabs()
    {
        var emitter = new PtxTensorCoreEmitter();
        string ptx = emitter.Emit(MatMul(512, 512, 512), Sm86Major, Sm86Minor);

        // 64x16 halves of A plus 16x64 of B, two bytes each.
        Assert.Equal((64 * 16 + 16 * 64) * 2, emitter.SharedMemoryBytes);
        Assert.Contains(".shared .align 16 .b8 stage[4096];", ptx, StringComparison.Ordinal);
        Assert.Contains("st.shared.u32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// THE GRIDS DIFFER, and not by a little: staged, four warps cover sixteen 16x16 tiles
    /// rather than four, so a 64x64 output is ONE block staged and FOUR naive. Launching the
    /// staged kernel on the naive grid would run it four times over, each pass accumulating
    /// into the same output again.
    /// </summary>
    [Theory]
    [InlineData(64, 64, 1)]
    [InlineData(128, 128, 4)]
    [InlineData(1024, 1024, 256)]
    public void StagedGrid_IsOneBlockPerBlockTile(int m, int n, int expectedBlocks)
    {
        var emitter = new PtxTensorCoreEmitter();
        var plan = PlanFor(MatMul(m, 64, n));

        Assert.Equal(expectedBlocks, emitter.BlockCount(plan));
        Assert.Equal(128, emitter.BlockThreads);
    }

    /// <summary>Turning staging off must restore the naive grid, or the A/B measurement lies.</summary>
    [Fact]
    public void StagingDisabled_RestoresTheNaiveLowering()
    {
        var emitter = new PtxTensorCoreEmitter { EnableStaging = false };
        var spec = MatMul(512, 512, 512);
        string ptx = emitter.Emit(spec, Sm86Major, Sm86Minor);

        Assert.False(emitter.Staged);
        Assert.Equal(0, emitter.SharedMemoryBytes);
        Assert.Contains("wmma.load.a.sync.aligned.row.m16n16k16.global.f16", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);

        var plan = PlanFor(spec);
        Assert.Equal(plan.TileCount / 4, emitter.BlockCount(plan));
    }

    /// <summary>The fused epilogue must survive staging -- it is the advantage over cuBLAS.</summary>
    [Fact]
    public void StagedKernel_KeepsTheFusedEpilogue()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(
            MatMul(512, 512, 512, CodegenActivationKind.ReLU), Sm86Major, Sm86Minor);

        // Once per accumulator register of all four tiles.
        Assert.Equal(32, CountOccurrences(ptx, "max.f32 %fc"));
        Assert.True(ptx.IndexOf("max.f32 %fc", StringComparison.Ordinal)
                  < ptx.IndexOf("wmma.store", StringComparison.Ordinal));
    }

    /// <summary>All four of the warp's tiles must be stored, not just the first.</summary>
    [Fact]
    public void StagedKernel_StoresAllFourTiles()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(MatMul(512, 512, 512), Sm86Major, Sm86Minor);

        Assert.Equal(4, CountOccurrences(ptx, "wmma.store.d.sync"));
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, index = 0;
        while ((index = haystack.IndexOf(needle, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += needle.Length;
        }
        return count;
    }
}
