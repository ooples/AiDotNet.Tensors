// Copyright (c) AiDotNet. All rights reserved.
// The tensor-core lowering, and -- more importantly -- what it refuses.
//
// The emitter emitted no wmma at all, which is why "dense GEMM at large K" is recorded in
// the blueprint as unwinnable: every generated matmul ran on the FP32 pipes while the
// competitor ran on the tensor cores. That is an order of magnitude of arithmetic
// throughput, not a tiling deficit.
//
// The dangerous failure here is not a compile error. wmma is warp-collective and its
// fragment layout is deliberately opaque, so a lowering that assumes which lane holds which
// element still runs, still produces numbers of the right magnitude, and is wrong. Most of
// these tests are therefore about the RECOGNISER: a spec that wmma cannot express exactly
// must be refused, with a reason, and fall back to the scalar path.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenTensorCoreTests
{
    private const int Sm86Major = 8, Sm86Minor = 6;

    private static CodegenKernelSpec MatMul(
        int m, int k, int n,
        CodegenElementType operands = CodegenElementType.Float16,
        CodegenElementType output = CodegenElementType.Float32,
        CodegenActivationKind activation = CodegenActivationKind.None,
        bool transposeB = false,
        CodegenReduceKind reduce = CodegenReduceKind.Sum,
        CodegenPreReduceOp preReduce = CodegenPreReduceOp.None)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", m), CodegenAxis.Parallel("n", n),
            CodegenAxis.Reduce("k", k));

        var a = new CodegenTensorBinding(0, "a", new[] { m, k },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(2) },
            elementType: operands);

        var b = transposeB
            ? new CodegenTensorBinding(1, "b", new[] { n, k },
                new[] { CodegenAffineExpr.Axis(1), CodegenAffineExpr.Axis(2) },
                elementType: operands)
            : new CodegenTensorBinding(1, "b", new[] { k, n },
                new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
                elementType: operands);

        var outBinding = new CodegenTensorBinding(2, "out", new[] { m, n },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) },
            isOutput: true, elementType: output);

        return new CodegenKernelSpec("tc", space, new[] { a, b }, outBinding,
            new[] { 0, 1 }, reduce, activation: activation, preReduce: preReduce);
    }

    /// <summary>A plain fp16 matmul on whole tiles is what this path is for.</summary>
    [Theory]
    [InlineData(16, 16, 16)]
    [InlineData(64, 64, 64)]
    [InlineData(512, 512, 512)]
    [InlineData(256, 2048, 256)]
    public void PlainFp16MatMul_IsEligible(int m, int k, int n)
    {
        Assert.True(PtxTensorCoreEmitter.TryPlan(
            MatMul(m, k, n), Sm86Major, Sm86Minor, out var plan, out string reason), reason);

        Assert.NotNull(plan);
        Assert.Equal(m, plan!.M);
        Assert.Equal(n, plan.N);
        Assert.Equal(k, plan.K);
        Assert.Equal((m / 16) * (n / 16), plan.TileCount);
    }

    /// <summary>The emitted kernel must actually use the tensor cores.</summary>
    [Fact]
    public void EmittedKernel_UsesWmma()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(MatMul(64, 64, 64), Sm86Major, Sm86Minor);

        Assert.Contains("wmma.load.a.sync.aligned.row.m16n16k16.global.f16", ptx, StringComparison.Ordinal);
        Assert.Contains("wmma.load.b.sync.aligned.row.m16n16k16.global.f16", ptx, StringComparison.Ordinal);
        Assert.Contains("wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("wmma.store.d.sync.aligned.row.m16n16k16.global.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// The tile index must come from the WARP, not the thread. wmma instructions are
    /// warp-synchronous: if lanes of one warp reached different tiles, or if some lanes took
    /// a bounds branch the others did not, the result is undefined rather than merely wrong.
    /// </summary>
    [Fact]
    public void TileIndex_ComesFromTheWarp()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(MatMul(64, 64, 64), Sm86Major, Sm86Minor);

        // tid >> 5 is the warp within the block; every lane of a warp gets the same value,
        // so the tile index and the guard that follows are warp-uniform by construction.
        Assert.Contains("mov.u32 %r1, %tid.x;", ptx, StringComparison.Ordinal);
        Assert.Contains("shr.u32 %r2, %r1, 5;", ptx, StringComparison.Ordinal);

        // ...and the tile actually comes from that warp index, not from %r1.
        Assert.Contains("mad.lo.u32 %r3, %r0, 4, %r2;", ptx, StringComparison.Ordinal);
        Assert.Contains("setp.ge.u32 %p0, %r3,", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// B transposed -- the [N, K] layout a linear layer's weights actually have -- must be
    /// handled by the layout qualifier, not by refusing or by transposing in memory.
    /// </summary>
    [Fact]
    public void TransposedB_UsesTheColumnLayoutQualifier()
    {
        Assert.True(PtxTensorCoreEmitter.TryPlan(
            MatMul(64, 64, 64, transposeB: true), Sm86Major, Sm86Minor,
            out var plan, out string reason), reason);

        Assert.False(plan!.BRowMajor);

        string ptx = new PtxTensorCoreEmitter().Emit(
            MatMul(64, 64, 64, transposeB: true), Sm86Major, Sm86Minor);
        Assert.Contains("wmma.load.b.sync.aligned.col.m16n16k16", ptx, StringComparison.Ordinal);
        Assert.Contains("wmma.mma.sync.aligned.row.col.m16n16k16", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// The fused epilogue is the ONLY structural advantage this path has over cuBLAS, which
    /// cannot fuse through its own call boundary. It must be applied to every accumulator
    /// register while the tile is still in registers.
    /// </summary>
    [Fact]
    public void Activation_IsFusedIntoTheAccumulatorFragment()
    {
        string ptx = new PtxTensorCoreEmitter().Emit(
            MatMul(64, 64, 64, activation: CodegenActivationKind.ReLU), Sm86Major, Sm86Minor);

        // Once per accumulator register, and before the store.
        int relus = CountOccurrences(ptx, "max.f32 %fc");
        Assert.Equal(8, relus);
        Assert.True(ptx.IndexOf("max.f32 %fc", StringComparison.Ordinal)
                  < ptx.IndexOf("wmma.store", StringComparison.Ordinal));
    }

    /// <summary>
    /// A long contraction must become a real loop. Unrolling it without bound is how the
    /// scalar emitter produced kernels ptxas could not allocate.
    /// </summary>
    [Fact]
    public void LongContraction_BecomesALoop()
    {
        var emitter = new PtxTensorCoreEmitter();
        string ptx = emitter.Emit(MatMul(64, 4096, 64), Sm86Major, Sm86Minor);

        Assert.False(emitter.Unrolled);
        Assert.Contains("KLOOP:", ptx, StringComparison.Ordinal);
        Assert.Equal(1, emitter.MmaInstructions);
    }

    /// <summary>A short contraction is unrolled, so the loop overhead is not paid.</summary>
    [Fact]
    public void ShortContraction_IsUnrolled()
    {
        var emitter = new PtxTensorCoreEmitter();
        emitter.Emit(MatMul(64, 128, 64), Sm86Major, Sm86Minor);

        Assert.True(emitter.Unrolled);
        Assert.Equal(8, emitter.MmaInstructions);        // 128 / 16
    }

    // ---- What the recogniser must REFUSE -------------------------------------------------
    //
    // Each of these would produce a kernel that runs and is wrong, so a bare `false` is not
    // enough: the reason has to name the property, otherwise "not eligible" is
    // indistinguishable from "the tensor cores never help".

    /// <summary>A shape that is not a whole number of tiles would read past the operands.</summary>
    [Theory]
    [InlineData(40, 64, 64)]
    [InlineData(64, 24, 64)]
    [InlineData(64, 64, 40)]
    public void PartialTiles_AreRefused(int m, int k, int n)
    {
        Assert.False(PtxTensorCoreEmitter.TryPlan(
            MatMul(m, k, n), Sm86Major, Sm86Minor, out _, out string reason));
        Assert.Contains("16x16x16", reason, StringComparison.Ordinal);
    }

    /// <summary>fp32 multiplicands are not what this wmma shape takes.</summary>
    [Fact]
    public void Fp32Operands_AreRefused()
    {
        Assert.False(PtxTensorCoreEmitter.TryPlan(
            MatMul(64, 64, 64, operands: CodegenElementType.Float32),
            Sm86Major, Sm86Minor, out _, out string reason));
        Assert.Contains("fp16", reason, StringComparison.Ordinal);
    }

    /// <summary>The tensor cores sum products; a maximum is a different reduction.</summary>
    [Fact]
    public void NonSumReduction_IsRefused()
    {
        Assert.False(PtxTensorCoreEmitter.TryPlan(
            MatMul(64, 64, 64, reduce: CodegenReduceKind.Max),
            Sm86Major, Sm86Minor, out _, out string reason));
        Assert.Contains("sum of products", reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// A pre-reduction transform would have to be applied per element BEFORE the multiply,
    /// which needs the fragment layout wmma does not expose.
    /// </summary>
    [Fact]
    public void PreReduceTransform_IsRefused()
    {
        Assert.False(PtxTensorCoreEmitter.TryPlan(
            MatMul(64, 64, 64, preReduce: CodegenPreReduceOp.Square),
            Sm86Major, Sm86Minor, out _, out string reason));
        Assert.Contains("fragment layout", reason, StringComparison.Ordinal);
    }

    /// <summary>A convolution is not a matmul, however much its inner loop looks like one.</summary>
    [Fact]
    public void NonMatMulIndexMap_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("m", 64), CodegenAxis.Parallel("n", 64),
            CodegenAxis.Reduce("k", 64));

        // A sliding window over A -- affine, valid, and exactly the map a convolution
        // produces. Its second index mixes two axes, so it is not a matrix operand.
        var a = new CodegenTensorBinding(0, "a", new[] { 64, 128 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Window(1, 2, 1, 0) },
            elementType: CodegenElementType.Float16);
        var b = new CodegenTensorBinding(1, "b", new[] { 64, 64 },
            new[] { CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(1) },
            elementType: CodegenElementType.Float16);
        var output = new CodegenTensorBinding(2, "out", new[] { 64, 64 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        var spec = new CodegenKernelSpec("tc_strided", space, new[] { a, b }, output,
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        Assert.False(PtxTensorCoreEmitter.TryPlan(
            spec, Sm86Major, Sm86Minor, out _, out string reason));
        Assert.Contains("plain 2-D matrix", reason, StringComparison.Ordinal);
    }

    /// <summary>Devices without tensor cores are refused rather than emitted for.</summary>
    [Fact]
    public void PreVoltaDevice_IsRefused()
    {
        Assert.False(PtxTensorCoreEmitter.TryPlan(
            MatMul(64, 64, 64), 6, 1, out _, out string reason));
        Assert.Contains("sm_70", reason, StringComparison.Ordinal);
    }

    /// <summary>Emitting an ineligible spec throws rather than producing a wrong kernel.</summary>
    [Fact]
    public void EmittingAnIneligibleSpec_Throws()
    {
        var ex = Assert.Throws<NotSupportedException>(() =>
            new PtxTensorCoreEmitter().Emit(MatMul(40, 64, 64), Sm86Major, Sm86Minor));
        Assert.Contains("16x16x16", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The launch geometry must cover every tile exactly once. Getting this wrong leaves a
    /// strip of the output untouched -- which reads as a correct kernel on any test whose
    /// tile count happens to divide the block size.
    /// </summary>
    [Theory]
    [InlineData(16, 16, 1)]        // 1 tile  -> 1 block
    [InlineData(64, 64, 4)]        // 16 tiles -> 4 blocks of 4 warps
    [InlineData(48, 32, 2)]        // 6 tiles  -> 2 blocks, last one partly idle
    public void LaunchGeometry_CoversEveryTile(int m, int n, int expectedBlocks)
    {
        var emitter = new PtxTensorCoreEmitter();
        Assert.True(PtxTensorCoreEmitter.TryPlan(
            MatMul(m, 64, n), Sm86Major, Sm86Minor, out var plan, out string reason), reason);

        Assert.Equal(expectedBlocks, emitter.BlockCount(plan!));
        Assert.Equal(128, emitter.BlockThreads);
        Assert.True(emitter.BlockCount(plan!) * emitter.WarpsPerBlock >= plan!.TileCount);
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
