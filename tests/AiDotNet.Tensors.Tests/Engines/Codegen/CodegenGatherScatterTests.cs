// Copyright (c) AiDotNet. All rights reserved.
// Data-dependent indexing: gather and scatter.
//
// The IR previously documented these as "deliberately not expressible ... out of scope for
// this layer", which meant every operator needing one -- embedding lookup and its backward,
// one-hot projection, sparse accumulation, deformable convolution's learned offsets -- had
// to be hand-written outside the generator.
//
// The dangerous case here is scatter. A destination reached through a run-time index cannot
// be proven unique, so two iterations may hit the same element; a plain store keeps whichever
// warp finished last. That is a race producing a DIFFERENT wrong answer per run, which reads
// as flakiness rather than as a bug, so several tests below are about the atomic.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGatherScatterTests
{
    /// <summary>
    /// Embedding lookup: <c>out[t, e] = table[ids[t], e]</c>. The row index is read from a
    /// tensor at run time; the column is ordinary and affine.
    /// </summary>
    private static CodegenKernelSpec EmbeddingGather(
        int tokens, int vocabulary, int width,
        CodegenIndexOutOfRange policy = CodegenIndexOutOfRange.Skip)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) },
            elementType: CodegenElementType.Int32);

        var table = new CodegenTensorBinding(1, "table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary, policy),
                null,
            });

        var output = new CodegenTensorBinding(2, "out", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        // No reduction axis: a gather copies a row, it does not contract one.
        return new CodegenKernelSpec("embedding_gather", space, new[] { ids, table }, output,
            new[] { 1 }, CodegenReduceKind.None);
    }

    /// <summary>
    /// The embedding backward: <c>grad_table[ids[t], e] += grad_out[t, e]</c>. Repeated
    /// tokens make the destination genuinely non-unique.
    /// </summary>
    private static CodegenKernelSpec EmbeddingScatter(int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) },
            elementType: CodegenElementType.Int32);

        var grad = new CodegenTensorBinding(1, "grad", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });

        var table = new CodegenTensorBinding(2, "grad_table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            isOutput: true,
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });

        return new CodegenKernelSpec("embedding_scatter", space, new[] { ids, grad }, table,
            new[] { 1 }, CodegenReduceKind.None);
    }

    private static double[] Ids(params int[] values)
    {
        var data = new double[values.Length];
        for (int i = 0; i < values.Length; i++) data[i] = values[i];
        return data;
    }

    private static double[] Ramp(int count, double scale = 1.0)
    {
        var data = new double[count];
        for (int i = 0; i < count; i++) data[i] = (i + 1) * scale;
        return data;
    }

    // ---- Gather ------------------------------------------------------------------------

    /// <summary>A gather must fetch the row the index names, against hand-written arithmetic.</summary>
    [Fact]
    public void Gather_FetchesTheIndexedRow()
    {
        var spec = EmbeddingGather(tokens: 3, vocabulary: 4, width: 2);
        var ids = Ids(2, 0, 3);
        var table = Ramp(4 * 2);          // rows: [1,2] [3,4] [5,6] [7,8]

        double[] got = spec.Interpret(new[] { ids, table });

        Assert.Equal(new double[] { 5, 6, 1, 2, 7, 8 }, got);
    }

    /// <summary>The same row fetched twice is not a special case.</summary>
    [Fact]
    public void Gather_HandlesRepeatedIndices()
    {
        var spec = EmbeddingGather(tokens: 3, vocabulary: 4, width: 2);
        double[] got = spec.Interpret(new[] { Ids(1, 1, 1), Ramp(4 * 2) });

        Assert.Equal(new double[] { 3, 4, 3, 4, 3, 4 }, got);
    }

    /// <summary>
    /// Under Skip, an out-of-range index contributes nothing -- what a padding row or a -1
    /// sentinel means.
    /// </summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(4)]
    [InlineData(9999)]
    public void Gather_SkipPolicy_YieldsZeroForAnOutOfRangeIndex(int bad)
    {
        var spec = EmbeddingGather(tokens: 2, vocabulary: 4, width: 2);
        double[] got = spec.Interpret(new[] { Ids(1, bad), Ramp(4 * 2) });

        Assert.Equal(new double[] { 3, 4, 0, 0 }, got);
    }

    /// <summary>Under Clamp, the edge row is genuinely read.</summary>
    [Theory]
    [InlineData(-1, 1.0, 2.0)]        // clamps to row 0
    [InlineData(99, 7.0, 8.0)]        // clamps to row 3
    public void Gather_ClampPolicy_ReadsTheEdgeRow(int bad, double first, double second)
    {
        var spec = EmbeddingGather(2, 4, 2, CodegenIndexOutOfRange.Clamp);
        double[] got = spec.Interpret(new[] { Ids(1, bad), Ramp(4 * 2) });

        Assert.Equal(new double[] { 3, 4, first, second }, got);
    }

    /// <summary>The emitted kernel must load the index and clamp it before addressing.</summary>
    [Fact]
    public void Gather_EmitsAnIndexLoadAndAClamp()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(EmbeddingGather(64, 512, 32), 8, 6);

        Assert.Equal(1, emitter.IndirectIndexLoads);
        Assert.Contains("ld.global.nc.u32", ptx, StringComparison.Ordinal);

        // THE CLAMP IS UNCONDITIONAL, and is not the caller's policy: it is what keeps a
        // malformed index tensor from forming an address outside the allocation. Predicating
        // the load alone would not do it -- the address is computed either way.
        Assert.Contains("max.s32", ptx, StringComparison.Ordinal);
        Assert.Contains("min.s32", ptx, StringComparison.Ordinal);
    }

    /// <summary>Under Skip the range test must also gate the access.</summary>
    [Fact]
    public void Gather_SkipPolicy_EmitsARangePredicate()
    {
        string ptx = new PtxAffineEmitter().Emit(EmbeddingGather(64, 512, 32), 8, 6);

        Assert.Contains("setp.lt.s32", ptx, StringComparison.Ordinal);
        Assert.Contains("@%p", ptx, StringComparison.Ordinal);          // a predicated load
    }

    // ---- Scatter -----------------------------------------------------------------------

    /// <summary>
    /// Scatter ACCUMULATES. Repeated tokens are the ordinary case in an embedding backward,
    /// not a corner case, and assigning would drop all but one gradient.
    /// </summary>
    [Fact]
    public void Scatter_AccumulatesRepeatedIndices()
    {
        var spec = EmbeddingScatter(tokens: 3, vocabulary: 2, width: 2);

        // tokens 0 and 2 both target row 1.
        var ids = Ids(1, 0, 1);
        var grad = new double[] { 1, 2, 10, 20, 100, 200 };

        double[] got = spec.Interpret(new[] { ids, grad });

        Assert.Equal(new double[] { 10, 20, 101, 202 }, got);
    }

    /// <summary>An out-of-range scatter index writes nowhere rather than clamping onto row 0.</summary>
    [Fact]
    public void Scatter_SkipPolicy_DropsAnOutOfRangeIndex()
    {
        var spec = EmbeddingScatter(tokens: 2, vocabulary: 2, width: 2);
        double[] got = spec.Interpret(new[] { Ids(0, -1), new double[] { 1, 2, 999, 999 } });

        Assert.Equal(new double[] { 1, 2, 0, 0 }, got);
    }

    /// <summary>
    /// The store must be atomic. This is decided by the structure, not by the caller: an
    /// output dimension addressed at run time cannot be proven injective.
    /// </summary>
    [Fact]
    public void Scatter_EmitsAnAtomicAccumulation()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(EmbeddingScatter(128, 512, 32), 8, 6);

        Assert.Equal(1, emitter.AtomicStores);
        Assert.Contains("red.global.add.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("st.global.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>An ordinary output is NOT made atomic -- that would be a needless cost.</summary>
    [Fact]
    public void DirectOutput_KeepsAPlainStore()
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(EmbeddingGather(64, 512, 32), 8, 6);

        Assert.Equal(0, emitter.AtomicStores);
        Assert.Contains("st.global.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global.add.f32", ptx, StringComparison.Ordinal);
    }

    /// <summary>The structural facts a caller can rely on.</summary>
    [Fact]
    public void BindingReportsItsIndirection()
    {
        var gather = EmbeddingGather(8, 16, 4);
        var scatter = EmbeddingScatter(8, 16, 4);

        Assert.True(gather.Inputs[1].HasIndirection);
        Assert.False(gather.Output.HasIndirection);
        Assert.False(gather.Output.NeedsAtomicStore);

        Assert.True(scatter.Output.HasIndirection);
        Assert.True(scatter.Output.NeedsAtomicStore);
    }

    // ---- What must be refused ----------------------------------------------------------

    /// <summary>
    /// Addressing with a float tensor would reinterpret its bit pattern as an integer --
    /// which neither faults nor looks wrong in the generated PTX.
    /// </summary>
    [Fact]
    public void IndexSourceThatIsNotAnIndexTensor_IsRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("t", 4));

        var floats = new CodegenTensorBinding(0, "not_ids", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) });          // fp32, not Int32

        var table = new CodegenTensorBinding(1, "table", new[] { 8 },
            new[] { CodegenAffineExpr.Const(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), 8),
            });

        var output = new CodegenTensorBinding(2, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { floats, table }, output, new[] { 1 }, CodegenReduceKind.None));

        Assert.Contains("rather than Int32", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>An index tensor read as arithmetic would compute with a bit pattern.</summary>
    [Fact]
    public void IndexTensorUsedAsAnOperand_IsRefused()
    {
        var space = new CodegenIterationSpace(CodegenAxis.Parallel("t", 4));

        var ids = new CodegenTensorBinding(0, "ids", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) },
            elementType: CodegenElementType.Int32);
        var output = new CodegenTensorBinding(1, "out", new[] { 4 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        // Refused when the SPEC is built, not at the load site. The emitter has several
        // load paths -- scalar, vectorised, staged, coarsened -- and a guard on one of them
        // is a guard on none; the first version of this check sat in EmitLoad and this test
        // caught a kernel that read the index buffer as fp32 without complaint.
        var ex = Assert.Throws<ArgumentException>(() => new CodegenKernelSpec(
            "bad", space, new[] { ids }, output, new[] { 0 }, CodegenReduceKind.None));

        Assert.Contains("index tensor", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The indirection's bound must be the dimension it addresses, or the guard would admit
    /// indices the allocation does not contain.
    /// </summary>
    [Fact]
    public void BoundThatDisagreesWithTheDimension_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() => new CodegenTensorBinding(
            1, "table", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), bound: 16),
                null,
            }));

        Assert.Contains("bounded at 16", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// An indirect dimension's affine map is dead, so it must be Const(0): any consumer that
    /// reads the map without consulting the indirection then computes an obviously wrong
    /// address rather than a plausible one.
    /// </summary>
    [Fact]
    public void LiveAffineMapOnAnIndirectDimension_IsRefused()
    {
        var ex = Assert.Throws<ArgumentException>(() => new CodegenTensorBinding(
            1, "table", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), bound: 8),
                null,
            }));

        Assert.Contains("must be Const(0)", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>A scatter destination must be fp32: there is no 16-bit atomic add here.</summary>
    [Fact]
    public void NarrowScatterDestination_IsRefused()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", 8), CodegenAxis.Parallel("e", 4));

        var ids = new CodegenTensorBinding(0, "ids", new[] { 8 },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);
        var grad = new CodegenTensorBinding(1, "grad", new[] { 8, 4 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var table = new CodegenTensorBinding(2, "grad_table", new[] { 16, 4 },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            isOutput: true, elementType: CodegenElementType.Float16,
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), 16),
                null,
            });

        var spec = new CodegenKernelSpec("narrow_scatter", space, new[] { ids, grad }, table,
            new[] { 1 }, CodegenReduceKind.None);

        var ex = Assert.Throws<NotSupportedException>(
            () => new PtxAffineEmitter().Emit(spec, 8, 6));
        Assert.Contains("must be fp32", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Affine-only kernels must emit exactly what they emitted before.</summary>
    [Fact]
    public void AffineOnlyKernels_AreUnchanged()
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", 32), CodegenAxis.Reduce("j", 16));
        var x = new CodegenTensorBinding(0, "x", new[] { 32, 16 },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var output = new CodegenTensorBinding(1, "out", new[] { 32 },
            new[] { CodegenAffineExpr.Axis(0) }, isOutput: true);

        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(new CodegenKernelSpec("plain", space, new[] { x }, output,
            new[] { 0 }, CodegenReduceKind.Sum), 8, 6);

        Assert.Equal(0, emitter.IndirectIndexLoads);
        Assert.Equal(0, emitter.AtomicStores);
        Assert.DoesNotContain("ld.global.nc.u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global.add.f32", ptx, StringComparison.Ordinal);
    }
}
