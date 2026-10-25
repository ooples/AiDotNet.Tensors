// Copyright (c) AiDotNet. All rights reserved.
// The PTX lowering of indirect indices (#1441), pinned where the device first disagreed with the interpreter: a shift
// is a value (never range-checked), a tiled axis read only through an indirect index still gives each lane its own load
// and range test, a vector load's guard reaches the term's identity select, and a rejected entry of a multi-operand
// product contributes the identity rather than 0 times its other operands.

using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenIndirectLoweringTests
{
    private static readonly CodegenLibraryFunction NotEqual =
        new("not_equal_scalar", "test", 2, a => a[0] != a[1] ? 1.0 : 0.0);

    private static string EmitPass(CodegenProgram program, string name, out PtxAffineEmitter emitter)
    {
        foreach (var pass in program.Passes)
            if (pass.Name == name)
            {
                emitter = new PtxAffineEmitter();
                return emitter.Emit(pass, 8, 6);
            }
        throw new Xunit.Sdk.XunitException("no pass " + name);
    }

    /// <summary>
    /// CSR SpMM's row-start test shifts by p - (nnz - 1), which is negative for every p but the last. Range-checking
    /// the shift as though it were an address rejected every entry of row r but the last, so rows came out 0.
    /// </summary>
    [Fact]
    public void AnIndirectShiftIsNotRangeChecked()
    {
        string ptx = EmitPass(CodegenBackendPrograms.CsrSpMM(17, 9, 1, 40, NotEqual), "csr_spmm", out _);
        Assert.DoesNotContain(", 2147483647;", ptx);
    }

    /// <summary>
    /// A deterministic scatter whose destination is the contiguous output axis, so it is the axis the emitter
    /// coarsens; the source reads it only through the indirect shift. Keyed on the index map alone, every lane reused
    /// lane 0's load and range test, and every destination summed lane 0's sources.
    /// </summary>
    [Fact]
    public void EachLaneOfATiledDestinationTestsItsOwnRange()
    {
        const int features = 8, destinations = 64, count = 1024;   // a rolled reduction loop: one trip in the body
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("f", features), CodegenAxis.Parallel("d", destinations), CodegenAxis.Reduce("i", count));
        var indices = new CodegenTensorBinding(0, "indices", new[] { count }, new[] { CodegenAffineExpr.Axis(2) },
            elementType: CodegenElementType.Int32);
        var source = new CodegenTensorBinding(1, "source", new[] { 1, count, features },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(2), CodegenAffineExpr.Axis(0) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(2), 1, CodegenIndexOutOfRange.Skip,
                    floatIndices: false, offset: CodegenAffineExpr.Axis(1)),
                null, null,
            });
        var output = new CodegenTensorBinding(2, "out", new[] { features, destinations },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(new CodegenKernelSpec("scatter_d_contiguous", space, new[] { indices, source }, output,
            new[] { 1 }, CodegenReduceKind.Sum), 8, 6);

        int rangeTests = Regex.Matches(ptx, @"setp\.lt\.s32 %p\d+, %r\d+, 1;").Count;
        int guardedLoads = Regex.Matches(ptx, @"@%p\d+ ld\.global").Count;
        Assert.True(rangeTests > 1, "the destination axis is expected to be coarsened here; got " + rangeTests + " tests");
        Assert.Equal(rangeTests, guardedLoads);
    }
    /// <summary>
    /// The scatter softmax's reciprocal sum loads four features with one guarded vector load and adds exp(x - max)
    /// per term: a rejected row must add 0, not exp(0 - max), so the vector's guard has to reach the identity select.
    /// </summary>
    [Fact]
    public void AGuardedVectorLoadSelectsTheIdentityUnderAPreReduce()
    {
        string ptx = EmitPass(CodegenBackendPrograms.ScatterSoftmaxRows(1000, 16, 37, NotEqual),
            "scatter_softmax_recipsum", out var emitter);
        Assert.True(emitter.VectorisedLoads > 0, "the source is expected to load as vectors here");
        Assert.Equal(4, Regex.Matches(ptx, @"selp\.f32").Count);
    }

    /// <summary>
    /// A CSR entry of another row reads 0 from its row-bound tests, and 0 times an infinite dense value is NaN: the
    /// product must select the identity where an indirect Skip test fails.
    /// </summary>
    [Fact]
    public void ARejectedEntryOfAProductContributesTheIdentity()
    {
        string ptx = EmitPass(CodegenBackendPrograms.CsrSpMM(17, 9, 1, 40, NotEqual), "csr_spmm", out _);
        Assert.Contains("selp.f32", ptx);
    }
}