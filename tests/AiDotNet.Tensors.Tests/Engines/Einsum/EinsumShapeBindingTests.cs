using AiDotNet.Tensors.Engines.Einsum;
using Xunit;

public class EinsumShapeBindingTests
{
    private static EinsumShapeBinding Bind(string equation, params int[][] shapes)
        => EinsumShapeBinding.Bind(EinsumEquation.Parse(equation), shapes);

    // --- Basic ---------------------------------------------------------

    [Fact]
    public void Matmul_BindsLabelSizesAndOutput()
    {
        var b = Bind("ij,jk->ik", new[] { 3, 5 }, new[] { 5, 7 });
        Assert.Equal(3, b.LabelSizes['i']);
        Assert.Equal(5, b.LabelSizes['j']);
        Assert.Equal(7, b.LabelSizes['k']);
        Assert.Equal(new[] { 3, 7 }, b.OutputShape);
        Assert.Empty(b.BatchDims);
    }

    [Fact]
    public void Transpose_OutputShape()
    {
        var b = Bind("ij->ji", new[] { 3, 5 });
        Assert.Equal(new[] { 5, 3 }, b.OutputShape);
    }

    [Fact]
    public void ScalarReduction_HasEmptyShape()
    {
        var b = Bind("ij->", new[] { 3, 5 });
        Assert.Empty(b.OutputShape);
    }

    [Fact]
    public void OuterProduct_OutputShape()
    {
        var b = Bind("i,j->ij", new[] { 3 }, new[] { 5 });
        Assert.Equal(new[] { 3, 5 }, b.OutputShape);
    }

    [Fact]
    public void ThreeWayContraction_BuildsShape()
    {
        var b = Bind("ab,bc,cd->ad", new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 });
        Assert.Equal(new[] { 2, 5 }, b.OutputShape);
    }

    // --- Diagonal / trace (repeated labels in one operand) ------------

    [Fact]
    public void Diagonal_SameSize_Succeeds()
    {
        var b = Bind("ii->i", new[] { 4, 4 });
        Assert.Equal(new[] { 4 }, b.OutputShape);
    }

    [Fact]
    public void Trace_SameSize_Succeeds()
    {
        var b = Bind("ii->", new[] { 6, 6 });
        Assert.Empty(b.OutputShape);
    }

    [Fact]
    public void Diagonal_DifferentSizes_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => Bind("ii->i", new[] { 4, 5 }));
        Assert.Contains("inconsistent sizes", ex.Message);
    }

    // --- Ellipsis happy paths -----------------------------------------

    [Fact]
    public void Ellipsis_BatchedMatmul()
    {
        var b = Bind("...ij,...jk->...ik",
            new[] { 2, 3, 4 },    // batch=2, i=3, j=4
            new[] { 2, 4, 5 });   // batch=2, j=4, k=5
        Assert.Equal(new[] { 2, 3, 5 }, b.OutputShape);
        Assert.Equal(new[] { 2 }, b.BatchDims);
    }

    [Fact]
    public void Ellipsis_BroadcastsSingleton()
    {
        var b = Bind("...ij,...jk->...ik",
            new[] { 1, 3, 4 },
            new[] { 5, 4, 6 });
        Assert.Equal(new[] { 5, 3, 6 }, b.OutputShape);
        Assert.Equal(new[] { 5 }, b.BatchDims);
    }

    [Fact]
    public void Ellipsis_RightAlignsDifferentRanks()
    {
        // first operand has 2 batch dims, second has 0 → broadcast to 2.
        var b = Bind("...ij,jk->...ik",
            new[] { 2, 3, 5, 4 },  // batch=[2,3], i=5, j=4
            new[] { 4, 6 });       // j=4, k=6
        Assert.Equal(new[] { 2, 3, 5, 6 }, b.OutputShape);
        Assert.Equal(new[] { 2, 3 }, b.BatchDims);
    }

    [Fact]
    public void Ellipsis_InMiddle_BindsCorrectly()
    {
        // "a...b" on shape [2, 3, 4, 5]: a=2, batch=[3,4], b=5.
        var b = Bind("a...b->a...b", new[] { 2, 3, 4, 5 });
        Assert.Equal(new[] { 2, 3, 4, 5 }, b.OutputShape);
        Assert.Equal(2, b.LabelSizes['a']);
        Assert.Equal(5, b.LabelSizes['b']);
        Assert.Equal(new[] { 3, 4 }, b.BatchDims);
    }

    [Fact]
    public void EllipsisOnly_OutputShapeIsBatchDims()
    {
        var b = Bind("...,...->...",
            new[] { 2, 3 },
            new[] { 2, 3 });
        Assert.Equal(new[] { 2, 3 }, b.OutputShape);
    }

    // --- Error cases ---------------------------------------------------

    [Fact]
    public void RankMismatch_NoEllipsis_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => Bind("ij,jk->ik", new[] { 3, 5, 7 }, new[] { 5, 7 }));
        Assert.Contains("rank 3", ex.Message);
    }

    [Fact]
    public void TooFewOperands_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => EinsumShapeBinding.Bind(
                EinsumEquation.Parse("ij,jk->ik"),
                new[] { new[] { 3, 5 } }));
        Assert.Contains("expects 2", ex.Message);
    }

    [Fact]
    public void LabelSizeMismatch_AcrossOperands_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => Bind("ij,jk->ik", new[] { 3, 4 }, new[] { 5, 7 }));
        Assert.Contains("inconsistent sizes", ex.Message);
    }

    [Fact]
    public void EllipsisBatchDim_NonBroadcastable_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => Bind("...ij,...jk->...ik",
                new[] { 2, 3, 4 },
                new[] { 5, 4, 6 }));
        Assert.Contains("broadcast", ex.Message);
    }

    [Fact]
    public void NegativeDim_Throws()
    {
        var ex = Assert.Throws<EinsumShapeException>(
            () => Bind("ij->", new[] { -1, 5 }));
        Assert.Contains("negative", ex.Message);
    }

    // --- Implicit-output shape ----------------------------------------

    [Fact]
    public void ImplicitOutput_ResolvesShape()
    {
        var b = Bind("ij,jk", new[] { 3, 5 }, new[] { 5, 7 });
        Assert.Equal(new[] { 3, 7 }, b.OutputShape);
    }

    [Fact]
    public void ImplicitOutputWithEllipsis_Prepends()
    {
        var b = Bind("...ij,...jk",
            new[] { 2, 3, 4 },
            new[] { 2, 4, 5 });
        // Implicit output = "...ik"; shape = [2, 3, 5]
        Assert.Equal(new[] { 2, 3, 5 }, b.OutputShape);
    }

    // --- ToString / OperandEllipsisDims --------------------------------

    [Fact]
    public void ToString_ShowsResolvedShape()
    {
        var b = Bind("ij,jk->ik", new[] { 3, 5 }, new[] { 5, 7 });
        Assert.Equal("ij,jk->ik -> [3, 7]", b.ToString());
    }

    [Fact]
    public void OperandEllipsisDims_ReflectsPerOperandEllipsis()
    {
        var b = Bind("...ij,jk->...ik",
            new[] { 2, 3, 5, 4 },
            new[] { 4, 6 });
        Assert.Equal(new[] { 2, 3 }, b.OperandEllipsisDims[0]);
        Assert.Empty(b.OperandEllipsisDims[1]);
    }
}
