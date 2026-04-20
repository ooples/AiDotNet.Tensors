using System.Linq;
using AiDotNet.Tensors.Engines.Einsum;
using Xunit;

public class EinsumEquationTests
{
    // --- Explicit-output happy paths ------------------------------------

    [Fact]
    public void Matmul_ExplicitOutput_Parses()
    {
        var eq = EinsumEquation.Parse("ij,jk->ik");
        Assert.True(eq.HasExplicitOutput);
        Assert.False(eq.HasEllipsis);
        Assert.Equal(2, eq.Operands.Count);
        Assert.Equal(new[] { 'i', 'j' }, eq.Operands[0].Labels);
        Assert.Equal(new[] { 'j', 'k' }, eq.Operands[1].Labels);
        Assert.Equal(new[] { 'i', 'k' }, eq.Output.Labels);
        Assert.Equal(new[] { 'j' }, eq.ContractionLabels.OrderBy(c => c));
    }

    [Fact]
    public void BatchedMatmul_WithExplicitOutput_Parses()
    {
        var eq = EinsumEquation.Parse("bij,bjk->bik");
        Assert.Equal(3, eq.Output.Labels.Count);
        Assert.Contains('j', eq.ContractionLabels);
        Assert.DoesNotContain('b', eq.ContractionLabels);
    }

    [Fact]
    public void AttentionScores_Parses()
    {
        var eq = EinsumEquation.Parse("bhqd,bhkd->bhqk");
        Assert.Equal(2, eq.Operands.Count);
        Assert.Equal(4, eq.Output.Labels.Count);
        Assert.Contains('d', eq.ContractionLabels);
    }

    [Fact]
    public void Whitespace_IsStripped()
    {
        var eq = EinsumEquation.Parse(" ij , jk -> ik ");
        Assert.Equal(new[] { 'i', 'j' }, eq.Operands[0].Labels);
        Assert.Equal(new[] { 'j', 'k' }, eq.Operands[1].Labels);
        Assert.Equal(new[] { 'i', 'k' }, eq.Output.Labels);
    }

    [Fact]
    public void UppercaseLabels_AreAllowed()
    {
        var eq = EinsumEquation.Parse("IJ,JK->IK");
        Assert.Equal(new[] { 'I', 'J' }, eq.Operands[0].Labels);
        Assert.Equal(new[] { 'I', 'K' }, eq.Output.Labels);
    }

    [Fact]
    public void ThreeOperands_Chain_Parses()
    {
        var eq = EinsumEquation.Parse("ab,bc,cd->ad");
        Assert.Equal(3, eq.Operands.Count);
        Assert.Equal(new[] { 'a', 'd' }, eq.Output.Labels);
        // 'b' and 'c' are contracted.
        Assert.Contains('b', eq.ContractionLabels);
        Assert.Contains('c', eq.ContractionLabels);
    }

    [Fact]
    public void Transpose_Parses()
    {
        var eq = EinsumEquation.Parse("ij->ji");
        Assert.Single(eq.Operands);
        Assert.Equal(new[] { 'j', 'i' }, eq.Output.Labels);
        Assert.Empty(eq.ContractionLabels);
    }

    [Fact]
    public void Reduction_ToScalar_Parses()
    {
        var eq = EinsumEquation.Parse("ij->");
        Assert.Empty(eq.Output.Labels);
        Assert.Equal(2, eq.ContractionLabels.Count);
    }

    [Fact]
    public void OuterProduct_Parses()
    {
        var eq = EinsumEquation.Parse("i,j->ij");
        Assert.Equal(2, eq.Operands.Count);
        Assert.Equal(new[] { 'i', 'j' }, eq.Output.Labels);
        Assert.Empty(eq.ContractionLabels);
    }

    // --- Trace / diagonal (repeated labels within one operand) ---------

    [Fact]
    public void Trace_ExplicitScalarOutput_Parses()
    {
        var eq = EinsumEquation.Parse("ii->");
        Assert.Single(eq.Operands);
        Assert.Equal(new[] { 'i', 'i' }, eq.Operands[0].Labels);
        Assert.Empty(eq.Output.Labels);
    }

    [Fact]
    public void Diagonal_ExplicitOutput_Parses()
    {
        var eq = EinsumEquation.Parse("ii->i");
        Assert.Equal(new[] { 'i' }, eq.Output.Labels);
    }

    // --- Implicit output ------------------------------------------------

    [Fact]
    public void ImplicitOutput_MatmulPattern()
    {
        var eq = EinsumEquation.Parse("ij,jk");
        Assert.False(eq.HasExplicitOutput);
        Assert.Equal(new[] { 'i', 'k' }, eq.Output.Labels);
        Assert.Equal(new[] { 'j' }, eq.ContractionLabels.OrderBy(c => c));
    }

    [Fact]
    public void ImplicitOutput_IsSortedAlphabetically()
    {
        // 'k' and 'i' both appear exactly once ⇒ output is "ik" (sorted).
        var eq = EinsumEquation.Parse("ki,ij");
        Assert.Equal(new[] { 'j', 'k' }, eq.Output.Labels);
    }

    [Fact]
    public void ImplicitOutput_Trace_IsScalar()
    {
        // "ii": label 'i' appears twice in the same operand → not in output.
        var eq = EinsumEquation.Parse("ii");
        Assert.Empty(eq.Output.Labels);
    }

    [Fact]
    public void ImplicitOutput_WithEllipsis_HasLeadingEllipsis()
    {
        var eq = EinsumEquation.Parse("...ij,...jk");
        Assert.True(eq.Output.HasEllipsis);
        Assert.Equal(0, eq.Output.EllipsisPosition);
        Assert.Equal(new[] { 'i', 'k' }, eq.Output.Labels);
    }

    // --- Ellipsis ------------------------------------------------------

    [Fact]
    public void Ellipsis_ExplicitOutput_Parses()
    {
        var eq = EinsumEquation.Parse("...ij,...jk->...ik");
        Assert.True(eq.HasEllipsis);
        Assert.Equal(0, eq.Operands[0].EllipsisPosition);
        Assert.Equal(0, eq.Operands[1].EllipsisPosition);
        Assert.Equal(0, eq.Output.EllipsisPosition);
        Assert.Equal(new[] { 'i', 'j' }, eq.Operands[0].Labels);
    }

    [Fact]
    public void Ellipsis_InTheMiddle_Parses()
    {
        var eq = EinsumEquation.Parse("a...b,b...c->a...c");
        var op0 = eq.Operands[0];
        Assert.Equal(1, op0.EllipsisPosition);
        Assert.Equal(new[] { 'a', 'b' }, op0.Labels);
        Assert.Equal(1, op0.CountBeforeEllipsis);
        Assert.Equal(1, op0.CountAfterEllipsis);
    }

    [Fact]
    public void Ellipsis_Only_Parses()
    {
        var eq = EinsumEquation.Parse("...,...->...");
        Assert.True(eq.HasEllipsis);
        Assert.All(eq.Operands, op => Assert.Empty(op.Labels));
        Assert.Empty(eq.Output.Labels);
    }

    [Fact]
    public void OperandToString_RoundTrips()
    {
        var eq = EinsumEquation.Parse("a...b,b...c->a...c");
        Assert.Equal("a...b", eq.Operands[0].ToString());
        Assert.Equal("b...c", eq.Operands[1].ToString());
        Assert.Equal("a...c", eq.Output.ToString());
    }

    // --- Error cases ---------------------------------------------------

    [Fact]
    public void LoneDot_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,.k"));
        Assert.Contains("...", ex.Message);
    }

    [Fact]
    public void DoubleEllipsis_InSameOperand_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("...ij...,jk"));
        Assert.Contains("more than one", ex.Message);
    }

    [Fact]
    public void MultipleArrows_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij->ik->il"));
        Assert.Contains("more than one", ex.Message);
    }

    [Fact]
    public void OutputLabel_NotInOperands_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,jk->ikl"));
        Assert.Contains("'l'", ex.Message);
    }

    [Fact]
    public void DuplicateOutputLabel_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,jk->iik"));
        Assert.Contains("more than once", ex.Message);
    }

    [Fact]
    public void OutputEllipsis_WithoutOperandEllipsis_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,jk->...ik"));
        Assert.Contains("no operand", ex.Message);
    }

    [Fact]
    public void NonLetterCharacter_Throws()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,j1->i1"));
        Assert.Contains("'1'", ex.Message);
    }

    [Fact]
    public void EmptyString_Throws()
    {
        // An empty equation has no operands.
        Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse(""));
    }

    [Fact]
    public void Error_IncludesCaretAtCorrectColumn()
    {
        var ex = Assert.Throws<EinsumSyntaxException>(() => EinsumEquation.Parse("ij,jk->iik"));
        // "iik" starts after "ij,jk->" which is 7 chars. Second 'i' is at
        // column 8 (0-based column 8 → 1-based column 9).
        Assert.Contains("column ", ex.Message);
        Assert.Equal(8, ex.Position);
    }

    // --- Labels and contraction set ------------------------------------

    [Fact]
    public void ContractionLabels_AreInputLabelsNotInOutput()
    {
        var eq = EinsumEquation.Parse("abc,bcd->ad");
        Assert.Contains('b', eq.ContractionLabels);
        Assert.Contains('c', eq.ContractionLabels);
        Assert.DoesNotContain('a', eq.ContractionLabels);
        Assert.DoesNotContain('d', eq.ContractionLabels);
    }

    [Fact]
    public void AllLabels_IncludesEveryDistinctOperandLabel()
    {
        var eq = EinsumEquation.Parse("abc,bcd->ad");
        Assert.Equal(new[] { 'a', 'b', 'c', 'd' }, eq.AllLabels.OrderBy(c => c));
    }

    [Fact]
    public void MinimumRank_ReflectsOperandLabels()
    {
        var eq = EinsumEquation.Parse("...ij,jk->...ik");
        Assert.Equal(2, eq.Operands[0].MinimumRank);
        Assert.Equal(2, eq.Operands[1].MinimumRank);
        Assert.True(eq.Operands[0].HasEllipsis);
        Assert.False(eq.Operands[1].HasEllipsis);
    }
}
