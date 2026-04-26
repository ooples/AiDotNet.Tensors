// Copyright (c) AiDotNet. All rights reserved.
// Phase A of issue #225: codegen IR definition — CodegenOpKind
// taxonomy, CodegenElementType, CodegenNode/Graph construction +
// invariants, and CodegenLowering helpers.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// Phase A guards. If these break, every downstream emitter in
/// Phase B + Phase C gets wrong output — they dispatch on the enum
/// and graph shape tested here.
/// </summary>
public class CodegenIrTests
{
    // ─── CodegenElementType taxonomy ──────────────────────────────────

    [Fact]
    public void ElementType_ByteWidth_MatchesStandardScalars()
    {
        Assert.Equal(4, CodegenElementType.Float32.GetByteWidth());
        Assert.Equal(8, CodegenElementType.Float64.GetByteWidth());
        Assert.Equal(2, CodegenElementType.Float16.GetByteWidth());
        Assert.Equal(2, CodegenElementType.BFloat16.GetByteWidth());
        Assert.Equal(4, CodegenElementType.Int32.GetByteWidth());
        Assert.Equal(8, CodegenElementType.Int64.GetByteWidth());
        Assert.Equal(1, CodegenElementType.Int8.GetByteWidth());
        Assert.Equal(1, CodegenElementType.UInt8.GetByteWidth());
    }

    [Fact]
    public void ElementType_BitWidth_ReportsSubByteExactly()
    {
        Assert.Equal(1, CodegenElementType.Int1.GetBitWidth());
        Assert.Equal(2, CodegenElementType.Int2.GetBitWidth());
        Assert.Equal(3, CodegenElementType.Int3.GetBitWidth());
        Assert.Equal(4, CodegenElementType.NF4.GetBitWidth());
        Assert.Equal(4, CodegenElementType.FP4.GetBitWidth());
        Assert.Equal(8, CodegenElementType.Int8.GetBitWidth());
        Assert.Equal(32, CodegenElementType.Float32.GetBitWidth());
    }

    [Fact]
    public void ElementType_IsFloatingPoint_RecognisesAllFloatVariants()
    {
        Assert.True(CodegenElementType.Float32.IsFloatingPoint());
        Assert.True(CodegenElementType.Float64.IsFloatingPoint());
        Assert.True(CodegenElementType.Float16.IsFloatingPoint());
        Assert.True(CodegenElementType.BFloat16.IsFloatingPoint());
        Assert.True(CodegenElementType.FP8_E4M3.IsFloatingPoint());
        Assert.True(CodegenElementType.FP8_E5M2.IsFloatingPoint());
        Assert.True(CodegenElementType.NF4.IsFloatingPoint());
        Assert.True(CodegenElementType.FP4.IsFloatingPoint());

        Assert.False(CodegenElementType.Int32.IsFloatingPoint());
        Assert.False(CodegenElementType.Int64.IsFloatingPoint());
        Assert.False(CodegenElementType.Int8.IsFloatingPoint());
        Assert.False(CodegenElementType.Bool.IsFloatingPoint());
    }

    [Fact]
    public void ElementType_IsSubByte_MatchesBitWidthBelow8()
    {
        Assert.True(CodegenElementType.Int1.IsSubByte());
        Assert.True(CodegenElementType.Int2.IsSubByte());
        Assert.True(CodegenElementType.Int3.IsSubByte());
        Assert.True(CodegenElementType.NF4.IsSubByte());
        Assert.True(CodegenElementType.FP4.IsSubByte());

        Assert.False(CodegenElementType.Int8.IsSubByte());
        Assert.False(CodegenElementType.Float32.IsSubByte());
    }

    // ─── CodegenOpKind categorisation ─────────────────────────────────

    [Fact]
    public void OpKind_Pointwise_IsClassifiedAsPointwise()
    {
        Assert.Equal(CodegenOpCategory.Pointwise, CodegenOpKinds.Categorize(CodegenOpKind.Add));
        Assert.Equal(CodegenOpCategory.Pointwise, CodegenOpKinds.Categorize(CodegenOpKind.Mul));
        Assert.Equal(CodegenOpCategory.Pointwise, CodegenOpKinds.Categorize(CodegenOpKind.ReLU));
        Assert.Equal(CodegenOpCategory.Pointwise, CodegenOpKinds.Categorize(CodegenOpKind.GELU));
        Assert.Equal(CodegenOpCategory.Pointwise, CodegenOpKinds.Categorize(CodegenOpKind.Exp));
    }

    [Fact]
    public void OpKind_Reduction_IsClassifiedAsReduction()
    {
        Assert.Equal(CodegenOpCategory.Reduction, CodegenOpKinds.Categorize(CodegenOpKind.ReduceSum));
        Assert.Equal(CodegenOpCategory.Reduction, CodegenOpKinds.Categorize(CodegenOpKind.ReduceMean));
        Assert.Equal(CodegenOpCategory.Reduction, CodegenOpKinds.Categorize(CodegenOpKind.Softmax));
    }

    [Fact]
    public void OpKind_MatMul_IsClassifiedAsMatmul()
    {
        Assert.Equal(CodegenOpCategory.Matmul, CodegenOpKinds.Categorize(CodegenOpKind.MatMul));
        Assert.Equal(CodegenOpCategory.Matmul, CodegenOpKinds.Categorize(CodegenOpKind.MatMulTransposeA));
        Assert.Equal(CodegenOpCategory.Matmul, CodegenOpKinds.Categorize(CodegenOpKind.BatchMatMul));
    }

    [Fact]
    public void OpKind_Attention_IsClassifiedAsAttention()
    {
        Assert.Equal(CodegenOpCategory.Attention,
            CodegenOpKinds.Categorize(CodegenOpKind.ScaledDotProductAttention));
    }

    [Fact]
    public void OpKind_IsPointwise_ExcludesLoadStoreAndReductions()
    {
        Assert.False(CodegenOpKinds.IsPointwise(CodegenOpKind.LoadInput));
        Assert.False(CodegenOpKinds.IsPointwise(CodegenOpKind.StoreOutput));
        Assert.False(CodegenOpKinds.IsPointwise(CodegenOpKind.ReduceSum));
        Assert.False(CodegenOpKinds.IsPointwise(CodegenOpKind.MatMul));

        Assert.True(CodegenOpKinds.IsPointwise(CodegenOpKind.Add));
        Assert.True(CodegenOpKinds.IsPointwise(CodegenOpKind.ReLU));
    }

    [Fact]
    public void OpKind_IsComparison_RecognisesAllCompareOps()
    {
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.Equal));
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.NotEqual));
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.Greater));
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.GreaterEqual));
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.Less));
        Assert.True(CodegenOpKinds.IsComparison(CodegenOpKind.LessEqual));

        Assert.False(CodegenOpKinds.IsComparison(CodegenOpKind.Add));
    }

    // ─── CodegenGraph construction + invariants ──────────────────────

    [Fact]
    public void Graph_AddNode_RejectsForwardInputReferences()
    {
        var g = new CodegenGraph();
        Assert.Throws<ArgumentException>(() => g.AddNode(
            new CodegenNode(CodegenOpKind.Add, new[] { 5 }, CodegenElementType.Float32, new[] { 4 })));
    }

    [Fact]
    public void Graph_AddNode_TracksInputAndOutputNodes()
    {
        var g = new CodegenGraph();
        int load = g.AddNode(new CodegenNode(
            CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 4 }, 0));
        int op = g.AddNode(new CodegenNode(
            CodegenOpKind.ReLU, new[] { load }, CodegenElementType.Float32, new[] { 4 }));
        int store = g.AddNode(new CodegenNode(
            CodegenOpKind.StoreOutput, new[] { op }, CodegenElementType.Float32, new[] { 4 }, 0));

        Assert.Equal(3, g.Count);
        Assert.Equal(new[] { load }, g.InputNodes);
        Assert.Equal(new[] { store }, g.OutputNodes);
    }

    [Fact]
    public void Graph_ConsumerTable_MatchesEdgeSet()
    {
        // Build: in → negate → store ; in → exp → store_2
        // consumers(in) = {negate, exp}, consumers(negate) = {store_1},
        // consumers(exp) = {store_2}, consumers(stores) = {}.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 2 }, 0));
        int neg = g.AddNode(new CodegenNode(CodegenOpKind.Negate, new[] { a },
            CodegenElementType.Float32, new[] { 2 }));
        int exp = g.AddNode(new CodegenNode(CodegenOpKind.Exp, new[] { a },
            CodegenElementType.Float32, new[] { 2 }));
        int s1 = g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { neg },
            CodegenElementType.Float32, new[] { 2 }, 0));
        int s2 = g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { exp },
            CodegenElementType.Float32, new[] { 2 }, 1));

        var consumers = g.BuildConsumerTable();
        Assert.Equal(new[] { neg, exp }, consumers[a].ToArray());
        Assert.Equal(new[] { s1 }, consumers[neg].ToArray());
        Assert.Equal(new[] { s2 }, consumers[exp].ToArray());
        Assert.Empty(consumers[s1]);
        Assert.Empty(consumers[s2]);
    }

    [Fact]
    public void Graph_ContentHash_IsDeterministicAndShapeSensitive()
    {
        var a = BuildChainGraph(shape: new[] { 4 });
        var b = BuildChainGraph(shape: new[] { 4 });
        var c = BuildChainGraph(shape: new[] { 5 }); // different shape → different hash

        Assert.Equal(a.ComputeContentHash(), b.ComputeContentHash());
        Assert.NotEqual(a.ComputeContentHash(), c.ComputeContentHash());
    }

    [Fact]
    public void Graph_Dump_ContainsEveryNode()
    {
        var g = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp },
            new[] { 2 });
        var dump = g.Dump();
        Assert.Contains("LoadInput", dump);
        Assert.Contains("Negate", dump);
        Assert.Contains("Exp", dump);
        Assert.Contains("StoreOutput", dump);
    }

    // ─── CodegenLowering helpers ──────────────────────────────────────

    [Fact]
    public void Lowering_UnaryPointwise_BuildsThreeNodeGraph()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 3, 4 });
        Assert.Equal(3, g.Count);
        Assert.Equal(CodegenOpKind.LoadInput, g[0].Op);
        Assert.Equal(CodegenOpKind.ReLU, g[1].Op);
        Assert.Equal(CodegenOpKind.StoreOutput, g[2].Op);

        Assert.Equal(CodegenElementType.Float32, g[1].Dtype);
        Assert.Equal(new[] { 3, 4 }, g[1].Shape);
    }

    [Fact]
    public void Lowering_UnaryPointwise_RejectsNonUnaryOps()
    {
        Assert.Throws<ArgumentException>(() =>
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Add, new[] { 4 }));
        Assert.Throws<ArgumentException>(() =>
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.MatMul, new[] { 4 }));
        Assert.Throws<ArgumentException>(() =>
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReduceSum, new[] { 4 }));
    }

    [Fact]
    public void Lowering_BinaryPointwise_BuildsFourNodeGraph()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Add, new[] { 8 });
        Assert.Equal(4, g.Count);
        Assert.Equal(CodegenOpKind.LoadInput, g[0].Op);
        Assert.Equal(CodegenOpKind.LoadInput, g[1].Op);
        Assert.Equal(CodegenOpKind.Add, g[2].Op);
        Assert.Equal(CodegenOpKind.StoreOutput, g[3].Op);
        Assert.Equal(new[] { 0, 1 }, g[2].Inputs);
    }

    [Fact]
    public void Lowering_UnaryChain_BuildsOneGraphPerOp()
    {
        var g = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sigmoid },
            new[] { 5 });
        // LoadInput + 3 ops + StoreOutput = 5 nodes.
        Assert.Equal(5, g.Count);
        Assert.Equal(CodegenOpKind.LoadInput, g[0].Op);
        Assert.Equal(CodegenOpKind.Negate, g[1].Op);
        Assert.Equal(CodegenOpKind.Exp, g[2].Op);
        Assert.Equal(CodegenOpKind.Sigmoid, g[3].Op);
        Assert.Equal(CodegenOpKind.StoreOutput, g[4].Op);
    }

    [Fact]
    public void Lowering_MapOpName_RecognisesEngineOpNames()
    {
        Assert.Equal(CodegenOpKind.Add, CodegenLowering.MapOpName("TensorAdd"));
        Assert.Equal(CodegenOpKind.Mul, CodegenLowering.MapOpName("TensorMultiply"));
        Assert.Equal(CodegenOpKind.MatMul, CodegenLowering.MapOpName("TensorMatMul"));
        Assert.Equal(CodegenOpKind.ReLU, CodegenLowering.MapOpName("ReLU"));
        Assert.Equal(CodegenOpKind.Opaque, CodegenLowering.MapOpName("SomeUnrecognisedOpName"));
    }

    [Fact]
    public void Lowering_ResolveElementType_CoversCommonScalars()
    {
        Assert.Equal(CodegenElementType.Float32, CodegenLowering.ResolveElementType<float>());
        Assert.Equal(CodegenElementType.Float64, CodegenLowering.ResolveElementType<double>());
        Assert.Equal(CodegenElementType.Int32, CodegenLowering.ResolveElementType<int>());
        Assert.Equal(CodegenElementType.Int64, CodegenLowering.ResolveElementType<long>());
        Assert.Equal(CodegenElementType.Int8, CodegenLowering.ResolveElementType<sbyte>());
        Assert.Equal(CodegenElementType.UInt8, CodegenLowering.ResolveElementType<byte>());
        Assert.Equal(CodegenElementType.Bool, CodegenLowering.ResolveElementType<bool>());

        Assert.Throws<NotSupportedException>(() =>
            CodegenLowering.ResolveElementType<decimal>());
    }

    // ─── helper ───────────────────────────────────────────────────────

    private static CodegenGraph BuildChainGraph(int[] shape)
        => CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp }, shape);
}
