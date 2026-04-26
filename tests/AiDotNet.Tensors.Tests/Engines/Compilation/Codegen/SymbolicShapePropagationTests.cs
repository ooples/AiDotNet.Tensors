// Copyright (c) AiDotNet. All rights reserved.
// Constraint propagation tests for issue #225 dynamic-shapes section.

#nullable disable

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// Symbolic shapes must survive fusion-pass-friendly traversals so
/// downstream passes (recompile-budget, autotune-cache) reason
/// correctly about which axes can vary at runtime.
/// </summary>
public class SymbolicShapePropagationTests
{
    [Fact]
    public void Propagate_PointwiseChain_PreservesSymbolicMask()
    {
        // input → ReLU → Negate → output. Symbolic at axis 0
        // (batch). Both ops must keep that mask.
        var g = new CodegenGraph();
        int x = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 8 }, 0));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { x },
            CodegenElementType.Float32, new[] { 4, 8 }));
        int neg = g.AddNode(new CodegenNode(CodegenOpKind.Negate, new[] { relu },
            CodegenElementType.Float32, new[] { 4, 8 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { neg },
            CodegenElementType.Float32, new[] { 4, 8 }, 0));

        var inShape = SymbolicShape.BatchDynamic(new[] { 4, 8 });
        var perNode = SymbolicShapePropagation.Propagate(g, new[] { inShape });

        Assert.NotNull(perNode[x]);
        Assert.NotNull(perNode[relu]);
        Assert.NotNull(perNode[neg]);
        // Axis 0 must remain symbolic at every node along the chain.
        Assert.Contains(0, perNode[relu]!.SymbolicDimensions);
        Assert.Contains(0, perNode[neg]!.SymbolicDimensions);
    }

    [Fact]
    public void Propagate_BinaryUnion_TakesUnionOfSymbolicMasks()
    {
        // a = [batch_dynamic, 8], b = [4, seq_dynamic]
        // Add(a, b) — output should be symbolic at axes {0, 1} (union).
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 8 }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 8 }, 1));
        int sum = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { a, b },
            CodegenElementType.Float32, new[] { 4, 8 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { sum },
            CodegenElementType.Float32, new[] { 4, 8 }, 0));

        var aShape = new SymbolicShape(new[] { 4, 8 }, 0);  // axis 0 dynamic
        var bShape = new SymbolicShape(new[] { 4, 8 }, 1);  // axis 1 dynamic

        var perNode = SymbolicShapePropagation.Propagate(g, new[] { aShape, bShape });
        var sumShape = perNode[sum];
        Assert.NotNull(sumShape);
        Assert.Contains(0, sumShape!.SymbolicDimensions);
        Assert.Contains(1, sumShape.SymbolicDimensions);
    }

    [Fact]
    public void Propagate_Reshape_ReturnsNullToForceReAnnotation()
    {
        // Reshape changes rank/shape arbitrarily — propagator can't
        // infer the new symbolic mask without knowing the mapping.
        // Returns null so the caller is forced to re-annotate.
        var g = new CodegenGraph();
        int x = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4, 8 }, 0));
        int rs = g.AddNode(new CodegenNode(CodegenOpKind.Reshape, new[] { x },
            CodegenElementType.Float32, new[] { 32 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { rs },
            CodegenElementType.Float32, new[] { 32 }, 0));

        var perNode = SymbolicShapePropagation.Propagate(g,
            new[] { SymbolicShape.BatchDynamic(new[] { 4, 8 }) });
        Assert.NotNull(perNode[x]);
        Assert.Null(perNode[rs]);
    }

    [Fact]
    public void Propagate_NullArgs_Throws()
    {
        Assert.Throws<ArgumentNullException>(
            () => SymbolicShapePropagation.Propagate(null, new SymbolicShape[0]));
        var g = new CodegenGraph();
        Assert.Throws<ArgumentNullException>(
            () => SymbolicShapePropagation.Propagate(g, null));
    }

    [Fact]
    public void Propagate_InputCountMismatch_Throws()
    {
        var g = new CodegenGraph();
        g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 0));
        // Pass two shapes for a single-input graph.
        Assert.Throws<ArgumentException>(
            () => SymbolicShapePropagation.Propagate(g,
                new[] { SymbolicShape.BatchDynamic(new[] { 4 }), SymbolicShape.BatchDynamic(new[] { 4 }) }));
    }

    [Fact]
    public void SymbolicShape_BatchSweep_ProducesSameKey()
    {
        // Across the batch sweep 8/16/32/64/128 with axis 0 dynamic,
        // ComputeKey must be invariant — that's the property the
        // recompile-budget gate relies on to keep the cache working.
        var batches = new[] { 8, 16, 32, 64, 128 };
        long? sharedKey = null;
        foreach (var b in batches)
        {
            var s = SymbolicShape.BatchDynamic(new[] { b, 16 });
            long k = s.ComputeKey();
            if (sharedKey is null) sharedKey = k;
            else Assert.Equal(sharedKey.Value, k);
        }
    }
}
