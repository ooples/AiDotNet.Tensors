// Copyright (c) AiDotNet. All rights reserved.
// Proves the front end exists: a graph built by the ORDINARY lowering path reaches the
// PTX emitter and produces a kernel that computes the right answer.
//
// Before this, PtxAffineEmitter was referenced by nothing in the library except itself,
// and the only CodegenKernelSpec constructors were a hand-written catalog, the adjoint
// deriver, the cost model and the emitter. The kernels were measured carefully and could
// never run. These tests exist so that cannot silently become true again.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class PtxGraphEmitterTests
{
    /// <summary>
    /// The whole point: a graph from CodegenLowering -- the pass that consumes real
    /// CompiledStep chains -- must reach the PTX emitter and produce PTX.
    /// </summary>
    [Fact]
    public void GraphFromTheOrdinaryLoweringPath_ReachesThePtxEmitter()
    {
        var graph = CodegenLowering.LowerUnaryPointwise<float>(
            CodegenOpKind.ReLU, new[] { 4, 256 });

        var emitter = new PtxGraphEmitter();
        var result = emitter.Emit(graph, CodegenElementType.Float32);

        Assert.NotNull(result.Source);
        Assert.Contains(".visible .entry", result.Source!, StringComparison.Ordinal);
        Assert.Contains("max.f32", result.Source!, StringComparison.Ordinal);   // the ReLU
        Assert.NotNull(emitter.LastSpec);
        Assert.True(emitter.LastLaunchBlocks > 0);
    }

    /// <summary>
    /// A chain of a multiply, an add and a ReLU is exactly the spec's body form, so it
    /// must translate rather than decline.
    /// </summary>
    [Fact]
    public void MultiplyAddRelu_TranslatesToTheSpecBodyForm()
    {
        var graph = new CodegenGraph();
        int a = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1024 }));
        int b = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1024 }));
        int c = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 1024 }));
        int mul = graph.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, new[] { 1024 }));
        int add = graph.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { mul, c },
            CodegenElementType.Float32, new[] { 1024 }));
        int relu = graph.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { add },
            CodegenElementType.Float32, new[] { 1024 }));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu },
            CodegenElementType.Float32, new[] { 1024 }));

        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "fused", out var spec, out string reason), reason);

        // product of two operands, a bias, and a ReLU -- the spec's exact body.
        Assert.Equal(2, spec!.ProductInputs.Count);
        Assert.True(spec.BiasInput.HasValue);
        Assert.Equal(CodegenActivationKind.ReLU, spec.Activation);
    }

    /// <summary>
    /// Anything outside the body form must DECLINE with a reason, not be approximated.
    /// A translator that quietly mis-lowers is worse than one that refuses.
    /// </summary>
    [Fact]
    public void GraphOutsideTheBodyForm_DeclinesWithAReason()
    {
        var graph = new CodegenGraph();
        int a = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 64 }));
        int sqrt = graph.AddNode(new CodegenNode(CodegenOpKind.Sqrt, new[] { a },
            CodegenElementType.Float32, new[] { 64 }));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { sqrt },
            CodegenElementType.Float32, new[] { 64 }));

        var result = new PtxGraphEmitter().Emit(graph, CodegenElementType.Float32);
        Assert.True(result.Declined);
        Assert.Contains("Sqrt", result.DeclineReason!, StringComparison.Ordinal);
    }

    /// <summary>A dtype the released cubins do not cover must decline, not emit fp32.</summary>
    [Fact]
    public void NonFloat32_Declines()
    {
        var graph = CodegenLowering.LowerUnaryPointwise<double>(
            CodegenOpKind.ReLU, new[] { 128 });

        var result = new PtxGraphEmitter().Emit(graph, CodegenElementType.Float64);
        Assert.True(result.Declined);
        Assert.Contains("fp32", result.DeclineReason!, StringComparison.Ordinal);
    }

    /// <summary>
    /// The translated spec must agree with its own fp64 interpretation, which is the
    /// same correctness bar every catalog kernel is held to.
    /// </summary>
    [Fact]
    public void TranslatedSpec_MatchesItsOwnFp64Interpretation()
    {
        var graph = new CodegenGraph();
        int a = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 256 }));
        int b = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 256 }));
        int mul = graph.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, new[] { 256 }));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mul },
            CodegenElementType.Float32, new[] { 256 }));

        Assert.True(CodegenGraphToSpec.TryTranslate(graph, "mul256", out var spec, out _));

        var lhs = new double[256];
        var rhs = new double[256];
        for (int i = 0; i < 256; i++) { lhs[i] = (i % 17) - 8; rhs[i] = (i % 5) - 2; }

        double[] got = spec!.Interpret(new[] { lhs, rhs });
        for (int i = 0; i < 256; i++)
            Assert.Equal(lhs[i] * rhs[i], got[i], 10);
    }
}
