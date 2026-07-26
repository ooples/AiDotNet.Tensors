// Copyright (c) AiDotNet. All rights reserved.
// Reductions must reach the front end, and must mean what they said.
//
// The first front end handled elementwise chains only, so every reduction declined --
// which is every matmul, so every linear layer. The PTX path was reachable end to end
// only for fusion chains, and the kernels carrying the actual wins stayed hand-built
// specs no model could produce.
//
// Translating a matmul is easy to do WRONGLY in a way that still emits: swap two axes in
// an index map and the kernel computes A times B-transpose, at full speed, silently. So
// these tests do not check that translation succeeds -- they check the translated spec's
// own fp64 interpretation against an independent matmul written out by hand.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGraphReductionTests
{
    private static double[] Fill(long count, int salt)
    {
        var v = new double[count];
        for (long i = 0; i < count; i++) v[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return v;
    }

    private static int Load(CodegenGraph g, int[] shape) =>
        g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));

    private static void Store(CodegenGraph g, int value, int[] shape) =>
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { value },
            CodegenElementType.Float32, shape));

    /// <summary>
    /// A matmul graph must translate AND compute a matmul. Checked against a hand-written
    /// triple loop, because an index map with two axes swapped translates and emits just
    /// as happily as a correct one.
    /// </summary>
    [Theory]
    [InlineData(CodegenOpKind.MatMul)]
    [InlineData(CodegenOpKind.MatMulTransposeA)]
    [InlineData(CodegenOpKind.MatMulTransposeB)]
    public void MatMul_TranslatesAndComputesTheContraction(CodegenOpKind op)
    {
        const int M = 6, K = 5, N = 4;
        int[] aShape = op == CodegenOpKind.MatMulTransposeA ? new[] { K, M } : new[] { M, K };
        int[] bShape = op == CodegenOpKind.MatMulTransposeB ? new[] { N, K } : new[] { K, N };
        int[] outShape = { M, N };

        var g = new CodegenGraph();
        int a = Load(g, aShape);
        int b = Load(g, bShape);
        int mm = g.AddNode(new CodegenNode(op, new[] { a, b }, CodegenElementType.Float32, outShape));
        Store(g, mm, outShape);

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "mm", out var spec, out string reason), reason);
        Assert.Equal(CodegenReduceKind.Sum, spec!.Reduce);

        double[] av = Fill((long)M * K, 1), bv = Fill((long)K * N, 2);
        double[] got = spec.Interpret(new[] { av, bv });

        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            double want = 0;
            for (int k = 0; k < K; k++)
            {
                double lhs = op == CodegenOpKind.MatMulTransposeA ? av[k * M + m] : av[m * K + k];
                double rhs = op == CodegenOpKind.MatMulTransposeB ? bv[n * K + k] : bv[k * N + n];
                want += lhs * rhs;
            }
            Assert.Equal(want, got[m * N + n], 9);
        }
    }

    /// <summary>A batched matmul must contract per batch, not across batches.</summary>
    [Fact]
    public void BatchMatMul_ContractsWithinEachBatch()
    {
        const int B = 3, M = 4, K = 5, N = 2;
        var g = new CodegenGraph();
        int a = Load(g, new[] { B, M, K });
        int b = Load(g, new[] { B, K, N });
        int mm = g.AddNode(new CodegenNode(CodegenOpKind.BatchMatMul, new[] { a, b },
            CodegenElementType.Float32, new[] { B, M, N }));
        Store(g, mm, new[] { B, M, N });

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "bmm", out var spec, out string reason), reason);

        double[] av = Fill((long)B * M * K, 3), bv = Fill((long)B * K * N, 4);
        double[] got = spec!.Interpret(new[] { av, bv });

        for (int p = 0; p < B; p++)
        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            double want = 0;
            for (int k = 0; k < K; k++)
                want += av[(p * M + m) * K + k] * bv[(p * K + k) * N + n];
            Assert.Equal(want, got[(p * M + m) * N + n], 9);
        }
    }

    /// <summary>
    /// The motivating case: a linear layer is matmul, a broadcast bias, and a ReLU. It
    /// must fuse into ONE spec, not decline because the bias is shaped [N] and the output
    /// is [M,N].
    /// </summary>
    [Fact]
    public void LinearLayer_FusesMatMulBiasAndReluIntoOneSpec()
    {
        const int M = 5, K = 6, N = 4;
        var g = new CodegenGraph();
        int a = Load(g, new[] { M, K });
        int w = Load(g, new[] { K, N });
        int bias = Load(g, new[] { N });
        int mm = g.AddNode(new CodegenNode(CodegenOpKind.MatMul, new[] { a, w },
            CodegenElementType.Float32, new[] { M, N }));
        int add = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { mm, bias },
            CodegenElementType.Float32, new[] { M, N }));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { add },
            CodegenElementType.Float32, new[] { M, N }));
        Store(g, relu, new[] { M, N });

        Assert.True(CodegenGraphToSpec.TryTranslate(
            g, "linear", out var spec, out var order, out string reason), reason);
        Assert.Equal(2, spec!.ProductInputs.Count);
        Assert.True(spec.BiasInput.HasValue);
        Assert.Equal(CodegenActivationKind.ReLU, spec.Activation);

        // The launcher binds by this order, so it has to name the right graph nodes.
        Assert.Equal(new[] { a, w, bias }, order);

        double[] av = Fill((long)M * K, 5), wv = Fill((long)K * N, 6), bv = Fill(N, 7);
        double[] got = spec.Interpret(new[] { av, wv, bv });

        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            double want = 0;
            for (int k = 0; k < K; k++) want += av[m * K + k] * wv[k * N + n];
            want = Math.Max(0, want + bv[n]);
            Assert.Equal(want, got[m * N + n], 9);
        }
    }

    /// <summary>An axis reduction must sum over the named axes and keep the others.</summary>
    [Fact]
    public void ReduceSum_SumsTheNamedAxisAndKeepsTheRest()
    {
        const int A = 4, Bx = 6, C = 3;
        var g = new CodegenGraph();
        int x = Load(g, new[] { A, Bx, C });
        int r = g.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { x },
            CodegenElementType.Float32, new[] { A, C }, new[] { 1 }));
        Store(g, r, new[] { A, C });

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "rsum", out var spec, out string reason), reason);
        Assert.Equal(CodegenReduceKind.Sum, spec!.Reduce);

        double[] xv = Fill((long)A * Bx * C, 8);
        double[] got = spec.Interpret(new[] { xv });

        for (int i = 0; i < A; i++)
        for (int k = 0; k < C; k++)
        {
            double want = 0;
            for (int j = 0; j < Bx; j++) want += xv[(i * Bx + j) * C + k];
            Assert.Equal(want, got[i * C + k], 9);
        }
    }

    /// <summary>Keepdims addresses the reduced dimension at index zero, not at the axis.</summary>
    [Fact]
    public void ReduceMax_WithKeepDims_AddressesTheReducedDimensionAtZero()
    {
        const int A = 5, Bx = 7;
        var g = new CodegenGraph();
        int x = Load(g, new[] { A, Bx });
        int r = g.AddNode(new CodegenNode(CodegenOpKind.ReduceMax, new[] { x },
            CodegenElementType.Float32, new[] { A, 1 }, new[] { 1 }));
        Store(g, r, new[] { A, 1 });

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "rmax", out var spec, out string reason), reason);
        Assert.Equal(CodegenReduceKind.Max, spec!.Reduce);

        double[] xv = Fill((long)A * Bx, 9);
        double[] got = spec.Interpret(new[] { xv });

        for (int i = 0; i < A; i++)
        {
            double want = double.NegativeInfinity;
            for (int j = 0; j < Bx; j++) want = Math.Max(want, xv[i * Bx + j]);
            Assert.Equal(want, got[i], 9);
        }
    }

    /// <summary>A negative reduction axis counts from the end, as everywhere else.</summary>
    [Fact]
    public void ReduceSum_AcceptsANegativeAxis()
    {
        var g = new CodegenGraph();
        int x = Load(g, new[] { 4, 8 });
        int r = g.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { x },
            CodegenElementType.Float32, new[] { 4 }, new[] { -1 }));
        Store(g, r, new[] { 4 });

        Assert.True(CodegenGraphToSpec.TryTranslate(g, "rneg", out var spec, out string reason), reason);

        double[] xv = Fill(32, 10);
        double[] got = spec!.Interpret(new[] { xv });
        for (int i = 0; i < 4; i++)
        {
            double want = 0;
            for (int j = 0; j < 8; j++) want += xv[i * 8 + j];
            Assert.Equal(want, got[i], 9);
        }
    }

    /// <summary>
    /// Reducing EVERY axis leaves no parallel axis, so it must decline with that reason
    /// rather than emit a one-thread kernel.
    /// </summary>
    [Fact]
    public void FullReductionToAScalar_Declines()
    {
        var g = new CodegenGraph();
        int x = Load(g, new[] { 4, 8 });
        int r = g.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { x },
            CodegenElementType.Float32, new[] { 1, 1 }, new[] { 0, 1 }));
        Store(g, r, new[] { 1, 1 });

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "rall", out _, out string reason));
        Assert.Contains("parallel axis", reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// Shapes that do not contract must be refused. This is the failure a swapped index
    /// map would otherwise turn into a fast wrong answer.
    /// </summary>
    [Fact]
    public void MatMulWithMismatchedInnerDimension_Declines()
    {
        var g = new CodegenGraph();
        int a = Load(g, new[] { 6, 5 });
        int b = Load(g, new[] { 7, 4 });     // k is 5 on the left and 7 on the right
        int mm = g.AddNode(new CodegenNode(CodegenOpKind.MatMul, new[] { a, b },
            CodegenElementType.Float32, new[] { 6, 4 }));
        Store(g, mm, new[] { 6, 4 });

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "bad", out _, out string reason));
        Assert.Contains("contract", reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// ReduceMean has no spec form -- the spec can scale by a TENSOR, not by 1/n -- so it
    /// must decline rather than silently return a sum.
    /// </summary>
    [Fact]
    public void ReduceMean_DeclinesRatherThanReturningASum()
    {
        var g = new CodegenGraph();
        int x = Load(g, new[] { 4, 8 });
        int r = g.AddNode(new CodegenNode(CodegenOpKind.ReduceMean, new[] { x },
            CodegenElementType.Float32, new[] { 4 }, new[] { 1 }));
        Store(g, r, new[] { 4 });

        Assert.False(CodegenGraphToSpec.TryTranslate(g, "mean", out _, out string reason));
        Assert.Contains("ReduceMean", reason, StringComparison.Ordinal);
    }

    /// <summary>A translated reduction must emit, not merely translate.</summary>
    [Fact]
    public void TranslatedMatMulAndReduction_Emit()
    {
        var mmGraph = new CodegenGraph();
        int a = Load(mmGraph, new[] { 64, 32 });
        int b = Load(mmGraph, new[] { 32, 16 });
        int mm = mmGraph.AddNode(new CodegenNode(CodegenOpKind.MatMul, new[] { a, b },
            CodegenElementType.Float32, new[] { 64, 16 }));
        Store(mmGraph, mm, new[] { 64, 16 });

        var reduceGraph = new CodegenGraph();
        int x = Load(reduceGraph, new[] { 128, 64 });
        int r = reduceGraph.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { x },
            CodegenElementType.Float32, new[] { 128 }, new[] { 1 }));
        Store(reduceGraph, r, new[] { 128 });

        foreach (var graph in new[] { mmGraph, reduceGraph })
        {
            var emitter = new PtxGraphEmitter();
            var result = emitter.Emit(graph, CodegenElementType.Float32);
            Assert.NotNull(result.Source);
            Assert.Contains(".visible .entry", result.Source!, StringComparison.Ordinal);
            Assert.NotNull(emitter.LastSpec);
            Assert.True(emitter.LastLaunchBlocks > 0);
        }
    }
}
