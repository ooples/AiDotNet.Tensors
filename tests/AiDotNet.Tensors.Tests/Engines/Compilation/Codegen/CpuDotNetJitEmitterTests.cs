// Copyright (c) AiDotNet. All rights reserved.
// Phase B of issue #225: verify that CpuDotNetJitEmitter produces
// compiled kernels whose numerical output matches the equivalent
// engine-composed op sequence to within float tolerance.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// Phase B correctness guards — if the emitter produces wrong code
/// the downstream GPU emitters (which share the IR contract) would
/// inherit the same bug.
/// </summary>
public class CpuDotNetJitEmitterTests
{
    private readonly CpuDotNetJitEmitter _emitter = new();

    // ─── Decline paths ────────────────────────────────────────────────

    [Fact]
    public void Emit_DeclinesUnsupportedDtype()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var r = _emitter.Emit(g, CodegenElementType.Int32);
        Assert.True(r.Declined);
        Assert.Contains("Float32", r.DeclineReason);
    }

    [Fact]
    public void Emit_DeclinesReductionOps()
    {
        var g = new CodegenGraph();
        int in0 = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 0));
        int red = g.AddNode(new CodegenNode(CodegenOpKind.ReduceSum, new[] { in0 },
            CodegenElementType.Float32, new[] { 4 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { red },
            CodegenElementType.Float32, new[] { 4 }, 0));

        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    [Fact]
    public void Emit_DeclinesMatmul()
    {
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 4 }, 1));
        int m = g.AddNode(new CodegenNode(CodegenOpKind.MatMul, new[] { a, b },
            CodegenElementType.Float32, new[] { 4 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { m },
            CodegenElementType.Float32, new[] { 4 }, 0));

        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    // ─── Successful emission ──────────────────────────────────────────

    [Fact]
    public void Emit_UnaryNegate_MatchesReference()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Negate, new[] { 8 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.NotNull(r.Kernel);
        Assert.NotNull(r.Source);

        var input = new float[] { 1, -2, 3, -4, 5, -6, 7, -8 };
        var output = new float[8];
        ExecuteUnary<float>(r.Kernel!, input, output);

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(-input[i], output[i]);
    }

    [Fact]
    public void Emit_UnaryChain_SigmoidExpNegate_MatchesPointwiseEvaluation()
    {
        var g = CodegenLowering.LowerUnaryChain<float>(
            new[] { CodegenOpKind.Negate, CodegenOpKind.Exp, CodegenOpKind.Sigmoid },
            new[] { 5 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);

        var input = new float[] { 0f, 0.5f, -0.3f, 1.2f, -2.0f };
        var output = new float[5];
        ExecuteUnary<float>(r.Kernel!, input, output);

        for (int i = 0; i < input.Length; i++)
        {
            float expected = Sigmoid(MathF.Exp(-input[i]));
            Assert.True(MathF.Abs(output[i] - expected) < 1e-5f,
                $"Element {i}: expected {expected}, got {output[i]}.");
        }
    }

    [Fact]
    public void Emit_BinaryAdd_MatchesSum()
    {
        var g = CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Add, new[] { 16 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);

        var a = new float[16]; for (int i = 0; i < 16; i++) a[i] = i * 0.25f;
        var b = new float[16]; for (int i = 0; i < 16; i++) b[i] = i * -0.125f;
        var output = new float[16];
        ExecuteBinary<float>(r.Kernel!, a, b, output);

        for (int i = 0; i < 16; i++)
            Assert.Equal(a[i] + b[i], output[i], precision: 5);
    }

    [Fact]
    public void Emit_DoublePrecision_WorksEndToEnd()
    {
        var g = CodegenLowering.LowerUnaryChain<double>(
            new[] { CodegenOpKind.Sqrt, CodegenOpKind.Log },
            new[] { 4 });
        var r = _emitter.Emit(g, CodegenElementType.Float64);
        Assert.False(r.Declined);

        var input = new double[] { 1, 4, 9, 16 };
        var output = new double[4];
        ExecuteUnary<double>(r.Kernel!, input, output);

        // log(sqrt(x)) = 0.5 * log(x). At x = 1, 4, 9, 16:
        //   0.5·log(1) = 0
        //   0.5·log(4) ≈ 0.6931
        //   0.5·log(9) ≈ 1.0986
        //   0.5·log(16) ≈ 1.3863
        Assert.Equal(0.0, output[0], precision: 6);
        Assert.Equal(0.5 * Math.Log(4), output[1], precision: 6);
        Assert.Equal(0.5 * Math.Log(9), output[2], precision: 6);
        Assert.Equal(0.5 * Math.Log(16), output[3], precision: 6);
    }

    [Fact]
    public void Emit_SourceDumpContainsKernelSignature()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 2 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.NotNull(r.Source);
        Assert.Contains("void Kernel(", r.Source);
        Assert.Contains("ReadOnlyMemory<float>", r.Source);
        Assert.Contains("for (int i = 0; i < count; i++)", r.Source);
        Assert.Contains("MathF.Max", r.Source); // ReLU → Max(x, 0)
    }

    [Fact]
    public void Emit_Kernel_ExposesDtypeAndTarget()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.Equal(CodegenElementType.Float32, r.Kernel!.Dtype);
        Assert.Equal(CodegenTarget.CpuDotNetJit, r.Kernel.Target);
        Assert.Same(g, r.Kernel.Graph);
    }

    [Fact]
    public void Execute_WrongDtype_Throws()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);

        var inputF = new float[] { 1, 2, 3, 4 };
        var outputF = new float[4];
        // Kernel is float32; passing double buffers should throw.
        Assert.Throws<ArgumentException>(() =>
            ExecuteUnary<double>(r.Kernel!, new double[4], new double[4]));
    }

    // ─── helpers ──────────────────────────────────────────────────────

    private static void ExecuteUnary<T>(CodegenKernel kernel, T[] input, T[] output)
        where T : unmanaged
    {
        kernel.Execute<T>(new[] { input }, new[] { output });
    }

    private static void ExecuteBinary<T>(CodegenKernel kernel, T[] a, T[] b, T[] output)
        where T : unmanaged
    {
        kernel.Execute<T>(new[] { a, b }, new[] { output });
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
