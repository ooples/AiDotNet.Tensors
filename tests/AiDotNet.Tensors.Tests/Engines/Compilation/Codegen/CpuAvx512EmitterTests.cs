// Copyright (c) AiDotNet. All rights reserved.
// Phase B+ of issue #225: validate the hand-emitted AVX-512 intrinsic
// kernel emitter. Tests early-out on hardware where AVX-512F isn't
// available — xUnit doesn't ship a built-in conditional-fact, so we
// guard each [Fact] with a runtime check that turns the body into a
// no-op pass on incompatible CPUs.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// CpuAvx512Emitter correctness — the emitter is selected over the
/// JIT path on AVX-512F-capable CPUs, so a regression here would
/// silently miscompute pointwise fusions in production.
/// </summary>
public class CpuAvx512EmitterTests
{
    private readonly CpuAvx512Emitter _emitter = new();

#if NET8_0_OR_GREATER
    private static bool Avx512Available => System.Runtime.Intrinsics.X86.Avx512F.IsSupported;
#else
    private static bool Avx512Available => false;
#endif

    [Fact]
    public void Target_IsCpuAvx512()
    {
        Assert.Equal(CodegenTarget.CpuAvx512, _emitter.Target);
    }

    [Fact]
    public void Emit_DeclinesUnsupportedDtype()
    {
        if (!Avx512Available) return;
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 16 });
        var r = _emitter.Emit(g, CodegenElementType.Int32);
        Assert.True(r.Declined);
    }

    [Fact]
    public void Emit_DeclinesUnsupportedOp()
    {
        if (!Avx512Available) return;
        // Sigmoid uses Exp under the hood — no AVX-512F intrinsic, so
        // this emitter must Decline so the JIT emitter can take it.
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Sigmoid, new[] { 16 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    [Fact]
    public void Emit_DeclinesEmptyGraph()
    {
        if (!Avx512Available) return;
        var g = new CodegenGraph();
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    [Fact]
    public void Emit_DeclinesWithoutAvx512_OnIncompatibleHardware()
    {
        // The complement of the gated tests: when AVX-512F is NOT
        // available, the emitter must Decline so the JIT path
        // can take over. On AVX-512-capable hardware this test is
        // a no-op (the assertion would invert).
        if (Avx512Available) return;
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 16 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.True(r.Declined);
    }

    [Fact]
    public void Emit_Float32_AddSubMul_MatchesReference()
    {
        if (!Avx512Available) return;
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 32 }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 32 }, 1));
        int sum = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { a, b },
            CodegenElementType.Float32, new[] { 32 }));
        int diff = g.AddNode(new CodegenNode(CodegenOpKind.Sub, new[] { a, b },
            CodegenElementType.Float32, new[] { 32 }));
        int prod = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { sum, diff },
            CodegenElementType.Float32, new[] { 32 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { prod },
            CodegenElementType.Float32, new[] { 32 }, 0));

        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);

        var aData = new float[32]; var bData = new float[32]; var outData = new float[32];
        for (int i = 0; i < 32; i++) { aData[i] = 0.5f * i; bData[i] = 0.25f * i; }

        r.Kernel.Execute(new[] { aData, bData }, new[] { outData });
        for (int i = 0; i < 32; i++)
        {
            float expected = aData[i] * aData[i] - bData[i] * bData[i];
            Assert.True(System.Math.Abs(expected - outData[i]) < 1e-4f,
                $"Mismatch at {i}: expected {expected}, got {outData[i]}.");
        }
    }

    [Fact]
    public void Emit_Float32_ReLUPreservesNaN()
    {
        if (!Avx512Available) return;
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 32 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);

        var inData = new float[32];
        for (int i = 0; i < 32; i++) inData[i] = i % 2 == 0 ? -1f * i : 1f * i;
        inData[7] = float.NaN;
        var outData = new float[32];

        r.Kernel.Execute(new[] { inData }, new[] { outData });

        Assert.True(float.IsNaN(outData[7]),
            "AVX-512 ReLU dropped NaN — should preserve via compare-and-AndNot, not Max.");
        for (int i = 0; i < 32; i++)
        {
            if (i == 7) continue;
            float expected = inData[i] < 0 ? 0 : inData[i];
            Assert.Equal(expected, outData[i]);
        }
    }

    [Fact]
    public void Emit_Float32_TailHandledCorrectly()
    {
        if (!Avx512Available) return;
        // Element count 19 = 16 SIMD + 3 scalar tail. The tail must
        // produce identical results to the SIMD body.
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Negate, new[] { 19 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);

        var inData = new float[19];
        for (int i = 0; i < 19; i++) inData[i] = i + 1f;
        var outData = new float[19];
        r.Kernel.Execute(new[] { inData }, new[] { outData });
        for (int i = 0; i < 19; i++)
            Assert.Equal(-inData[i], outData[i]);
    }

    [Fact]
    public void Emit_Float64_8WideKernel_Works()
    {
        if (!Avx512Available) return;
        // Float64 packs 8 lanes per Vector512; element count 24 = 3 SIMD blocks.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float64, new[] { 24 }, 0));
        int sq = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, a },
            CodegenElementType.Float64, new[] { 24 }));
        int rt = g.AddNode(new CodegenNode(CodegenOpKind.Sqrt, new[] { sq },
            CodegenElementType.Float64, new[] { 24 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { rt },
            CodegenElementType.Float64, new[] { 24 }, 0));

        var r = _emitter.Emit(g, CodegenElementType.Float64);
        Assert.False(r.Declined);

        var inData = new double[24];
        for (int i = 0; i < 24; i++) inData[i] = i * 0.1 + 0.05;
        var outData = new double[24];
        r.Kernel.Execute(new[] { inData }, new[] { outData });
        for (int i = 0; i < 24; i++)
            Assert.True(System.Math.Abs(inData[i] - outData[i]) < 1e-9,
                $"sqrt(x²) mismatch at {i}: in {inData[i]}, out {outData[i]}.");
    }

    [Fact]
    public void Emit_Source_DocumentsIntrinsicShape()
    {
        if (!Avx512Available) return;
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 16 });
        var r = _emitter.Emit(g, CodegenElementType.Float32);
        Assert.False(r.Declined);
        Assert.Contains("AVX-512F", r.Source);
        Assert.Contains("16-wide", r.Source);
    }
}
