// Copyright (c) AiDotNet. All rights reserved.
// Backend dispatch wiring tests for issue #225 Phase C.5.

#nullable disable

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// Verifies that the central CodegenDispatcher selects the right CPU
/// emitter for a given graph and that its output kernels execute
/// correctly end-to-end.
/// </summary>
public class CodegenDispatcherTests
{
    [Fact]
    public void CpuEmitters_OrderIsAvx512BeforeJit()
    {
        // Priority order matters — AVX-512 must be tried first so
        // graphs that fit its supported subset get the SIMD-guaranteed
        // path rather than RyuJIT's heuristic vectorisation.
        var emitters = CodegenDispatcher.CpuEmitters;
        Assert.Equal(2, emitters.Count);
        Assert.Equal(CodegenTarget.CpuAvx512, emitters[0].Target);
        Assert.Equal(CodegenTarget.CpuDotNetJit, emitters[1].Target);
    }

    [Fact]
    public void TryEmitCpu_PointwiseAdd_ProducesExecutableKernel()
    {
        // out[i] = a[i] + b[i] — every emitter handles this.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 32 }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 32 }, 1));
        int sum = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { a, b },
            CodegenElementType.Float32, new[] { 32 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { sum },
            CodegenElementType.Float32, new[] { 32 }, 0));

        var kernel = CodegenDispatcher.TryEmitCpu(g, CodegenElementType.Float32, out var declines);
        Assert.NotNull(kernel);

        var aData = new float[32]; var bData = new float[32]; var outData = new float[32];
        for (int i = 0; i < 32; i++) { aData[i] = i; bData[i] = -i + 1f; }
        kernel.Execute(new[] { aData, bData }, new[] { outData });
        for (int i = 0; i < 32; i++) Assert.Equal(1f, outData[i]);
    }

    [Fact]
    public void TryEmitCpu_TranscendentalGraph_FallsBackToJit()
    {
        // Sigmoid uses Exp internally — AVX-512 emitter declines it.
        // The dispatcher should fall through to the JIT emitter and
        // still produce a working kernel.
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.Sigmoid, new[] { 32 });
        var kernel = CodegenDispatcher.TryEmitCpu(g, CodegenElementType.Float32, out var declines);
        Assert.NotNull(kernel);

        // The decline list shows AVX-512 was tried first and refused.
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
        {
            Assert.Contains(declines, d => d.Target == CodegenTarget.CpuAvx512);
        }
#endif
    }

    [Fact]
    public void TryEmitCpu_UnsupportedDtype_AllEmittersDecline()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 16 });
        var kernel = CodegenDispatcher.TryEmitCpu(g, CodegenElementType.Int32, out var declines);
        Assert.Null(kernel);
        Assert.Equal(2, declines.Count);  // Both emitters declined.
        Assert.All(declines, d => Assert.NotEmpty(d.Reason));
    }

    [Fact]
    public void TryEmitCpu_NullGraph_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            CodegenDispatcher.TryEmitCpu(null!, CodegenElementType.Float32));
    }

    [Fact]
    public void TryEmitCpu_RealisticPointwise_MatchesScalarReference()
    {
        // out[i] = ReLU(a[i] * b[i] + c[i]) — the kind of fusion a
        // production pass would emit. Every emitter handles every op
        // here so the dispatcher succeeds at the first level.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 64 }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 64 }, 1));
        int c = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { 64 }, 2));
        int prod = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, new[] { 64 }));
        int sum = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { prod, c },
            CodegenElementType.Float32, new[] { 64 }));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { sum },
            CodegenElementType.Float32, new[] { 64 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu },
            CodegenElementType.Float32, new[] { 64 }, 0));

        var kernel = CodegenDispatcher.TryEmitCpu(g, CodegenElementType.Float32);
        Assert.NotNull(kernel);

        var aData = new float[64];
        var bData = new float[64];
        var cData = new float[64];
        var outData = new float[64];
        for (int i = 0; i < 64; i++)
        {
            aData[i] = (i % 8) - 4f;     // -4 .. 3
            bData[i] = 0.5f;              // small positive scale
            cData[i] = -1f;               // bias makes some lanes negative pre-ReLU
        }

        kernel.Execute(new[] { aData, bData, cData }, new[] { outData });

        for (int i = 0; i < 64; i++)
        {
            float expected = aData[i] * bData[i] + cData[i];
            expected = expected < 0 ? 0 : expected;
            Assert.True(System.Math.Abs(expected - outData[i]) < 1e-5f,
                $"Mismatch at {i}: expected {expected}, got {outData[i]}.");
        }
    }
}
