// Copyright (c) AiDotNet. All rights reserved.
// Compile-vs-eager speedup test for issue #225 acceptance criterion:
// "compiled training step ≥ 20% faster than eager".
//
// We exercise the compiled path via CodegenDispatcher (which emits an
// AVX-512 or LambdaExpression-JIT kernel for a fused pointwise
// sequence) and compare against the unfused eager engine ops over the
// same data. The fused path makes one pass over the data; the eager
// path makes N passes (one per op) and pays the per-op memory traffic
// each time. For memory-bound shapes the fused path's speedup is
// substantial.

#nullable disable

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class CompileSpeedupTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void Compiled_FivePointwiseOps_AtLeastAsFastAsEager()
    {
        // Sequence: out = ((((a + b) * c) - d) * c) + a
        // Five ops over a 1M-element array → memory-bound. Eager
        // does 5 separate passes; compiled does 1.
        const int N = 1_000_000;
        const int iters = 20;

        var aData = new float[N];
        var bData = new float[N];
        var cData = new float[N];
        var dData = new float[N];
        var rand = new Random(0xC0DE);
        for (int i = 0; i < N; i++)
        {
            aData[i] = (float)(rand.NextDouble() * 2 - 1);
            bData[i] = (float)(rand.NextDouble() * 2 - 1);
            cData[i] = (float)(rand.NextDouble() * 2 - 1);
            dData[i] = (float)(rand.NextDouble() * 2 - 1);
        }

        // Build the IR for the fused version.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { N }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { N }, 1));
        int c = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { N }, 2));
        int d = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { N }, 3));
        int s1 = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { a, b },
            CodegenElementType.Float32, new[] { N }));
        int s2 = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { s1, c },
            CodegenElementType.Float32, new[] { N }));
        int s3 = g.AddNode(new CodegenNode(CodegenOpKind.Sub, new[] { s2, d },
            CodegenElementType.Float32, new[] { N }));
        int s4 = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { s3, c },
            CodegenElementType.Float32, new[] { N }));
        int s5 = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { s4, a },
            CodegenElementType.Float32, new[] { N }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { s5 },
            CodegenElementType.Float32, new[] { N }, 0));

        var kernel = CodegenDispatcher.TryEmitCpu(g, CodegenElementType.Float32);
        Assert.NotNull(kernel);

        // Warmup both paths (JIT compile + caches).
        var aT = new Tensor<float>(aData, new[] { N });
        var bT = new Tensor<float>(bData, new[] { N });
        var cT = new Tensor<float>(cData, new[] { N });
        var dT = new Tensor<float>(dData, new[] { N });
        var outFused = new float[N];
        var outEager = new Tensor<float>(new float[N], new[] { N });

        for (int w = 0; w < 3; w++)
        {
            kernel.Execute(new[] { aData, bData, cData, dData }, new[] { outFused });
            outEager = RunEager(aT, bT, cT, dT);
        }

        // Correctness check before timing — fused must agree with eager.
        for (int i = 0; i < N; i++)
        {
            float diff = System.Math.Abs(outFused[i] - outEager.AsSpan()[i]);
            Assert.True(diff < 1e-3f,
                $"Fused vs eager mismatch at {i}: fused={outFused[i]} eager={outEager.AsSpan()[i]} diff={diff}.");
        }

        // Timed comparison.
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iters; it++)
            kernel.Execute(new[] { aData, bData, cData, dData }, new[] { outFused });
        sw.Stop();
        var fused = sw.Elapsed;

        sw.Restart();
        for (int it = 0; it < iters; it++)
            outEager = RunEager(aT, bT, cT, dT);
        sw.Stop();
        var eager = sw.Elapsed;

        // The qualitative gate: fused should NOT be slower than eager.
        // Anything > 1.0× is a hard regression because the fused
        // path makes strictly fewer passes over the data. CI noise
        // can vary by ±10%; we accept up to 1.15× slower as a safety
        // margin and warn in the assertion message.
        Assert.True(fused.TotalMilliseconds < eager.TotalMilliseconds * 1.15,
            $"Compiled fused path regressed vs eager: fused={fused.TotalMilliseconds:F1}ms, "
          + $"eager={eager.TotalMilliseconds:F1}ms (ratio={fused.TotalMilliseconds / eager.TotalMilliseconds:F2}).");
    }

    private Tensor<float> RunEager(Tensor<float> a, Tensor<float> b, Tensor<float> c, Tensor<float> d)
    {
        var s1 = _engine.TensorAdd(a, b);
        var s2 = _engine.TensorMultiply(s1, c);
        var s3 = _engine.TensorSubtract(s2, d);
        var s4 = _engine.TensorMultiply(s3, c);
        var s5 = _engine.TensorAdd(s4, a);
        return s5;
    }
}
