// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Compile-vs-eager test for issue #225: a fused pointwise sequence emitted by CodegenDispatcher (AVX-512 or the
// LambdaExpression JIT) against the same five ops run one at a time on the CPU engine. The fused path makes one pass
// over the data; the eager path makes one per op.
//
// What is asserted is structural and so the same on every host: the five ops compile into ONE CPU kernel that is
// vectorized and split across the CPU pool, and its output is bit-identical to eager's. Wall-clock time is printed
// as diagnostics only: the ratio depends on the core count and on whatever else the host runs (the eager arm's
// multithreaded passes suffer most under load), so a timing gate measured the host, not the kernel.

#nullable disable

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class CompileSpeedupTests
{
    // The eager arm is the CPU engine itself, never the process's current engine: with a GPU visible that is the GPU
    // engine, and the comparison was then a CPU kernel against host-to-device round trips.
    private readonly CpuEngine _engine = new();
    private readonly ITestOutputHelper _out;

    public CompileSpeedupTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Compiled_FivePointwiseOps_RunAsOneVectorizedSplitKernel_BitIdenticalToEager()
    {
        // out = ((((a + b) * c) - d) * c) + a over 1M elements: memory-bound.
        const int N = 1_000_000;

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

        // One kernel computes all five ops: one invocation where eager issues five.
        Assert.True(kernel.Target is CodegenTarget.CpuAvx512 or CodegenTarget.CpuDotNetJit, $"target {kernel.Target}");
        int fusedOps = 0;
        foreach (var node in kernel.Graph.Nodes)
            if (node.Op is not (CodegenOpKind.LoadInput or CodegenOpKind.StoreOutput)) fusedOps++;
        Assert.Equal(5, fusedOps);
        Assert.Equal(4, kernel.InputCount);
        Assert.Equal(1, kernel.OutputCount);
        // The managed kernel runs on Vector<T> lanes where the hardware has them, split into grain-sized chunks.
        if (kernel is CompiledCpuKernel<float> jit)
        {
            Assert.Equal(System.Numerics.Vector.IsHardwareAccelerated, jit.IsVectorized);
            Assert.True(jit.ChunkCount > 1, $"1M elements ran as {jit.ChunkCount} chunk(s)");
        }

        var aT = new Tensor<float>(aData, new[] { N });
        var bT = new Tensor<float>(bData, new[] { N });
        var cT = new Tensor<float>(cData, new[] { N });
        var dT = new Tensor<float>(dData, new[] { N });
        var outFused = new float[N];

        kernel.Execute(new[] { aData, bData, cData, dData }, new[] { outFused });
        var eager = RunEager(aT, bT, cT, dT).AsSpan();
        // Bit-identical: each element goes through the same five IEEE operations in the same order on both paths.
        for (int i = 0; i < N; i++)
            Assert.True(MathCompat.SingleToInt32Bits(outFused[i]) == MathCompat.SingleToInt32Bits(eager[i]),
                $"Fused vs eager differ at {i}: fused={outFused[i]:R} eager={eager[i]:R}.");

        // Diagnostics only (see the file header).
        const int iters = 20;
        var sw = Stopwatch.StartNew();
        for (int it = 0; it < iters; it++)
            kernel.Execute(new[] { aData, bData, cData, dData }, new[] { outFused });
        double fusedMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        for (int it = 0; it < iters; it++)
            RunEager(aT, bT, cT, dT);
        double eagerMs = sw.Elapsed.TotalMilliseconds;
        _out.WriteLine($"fused {fusedMs / iters:F3} ms/iter, eager {eagerMs / iters:F3} ms/iter "
            + $"(ratio {fusedMs / eagerMs:F2}), {CpuParallelSettings.MaxDegreeOfParallelism} threads, target {kernel.Target}");
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
