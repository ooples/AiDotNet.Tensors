using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

/// <summary>
/// The managed CPU kernel runs its graph on Vector&lt;T&gt; lanes and splits large inputs into grain-sized chunks across
/// the CPU pool. Each element is computed alone, so neither the lanes nor the split may change a single bit against
/// one serial scalar pass, whatever the size (vector and chunk tails included) or the element type.
/// </summary>
public class CpuJitKernelSplitTests
{
    private static CodegenGraph Graph(CodegenElementType dtype, int n, params CodegenOpKind[] unaryChain)
    {
        // out = chain((a - b) * a / (b + b)), the unary chain applied to that in order.
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, new[] { n }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, new[] { n }, 1));
        int s = g.AddNode(new CodegenNode(CodegenOpKind.Sub, new[] { a, b }, dtype, new[] { n }));
        int m = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { s, a }, dtype, new[] { n }));
        int p = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { b, b }, dtype, new[] { n }));
        int v = g.AddNode(new CodegenNode(CodegenOpKind.Div, new[] { m, p }, dtype, new[] { n }));
        foreach (var op in unaryChain)
            v = g.AddNode(new CodegenNode(op, new[] { v }, dtype, new[] { n }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { v }, dtype, new[] { n }, 0));
        return g;
    }

    private static T[] Data<T>(int n, int seed, Func<double, T> make)
    {
        var rng = new Random(seed);
        var data = new T[n];
        for (int i = 0; i < n; i++) data[i] = make(rng.NextDouble() * 4 - 2);
        return data;
    }

    // Beyond one chunk the runner dispatches to the pool whenever this process allows more than one thread: the same
    // decision it makes, so a split theory cannot pass on a silent serial fallback.
    private static void AssertSplitsAcrossThePool(int chunkCount, int n, int buffers)
    {
        if (chunkCount < 2 || CpuParallelSettings.MaxDegreeOfParallelism < 2 || CpuParallelSettings.IsInParallelRegion)
            return;
        Assert.True(CpuParallelSettings.TryPlanIndependentDispatch(chunkCount, (long)n * buffers, int.MaxValue,
            deterministicSafe: true, out int tasks, out _));
        Assert.True(tasks > 1, $"n={n}: {chunkCount} chunks planned as {tasks} task(s)");
    }

    public static TheoryData<int> Sizes => new()
    {
        1, 7, 1000,
        CpuParallelSettings.ElementwiseGrain - 3,
        2 * CpuParallelSettings.ElementwiseGrain + 5,
        1_000_003,
    };

    [Theory]
    [MemberData(nameof(Sizes))]
    public void Float_SplitVectorRun_IsBitIdenticalToOneSerialPass(int n)
    {
        var g = Graph(CodegenElementType.Float32, n, CodegenOpKind.Negate, CodegenOpKind.Abs, CodegenOpKind.Sqrt,
            CodegenOpKind.Negate, CodegenOpKind.ReLU);
        var kernel = Assert.IsType<CompiledCpuKernel<float>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel);
        Assert.Equal(System.Numerics.Vector.IsHardwareAccelerated, kernel.IsVectorized);
        var a = Data(n, 1, x => (float)x);
        var b = Data(n, 2, x => (float)x);
        // Signed zeros and NaNs must pass the ReLU exactly as the scalar body does.
        if (n > 4) { a[0] = -0f; b[1] = float.NaN; a[3] = float.NaN; }
        AssertSplitsAcrossThePool(kernel.ChunkCount, n, 3);
        var split = new float[n];
        var serial = new float[n];
        kernel.Execute(new[] { a, b }, new[] { split });
        kernel.ExecuteSerial(new[] { a, b }, new[] { serial });
        for (int i = 0; i < n; i++)
            Assert.True(MathCompat.SingleToInt32Bits(split[i]) == MathCompat.SingleToInt32Bits(serial[i]),
                $"n={n}, element {i}: split {split[i]:R} serial {serial[i]:R}");
        // And the scalar reference itself is the plain expression, computed here in C#.
        for (int i = 0; i < n; i++)
        {
            float v = (a[i] - b[i]) * a[i] / (b[i] + b[i]);
            v = -v; v = MathF.Abs(v); v = MathF.Sqrt(v); v = -v; v = v < 0f ? 0f : v;
            Assert.True(MathCompat.SingleToInt32Bits(v) == MathCompat.SingleToInt32Bits(serial[i]),
                $"n={n}, element {i}: expected {v:R} kernel {serial[i]:R}");
        }
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void Double_SplitVectorRun_IsBitIdenticalToOneSerialPass(int n)
    {
        var g = Graph(CodegenElementType.Float64, n, CodegenOpKind.Abs, CodegenOpKind.Sqrt);
        var kernel = Assert.IsType<CompiledCpuKernel<double>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float64).Kernel);
        var a = Data(n, 3, x => x);
        var b = Data(n, 4, x => x);
        var split = new double[n];
        var serial = new double[n];
        kernel.Execute(new[] { a, b }, new[] { split });
        kernel.ExecuteSerial(new[] { a, b }, new[] { serial });
        for (int i = 0; i < n; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(split[i]) == BitConverter.DoubleToInt64Bits(serial[i]),
                $"n={n}, element {i}: split {split[i]:R} serial {serial[i]:R}");
    }

    [Fact]
    public void AGraphWithoutALaneExactVectorForm_StaysScalar_AndSplitsIdentically()
    {
        // Exp has no Vector<T> form that matches MathF.Exp bit for bit, so the kernel keeps the scalar body.
        int n = 3 * CpuParallelSettings.ElementwiseGrain + 11;
        var g = Graph(CodegenElementType.Float32, n, CodegenOpKind.Exp);
        var kernel = Assert.IsType<CompiledCpuKernel<float>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel);
        Assert.False(kernel.IsVectorized);
        Assert.True(kernel.ChunkCount > 1);
        var a = Data(n, 5, x => (float)x);
        var b = Data(n, 6, x => (float)x);
        var split = new float[n];
        var serial = new float[n];
        kernel.Execute(new[] { a, b }, new[] { split });
        kernel.ExecuteSerial(new[] { a, b }, new[] { serial });
        for (int i = 0; i < n; i++)
            Assert.True(MathCompat.SingleToInt32Bits(split[i]) == MathCompat.SingleToInt32Bits(serial[i]), $"element {i}");
    }

    [Fact]
    public void AnInputBelowOneGrain_RunsAsOneChunk()
    {
        var g = Graph(CodegenElementType.Float32, CpuParallelSettings.ElementwiseGrain / 2);
        var kernel = Assert.IsType<CompiledCpuKernel<float>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel);
        Assert.Equal(1, kernel.ChunkCount);
    }

    [Fact]
    public void ConcurrentCallsOfOneKernel_EachGetTheirOwnResult()
    {
        // The runner caches one call slot; a second concurrent caller must take a slot of its own, never share it.
        int n = 4 * CpuParallelSettings.ElementwiseGrain + 9;
        var g = Graph(CodegenElementType.Float32, n);
        var kernel = Assert.IsType<CompiledCpuKernel<float>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel);
        const int callers = 6;
        var inputs = Enumerable.Range(0, callers)
            .Select(c => new[] { Data(n, 10 + c, x => (float)x), Data(n, 20 + c, x => (float)x) }).ToArray();
        var expected = inputs.Select(io =>
        {
            var o = new float[n];
            kernel.ExecuteSerial(io, new[] { o });
            return o;
        }).ToArray();
        var outputs = Enumerable.Range(0, callers).Select(_ => new float[n]).ToArray();
        Parallel.For(0, callers * 8, new ParallelOptions { MaxDegreeOfParallelism = callers }, k =>
        {
            int c = k % callers;
            kernel.Execute(inputs[c], new[] { outputs[c] });
        });
        for (int c = 0; c < callers; c++)
            for (int i = 0; i < n; i++)
                Assert.True(MathCompat.SingleToInt32Bits(outputs[c][i]) == MathCompat.SingleToInt32Bits(expected[c][i]),
                    $"caller {c}, element {i}");
    }

    [Fact]
    public void AKernelCalledInsideAParallelRegion_RunsSeriallyAndIdentically()
    {
        // Nested inside another parallel loop the runner must not dispatch again (the pool would serialize or
        // starve); it runs the whole range on the calling worker, with the same bits.
        int n = 3 * CpuParallelSettings.ElementwiseGrain + 1;
        var g = Graph(CodegenElementType.Float32, n);
        var kernel = Assert.IsType<CompiledCpuKernel<float>>(new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel);
        var a = Data(n, 7, x => (float)x);
        var b = Data(n, 8, x => (float)x);
        var serial = new float[n];
        kernel.ExecuteSerial(new[] { a, b }, new[] { serial });
        var outputs = Enumerable.Range(0, 4).Select(_ => new float[n]).ToArray();
        bool[] nested = new bool[outputs.Length];
        CpuParallelSettings.ParallelForOrSerial(0, outputs.Length, long.MaxValue / 2, k =>
        {
            nested[k] = CpuParallelSettings.IsInParallelRegion;
            kernel.Execute(new[] { a, b }, new[] { outputs[k] });
        }, deterministicSafe: true);
        for (int k = 0; k < outputs.Length; k++)
            for (int i = 0; i < n; i++)
                Assert.True(MathCompat.SingleToInt32Bits(outputs[k][i]) == MathCompat.SingleToInt32Bits(serial[i]),
                    $"call {k} (nested {nested[k]}), element {i}");
    }

#if NET8_0_OR_GREATER
    // The AVX-512 emitter shares the chunked runner. Runs wherever AVX-512F is available (many CI runners), and skips
    // on hosts without it.
    [SkippableTheory]
    [MemberData(nameof(Sizes))]
    public void Avx512_SplitRun_IsBitIdenticalToOneSerialPass(int n)
    {
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx512F.IsSupported, "AVX-512F unavailable on this host.");
        // out = relu(-sqrt(((a - b) * a + b) * b)): only ops the AVX-512 emitter takes.
        var dtype = CodegenElementType.Float32;
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, new[] { n }, 0));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, new[] { n }, 1));
        int v = g.AddNode(new CodegenNode(CodegenOpKind.Sub, new[] { a, b }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { v, a }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { v, b }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { v, b }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.Sqrt, new[] { v }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.Negate, new[] { v }, dtype, new[] { n }));
        v = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { v }, dtype, new[] { n }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { v }, dtype, new[] { n }, 0));
        var kernel = Assert.IsType<CompiledAvx512Kernel<float>>(new CpuAvx512Emitter().Emit(g, dtype).Kernel);
        var x = Data(n, 11, d => (float)d);
        var y = Data(n, 12, d => (float)d);
        if (n > 4) { x[0] = -0f; y[1] = float.NaN; }
        AssertSplitsAcrossThePool(kernel.ChunkCount, n, 3);
        var split = new float[n];
        var serial = new float[n];
        kernel.Execute(new[] { x, y }, new[] { split });
        kernel.ExecuteSerial(new[] { x, y }, new[] { serial });
        for (int i = 0; i < n; i++)
        {
            Assert.True(MathCompat.SingleToInt32Bits(split[i]) == MathCompat.SingleToInt32Bits(serial[i]),
                $"n={n}, element {i}: split {split[i]:R} serial {serial[i]:R}");
            float r = MathF.Sqrt(((x[i] - y[i]) * x[i] + y[i]) * y[i]);
            r = -r; r = r < 0f ? 0f : r;
            Assert.True(MathCompat.SingleToInt32Bits(r) == MathCompat.SingleToInt32Bits(serial[i]),
                $"n={n}, element {i}: expected {r:R} kernel {serial[i]:R}");
        }
    }
#endif
}
