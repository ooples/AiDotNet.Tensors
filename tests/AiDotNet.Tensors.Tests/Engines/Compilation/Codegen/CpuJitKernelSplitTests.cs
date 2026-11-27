using System;
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
}
