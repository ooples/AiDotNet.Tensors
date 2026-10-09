using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class CpuKernelContractTests
{
    private static IKernelEmitter Emitter(bool simd)
    {
        if (!simd) return new CpuDotNetJitEmitter();
#if NET8_0_OR_GREATER
        Skip.IfNot(System.Runtime.Intrinsics.X86.Avx512F.IsSupported, "AVX-512F unavailable.");
#else
        Skip.If(true, "AVX-512F requires net8.0+.");
#endif
        return new CpuAvx512Emitter();
    }

    private static CodegenGraph Graph(int[] shape, CodegenOpKind op = CodegenOpKind.Negate,
        CodegenElementType dtype = CodegenElementType.Float64, bool twoOutputs = false)
    {
        var graph = new CodegenGraph();
        int input = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), dtype, shape, 0));
        int value = graph.AddNode(new CodegenNode(op, new[] { input }, dtype, shape));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { value }, dtype, shape, 0));
        if (twoOutputs)
            graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { input }, dtype, shape, 1));
        return graph;
    }

    private static CodegenKernel Compile(IKernelEmitter emitter, CodegenGraph graph, CodegenElementType dtype)
    {
        var result = emitter.Emit(graph, dtype);
        Assert.False(result.Declined, result.DeclineReason);
        return Assert.IsAssignableFrom<CodegenKernel>(result.Kernel);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void InvalidBuffersAreRejectedBeforeAnyOutputIsWritten(bool simd)
    {
        var kernel = Compile(Emitter(simd), Graph(new[] { 33 }, twoOutputs: true), CodegenElementType.Float64);
        var input = Enumerable.Repeat(2d, 33).ToArray();
        var first = Enumerable.Repeat(123d, 33).ToArray();
        var second = Enumerable.Repeat(456d, 33).ToArray();

        Assert.Throws<ArgumentException>(() => kernel.Execute(new[] { input }, new[] { first, new double[32] }));
        Assert.Throws<ArgumentException>(() => kernel.Execute(new[] { input }, new[] { first, (double[])null! }));
        Assert.Throws<ArgumentException>(() => kernel.Execute(new[] { new double[32] }, new[] { first, second }));
        Assert.Throws<ArgumentException>(() => kernel.Execute(new[] { (double[])null! }, new[] { first, second }));
        Assert.All(first, x => Assert.Equal(123d, x));
        Assert.All(second, x => Assert.Equal(456d, x));

        // Admission failure does not poison the reusable kernel.
        kernel.Execute(new[] { input }, new[] { first, second });
        Assert.All(first, x => Assert.Equal(-2d, x));
        Assert.All(second, x => Assert.Equal(2d, x));
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void InvalidShapesDeclineInsteadOfOverflowingOrEmittingEmptyLoops(bool simd)
    {
        var emitter = Emitter(simd);
        foreach (var shape in new[]
        {
            new[] { -1 }, new[] { -2, -2 }, new[] { 0, -1 },
            new[] { int.MaxValue, int.MaxValue, int.MaxValue },
            new[] { 65536, 65536, 65536, 65536 }
        })
        {
            var result = emitter.Emit(Graph(shape), CodegenElementType.Float64);
            Assert.True(result.Declined);
            Assert.Null(result.Kernel);
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScalarAndEmptyShapesRemainValid(bool simd)
    {
        var emitter = Emitter(simd);
        var scalar = Compile(emitter, Graph(Array.Empty<int>()), CodegenElementType.Float64);
        var output = new double[1];
        scalar.Execute(new[] { new[] { 3d } }, new[] { output });
        Assert.Equal(-3d, output[0]);
        var empty = Compile(emitter, Graph(new[] { int.MaxValue, int.MaxValue, int.MaxValue, 0 }), CodegenElementType.Float64);
        empty.Execute(new[] { Array.Empty<double>() }, new[] { Array.Empty<double>() });
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void MismatchedDtypesMalformedOperandsAndMutatedWiringDecline(bool simd)
    {
        var emitter = Emitter(simd);
        Assert.True(emitter.Emit(Graph(new[] { 4 }), CodegenElementType.Float32).Declined);
        Assert.True(emitter.Emit(Graph(new[] { 4 }, CodegenOpKind.Add), CodegenElementType.Float64).Declined);
        var graph = Graph(new[] { 4 });
        graph[1].Inputs[0] = 1; // Mutated after AddNode's topological check.
        Assert.True(emitter.Emit(graph, CodegenElementType.Float64).Declined);
    }

    [SkippableTheory]
    [InlineData(false, CodegenOpKind.Negate)]
    [InlineData(true, CodegenOpKind.Negate)]
    [InlineData(false, CodegenOpKind.ReLU)]
    [InlineData(true, CodegenOpKind.ReLU)]
    public void FloatAndDoubleBodiesAndTailsPreserveIeeeSigns(bool simd, CodegenOpKind op)
    {
        var emitter = Emitter(simd);
        // Repeat each exceptional value in both the vector body and scalar tail.
        var samples = new[] { 0d, -0d, double.Epsilon, -double.Epsilon, double.PositiveInfinity,
            double.NegativeInfinity, BitConverter.Int64BitsToDouble(0x7ff8000000000042L), -3d, 2d };
        var doubles = Enumerable.Range(0, 41).Select(i => samples[i % samples.Length]).ToArray();
        var doublesOut = new double[doubles.Length];
        Compile(emitter, Graph(new[] { doubles.Length }, op), CodegenElementType.Float64)
            .Execute(new[] { doubles }, new[] { doublesOut });
        for (int i = 0; i < doubles.Length; i++)
        {
            long expected = op == CodegenOpKind.Negate
                ? BitConverter.DoubleToInt64Bits(doubles[i]) ^ long.MinValue
                : BitConverter.DoubleToInt64Bits(doubles[i] < 0 ? 0 : doubles[i]);
            Assert.Equal(expected, BitConverter.DoubleToInt64Bits(doublesOut[i]));
        }
        var floats = doubles.Select(x => (float)x).ToArray();
        var floatsOut = new float[floats.Length];
        Compile(emitter, Graph(new[] { floats.Length }, op, CodegenElementType.Float32), CodegenElementType.Float32)
            .Execute(new[] { floats }, new[] { floatsOut });
        for (int i = 0; i < floats.Length; i++)
        {
            int expected = op == CodegenOpKind.Negate ? Bits(floats[i]) ^ int.MinValue : Bits(floats[i] < 0 ? 0 : floats[i]);
            Assert.Equal(expected, Bits(floatsOut[i]));
        }
    }

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
}
