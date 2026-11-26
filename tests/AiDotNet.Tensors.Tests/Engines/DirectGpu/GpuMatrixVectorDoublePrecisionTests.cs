// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="GpuDoublePrecisionSweepTests"/> covers the tensor ops; these are the <see cref="Matrix{T}"/> and
/// <see cref="Vector{T}"/> ops of the GPU engine. Under the default (type-preserving) policy none may compute a double
/// through a float kernel: operands carry low bits a float cannot hold and magnitudes a float cannot reach, and the
/// result must be the CPU engine's.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuMatrixVectorDoublePrecisionTests
{
    private const int Rows = 96, Columns = 80;       // large enough that no small-size host path hides the device one

    // Positive, distinct, with bits below a float's 24: a float round trip moves every one of them by about 1e-8.
    private static double Value(int i, double seed) => 1.0 + seed + i * 1.0000000123e-3 + ((i * 2654435761u) % 1000) * 1e-13;

    private static Matrix<double> MatrixOf(double seed)
    {
        var matrix = new Matrix<double>(Rows, Columns);
        var span = matrix.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = Value(i, seed);
        return matrix;
    }

    private static Vector<double> VectorOf(double seed, int length = Rows * Columns)
    {
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = Value(i, seed);
        return new Vector<double>(values);
    }

    // Alternating signs with low bits a float cannot hold: Abs and ReLU then see magnitudes that a float round trip
    // would move, and ReLU also exercises its zero branch.
    private static Vector<double> SignedVectorOf(double seed)
    {
        var values = VectorOf(seed).ToArray();
        for (int i = 0; i < values.Length; i += 2) values[i] = -values[i];
        return new Vector<double>(values);
    }

    private static IEngine GpuEngineOrSkip()
    {
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "No GPU engine on this host.");
        return AiDotNetEngine.Current;
    }

    private static void AssertSame(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, string op)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual[i];
            if (BitConverter.DoubleToInt64Bits(e) == BitConverter.DoubleToInt64Bits(a)) continue;
            double scale = Math.Max(Math.Abs(e), double.Epsilon);
            Assert.True(Math.Abs(e - a) <= 1e-13 * scale, $"{op}[{i}]: CPU {e:R}, GPU {a:R}");
        }
    }

    public static IEnumerable<object[]> MatrixOps()
    {
        yield return new object[] { "MatrixAdd" };
        yield return new object[] { "MatrixSubtract" };
        yield return new object[] { "MatrixMultiplyScalar" };
        yield return new object[] { "MatrixMultiply" };
        yield return new object[] { "MatrixTranspose" };
    }

    [SkippableTheory]
    [MemberData(nameof(MatrixOps))]
    public void MatrixOp_OnDoubles_IsNotComputedInFloat(string op)
    {
        var gpu = GpuEngineOrSkip();
        var cpu = (IEngine)new CpuEngine();
        var a = MatrixOf(0.25);
        var b = MatrixOf(0.5);
        var square = new Matrix<double>(Columns, Rows);
        var squareSpan = square.AsWritableSpan();
        for (int i = 0; i < squareSpan.Length; i++) squareSpan[i] = Value(i, 0.75);

        Matrix<double> Run(IEngine engine) => op switch
        {
            "MatrixAdd" => engine.MatrixAdd(a, b),
            "MatrixSubtract" => engine.MatrixSubtract(a, b),
            "MatrixMultiplyScalar" => engine.MatrixMultiplyScalar(a, 0.3000000000000123),
            "MatrixMultiply" => engine.MatrixMultiply(a, square),
            "MatrixTranspose" => engine.MatrixTranspose(a),
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "Unknown op."),
        };

        AssertSame(Run(cpu).AsSpan(), Run(gpu).AsSpan(), op);
    }

    /// <summary>Magnitudes outside a float's range: narrowed, 1.5e300 comes back infinite and 1e-300 zero.</summary>
    [SkippableFact]
    public void MatrixMultiplyScalar_KeepsMagnitudesAFloatCannotHold()
    {
        var gpu = GpuEngineOrSkip();
        var matrix = new Matrix<double>(Rows, Columns);
        var span = matrix.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (i % 3) switch { 0 => 1.5e300, 1 => 1e-300, _ => -0.3 };

        var result = gpu.MatrixMultiplyScalar(matrix, 0.5).AsSpan();
        for (int i = 0; i < result.Length; i++)
        {
            double expected = (i % 3) switch { 0 => 0.75e300, 1 => 0.5e-300, _ => -0.15 };
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(result[i]),
                $"[{i}]: expected {expected:R}, got {result[i]:R}");
        }
    }

    public static IEnumerable<object[]> VectorOps()
    {
        foreach (string op in new[]
        {
            "Add", "Subtract", "Multiply", "MultiplyScalar", "Divide", "DivideScalar", "Max", "Min", "Abs", "Exp", "Exp2",
            "Exp10", "Log", "Log2", "Sqrt", "Power", "Tanh", "Sigmoid", "ReLU", "GELU", "DotProduct", "StridedGather", "StridedScatter",
        })
            yield return new object[] { op };
    }

    [SkippableTheory]
    [MemberData(nameof(VectorOps))]
    public void VectorOp_OnDoubles_IsNotComputedInFloat(string op)
    {
        var gpu = GpuEngineOrSkip();
        var cpu = (IEngine)new CpuEngine();
        var a = VectorOf(0.25);
        var b = VectorOf(0.5);

        double[] Run(IEngine engine) => op switch
        {
            "Add" => engine.Add(a, b).ToArray(),
            "Subtract" => engine.Subtract(a, b).ToArray(),
            "Multiply" => engine.Multiply(a, b).ToArray(),
            "MultiplyScalar" => engine.Multiply(a, 0.3000000000000123).ToArray(),
            "Divide" => engine.Divide(a, b).ToArray(),
            "DivideScalar" => engine.Divide(a, 0.3000000000000123).ToArray(),
            "Max" => engine.Max(a, b).ToArray(),
            "Min" => engine.Min(a, b).ToArray(),
            "Abs" => engine.Abs(SignedVectorOf(0.25)).ToArray(),
            "Exp" => engine.Exp(a).ToArray(),
            "Exp2" => engine.Exp2(a).ToArray(),
            "Exp10" => engine.Exp10(a).ToArray(),
            "Log" => engine.Log(a).ToArray(),
            "Log2" => engine.Log2(a).ToArray(),
            "Sqrt" => engine.Sqrt(a).ToArray(),
            "Power" => engine.Power(a, 1.5000000000000123).ToArray(),
            "Tanh" => engine.Tanh(engine.Divide(a, 16.0)).ToArray(),
            "Sigmoid" => engine.Sigmoid(engine.Divide(a, 16.0)).ToArray(),
            "ReLU" => engine.ReLU(SignedVectorOf(0.25)).ToArray(),
            "GELU" => engine.GELU(engine.Divide(a, 16.0)).ToArray(),
            "DotProduct" => new[] { engine.DotProduct(a, b) },
            "StridedGather" => engine.StridedGather(a, 3, 5, 1000).ToArray(),
            "StridedScatter" => Scatter(engine),
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "Unknown op."),
        };

        // Writes every fifth element; the others must come back as they were, bits included.
        double[] Scatter(IEngine engine)
        {
            var destination = VectorOf(0.75);
            engine.StridedScatter(destination, VectorOf(0.125, 1000), 3, 5);
            return destination.ToArray();
        }

        AssertSame(Run(cpu), Run(gpu), op);
    }

    [SkippableFact]
    public void StridedScatter_SpeedFirst_LeavesUntouchedDoublesExact()
    {
        var gpu = GpuEngineOrSkip();
        // Speed-first may round the scattered values through float, never the elements the scatter does not write.
        using var speedFirst = new GpuExecutionPolicyScope(GpuExecutionPolicy.Default);
        var destination = VectorOf(0.75);
        var before = destination.ToArray();
        gpu.StridedScatter(destination, VectorOf(0.125, 1000), 3, 5);
        var after = destination.ToArray();
        for (int i = 0; i < after.Length; i++)
        {
            if (i >= 3 && (i - 3) % 5 == 0 && (i - 3) / 5 < 1000) continue;
            Assert.True(BitConverter.DoubleToInt64Bits(before[i]) == BitConverter.DoubleToInt64Bits(after[i]),
                $"[{i}]: untouched element changed from {before[i]:R} to {after[i]:R}");
        }
    }
}
