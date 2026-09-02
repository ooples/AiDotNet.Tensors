// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #624 Stage 2 — "inference on a clone never privatizes." Stage 1 proved a COW clone
/// (<see cref="TensorBase{T}.CloneShared"/>) isolates on write; Stage 2 unlocks the actual benefit:
/// when a cloned model runs inference, its shared WEIGHTS flow through the engine ops as read-only
/// inputs and must NOT trigger copy-on-write privatization (otherwise an O(1) clone silently
/// becomes a full weight-buffer copy on the first matmul — zero benefit for the large-model case).
///
/// <para>Each test clones a weight tensor, runs the op with the clone as the weight operand, and
/// asserts (1) the clone is STILL COW-shared afterwards (the op read it read-only) and (2) the
/// result is byte-identical to running the op against an independent non-shared weight. A failing
/// "still shared" assertion is a precise pointer to an op whose input read still routes through the
/// privatizing <c>GetDataArray()</c> instead of the read-only <c>GetReadOnlyDataArray()</c>.</para>
/// </summary>
public class TensorCowInferenceReadPathTests
{
    public enum BinaryOperation
    {
        Add,
        Subtract,
        Multiply,
        Divide
    }

    private static readonly CpuEngine Engine = new CpuEngine();

    private static Tensor<float> Filled(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        // Deterministic, non-trivial values (no RNG needed; reproducible).
        for (int i = 0; i < n; i++)
            data[i] = (float)Math.Sin(0.123 * (i + seed) + 0.7);
        return new Tensor<float>(data, (int[])shape.Clone());
    }

    private static Tensor<float> PositiveFilled(int[] shape, int seed)
    {
        var tensor = Filled(shape, seed);
        var data = tensor.ToArray();
        for (int i = 0; i < data.Length; i++)
            data[i] = 1.25f + Math.Abs(data[i]);
        return new Tensor<float>(data, (int[])shape.Clone());
    }

    private static Tensor<double> FilledDouble(int[] shape, int seed, bool positive = false)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            double value = Math.Sin(0.123 * (i + seed) + 0.7);
            data[i] = positive ? 1.25 + Math.Abs(value) : value;
        }
        return new Tensor<double>(data, (int[])shape.Clone());
    }

    /// <summary>An independent non-shared weight + a COW clone of the same data.</summary>
    private static (Tensor<float> independent, Tensor<float> cowClone) Weight(int[] shape, int seed)
    {
        var w = Filled(shape, seed);
        var cowSource = Filled(shape, seed);            // separate buffer, identical values
        var clone = (Tensor<float>)cowSource.CloneShared();
        Assert.True(clone.IsCowShared, "precondition: CloneShared must flag the clone COW");
        return (w, clone);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-4f)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol + 1e-3f * Math.Abs(e[i]),
                $"mismatch at {i}: expected {e[i]}, got {a[i]}");
    }

    private static void AssertClose(Tensor<double> expected, Tensor<double> actual, double tol = 1e-10)
    {
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tol + 1e-9 * Math.Abs(e[i]),
                $"mismatch at {i}: expected {e[i]}, got {a[i]}");
    }

    private static Tensor<T> ApplyBinary<T>(BinaryOperation operation, Tensor<T> left, Tensor<T> right)
    {
        return operation switch
        {
            BinaryOperation.Add => Engine.TensorAdd(left, right),
            BinaryOperation.Subtract => Engine.TensorSubtract(left, right),
            BinaryOperation.Multiply => Engine.TensorMultiply(left, right),
            BinaryOperation.Divide => Engine.TensorDivide(left, right),
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
    }

    private static void ApplyBinaryInto<T>(BinaryOperation operation, Tensor<T> destination, Tensor<T> left, Tensor<T> right)
    {
        switch (operation)
        {
            case BinaryOperation.Add:
                Engine.TensorAddInto(destination, left, right);
                break;
            case BinaryOperation.Subtract:
                Engine.TensorSubtractInto(destination, left, right);
                break;
            case BinaryOperation.Multiply:
                Engine.TensorMultiplyInto(destination, left, right);
                break;
            case BinaryOperation.Divide:
                Engine.TensorDivideInto(destination, left, right);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(operation));
        }
    }

    [Theory]
    [InlineData(2, 4, 3)]    // small
    [InlineData(8, 16, 32)]  // medium-M float fast path
    [InlineData(64, 128, 96)]
    public void Matmul_DoesNotPrivatizeCowWeight(int m, int k, int n)
    {
        var x = Filled(new[] { m, k }, 1);
        var (w, wClone) = Weight(new[] { k, n }, 100);

        var expected = Engine.TensorMatMul(x, w);
        var actual = Engine.TensorMatMul(x, wClone);

        Assert.True(wClone.IsCowShared, "matmul privatized the COW weight (operand B read through a write accessor)");
        AssertClose(expected, actual);
    }

    [Fact]
    public void LayerNorm_DoesNotPrivatizeCowGammaBeta()
    {
        var x = Filled(new[] { 4, 8 }, 1);
        var (gamma, gammaClone) = Weight(new[] { 8 }, 200);
        var (beta, betaClone) = Weight(new[] { 8 }, 300);

        var expected = Engine.TensorLayerNorm(x, gamma, beta);
        var actual = Engine.TensorLayerNorm(x, gammaClone, betaClone);

        Assert.True(gammaClone.IsCowShared, "layernorm privatized the COW gamma");
        Assert.True(betaClone.IsCowShared, "layernorm privatized the COW beta");
        AssertClose(expected, actual);
    }

    [Fact]
    public void Conv2D_DoesNotPrivatizeCowKernel()
    {
        // input [N=1, C=2, H=5, W=5], kernel [outC=3, inC=2, kh=3, kw=3]
        var x = Filled(new[] { 1, 2, 5, 5 }, 1);
        var (kernel, kernelClone) = Weight(new[] { 3, 2, 3, 3 }, 400);

        var expected = Engine.TensorConv2D(x, kernel, stride: 1, padding: 1, dilation: 1);
        var actual = Engine.TensorConv2D(x, kernelClone, stride: 1, padding: 1, dilation: 1);

        Assert.True(kernelClone.IsCowShared, "conv2d privatized the COW kernel");
        AssertClose(expected, actual);
    }

    [Fact]
    public void Embedding_DoesNotPrivatizeCowTable()
    {
        var indices = new Tensor<int>(new[] { 0, 3, 1, 2 }, new[] { 4 });
        var (table, tableClone) = Weight(new[] { 5, 6 }, 500);

        var expected = Engine.Embedding(indices, table);
        var actual = Engine.Embedding(indices, tableClone);

        Assert.True(tableClone.IsCowShared, "embedding privatized the COW table");
        AssertClose(expected, actual);
    }

    [Fact]
    public void BatchMatMul_DoesNotPrivatizeCowOperand()
    {
        // attention-style batched matmul [B, M, K] x [B, K, N]
        var a = Filled(new[] { 2, 3, 4 }, 1);
        var (b, bClone) = Weight(new[] { 2, 4, 5 }, 600);

        var expected = Engine.TensorBatchMatMul(a, b);
        var actual = Engine.TensorBatchMatMul(a, bClone);

        Assert.True(bClone.IsCowShared, "batch matmul privatized the COW operand");
        AssertClose(expected, actual);
    }

    [Theory]
    [InlineData(BinaryOperation.Add)]
    [InlineData(BinaryOperation.Subtract)]
    [InlineData(BinaryOperation.Multiply)]
    [InlineData(BinaryOperation.Divide)]
    public void ElementwiseBinaryFloat_DoesNotPrivatizeEitherCowFamily(BinaryOperation operation)
    {
        var leftSource = Filled(new[] { 4, 8 }, 2400);
        var rightSource = operation == BinaryOperation.Divide
            ? PositiveFilled(new[] { 4, 8 }, 2500)
            : Filled(new[] { 4, 8 }, 2500);
        var leftClone = (Tensor<float>)leftSource.CloneShared();
        var rightClone = (Tensor<float>)rightSource.CloneShared();
        var expectedLeft = Filled(new[] { 4, 8 }, 2400);
        var expectedRight = operation == BinaryOperation.Divide
            ? PositiveFilled(new[] { 4, 8 }, 2500)
            : Filled(new[] { 4, 8 }, 2500);

        var expected = ApplyBinary(operation, expectedLeft, expectedRight);
        var actual = ApplyBinary(operation, leftClone, rightClone);

        Assert.True(leftSource.IsCowShared, $"{operation} privatized the source-side left operand");
        Assert.True(rightSource.IsCowShared, $"{operation} privatized the source-side right operand");
        Assert.True(leftClone.IsCowShared, $"{operation} privatized the cloned left operand");
        Assert.True(rightClone.IsCowShared, $"{operation} privatized the cloned right operand");
        AssertClose(expected, actual);
    }

    [Theory]
    [InlineData(BinaryOperation.Add)]
    [InlineData(BinaryOperation.Subtract)]
    [InlineData(BinaryOperation.Multiply)]
    [InlineData(BinaryOperation.Divide)]
    public void ElementwiseBinaryDouble_DoesNotPrivatizeEitherCowFamily(BinaryOperation operation)
    {
        bool positiveRight = operation == BinaryOperation.Divide;
        var leftSource = FilledDouble(new[] { 4, 8 }, 2600);
        var rightSource = FilledDouble(new[] { 4, 8 }, 2700, positiveRight);
        var leftClone = (Tensor<double>)leftSource.CloneShared();
        var rightClone = (Tensor<double>)rightSource.CloneShared();
        var expectedLeft = FilledDouble(new[] { 4, 8 }, 2600);
        var expectedRight = FilledDouble(new[] { 4, 8 }, 2700, positiveRight);

        var expected = ApplyBinary(operation, expectedLeft, expectedRight);
        var actual = ApplyBinary(operation, leftClone, rightClone);

        Assert.True(leftSource.IsCowShared, $"double {operation} privatized the source-side left operand");
        Assert.True(rightSource.IsCowShared, $"double {operation} privatized the source-side right operand");
        Assert.True(leftClone.IsCowShared, $"double {operation} privatized the cloned left operand");
        Assert.True(rightClone.IsCowShared, $"double {operation} privatized the cloned right operand");
        AssertClose(expected, actual);
    }

    [Theory]
    [InlineData(BinaryOperation.Add)]
    [InlineData(BinaryOperation.Subtract)]
    [InlineData(BinaryOperation.Multiply)]
    [InlineData(BinaryOperation.Divide)]
    public void ElementwiseBinaryIntoFloat_DoesNotPrivatizeCowInputs(BinaryOperation operation)
    {
        var leftSource = Filled(new[] { 4, 8 }, 2800);
        var rightSource = operation == BinaryOperation.Divide
            ? PositiveFilled(new[] { 4, 8 }, 2900)
            : Filled(new[] { 4, 8 }, 2900);
        var leftClone = (Tensor<float>)leftSource.CloneShared();
        var rightClone = (Tensor<float>)rightSource.CloneShared();
        var expectedLeft = Filled(new[] { 4, 8 }, 2800);
        var expectedRight = operation == BinaryOperation.Divide
            ? PositiveFilled(new[] { 4, 8 }, 2900)
            : Filled(new[] { 4, 8 }, 2900);
        var expected = ApplyBinary(operation, expectedLeft, expectedRight);
        var destination = new Tensor<float>(new[] { 4, 8 });

        ApplyBinaryInto(operation, destination, leftClone, rightClone);

        Assert.True(leftSource.IsCowShared, $"{operation}Into privatized the source-side left operand");
        Assert.True(rightSource.IsCowShared, $"{operation}Into privatized the source-side right operand");
        Assert.True(leftClone.IsCowShared, $"{operation}Into privatized the cloned left operand");
        Assert.True(rightClone.IsCowShared, $"{operation}Into privatized the cloned right operand");
        AssertClose(expected, destination);
    }

    [Theory]
    [InlineData(BinaryOperation.Add)]
    [InlineData(BinaryOperation.Subtract)]
    [InlineData(BinaryOperation.Multiply)]
    [InlineData(BinaryOperation.Divide)]
    public void ElementwiseBinaryIntoDouble_DoesNotPrivatizeCowInputs(BinaryOperation operation)
    {
        bool positiveRight = operation == BinaryOperation.Divide;
        var leftSource = FilledDouble(new[] { 4, 8 }, 3000);
        var rightSource = FilledDouble(new[] { 4, 8 }, 3100, positiveRight);
        var leftClone = (Tensor<double>)leftSource.CloneShared();
        var rightClone = (Tensor<double>)rightSource.CloneShared();
        var expectedLeft = FilledDouble(new[] { 4, 8 }, 3000);
        var expectedRight = FilledDouble(new[] { 4, 8 }, 3100, positiveRight);
        var expected = ApplyBinary(operation, expectedLeft, expectedRight);
        var destination = new Tensor<double>(new[] { 4, 8 });

        ApplyBinaryInto(operation, destination, leftClone, rightClone);

        Assert.True(leftSource.IsCowShared, $"double {operation}Into privatized the source-side left operand");
        Assert.True(rightSource.IsCowShared, $"double {operation}Into privatized the source-side right operand");
        Assert.True(leftClone.IsCowShared, $"double {operation}Into privatized the cloned left operand");
        Assert.True(rightClone.IsCowShared, $"double {operation}Into privatized the cloned right operand");
        AssertClose(expected, destination);
    }

    [Fact]
    public void BroadcastDivide_DoesNotPrivatizeCowOperands()
    {
        var numeratorSource = Filled(new[] { 2, 4, 3, 3 }, 1600);
        var divisorSource = PositiveFilled(new[] { 1, 4, 1, 1 }, 1700);
        var numeratorClone = (Tensor<float>)numeratorSource.CloneShared();
        var divisorClone = (Tensor<float>)divisorSource.CloneShared();

        var expected = Engine.TensorDivide(numeratorSource, divisorSource);
        var actual = Engine.TensorDivide(numeratorClone, divisorClone);

        Assert.True(numeratorSource.IsCowShared, "broadcast divide privatized the source-side numerator");
        Assert.True(divisorSource.IsCowShared, "broadcast divide privatized the source-side divisor");
        Assert.True(numeratorClone.IsCowShared, "broadcast divide privatized the cloned numerator");
        Assert.True(divisorClone.IsCowShared, "broadcast divide privatized the cloned divisor");
        AssertClose(expected, actual);
    }

    [Fact]
    public void StridedDivide_DoesNotPrivatizeCowViewFamilies()
    {
        var leftSource = Filled(new[] { 2, 3 }, 1800);
        var leftClone = (Tensor<float>)leftSource.CloneShared();
        var sourceView = leftSource.Transpose(new[] { 1, 0 });
        var cloneView = leftClone.Transpose(new[] { 1, 0 });
        var divisor = PositiveFilled(new[] { 3, 2 }, 1900);

        var expected = Engine.TensorDivide(sourceView, divisor);
        var actual = Engine.TensorDivide(cloneView, divisor);

        Assert.True(leftSource.IsCowShared, "strided divide privatized the source alias family");
        Assert.True(leftClone.IsCowShared, "strided divide privatized the clone alias family");
        Assert.True(sourceView.IsCowShared, "strided divide privatized the source view");
        Assert.True(cloneView.IsCowShared, "strided divide privatized the cloned view");
        AssertClose(expected, actual);
    }

    [SkippableFact]
    public void DirectGpuDivide_DoesNotPrivatizeCowInputs()
    {
        DirectGpuTensorEngine gpu;
        try
        {
            gpu = new DirectGpuTensorEngine();
        }
        catch
        {
            Skip.If(true, "No DirectGpu backend can be initialized on this machine.");
            return;
        }

        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend is available on this machine.");

            var leftSource = Filled(new[] { 8, 8 }, 2200);
            var rightSource = PositiveFilled(new[] { 8, 8 }, 2300);
            var leftClone = (Tensor<float>)leftSource.CloneShared();
            var rightClone = (Tensor<float>)rightSource.CloneShared();

            var expected = Engine.TensorDivide(leftSource, rightSource);
            var actual = gpu.TensorDivide(leftClone, rightClone);

            Assert.True(leftClone.IsCowShared, "DirectGpu divide privatized the cloned left operand");
            Assert.True(rightClone.IsCowShared, "DirectGpu divide privatized the cloned right operand");
            AssertClose(expected, actual, tol: 2e-4f);
        }
    }

    [Fact]
    public void GroupNorm_DoesNotPrivatizeCowGammaBeta()
    {
        // [N=1, C=4, H=2, W=2], 2 groups
        var x = Filled(new[] { 1, 4, 2, 2 }, 1);
        var (gamma, gammaClone) = Weight(new[] { 4 }, 800);
        var (beta, betaClone) = Weight(new[] { 4 }, 900);

        var expected = Engine.GroupNorm(x, 2, gamma, beta, 1e-5, out _, out _);
        var actual = Engine.GroupNorm(x, 2, gammaClone, betaClone, 1e-5, out _, out _);

        Assert.True(gammaClone.IsCowShared, "groupnorm privatized the COW gamma");
        Assert.True(betaClone.IsCowShared, "groupnorm privatized the COW beta");
        AssertClose(expected, actual);
    }

    [Fact]
    public void FusedLinear_DoesNotPrivatizeCowWeightsBias()
    {
        var x = Filled(new[] { 2, 4 }, 1);
        var (w, wClone) = Weight(new[] { 4, 3 }, 1000);
        var (bias, biasClone) = Weight(new[] { 3 }, 1100);

        var expected = Engine.FusedLinear(x, w, bias, AiDotNet.Tensors.Engines.FusedActivationType.None);
        var actual = Engine.FusedLinear(x, wClone, biasClone, AiDotNet.Tensors.Engines.FusedActivationType.None);

        Assert.True(wClone.IsCowShared, "fused linear privatized the COW weights");
        Assert.True(biasClone.IsCowShared, "fused linear privatized the COW bias");
        AssertClose(expected, actual);
    }
}
