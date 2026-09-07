// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.ExceptionServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
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

    public enum ConvolutionEpilogue
    {
        Allocating,
        InPlace
    }

    public enum NormalizationForwardPath
    {
        GroupNorm,
        GroupNormInto,
        GroupNormSwishInto,
        BatchNorm,
        RmsNorm,
        InstanceNorm
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

    /// <summary>
    /// Runtime half of the source-generated mixed-residency contract. The generator supplies a
    /// compile-time-bound IEngine call for each annotated operation, so the coverage cannot drift
    /// through a string-based operation registry.
    /// </summary>
    public static void VerifyCpuConsumesMixedResidencyInputs(
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> operation)
    {
        var shape = new[] { 4, 8 };
        var left = Filled(shape, 3400);
        var right = PositiveFilled(shape, 3500);
        var expected = operation(Engine, left, right);

        using var nativeLeft = global::AiDotNet.Tensors.Helpers.TensorAllocator.RentNative<float>(shape);
        using var nativeRight = global::AiDotNet.Tensors.Helpers.TensorAllocator.RentNative<float>(shape);
        nativeLeft.CopyFromArray(left.ToArray());
        nativeRight.CopyFromArray(right.ToArray());
        var nativeActual = operation(Engine, nativeLeft, nativeRight);
        AssertClose(expected, nativeActual);

        using (var gpu = new DirectGpuTensorEngine())
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend is available on this machine.");

            var backend = gpu.TestBackend;
            Assert.NotNull(backend);
            if (backend is null)
                return;
            using var gpuLeft = Tensor<float>.FromGpuBuffer(
                backend,
                backend.AllocateBuffer(left.ToArray()),
                shape,
                ownsBuffer: true);
            using var gpuRight = Tensor<float>.FromGpuBuffer(
                backend,
                backend.AllocateBuffer(right.ToArray()),
                shape,
                ownsBuffer: true);

            Assert.True(gpuLeft.IsGpuResident, "precondition: left operand must be GPU-resident");
            Assert.True(gpuRight.IsGpuResident, "precondition: right operand must be GPU-resident");

            var actual = operation(Engine, gpuLeft, gpuRight);

            AssertClose(expected, actual);
            Assert.True(gpuLeft.IsGpuResident, "CPU read changed the left operand's device ownership");
            Assert.True(gpuRight.IsGpuResident, "CPU read changed the right operand's device ownership");
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
    public void MatmulNdWith2DWeight_DoesNotPrivatizeCowWeight()
    {
        var input = Filled(new[] { 2, 3, 4 }, 110);
        var weightSource = Filled(new[] { 4, 5 }, 120);
        var weightClone = (Tensor<float>)weightSource.CloneShared();
        var expected = Engine.TensorMatMul(input, Filled(new[] { 4, 5 }, 120));

        var actual = Engine.TensorMatMul(input, weightClone);

        Assert.True(weightSource.IsCowShared, "ND x 2D matmul privatized the source-side weight");
        Assert.True(weightClone.IsCowShared, "ND x 2D matmul privatized the cloned weight");
        AssertClose(expected, actual);
    }

    [Fact]
    public void MatmulGemv_DoesNotPrivatizeCowWeight()
    {
        var input = Filled(new[] { 3, 4 }, 130);
        var weightSource = Filled(new[] { 4, 1 }, 140);
        var weightClone = (Tensor<float>)weightSource.CloneShared();
        var expected = Engine.TensorMatMul(input, Filled(new[] { 4, 1 }, 140));

        var actual = Engine.TensorMatMul(input, weightClone);

        Assert.True(weightSource.IsCowShared, "GEMV matmul privatized the source-side weight");
        Assert.True(weightClone.IsCowShared, "GEMV matmul privatized the cloned weight");
        AssertClose(expected, actual);
    }

    [Fact]
    public void MatmulNdWith2DWeight_CompiledCapturePreservesCowWeight()
    {
        var input = Filled(new[] { 2, 3, 4 }, 150);
        var weightSource = Filled(new[] { 4, 5 }, 160);
        var weightClone = (Tensor<float>)weightSource.CloneShared();
        var expected = Engine.TensorMatMul(input, Filled(new[] { 4, 5 }, 160));
        CompiledInferencePlan<float> plan;

        using (var scope = GraphMode.Enable())
        {
            Engine.TensorMatMul(input, weightClone);
            Assert.True(weightClone.IsCowShared, "graph recording privatized the cloned ND x 2D weight");
            plan = scope.CompileInference<float>();
            Assert.True(weightClone.IsCowShared, "inference-plan compilation privatized the cloned ND x 2D weight");
        }

        using (plan)
        using (var actual = plan.Execute())
        {
            AssertClose(expected, actual);
            Assert.True(weightClone.IsCowShared, "inference-plan execution privatized the cloned ND x 2D weight");
        }

        Assert.True(weightSource.IsCowShared, "compiled ND x 2D matmul privatized the source-side weight");
        Assert.True(weightClone.IsCowShared, "compiled ND x 2D matmul privatized the cloned weight");
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

    [Theory]
    [InlineData(BinaryOperation.Add)]
    [InlineData(BinaryOperation.Subtract)]
    [InlineData(BinaryOperation.Multiply)]
    [InlineData(BinaryOperation.Divide)]
    public void ElementwiseBinaryIntoFloat_DetachesAliasedCowDestinationWithoutChangingPeer(
        BinaryOperation operation)
    {
        var source = Filled(new[] { 4, 8 }, 3200);
        var originalSourceValues = Filled(new[] { 4, 8 }, 3200);
        var destination = (Tensor<float>)source.CloneShared();
        var right = operation == BinaryOperation.Divide
            ? PositiveFilled(new[] { 4, 8 }, 3300)
            : Filled(new[] { 4, 8 }, 3300);
        var expected = ApplyBinary(operation, originalSourceValues, right);

        ApplyBinaryInto(operation, destination, destination, right);

        AssertClose(originalSourceValues, source);
        AssertClose(expected, destination);
        Assert.False(destination.IsCowShared,
            $"{operation}Into left a writable destination attached to its COW peer");
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
        using (var gpu = new DirectGpuTensorEngine())
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

    [Theory]
    [InlineData(NormalizationForwardPath.GroupNorm)]
    [InlineData(NormalizationForwardPath.GroupNormInto)]
    [InlineData(NormalizationForwardPath.GroupNormSwishInto)]
    [InlineData(NormalizationForwardPath.BatchNorm)]
    [InlineData(NormalizationForwardPath.RmsNorm)]
    [InlineData(NormalizationForwardPath.InstanceNorm)]
    public void NormalizationForward_DoesNotPrivatizeCowOperands(NormalizationForwardPath path)
    {
        var baselineInput = Filled(new[] { 3, 4 }, 910);
        var inputSource = Filled(new[] { 3, 4 }, 910);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var baselineGamma = Filled(new[] { 4 }, 920);
        var gammaSource = Filled(new[] { 4 }, 920);
        var gammaClone = (Tensor<float>)gammaSource.CloneShared();
        var baselineBeta = Filled(new[] { 4 }, 930);
        var betaSource = Filled(new[] { 4 }, 930);
        var betaClone = (Tensor<float>)betaSource.CloneShared();

        var expected = ApplyNormalizationForward(path, baselineInput, baselineGamma, baselineBeta);
        var actual = ApplyNormalizationForward(path, inputClone, gammaClone, betaClone);

        Assert.True(inputSource.IsCowShared, $"{path} privatized the source-side input");
        Assert.True(inputClone.IsCowShared, $"{path} privatized the cloned input");
        Assert.True(gammaSource.IsCowShared, $"{path} privatized the source-side gamma");
        Assert.True(gammaClone.IsCowShared, $"{path} privatized the cloned gamma");
        Assert.True(betaSource.IsCowShared, $"{path} privatized the source-side beta");
        Assert.True(betaClone.IsCowShared, $"{path} privatized the cloned beta");
        AssertClose(expected, actual);
    }

    private static Tensor<float> ApplyNormalizationForward(
        NormalizationForwardPath path,
        Tensor<float> input,
        Tensor<float> gamma,
        Tensor<float> beta)
    {
        switch (path)
        {
            case NormalizationForwardPath.GroupNorm:
                return Engine.GroupNorm(input, 2, gamma, beta, 1e-5, out _, out _);
            case NormalizationForwardPath.GroupNormInto:
            {
                var output = new Tensor<float>(new float[input.Length], input.Shape.ToArray());
                Engine.GroupNormInto(output, input, 2, gamma, beta, 1e-5, out _, out _);
                return output;
            }
            case NormalizationForwardPath.GroupNormSwishInto:
            {
                var output = new Tensor<float>(new float[input.Length], input.Shape.ToArray());
                Engine.GroupNormSwishInto(output, input, 2, gamma, beta, 1e-5);
                return output;
            }
            case NormalizationForwardPath.BatchNorm:
                return Engine.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
            case NormalizationForwardPath.RmsNorm:
                return Engine.RMSNorm(input, gamma, 1e-5, out _);
            case NormalizationForwardPath.InstanceNorm:
                return Engine.InstanceNorm(input, gamma, beta, 1e-5, out _, out _);
            default:
                throw new ArgumentOutOfRangeException(nameof(path));
        }
    }

    [SkippableTheory]
    [InlineData(NormalizationForwardPath.GroupNormInto)]
    [InlineData(NormalizationForwardPath.GroupNormSwishInto)]
    public void DirectGpuNormalizationInto_PreservesCowOperands(NormalizationForwardPath path)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend is available on this machine.");
        IEngine gpuEngine = gpu;

        var baselineInput = Filled(new[] { 1, 4, 2, 2 }, 940);
        var inputSource = Filled(new[] { 1, 4, 2, 2 }, 940);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var baselineGamma = Filled(new[] { 4 }, 950);
        var gammaSource = Filled(new[] { 4 }, 950);
        var gammaClone = (Tensor<float>)gammaSource.CloneShared();
        var baselineBeta = Filled(new[] { 4 }, 960);
        var betaSource = Filled(new[] { 4 }, 960);
        var betaClone = (Tensor<float>)betaSource.CloneShared();
        var expected = ApplyNormalizationForward(path, baselineInput, baselineGamma, baselineBeta);
        var actual = new Tensor<float>(new float[inputClone.Length], inputClone.Shape.ToArray());

        switch (path)
        {
            case NormalizationForwardPath.GroupNormInto:
                gpuEngine.GroupNormInto(actual, inputClone, 2, gammaClone, betaClone, 1e-5, out _, out _);
                break;
            case NormalizationForwardPath.GroupNormSwishInto:
                gpuEngine.GroupNormSwishInto(actual, inputClone, 2, gammaClone, betaClone, 1e-5);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(path));
        }

        Assert.True(inputSource.IsCowShared, $"DirectGpu {path} privatized the source-side input");
        Assert.True(inputClone.IsCowShared, $"DirectGpu {path} privatized the cloned input");
        Assert.True(gammaSource.IsCowShared, $"DirectGpu {path} privatized the source-side gamma");
        Assert.True(gammaClone.IsCowShared, $"DirectGpu {path} privatized the cloned gamma");
        Assert.True(betaSource.IsCowShared, $"DirectGpu {path} privatized the source-side beta");
        Assert.True(betaClone.IsCowShared, $"DirectGpu {path} privatized the cloned beta");
        AssertClose(expected, actual, tol: 2e-4f);
    }

    [SkippableFact]
    public void DirectGpuAddGroupNormInto_PreservesCowOperands()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend is available on this machine.");
        IEngine gpuEngine = gpu;

        var leftSource = Filled(new[] { 1, 4, 2, 2 }, 970);
        var leftClone = (Tensor<float>)leftSource.CloneShared();
        var rightSource = Filled(new[] { 1, 4, 2, 2 }, 980);
        var rightClone = (Tensor<float>)rightSource.CloneShared();
        var gammaSource = Filled(new[] { 4 }, 990);
        var gammaClone = (Tensor<float>)gammaSource.CloneShared();
        var betaSource = Filled(new[] { 4 }, 1000);
        var betaClone = (Tensor<float>)betaSource.CloneShared();
        var expectedSum = Engine.TensorAdd(
            Filled(new[] { 1, 4, 2, 2 }, 970),
            Filled(new[] { 1, 4, 2, 2 }, 980));
        var expected = Engine.GroupNorm(
            expectedSum,
            2,
            Filled(new[] { 4 }, 990),
            Filled(new[] { 4 }, 1000),
            1e-5,
            out _,
            out _);
        var actual = new Tensor<float>(new float[leftClone.Length], leftClone.Shape.ToArray());

        gpuEngine.AddGroupNormInto(actual, leftClone, rightClone, 2, gammaClone, betaClone, 1e-5);

        Assert.True(leftSource.IsCowShared, "DirectGpu AddGroupNormInto privatized the source-side left input");
        Assert.True(leftClone.IsCowShared, "DirectGpu AddGroupNormInto privatized the cloned left input");
        Assert.True(rightSource.IsCowShared, "DirectGpu AddGroupNormInto privatized the source-side right input");
        Assert.True(rightClone.IsCowShared, "DirectGpu AddGroupNormInto privatized the cloned right input");
        Assert.True(gammaSource.IsCowShared, "DirectGpu AddGroupNormInto privatized the source-side gamma");
        Assert.True(gammaClone.IsCowShared, "DirectGpu AddGroupNormInto privatized the cloned gamma");
        Assert.True(betaSource.IsCowShared, "DirectGpu AddGroupNormInto privatized the source-side beta");
        Assert.True(betaClone.IsCowShared, "DirectGpu AddGroupNormInto privatized the cloned beta");
        AssertClose(expected, actual, tol: 2e-4f);
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

    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void FusedConv2DFloat_DoesNotPrivatizeCowKernelOrBias(FusedActivationType activation)
    {
        var input = Filled(new[] { 1, 2, 5, 5 }, 3600);
        var inputSource = Filled(new[] { 1, 2, 5, 5 }, 3600);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var kernel = Filled(new[] { 3, 2, 3, 3 }, 3700);
        var kernelSource = Filled(new[] { 3, 2, 3, 3 }, 3700);
        var kernelClone = (Tensor<float>)kernelSource.CloneShared();
        var bias = Filled(new[] { 3 }, 3800);
        var biasSource = Filled(new[] { 3 }, 3800);
        var biasClone = (Tensor<float>)biasSource.CloneShared();

        var expected = Engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, activation);
        var actual = Engine.FusedConv2D(inputClone, kernelClone, biasClone, 1, 1, 1, 1, 1, 1, activation);

        Assert.True(inputSource.IsCowShared, "fused conv2d privatized the source-side float input");
        Assert.True(kernelSource.IsCowShared, "fused conv2d privatized the source-side float kernel");
        Assert.True(biasSource.IsCowShared, "fused conv2d privatized the source-side float bias");
        Assert.True(inputClone.IsCowShared, "fused conv2d privatized the cloned float input");
        Assert.True(kernelClone.IsCowShared, "fused conv2d privatized the cloned float kernel");
        Assert.True(biasClone.IsCowShared, "fused conv2d privatized the cloned float bias");
        AssertClose(expected, actual);
    }

    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void FusedConv2DDouble_DoesNotPrivatizeCowKernelOrBias(FusedActivationType activation)
    {
        var input = FilledDouble(new[] { 1, 2, 5, 5 }, 3900);
        var inputSource = FilledDouble(new[] { 1, 2, 5, 5 }, 3900);
        var inputClone = (Tensor<double>)inputSource.CloneShared();
        var kernel = FilledDouble(new[] { 3, 2, 3, 3 }, 4000);
        var kernelSource = FilledDouble(new[] { 3, 2, 3, 3 }, 4000);
        var kernelClone = (Tensor<double>)kernelSource.CloneShared();
        var bias = FilledDouble(new[] { 3 }, 4100);
        var biasSource = FilledDouble(new[] { 3 }, 4100);
        var biasClone = (Tensor<double>)biasSource.CloneShared();

        var expected = Engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, activation);
        var actual = Engine.FusedConv2D(inputClone, kernelClone, biasClone, 1, 1, 1, 1, 1, 1, activation);

        Assert.True(inputSource.IsCowShared, "fused conv2d privatized the source-side double input");
        Assert.True(kernelSource.IsCowShared, "fused conv2d privatized the source-side double kernel");
        Assert.True(biasSource.IsCowShared, "fused conv2d privatized the source-side double bias");
        Assert.True(inputClone.IsCowShared, "fused conv2d privatized the cloned double input");
        Assert.True(kernelClone.IsCowShared, "fused conv2d privatized the cloned double kernel");
        Assert.True(biasClone.IsCowShared, "fused conv2d privatized the cloned double bias");
        AssertClose(expected, actual);
    }

    [Fact]
    public void ChannelBiasAdd_DoesNotCreateAViewOrPrivatizeCowOperands()
    {
        var inputSource = Filled(new[] { 2, 4, 3, 5 }, 4120);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var biasSource = Filled(new[] { 4 }, 4130);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var expected = Engine.TensorAdd(
            Filled(new[] { 2, 4, 3, 5 }, 4120),
            Filled(new[] { 4 }, 4130).Reshape(1, 4, 1, 1));

        var actual = Engine.TensorChannelBiasAdd(inputClone, biasClone);

        Assert.True(inputSource.IsCowShared, "channel bias add privatized the source-side input");
        Assert.True(inputClone.IsCowShared, "channel bias add privatized the cloned input");
        Assert.True(biasSource.IsCowShared, "channel bias add created or wrote through a source-side bias alias");
        Assert.True(biasClone.IsCowShared, "channel bias add created or wrote through a cloned bias alias");
        AssertClose(expected, actual);
    }

    [Fact]
    public void ChannelBiasAdd_ThirdPartyEngineFallbackPreservesCompatibilityAndCowBias()
    {
        IEngine proxy = DispatchProxy.Create<IEngine, ForwardingEngineProxy>();
        var forwarding = (ForwardingEngineProxy)(object)proxy;
        forwarding.Inner = new CpuEngine();
        var inputSource = Filled(new[] { 2, 4, 3, 5 }, 4120);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var biasSource = Filled(new[] { 4 }, 4130);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var expected = Engine.TensorChannelBiasAdd(inputSource, biasSource);

        var actual = proxy.TensorChannelBiasAdd(inputClone, biasClone);

        Assert.Contains(nameof(IEngine.TensorPermute), forwarding.Invocations);
        Assert.Contains(nameof(IEngine.TensorAdd), forwarding.Invocations);
        Assert.True(inputSource.IsCowShared, "third-party fallback privatized the source-side input");
        Assert.True(inputClone.IsCowShared, "third-party fallback privatized the cloned input");
        Assert.True(biasSource.IsCowShared, "third-party fallback privatized the source-side bias");
        Assert.True(biasClone.IsCowShared, "third-party fallback privatized the cloned bias");
        AssertClose(expected, actual);
    }

    [Fact]
    public void FusedConv2D_CompiledCaptureDoesNotRetainParameterViews()
    {
        var input = Filled(new[] { 1, 2, 5, 5 }, 4140);
        var kernelSource = Filled(new[] { 3, 2, 3, 3 }, 4150);
        var kernelClone = (Tensor<float>)kernelSource.CloneShared();
        var biasSource = Filled(new[] { 3 }, 4160);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var expected = Engine.FusedConv2D(
            input, kernelSource, biasSource,
            1, 1, 1, 1, 1, 1, FusedActivationType.None);

        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Engine.FusedConv2D(
                input, kernelClone, biasClone,
                1, 1, 1, 1, 1, 1, FusedActivationType.None);
            plan = scope.CompileInference<float>();
        }

        using (plan)
        using (var actual = plan.Execute())
        {
            AssertClose(expected, actual);
        }

        Assert.True(kernelSource.IsCowShared, "compiled capture privatized the source-side kernel");
        Assert.True(kernelClone.IsCowShared, "compiled capture privatized the cloned kernel");
        Assert.True(biasSource.IsCowShared, "compiled capture retained a source-side bias view");
        Assert.True(biasClone.IsCowShared, "compiled capture retained a cloned bias view");
    }

    [Fact]
    public void ChannelBiasAdd_BackwardReducesBatchAndSpatialAxes()
    {
        var input = Filled(new[] { 2, 3, 2, 4 }, 4170);
        var bias = Filled(new[] { 3 }, 4180);
        Dictionary<Tensor<float>, Tensor<float>> gradients;

        using (var tape = new GradientTape<float>())
        {
            var output = Engine.TensorChannelBiasAdd(input, bias);
            var loss = Engine.ReduceSum(output, null);
            gradients = tape.ComputeGradients(loss, new[] { input, bias });
        }

        foreach (float value in gradients[input].ToArray())
            Assert.Equal(1f, value);
        foreach (float value in gradients[bias].ToArray())
            Assert.Equal(16f, value);
    }

    [SkippableFact]
    public void DirectGpuChannelBiasAdd_StaysResidentAndPreservesCowBias()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend is available on this machine.");

        var input = Filled(new[] { 2, 4, 3, 5 }, 4190);
        var biasSource = Filled(new[] { 4 }, 4200);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var expected = Engine.TensorChannelBiasAdd(input, biasSource);

        using var actual = gpu.TensorChannelBiasAdd(input, biasClone);

        Assert.True(biasSource.IsCowShared, "GPU channel bias add privatized the source-side bias");
        Assert.True(biasClone.IsCowShared, "GPU channel bias add privatized the cloned bias");
        Assert.True(actual.IsGpuResident, "GPU channel bias add did not keep the result resident.");
        AssertClose(expected, actual, tol: 2e-4f);
    }

    [Fact]
    public void DirectGpuChannelBiasAdd_EmptyChannelsDoNotDivideByZero()
    {
        using var gpu = new DirectGpuTensorEngine();
        var input = new Tensor<float>(Array.Empty<float>(), new[] { 1, 0, 2, 3 });
        var bias = new Tensor<float>(Array.Empty<float>(), new[] { 0 });

        using var actual = gpu.TensorChannelBiasAdd(input, bias);

        Assert.Equal(new[] { 1, 0, 2, 3 }, actual.Shape.ToArray());
        Assert.Equal(0, actual.Length);
    }

    [Fact]
    public void BroadcastAddInPlace_DoesNotPrivatizeCowBroadcastOperandOrItsView()
    {
        var biasSource = Filled(new[] { 4 }, 4200);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var biasView = biasClone.Reshape(1, 4, 1, 1);
        var destination = Filled(new[] { 1, 4, 3, 3 }, 4300);
        var expected = Engine.TensorAdd(
            Filled(new[] { 1, 4, 3, 3 }, 4300),
            Filled(new[] { 4 }, 4200).Reshape(1, 4, 1, 1));

        Engine.TensorBroadcastAddInPlace(destination, biasView);

        Assert.True(biasSource.IsCowShared, "in-place broadcast add privatized the source-side bias");
        Assert.True(biasClone.IsCowShared, "in-place broadcast add privatized the cloned bias");
        Assert.True(biasView.IsCowShared, "in-place broadcast add privatized the cloned bias view");
        AssertClose(expected, destination);
    }

    [Theory]
    [InlineData(ConvolutionEpilogue.Allocating)]
    [InlineData(ConvolutionEpilogue.InPlace)]
    public void ScalarConv2DWithBias_DoesNotPrivatizeCowOperands(ConvolutionEpilogue epilogue)
    {
        var inputSource = Filled(new[] { 1, 4, 16, 16 }, 4400);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var kernelSource = Filled(new[] { 32, 4, 3, 3 }, 4500);
        var kernelClone = (Tensor<float>)kernelSource.CloneShared();
        var biasSource = Filled(new[] { 32 }, 4600);
        var biasClone = (Tensor<float>)biasSource.CloneShared();
        var biasView = biasClone.Reshape(1, 32, 1, 1);

        var convolution = Engine.Conv2D(inputClone, kernelClone, stride: 1, padding: 1, dilation: 1);
        Tensor<float> actual;
        switch (epilogue)
        {
            case ConvolutionEpilogue.Allocating:
                actual = Engine.TensorAdd(convolution, biasView);
                break;
            case ConvolutionEpilogue.InPlace:
                Engine.TensorBroadcastAddInPlace(convolution, biasView);
                actual = convolution;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(epilogue));
        }

        Assert.True(inputSource.IsCowShared, $"{epilogue} conv privatized the source-side input");
        Assert.True(kernelSource.IsCowShared, $"{epilogue} conv privatized the source-side kernel");
        Assert.True(biasSource.IsCowShared, $"{epilogue} conv privatized the source-side bias");
        Assert.True(inputClone.IsCowShared, $"{epilogue} conv privatized the cloned input");
        Assert.True(kernelClone.IsCowShared, $"{epilogue} conv privatized the cloned kernel");
        Assert.True(biasClone.IsCowShared, $"{epilogue} conv privatized the cloned bias");
        Assert.True(biasView.IsCowShared, $"{epilogue} conv privatized the cloned bias view");
        Assert.Equal(new[] { 1, 32, 16, 16 }, actual.Shape.ToArray());
    }

    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void FusedConvTranspose2D_DoesNotPrivatizeCowOperands(FusedActivationType activation)
    {
        var inputSource = Filled(new[] { 1, 2, 4, 4 }, 4700);
        var inputClone = (Tensor<float>)inputSource.CloneShared();
        var kernelSource = Filled(new[] { 2, 3, 3, 3 }, 4800);
        var kernelClone = (Tensor<float>)kernelSource.CloneShared();
        var biasSource = Filled(new[] { 3 }, 4900);
        var biasClone = (Tensor<float>)biasSource.CloneShared();

        var actual = Engine.FusedConvTranspose2D(
            inputClone, kernelClone, biasClone,
            strideH: 1, strideW: 1, padH: 1, padW: 1,
            outputPadH: 0, outputPadW: 0, activation);

        Assert.True(inputSource.IsCowShared, "fused conv transpose privatized the source-side input");
        Assert.True(kernelSource.IsCowShared, "fused conv transpose privatized the source-side kernel");
        Assert.True(biasSource.IsCowShared, "fused conv transpose privatized the source-side bias");
        Assert.True(inputClone.IsCowShared, "fused conv transpose privatized the cloned input");
        Assert.True(kernelClone.IsCowShared, "fused conv transpose privatized the cloned kernel");
        Assert.True(biasClone.IsCowShared, "fused conv transpose privatized the cloned bias");
        Assert.Equal(new[] { 1, 3, 4, 4 }, actual.Shape.ToArray());
    }

    public class ForwardingEngineProxy : DispatchProxy
    {
        internal CpuEngine? Inner { get; set; }
        internal List<string> Invocations { get; } = new();

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null || args is null)
                throw new InvalidOperationException("Missing forwarded engine invocation metadata.");
            var inner = Inner;
            if (inner is null)
                throw new InvalidOperationException("The forwarding engine has not been configured.");

            Invocations.Add(targetMethod.Name);
            try
            {
                return targetMethod.Invoke(inner, args);
            }
            catch (TargetInvocationException exception) when (exception.InnerException is not null)
            {
                ExceptionDispatchInfo.Capture(exception.InnerException).Throw();
                throw;
            }
        }
    }
}
