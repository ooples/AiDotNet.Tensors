// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Double tensors on the device: the element-wise kernels equal the host bit for bit, and an op's result stays on the
/// device until the host reads it.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuFloat64ResidencyTests
{
    private const int Large = DirectGpuTensorEngine.Float64ReductionMinLength + 4097;

    private static (DirectGpuTensorEngine Engine, IDirectGpuBackend Backend, IGpuFloat64Elementwise Elementwise)? Device()
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu) return null;
        if (gpu.GetBackend() is not { } backend) return null;
        if (backend is not IGpuFloat64Elementwise elementwise || !elementwise.SupportsFloat64Elementwise) return null;
        if (backend is not IGpuFloat64Reductions reductions || !reductions.SupportsFloat64Reductions) return null;
        return backend.SupportsElementType(GpuElementType.Float64) ? (gpu, backend, elementwise) : null;
    }

    // Values with full 53-bit mantissas and mixed signs and magnitudes: a float round trip, a fused multiply-add or a
    // reciprocal-multiply division would each change the last bits.
    private static double[] Values(int length, int seed)
    {
        var values = new double[length];
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 1UL;
        for (int i = 0; i < length; i++)
        {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            double unit = (state >> 11) * (1.0 / 9007199254740992.0);
            double magnitude = Math.Pow(10.0, (i % 13) - 6);
            values[i] = (unit - 0.5) * magnitude + ((i % 7 == 0) ? 3.0 : 0.0);
            if (values[i] == 0.0) values[i] = 1.0;
        }

        return values;
    }

    [SkippableTheory]
    [InlineData(GpuFloat64BinaryOperation.Add)]
    [InlineData(GpuFloat64BinaryOperation.Subtract)]
    [InlineData(GpuFloat64BinaryOperation.Multiply)]
    [InlineData(GpuFloat64BinaryOperation.Divide)]
    public void BinaryKernel_EqualsTheHost_BitForBit(GpuFloat64BinaryOperation operation)
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var (_, backend, elementwise) = device.Value;
        foreach (int size in new[] { 1, 2, 255, 256, 257, 4097, 100_003 })
        {
            var left = Values(size, 1);
            var right = Values(size, 2);
            using var a = backend.AllocateBuffer(size, GpuElementType.Float64);
            using var b = backend.AllocateBuffer(size, GpuElementType.Float64);
            using var output = backend.AllocateBuffer(size, GpuElementType.Float64);
            backend.UploadBuffer<double>(a, left);
            backend.UploadBuffer<double>(b, right);
            elementwise.BinaryFloat64(operation, a, b, output, size);
            var actual = new double[size];
            backend.DownloadBuffer<double>(output, actual);
            for (int i = 0; i < size; i++)
            {
                double expected = operation switch
                {
                    GpuFloat64BinaryOperation.Add => left[i] + right[i],
                    GpuFloat64BinaryOperation.Subtract => left[i] - right[i],
                    GpuFloat64BinaryOperation.Multiply => left[i] * right[i],
                    _ => left[i] / right[i],
                };
                Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual[i]),
                    $"{operation} at {i} of {size}: host {expected:R}, device {actual[i]:R}.");
            }
        }
    }

    [SkippableFact]
    public void BinaryKernel_WritesOnlyTheRequestedPrefix_AndRefusesOtherBuffers()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var (_, backend, elementwise) = device.Value;
        var left = Values(1000, 3);
        var sentinel = new double[1000];
        for (int i = 0; i < sentinel.Length; i++) sentinel[i] = -7.0;
        using var a = backend.AllocateBuffer(1000, GpuElementType.Float64);
        using var output = backend.AllocateBuffer(1000, GpuElementType.Float64);
        using var floats = backend.AllocateBuffer(1000);
        backend.UploadBuffer<double>(a, left);
        backend.UploadBuffer<double>(output, sentinel);
        elementwise.BinaryFloat64(GpuFloat64BinaryOperation.Add, a, a, output, 300);
        var actual = new double[1000];
        backend.DownloadBuffer<double>(output, actual);
        for (int i = 0; i < 1000; i++)
            Assert.Equal(i < 300 ? left[i] + left[i] : -7.0, actual[i]);

        Assert.Throws<ArgumentException>(() => elementwise.BinaryFloat64(GpuFloat64BinaryOperation.Add, a, floats, output, 10));
        Assert.Throws<ArgumentException>(() => elementwise.BinaryFloat64(GpuFloat64BinaryOperation.Add, a, a, floats, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => elementwise.BinaryFloat64(GpuFloat64BinaryOperation.Add, a, a, output, 1001));
    }

    [SkippableFact]
    public void Planner_RoutesDoubleArithmeticToTheDevice_AndEverythingElseDoubleToTheCpu()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var backend = device.Value.Backend;
        foreach (var operation in new[]
                 {
                     GpuPrecisionOperation.Add, GpuPrecisionOperation.Subtract, GpuPrecisionOperation.Multiply,
                     GpuPrecisionOperation.Divide, GpuPrecisionOperation.MatMul,
                 })
        {
            var plan = GpuPrecisionPlanner.CreatePlan<double>(backend, operation, "probe");
            Assert.Equal(GpuExecutionRoute.Gpu, plan.Route);
            Assert.Equal(GpuScalarType.Float64, plan.InputStorage);
            Assert.Equal(GpuScalarType.Float64, plan.OutputStorage);
        }

        // No double kernel exists for these, so the default (preserving) policy must keep them exact on the CPU.
        foreach (var operation in new[]
                 {
                     GpuPrecisionOperation.Elementwise, GpuPrecisionOperation.Relu, GpuPrecisionOperation.BatchMatMul,
                     GpuPrecisionOperation.MatMulTransposed, GpuPrecisionOperation.Convolution,
                     GpuPrecisionOperation.Reduction, GpuPrecisionOperation.General,
                 })
            Assert.Equal(GpuExecutionRoute.Cpu, GpuPrecisionPlanner.CreatePlan<double>(backend, operation, "probe").Route);

        // Float is unaffected: it keeps its single-precision route.
        Assert.Equal(GpuScalarType.Float32,
            GpuPrecisionPlanner.CreatePlan<float>(backend, GpuPrecisionOperation.Multiply, "probe").InputStorage);
    }


    // Leaves registered as persistent are uploaded once and stay on the device; an unregistered host tensor is not
    // cached there (see the engine's residency notes), so two of them are computed by the CPU engine.
    private static (Tensor<double> X, Tensor<double> Y) PersistentLeaves(IEngine gpu, int length, int seed)
    {
        var x = new Tensor<double>(Values(length, seed), new[] { length });
        var y = new Tensor<double>(Values(length, seed + 1), new[] { length });
        gpu.RegisterPersistentTensor(x, PersistentTensorRole.Weights);
        gpu.RegisterPersistentTensor(y, PersistentTensorRole.Weights);
        return (x, y);
    }

    private static void AssertBitEqual(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(actual[i]),
                $"Element {i}: cpu {expected[i]:R}, device {actual[i]:R}.");
    }

    [SkippableFact]
    public void Chain_OverPersistentLeaves_StaysOnTheDevice_AndEqualsTheCpuEngine()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        var cpu = (IEngine)new CpuEngine();
        var (x, y) = PersistentLeaves(gpu, Large, 4);
        try
        {
            long before = GpuLaunchProbe.Readbacks;
            var sum = gpu.TensorAdd(x, y);
            var product = gpu.TensorMultiply(sum, x);
            var difference = gpu.TensorSubtract(product, y);
            var quotient = gpu.TensorDivide(difference, x);
            Assert.True(gpuEngine.IsFloat64Resident(x), "The persistent leaf is not resident on the device.");
            Assert.True(gpuEngine.IsFloat64Resident(sum), "The sum is not resident on the device.");
            Assert.True(gpuEngine.IsFloat64Resident(quotient), "The quotient is not resident on the device.");
            Assert.Equal(before, GpuLaunchProbe.Readbacks);

            // The reduction of a resident tensor reads one scalar back and nothing else.
            double total = gpu.TensorSum(quotient);
            Assert.Equal(before + 1, GpuLaunchProbe.Readbacks);

            var expected = cpu.TensorDivide(cpu.TensorSubtract(cpu.TensorMultiply(cpu.TensorAdd(x, y), x), y), x);
            // Element-wise values are single IEEE operations: identical. The total differs only by summation order.
            AssertBitEqual(expected.AsSpan(), quotient.AsSpan());
            double cpuTotal = cpu.TensorSum(expected);
            Assert.True(Math.Abs(total - cpuTotal) <= 1e-9 * Math.Abs(cpuTotal) + 1e-9, $"device {total:R}, cpu {cpuTotal:R}");
            Assert.Equal(new[] { Large }, quotient.Shape.ToArray());
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
        }
    }

    [SkippableTheory]
    [InlineData(4096)]
    [InlineData(Large)]
    public void TwoPlainHostOperands_AreComputedOnTheCpu_AtAnySize(int length)
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        var x = new Tensor<double>(Values(length, 6), new[] { length });
        var y = new Tensor<double>(Values(length, 7), new[] { length });
        long uploadsBefore = GpuLaunchProbe.TuningUploads;
        long readbacksBefore = GpuLaunchProbe.Readbacks;
        var sum = gpu.TensorAdd(x, y);
        Assert.False(gpuEngine.IsFloat64Resident(sum));
        Assert.False(gpuEngine.IsFloat64Resident(x));
        Assert.Equal(readbacksBefore, GpuLaunchProbe.Readbacks);
        Assert.Equal(uploadsBefore, GpuLaunchProbe.TuningUploads);
        var expected = new double[length];
        for (int i = 0; i < length; i++) expected[i] = x[i] + y[i];
        AssertBitEqual(expected, sum.AsSpan());
    }

    [SkippableFact]
    public void SmallPersistentOperands_AreComputedOnTheCpu()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        int length = DirectGpuTensorEngine.Float64ElementwiseMinLength - 1;
        var (x, y) = PersistentLeaves(gpu, length, 20);
        try
        {
            var product = gpu.TensorMultiply(x, y);
            Assert.False(gpuEngine.IsFloat64Resident(product));
            Assert.Equal(x[length - 1] * y[length - 1], product[length - 1]);
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
        }
    }

    [SkippableFact]
    public void HostWrites_ToAResult_AndToAPersistentLeaf_AreSeenByTheNextOp()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        var (x, y) = PersistentLeaves(gpu, Large, 8);
        try
        {
            var sum = gpu.TensorAdd(x, y);
            Assert.True(gpuEngine.IsFloat64Resident(sum));
            double untouched = x[18] + y[18];

            // The write brings the device values down, then replaces one; the device copy is now stale.
            sum[17] = 123.5;
            Assert.False(gpuEngine.IsFloat64Resident(sum));
            var doubled = gpu.TensorAdd(sum, sum);
            Assert.Equal(247.0, doubled[17]);
            Assert.Equal(untouched * 2.0, doubled[18]);
            Assert.Equal(x[0] + y[0], sum[0]);

            // A persistent leaf written on the host is uploaded again.
            x[5] = 1000.0;
            Assert.False(gpuEngine.IsFloat64Resident(x));
            var again = gpu.TensorAdd(x, y);
            Assert.True(gpuEngine.IsFloat64Resident(x));
            Assert.Equal(1000.0 + y[5], again[5]);
            Assert.Equal(x[6] + y[6], again[6]);
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
        }
    }

    [SkippableFact]
    public void UnregisteringOrInvalidating_APersistentLeaf_FreesItsDeviceCopy()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        gpuEngine.ClearFloat64Residents();
        var (x, y) = PersistentLeaves(gpu, Large, 30);
        try
        {
            var sum = gpu.TensorAdd(x, y);
            Assert.True(gpuEngine.IsFloat64Resident(x));
            gpu.InvalidatePersistentTensor(x);
            Assert.False(gpuEngine.IsFloat64Resident(x));
            Assert.True(gpuEngine.IsFloat64Resident(y));
            Assert.Equal(x[9] + y[9], sum[9]);
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
        }

        Assert.False(gpuEngine.IsFloat64Resident(y));
        // No longer persistent: two plain host operands again.
        Assert.False(gpuEngine.IsFloat64Resident(gpu.TensorAdd(x, y)));
    }

    [SkippableFact]
    public void ResultsNothingCanRead_AreFreedWithoutADownload()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        gpuEngine.ClearFloat64Residents();
        long? savedBudget = DirectGpuTensorEngine.Float64ResidentBudgetOverride;
        var (x, y) = PersistentLeaves(gpu, Large, 40);
        try
        {
            long leafBytes = 2L * Large * sizeof(double);
            long readbacksBefore = GpuLaunchProbe.Readbacks;
            RunAndDropIntermediates(gpu, x, y);
            Assert.True(gpuEngine.Float64ResidentBytes > leafBytes, "The chain left nothing on the device to free.");

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            // Over a budget with room for the leaves and one result, the next op reclaims the dead results: its own
            // result (in a buffer one of them gave up) is the only one left beside the leaves.
            DirectGpuTensorEngine.Float64ResidentBudgetOverride = 3L * Large * sizeof(double) + 1024;
            var kept = gpu.TensorAdd(x, y);
            Assert.Equal(leafBytes + (long)Large * sizeof(double), gpuEngine.Float64ResidentBytes);
            Assert.True(gpuEngine.Float64PooledBytes > 0, "No dead result's buffer was pooled.");
            Assert.Equal(readbacksBefore, GpuLaunchProbe.Readbacks);
            Assert.Equal(x[3] + y[3], kept[3]);
        }
        finally
        {
            DirectGpuTensorEngine.Float64ResidentBudgetOverride = savedBudget;
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
        }
    }

    // Its own frame, so no local keeps an intermediate reachable once it returns.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void RunAndDropIntermediates(IEngine gpu, Tensor<double> x, Tensor<double> y)
    {
        var value = gpu.TensorAdd(x, y);
        for (int i = 0; i < 6; i++) value = gpu.TensorMultiply(value, y);
    }

    [SkippableFact]
    public void ResidentResults_OverTheBudget_ComeBackToTheHostIntact()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        long? savedBudget = DirectGpuTensorEngine.Float64ResidentBudgetOverride;
        gpuEngine.ClearFloat64Residents();
        var (x, y) = PersistentLeaves(gpu, Large, 10);
        try
        {
            // Room for the two leaves and two results; the third result pushes the oldest result out.
            DirectGpuTensorEngine.Float64ResidentBudgetOverride = 4L * Large * sizeof(double) + 1024;
            var first = gpu.TensorAdd(x, y);
            var second = gpu.TensorMultiply(x, y);
            var third = gpu.TensorSubtract(x, y);
            Assert.False(gpuEngine.IsFloat64Resident(first));
            Assert.True(gpuEngine.IsFloat64Resident(second));
            Assert.True(gpuEngine.IsFloat64Resident(third));
            Assert.True(gpuEngine.Float64ResidentBytes <= 4L * Large * sizeof(double) + 1024);
            Assert.Equal(x[5] + y[5], first[5]);
            Assert.Equal(x[Large - 1] + y[Large - 1], first[Large - 1]);

            gpuEngine.ClearFloat64Residents();
            Assert.Equal(0L, gpuEngine.Float64ResidentBytes);
            Assert.Equal(x[5] * y[5], second[5]);
            Assert.Equal(x[Large - 1] - y[Large - 1], third[Large - 1]);
        }
        finally
        {
            DirectGpuTensorEngine.Float64ResidentBudgetOverride = savedBudget;
            gpu.UnregisterPersistentTensor(x);
            gpu.UnregisterPersistentTensor(y);
            gpuEngine.ClearFloat64Residents();
        }
    }

    [SkippableFact]
    public void Views_AndSelfOperands_OfAResidentResult_ReadTheRightValues()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        int rows = 1100;
        int columns = 1000;
        var x = new Tensor<double>(Values(rows * columns, 12), new[] { rows, columns });
        gpu.RegisterPersistentTensor(x, PersistentTensorRole.Weights);
        try
        {
            var sum = gpu.TensorAdd(x, x);
            Assert.True(gpuEngine.IsFloat64Resident(sum));
            var squared = gpu.TensorMultiply(sum, sum);
            Assert.True(gpuEngine.IsFloat64Resident(squared));
            Assert.Equal(new[] { rows, columns }, squared.Shape.ToArray());
            Assert.Equal((x[3, 4] + x[3, 4]) * (x[3, 4] + x[3, 4]), squared[3, 4]);

            // A reshape shares the resident result's vector and layout: still resident.
            var flat = sum.Reshape(new[] { rows * columns });
            var flatDoubled = gpu.TensorAdd(flat, flat);
            Assert.True(gpuEngine.IsFloat64Resident(flatDoubled));
            Assert.Equal(4.0 * x[7, 9], flatDoubled[7 * columns + 9]);

            // A permuted view shares the vector but not the layout: it must not be read as the resident buffer.
            var transposed = sum.Transpose(new[] { 1, 0 });
            var transposedSum = gpu.TensorAdd(transposed, transposed);
            Assert.Equal(new[] { columns, rows }, transposedSum.Shape.ToArray());
            Assert.Equal(4.0 * x[7, 9], transposedSum[9, 7]);
        }
        finally
        {
            gpu.UnregisterPersistentTensor(x);
        }
    }

    // --- Matrix multiply -------------------------------------------------------------------------------------------

    private static double[] HostMatMul(double[] a, double[] b, int m, int n, int k)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double x = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += x * b[p * n + j];
            }

        return c;
    }

    // The device accumulates in another order (and may fuse the multiply-add), so the products agree to rounding of
    // the sum of magnitudes, not bit for bit.
    private static void AssertProductClose(double[] a, double[] b, int m, int n, int k, ReadOnlySpan<double> actual)
    {
        var expected = HostMatMul(a, b, m, n, k);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double magnitude = 0.0;
                for (int p = 0; p < k; p++) magnitude += Math.Abs(a[i * k + p] * b[p * n + j]);
                double bound = 4.0 * k * 2.220446049250313e-16 * magnitude + 1e-300;
                Assert.True(Math.Abs(expected[i * n + j] - actual[i * n + j]) <= bound,
                    $"[{i},{j}] of {m}x{n} (k={k}): host {expected[i * n + j]:R}, device {actual[i * n + j]:R}, bound {bound:E2}.");
            }
    }

    [SkippableTheory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 5, 7)]
    [InlineData(33, 65, 17)]
    [InlineData(100, 257, 64)]
    [InlineData(64, 1, 300)]
    [InlineData(1, 300, 64)]
    public void MatrixMultiplyKernel_AgreesWithTheHost_ToDoubleRounding(int m, int n, int k)
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var backend = device.Value.Backend;
        Skip.If(backend is not IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true }, "No double matrix multiply on this device.");
        var multiply = (IGpuFloat64MatrixMultiply)backend;
        var left = Values(m * k, 50);
        var right = Values(k * n, 51);
        using var a = backend.AllocateBuffer(m * k, GpuElementType.Float64);
        using var b = backend.AllocateBuffer(k * n, GpuElementType.Float64);
        using var c = backend.AllocateBuffer(m * n, GpuElementType.Float64);
        backend.UploadBuffer<double>(a, left);
        backend.UploadBuffer<double>(b, right);
        multiply.MatrixMultiplyFloat64(a, b, c, m, n, k);
        var actual = new double[m * n];
        backend.DownloadBuffer<double>(c, actual);
        AssertProductClose(left, right, m, n, k, actual);
    }

    [SkippableFact]
    public void MatrixMultiplyKernel_KeepsThePrecisionAFloatMultiplyLoses_AndRefusesBadOperands()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var backend = device.Value.Backend;
        Skip.If(backend is not IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true }, "No double matrix multiply on this device.");
        var multiply = (IGpuFloat64MatrixMultiply)backend;
        // (1 + 2^-40)^2 = 1 + 2^-39 + 2^-80: the 2^-39 term is far below float's resolution and exact in double.
        double value = 1.0 + Math.Pow(2.0, -40);
        using var a = backend.AllocateBuffer(1, GpuElementType.Float64);
        using var c = backend.AllocateBuffer(1, GpuElementType.Float64);
        using var floats = backend.AllocateBuffer(4);
        backend.UploadBuffer<double>(a, new[] { value });
        using var b = backend.AllocateBuffer(1, GpuElementType.Float64);
        backend.UploadBuffer<double>(b, new[] { value });
        multiply.MatrixMultiplyFloat64(a, b, c, 1, 1, 1);
        var product = new double[1];
        backend.DownloadBuffer<double>(c, product);
        Assert.Equal(value * value, product[0]);
        Assert.NotEqual(1.0, product[0]);

        Assert.Throws<ArgumentException>(() => multiply.MatrixMultiplyFloat64(floats, b, c, 1, 1, 1));
        Assert.Throws<ArgumentException>(() => multiply.MatrixMultiplyFloat64(a, b, a, 1, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => multiply.MatrixMultiplyFloat64(a, b, c, 2, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => multiply.MatrixMultiplyFloat64(a, b, c, 0, 1, 1));
    }

    [SkippableFact]
    public void TensorMatMul_OnTheDevice_LeavesTheProductThere_ForRank2_AndLeadingBatchDimensions()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        Skip.If(device.Value.Backend is not IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true }, "No double matrix multiply on this device.");
        var gpu = (IEngine)gpuEngine;
        bool? saved = DirectGpuTensorEngine.Float64MatMulForceDevice;
        try
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = true;
            int m = 37, k = 29, n = 41;
            var left = Values(m * k, 60);
            var right = Values(k * n, 61);
            var a = new Tensor<double>(left, new[] { m, k });
            var b = new Tensor<double>(right, new[] { k, n });
            long readbacksBefore = GpuLaunchProbe.Readbacks;
            var product = gpu.TensorMatMul(a, b);
            Assert.True(gpuEngine.IsFloat64Resident(product), "The product is not resident on the device.");
            Assert.Equal(readbacksBefore, GpuLaunchProbe.Readbacks);
            Assert.Equal(new[] { m, n }, product.Shape.ToArray());
            AssertProductClose(left, right, m, n, k, product.AsSpan());

            // [2, 3, k] x [k, n]: the leading dimensions are rows of one multiply.
            var batchedLeft = Values(2 * 3 * k, 62);
            var batched = new Tensor<double>(batchedLeft, new[] { 2, 3, k });
            var batchedProduct = gpu.TensorMatMul(batched, b);
            Assert.True(gpuEngine.IsFloat64Resident(batchedProduct));
            Assert.Equal(new[] { 2, 3, n }, batchedProduct.Shape.ToArray());
            AssertProductClose(batchedLeft, right, 6, n, k, batchedProduct.AsSpan());

            Assert.Throws<ArgumentException>(() => gpu.TensorMatMul(a, a));
        }
        finally
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = saved;
        }
    }

    [SkippableFact]
    public void TensorMatMul_ChoosesARouteByMeasuring_AndIsCorrectOnEither()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        var gpuEngine = device.Value.Engine;
        var gpu = (IEngine)gpuEngine;
        bool? saved = DirectGpuTensorEngine.Float64MatMulForceDevice;
        try
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = null;
            // Below the floor: the CPU engine, no measurement.
            var small = new Tensor<double>(Values(16 * 16, 70), new[] { 16, 16 });
            Assert.False(gpuEngine.IsFloat64Resident(gpu.TensorMatMul(small, small)));

            // Above it the first call of the size class measures both routes and every later call uses the winner:
            // which one is this machine's business, the product is not.
            int size = 336; // 336^3 is just over 2^25
            Assert.True((long)size * size * size >= DirectGpuTensorEngine.Float64MatMulMinWork);
            var left = Values(size * size, 71);
            var right = Values(size * size, 72);
            var a = new Tensor<double>(left, new[] { size, size });
            var b = new Tensor<double>(right, new[] { size, size });
            var first = gpu.TensorMatMul(a, b);
            var second = gpu.TensorMatMul(a, b);
            Assert.Equal(gpuEngine.IsFloat64Resident(first), gpuEngine.IsFloat64Resident(second));
            AssertProductClose(left, right, size, size, size, first.AsSpan());
            AssertProductClose(left, right, size, size, size, second.AsSpan());

            DirectGpuTensorEngine.Float64MatMulForceDevice = false;
            Assert.False(gpuEngine.IsFloat64Resident(gpu.TensorMatMul(a, b)));
        }
        finally
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = saved;
        }
    }

    [SkippableFact]
    public void TensorMatMul_RecordsItsGradient_WhenTheProductIsOnTheDevice()
    {
        var device = Device();
        Skip.If(device is null, "No device with double-precision kernels on this host.");
        Skip.If(device.Value.Backend is not IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true }, "No double matrix multiply on this device.");
        var gpu = (IEngine)device.Value.Engine;
        var cpu = (IEngine)new CpuEngine();
        bool? saved = DirectGpuTensorEngine.Float64MatMulForceDevice;
        try
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = true;
            int m = 9, k = 7, n = 5;
            var a = new Tensor<double>(Values(m * k, 80), new[] { m, k });
            var b = new Tensor<double>(Values(k * n, 81), new[] { k, n });

            Tensor<double> Gradient(IEngine engine)
            {
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
                var product = engine.TensorMatMul(a, b);
                var loss = engine.ReduceSum(product, null);
                return tape.ComputeGradients(loss, new[] { a })[a];
            }

            var expected = Gradient(cpu).AsSpan();
            var actual = Gradient(gpu).AsSpan();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12 * (1.0 + Math.Abs(expected[i])),
                    $"Gradient {i}: cpu {expected[i]:R}, device route {actual[i]:R}.");
        }
        finally
        {
            DirectGpuTensorEngine.Float64MatMulForceDevice = saved;
        }
    }
}
