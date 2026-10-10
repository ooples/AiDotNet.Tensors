using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Double scalar reductions (TensorSum / TensorMaxValue / TensorMinValue) on the CPU engine: exact
/// values across the 32K-element parallel chunk edges, and bit-identical results for any thread count.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class CpuDoubleReductionTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(1000)]
    [InlineData(32 * 1024 - 1)]
    [InlineData(32 * 1024)]
    [InlineData(32 * 1024 + 1)]
    [InlineData(64 * 1024)]
    [InlineData(100_000)]
    [InlineData(1_000_003)]
    public void SumMaxMin_MatchReference(int length)
    {
        var data = Data(length, seed: length);
        var t = new Tensor<double>((double[])data.Clone(), new[] { length });
        IEngine cpu = new CpuEngine();

        double max = double.NegativeInfinity, min = double.PositiveInfinity;
        decimal exact = 0m;
        foreach (var v in data) { max = Math.Max(max, v); min = Math.Min(min, v); exact += (decimal)v; }

        Assert.Equal(max, cpu.TensorMaxValue(t));
        Assert.Equal(min, cpu.TensorMinValue(t));
        // |values| < 50 and length <= ~1M: fp64 accumulation error is around 1e-8 at most.
        Assert.Equal((double)exact, cpu.TensorSum(t), 1e-6);
    }

    [Theory]
    [InlineData(100_000)]
    [InlineData(1_000_003)]
    public void Sum_IsBitIdentical_AcrossThreadCounts(int length)
    {
        var t = new Tensor<double>(Data(length, seed: 7), new[] { length });
        IEngine cpu = new CpuEngine();
        int original = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            double serial = cpu.TensorSum(t);
            foreach (int threads in new[] { 2, 3, Math.Max(4, Environment.ProcessorCount) })
            {
                CpuParallelSettings.MaxDegreeOfParallelism = threads;
                Assert.Equal(BitConverter.DoubleToInt64Bits(serial), BitConverter.DoubleToInt64Bits(cpu.TensorSum(t)));
            }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = original;
        }
    }

    internal static double[] Data(int length, int seed)
    {
        var rnd = new Random(seed);
        var data = new double[length];
        for (int i = 0; i < length; i++) data[i] = rnd.NextDouble() * 100 - 50;
        return data;
    }
}

#if NET6_0_OR_GREATER
/// <summary>
/// DirectGpuTensorEngine scalar reductions run where the data lives: a host-resident tensor is reduced
/// on the CPU (no upload + synchronize for one number), and double reductions keep fp64 instead of the
/// backend's fp32 kernels. Calls go through <see cref="IEngine"/> because these are explicit interface
/// implementations; a call on a DirectGpuTensorEngine-typed variable would bind to the CPU base.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuEngineScalarReductionDispatchTests
{
    [SkippableFact]
    public void HostResidentDouble_ReductionsMatchCpuFp64Exactly()
    {
        // The constructor reports a missing backend through IsGpuAvailable rather than throwing, so a constructor
        // exception is a real failure and is left to fail the test.
        var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        try
        {
            const int length = 100_000;
            // Values with wide dynamic range: an fp32 accumulation visibly diverges from fp64.
            var data = CpuDoubleReductionTests.Data(length, seed: 11);
            for (int i = 0; i < length; i += 3) data[i] *= 1e6;
            var t = new Tensor<double>(data, new[] { length });
            IEngine gpuEngine = gpu;
            IEngine cpu = new CpuEngine();

            Assert.Equal(cpu.TensorSum(t), gpuEngine.TensorSum(t));
            Assert.Equal(cpu.TensorMaxValue(t), gpuEngine.TensorMaxValue(t));
            Assert.Equal(cpu.TensorMinValue(t), gpuEngine.TensorMinValue(t));
            Assert.Equal(cpu.TensorMean(t), gpuEngine.TensorMean(t));
        }
        finally
        {
            gpu.Dispose();
        }
    }

    [SkippableFact]
    public void DeviceResidentFloat_ReductionsRunOnTheDeviceAndMatchCpu()
    {
        var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        try
        {
            const int length = 4096;
            var data = new float[length];
            var rnd = new Random(5);
            for (int i = 0; i < length; i++) data[i] = (float)(rnd.NextDouble() * 10 - 5);
            var host = new Tensor<float>((float[])data.Clone(), new[] { length });
            var resident = gpu.UploadToGpu(new Tensor<float>((float[])data.Clone(), new[] { length }),
                AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.General);
            Assert.True(resident.IsGpuResident, "The uploaded tensor should be device-resident.");
            IEngine gpuEngine = gpu;
            IEngine cpu = new CpuEngine();

            // A device-resident tensor reduces on the device, whose summation order differs from the CPU's, so the
            // sum is compared with a tolerance; max and min are order-independent and must match exactly.
            float cpuSum = cpu.TensorSum(host);
            float sum, max, min;
            long fullTensorBytes = (long)length * sizeof(float);
            using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true))
            {
                sum = gpuEngine.TensorSum(resident);
                max = gpuEngine.TensorMaxValue(resident);
                min = gpuEngine.TensorMinValue(resident);

                // A device reduction downloads only its scalar. A CPU fallback would have to pull the whole tensor
                // back first, so no single download may be as large as the input.
                foreach (var transfer in scope.Events)
                {
                    Assert.False(transfer.Kind == AiDotNet.Tensors.Engines.Diagnostics.GpuTransferKind.DeviceToHost
                                 && transfer.Bytes >= fullTensorBytes,
                        $"A reduction downloaded {transfer.Bytes} bytes ({transfer.Operation}): it ran on the CPU.");
                }
            }

            // Positive control: pulling the tensor to the host is a full-size download, and the same check sees it.
            using (var control = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin())
            {
                _ = resident.ToArray();
                Assert.True(control.BytesDownloaded >= fullTensorBytes,
                    $"The residency scope did not record a full download ({control.BytesDownloaded} bytes); the check above proves nothing on this backend.");
            }

            Assert.Equal(cpuSum, sum, 1e-2f);
            Assert.Equal(cpu.TensorMaxValue(host), max);
            Assert.Equal(cpu.TensorMinValue(host), min);
        }
        finally
        {
            gpu.Dispose();
        }
    }

    [SkippableFact]
    public void HostResidentFloat_ReductionsMatchCpu()
    {
        // The constructor reports a missing backend through IsGpuAvailable rather than throwing, so a constructor
        // exception is a real failure and is left to fail the test.
        var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        try
        {
            const int length = 4096;
            var data = new float[length];
            var rnd = new Random(3);
            for (int i = 0; i < length; i++) data[i] = (float)(rnd.NextDouble() * 10 - 5);
            var t = new Tensor<float>(data, new[] { length });
            IEngine gpuEngine = gpu;
            IEngine cpu = new CpuEngine();

            // A host-resident tensor reduces on the CPU, so the results are identical, not merely close.
            Assert.Equal(cpu.TensorSum(t), gpuEngine.TensorSum(t));
            Assert.Equal(cpu.TensorMaxValue(t), gpuEngine.TensorMaxValue(t));
            Assert.Equal(cpu.TensorMinValue(t), gpuEngine.TensorMinValue(t));
        }
        finally
        {
            gpu.Dispose();
        }
    }
}
#endif
