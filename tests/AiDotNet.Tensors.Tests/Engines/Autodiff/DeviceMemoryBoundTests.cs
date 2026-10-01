using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Runs alone: it reads process-wide device allocation counters, which concurrent GPU tests also move.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DeviceMemoryAccountingCollection
{
    public const string Name = "DeviceMemoryAccountingSerial";
}

[Collection(DeviceMemoryAccountingCollection.Name)]
public sealed class DeviceMemoryBoundTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public DeviceMemoryBoundTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>
    /// A warm training loop's device memory stays bounded. Dead results return to the device pool when the GC
    /// finalizes them, and the backends collect after every <see cref="DeviceMemoryReclaim.CollectAfterDriverBytes"/>
    /// of driver allocation, so memory can grow by about that much between collections and no more. Before: the pool
    /// kept four buffers per size and sent the rest to the driver behind completion markers, collections waited on
    /// managed allocation, and this loop swung between 20 and 330 MB with a ~10 MB working set. Checked with the
    /// OpenCL allocation counters (GpuKernelDiagnostics), which the other backends do not maintain.
    /// </summary>
    [SkippableFact]
    public void SteadyTrainingLoop_KeepsDeviceMemoryBounded()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend,
            "the device allocation counters are maintained by the OpenCL backend.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            const int WarmupSteps = 100, MeasuredSteps = 300;
            const float LearningRate = 1e-3f;
            // Large enough that each step leaves several megabytes of dead results for the GC to find.
            var x = gpu.UploadToGpu(Rand(new[] { 256, 512 }, 1), AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.General);
            var w1 = gpu.UploadToGpu(Rand(new[] { 512, 1024 }, 2), AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.General);
            var w2 = gpu.UploadToGpu(Rand(new[] { 1024, 256 }, 3), AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.General);
            void Step()
            {
                Dictionary<Tensor<float>, Tensor<float>> grads;
                using (var tape = new GradientTape<float>())
                {
                    var hidden = gpu.TensorTanh(gpu.TensorMatMul(x, w1));
                    var output = gpu.TensorMatMul(hidden, w2);
                    var loss = gpu.ReduceMean(gpu.TensorMultiply(output, output), null, false);
                    grads = tape.ComputeGradients(loss, new[] { w1, w2 });
                }
                gpu.TensorSubtractInPlace(w1, gpu.TensorMultiplyScalar(grads[w1], LearningRate));
                gpu.TensorSubtractInPlace(w2, gpu.TensorMultiplyScalar(grads[w2], LearningRate));
            }
            for (int i = 0; i < WarmupSteps; i++) Step();
            long warmBytes = GpuKernelDiagnostics.LiveBufferBytes;
            long allocatedBefore = GpuKernelDiagnostics.TotalBuffersAllocated;
            long peakBytes = warmBytes;
            for (int i = 0; i < MeasuredSteps; i++)
            {
                Step();
                peakBytes = Math.Max(peakBytes, GpuKernelDiagnostics.LiveBufferBytes);
            }
            // Measured on OpenCL before the change: +257 MB and 3,900 driver allocations over these 300 steps; after: none.
            long allowedGrowth = DeviceMemoryReclaim.CollectAfterDriverBytes;
            long newAllocations = GpuKernelDiagnostics.TotalBuffersAllocated - allocatedBefore;
            Assert.True(peakBytes - warmBytes <= allowedGrowth,
                $"device memory grew {(peakBytes - warmBytes) / 1048576.0:F1} MB over {MeasuredSteps} warm steps (allowed "
                + $"{allowedGrowth / 1048576.0:F0} MB), with {newAllocations} driver allocations; "
                + GpuKernelDiagnostics.DescribeBufferResidency());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
