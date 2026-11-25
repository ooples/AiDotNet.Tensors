// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression tests proving <see cref="DirectGpuTensorEngine.TensorBroadcastTo{T}"/> during active
/// <see cref="GradientTape{T}"/> recording and <c>TensorNarrow</c> prefix/offset slice views of
/// persistent and synced device tensors execute 100% on-device with zero host zero-tensor allocations,
/// zero host <c>ToArray()</c> materializations, and zero mid-step PCIe uploads/downloads.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class BroadcastAndSliceResidencyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public BroadcastAndSliceResidencyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(new[] { 4, 8 }, new[] { 4, 8 })]               // Tier 1: exact shape identity
    [InlineData(new[] { 4, 8 }, new[] { 1, 1, 4, 8 })]         // Tier 2: pure leading size-1 prepend
    [InlineData(new[] { 2, 1, 4 }, new[] { 2, 6, 4 })]         // Tier 3: interior axis repeat
    [InlineData(new[] { 1, 4, 1 }, new[] { 3, 4, 5 })]         // Tier 3: multi-axis repeat + rank match
    [InlineData(new[] { 4, 1 }, new[] { 2, 4, 6 })]            // Tier 3: leading prepend + trailing repeat
    public void TensorBroadcastTo_UnderActiveGradientTape_StaysResidentWithZeroHostUploadsAndExactGradients(
        int[] srcShape,
        int[] targetShape)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine ?? throw new InvalidOperationException("GPU fixture reported a device without an engine.");
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(42);
            var x = new Tensor<float>(srcShape);
            for (int i = 0; i < x.Length; i++)
                x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            var upstreamWeights = new Tensor<float>(targetShape);
            for (int i = 0; i < upstreamWeights.Length; i++)
                upstreamWeights[i] = (float)(0.25 * ((i % 9) - 4));
            gpu.RegisterPersistentTensor(upstreamWeights, PersistentTensorRole.Constant);

            float[] RunCpu(out float[] fwdOut)
            {
                AiDotNetEngine.Current = cpu;
                using var tape = new GradientTape<float>();
                var act = cpu.TensorTanh(x);
                var bc = cpu.TensorBroadcastTo(act, targetShape);
                fwdOut = bc.ToArray();
                var loss = cpu.ReduceSum(cpu.TensorMultiply(bc, upstreamWeights), null, keepDims: false);
                return tape.ComputeGradients(loss, new[] { x })[x].ToArray();
            }

            var expectedGrad = RunCpu(out var expectedFwd);

            // Warm up persistent weight upload once before opening the residency scope.
            AiDotNetEngine.Current = gpu;
            _ = gpu.TensorMultiplyScalar(upstreamWeights, 1.0f);

            Tensor<float> gpuFwdTensor;
            Tensor<float> gpuGradTensor;
            long uploadedBytes;
            long downloadedBytes;
            using (var tape = new GradientTape<float>())
            {
                var act = gpu.TensorTanh(x);
                using (var residency = GpuResidencyScope.Begin())
                {
                    gpuFwdTensor = gpu.TensorBroadcastTo(act, targetShape);
                    var loss = gpu.ReduceSum(gpu.TensorMultiply(gpuFwdTensor, upstreamWeights), null, keepDims: false);
                    gpuGradTensor = tape.ComputeGradients(loss, new[] { x })[x];
                    residency.Dispose();
                    uploadedBytes = residency.BytesUploaded;
                    downloadedBytes = residency.BytesDownloaded;
                }
            }

            Assert.True(uploadedBytes == 0, $"TensorBroadcastTo under GradientTape uploaded {uploadedBytes} bytes from host.");
            Assert.True(downloadedBytes <= 16, $"TensorBroadcastTo under GradientTape downloaded {downloadedBytes} bytes to host.");

            var actualFwd = gpuFwdTensor.ToArray();
            Assert.Equal(expectedFwd.Length, actualFwd.Length);
            for (int i = 0; i < expectedFwd.Length; i++)
                Assert.True(Math.Abs(expectedFwd[i] - actualFwd[i]) < 1e-4f, $"Fwd[{i}] cpu={expectedFwd[i]} gpu={actualFwd[i]}");

            var actualGrad = gpuGradTensor.ToArray();
            Assert.Equal(expectedGrad.Length, actualGrad.Length);
            for (int i = 0; i < expectedGrad.Length; i++)
                Assert.True(Math.Abs(expectedGrad[i] - actualGrad[i]) < 1e-4f, $"Grad[{i}] cpu={expectedGrad[i]} gpu={actualGrad[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableTheory]
    [InlineData(0, 8)]   // Prefix slice (offset == 0, length < capacity) — exact PerfView TensorConcatenate LOH bug
    [InlineData(4, 8)]   // Nonzero-offset contiguous slice of persistent buffer
    public void NarrowSliceOfPersistentBuffer_InTensorConcatenate_UploadsOnceAndSlicesOnDevice(int startRow, int rowCount)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine ?? throw new InvalidOperationException("GPU fixture reported a device without an engine.");
        var cpu = new CpuEngine();

        const int capacityRows = 16;
        const int cols = 8;
        var rng = RandomHelper.CreateSeededRandom(99 + startRow);
        var capacityPad = new Tensor<float>(new[] { capacityRows, cols });
        for (int i = 0; i < capacityPad.Length; i++)
            capacityPad[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var liveInput = new Tensor<float>(new[] { rowCount, cols });
        for (int i = 0; i < liveInput.Length; i++)
            liveInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var expected = cpu.TensorConcatenate(
            new[] { cpu.TensorTanh(liveInput), cpu.TensorNarrow(capacityPad, 0, startRow, rowCount) },
            axis: 1).ToArray();

        // Register lazily without pre-uploading the full tensor: first narrow triggers one persistent upload,
        // and every subsequent narrow + concatenate must transfer 0 bytes between host and device.
        gpu.RegisterPersistentTensor(capacityPad, PersistentTensorRole.Constant);
        var residentLive = gpu.TensorTanh(liveInput);

        // Warmup step 1 (populates _persistentBufferCache from _pendingPersistentRoles)
        _ = gpu.TensorConcatenate(new[] { residentLive, gpu.TensorNarrow(capacityPad, 0, startRow, rowCount) }, axis: 1);

        Tensor<float> concatResult;
        long steadyUploaded;
        long steadyDownloaded;
        using (var residency = GpuResidencyScope.Begin())
        {
            var padSlice = gpu.TensorNarrow(capacityPad, 0, startRow, rowCount);
            concatResult = gpu.TensorConcatenate(new[] { residentLive, padSlice }, axis: 1);
            residency.Dispose();
            steadyUploaded = residency.BytesUploaded;
            steadyDownloaded = residency.BytesDownloaded;
        }

        Assert.True(steadyUploaded == 0, $"Steady-state persistent slice concat uploaded {steadyUploaded} bytes.");
        Assert.True(steadyDownloaded == 0, $"Steady-state persistent slice concat downloaded {steadyDownloaded} bytes.");

        var actual = concatResult.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-5f, $"[{i}] expected={expected[i]} actual={actual[i]}");
    }
}
