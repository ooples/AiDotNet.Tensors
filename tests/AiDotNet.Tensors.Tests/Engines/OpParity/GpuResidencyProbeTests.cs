// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). EMPIRICAL GPU-residency guard.
//
// The reflection guard (BackendCompletenessTests) reports which ops lack a DEDICATED GPU override —
// a SUPERSET, because a composed op (e.g. TensorSquare -> TensorMultiply) has no override of its own
// yet still runs on the GPU through its primitives. That metric can only OVER-count gaps, never prove
// one covered. This guard does the opposite with a MEASUREMENT, not a guess: it runs every registry
// op on the GPU engine while GpuLaunchProbe counts real kernel dispatches, and flags any op that fires
// ZERO kernels. A zero-launch op computed entirely on the host when the caller asked for the GPU
// engine — the TRUE coverage gap. Projected cases require an additional dispatch so that the
// projection kernel cannot hide a host-computed result from the operation under test.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

[Collection("OpParity")]
public sealed class GpuResidencyProbeTests
{
    private readonly OpParityFixture _fx;
    public GpuResidencyProbeTests(OpParityFixture fx) => _fx = fx;

    // Registry ops that execute fewer GPU kernels than their case requires — i.e. silently run the
    // operation on the CPU, even if a result-projection helper subsequently launches a GPU kernel.
    // GOAL 0. Ratchets DOWN only: add a kernel / GPU-primitive composition to lower it. Never raise it
    // to hide a regression. Measured in isolation (see class remarks); parallel runs can only observe
    // an equal-or-lower count (a stray cross-collection launch never manufactures a fallback), so the
    // <= assertion cannot false-fail.
    private const int CpuFallbackFloor = 0;

    // Ops whose GPU override throws kernel-not-found and silently falls back to the CPU (a hollow
    // override — reflection counts it covered, the probe proves it does zero GPU work). GOAL 0.
    // Ratchets DOWN only: register/write the missing kernel or compose from a working primitive.
    private const int HollowOverrideFloor = 0;

    [SkippableFact]
    public void EveryRegistryOp_ActuallyLaunchesAGpuKernel()
    {
        if (!_fx.GpuReady)
        {
            if (_fx.RequireGpu)
                throw new InvalidOperationException(
                    "GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", _fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        GpuLaunchProbe.CaptureReadbackSites = true;
        var fallback = new SortedSet<string>(StringComparer.Ordinal); // ran on CPU (0 launches)
        var metadata = new SortedSet<string>(StringComparer.Ordinal); // verified zero-compute views/identity ops
        var hostContracts = new SortedSet<string>(StringComparer.Ordinal); // managed delegate/codec boundaries
        var covered = new SortedSet<string>(StringComparer.Ordinal);  // launched >= 1 kernel
        var errored = new SortedSet<string>(StringComparer.Ordinal);  // threw (can't classify)
        var readbacks = new SortedSet<string>(StringComparer.Ordinal); // downloaded values before materialization
        // Of the fallbacks: HOLLOW = 0 launches AND a kernel-not-found miss (a GPU override asked for an
        // unregistered kernel and silently fell to the CPU). These are BUGS. The rest (0 launches, no miss)
        // are legitimately zero-compute views (reshape/permute/narrow) or CPU-guarded paths.
        var hollow = new SortedSet<string>(StringComparer.Ordinal);

        foreach (var op in OpParityRegistry.All())
        {
            // A quarantined op makes no residency claim (its GPU path is skipped/known-broken).
            if (op.KnownDivergence is not null || op.GpuUnsafe) continue;

            GpuLaunchProbe.Reset();
            long launches;
            long internalReadbacks;
            long internalReadbackBytes;
            string[] internalReadbackSites;
            try
            {
                using var result = op.RunFloat(gpu);
                launches = GpuLaunchProbe.Count;
                internalReadbacks = GpuLaunchProbe.Readbacks;
                internalReadbackBytes = GpuLaunchProbe.ReadbackBytes;
                internalReadbackSites = GpuLaunchProbe.ReadbackSites;
                _ = result.ToArray();
            }
            catch (Exception ex)
            {
                errored.Add($"{op.Name}\t{ex.ToString().Replace(Environment.NewLine, " | ")}");
                continue;
            }

            if (internalReadbacks > op.GpuAllowedReadbacks)
                readbacks.Add($"{op.Name}\t{op.Category}\t{internalReadbacks} transfers\t" +
                    $"{internalReadbackBytes} bytes\tallowance {op.GpuAllowedReadbacks}\t" +
                    string.Join("; ", internalReadbackSites));

            if (op.GpuProbeExpectation == GpuProbeExpectation.HostContract)
            {
                Assert.False(string.IsNullOrWhiteSpace(op.GpuHostContractReason),
                    $"{op.Name} is marked HostContract without a contract-level reason.");
                hostContracts.Add($"{op.Name}\t{op.Category}\t{op.GpuHostContractReason}");
                continue;
            }
            Assert.True(op.GpuMinimumKernelLaunches > 0,
                $"{op.Name} has an invalid minimum GPU kernel launch count: {op.GpuMinimumKernelLaunches}.");
            if (launches >= op.GpuMinimumKernelLaunches) { covered.Add(op.Name); continue; }
            if (op.GpuProbeExpectation == GpuProbeExpectation.MetadataOnly)
            {
                metadata.Add($"{op.Name}\t{op.Category}");
                continue;
            }
            fallback.Add($"{op.Name}\t{op.Category}\tlaunches {launches}/{op.GpuMinimumKernelLaunches}");
            if (GpuLaunchProbe.KernelMisses > 0)
                hollow.Add($"{op.Name}\t{op.Category}\tmissing-kernel: {string.Join(",", GpuLaunchProbe.MissedKernelNames)}");
        }

        WriteReport(covered, metadata, hostContracts, fallback, errored, hollow, readbacks);
        GpuLaunchProbe.CaptureReadbackSites = false;

        Assert.True(fallback.Count <= CpuFallbackFloor,
            $"{fallback.Count} registry ops dispatched fewer GPU kernels than required (floor " +
            $"{CpuFallbackFloor}) — the operation may have run on the CPU when the caller selected the GPU engine. " +
            $"See gpu-cpu-fallback.md for the list. Write a kernel or GPU-primitive composition to move " +
            $"them on-device; do not raise the floor.");

        Assert.True(hollow.Count <= HollowOverrideFloor,
            $"{hollow.Count} ops have a GPU override that threw kernel-not-found and SILENTLY fell back to " +
            $"the CPU (floor {HollowOverrideFloor}) — reflection thinks they are GPU-covered but they do zero " +
            $"GPU work. See gpu-hollow-overrides.md (lists the missing kernel per op). Register/write the " +
            $"kernel or compose from a working primitive; do not raise the floor.");

        Assert.Empty(readbacks);
    }

    [SkippableFact]
    public void MetadataViews_PreserveResidentStorageWithoutLaunchingKernels()
    {
        if (!_fx.GpuReady)
        {
            if (_fx.RequireGpu)
                throw new InvalidOperationException(
                    "GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", _fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var backend = gpu.TestBackend!;
        var cases = new (string Name, int[] Shape, Func<Tensor<float>, Tensor<float>> Run, float[] Expected)[]
        {
            ("Reshape", new[] { 2, 3 }, x => gpu.Reshape(x, new[] { 6 }), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("AtLeast1D", new[] { 6 }, x => gpu.TensorAtLeast1D(x), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("AtLeast2D", new[] { 6 }, x => gpu.TensorAtLeast2D(x), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("AtLeast3D", new[] { 6 }, x => gpu.TensorAtLeast3D(x), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("ExpandDims", new[] { 2, 3 }, x => gpu.TensorExpandDims(x, 1), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("Flatten", new[] { 2, 3 }, x => gpu.TensorFlatten(x), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("Squeeze", new[] { 2, 1, 3 }, x => gpu.TensorSqueeze(x, 1), new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
            ("Permute", new[] { 2, 3 }, x => gpu.TensorPermute(x, new[] { 1, 0 }), new[] { 1f, 4f, 2f, 5f, 3f, 6f }),
            ("MoveDim", new[] { 2, 3 }, x => gpu.TensorMoveDim(x, 0, 1), new[] { 1f, 4f, 2f, 5f, 3f, 6f }),
            ("SwapAxes", new[] { 2, 3 }, x => gpu.TensorSwapAxes(x, 0, 1), new[] { 1f, 4f, 2f, 5f, 3f, 6f }),
            ("ReorderToNchw", new[] { 1, 1, 2, 3 }, x => gpu.ReorderToNchw(x), new[] { 1f, 2f, 3f, 4f, 5f, 6f })
        };

        foreach (var c in cases)
        {
            using var source = Tensor<float>.FromGpuBuffer(
                backend,
                backend.AllocateBuffer(new[] { 1f, 2f, 3f, 4f, 5f, 6f }),
                c.Shape,
                ownsBuffer: true);
            Assert.True(source.IsGpuResident, $"{c.Name}: source was not resident");
            var sourceBuffer = source.TryGetGpuBuffer();
            Assert.NotNull(sourceBuffer);

            GpuLaunchProbe.Reset();
            using var view = c.Run(source);
            Assert.Equal(0, GpuLaunchProbe.Count);
            Assert.True(view.IsGpuResident, $"{c.Name}: result lost residency");
            Assert.Same(sourceBuffer, view.TryGetGpuBuffer());
            Assert.Equal(c.Expected, view.ToArray());
        }
    }

    [SkippableFact]
    public void MuLawEncodeDecodeChain_KeepsIntegerCodesResident()
    {
        if (!_fx.GpuReady)
        {
            if (_fx.RequireGpu)
                throw new InvalidOperationException(
                    "GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", _fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend available.");
            return;
        }

        var gpu = _fx.Gpu!;
        using var source = new Tensor<float>(
            new[] { -1f, -0.75f, -0.1f, 0f, 0.1f, 0.75f, 1f }, new[] { 7 });
        using var encoded = gpu.MuLawEncoding(source, 256);
        Assert.True(encoded.IsGpuResident, "MuLawEncoding did not leave its integer output resident.");

        GpuLaunchProbe.Reset();
        using var decoded = gpu.MuLawDecoding<float>(encoded, 256);

        Assert.True(decoded.IsGpuResident, "MuLawDecoding did not leave its output resident.");
        Assert.True(GpuLaunchProbe.Count > 0, "MuLawDecoding launched no GPU work.");
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        float[] values = decoded.ToArray();
        Assert.All(values, value => Assert.True(float.IsFinite(value)));
    }

    [SkippableFact]
    public void AudioElementwiseChain_KeepsOutputsResident()
    {
        if (!_fx.GpuReady)
        {
            if (_fx.RequireGpu)
                throw new InvalidOperationException(
                    "GPU required (AIDOTNET_REQUIRE_GPU_TESTS=1) but no DirectGpu backend is available.", _fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend available.");
            return;
        }

        var gpu = _fx.Gpu!;
        using var source = new Tensor<float>(
            new[] { 0.1f, 0.2f, 0.4f, 0.8f, 0.7f, 0.5f, 0.3f, 0.15f }, new[] { 1, 8 });

        GpuLaunchProbe.Reset();
        using var decibels = gpu.AmplitudeToDB(source);
        using var deltas = gpu.ComputeDeltas(decibels, 5);

        Assert.True(decibels.IsGpuResident, "AmplitudeToDB did not leave its output resident.");
        Assert.True(deltas.IsGpuResident, "ComputeDeltas did not leave its output resident.");
        Assert.True(GpuLaunchProbe.Count >= 2, "The audio chain did not launch both GPU kernels.");
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        float[] values = deltas.ToArray();
        Assert.All(values, value => Assert.True(float.IsFinite(value)));
    }

    [SkippableFact]
    public void OwnedResidentBuffer_OutlivesItsSourceView()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var backend = _fx.Gpu!.TestBackend!;
        var source = Tensor<float>.FromGpuBuffer(
            backend, backend.AllocateBuffer(new[] { 1f, 2f, 3f, 4f }), new[] { 2, 2 }, ownsBuffer: true);
        using var view = source.Reshape(4);
        source.Dispose();
        Assert.True(view.IsGpuResident);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, view.ToArray());
    }

    [SkippableFact]
    public void TensorIndexPooling_ForwardAndBackward_StayResidentAndMatchCpu()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var cpu = new CpuEngine();

        var input2 = new Tensor<float>(
            Enumerable.Range(0, 48).Select(i => (float)((i * 17) % 31 - 15)).ToArray(),
            new[] { 1, 2, 4, 6 });
        var grad2 = new Tensor<float>(
            Enumerable.Range(1, 12).Select(i => i * 0.125f).ToArray(),
            new[] { 1, 2, 2, 3 });
        var shape2 = new[] { 1, 2, 4, 6 };
        var pool2 = new[] { 2, 2 };

        GpuLaunchProbe.Reset();
        using var gpuOut2 = gpu.MaxPool2DWithTensorIndices(input2, pool2, pool2, out var gpuIdx2);
        using var gpuGrad2 = gpu.MaxPool2DBackwardWithTensorIndices(grad2, gpuIdx2, shape2, pool2, pool2);
        Assert.True(gpuOut2.IsGpuResident);
        Assert.True(gpuIdx2.IsGpuResident);
        Assert.True(gpuGrad2.IsGpuResident);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        using var cpuOut2 = cpu.MaxPool2DWithTensorIndices(input2, pool2, pool2, out var cpuIdx2);
        using var cpuGrad2 = cpu.MaxPool2DBackwardWithTensorIndices(grad2, cpuIdx2, shape2, pool2, pool2);
        Assert.Equal(cpuOut2.ToArray(), gpuOut2.ToArray());
        Assert.Equal(cpuIdx2.ToArray(), gpuIdx2.ToArray());
        Assert.Equal(cpuGrad2.ToArray(), gpuGrad2.ToArray());

        var input3 = new Tensor<float>(
            Enumerable.Range(0, 64).Select(i => (float)((i * 13) % 37 - 18)).ToArray(),
            new[] { 1, 1, 4, 4, 4 });
        var grad3 = new Tensor<float>(
            Enumerable.Range(1, 8).Select(i => i * -0.25f).ToArray(),
            new[] { 1, 1, 2, 2, 2 });
        var shape3 = new[] { 1, 1, 4, 4, 4 };
        var pool3 = new[] { 2, 2, 2 };

        GpuLaunchProbe.Reset();
        using var gpuOut3 = gpu.MaxPool3DWithTensorIndices(input3, pool3, pool3, out var gpuIdx3);
        using var gpuGrad3 = gpu.MaxPool3DBackwardWithTensorIndices(grad3, gpuIdx3, shape3, pool3, pool3);
        Assert.True(gpuOut3.IsGpuResident);
        Assert.True(gpuIdx3.IsGpuResident);
        Assert.True(gpuGrad3.IsGpuResident);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        using var cpuOut3 = cpu.MaxPool3DWithTensorIndices(input3, pool3, pool3, out var cpuIdx3);
        using var cpuGrad3 = cpu.MaxPool3DBackwardWithTensorIndices(grad3, cpuIdx3, shape3, pool3, pool3);
        Assert.Equal(cpuOut3.ToArray(), gpuOut3.ToArray());
        Assert.Equal(cpuIdx3.ToArray(), gpuIdx3.ToArray());
        Assert.Equal(cpuGrad3.ToArray(), gpuGrad3.ToArray());

        gpuIdx2.Dispose();
        cpuIdx2.Dispose();
        gpuIdx3.Dispose();
        cpuIdx3.Dispose();
    }

    [SkippableFact]
    public void TensorIndexReduceMax_ForwardAndBackward_StayResidentAndMatchCpu()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var cpu = new CpuEngine();
        var inputData = Enumerable.Range(0, 24).Select(i => -100f - i).ToArray();
        inputData[1] = 10f;
        inputData[18] = 20f;
        inputData[11] = 30f;
        using var input = new Tensor<float>(inputData, new[] { 2, 3, 4 });
        using var gradOutput = new Tensor<float>(
            new[] { 1.25f, -2.5f, 4f }, new[] { 3 });
        var axes = new[] { 0, 2 };
        var inputShape = new[] { 2, 3, 4 };

        GpuLaunchProbe.Reset();
        using var gpuOutput = gpu.ReduceMaxWithTensorIndices(
            input, axes, keepDims: false, out var gpuIndices);
        using var gpuGradient = gpu.ReduceMaxBackwardWithTensorIndices(
            gradOutput, gpuIndices, inputShape, axes);
        Assert.True(gpuOutput.IsGpuResident);
        Assert.True(gpuIndices.IsGpuResident);
        Assert.True(gpuGradient.IsGpuResident);
        Assert.True(GpuLaunchProbe.Count > 0);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);

        using var cpuOutput = cpu.ReduceMaxWithTensorIndices(
            input, axes, keepDims: false, out var cpuIndices);
        using var cpuGradient = cpu.ReduceMaxBackwardWithTensorIndices(
            gradOutput, cpuIndices, inputShape, axes);
        Assert.Equal(cpuOutput.ToArray(), gpuOutput.ToArray());
        Assert.Equal(cpuIndices.ToArray(), gpuIndices.ToArray());
        Assert.Equal(cpuGradient.ToArray(), gpuGradient.ToArray());

        gpuIndices.Dispose();
        cpuIndices.Dispose();

        GpuLaunchProbe.Reset();
        Tensor<float> gpuTapeGradient;
        using (var tape = new GradientTape<float>())
        {
            using var tapeOutput = gpu.ReduceMax(input, axes, keepDims: false);
            gpuTapeGradient = tape.ComputeGradients(tapeOutput, new[] { input })[input];
        }
        using (gpuTapeGradient)
        {
            Assert.True(gpuTapeGradient.IsGpuResident);
            Assert.Equal(0, GpuLaunchProbe.Readbacks);

            Tensor<float> cpuTapeGradient;
            using (var tape = new GradientTape<float>())
            {
                using var tapeOutput = cpu.ReduceMax(input, axes, keepDims: false);
                cpuTapeGradient = tape.ComputeGradients(tapeOutput, new[] { input })[input];
            }
            using (cpuTapeGradient)
                Assert.Equal(cpuTapeGradient.ToArray(), gpuTapeGradient.ToArray());
        }
    }

    [SkippableFact]
    public void FirstAxisBroadcastMultiply_RemainsResidentAndMatchesRowBroadcast()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var backend = gpu.TestBackend!;
        using var input = Tensor<float>.FromGpuBuffer(
            backend,
            backend.AllocateBuffer([1f, 2f, 3f, 4f, -5f, -6f]),
            [2, 3],
            ownsBuffer: true);
        using var weights = Tensor<float>.FromGpuBuffer(
            backend,
            backend.AllocateBuffer([2f, -0.5f]),
            [2, 1],
            ownsBuffer: true);

        GpuLaunchProbe.Reset();
        using var output = gpu.BroadcastMultiplyColumnGpu(input, weights);

        Assert.True(output.IsGpuResident);
        Assert.True(GpuLaunchProbe.Count > 0);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        Assert.Equal([2f, 4f, 6f, -2f, 2.5f, 3f], output.ToArray());
    }

    [SkippableFact]
    public void HashEncoding_ForwardAndBackwardRemainResident()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var backend = gpu.TestBackend!;
        using var positions = Tensor<float>.FromGpuBuffer(backend,
            backend.AllocateBuffer([0.1f, 0.2f, 0.3f, 0.9f, 0.8f, 0.7f]),
            [2, 3], ownsBuffer: true);
        using var table0 = Tensor<float>.FromGpuBuffer(backend,
            backend.AllocateBuffer([0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f]),
            [4, 2], ownsBuffer: true);
        using var table1 = Tensor<float>.FromGpuBuffer(backend,
            backend.AllocateBuffer([
                0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f]),
            [8, 2], ownsBuffer: true);

        using var encoded = gpu.MultiresolutionHashEncoding(
            positions, [table0, table1], [4, 8], 2);
        Assert.True(encoded.IsGpuResident);
        Assert.Equal(new[] { 2, 4 }, encoded.Shape.ToArray());

        using var outputGradient = Tensor<float>.FromGpuBuffer(backend,
            backend.AllocateBuffer([1f, -1f, 0.5f, -0.5f, 2f, -2f, 1.5f, -1.5f]),
            [2, 4], ownsBuffer: true);
        var gradients = gpu.MultiresolutionHashEncodingBackward(
            positions, [table0, table1], [4, 8], 2, outputGradient);
        try
        {
            Assert.Equal(2, gradients.Length);
            Assert.All(gradients, gradient => Assert.True(gradient.IsGpuResident));
            Assert.Equal(new[] { 4, 2 }, gradients[0].Shape.ToArray());
            Assert.Equal(new[] { 8, 2 }, gradients[1].Shape.ToArray());
        }
        finally
        {
            foreach (var gradient in gradients) gradient.Dispose();
        }
    }

    [SkippableFact]
    public void UniqueConsecutive_ResidentInputProducesResidentValues()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        var gpu = _fx.Gpu!;
        var backend = gpu.TestBackend!;
        using var input = Tensor<float>.FromGpuBuffer(backend,
            backend.AllocateBuffer([1f, 1f, 2f, 2f, 2f, 1f, float.NaN, float.NaN]),
            [8], ownsBuffer: true);
        using var result = gpu.TensorUniqueConsecutive(input);
        Assert.True(result.IsGpuResident);
        Assert.Equal(new[] { 5 }, result.Shape.ToArray());
        var values = result.ToArray();
        Assert.Equal(1f, values[0]);
        Assert.Equal(2f, values[1]);
        Assert.Equal(1f, values[2]);
        Assert.True(float.IsNaN(values[3]));
        Assert.True(float.IsNaN(values[4]));
    }

    [SkippableFact]
    public void DropoutEvaluation_ReturnsResidentOutputAndMask()
    {
        if (!_fx.GpuReady)
        {
            Skip.If(true, "No DirectGpu backend (CUDA/OpenCL/HIP) available on this system.");
            return;
        }

        GpuLaunchProbe.Reset();
        using var output = _fx.Gpu!.Dropout(
            new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 }),
            0.5,
            training: false,
            out var mask);
        using (mask)
        {
            Assert.True(output.IsGpuResident);
            Assert.True(mask.IsGpuResident);
            Assert.True(GpuLaunchProbe.Count > 0);
            Assert.Equal(new[] { 1f, 2f, 3f, 4f }, output.ToArray());
            Assert.Equal(new[] { 1f, 1f, 1f, 1f }, mask.ToArray());
        }
    }

    private static void WriteReport(IEnumerable<string> covered, IEnumerable<string> metadata,
        IEnumerable<string> hostContracts,
        IEnumerable<string> fallback,
        IEnumerable<string> errored, IEnumerable<string> hollow, IEnumerable<string> readbacks)
    {
        var cov = covered.ToList();
        var md = metadata.ToList();
        var hc = hostContracts.ToList();
        var fb = fallback.ToList();
        var er = errored.ToList();
        var hw = hollow.ToList();
        var rb = readbacks.ToList();
        var sb = new StringBuilder();
        sb.AppendLine("# GPU residency probe — measured kernel launches per op (#775)");
        sb.AppendLine($"met the per-case GPU kernel minimum (on-device) : {cov.Count}");
        sb.AppendLine($"metadata-only resident views / identity ops : {md.Count}");
        sb.AppendLine($"host-defined managed/codec contracts : {hc.Count}");
        sb.AppendLine($"below the per-case kernel minimum (CPU fallback, the worklist) : {fb.Count} (goal 0)");
        sb.AppendLine($"  of which HOLLOW (override threw kernel-not-found -> silent CPU) : {hw.Count}");
        sb.AppendLine($"threw before classifying : {er.Count}");
        sb.AppendLine();
        sb.AppendLine("## metadata-only (zero compute by contract; resident aliases tested separately)");
        foreach (var n in md) sb.AppendLine($"  [x] {n}");
        sb.AppendLine();
        sb.AppendLine("## host contracts (not portable GPU-kernel signatures; op\tcategory\treason)");
        foreach (var n in hc) sb.AppendLine($"  [~] {n}");
        sb.AppendLine();
        sb.AppendLine("## CPU fallback — below required dispatch count (op\\tcategory\\tobserved/required)");
        foreach (var n in fb) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine("## errored (excluded from the count)");
        foreach (var n in er) sb.AppendLine($"  [!] {n}");
        WriteFile("gpu-cpu-fallback.md", sb.ToString());

        var hb = new StringBuilder();
        hb.AppendLine("# Hollow GPU overrides — threw kernel-not-found, silently fell back to CPU (#775)");
        hb.AppendLine($"count : {hw.Count} (goal 0). Reflection counts these GPU-covered; they do ZERO GPU work.");
        hb.AppendLine("Each: op<TAB>category<TAB>the kernel name(s) that were looked up but not registered.");
        hb.AppendLine();
        foreach (var n in hw) hb.AppendLine($"  [ ] {n}");
        WriteFile("gpu-hollow-overrides.md", hb.ToString());

        var rbReport = new StringBuilder();
        rbReport.AppendLine("# GPU internal readbacks before result materialization (#775)");
        rbReport.AppendLine($"count : {rb.Count} (goal 0)");
        rbReport.AppendLine("Audited dynamic-shape scalars and synchronous shape/index metadata arrays are excluded; tensor values must remain resident.");
        rbReport.AppendLine();
        foreach (var n in rb) rbReport.AppendLine($"  [ ] {n}");
        WriteFile("gpu-internal-readbacks.md", rbReport.ToString());
    }

    private static void WriteFile(string name, string content)
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, name), content);
        }
        catch { /* best-effort */ }
    }
}
#endif
