#if NET6_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Sparse routing ops and readout-detached fused cross-entropy on the GPU engine under a tape: forward and gradients
/// must match the CPU engine and remain device-resident.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class SparseRoutingTapeGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public SparseRoutingTapeGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    internal static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    internal static Tensor<int> RandIndices(int[] shape, int numBins, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<int>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.Next(0, numBins);
        return t;
    }

    internal static void AssertClose(float[] e, float[] a, string name)
    {
        Assert.Equal(e.Length, a.Length);
        double scale = Math.Max(e.Select(v => (double)Math.Abs(v)).DefaultIfEmpty(0).Max(), 1e-6);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 1e-4 * scale, $"{name}[{i}]: cpu {e[i]}, gpu {a[i]} (scale {scale})");
    }

    [SkippableTheory]
    [InlineData(2, 9, 16, 4, 40)]
    [InlineData(3, 17, 32, 6, 70)]
    public void SparseRouting_TapeForwardAndGradients_MatchCpu_OnDevice(int b, int s, int g, int k, int c)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();

        var logits = Rand([b, s, g], 1);
        var indices = RandIndices([b, s, k], g, 2);
        var table = Rand([g, c], 3);
        var upstream = Rand([b, s, c], 4);

        (float[] Y, float[] Dl, float[] Dt, float[] Counts) Run(CpuEngine e, bool onDevice)
        {
            using var tape = new GradientTape<float>();
            var picked = e.TensorGatherRowValues(logits, indices);
            var weights = e.Softmax(picked, -1);
            var mixed = e.TopKWeightedRowMix(table, indices, weights);
            var counts = e.TensorCountIndices<float>(indices, g);
            if (onDevice)
            {
                Assert.True(picked.IsGpuResident || picked.HasPendingGpuData, "TensorGatherRowValues ran on host");
                Assert.True(mixed.IsGpuResident || mixed.HasPendingGpuData, "TopKWeightedRowMix ran on host");
                Assert.True(counts.IsGpuResident || counts.HasPendingGpuData, "TensorCountIndices ran on host");
            }
            var loss = e.ReduceSum(e.TensorMultiply(mixed, upstream), [0, 1, 2], keepDims: false);
            var grads = tape.ComputeGradients(loss, [logits, table]);
            if (onDevice)
            {
                Assert.True(grads[logits].IsGpuResident || grads[logits].HasPendingGpuData, "dLogits ran on host");
                Assert.True(grads[table].IsGpuResident || grads[table].HasPendingGpuData, "dTable ran on host");
            }
            return (mixed.ToArray(), grads[logits].ToArray(), grads[table].ToArray(), counts.ToArray());
        }

        var cpuRun = Run(cpu, false);
        var gpuRun = Run(gpu, true);
        AssertClose(cpuRun.Y, gpuRun.Y, "mixed");
        AssertClose(cpuRun.Dl, gpuRun.Dl, "dLogits");
        AssertClose(cpuRun.Dt, gpuRun.Dt, "dTable");
        AssertClose(cpuRun.Counts, gpuRun.Counts, "counts");
    }

    [SkippableFact]
    public void FusedLinearCrossEntropy_DetachedReadout_MatchesCpu_OnDevice()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();

        int n = 32, d = 24, vocab = 96;
        var hInter = Rand([n, d], 10);
        var hFinal = Rand([n, d], 11);
        var w = Rand([d, vocab], 12);
        var b = Rand([vocab], 13);
        var targets = RandIndices([n], vocab, 14);

        (float Loss, float[] DhInter, float[] DhFinal, float[] Dw, float[] Db) Run(CpuEngine e, bool onDevice)
        {
            using var tape = new GradientTape<float>();
            var l1 = e.FusedLinearCrossEntropyWithLogits(hInter, w, b, targets, computeReadoutGrads: false);
            var l2 = e.FusedLinearCrossEntropyWithLogits(hFinal, w, b, targets, computeReadoutGrads: true);
            var total = e.TensorAdd(e.TensorMultiplyScalar(l1, 0.5f), l2);
            var grads = tape.ComputeGradients(total, [hInter, hFinal, w, b]);
            if (onDevice)
            {
                Assert.True(grads[hInter].IsGpuResident || grads[hInter].HasPendingGpuData, "dhInter ran on host");
                Assert.True(grads[w].IsGpuResident || grads[w].HasPendingGpuData, "dw ran on host");
            }
            return (total[0], grads[hInter].ToArray(), grads[hFinal].ToArray(), grads[w].ToArray(), grads[b].ToArray());
        }

        var c = Run(cpu, false);
        var g = Run(gpu, true);
        Assert.True(Math.Abs(c.Loss - g.Loss) <= 1e-4f, $"loss: cpu {c.Loss}, gpu {g.Loss}");
        AssertClose(c.DhInter, g.DhInter, "dhInter");
        AssertClose(c.DhFinal, g.DhFinal, "dhFinal");
        AssertClose(c.Dw, g.Dw, "dw");
        AssertClose(c.Db, g.Db, "db");
    }
}

/// <summary>
/// Every backend's <see cref="ISparseRoutingBackend"/> kernels driven directly against the CPU engine, verifying zero
/// host-device transfers during kernel execution across all 6 GPU backends.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class SparseRoutingBackendParityTests
{
    public static TheoryData<int, int, int, int> Shapes => new()
    {
        { 18, 16, 4, 40 },
        { 9, 24, 1, 33 },    // top-1; channels not a multiple of 32
        { 31, 12, 5, 70 },   // rows not a multiple of 8; channels > 64
    };

    private static void Check(IDirectGpuBackend backend, int rows, int numEntries, int k, int channels)
    {
        var sr = Assert.IsAssignableFrom<ISparseRoutingBackend>(backend);
        var cpu = new CpuEngine();

        var values = SparseRoutingTapeGpuTests.Rand([rows, numEntries], 21);
        var indices = SparseRoutingTapeGpuTests.RandIndices([rows, k], numEntries, 22);
        var gradGather = SparseRoutingTapeGpuTests.Rand([rows, k], 23);
        var table = SparseRoutingTapeGpuTests.Rand([numEntries, channels], 24);
        var weights = SparseRoutingTapeGpuTests.Rand([rows, k], 25);
        var gradMix = SparseRoutingTapeGpuTests.Rand([rows, channels], 26);

        var yGather = cpu.TensorGatherRowValues(values, indices).ToArray();
        var dvGather = cpu.TensorGatherRowValuesBackward(gradGather, indices, numEntries).ToArray();
        var yMix = cpu.TopKWeightedRowMix(table, indices, weights).ToArray();
        var dwMix = cpu.TopKWeightedRowMixBackwardWeights(gradMix, table, indices).ToArray();
        var dtMix = cpu.TopKWeightedRowMixBackwardTable(gradMix, indices, weights, numEntries).ToArray();
        var counts = cpu.TensorCountIndices<float>(indices, numEntries).ToArray();

        using var valB = backend.AllocateBuffer(values.ToArray());
        using var idxB = backend.AllocateIntBuffer(indices.ToArray());
        using var ggB = backend.AllocateBuffer(gradGather.ToArray());
        using var tblB = backend.AllocateBuffer(table.ToArray());
        using var wB = backend.AllocateBuffer(weights.ToArray());
        using var gmB = backend.AllocateBuffer(gradMix.ToArray());

        using var ygB = backend.AllocateBuffer(rows * k);
        using var dvB = backend.AllocateBuffer(rows * numEntries);
        using var ymB = backend.AllocateBuffer(rows * channels);
        using var dwB = backend.AllocateBuffer(rows * k);
        using var dtB = backend.AllocateBuffer(numEntries * channels);
        using var cntB = backend.AllocateBuffer(numEntries);

        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
        {
            sr.GatherRowValuesForward(valB, idxB, ygB, rows, numEntries, k);
            sr.GatherRowValuesBackward(ggB, idxB, dvB, rows, numEntries, k);
            sr.TopKWeightedRowMixForward(tblB, idxB, wB, ymB, rows, numEntries, k, channels);
            sr.TopKWeightedRowMixBackwardWeights(gmB, tblB, idxB, dwB, rows, numEntries, k, channels);
            sr.TopKWeightedRowMixBackwardTable(gmB, idxB, wB, dtB, rows, numEntries, k, channels);
            sr.CountIndicesInt32(idxB, cntB, rows * k, numEntries);
        }

        Assert.True(scope.Uploads + scope.Downloads == 0,
            "sparse routing crossed the host-device boundary: "
            + string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));

        SparseRoutingTapeGpuTests.AssertClose(yGather, backend.DownloadBuffer(ygB), "yGather");
        SparseRoutingTapeGpuTests.AssertClose(dvGather, backend.DownloadBuffer(dvB), "dvGather");
        SparseRoutingTapeGpuTests.AssertClose(yMix, backend.DownloadBuffer(ymB), "yMix");
        SparseRoutingTapeGpuTests.AssertClose(dwMix, backend.DownloadBuffer(dwB), "dwMix");
        SparseRoutingTapeGpuTests.AssertClose(dtMix, backend.DownloadBuffer(dtB), "dtMix");
        SparseRoutingTapeGpuTests.AssertClose(counts, backend.DownloadBuffer(cntB), "counts");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend, rows, numEntries, k, channels);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Vulkan_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, rows, numEntries, k, channels);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Cuda_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend, rows, numEntries, k, channels);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Hip_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, rows, numEntries, k, channels);
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Metal_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, rows, numEntries, k, channels);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void WebGpu_MatchesCpu_OnDevice(int rows, int numEntries, int k, int channels)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, rows, numEntries, k, channels);
    }
#endif
}
#endif
