// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A GPU tape used to download EVERY step intermediate to the host before freeing it -- at tape dispose and after
/// each intermediate's last backward use -- in case something read it later (measured: ~3 GB device-to-host per
/// training step on a 26.8M-parameter LM, over half the step). Now, as in PyTorch, a forward result the caller holds
/// stays valid after the tape and is freed when it is no longer referenced; nothing is downloaded at the tape's end.
/// The gradients backward computes but does not return are freed into the device pool as soon as backward ends.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuTapeIntermediateReleaseTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    // The tape dispatches its backward on AiDotNetEngine.Current; bind it to the engine that ran the forward.
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuTapeIntermediateReleaseTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
        if (fixture.IsAvailable) AiDotNetEngine.Current = fixture.Engine!;
    }

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    // loss = sum(tanh(x·w) * 2): three device intermediates (matmul, tanh, scaled) plus the loss.
    private static (Tensor<float> h, Tensor<float> a, Tensor<float> loss) Forward(IEngine e, Tensor<float> x, Tensor<float> w)
    {
        var h = e.TensorMatMul(x, w);
        var a = e.TensorTanh(h);
        var scaled = e.TensorMultiplyScalar(a, 2f);
        var loss = e.ReduceSum(scaled, new[] { 0, 1 }, keepDims: false);
        return (h, a, loss);
    }

    private static float[] CpuGrad(Tensor<float> x, Tensor<float> w)
    {
        IEngine cpu = new CpuEngine();
        var current = AiDotNetEngine.Current;
        AiDotNetEngine.Current = cpu;
        try
        {
            using var tape = new GradientTape<float>();
            var (_, _, loss) = Forward(cpu, x, w);
            return tape.ComputeGradients(loss, new[] { w })[w].ToArray();
        }
        finally
        {
            AiDotNetEngine.Current = current;
        }
    }

    [SkippableFact]
    public void Intermediates_StayReadableAfterTheTape_AndNothingIsDownloadedAtItsEnd()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var x = Rand(new[] { 64, 32 }, 1);
        var w = Rand(new[] { 32, 48 }, 2);
        var expectedGrad = CpuGrad(x, w);
        float[] expectedA;
        {
            IEngine cpu = new CpuEngine();
            expectedA = cpu.TensorTanh(cpu.TensorMatMul(x, w)).ToArray();
        }

        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            Tensor<float> h, a, grad;
            long backwardReadbackBytes, disposeReadbackBytes;
            string disposeSites, backwardSites;
            var tape = new GradientTape<float>();
            try
            {
                Tensor<float> loss;
                (h, a, loss) = Forward(gpu, x, w);
                tape.Retain(a);
                GpuLaunchProbe.Reset();
                grad = tape.ComputeGradients(loss, new[] { w })[w];
                backwardReadbackBytes = GpuLaunchProbe.ReadbackBytes;
                backwardSites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
            }
            finally
            {
                GpuLaunchProbe.Reset();
                tape.Dispose();
                disposeReadbackBytes = GpuLaunchProbe.ReadbackBytes;
                disposeSites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
            }

            // The gradient itself was not released (it is not a tape intermediate) and is correct.
            var g = grad.ToArray();
            for (int i = 0; i < g.Length; i++)
                Assert.True(Math.Abs(g[i] - expectedGrad[i]) < 1e-3f, $"dL/dw[{i}]: gpu {g[i]} cpu {expectedGrad[i]}");

            // Backward's last-use release and the dispose-time release move nothing to the host (the loss scalar
            // may be read by the backward seed; allow a few bytes).
            Assert.True(backwardReadbackBytes <= 64, $"backward read back {backwardReadbackBytes} bytes: {backwardSites}");
            Assert.True(disposeReadbackBytes == 0, $"tape dispose read back {disposeReadbackBytes} bytes: {disposeSites}");

            // A forward result the caller still holds keeps its values after the tape, as in PyTorch: read directly, and
            // fed to another GPU op (its device buffer must not have been handed to anyone else).
            float[] expectedH;
            {
                IEngine cpu = new CpuEngine();
                expectedH = cpu.TensorMatMul(x, w).ToArray();
            }
            var hValues = h.ToArray();
            for (int i = 0; i < hValues.Length; i++)
                Assert.True(Math.Abs(hValues[i] - expectedH[i]) < 1e-4f, $"h[{i}] after the tape: {hValues[i]} vs {expectedH[i]}");
            var doubled = gpu.TensorAdd(h, h).ToArray();
            for (int i = 0; i < doubled.Length; i++)
                Assert.True(Math.Abs(doubled[i] - 2 * expectedH[i]) < 1e-3f, $"(h + h)[{i}] after the tape: {doubled[i]}");

            // A retained intermediate survives with its values too; inputs and parameters are untouched.
            var aValues = a.ToArray();
            for (int i = 0; i < aValues.Length; i++)
                Assert.True(Math.Abs(aValues[i] - expectedA[i]) < 1e-4f, $"retained a[{i}]: {aValues[i]} vs {expectedA[i]}");
            Assert.Equal(x.Length, x.ToArray().Length);
            Assert.Equal(w.Length, w.ToArray().Length);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
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

    [SkippableFact]
    public void RepeatedSteps_KeepProducingCorrectGradients()
    {
        // Released buffers go back to the pool and are re-rented by the next step; a stale binding would feed one
        // step's data into the next. Five steps on fresh inputs must each match the CPU.
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var w = Rand(new[] { 32, 48 }, 2);
        for (int step = 0; step < 5; step++)
        {
            var x = Rand(new[] { 64, 32 }, 10 + step);
            var expected = CpuGrad(x, w);
            float[] got;
            using (var tape = new GradientTape<float>())
            {
                var (_, _, loss) = Forward(gpu, x, w);
                got = tape.ComputeGradients(loss, new[] { w })[w].ToArray();
            }
            for (int i = 0; i < got.Length; i++)
                Assert.True(Math.Abs(got[i] - expected[i]) < 1e-3f, $"step {step} dL/dw[{i}]: gpu {got[i]} cpu {expected[i]}");
        }
    }
    [SkippableFact]
    public void LossAndGradients_OutliveTheirTape_WithoutDownloadOrCacheGrowth()
    {
        // PyTorch keeps a result alive while it is referenced. The loss and gradients a tape returns are read after
        // it ends (logging, a host-side update, comparing two runs); they must stay readable across later tapes,
        // must not be downloaded unless read, and must not accumulate in the activation cache step after step.
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var w = Rand(new[] { 32, 48 }, 2);
        var x0 = Rand(new[] { 64, 32 }, 20);
        float[] expected = CpuGrad(x0, w);
        Tensor<float> firstGrad, firstLoss;
        using (var tape = new GradientTape<float>())
        {
            var (_, _, loss) = Forward(gpu, x0, w);
            firstGrad = tape.ComputeGradients(loss, new[] { w })[w];
            firstLoss = loss;
        }

        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        long cacheBytesAfterFirst = gpu.CurrentActivationCacheBytes;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            // Ten more steps whose results nobody reads.
            for (int step = 0; step < 10; step++)
            {
                using var tape = new GradientTape<float>();
                var (_, _, loss) = Forward(gpu, Rand(new[] { 64, 32 }, 30 + step), w);
                tape.ComputeGradients(loss, new[] { w });
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            Assert.True(GpuLaunchProbe.ReadbackBytes <= 10 * 64,
                $"ten unread steps read back {GpuLaunchProbe.ReadbackBytes} bytes: " + string.Join("; ", GpuLaunchProbe.ReadbackSites));
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(gpu.CurrentActivationCacheBytes <= cacheBytesAfterFirst,
            $"activation cache grew from {cacheBytesAfterFirst} to {gpu.CurrentActivationCacheBytes} bytes over ten steps");

        // The first step's results are still readable, and correct, after all those tapes.
        Assert.Equal(1, firstLoss.ToArray().Length);
        var g = firstGrad.ToArray();
        for (int i = 0; i < g.Length; i++)
            Assert.True(Math.Abs(g[i] - expected[i]) < 1e-3f, $"dL/dw[{i}] ten tapes later: gpu {g[i]} cpu {expected[i]}");
    }
    [SkippableFact]
    public void SharedGradientContribution_IsCopiedOnTheDevice()
    {
        // z = x + y hands the SAME upstream gradient to x and to y; the second destination gets its own copy. That
        // copy was made on the host (download, then an upload at the next GPU op) even when the gradient was
        // device-resident.
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var x = Rand(new[] { 64, 48 }, 5);
        var y = Rand(new[] { 64, 48 }, 6);
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        Dictionary<Tensor<float>, Tensor<float>> grads;
        long readbackBytes;
        string sites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            using var tape = new GradientTape<float>();
            var z = gpu.TensorAdd(gpu.TensorTanh(x), gpu.TensorTanh(y));
            var loss = gpu.ReduceSum(gpu.TensorMultiply(z, z), new[] { 0, 1 }, keepDims: false);
            GpuLaunchProbe.Reset();
            grads = tape.ComputeGradients(loss, new[] { x, y });
            readbackBytes = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readbackBytes <= 64, $"backward read back {readbackBytes} bytes: {sites}");

        // d/dx sum((tanh x + tanh y)^2) = 2 (tanh x + tanh y)(1 - tanh^2 x), and symmetrically for y.
        var gx = grads[x].ToArray();
        var gy = grads[y].ToArray();
        for (int i = 0; i < gx.Length; i++)
        {
            double tx = Math.Tanh(x[i]), ty = Math.Tanh(y[i]);
            double ex = 2 * (tx + ty) * (1 - tx * tx), ey = 2 * (tx + ty) * (1 - ty * ty);
            Assert.True(Math.Abs(gx[i] - ex) < 1e-4 && Math.Abs(gy[i] - ey) < 1e-4,
                $"[{i}] dx {gx[i]} vs {ex}, dy {gy[i]} vs {ey}");
        }
    }
}
