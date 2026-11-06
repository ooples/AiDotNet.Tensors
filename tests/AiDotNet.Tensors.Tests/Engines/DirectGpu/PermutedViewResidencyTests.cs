// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorPermute returns a strided view. Feeding a view of a GPU-resident result to another GPU op materialized it
/// on the host (download the base, permute on the CPU, upload) -- per step on an LM's attention feature path. It is
/// now permuted on the device.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class PermutedViewResidencyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public PermutedViewResidencyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    public static IEnumerable<object[]> Cases() => new[]
    {
        new object[] { new[] { 4, 6, 8 }, new[] { 0, 2, 1 } },
        new object[] { new[] { 4, 6, 8 }, new[] { 2, 0, 1 } },
        new object[] { new[] { 2, 3, 4, 5 }, new[] { 0, 2, 1, 3 } },
        new object[] { new[] { 3, 1, 5 }, new[] { 2, 1, 0 } },   // a size-1 axis
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void PermutedViewOfResidentResult_IsPermutedOnTheDevice(int[] shape, int[] axes)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var rng = new Random(3);
        var x = new Tensor<float>(shape);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var expected = cpu.TensorMultiplyScalar(cpu.TensorPermute(cpu.TensorTanh(x), axes), 2f).ToArray();

        var resident = gpu.TensorTanh(x);          // device-only result
        var view = gpu.TensorPermute(resident, axes);
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        Tensor<float> scaled;
        long readbackBytes;
        string sites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            scaled = gpu.TensorMultiplyScalar(view, 2f);
            readbackBytes = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readbackBytes == 0, $"consuming the permuted view read back {readbackBytes} bytes: {sites}");
        var got = scaled.ToArray();
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-5f, $"[{i}] gpu {got[i]} cpu {expected[i]}");
    }

    /// <summary>
    /// The attention head split: Permute(Reshape(x·W, [b, s, h, d]), [0, 2, 1, 3]). The base is a matmul result and
    /// the permute is applied to a reshape VIEW of it.
    /// </summary>
    [SkippableFact]
    public void HeadSplitViewOfMatMulResult_IsPermutedOnTheDevice()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var rng = new Random(8);
        var x = new Tensor<float>(new[] { 8, 12 });
        var w = new Tensor<float>(new[] { 12, 24 });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        Tensor<float> Split(IEngine e) => e.TensorPermute(e.Reshape(e.TensorMatMul(x, w), new[] { 2, 4, 3, 8 }), new[] { 0, 2, 1, 3 });
        var expected = cpu.TensorMultiplyScalar(Split(cpu), 2f).ToArray();

        var view = Split(gpu);
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        long readbackBytes;
        string sites;
        Tensor<float> scaled;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            scaled = gpu.TensorMultiplyScalar(view, 2f);
            readbackBytes = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readbackBytes == 0, $"the head split read back {readbackBytes} bytes: {sites}");
        var got = scaled.ToArray();
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) < 1e-4f, $"[{i}] gpu {got[i]} cpu {expected[i]}");
    }

    /// <summary>
    /// Head merge (permute, then reshape) of a device-only result: the reshape needs a contiguous copy, which the base
    /// made on the host. It must stay on the device and keep the Reshape tape node, so gradients still flow.
    /// </summary>
    [SkippableFact]
    public void ReshapeOfPermutedView_StaysOnDevice_AndKeepsTheGradient()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = new Random(4);
            var x = new Tensor<float>(new[] { 2, 3, 4, 5 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

            float[] Grad(IEngine e, out long readback)
            {
                AiDotNetEngine.Current = e;
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var activated = e.TensorTanh(x);
                GpuLaunchProbe.Reset();
                var merged = e.Reshape(e.TensorPermute(activated, new[] { 0, 2, 1, 3 }), new[] { 2, 4, 15 });
                var weights = new Tensor<float>(new[] { 2, 4, 15 });
                for (int i = 0; i < weights.Length; i++) weights[i] = (i % 7) - 3;
                var loss = e.ReduceSum(e.TensorMultiply(merged, weights), new[] { 0, 1, 2 }, keepDims: false);
                readback = GpuLaunchProbe.ReadbackBytes;
                return tape.ComputeGradients(loss, new[] { x })[x].ToArray();
            }

            var cpuGrad = Grad(new CpuEngine(), out _);
            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            float[] gpuGrad;
            long forwardReadback;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;
                gpuGrad = Grad(gpu, out forwardReadback);
            }
            finally
            {
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            }
            Assert.True(forwardReadback == 0, $"the merged heads' forward read back {forwardReadback} bytes");
            for (int i = 0; i < cpuGrad.Length; i++)
                Assert.True(Math.Abs(cpuGrad[i] - gpuGrad[i]) < 1e-4f, $"d/dx[{i}] cpu {cpuGrad[i]} gpu {gpuGrad[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
    /// <summary>
    /// PermuteBackward returns a strided view of the upstream gradient, which gradient accumulation makes contiguous;
    /// Tensor.Contiguous walked it on the host (a download per permute per step on an LM). It now permutes on the
    /// device and records the same Contiguous node.
    /// </summary>
    [SkippableFact]
    public void PermuteBackward_StaysOnTheDevice()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = new Random(12);
            var x = new Tensor<float>(new[] { 2, 3, 4, 5 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var weights = new Tensor<float>(new[] { 2, 4, 3, 5 });
            for (int i = 0; i < weights.Length; i++) weights[i] = (i % 5) - 2;

            float[] Grad(IEngine e, out long backwardReadback, out string sites)
            {
                AiDotNetEngine.Current = e;
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var permuted = e.TensorPermute(e.TensorTanh(x), new[] { 0, 2, 1, 3 });
                var loss = e.ReduceSum(e.TensorMultiply(permuted, weights), new[] { 0, 1, 2, 3 }, keepDims: false);
                GpuLaunchProbe.Reset();
                var g = tape.ComputeGradients(loss, new[] { x })[x];
                backwardReadback = GpuLaunchProbe.ReadbackBytes;
                sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                return g.ToArray();
            }

            var cpuGrad = Grad(new CpuEngine(), out _, out _);
            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            float[] gpuGrad;
            long readback;
            string where;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;
                gpuGrad = Grad(gpu, out readback, out where);
            }
            finally
            {
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            }
            Assert.True(readback <= 64, $"the permute's backward read back {readback} bytes: {where}");
            for (int i = 0; i < cpuGrad.Length; i++)
                Assert.True(Math.Abs(cpuGrad[i] - gpuGrad[i]) < 1e-4f, $"d/dx[{i}] cpu {cpuGrad[i]} gpu {gpuGrad[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void PermutedViewOfPersistent3DWeight_StaysOnTheDevice_BeforeAndAfterOnDeviceOptimizerStep()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = new Random(19);
            var x = new Tensor<float>(new[] { 4, 5, 6 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var wCpu = new Tensor<float>(new[] { 4, 6, 8 });
            var wGpu = new Tensor<float>(new[] { 4, 6, 8 });
            for (int i = 0; i < wCpu.Length; i++)
            {
                float v = (float)(rng.NextDouble() * 0.5 - 0.25);
                wCpu[i] = v;
                wGpu[i] = v;
            }

            gpu.RegisterPersistentTensor(wGpu, PersistentTensorRole.Weights);
            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? mBuf = null;
            AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer? vBuf = null;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;

                // Step 1: persistent 3D weight in _persistentBufferCache (SyncedDeviceViewBase path)
                AiDotNetEngine.Current = cpu;
                Tensor<float> cpuGradX, cpuGradW;
                using (var cpuTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
                {
                    var cpuLoss = cpu.ReduceSum(cpu.BatchMatMul(x, wCpu), new[] { 0, 1, 2 }, keepDims: false);
                    var grads = cpuTape.ComputeGradients(cpuLoss, new[] { x, wCpu });
                    cpuGradX = grads[x];
                    cpuGradW = grads[wCpu];
                }

                AiDotNetEngine.Current = gpu;
                Tensor<float> gpuGradX, gpuGradW;
                long step1Readback;
                string step1Sites;
                using (var gpuTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
                {
                    var gpuOut = gpu.BatchMatMul(x, wGpu);
                    var gpuLoss = gpu.ReduceSum(gpuOut, new[] { 0, 1, 2 }, keepDims: false);
                    GpuLaunchProbe.Reset();
                    var grads = gpuTape.ComputeGradients(gpuLoss, new[] { x, wGpu });
                    step1Readback = GpuLaunchProbe.ReadbackBytes;
                    step1Sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                    gpuGradX = grads[x];
                    gpuGradW = grads[wGpu];
                }

                Assert.True(step1Readback <= 64, $"Step 1 BatchMatMul backward read back {step1Readback} bytes: {step1Sites}");
                var gpuGradXArr = gpuGradX.ToArray();
                var cpuGradXArr = cpuGradX.ToArray();
                for (int i = 0; i < cpuGradXArr.Length; i++)
                    Assert.True(Math.Abs(cpuGradXArr[i] - gpuGradXArr[i]) < 1e-4f, $"step1 d/dx[{i}] cpu {cpuGradXArr[i]} gpu {gpuGradXArr[i]}");

                // Step 2: on-device optimizer updates wGpu in-place (DeviceAuthoritativeViewBase path)
                if (AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAllocBf16MomentState(wGpu.Length, out mBuf, out vBuf) &&
                    AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamBf16Step(
                        wGpu, gpuGradW, mBuf, vBuf, lr: 1e-2f, beta1: 0.9f, beta2: 0.999f,
                        epsilon: 1e-8f, weightDecay: 0f, step: 1, decoupledWeightDecay: true))
                {
                    long step2Readback;
                    string step2Sites;
                    using (var gpuTape2 = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
                    {
                        var gpuOut2 = gpu.BatchMatMul(x, wGpu);
                        var gpuLoss2 = gpu.ReduceSum(gpuOut2, new[] { 0, 1, 2 }, keepDims: false);
                        GpuLaunchProbe.Reset();
                        _ = gpuTape2.ComputeGradients(gpuLoss2, new[] { x, wGpu });
                        step2Readback = GpuLaunchProbe.ReadbackBytes;
                        step2Sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                    }
                    Assert.True(step2Readback <= 64, $"Step 2 BatchMatMul backward read back {step2Readback} bytes: {step2Sites}");
                }
            }
            finally
            {
                mBuf?.Dispose();
                vBuf?.Dispose();
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
                gpu.UnregisterPersistentTensor(wGpu);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void DepthwiseConv1D_BackwardGradients_StayDeviceResidentForClipGradNorm()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = new Random(29);
            var x = new Tensor<float>(new[] { 2, 8, 16 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var k = new Tensor<float>(new[] { 8, 1, 4 });
            for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

            AiDotNetEngine.Current = cpu;
            float[] cpuGx, cpuGk;
            using (var cpuTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                var y = cpu.DepthwiseConv1D(x, k, stride: 1, padding: 1);
                var loss = cpu.ReduceSum(y, new[] { 0, 1, 2 }, keepDims: false);
                var grads = cpuTape.ComputeGradients(loss, new[] { x, k });
                cpuGx = grads[x].ToArray();
                cpuGk = grads[k].ToArray();
            }

            AiDotNetEngine.Current = gpu;
            Tensor<float> gpuGx, gpuGk;
            using (var gpuTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                var y = gpu.DepthwiseConv1D(x, k, stride: 1, padding: 1);
                var loss = gpu.ReduceSum(y, new[] { 0, 1, 2 }, keepDims: false);
                var grads = gpuTape.ComputeGradients(loss, new[] { x, k });
                gpuGx = grads[x];
                gpuGk = grads[k];
            }

            Assert.True(gpuGx.IsGpuResident, "DepthwiseConv1D input gradient should remain GPU-resident.");
            Assert.True(gpuGk.IsGpuResident, "DepthwiseConv1D kernel gradient should remain GPU-resident.");

            // Check numerical parity before in-place norm clipping
            double expectedNormSq = 0.0;
            for (int i = 0; i < cpuGx.Length; i++) expectedNormSq += (double)cpuGx[i] * cpuGx[i];
            for (int i = 0; i < cpuGk.Length; i++) expectedNormSq += (double)cpuGk[i] * cpuGk[i];

            using var devNorm = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryClipGradNorm(new[] { gpuGx, gpuGk }, maxNorm: 1000.0);
            Assert.NotNull(devNorm);
            Assert.True(devNorm!.IsFinite);
            Assert.True(Math.Abs(Math.Sqrt(expectedNormSq) - devNorm.TotalNorm) < 1e-2,
                $"Expected norm {Math.Sqrt(expectedNormSq)} vs GPU norm {devNorm.TotalNorm}");

            var gpuGxArr = gpuGx.ToArray();
            var gpuGkArr = gpuGk.ToArray();
            for (int i = 0; i < cpuGx.Length; i++)
                Assert.True(Math.Abs(cpuGx[i] - gpuGxArr[i]) < 1e-4f, $"d/dx[{i}] cpu {cpuGx[i]} gpu {gpuGxArr[i]}");
            for (int i = 0; i < cpuGk.Length; i++)
                Assert.True(Math.Abs(cpuGk[i] - gpuGkArr[i]) < 1e-4f, $"d/dk[{i}] cpu {cpuGk[i]} gpu {gpuGkArr[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
