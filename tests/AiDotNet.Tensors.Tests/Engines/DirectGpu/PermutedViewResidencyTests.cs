// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
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
        IEngine gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var cpu = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(3);
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
        IEngine gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var cpu = new CpuEngine();
        var rng = RandomHelper.CreateSeededRandom(8);
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
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(4);
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
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(12);
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
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(19);
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
                AssertGradientsMatch(cpuGradW, gpuGradW, "step1 d/dW");

                // Step 2: on-device optimizer updates wGpu in-place (DeviceAuthoritativeViewBase path). Every backend
                // implements the compressed-moment optimizer, so a declined step is a failure, not a skip.
                Assert.True(AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAllocBf16MomentState(wGpu.Length, out mBuf, out vBuf),
                    "the bf16 moment state was not allocated on the device");
                Assert.True(AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamBf16Step(
                        wGpu, gpuGradW, mBuf, vBuf, lr: 1e-2f, beta1: 0.9f, beta2: 0.999f,
                        epsilon: 1e-8f, weightDecay: 0f, step: 1, decoupledWeightDecay: true),
                    "the on-device Adam step declined");
                long step2Readback;
                string step2Sites;
                Tensor<float> gpuGradX2, gpuGradW2;
                using (var gpuTape2 = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
                {
                    var gpuOut2 = gpu.BatchMatMul(x, wGpu);
                    var gpuLoss2 = gpu.ReduceSum(gpuOut2, new[] { 0, 1, 2 }, keepDims: false);
                    GpuLaunchProbe.Reset();
                    var grads2 = gpuTape2.ComputeGradients(gpuLoss2, new[] { x, wGpu });
                    step2Readback = GpuLaunchProbe.ReadbackBytes;
                    step2Sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
                    gpuGradX2 = grads2[x];
                    gpuGradW2 = grads2[wGpu];
                }
                Assert.True(step2Readback <= 64, $"Step 2 BatchMatMul backward read back {step2Readback} bytes: {step2Sites}");

                // The CPU reference for step 2 runs on a copy of the updated weight, taken only now so the copy's
                // readback is outside the measured window.
                AiDotNetEngine.Current = cpu;
                var wUpdated = new Tensor<float>(wGpu.ToArray(), new[] { 4, 6, 8 });

                // The update itself, against an independent Adam step: with bias correction the first step moves each
                // weight by lr * g / (|g| + eps). The bf16 moments round m and v to about 0.4%, well inside 2e-4.
                var wArr = wCpu.ToArray();
                var gArr = cpuGradW.ToArray();
                var updatedArr = wUpdated.ToArray();
                for (int i = 0; i < wArr.Length; i++)
                {
                    float expected = wArr[i] - 1e-2f * gArr[i] / (Math.Abs(gArr[i]) + 1e-8f);
                    Assert.True(Math.Abs(expected - updatedArr[i]) < 2e-4f,
                        $"step2 W[{i}] after Adam: expected {expected}, device {updatedArr[i]}");
                }

                // d/dW of sum(x @ W) does not depend on W; d/dx does, so it is the one that catches a stale weight.
                using (var cpuTape2 = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
                {
                    var cpuLoss2 = cpu.ReduceSum(cpu.BatchMatMul(x, wUpdated), new[] { 0, 1, 2 }, keepDims: false);
                    var cpuGrads2 = cpuTape2.ComputeGradients(cpuLoss2, new[] { x, wUpdated });
                    AssertGradientsMatch(cpuGrads2[x], gpuGradX2, "step2 d/dx");
                    AssertGradientsMatch(cpuGrads2[wUpdated], gpuGradW2, "step2 d/dW");
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

    private static void AssertGradientsMatch(Tensor<float> cpu, Tensor<float> gpu, string name)
    {
        var c = cpu.ToArray();
        var g = gpu.ToArray();
        Assert.Equal(c.Length, g.Length);
        for (int i = 0; i < c.Length; i++)
            Assert.True(Math.Abs(c[i] - g[i]) < 1e-4f, $"{name}[{i}] cpu {c[i]} gpu {g[i]}");
    }

    [SkippableFact]
    public void DepthwiseConv1D_BackwardGradients_StayDeviceResidentForClipGradNorm()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(29);
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

            // A threshold of half the norm, so the clip actually scales the gradients in place.
            double expectedNorm = Math.Sqrt(expectedNormSq);
            double maxNorm = expectedNorm / 2;
            using var devNorm = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryClipGradNorm(new[] { gpuGx, gpuGk }, maxNorm);
            Assert.NotNull(devNorm);
            if (devNorm is null) return;
            Assert.True(devNorm.IsFinite);
            Assert.True(Math.Abs(expectedNorm - devNorm.TotalNorm) < 1e-2,
                $"Expected pre-clip norm {expectedNorm} vs GPU norm {devNorm.TotalNorm}");
            float coefficient = (float)(maxNorm / (expectedNorm + 1e-6));
            Assert.True(Math.Abs(coefficient - devNorm.ClipCoefficient) < 1e-4f,
                $"Expected clip coefficient {coefficient} vs GPU {devNorm.ClipCoefficient}");

            var gpuGxArr = gpuGx.ToArray();
            var gpuGkArr = gpuGk.ToArray();
            for (int i = 0; i < cpuGx.Length; i++)
                Assert.True(Math.Abs(cpuGx[i] * coefficient - gpuGxArr[i]) < 1e-4f, $"d/dx[{i}] cpu {cpuGx[i] * coefficient} gpu {gpuGxArr[i]}");
            for (int i = 0; i < cpuGk.Length; i++)
                Assert.True(Math.Abs(cpuGk[i] * coefficient - gpuGkArr[i]) < 1e-4f, $"d/dk[{i}] cpu {cpuGk[i] * coefficient} gpu {gpuGkArr[i]}");
            double clippedNorm = Math.Sqrt(gpuGxArr.Concat(gpuGkArr).Sum(g => (double)g * g));
            Assert.True(Math.Abs(clippedNorm - maxNorm) < 1e-3 * maxNorm, $"clipped norm {clippedNorm}, max norm {maxNorm}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void TensorFlip_CumSum_And_TensorBatchMatMul_Gradients_StayDeviceResidentForClipGradNorm()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        try
        {
            var rng = RandomHelper.CreateSeededRandom(37);
            var x = new Tensor<float>(new[] { 4, 6, 8 });
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var convW = new Tensor<float>(new[] { 8, 4 });
            for (int i = 0; i < convW.Length; i++) convW[i] = (float)(rng.NextDouble() * 0.5 - 0.25);
            var expertW = new Tensor<float>(new[] { 4, 8, 8 });
            for (int i = 0; i < expertW.Length; i++) expertW[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

            float[] RunCpu()
            {
                AiDotNetEngine.Current = cpu;
                IEngine eng = cpu;
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var flipped = eng.Reshape(eng.TensorFlip(convW, new[] { 1 }), new[] { 8, 1, 4 });
                var conv = eng.TensorPermute(
                    eng.TensorSlice(eng.DepthwiseConv1D(eng.TensorPermute(x, new[] { 0, 2, 1 }), flipped, 1, 3), new[] { 0, 0, 0 }, new[] { 4, 8, 6 }),
                    new[] { 0, 2, 1 });
                var cum = eng.TensorCumSum(conv, 1);
                var bmm = eng.TensorBatchMatMul(cum, expertW);
                var loss = eng.ReduceSum(bmm, new[] { 0, 1, 2 }, false);
                var grads = tape.ComputeGradients(loss, new[] { x, convW, expertW });
                return grads[x].ToArray().Concat(grads[convW].ToArray()).Concat(grads[expertW].ToArray()).ToArray();
            }

            var cpuGrads = RunCpu();

            AiDotNetEngine.Current = gpu;
            gpu.RegisterPersistentTensor(convW, PersistentTensorRole.Weights);
            gpu.RegisterPersistentTensor(expertW, PersistentTensorRole.Weights);
            IEngine gpuEng = gpu;
            Tensor<float> gx, gConvW, gExpertW;
            using (var gpuTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
            {
                var flipped = gpuEng.Reshape(gpuEng.TensorFlip(convW, new[] { 1 }), new[] { 8, 1, 4 });
                var conv = gpuEng.TensorPermute(
                    gpuEng.TensorSlice(gpuEng.DepthwiseConv1D(gpuEng.TensorPermute(x, new[] { 0, 2, 1 }), flipped, 1, 3), new[] { 0, 0, 0 }, new[] { 4, 8, 6 }),
                    new[] { 0, 2, 1 });
                var cum = gpuEng.TensorCumSum(conv, 1);
                var bmm = gpuEng.TensorBatchMatMul(cum, expertW);
                var loss = gpuEng.ReduceSum(bmm, new[] { 0, 1, 2 }, false);
                var grads = gpuTape.ComputeGradients(loss, new[] { x, convW, expertW });
                gx = grads[x];
                gConvW = grads[convW];
                gExpertW = grads[expertW];
            }

            Assert.True(gx.IsGpuResident, "CumSum/Conv input gradient should stay GPU-resident.");
            Assert.True(gConvW.IsGpuResident, "TensorFlip kernel gradient should stay GPU-resident.");
            Assert.True(gExpertW.IsGpuResident, "TensorBatchMatMul 3D persistent weight gradient should stay GPU-resident.");

            // A threshold of half the CPU reference norm, so the clip actually scales every gradient in place.
            double expectedNorm = Math.Sqrt(cpuGrads.Sum(g => (double)g * g));
            double maxNorm = expectedNorm / 2;
            using var devNorm = AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryClipGradNorm(new[] { gx, gConvW, gExpertW }, maxNorm);
            Assert.True(devNorm is { IsFinite: true }, "the device gradient-norm clip declined or saw a non-finite norm");
            double deviceNorm = devNorm is null ? double.NaN : devNorm.TotalNorm;
            Assert.True(Math.Abs(expectedNorm - deviceNorm) < 1e-3 * expectedNorm,
                $"Expected pre-clip norm {expectedNorm} vs GPU norm {deviceNorm}");
            float coefficient = (float)(maxNorm / (expectedNorm + 1e-6));

            var gpuGrads = gx.ToArray().Concat(gConvW.ToArray()).Concat(gExpertW.ToArray()).ToArray();
            Assert.Equal(cpuGrads.Length, gpuGrads.Length);
            for (int i = 0; i < cpuGrads.Length; i++)
                Assert.True(Math.Abs(cpuGrads[i] * coefficient - gpuGrads[i]) < 1e-3f,
                    $"grad[{i}] cpu {cpuGrads[i] * coefficient} gpu {gpuGrads[i]} (clip coefficient {coefficient})");
            double clippedNorm = Math.Sqrt(gpuGrads.Sum(g => (double)g * g));
            Assert.True(Math.Abs(clippedNorm - maxNorm) < 1e-3 * maxNorm, $"clipped norm {clippedNorm}, max norm {maxNorm}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
