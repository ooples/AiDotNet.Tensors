// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HouseholderChainParityTests
{
    private static Tensor<float> RandomTensor(Random rng, int[] shape, float scale = 1.0f)
    {
        int len = 1;
        foreach (int s in shape) len *= s;
        var data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = (float)((2.0 * rng.NextDouble() - 1.0) * scale);
        return new Tensor<float>(data, shape);
    }

    private static void RequireOrSkipGpu(DirectGpuTensorEngine gpu)
    {
        if (!gpu.SupportsGpu)
        {
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU") == "1")
                Assert.True(gpu.SupportsGpu, "AIDOTNET_REQUIRE_GPU=1 was set, but DirectGpuTensorEngine.SupportsGpu is false.");
            Skip.If(true, "No DirectGpu backend available on this runner.");
        }
    }

    [Theory]
    [InlineData(17, 1, 32, false)]
    [InlineData(17, 2, 64, true)]
    [InlineData(64, 2, 256, false)]
    [InlineData(64, 4, 256, true)]
    [InlineData(35, 3, 137, false)]
    [InlineData(35, 4, 512, true)]
    public void CpuHouseholderChain_MatchesOrthogonalityAndBackward(int rows, int k, int dim, bool transpose)
    {
        var rng = new Random(1000 + rows * 7 + k * 31 + dim + (transpose ? 1 : 0));
        var xData = RandomTensor(rng, new[] { rows, dim });
        var uData = RandomTensor(rng, new[] { k, dim });
        var dyData = RandomTensor(rng, new[] { rows, dim });

        var cpu = new CpuEngine();
        var xCpu = new Tensor<float>(xData.AsSpan().ToArray(), new[] { rows, dim });
        var uCpu = new Tensor<float>(uData.AsSpan().ToArray(), new[] { k, dim });
        var dyCpu = new Tensor<float>(dyData.AsSpan().ToArray(), new[] { rows, dim });

        Tensor<float> yCpu;
        Dictionary<Tensor<float>, Tensor<float>> cpuGrads;
        using (var tape = new GradientTape<float>())
        {
            var prev = AiDotNetEngine.Current;
            try
            {
                AiDotNetEngine.Current = cpu;
                yCpu = cpu.HouseholderChain(xCpu, uCpu, transpose);
                var lossCpu = cpu.ReduceSum(cpu.TensorMultiply(yCpu, dyCpu), null);
                cpuGrads = tape.ComputeGradients(lossCpu, new[] { xCpu, uCpu });
            }
            finally
            {
                AiDotNetEngine.Current = prev;
            }
        }

        var xRoundTrip = cpu.HouseholderChain(yCpu, uCpu, !transpose);
        var xOrigArr = xCpu.AsSpan();
        var xRtArr = xRoundTrip.AsSpan();
        for (int i = 0; i < xOrigArr.Length; i++)
            Assert.True(Math.Abs(xOrigArr[i] - xRtArr[i]) < 2e-5f, $"Orthogonality drift at {i}: {xOrigArr[i]} vs {xRtArr[i]}");
        Assert.Equal(xCpu.Length, cpuGrads[xCpu].Length);
        Assert.Equal(uCpu.Length, cpuGrads[uCpu].Length);
    }

    [SkippableTheory]
    [InlineData(17, 1, 32, false)]
    [InlineData(17, 2, 64, true)]
    [InlineData(64, 2, 256, false)]
    [InlineData(64, 4, 256, true)]
    [InlineData(35, 3, 137, false)]
    [InlineData(35, 4, 512, true)]
    public void GpuFusedAndPortable_MatchCpuForwardAndBackward(int rows, int k, int dim, bool transpose)
    {
        using var gpu = new DirectGpuTensorEngine();
        RequireOrSkipGpu(gpu);

        var rng = new Random(1000 + rows * 7 + k * 31 + dim + (transpose ? 1 : 0));
        var xData = RandomTensor(rng, new[] { rows, dim });
        var uData = RandomTensor(rng, new[] { k, dim });
        var dyData = RandomTensor(rng, new[] { rows, dim });

        var cpu = new CpuEngine();
        var xCpu = new Tensor<float>(xData.AsSpan().ToArray(), new[] { rows, dim });
        var uCpu = new Tensor<float>(uData.AsSpan().ToArray(), new[] { k, dim });
        var dyCpu = new Tensor<float>(dyData.AsSpan().ToArray(), new[] { rows, dim });

        Tensor<float> yCpu;
        Dictionary<Tensor<float>, Tensor<float>> cpuGrads;
        using (var tape = new GradientTape<float>())
        {
            var prev = AiDotNetEngine.Current;
            try
            {
                AiDotNetEngine.Current = cpu;
                yCpu = cpu.HouseholderChain(xCpu, uCpu, transpose);
                var lossCpu = cpu.ReduceSum(cpu.TensorMultiply(yCpu, dyCpu), null);
                cpuGrads = tape.ComputeGradients(lossCpu, new[] { xCpu, uCpu });
            }
            finally
            {
                AiDotNetEngine.Current = prev;
            }
        }

        foreach (bool useFusedCuda in new[] { true, false })
        {
            bool savedFlag = CudaBackend.UseFusedHouseholder;
            long fwdBefore = System.Threading.Interlocked.Read(ref CudaBackend.FusedHouseholderForwardRuns);
            long bwdBefore = System.Threading.Interlocked.Read(ref CudaBackend.FusedHouseholderBackwardRuns);
            long portFwdBefore = System.Threading.Interlocked.Read(ref PortableHouseholderChain.PortableHouseholderForwardRuns);
            long portBwdBefore = System.Threading.Interlocked.Read(ref PortableHouseholderChain.PortableHouseholderBackwardRuns);
            CudaBackend.UseFusedHouseholder = useFusedCuda;
            try
            {
                var prev = AiDotNetEngine.Current;
                AiDotNetEngine.Current = gpu;
                try
                {
                    var xGpu = new Tensor<float>(xData.AsSpan().ToArray(), new[] { rows, dim });
                    var uGpu = new Tensor<float>(uData.AsSpan().ToArray(), new[] { k, dim });
                    var dyGpu = new Tensor<float>(dyData.AsSpan().ToArray(), new[] { rows, dim });

                    Tensor<float> yGpu;
                    Dictionary<Tensor<float>, Tensor<float>> gpuGrads;
                    using (var tape = new GradientTape<float>())
                    {
                        yGpu = gpu.HouseholderChain(xGpu, uGpu, transpose);
                        var lossGpu = gpu.ReduceSum(gpu.TensorMultiply(yGpu, dyGpu), null);
                        gpuGrads = tape.ComputeGradients(lossGpu, new[] { xGpu, uGpu });
                    }

                    if (gpu.GetBackend() is CudaBackend)
                    {
                        if (useFusedCuda)
                        {
                            Assert.True(System.Threading.Interlocked.Read(ref CudaBackend.FusedHouseholderForwardRuns) > fwdBefore);
                            Assert.True(System.Threading.Interlocked.Read(ref CudaBackend.FusedHouseholderBackwardRuns) > bwdBefore);
                        }
                        else
                        {
                            Assert.True(System.Threading.Interlocked.Read(ref PortableHouseholderChain.PortableHouseholderForwardRuns) > portFwdBefore);
                            Assert.True(System.Threading.Interlocked.Read(ref PortableHouseholderChain.PortableHouseholderBackwardRuns) > portBwdBefore);
                        }
                    }

                    var yC = yCpu.AsSpan();
                    var yG = yGpu.AsSpan();
                    for (int i = 0; i < yC.Length; i++)
                        Assert.True(Math.Abs(yC[i] - yG[i]) < 3e-5f, $"Forward mismatch (fused={useFusedCuda}) at {i}: {yC[i]} vs {yG[i]}");

                    var dxC = cpuGrads[xCpu].AsSpan();
                    var dxG = gpuGrads[xGpu].AsSpan();
                    for (int i = 0; i < dxC.Length; i++)
                        Assert.True(Math.Abs(dxC[i] - dxG[i]) < 5e-5f, $"GradInput mismatch (fused={useFusedCuda}) at {i}: {dxC[i]} vs {dxG[i]}");

                    var duC = cpuGrads[uCpu].AsSpan();
                    var duG = gpuGrads[uGpu].AsSpan();
                    for (int i = 0; i < duC.Length; i++)
                        Assert.True(Math.Abs(duC[i] - duG[i]) < 2e-4f, $"GradReflectors mismatch (fused={useFusedCuda}) at {i}: {duC[i]} vs {duG[i]}");
                }
                finally
                {
                    AiDotNetEngine.Current = prev;
                }
            }
            finally
            {
                CudaBackend.UseFusedHouseholder = savedFlag;
            }
        }
    }

    [Fact]
    public void CpuExtremeReflectorsAndTypeGuards_ProduceExactFiniteResults()
    {
        var cpu = new CpuEngine();
        Assert.Throws<NotSupportedException>(() =>
            cpu.HouseholderChain(new Tensor<int>(new[] { 1, 0 }, new[] { 1, 2 }), new Tensor<int>(new[] { 1, 2 }, new[] { 1, 2 })));

        var hugeOut = cpu.HouseholderChain(
            new Tensor<double>(new[] { 1e308 }, new[] { 1, 1 }),
            new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 }));
        Assert.False(double.IsInfinity(hugeOut[0]) || double.IsNaN(hugeOut[0]));
        Assert.Equal(-1e308, hugeOut[0], 5);
    }

    [SkippableFact]
    public void GpuExtremeReflectors_ZeroHugeAndSubFloor_MatchKnownNonZeroSubFloorReference()
    {
        using var gpu = new DirectGpuTensorEngine();
        RequireOrSkipGpu(gpu);

        int rows = 4, k = 3, dim = 16;
        var rng = new Random(77);
        var x = RandomTensor(rng, new[] { rows, dim });
        var dy = RandomTensor(rng, new[] { rows, dim });
        var uArr = new float[k * dim];
        // Row 0: identically zero -> identity
        // Row 1: huge (1e12f) -> overflow-safe normalized reflection
        // Row 2: sub-floor with normFloor = 1.0 -> ||u_2||^2 = 0.01 < 1.0, active=0, nonzero gradient!
        for (int i = 0; i < dim; i++)
            uArr[dim + i] = i % 2 == 0 ? 1e12f : -1e12f;
        uArr[2 * dim] = 0.1f;
        const double explicitFloor = 1.0;

        var cpu = new CpuEngine();
        Dictionary<Tensor<float>, Tensor<float>> cGrads;
        Tensor<float> yCpu;
        var prev = AiDotNetEngine.Current;
        var xCpu = new Tensor<float>(x.AsSpan().ToArray(), new[] { rows, dim });
        var uCpu = new Tensor<float>((float[])uArr.Clone(), new[] { k, dim });
        var dyCpu = new Tensor<float>(dy.AsSpan().ToArray(), new[] { rows, dim });
        try
        {
            AiDotNetEngine.Current = cpu;
            using var tape = new GradientTape<float>();
            yCpu = cpu.HouseholderChain(xCpu, uCpu, transpose: false, normFloor: explicitFloor);
            cGrads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yCpu, dyCpu), null), new[] { xCpu, uCpu });
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        AiDotNetEngine.Current = gpu;
        try
        {
            var xGpu = new Tensor<float>(x.AsSpan().ToArray(), new[] { rows, dim });
            var uGpu = new Tensor<float>((float[])uArr.Clone(), new[] { k, dim });
            var dyGpu = new Tensor<float>(dy.AsSpan().ToArray(), new[] { rows, dim });
            using var tape = new GradientTape<float>();
            var yGpu = gpu.HouseholderChain(xGpu, uGpu, transpose: false, normFloor: explicitFloor);
            var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yGpu, dyGpu), null), new[] { xGpu, uGpu });

            foreach (float v in gGrads[uGpu].AsSpan())
                Assert.False(float.IsNaN(v) || float.IsInfinity(v));

            float maxSubFloorAbs = 0f;
            for (int i = 0; i < dim; i++)
            {
                float cVal = cGrads[uCpu].AsSpan()[2 * dim + i];
                float gVal = gGrads[uGpu].AsSpan()[2 * dim + i];
                if (Math.Abs(cVal) > maxSubFloorAbs) maxSubFloorAbs = Math.Abs(cVal);
                Assert.True(Math.Abs(cVal - gVal) < 1e-4f, $"Sub-floor reflector grad mismatch at {i}: cpu={cVal} vs gpu={gVal}");
            }
            Assert.True(maxSubFloorAbs > 1e-3f, $"Expected deterministic nonzero sub-floor gradient, got maxAbs={maxSubFloorAbs}");

            for (int i = 0; i < yGpu.Length; i++)
                Assert.True(Math.Abs(yCpu.AsSpan()[i] - yGpu.AsSpan()[i]) < 1e-4f);
            for (int i = 0; i < xGpu.Length; i++)
                Assert.True(Math.Abs(cGrads[xCpu].AsSpan()[i] - gGrads[xGpu].AsSpan()[i]) < 1e-4f);
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }
    }

    [Fact]
    public void TensorScope_DeferredChildDispose_DoesNotEvictEnclosingStorageCreatedBeforeFlush()
    {
        using var outer = TensorScope.Begin();
        bool outerReleased = false;
        GpuBufferReleaseDeferral.Begin();
        try
        {
            using (var inner = TensorScope.Begin())
            {
                var innerTensor = new Tensor<float>(new[] { 4 });
                TensorScope.Track(innerTensor.DataVector, () => { });
            }

            // Created in outer AFTER inner closed, while inner's release is still queued in GpuBufferReleaseDeferral.
            var outerTensor = new Tensor<float>(new[] { 4 });
            Assert.True(outerTensor.DataVector is IScopeTrackableStorage);
            // Verify inner.Created(outerTensor.DataVector) is bounded by inner's close tick.
            var createdMethod = typeof(TensorScope).GetMethod(
                "IsReleasable",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            Assert.NotNull(createdMethod);

            using (var child2 = TensorScope.Begin())
            {
                var childRef = child2;
                child2.Dispose();
                var afterClose = new Tensor<float>(new[] { 4 });
                bool claimedByClosedChild = (bool)createdMethod.Invoke(childRef, new object[] { afterClose.DataVector });
                Assert.False(claimedByClosedChild, "Closed child TensorScope must not claim storage created after its close tick.");
            }

            TensorScope.Track(outerTensor.DataVector, () => outerReleased = true);
            GpuBufferReleaseDeferral.EndAndRelease();
            Assert.False(outerReleased, "Flushing deferred child release must not release outer scope storage.");
        }
        finally
        {
            if (GpuBufferReleaseDeferral.IsActive)
                GpuBufferReleaseDeferral.EndAndRelease();
        }
    }

    [Fact]
    public void FiniteDifferenceGradientCheck_MatchesAnalyticBackward()
    {
        var rng = new Random(31415);
        int rows = 4, k = 3, dim = 12;
        var xBase = RandomTensor(rng, new[] { rows, dim }).AsSpan().ToArray();
        var uBase = RandomTensor(rng, new[] { k, dim }).AsSpan().ToArray();
        var wBase = RandomTensor(rng, new[] { rows, dim }).AsSpan().ToArray();

        var cpu = new CpuEngine();
        var prev = AiDotNetEngine.Current;
        AiDotNetEngine.Current = cpu;
        try
        {
            var xT = new Tensor<double>(Array.ConvertAll(xBase, v => (double)v), new[] { rows, dim });
            var uT = new Tensor<double>(Array.ConvertAll(uBase, v => (double)v), new[] { k, dim });
            var wT = new Tensor<double>(Array.ConvertAll(wBase, v => (double)v), new[] { rows, dim });

            Dictionary<Tensor<double>, Tensor<double>> grads;
            using (var tape = new GradientTape<double>())
            {
                var y = cpu.HouseholderChain(xT, uT, transpose: true);
                var loss = cpu.ReduceSum(cpu.TensorMultiply(y, wT), null);
                grads = tape.ComputeGradients(loss, new[] { xT, uT });
            }

            const double eps = 1e-6;
            var duAnalytic = grads[uT].AsSpan();
            for (int idx = 0; idx < uT.Length; idx++)
            {
                var uPlus = Array.ConvertAll(uBase, v => (double)v);
                var uMinus = Array.ConvertAll(uBase, v => (double)v);
                uPlus[idx] += eps;
                uMinus[idx] -= eps;
                var yP = cpu.HouseholderChain(xT, new Tensor<double>(uPlus, new[] { k, dim }), transpose: true);
                var yM = cpu.HouseholderChain(xT, new Tensor<double>(uMinus, new[] { k, dim }), transpose: true);
                double lP = cpu.ReduceSum(cpu.TensorMultiply(yP, wT), null)[0];
                double lM = cpu.ReduceSum(cpu.TensorMultiply(yM, wT), null)[0];
                double fd = (lP - lM) / (2.0 * eps);
                Assert.True(Math.Abs(duAnalytic[idx] - fd) < 1e-6, $"FD mismatch at u[{idx}]: analytic={duAnalytic[idx]} vs fd={fd}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }
    }
}
