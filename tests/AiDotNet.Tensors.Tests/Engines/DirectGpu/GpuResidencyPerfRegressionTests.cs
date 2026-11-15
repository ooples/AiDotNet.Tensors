// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NETCOREAPP
#nullable disable

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU vs CPU parity and zero-host-materialization regression tests for PerfView-identified
/// training bottlenecks:
/// 1. <c>TryNarrowBackwardOnDevice</c> single-launch 2D <c>RectSlice(..., scatter: true)</c> on outer/inner axes.
/// 2. <c>TryIndexSelectBackwardOnDevice</c> device-resident <c>ScatterAddRows</c> (axis 0) and <c>IndexAdd</c> (axis &gt; 0).
/// 3. <c>TryRunUnaryInPlaceResident</c> avoiding implicit <c>Vector&lt;T&gt; -&gt; T[]</c> DtoH download on GPU-resident tensors.
/// 4. <c>ApplyRoPEInterleaved</c> deferred GPU output and <c>PersistentTensorRole.Constant</c> preservation across <c>InvalidateAllWeightCaches</c>.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuResidencyPerfRegressionTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly CpuEngine _cpu = new();
    private readonly bool _gpuReady;

    public GpuResidencyPerfRegressionTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuReady = _gpu.IsGpuAvailable;
        }
        catch
        {
            _gpuReady = false;
        }
    }

    public void Dispose() => _gpu?.Dispose();

    private static Tensor<float> RandomTensor(int seed, int[] shape)
    {
        var rng = new Random(seed);
        int len = 1;
        foreach (int s in shape) len *= s;
        var data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(shape, new Vector<float>(data));
    }

    private static void AssertClose(Tensor<float> actual, Tensor<float> expected, float tol = 1e-4f)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var a = actual.ToArray();
        var e = expected.ToArray();
        float maxErr = 0f;
        for (int i = 0; i < e.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(a[i] - e[i]));
        Assert.True(maxErr <= tol, $"Max abs diff {maxErr:E3} exceeded tolerance {tol:E3}");
    }

    [SkippableFact]
    public void TensorNarrow_Backward_Rank4_InnerAndOuterAxes_MatchesCpu()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        Tensor<float> RunGrad(IEngine engine)
        {
            var prev = AiDotNetEngine.Current;
            AiDotNetEngine.Current = engine;
            try
            {
                var x = RandomTensor(42, new[] { 2, 16, 4, 33 });
                var wNum = RandomTensor(43, new[] { 2, 16, 4, 32 });
                var wDen = RandomTensor(44, new[] { 2, 16, 4, 1 });
                using var tape = new GradientTape<float>();
                var num = engine.TensorNarrow(x, 3, 0, 32);
                var den = engine.TensorNarrow(x, 3, 32, 1);
                var sub = engine.TensorNarrow(num, 1, 3, 10);
                var wSub = engine.TensorNarrow(wNum, 1, 3, 10);
                var term1 = engine.ReduceSum(engine.TensorMultiply(sub, wSub), new[] { 0, 1, 2, 3 }, keepDims: false);
                var term2 = engine.ReduceSum(engine.TensorMultiply(den, wDen), new[] { 0, 1, 2, 3 }, keepDims: false);
                var loss = engine.TensorAdd(term1, term2);
                var grads = tape.ComputeGradients(loss, new[] { x });
                return grads[x];
            }
            finally
            {
                AiDotNetEngine.Current = prev;
            }
        }

        var cpuGrad = RunGrad(_cpu);
        var gpuGrad = RunGrad(_gpu);
        AssertClose(gpuGrad, cpuGrad);
    }

    [SkippableFact]
    public void IndexSelect_Backward_Axis0AndInnerAxis_WithRepeatedIndices_MatchesCpu()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        (Tensor<float> Grad0, Tensor<float> Grad1) RunGrads(IEngine engine)
        {
            var prev = AiDotNetEngine.Current;
            AiDotNetEngine.Current = engine;
            try
            {
                var table = RandomTensor(101, new[] { 64, 24 });
                var idx0 = new Tensor<int>(new[] { 12 }, new Vector<int>(new[] { 3, 7, 3, 0, 63, 7, 3, 19, 0, 42, 63, 12 }));
                var w0 = RandomTensor(102, new[] { 12, 24 });

                var vol = RandomTensor(103, new[] { 3, 16, 8 });
                var idx1 = new Tensor<int>(new[] { 7 }, new Vector<int>(new[] { 5, 2, 5, 15, 0, 2, 5 }));
                var w1 = RandomTensor(104, new[] { 3, 7, 8 });

                using var tape = new GradientTape<float>();
                var sel0 = engine.TensorIndexSelect(table, idx0, axis: 0);
                var sel1 = engine.TensorIndexSelect(vol, idx1, axis: 1);
                var l0 = engine.ReduceSum(engine.TensorMultiply(sel0, w0), new[] { 0, 1 }, keepDims: false);
                var l1 = engine.ReduceSum(engine.TensorMultiply(sel1, w1), new[] { 0, 1, 2 }, keepDims: false);
                var loss = engine.TensorAdd(l0, l1);
                var grads = tape.ComputeGradients(loss, new[] { table, vol });
                return (grads[table], grads[vol]);
            }
            finally
            {
                AiDotNetEngine.Current = prev;
            }
        }

        var (cpuG0, cpuG1) = RunGrads(_cpu);
        var (gpuG0, gpuG1) = RunGrads(_gpu);
        AssertClose(gpuG0, cpuG0);
        AssertClose(gpuG1, cpuG1);
    }

    [SkippableFact]
    public void TensorMultiplyScalarInPlace_GpuResident_AvoidsHostDownload_AndMatchesCpu()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        var a = RandomTensor(201, new[] { 512, 512 }); // 262,144 floats = 1 MB (LOH if materialized)
        var b = RandomTensor(202, new[] { 512, 512 });
        var cpuExpected = _cpu.TensorAdd(a, b);
        _cpu.TensorMultiplyScalarInPlace(cpuExpected, 0.375f);
        IEngine gpu = _gpu;
        var gpuResident = gpu.TensorAdd(a, b);
        var warmup = gpu.TensorAdd(a, b);
        gpu.TensorMultiplyScalarInPlace(warmup, 0.5f); // warm up Scale kernel + JIT once

        long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
        gpu.TensorMultiplyScalarInPlace(gpuResident, 0.375f);
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

        // Before the (object?) cast fix in TryRunUnaryInPlaceResident, C# T[]? ?? Vector<T>
        // invoked Vector<T>.op_Implicit -> ToArray() -> DownloadBuffer, allocating >= 2 MB LOH!
        Assert.True(allocatedBytes < 4_096, $"Expected < 4 KB steady-state allocation (zero LOH download), but allocated {allocatedBytes} bytes.");
        AssertClose(gpuResident, cpuExpected);
    }

    [SkippableFact]
    public void ApplyRoPEInterleaved_PreservesConstantsAcrossWeightInvalidation_AndMatchesCpu()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        var x1 = RandomTensor(301, new[] { 4, 8, 64, 16 }); // 32,768 floats = 128 KB (LOH if host output allocated)
        var x2 = RandomTensor(302, new[] { 4, 8, 64, 16 });
        var cos = RandomTensor(303, new[] { 128, 8 });
        var sin = RandomTensor(304, new[] { 128, 8 });

        var cpu1 = _cpu.ApplyRoPEInterleaved(x1, cos, sin, startPosition: 0);
        var cpu2 = _cpu.ApplyRoPEInterleaved(x2, cos, sin, startPosition: 4);

        var x1Gpu = _gpu.TensorAddScalar(x1, 0f);
        var x2Gpu = _gpu.TensorAddScalar(x2, 0f);
        var gpu1 = _gpu.ApplyRoPEInterleaved(x1Gpu, cos, sin, startPosition: 0);
        _gpu.InvalidateAllWeightCaches();

        long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
        var gpu2 = _gpu.ApplyRoPEInterleaved(x2Gpu, cos, sin, startPosition: 4);
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

        Assert.True(allocatedBytes < 32_768, $"Expected deferred GPU output without 128 KB host array, allocated {allocatedBytes} bytes.");

        AssertClose(gpu1, cpu1);
        AssertClose(gpu2, cpu2);
    }

    [SkippableFact]
    public void TryAccumulateSumOfSquares_MultiTensorAndPostReadbackReuse_MatchesCpuWithoutSqBufferAlloc()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        var t1 = RandomTensor(401, new[] { 512, 512 }); // 1 MB if materialized on host
        var t2 = RandomTensor(402, new[] { 128, 64 });
        var t3 = RandomTensor(403, new[] { 37 });

        IEngine cpu = _cpu;
        IEngine gpu = _gpu;
        float expectedFirst = cpu.TensorSumOfSquares(t1) + cpu.TensorSumOfSquares(t2);
        float expectedSecond = expectedFirst + cpu.TensorSumOfSquares(t3);

        var g1 = gpu.TensorAddScalar(t1, 0f);
        var g2 = gpu.TensorAddScalar(t2, 0f);
        var g3 = gpu.TensorAddScalar(t3, 0f);

        // Warm up both initial DotProduct (acc == null) and accumulation DotProduct + Add (acc != null) kernels
        Tensor<float> warmAcc = null;
        Assert.True(_gpu.TryAccumulateSumOfSquares(g3, ref warmAcc));
        Assert.True(_gpu.TryAccumulateSumOfSquares(g3, ref warmAcc));
        _ = warmAcc.ToArray()[0];

        Tensor<float> acc = null;
        Assert.True(_gpu.TryAccumulateSumOfSquares(g1, ref acc));

        long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
        Assert.True(_gpu.TryAccumulateSumOfSquares(g2, ref acc));
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

        Assert.True(allocatedBytes < 8_192, $"Expected < 8 KB host allocation for steady-state TryAccumulateSumOfSquares, got {allocatedBytes} bytes.");

        float actualFirst = acc.ToArray()[0];
        Assert.True(Math.Abs(actualFirst - expectedFirst) <= Math.Max(1e-2f, Math.Abs(expectedFirst) * 1e-4f),
            $"First sum-of-squares mismatch: GPU={actualFirst}, CPU={expectedFirst}");

        // Verify accumulating again after ToArray() host readback invalidates stale host cache and updates GPU scalar
        Assert.True(_gpu.TryAccumulateSumOfSquares(g3, ref acc));
        float actualSecond = acc.ToArray()[0];
        Assert.True(Math.Abs(actualSecond - expectedSecond) <= Math.Max(1e-2f, Math.Abs(expectedSecond) * 1e-4f),
            $"Post-readback sum-of-squares mismatch: GPU={actualSecond}, CPU={expectedSecond}");

        // Also verify IEngine.TensorSumOfSquares parity on GPU-resident tensor
        float singleGpu = gpu.TensorSumOfSquares(g1);
        float singleCpu = cpu.TensorSumOfSquares(t1);
        Assert.True(Math.Abs(singleGpu - singleCpu) <= Math.Max(1e-2f, Math.Abs(singleCpu) * 1e-4f),
            $"TensorSumOfSquares mismatch: GPU={singleGpu}, CPU={singleCpu}");
    }

    [Fact]
    public void TensorScope_DetachPreExistingTensor_AvoidsLohKeptListGrowth_AndPreservesScopeCreated()
    {
        var preExisting = RandomTensor(501, new[] { 16 });
        Tensor<float> survivedScopeCreated;

        // Warm up TensorScope TLS
        using (var warm = TensorScope.Begin())
        {
            TensorScope.Detach(preExisting);
        }

        long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
        using (var scope = TensorScope.Begin())
        {
            // Previously, GradientCheckpointing detaching pre-existing model weights pushed every call
            // into _kept (500,000 calls -> 4+ MB LOH List<TensorStorage> backing array).
            for (int i = 0; i < 500_000; i++)
            {
                TensorScope.Detach(preExisting);
            }

            survivedScopeCreated = _cpu.TensorAddScalar(preExisting, 1.0f);
            TensorScope.Detach(survivedScopeCreated);
        }
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

        Assert.True(allocatedBytes < 32_768, $"Expected zero LOH growth from detaching pre-existing tensors, but allocated {allocatedBytes} bytes.");
        Assert.Equal(preExisting.ToArray()[0] + 1.0f, survivedScopeCreated.ToArray()[0], 4);
    }

    [Fact]
    public void DeviceMemoryReclaim_InsideActiveOrSuspendedScope_DefersBlockingGcUntilStepBoundary()
    {
        var prevTapeOpen = DeviceMemoryReclaim.TapeOpen;
        DeviceMemoryReclaim.TapeOpen = () => true;
        try
        {
            DeviceMemoryReclaim.ResetCountsForTests();
            using (var scope = TensorScope.Begin())
            {
                Assert.True(TensorScope.HasActiveOrSuspendedScope);

                // Allocate 2x CollectInsideTapeAfterDeviceBytes (8 GiB virtual counter): inside a deterministic
                // TensorScope, OnDeviceAllocation must mark CollectionDue rather than firing a blocking Gen-2 GC.
                long allocBytes = 2 * DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes;
                long epoch1 = DeviceMemoryReclaim.OnDeviceAllocation(allocBytes);
                Assert.True(DeviceMemoryReclaim.CollectionDue);

                long epoch2;
                using (TensorScope.Suspend())
                {
                    Assert.True(TensorScope.HasActiveOrSuspendedScope);
                    epoch2 = DeviceMemoryReclaim.OnDeviceAllocation(DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes);
                    Assert.True(DeviceMemoryReclaim.CollectionDue);
                }

                // Deterministic scope disposal credits all device bytes back before tape step boundary
                DeviceMemoryReclaim.OnDeviceRelease(allocBytes, epoch1);
                DeviceMemoryReclaim.OnDeviceRelease(DeviceMemoryReclaim.CollectInsideTapeAfterDeviceBytes, epoch2);
            }

            Assert.False(TensorScope.HasActiveOrSuspendedScope);
            Assert.False(DeviceMemoryReclaim.CollectIfDueAtStepBoundary());
            Assert.Equal(0, DeviceMemoryReclaim.BytesSinceGen2CollectionForTests);
        }
        finally
        {
            DeviceMemoryReclaim.TapeOpen = prevTapeOpen;
            DeviceMemoryReclaim.ResetCountsForTests();
        }
    }

    [SkippableFact]
    public void AllocateIntBuffer_WithData_RoundTripsAndReusesPoolAcrossIterations()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");
        var backend = _gpu.GetBackend();
        Skip.If(backend is null, "Backend instance unavailable.");

        var table = RandomTensor(601, new[] { 128, 32 });
        var tableGpu = _gpu.TensorAddScalar(table, 0f);
        var indices = new int[256];
        for (int i = 0; i < indices.Length; i++)
            indices[i] = (i * 7 + 3) % 128;

        var expected = _cpu.TensorIndexSelect(table, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0);

        for (int iter = 0; iter < 32; iter++)
        {
            using var intBuf = backend.AllocateIntBuffer(indices);
            Assert.Equal(indices.Length, intBuf.Size);

            var rawBytes = backend.DownloadByteBuffer(intBuf, indices.Length * sizeof(int));
            var roundTrip = new int[indices.Length];
            Buffer.BlockCopy(rawBytes, 0, roundTrip, 0, rawBytes.Length);
            Assert.Equal(indices, roundTrip);
        }

        var actual = _gpu.TensorIndexSelect(tableGpu, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0);
        AssertClose(actual, expected);
    }

    [Fact]
    public void StreamingTape_NonLeafComposedEmbeddingLookup_PropagatesUpstreamGradients()
    {
        var prev = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _cpu;
        try
        {
            var leftFactor = RandomTensor(701, new[] { 32, 8 });
            var rightFactor = RandomTensor(702, new[] { 8, 16 });
            var floatIds = new Tensor<float>(new[] { 6 }, new Vector<float>(new float[] { 3f, 7f, 3f, 0f, 31f, 12f }));
            var upstreamWeights = RandomTensor(703, new[] { 6, 16 });
            var sources = new[] { leftFactor, rightFactor };

            // Reference: standard non-streaming ComputeGradients
            Tensor<float> expectedLeftGrad, expectedRightGrad;
            using (var refTape = new GradientTape<float>())
            {
                var composedTable = _cpu.TensorMatMul(leftFactor, rightFactor); // non-leaf [32, 16] with GradFn != null
                var lookedUp = _cpu.TensorEmbeddingLookupFromFloatIndices(composedTable, floatIds);
                var loss = _cpu.ReduceSum(_cpu.TensorMultiply(lookedUp, upstreamWeights), new[] { 0, 1 }, keepDims: false);
                var refGrads = refTape.ComputeGradients(loss, sources);
                expectedLeftGrad = refGrads[leftFactor];
                expectedRightGrad = refGrads[rightFactor];
            }

            // Streaming backward previously assigned _gradIndex >= 0 to non-leaf composedTable, causing
            // AccumulateSparseEmbeddingGrad to stash a sparse grad for an intermediate tensor and drop
            // grads[composedTable] completely!
            Tensor<float> actualLeftGrad = null;
            Tensor<float> actualRightGrad = null;
            using (var streamTape = new GradientTape<float>())
            {
                var composedTable = _cpu.TensorMatMul(leftFactor, rightFactor);
                var lookedUp = _cpu.TensorEmbeddingLookupFromFloatIndices(composedTable, floatIds);
                var loss = _cpu.ReduceSum(_cpu.TensorMultiply(lookedUp, upstreamWeights), new[] { 0, 1 }, keepDims: false);
                streamTape.ComputeGradientsStreaming(loss, sources, (src, grad) =>
                {
                    if (ReferenceEquals(src, leftFactor)) actualLeftGrad = grad.Clone();
                    if (ReferenceEquals(src, rightFactor)) actualRightGrad = grad.Clone();
                });
            }

            Assert.NotNull(actualLeftGrad);
            Assert.NotNull(actualRightGrad);
            AssertClose(actualLeftGrad, expectedLeftGrad);
            AssertClose(actualRightGrad, expectedRightGrad);
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }
    }

    [SkippableFact]
    public void TensorTopK_And_TensorClampMinMax_GpuResident_AvoidHostArrayAllocations_AndMatchCpu()
    {
        Skip.If(!_gpuReady, "DirectGpu backend not available.");

        IEngine gpu = _gpu;
        IEngine cpu = _cpu;
        var logits = RandomTensor(801, new[] { 1024, 64 }); // 256 KB
        var logitsGpu = _gpu.TensorAddScalar(logits, 0f);

        // Warm up TopK + Clamp kernels
        _ = gpu.TensorTopK(logitsGpu, 8, axis: 1, out _);
        _ = _gpu.TensorClampMin(logitsGpu, -0.25f);
        _ = _gpu.TensorClampMax(logitsGpu, 0.25f);

        long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
        var topValsGpu = gpu.TensorTopK(logitsGpu, 8, axis: 1, out var topIdxGpu);
        var clampedMinGpu = _gpu.TensorClampMin(logitsGpu, -0.25f);
        var clampedMaxGpu = _gpu.TensorClampMax(clampedMinGpu, 0.25f);
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - beforeBytes;

        // Previously: TensorTopK allocated int[8192] + downloaded via TapeIndexCopy, and ClampMin/Max allocated float[65536] (256 KB LOH each)!
        Assert.True(allocatedBytes < 16_384, $"Expected deferred GPU outputs without host array/LOH allocations, but allocated {allocatedBytes} bytes.");

        var topValsCpu = cpu.TensorTopK(logits, 8, axis: 1, out var topIdxCpu);
        var clampedCpu = _cpu.TensorClampMax(_cpu.TensorClampMin(logits, -0.25f), 0.25f);
        AssertClose(topValsGpu, topValsCpu);
        Assert.Equal(topIdxCpu.ToArray(), topIdxGpu.ToArray());
        AssertClose(clampedMaxGpu, clampedCpu);
    }

    [Fact]
    public void RebindablePlanCache_MultiStepReplay_MatchesFreshWalkGradients()
    {
        IEngine engine = _gpuReady ? _gpu : _cpu;
        var prev = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w1 = RandomTensor(901, new[] { 16, 24 });
            var w2 = RandomTensor(902, new[] { 24, 8 });
            var sources = new[] { w1, w2 };

            for (int step = 0; step < 3; step++)
            {
                var x = RandomTensor(910 + step, new[] { 12, 16 });
                var target = RandomTensor(920 + step, new[] { 12, 8 });

                Dictionary<Tensor<float>, Tensor<float>> replayGrads;
                using (var tape = new GradientTape<float>())
                {
                    var h = engine.TensorReLU(engine.TensorMatMul(x, w1));
                    var y = engine.TensorMatMul(h, w2);
                    var diff = engine.TensorSubtract(y, target);
                    var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                    replayGrads = tape.ComputeGradients(loss, sources);
                }

                // Compare against fresh CPU reference on every step (steps 1 and 2 exercise RebindablePlanCache.TryExecute)
                AiDotNetEngine.Current = _cpu;
                using var cpuTape = new GradientTape<float>();
                var hc = _cpu.TensorReLU(_cpu.TensorMatMul(x, w1));
                var yc = _cpu.TensorMatMul(hc, w2);
                var dc = _cpu.TensorSubtract(yc, target);
                var lc = _cpu.ReduceSum(_cpu.TensorMultiply(dc, dc), new[] { 0, 1 }, keepDims: false);
                var cpuGrads = cpuTape.ComputeGradients(lc, sources);
                AiDotNetEngine.Current = engine;

                AssertClose(replayGrads[w1], cpuGrads[w1]);
                AssertClose(replayGrads[w2], cpuGrads[w2]);
            }
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }
    }
}
#endif
