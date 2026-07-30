using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Phase-1 experiment for a driver-only, hand-emitted PTX attention kernel.
/// The timed region contains device work only; allocation, upload, PTX JIT,
/// and cuBLAS handle creation are deliberately outside it.
/// </summary>
internal static class DirectPtxAttentionExperiment
{
    private readonly record struct Shape(string Name, int BatchHeads, int M, int N, int K, int Seed);
    private readonly record struct Distribution(double Min, double Mean, double Median, double P95, double P99, double Max);

    private static readonly Shape[] Shapes =
    [
        new("AIsEval MHA",       512,  32,  32, 16, 20260721),
        new("BERT-base MHA",      96, 128, 128, 64, 20260722),
        new("Long-context MHA",   16, 256, 256, 64, 20260723),
        new("Decoder MHA",        12, 512, 512, 64, 20260724),
    ];

    internal static void Run()
    {
        if (!DirectPtxRuntime.IsAvailable)
        {
            Console.WriteLine("NVIDIA CUDA Driver API is unavailable.");
            return;
        }

        using var runtime = new DirectPtxRuntime();
        Console.WriteLine("AiDotNet direct-PTX attention Q*K^T experiment");
        Console.WriteLine($"GPU: {runtime.DeviceName} (SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor})");
        Console.WriteLine("Custom path: emitted PTX -> CUDA Driver JIT -> SASS -> cuLaunchKernel");
        Console.WriteLine("Excluded from custom path: CUDA Runtime, NVRTC, cuBLAS, cuDNN");
        Console.WriteLine("Math: FP16 inputs, FP32 accumulation/output, fused 1/sqrt(K) scale");
        Console.WriteLine();
        Console.WriteLine($"{"Shape",-21} {"Method",-21} {"med us",9} {"p95 us",9} {"p99 us",9} {"mean us",9} {"TF/s",8} {"E2E med",9} {"E2E p95",9} {"E2E p99",9} {"B/call",9}");
        Console.WriteLine(new string('-', 131));

        foreach (Shape shape in Shapes)
            RunShape(runtime, shape);

        Console.WriteLine();
        Console.WriteLine("med/p95/p99/mean are CUDA-event device times; E2E is host launch + synchronize time (microseconds).");
        Console.WriteLine("B/call is managed allocation on the calling thread. Device allocation and input/output buffers are setup-only.");
        Console.WriteLine("AiDotNet per-head floor is the current cublasGemmEx launch granularity without scheduler/list overhead.");
        Console.WriteLine();
        RunCurrentProductionApi();
    }

    private static void RunShape(DirectPtxRuntime runtime, Shape shape)
    {
        float scale = 1.0f / MathF.Sqrt(shape.K);
        var setupTimer = Stopwatch.StartNew();
        using var kernel = new PtxWmmaBatchedQkKernel(
            runtime, shape.M, shape.N, shape.K, shape.BatchHeads, scale);
        setupTimer.Stop();
        using var q = runtime.AllocateBytes(kernel.ABytes);
        using var keys = runtime.AllocateBytes(kernel.BBytes);
        using var ptxScores = runtime.AllocateBytes(kernel.CBytes);
        using var blasScores = runtime.AllocateBytes(kernel.CBytes);

        var random = new Random(shape.Seed);
        ushort[] qHost = RandomHalf(random, checked(shape.BatchHeads * shape.M * shape.K));
        ushort[] kHost = RandomHalf(random, checked(shape.BatchHeads * shape.N * shape.K));
        q.Upload<ushort>(qHost);
        keys.Upload<ushort>(kHost);

        var blasSetupTimer = Stopwatch.StartNew();
        using var blas = new CuBlasQkComparator(runtime, shape, scale);
        blasSetupTimer.Stop();
        kernel.Launch(q, keys, ptxScores);
        blas.LaunchStrided(q, keys, blasScores);
        runtime.Synchronize();
        var error = Validate(ptxScores, blasScores, checked(shape.BatchHeads * shape.M * shape.N), shape.Name);

        Action ptxAction = () => kernel.Launch(q, keys, ptxScores);
        Action blasAction = () => blas.LaunchStrided(q, keys, blasScores);
        Action legacyAction = () => blas.LaunchPerHead(q, keys, blasScores);

        Distribution ptxDevice = Summarize(runtime.MeasureKernelSamples(
            ptxAction, warmup: 30, samples: 101, launchesPerSample: 10));
        Distribution blasDevice = Summarize(runtime.MeasureKernelSamples(
            blasAction, warmup: 30, samples: 101, launchesPerSample: 10));

        // The legacy path can contain hundreds of library launches per sample;
        // use one launch group and 51 samples while retaining p99 visibility.
        Distribution legacyDevice = Summarize(runtime.MeasureKernelSamples(
            legacyAction, warmup: 2, samples: 51, launchesPerSample: 1));

        Distribution ptxE2e = MeasureEndToEnd(runtime, ptxAction, 101);
        Distribution blasE2e = MeasureEndToEnd(runtime, blasAction, 101);
        Distribution legacyE2e = MeasureEndToEnd(runtime, legacyAction, 51);
        long ptxAllocation = MeasureAllocation(runtime, ptxAction, 200);
        long blasAllocation = MeasureAllocation(runtime, blasAction, 200);
        long legacyAllocation = MeasureAllocation(runtime, legacyAction, 20);

        Print(shape.Name, "Direct PTX (new)", ptxDevice, ptxE2e,
            kernel.EffectiveTflops((float)ptxDevice.Median), ptxAllocation);
        Print(shape.Name, "cuBLAS best", blasDevice, blasE2e,
            kernel.EffectiveTflops((float)blasDevice.Median), blasAllocation);
        Print(shape.Name, "AiDotNet per-head floor", legacyDevice, legacyE2e,
            kernel.EffectiveTflops((float)legacyDevice.Median), legacyAllocation);
        Console.WriteLine($"  setup: PTX emit+driver JIT={setupTimer.Elapsed.TotalMilliseconds:F3} ms, " +
            $"cuBLAS handle={blasSetupTimer.Elapsed.TotalMilliseconds:F3} ms, " +
            $"max |PTX-cuBLAS|={error.maxAbsolute:G4}, max relative={error.maxRelative:G4}");
    }

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < length; i++)
            result[i] = BitConverter.HalfToUInt16Bits((Half)(random.NextDouble() * 0.5 - 0.25));
        return result;
    }

    private static (float maxAbsolute, float maxRelative) Validate(
        DirectPtxBuffer ptx, DirectPtxBuffer blas, int length, string name)
    {
        var actual = new float[length];
        var expected = new float[length];
        ptx.Download<float>(actual);
        blas.Download<float>(expected);

        float maxAbsolute = 0;
        float maxRelative = 0;
        for (int i = 0; i < length; i++)
        {
            float absolute = MathF.Abs(actual[i] - expected[i]);
            float relative = absolute / MathF.Max(MathF.Abs(expected[i]), 1e-5f);
            maxAbsolute = MathF.Max(maxAbsolute, absolute);
            maxRelative = MathF.Max(maxRelative, relative);
        }

        if (maxAbsolute > 0.003f)
            throw new InvalidOperationException(
                $"{name}: PTX/cuBLAS mismatch; max abs={maxAbsolute:G9}, max rel={maxRelative:G9}.");
        return (maxAbsolute, maxRelative);
    }

    private static Distribution MeasureEndToEnd(
        DirectPtxRuntime runtime, Action action, int sampleCount)
    {
        for (int i = 0; i < 10; i++) action();
        runtime.Synchronize();
        var samples = new double[sampleCount];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            runtime.Synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        return Summarize(samples);
    }

    private static long MeasureAllocation(DirectPtxRuntime runtime, Action action, int calls)
    {
        action();
        runtime.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < calls; i++)
        {
            action();
            runtime.Synchronize();
        }
        long after = GC.GetAllocatedBytesForCurrentThread();
        return (after - before) / calls;
    }

    private static Distribution Summarize(float[] samples)
    {
        var converted = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++) converted[i] = samples[i];
        return Summarize(converted);
    }

    private static Distribution Summarize(double[] samples)
    {
        var sorted = (double[])samples.Clone();
        Array.Sort(sorted);
        return new Distribution(
            sorted[0],
            samples.Average(),
            Percentile(sorted, 0.50),
            Percentile(sorted, 0.95),
            Percentile(sorted, 0.99),
            sorted[^1]);
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        double fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    private static void Print(
        string shape, string method, Distribution device, Distribution e2e,
        double tflops, long allocatedBytes)
    {
        Console.WriteLine(
            $"{shape,-21} {method,-21} {device.Median * 1000,9:F2} {device.P95 * 1000,9:F2} " +
            $"{device.P99 * 1000,9:F2} {device.Mean * 1000,9:F2} {tflops,8:F3} " +
            $"{e2e.Median * 1000,9:F2} {e2e.P95 * 1000,9:F2} {e2e.P99 * 1000,9:F2} {allocatedBytes,9}");
    }

    private static void RunCurrentProductionApi()
    {
        Console.WriteLine("Current AiDotNet production API (CUDA C/NVRTC backend + per-head cuBLAS scheduler)");
        Console.WriteLine($"{"Shape",-21} {"median us",11} {"p95 us",11} {"p99 us",11} {"mean us",11} {"TF/s",9} {"B/call",12}");
        Console.WriteLine(new string('-', 94));

        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("Current CUDA backend unavailable (it requires both CUDA Driver and NVRTC).");
            return;
        }
        using var pool = new GpuStreamPool(backend);
        using var scheduler = new GpuStreamScheduler(pool);

        foreach (Shape shape in Shapes)
        {
            int qElements = checked(shape.BatchHeads * shape.M * shape.K);
            int kElements = checked(shape.BatchHeads * shape.N * shape.K);
            int scoreElements = checked(shape.BatchHeads * shape.M * shape.N);
            using var q = backend.AllocateByteBuffer(checked(qElements * sizeof(ushort)));
            using var keys = backend.AllocateByteBuffer(checked(kElements * sizeof(ushort)));
            using var scores = backend.AllocateBuffer(scoreElements);

            var random = new Random(shape.Seed);
            backend.UploadByteBuffer(q, ToBytes(RandomHalf(random, qElements)));
            backend.UploadByteBuffer(keys, ToBytes(RandomHalf(random, kElements)));
            float scale = 1.0f / MathF.Sqrt(shape.K);
            Action action = () => backend.MultiHeadAttentionScoresFanoutMixed(
                q, keys, scores,
                batch: 1, numHeads: shape.BatchHeads,
                seqLen: shape.M, headDim: shape.K,
                scheduler,
                useBFloat16: false,
                alpha: scale, beta: 0);

            for (int i = 0; i < 3; i++) action();
            var samples = new double[101];
            for (int i = 0; i < samples.Length; i++)
            {
                long start = Stopwatch.GetTimestamp();
                action();
                backend.Synchronize();
                samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
            Distribution stats = Summarize(samples);

            action();
            backend.Synchronize();
            long before = GC.GetAllocatedBytesForCurrentThread();
            const int allocationCalls = 10;
            for (int i = 0; i < allocationCalls; i++)
            {
                action();
                backend.Synchronize();
            }
            long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / allocationCalls;
            double flops = 2.0 * shape.BatchHeads * shape.M * shape.N * shape.K;
            double tflops = flops / (stats.Median * 1e-3) / 1e12;
            Console.WriteLine(
                $"{shape.Name,-21} {stats.Median * 1000,11:F2} {stats.P95 * 1000,11:F2} " +
                $"{stats.P99 * 1000,11:F2} {stats.Mean * 1000,11:F2} {tflops,9:F3} {allocation,12}");
        }
    }

    private static byte[] ToBytes(ushort[] values)
    {
        var bytes = new byte[checked(values.Length * sizeof(ushort))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private sealed class CuBlasQkComparator : IDisposable
    {
        private readonly DirectPtxRuntime _runtime;
        private readonly Shape _shape;
        private readonly float _scale;
        private IntPtr _handle;

        internal CuBlasQkComparator(DirectPtxRuntime runtime, Shape shape, float scale)
        {
            _runtime = runtime;
            _shape = shape;
            _scale = scale;
            using var _ = runtime.Enter();
            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasCreate(out _handle), "cublasCreate(PTX comparator)");
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSetMathMode(_handle, CuBlasNative.CUBLAS_TENSOR_OP_MATH),
                "cublasSetMathMode(PTX comparator)");
        }

        internal unsafe void LaunchStrided(DirectPtxBuffer q, DirectPtxBuffer keys, DirectPtxBuffer scores)
        {
            using var _ = _runtime.Enter();
            float alpha = _scale, beta = 0;
            AiDotNet.Tensors.Engines.CublasStatus status = CuBlasNative.cublasGemmStridedBatchedEx(
                _handle,
                CublasOperation.Transpose, CublasOperation.None,
                _shape.N, _shape.M, _shape.K,
                (IntPtr)(&alpha),
                keys.Pointer, CuBlasNative.CUDA_R_16F, _shape.K, (long)_shape.N * _shape.K,
                q.Pointer, CuBlasNative.CUDA_R_16F, _shape.K, (long)_shape.M * _shape.K,
                (IntPtr)(&beta),
                scores.Pointer, CuBlasNative.CUDA_R_32F, _shape.N, (long)_shape.M * _shape.N,
                _shape.BatchHeads,
                CuBlasNative.CUBLAS_COMPUTE_32F,
                0);
            CuBlasNative.CheckCublasStatus(status, "cublasGemmStridedBatchedEx(QK^T)");
        }

        internal unsafe void LaunchPerHead(DirectPtxBuffer q, DirectPtxBuffer keys, DirectPtxBuffer scores)
        {
            using var _ = _runtime.Enter();
            float alpha = _scale, beta = 0;
            long qStrideBytes = (long)_shape.M * _shape.K * sizeof(ushort);
            long kStrideBytes = (long)_shape.N * _shape.K * sizeof(ushort);
            long cStrideBytes = (long)_shape.M * _shape.N * sizeof(float);
            for (int batch = 0; batch < _shape.BatchHeads; batch++)
            {
                AiDotNet.Tensors.Engines.CublasStatus status = CuBlasNative.cublasGemmEx(
                    _handle,
                    CublasOperation.Transpose, CublasOperation.None,
                    _shape.N, _shape.M, _shape.K,
                    (IntPtr)(&alpha),
                    Add(keys.Pointer, batch * kStrideBytes), CuBlasNative.CUDA_R_16F, _shape.K,
                    Add(q.Pointer, batch * qStrideBytes), CuBlasNative.CUDA_R_16F, _shape.K,
                    (IntPtr)(&beta),
                    Add(scores.Pointer, batch * cStrideBytes), CuBlasNative.CUDA_R_32F, _shape.N,
                    CuBlasNative.CUBLAS_COMPUTE_32F,
                    0);
                CuBlasNative.CheckCublasStatus(status, "cublasGemmEx(per-head QK^T)");
            }
        }

        private static IntPtr Add(IntPtr pointer, long byteOffset) =>
            new(checked(pointer.ToInt64() + byteOffset));

        public void Dispose()
        {
            if (_handle == IntPtr.Zero) return;
            using var _ = _runtime.Enter();
            CuBlasNative.CheckCublasStatus(CuBlasNative.cublasDestroy(_handle), "cublasDestroy(PTX comparator)");
            _handle = IntPtr.Zero;
        }
    }
}
