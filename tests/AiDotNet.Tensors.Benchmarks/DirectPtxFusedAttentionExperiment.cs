using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Championship-cell experiment for a complete resident attention forward:
/// [BH=512,S=32,D=16], FP16 Q/K/V, FP32 output. The custom kernel never
/// materializes scores or probabilities in global memory.
/// </summary>
internal static class DirectPtxFusedAttentionExperiment
{
    private const int BatchHeads = 512;
    private const int Sequence = 32;
    private const int Dimension = 16;
    private const float Scale = 0.25f;

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    internal static void Run()
    {
        if (!DirectPtxRuntime.IsAvailable)
        {
            Console.WriteLine("NVIDIA CUDA Driver API is unavailable.");
            return;
        }

        using var runtime = new DirectPtxRuntime();
        Console.WriteLine("Direct-PTX fused-attention championship cell");
        Console.WriteLine($"GPU: {runtime.DeviceName} (SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor})");
        Console.WriteLine("Shape: batch-heads=512, sequence=32, head-dim=16");
        Console.WriteLine("Math: FP16 Q/K/V, FP32 accumulation/output; scores and probabilities are shared-memory-only");
        Console.WriteLine();
        Console.WriteLine($"{"Mode",-12} {"Method",-34} {"med us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"GF/s",9} {"B/call",11} {"tmp MiB",9} {"max err",11}");
        Console.WriteLine(new string('-', 133));

        RunMode(runtime, isCausal: false);
        RunMode(runtime, isCausal: true);
        RunCurrentAiDotNet();
        RunTorchSharp();
    }

    private static void RunMode(DirectPtxRuntime runtime, bool isCausal)
    {
        using var kernel = new PtxWmmaFusedAttention32x16Kernel(runtime, BatchHeads, isCausal, Scale);
        using var q = runtime.AllocateBytes(kernel.QBytes);
        using var k = runtime.AllocateBytes(kernel.KBytes);
        using var v = runtime.AllocateBytes(kernel.VBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        using var cuBlasOutput = runtime.AllocateBytes(kernel.OutputBytes);
        using var cuBlas = new CuBlasAttentionComparator(runtime, BatchHeads, isCausal, Scale);

        var random = new Random(isCausal ? 20260728 : 20260727);
        ushort[] qHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
        ushort[] kHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
        ushort[] vHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
        q.Upload<ushort>(qHost);
        k.Upload<ushort>(kHost);
        v.Upload<ushort>(vHost);

        Action action = () => kernel.Launch(q, k, v, output);
        action();
        runtime.Synchronize();
        float maxError = ValidateFirstHeads(output, qHost, kHost, vHost, isCausal, headsToCheck: 4);

        Distribution device = Summarize(runtime.MeasureKernelSamples(
            action, warmup: 100, samples: 101, launchesPerSample: 10));
        Distribution e2e = MeasureEndToEnd(runtime, action);
        long allocation = MeasureAllocation(runtime, action);
        Print(isCausal ? "causal" : "unmasked", "Direct PTX fused (new)", device,
            kernel.EffectiveTflops((float)device.Median), allocation, temporaryBytes: 0, maxError);
        Print(isCausal ? "causal" : "unmasked", "Direct PTX fused E2E", e2e,
            kernel.EffectiveTflops((float)e2e.Median), allocation, temporaryBytes: 0, maxError);

        Action cuBlasAction = () => cuBlas.Launch(q, k, v, cuBlasOutput);
        cuBlasAction();
        runtime.Synchronize();
        float cuBlasError = ValidateFirstHeads(
            cuBlasOutput, qHost, kHost, vHost, isCausal, headsToCheck: 4);
        Distribution cuBlasDevice = Summarize(runtime.MeasureKernelSamples(
            cuBlasAction, warmup: 100, samples: 101, launchesPerSample: 10));
        Distribution cuBlasE2e = MeasureEndToEnd(runtime, cuBlasAction);
        long cuBlasAllocation = MeasureAllocation(runtime, cuBlasAction);
        long temporaryBytes = checked((long)cuBlas.ScoreBytes + (long)cuBlas.ProbabilityBytes);
        Print(isCausal ? "causal" : "unmasked", "cuBLAS GEMMs + PTX softmax", cuBlasDevice,
            EffectiveTflops(cuBlasDevice.Median), cuBlasAllocation, temporaryBytes, cuBlasError);
        Print(isCausal ? "causal" : "unmasked", "cuBLAS composition E2E", cuBlasE2e,
            EffectiveTflops(cuBlasE2e.Median), cuBlasAllocation, temporaryBytes, cuBlasError);
    }

    private static void RunCurrentAiDotNet()
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("Current AiDotNet CUDA backend unavailable.");
            return;
        }

        int elements = BatchHeads * Sequence * Dimension;
        var random = new Random(20260727);
        ushort[] qHalf = RandomHalf(random, elements);
        ushort[] kHalf = RandomHalf(random, elements);
        ushort[] vHalf = RandomHalf(random, elements);
        float[] qHost = ToFloat(qHalf);
        float[] kHost = ToFloat(kHalf);
        float[] vHost = ToFloat(vHalf);

        using var q = backend.AllocateBuffer(qHost);
        using var k = backend.AllocateBuffer(kHost);
        using var v = backend.AllocateBuffer(vHost);
        using var output = backend.AllocateBuffer(elements);
        using var stats = backend.AllocateBuffer(BatchHeads * Sequence);

        foreach (bool isCausal in new[] { false, true })
        {
            Action action = () => backend.FlashAttentionV2(
                q, k, v, output, stats,
                batch: 1, numHeads: BatchHeads,
                seqQ: Sequence, seqK: Sequence, headDim: Dimension,
                scale: Scale, isCausal: isCausal);

            for (int i = 0; i < 20; i++) action();
            backend.Synchronize();
            float maxError = ValidateFirstHeads(
                backend.DownloadBuffer(output), qHalf, kHalf, vHalf, isCausal, headsToCheck: 4);
            var samples = new double[101];
            for (int i = 0; i < samples.Length; i++)
            {
                long start = Stopwatch.GetTimestamp();
                action();
                backend.Synchronize();
                samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
            Distribution e2e = Summarize(samples);

            action();
            backend.Synchronize();
            long before = GC.GetAllocatedBytesForCurrentThread();
            const int calls = 50;
            for (int i = 0; i < calls; i++)
            {
                action();
                backend.Synchronize();
            }
            long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;

            double tflops = EffectiveTflops(e2e.Median);
            Print(isCausal ? "causal" : "unmasked", "AiDotNet NVRTC FlashAttn E2E", e2e,
                tflops, allocation, BatchHeads * Sequence * sizeof(float), maxError);
        }
    }

    private static float ValidateFirstHeads(
        DirectPtxBuffer output, ushort[] q, ushort[] k, ushort[] v, bool isCausal, int headsToCheck)
    {
        var actual = new float[BatchHeads * Sequence * Dimension];
        output.Download<float>(actual);
        return ValidateFirstHeads(actual, q, k, v, isCausal, headsToCheck);
    }

    private static float ValidateFirstHeads(
        float[] actual, ushort[] q, ushort[] k, ushort[] v, bool isCausal, int headsToCheck)
    {
        float maxError = 0;
        Span<float> scores = stackalloc float[Sequence];
        for (int bh = 0; bh < headsToCheck; bh++)
        for (int row = 0; row < Sequence; row++)
        {
            int lastKey = isCausal ? row : Sequence - 1;
            float rowMax = float.NegativeInfinity;
            for (int column = 0; column <= lastKey; column++)
            {
                float score = 0;
                for (int inner = 0; inner < Dimension; inner++)
                {
                    score += HalfAt(q, (bh * Sequence + row) * Dimension + inner) *
                        HalfAt(k, (bh * Sequence + column) * Dimension + inner);
                }
                scores[column] = score * Scale;
                rowMax = MathF.Max(rowMax, scores[column]);
            }

            float sum = 0;
            for (int column = 0; column <= lastKey; column++)
            {
                scores[column] = MathF.Exp(scores[column] - rowMax);
                sum += scores[column];
            }
            for (int outputColumn = 0; outputColumn < Dimension; outputColumn++)
            {
                float expected = 0;
                for (int column = 0; column <= lastKey; column++)
                    expected += scores[column] / sum *
                        HalfAt(v, (bh * Sequence + column) * Dimension + outputColumn);
                float got = actual[(bh * Sequence + row) * Dimension + outputColumn];
                maxError = MathF.Max(maxError, MathF.Abs(got - expected));
            }
        }

        if (maxError > 0.003f)
            throw new InvalidOperationException($"Fused attention validation failed: max abs error {maxError:G9}.");
        return maxError;
    }

    private static void RunTorchSharp()
    {
        try
        {
            if (!torch.cuda.is_available())
            {
                Console.WriteLine("TorchSharp/libtorch CUDA is unavailable; PyTorch eager baseline skipped.");
                return;
            }

            int elements = BatchHeads * Sequence * Dimension;
            var random = new Random(20260727);
            ushort[] qHalf = RandomHalf(random, elements);
            ushort[] kHalf = RandomHalf(random, elements);
            ushort[] vHalf = RandomHalf(random, elements);
            using TorchTensor qFloat = torch.tensor(
                ToFloat(qHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor kFloat = torch.tensor(
                ToFloat(kHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor vFloat = torch.tensor(
                ToFloat(vHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor q = qFloat.half();
            using TorchTensor k = kFloat.half();
            using TorchTensor v = vFloat.half();

            foreach (bool isCausal in new[] { false, true })
            {
                TorchTensor? mask = null;
                if (isCausal)
                {
                    var maskHost = new float[Sequence * Sequence];
                    for (int row = 0; row < Sequence; row++)
                    for (int column = row + 1; column < Sequence; column++)
                        maskHost[row * Sequence + column] = float.NegativeInfinity;
                    using TorchTensor maskFloat = torch.tensor(
                        maskHost, [Sequence, Sequence], device: torch.CUDA);
                    mask = maskFloat.half();
                }

                try
                {
                    using (TorchTensor validation = TorchAttention(q, k, v, mask))
                    using (TorchTensor validationCpu = validation.cpu())
                    {
                        float[] actual = validationCpu.data<float>().ToArray();
                        float error = ValidateFirstHeads(
                            actual, qHalf, kHalf, vHalf, isCausal, headsToCheck: 4);

                        for (int i = 0; i < 20; i++)
                        {
                            using TorchTensor warmup = TorchAttention(q, k, v, mask);
                        }
                        torch.cuda.synchronize();

                        var samples = new double[101];
                        for (int i = 0; i < samples.Length; i++)
                        {
                            long start = Stopwatch.GetTimestamp();
                            using TorchTensor result = TorchAttention(q, k, v, mask);
                            torch.cuda.synchronize();
                            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                        }
                        Distribution e2e = Summarize(samples);

                        using (TorchTensor allocationWarmup = TorchAttention(q, k, v, mask))
                            torch.cuda.synchronize();
                        long before = GC.GetAllocatedBytesForCurrentThread();
                        const int calls = 50;
                        for (int i = 0; i < calls; i++)
                        {
                            using TorchTensor result = TorchAttention(q, k, v, mask);
                            torch.cuda.synchronize();
                        }
                        long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;

                        long minimumTemporaries =
                            2L * BatchHeads * Sequence * Sequence * sizeof(ushort) +
                            (long)BatchHeads * Sequence * Dimension * sizeof(ushort);
                        Print(isCausal ? "causal" : "unmasked", "PyTorch eager composition E2E", e2e,
                            EffectiveTflops(e2e.Median), allocation, minimumTemporaries, error);
                    }
                }
                finally
                {
                    mask?.Dispose();
                }
            }

            bool previousFlash = torch.backends.cuda.flash_sdp_enabled();
            bool previousMath = torch.backends.cuda.math_sdp_enabled();
            try
            {
                torch.backends.cuda.enable_flash_sdp(true);
                torch.backends.cuda.enable_math_sdp(false);
                foreach (bool isCausal in new[] { false, true })
                    RunTorchSharpFused(q, k, v, qHalf, kHalf, vHalf, isCausal);
            }
            finally
            {
                torch.backends.cuda.enable_flash_sdp(previousFlash);
                torch.backends.cuda.enable_math_sdp(previousMath);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TorchSharp/libtorch baseline unavailable: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static TorchTensor TorchAttention(
        TorchTensor q, TorchTensor k, TorchTensor v, TorchTensor? mask)
    {
        using TorchTensor kt = k.transpose(2, 3);
        using TorchTensor scores = torch.matmul(q, kt);
        scores.mul_(Scale);
        if (mask is not null) scores.add_(mask);
        using TorchTensor probabilities = torch.nn.functional.softmax(scores, -1);
        using TorchTensor halfOutput = torch.matmul(probabilities, v);
        return halfOutput.to_type(torch.ScalarType.Float32);
    }

    private static void RunTorchSharpFused(
        TorchTensor q, TorchTensor k, TorchTensor v,
        ushort[] qHost, ushort[] kHost, ushort[] vHost, bool isCausal)
    {
        using (TorchTensor validation = TorchFusedAttention(q, k, v, isCausal))
        using (TorchTensor validationFloat = validation.to_type(torch.ScalarType.Float32))
        using (TorchTensor validationCpu = validationFloat.cpu())
        {
            float error = ValidateFirstHeads(
                validationCpu.data<float>().ToArray(), qHost, kHost, vHost, isCausal, headsToCheck: 4);
            for (int i = 0; i < 20; i++)
            {
                using TorchTensor warmup = TorchFusedAttention(q, k, v, isCausal);
            }
            torch.cuda.synchronize();

            var samples = new double[101];
            for (int i = 0; i < samples.Length; i++)
            {
                long start = Stopwatch.GetTimestamp();
                using TorchTensor result = TorchFusedAttention(q, k, v, isCausal);
                torch.cuda.synchronize();
                samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
            Distribution e2e = Summarize(samples);

            using (TorchTensor allocationWarmup = TorchFusedAttention(q, k, v, isCausal))
                torch.cuda.synchronize();
            long before = GC.GetAllocatedBytesForCurrentThread();
            const int calls = 50;
            for (int i = 0; i < calls; i++)
            {
                using TorchTensor result = TorchFusedAttention(q, k, v, isCausal);
                torch.cuda.synchronize();
            }
            long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;
            Print(isCausal ? "causal" : "unmasked", "TorchSharp flash-preferred SDPA E2E", e2e,
                EffectiveTflops(e2e.Median), allocation, temporaryBytes: -1, error);
        }
    }

    private static TorchTensor TorchFusedAttention(
        TorchTensor q, TorchTensor k, TorchTensor v, bool isCausal) =>
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, null, 0.0, isCausal);

    private static Distribution MeasureEndToEnd(DirectPtxRuntime runtime, Action action)
    {
        for (int i = 0; i < 20; i++) action();
        runtime.Synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            runtime.Synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        return Summarize(samples);
    }

    private static long MeasureAllocation(DirectPtxRuntime runtime, Action action)
    {
        action();
        runtime.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        const int calls = 200;
        for (int i = 0; i < calls; i++)
        {
            action();
            runtime.Synchronize();
        }
        long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;
        return allocation;
    }

    private static void Print(
        string mode, string method, Distribution distribution,
        double tflops, long allocation, long temporaryBytes, double maxError)
    {
        string error = double.IsNaN(maxError) ? "n/a" : maxError.ToString("G4");
        string temporary = temporaryBytes < 0
            ? "n/a"
            : (temporaryBytes / 1048576.0).ToString("F3");
        Console.WriteLine(
            $"{mode,-12} {method,-34} {distribution.Median * 1000,10:F2} " +
            $"{distribution.P95 * 1000,10:F2} {distribution.P99 * 1000,10:F2} " +
            $"{distribution.Mean * 1000,10:F2} {tflops * 1000,9:F2} {allocation,11} " +
            $"{temporary,9} {error,11}");
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
        return new Distribution(samples.Average(), Percentile(sorted, 0.50),
            Percentile(sorted, 0.95), Percentile(sorted, 0.99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double EffectiveTflops(double milliseconds)
    {
        double flops = 4.0 * BatchHeads * Sequence * Sequence * Dimension;
        return flops / (milliseconds * 1e-3) / 1e12;
    }

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < length; i++)
            result[i] = BitConverter.HalfToUInt16Bits((Half)(random.NextDouble() * 0.5 - 0.25));
        return result;
    }

    private static float[] ToFloat(ushort[] source)
    {
        var result = new float[source.Length];
        for (int i = 0; i < source.Length; i++) result[i] = HalfAt(source, i);
        return result;
    }

    private static float HalfAt(ushort[] source, int index) =>
        (float)BitConverter.UInt16BitsToHalf(source[index]);

    /// <summary>
    /// Best-case decomposed vendor-library boundary: one strided batched
    /// cuBLAS Tensor Core GEMM for Q*K^T, a shape-specialized warp softmax,
    /// then one strided batched cuBLAS Tensor Core GEMM for P*V.
    /// </summary>
    private sealed class CuBlasAttentionComparator : IDisposable
    {
        private readonly DirectPtxRuntime _runtime;
        private readonly PtxAttentionSoftmax32Kernel _softmax;
        private readonly DirectPtxBuffer _scores;
        private readonly DirectPtxBuffer _probabilities;
        private readonly float _scale;
        private IntPtr _handle;

        internal nuint ScoreBytes => _softmax.ScoreBytes;
        internal nuint ProbabilityBytes => _softmax.ProbabilityBytes;

        internal CuBlasAttentionComparator(
            DirectPtxRuntime runtime, int batchHeads, bool isCausal, float scale)
        {
            _runtime = runtime;
            _scale = scale;
            _softmax = new PtxAttentionSoftmax32Kernel(runtime, batchHeads, isCausal);
            _scores = runtime.AllocateBytes(_softmax.ScoreBytes);
            _probabilities = runtime.AllocateBytes(_softmax.ProbabilityBytes);
            using var _ = runtime.Enter();
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasCreate(out _handle), "cublasCreate(fused-attention comparator)");
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSetMathMode(_handle, CuBlasNative.CUBLAS_TENSOR_OP_MATH),
                "cublasSetMathMode(fused-attention comparator)");
        }

        internal unsafe void Launch(
            DirectPtxBuffer q, DirectPtxBuffer k, DirectPtxBuffer v, DirectPtxBuffer output)
        {
            using var _ = _runtime.Enter();
            float alphaQk = _scale;
            float alphaPv = 1f;
            float beta = 0f;

            AiDotNet.Tensors.Engines.CublasStatus qkStatus = CuBlasNative.cublasGemmStridedBatchedEx(
                _handle,
                CublasOperation.Transpose, CublasOperation.None,
                Sequence, Sequence, Dimension,
                (IntPtr)(&alphaQk),
                k.Pointer, CuBlasNative.CUDA_R_16F, Dimension, Sequence * Dimension,
                q.Pointer, CuBlasNative.CUDA_R_16F, Dimension, Sequence * Dimension,
                (IntPtr)(&beta),
                _scores.Pointer, CuBlasNative.CUDA_R_32F, Sequence, Sequence * Sequence,
                BatchHeads,
                CuBlasNative.CUBLAS_COMPUTE_32F,
                0);
            CuBlasNative.CheckCublasStatus(qkStatus, "cublasGemmStridedBatchedEx(QK^T)");

            _softmax.Launch(_scores, _probabilities);

            // Row-major P*V is column-major V^T*P^T to cuBLAS.
            AiDotNet.Tensors.Engines.CublasStatus pvStatus = CuBlasNative.cublasGemmStridedBatchedEx(
                _handle,
                CublasOperation.None, CublasOperation.None,
                Dimension, Sequence, Sequence,
                (IntPtr)(&alphaPv),
                v.Pointer, CuBlasNative.CUDA_R_16F, Dimension, Sequence * Dimension,
                _probabilities.Pointer, CuBlasNative.CUDA_R_16F, Sequence, Sequence * Sequence,
                (IntPtr)(&beta),
                output.Pointer, CuBlasNative.CUDA_R_32F, Dimension, Sequence * Dimension,
                BatchHeads,
                CuBlasNative.CUBLAS_COMPUTE_32F,
                0);
            CuBlasNative.CheckCublasStatus(pvStatus, "cublasGemmStridedBatchedEx(PV)");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                using var _ = _runtime.Enter();
                CuBlasNative.CheckCublasStatus(
                    CuBlasNative.cublasDestroy(_handle), "cublasDestroy(fused-attention comparator)");
                _handle = IntPtr.Zero;
            }
            _probabilities.Dispose();
            _scores.Dispose();
            _softmax.Dispose();
        }
    }
}
