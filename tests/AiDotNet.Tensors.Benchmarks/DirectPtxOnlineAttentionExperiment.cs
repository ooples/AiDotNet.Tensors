using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Hardware smoke test for the async online S=128,D=64 specialization.</summary>
internal static class DirectPtxOnlineAttentionExperiment
{
    private const int BatchHeads = 128;
    private const int Sequence = 128;
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const float AttentionMaxAbsoluteTolerance = 5e-4f;
    private const float FusedMaxAbsoluteTolerance = 3e-3f;
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Error(float MaxAbsolute, float MaxRelative);

    internal static void Run()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("online-attention-start");
        GpuBenchmarkEnvironment.PrintSnapshot("start");
        using var runtime = new DirectPtxRuntime();
        // Model a resident execution thread: establish the context once so
        // host timing measures launch/sync, not repeated context switching.
        Console.WriteLine("Async online FlashAttention championship: [BH=128,S=128,D=64]");
        Console.WriteLine($"GPU: {runtime.DeviceName} (SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor})");
        Console.WriteLine(
            "Physical direct/cuBLAS/TorchSharp lanes use FP16 Q/K/V; current AiDotNet NVRTC uses " +
            "the same half-rounded values in FP32 storage and is reported separately.");
        Console.WriteLine(
            $"Correctness max-absolute tolerance: attention={AttentionMaxAbsoluteTolerance:G4}; " +
            $"attention+epilogue={FusedMaxAbsoluteTolerance:G4}.");
        Console.WriteLine($"{"Mode",-9} {"Method",-52} {"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"GFLOPS",10} {"TFLOPS",9} {"managed B",10} {"tmp MiB",9} {"max abs",10} {"max rel",10} {"regs",6} {"static B",8} {"dynamic B",9} {"local B",8} {"occ",7}");
        Console.WriteLine(new string('-', 220));

        using (runtime.Enter())
        {
        foreach (bool causal in new[] { false, true })
        foreach (bool epilogue in new[] { false, true })
        {
            GpuBenchmarkEnvironment.RequireNoForeignCompute(
                $"online-attention-{(causal ? "causal" : "plain")}-{(epilogue ? "epilogue" : "attention")}");
            using var kernel = new PtxOnlineFusedAttention128x64Kernel(
                runtime, BatchHeads, causal, epilogue, Scale,
                emitSoftmaxStats: false);
            using var q = runtime.AllocateBytes(kernel.QBytes);
            using var k = runtime.AllocateBytes(kernel.KBytes);
            using var v = runtime.AllocateBytes(kernel.VBytes);
            using var gamma = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
            using var beta = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);

            var random = new Random(1771);
            ushort[] qHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
            ushort[] kHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
            ushort[] vHost = RandomHalf(random, BatchHeads * Sequence * Dimension);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            float[] betaHost = Enumerable.Range(0, Dimension).Select(i => (i - 32) / 512f).ToArray();
            q.Upload<ushort>(qHost);
            k.Upload<ushort>(kHost);
            v.Upload<ushort>(vHost);
            gamma.Upload<float>(gammaHost);
            beta.Upload<float>(betaHost);

            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]),
                default);

            launch();
            runtime.Synchronize();
            var actual = new float[BatchHeads * Sequence * Dimension];
            output.Download<float>(actual);
            Error error = Validate(actual, qHost, kHost, vHost, gammaHost, betaHost, causal, epilogue);
            Distribution device = Summarize(runtime.MeasureKernelSamples(
                launch, warmup: 100, samples: 101, launchesPerSample: 10));
            Distribution e2e = MeasureEndToEnd(runtime, launch);
            long allocation = MeasureAllocation(runtime, launch);
            string mode = causal ? "causal" : "unmasked";
            string suffix = epilogue ? "+LN+GELU" : string.Empty;
            double occupancy = kernel.Audit.ActiveBlocksPerMultiprocessor *
                kernel.Audit.BlockThreads / (double)runtime.MaxThreadsPerMultiprocessor;
            Print(mode, $"Direct PTX online{suffix} [device]", device,
                kernel.AttentionTflops((float)device.Median), allocation, 0, error,
                kernel.FunctionInfo.RegistersPerThread, kernel.FunctionInfo.StaticSharedBytes,
                0,
                kernel.FunctionInfo.LocalBytesPerThread, occupancy);
            Print(mode, $"Direct PTX online{suffix} [E2E]", e2e,
                kernel.AttentionTflops((float)e2e.Median), allocation, 0, error,
                kernel.FunctionInfo.RegistersPerThread, kernel.FunctionInfo.StaticSharedBytes,
                0,
                kernel.FunctionInfo.LocalBytesPerThread, occupancy);

            if (!epilogue)
            {
                using var cuBlas = new CuBlasAttentionComparator(
                    runtime, BatchHeads, Sequence, Dimension, causal, Scale);
                Action cuBlasLaunch = () => cuBlas.Launch(q, k, v, output);
                cuBlasLaunch();
                runtime.Synchronize();
                output.Download<float>(actual);
                Error cuBlasError = Validate(
                    actual, qHost, kHost, vHost, gammaHost, betaHost,
                    causal, epilogue: false);
                Distribution cuBlasDevice = Summarize(runtime.MeasureKernelSamples(
                    cuBlasLaunch, warmup: 100, samples: 101, launchesPerSample: 10));
                Distribution cuBlasE2e = MeasureEndToEnd(runtime, cuBlasLaunch);
                long cuBlasAllocation = MeasureAllocation(runtime, cuBlasLaunch);
                long temporaries = checked((long)cuBlas.ScoreBytes + (long)cuBlas.ProbabilityBytes);
                Print(mode, "cuBLAS GEMMs + PTX softmax [device]", cuBlasDevice,
                    EffectiveTflops(cuBlasDevice.Median), cuBlasAllocation,
                    temporaries, cuBlasError);
                Print(mode, "cuBLAS GEMMs + PTX softmax [E2E]", cuBlasE2e,
                    EffectiveTflops(cuBlasE2e.Median), cuBlasAllocation,
                    temporaries, cuBlasError);
            }
        }
        }

        GpuBenchmarkEnvironment.RequireIdleGpu("online-attention-framework-baselines");
        RunAiDotNetNvrtc();
        RunTorchSharp();
        GpuBenchmarkEnvironment.RequireNoForeignCompute("online-attention-end");
        GpuBenchmarkEnvironment.PrintSnapshot("end");
    }

    private static void RunAiDotNetNvrtc()
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        int elements = BatchHeads * Sequence * Dimension;
        int rows = BatchHeads * Sequence;
        var random = new Random(1771);
        ushort[] qHalf = RandomHalf(random, elements);
        ushort[] kHalf = RandomHalf(random, elements);
        ushort[] vHalf = RandomHalf(random, elements);
        float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Enumerable.Range(0, Dimension).Select(i => (i - 32) / 512f).ToArray();
        using var q = backend.AllocateBuffer(ToFloat(qHalf));
        using var k = backend.AllocateBuffer(ToFloat(kHalf));
        using var v = backend.AllocateBuffer(ToFloat(vHalf));
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var attention = backend.AllocateBuffer(elements);
        using var normalized = backend.AllocateBuffer(elements);
        using var final = backend.AllocateBuffer(elements);
        using var stats = backend.AllocateBuffer(rows);
        using var means = backend.AllocateBuffer(rows);
        using var inverseVariances = backend.AllocateBuffer(rows);

        foreach (bool causal in new[] { false, true })
        {
            Action attentionAction = () => backend.FlashAttentionV2(
                q, k, v, attention, stats,
                1, BatchHeads, Sequence, Sequence, Dimension, Scale, causal);
            attentionAction();
            backend.Synchronize();
            Error attentionError = Validate(
                backend.DownloadBuffer(attention), qHalf, kHalf, vHalf,
                gammaHost, betaHost, causal, epilogue: false);
            Distribution attentionTime = MeasureEndToEnd(backend, attentionAction);
            long attentionAllocation = MeasureAllocation(backend, attentionAction);
            Print(causal ? "causal" : "unmasked", "AiDotNet NVRTC FlashAttn [E2E]", attentionTime,
                EffectiveTflops(attentionTime.Median), attentionAllocation,
                rows * sizeof(float), attentionError);

            Action composition = () =>
            {
                backend.FlashAttentionV2(
                    q, k, v, attention, stats,
                    1, BatchHeads, Sequence, Sequence, Dimension, Scale, causal);
                backend.LayerNorm(
                    attention, normalized, gamma, beta, means, inverseVariances,
                    rows, Dimension, 1e-5f);
                backend.Gelu(normalized, final, elements);
            };
            composition();
            backend.Synchronize();
            Error fusedError = Validate(
                backend.DownloadBuffer(final), qHalf, kHalf, vHalf,
                gammaHost, betaHost, causal, epilogue: true);
            Distribution compositionTime = MeasureEndToEnd(backend, composition);
            long compositionAllocation = MeasureAllocation(backend, composition);
            long temporaryBytes =
                (long)rows * sizeof(float) * 3 + (long)elements * sizeof(float) * 2;
            Print(causal ? "causal" : "unmasked", "AiDotNet NVRTC+LN+GELU [E2E]", compositionTime,
                EffectiveTflops(compositionTime.Median), compositionAllocation,
                temporaryBytes, fusedError);
        }
    }

    private static void RunTorchSharp()
    {
        try
        {
            if (!torch.cuda.is_available()) return;
            int elements = BatchHeads * Sequence * Dimension;
            var random = new Random(1771);
            ushort[] qHalf = RandomHalf(random, elements);
            ushort[] kHalf = RandomHalf(random, elements);
            ushort[] vHalf = RandomHalf(random, elements);
            float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
            float[] betaHost = Enumerable.Range(0, Dimension).Select(i => (i - 32) / 512f).ToArray();
            using TorchTensor qFloat = torch.tensor(ToFloat(qHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor kFloat = torch.tensor(ToFloat(kHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor vFloat = torch.tensor(ToFloat(vHalf), [1, BatchHeads, Sequence, Dimension], device: torch.CUDA);
            using TorchTensor q = qFloat.half();
            using TorchTensor k = kFloat.half();
            using TorchTensor v = vFloat.half();
            using TorchTensor gamma = torch.tensor(gammaHost, device: torch.CUDA);
            using TorchTensor beta = torch.tensor(betaHost, device: torch.CUDA);
            bool oldFlash = torch.backends.cuda.flash_sdp_enabled();
            bool oldMath = torch.backends.cuda.math_sdp_enabled();
            try
            {
                torch.backends.cuda.enable_flash_sdp(true);
                torch.backends.cuda.enable_math_sdp(false);
                foreach (bool causal in new[] { false, true })
                {
                    TorchTensor Attention()
                    {
                        using TorchTensor half = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, null, 0.0, causal);
                        return half.to_type(torch.ScalarType.Float32);
                    }
                    TorchTensor Composition()
                    {
                        using TorchTensor attention = Attention();
                        using TorchTensor normalized = torch.nn.functional.layer_norm(
                            attention, [Dimension], gamma, beta, 1e-5);
                        return TanhGelu(normalized);
                    }

                    using (TorchTensor check = Attention())
                    using (TorchTensor checkCpu = check.cpu())
                    {
                        Error error = Validate(checkCpu.data<float>().ToArray(), qHalf, kHalf, vHalf,
                            gammaHost, betaHost, causal, epilogue: false);
                        Distribution time = MeasureTorch(Attention);
                        long allocation = MeasureTorchAllocation(Attention);
                        Print(causal ? "causal" : "unmasked", "TorchSharp flash-preferred SDPA [E2E]", time,
                            EffectiveTflops(time.Median), allocation, -1, error);
                    }

                    using (TorchTensor check = Composition())
                    using (TorchTensor checkCpu = check.cpu())
                    {
                        Error error = Validate(checkCpu.data<float>().ToArray(), qHalf, kHalf, vHalf,
                            gammaHost, betaHost, causal, epilogue: true);
                        Distribution time = MeasureTorch(Composition);
                        long allocation = MeasureTorchAllocation(Composition);
                        Print(causal ? "causal" : "unmasked", "TorchSharp flash-preferred SDPA+LN+GELU [E2E]", time,
                            EffectiveTflops(time.Median), allocation, -1, error);
                    }
                }
            }
            finally
            {
                torch.backends.cuda.enable_flash_sdp(oldFlash);
                torch.backends.cuda.enable_math_sdp(oldMath);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TorchSharp flash-preferred baseline unavailable: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static TorchTensor TanhGelu(TorchTensor x)
    {
        using TorchTensor square = x.mul(x);
        using TorchTensor cube = square.mul(x);
        using TorchTensor corrected = cube.mul(0.044715).add(x);
        using TorchTensor scaled = corrected.mul(0.7978845608028654);
        using TorchTensor activated = scaled.tanh().add(1.0);
        return activated.mul(x).mul_(0.5);
    }

    private static Error Validate(
        float[] actual,
        ushort[] q,
        ushort[] k,
        ushort[] v,
        float[] gamma,
        float[] beta,
        bool causal,
        bool epilogue)
    {
        var scores = new float[Sequence];
        var expectedRow = new float[Dimension];
        float maxAbsolute = 0;
        float maxRelative = 0;
        foreach (int head in new[] { 0, BatchHeads / 2, BatchHeads - 1 })
        for (int row = 0; row < Sequence; row++)
        {
            int headOffset = head * Sequence * Dimension;
            int lastKey = causal ? row : Sequence - 1;
            float maximum = float.NegativeInfinity;
            for (int column = 0; column <= lastKey; column++)
            {
                float score = 0;
                for (int d = 0; d < Dimension; d++)
                    score += HalfAt(q, headOffset + row * Dimension + d) *
                        HalfAt(k, headOffset + column * Dimension + d);
                scores[column] = score * Scale;
                maximum = MathF.Max(maximum, scores[column]);
            }

            float sum = 0;
            for (int column = 0; column <= lastKey; column++)
            {
                scores[column] = MathF.Exp(scores[column] - maximum);
                sum += scores[column];
            }
            for (int d = 0; d < Dimension; d++)
            {
                float value = 0;
                for (int column = 0; column <= lastKey; column++)
                    value += scores[column] / sum *
                        HalfAt(v, headOffset + column * Dimension + d);
                expectedRow[d] = value;
            }

            if (epilogue)
            {
                float mean = expectedRow.Average();
                float variance = 0;
                for (int d = 0; d < Dimension; d++)
                    variance += (expectedRow[d] - mean) * (expectedRow[d] - mean);
                float inverseStd = 1f / MathF.Sqrt(variance / Dimension + 1e-5f);
                for (int d = 0; d < Dimension; d++)
                {
                    float x = (expectedRow[d] - mean) * inverseStd * gamma[d] + beta[d];
                    expectedRow[d] = 0.5f * x *
                        (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                }
            }

            for (int d = 0; d < Dimension; d++)
            {
                float actualValue = actual[headOffset + row * Dimension + d];
                float absolute = MathF.Abs(actualValue - expectedRow[d]);
                float relative = 2f * absolute /
                    (MathF.Abs(actualValue) + MathF.Abs(expectedRow[d]) + 1e-3f);
                maxAbsolute = MathF.Max(maxAbsolute, absolute);
                maxRelative = MathF.Max(maxRelative, relative);
            }
        }

        if (!float.IsFinite(maxAbsolute) || !float.IsFinite(maxRelative) ||
            maxAbsolute > (epilogue ? FusedMaxAbsoluteTolerance : AttentionMaxAbsoluteTolerance))
            throw new InvalidOperationException(
                $"Online attention validation failed: max abs error {maxAbsolute:G9}, " +
                $"max symmetric relative error {maxRelative:G9}.");
        return new Error(maxAbsolute, maxRelative);
    }

    private static Distribution MeasureEndToEnd(DirectPtxRuntime runtime, Action action)
    {
        for (int i = 0; i < 30; i++) action();
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

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 30; i++) action();
        backend.Synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureTorch(Func<TorchTensor> action)
    {
        for (int i = 0; i < 30; i++) using (TorchTensor warmup = action()) { }
        torch.cuda.synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            using TorchTensor result = action();
            torch.cuda.synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        return Summarize(samples);
    }

    private static long MeasureAllocation(DirectPtxRuntime runtime, Action action)
    {
        action();
        runtime.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        const int calls = 100;
        for (int i = 0; i < calls; i++) { action(); runtime.Synchronize(); }
        long result = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;
        return result;
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        const int calls = 100;
        for (int i = 0; i < calls; i++) { action(); backend.Synchronize(); }
        long result = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;
        return result;
    }

    private static long MeasureTorchAllocation(Func<TorchTensor> action)
    {
        using (TorchTensor warmup = action()) torch.cuda.synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        const int calls = 100;
        for (int i = 0; i < calls; i++)
        {
            using TorchTensor result = action();
            torch.cuda.synchronize();
        }
        long allocated = (GC.GetAllocatedBytesForCurrentThread() - before) / calls;
        return allocated;
    }

    private static Distribution Summarize(float[] samples) =>
        Summarize(samples.Select(static value => (double)value).ToArray());

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

    private static void Print(
        string mode,
        string method,
        Distribution distribution,
        double tflops,
        long allocation,
        long temporaryBytes,
        Error error,
        int registers = 0,
        int staticSharedBytes = 0,
        int dynamicSharedBytes = 0,
        int localBytes = 0,
        double occupancy = double.NaN)
    {
        string temporary = temporaryBytes < 0 ? "n/a" : (temporaryBytes / 1048576.0).ToString("F3");
        string registerText = registers == 0 ? "n/a" : registers.ToString();
        string staticSharedText = registers == 0 ? "n/a" : staticSharedBytes.ToString();
        string dynamicSharedText = registers == 0 ? "n/a" : dynamicSharedBytes.ToString();
        string localText = registers == 0 ? "n/a" : localBytes.ToString();
        string occupancyText = double.IsNaN(occupancy) ? "n/a" : occupancy.ToString("P0");
        Console.WriteLine(
            $"{mode,-9} {method,-52} {distribution.Median * 1000,10:F2} " +
            $"{distribution.P95 * 1000,10:F2} {distribution.P99 * 1000,10:F2} " +
            $"{distribution.Mean * 1000,10:F2} {tflops * 1000,10:F2} {tflops,9:F3} {allocation,10} " +
            $"{temporary,9} {error.MaxAbsolute,10:G4} {error.MaxRelative,10:G4} " +
            $"{registerText,6} {staticSharedText,8} {dynamicSharedText,9} " +
            $"{localText,8} {occupancyText,7}");
    }

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.HalfToUInt16Bits((Half)((random.NextSingle() - 0.5f) * 0.5f));
        return result;
    }

    private static float HalfAt(ushort[] source, int index) =>
        (float)BitConverter.UInt16BitsToHalf(source[index]);

    private static float[] ToFloat(ushort[] source)
    {
        var result = new float[source.Length];
        for (int i = 0; i < result.Length; i++) result[i] = HalfAt(source, i);
        return result;
    }

    /// <summary>
    /// Vendor-library attention boundary: Tensor-Core strided-batched cuBLAS
    /// Q*K^T, shape-baked PTX softmax, Tensor-Core cuBLAS P*V.
    /// </summary>
    internal sealed class CuBlasAttentionComparator : IDisposable
    {
        private readonly DirectPtxRuntime _runtime;
        private readonly PtxAttentionSoftmax32Kernel _softmax;
        private readonly DirectPtxBuffer _scores;
        private readonly DirectPtxBuffer _probabilities;
        private readonly int _batchHeads;
        private readonly int _sequence;
        private readonly int _dimension;
        private readonly float _scale;
        private IntPtr _handle;

        internal nuint ScoreBytes => _softmax.ScoreBytes;
        internal nuint ProbabilityBytes => _softmax.ProbabilityBytes;

        internal CuBlasAttentionComparator(
            DirectPtxRuntime runtime,
            int batchHeads,
            int sequence,
            int dimension,
            bool isCausal,
            float scale)
        {
            _runtime = runtime;
            _batchHeads = batchHeads;
            _sequence = sequence;
            _dimension = dimension;
            _scale = scale;
            _softmax = new PtxAttentionSoftmax32Kernel(
                runtime, batchHeads, isCausal, sequence);
            _scores = runtime.AllocateBytes(_softmax.ScoreBytes);
            _probabilities = runtime.AllocateBytes(_softmax.ProbabilityBytes);
            using var _ = runtime.Enter();
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasCreate(out _handle), "cublasCreate(online-attention comparator)");
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasSetMathMode(_handle, CuBlasNative.CUBLAS_TENSOR_OP_MATH),
                "cublasSetMathMode(online-attention comparator)");
        }

        internal unsafe void Launch(
            DirectPtxBuffer q,
            DirectPtxBuffer k,
            DirectPtxBuffer v,
            DirectPtxBuffer output)
        {
            using var _ = _runtime.Enter();
            float alphaQk = _scale;
            float alphaPv = 1f;
            float beta = 0f;
            int inputStride = _sequence * _dimension;
            int scoreStride = _sequence * _sequence;

            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasGemmStridedBatchedEx(
                    _handle,
                    CublasOperation.Transpose, CublasOperation.None,
                    _sequence, _sequence, _dimension,
                    (IntPtr)(&alphaQk),
                    k.Pointer, CuBlasNative.CUDA_R_16F, _dimension, inputStride,
                    q.Pointer, CuBlasNative.CUDA_R_16F, _dimension, inputStride,
                    (IntPtr)(&beta),
                    _scores.Pointer, CuBlasNative.CUDA_R_32F, _sequence, scoreStride,
                    _batchHeads, CuBlasNative.CUBLAS_COMPUTE_32F, 0),
                "cublasGemmStridedBatchedEx(QK^T)");
            _softmax.Launch(_scores, _probabilities);
            CuBlasNative.CheckCublasStatus(
                CuBlasNative.cublasGemmStridedBatchedEx(
                    _handle,
                    CublasOperation.None, CublasOperation.None,
                    _dimension, _sequence, _sequence,
                    (IntPtr)(&alphaPv),
                    v.Pointer, CuBlasNative.CUDA_R_16F, _dimension, inputStride,
                    _probabilities.Pointer, CuBlasNative.CUDA_R_16F, _sequence, scoreStride,
                    (IntPtr)(&beta),
                    output.Pointer, CuBlasNative.CUDA_R_32F, _dimension, inputStride,
                    _batchHeads, CuBlasNative.CUBLAS_COMPUTE_32F, 0),
                "cublasGemmStridedBatchedEx(PV)");
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                using var _ = _runtime.Enter();
                CuBlasNative.CheckCublasStatus(
                    CuBlasNative.cublasDestroy(_handle), "cublasDestroy(online-attention comparator)");
                _handle = IntPtr.Zero;
            }
            _probabilities.Dispose();
            _scores.Dispose();
            _softmax.Dispose();
        }
    }
}
