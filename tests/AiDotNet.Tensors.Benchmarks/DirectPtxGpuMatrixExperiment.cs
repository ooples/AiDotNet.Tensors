using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// NVIDIA-only, apples-to-apples matrix for the online attention family.
/// Every row executes resident GPU buffers and the same mathematical lane.
/// </summary>
internal static class DirectPtxGpuMatrixExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const float AttentionMaxAbsoluteTolerance = 5e-4f;
    private const float FusedMaxAbsoluteTolerance = 3e-3f;
    private const int LaunchesPerSample = 1;
    private const int DeviceLaunchesPerSample = 10;
    private readonly record struct Shape(string Name, int BatchHeads, int Sequence);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Error(float MaxAbsolute, float MaxRelative);

    private static readonly float[] GammaHost = Enumerable.Range(0, Dimension)
        .Select(static i => 0.75f + i / 256f).ToArray();
    private static readonly float[] BetaHost = Enumerable.Range(0, Dimension)
        .Select(static i => (i - 32) / 512f).ToArray();

    private static readonly Shape[] Shapes =
    [
        new("single-s16", 1, 16),
        new("single-s128", 1, 128),
        new("decode", 12, 16),
        new("short", 12, 32),
        new("medium", 12, 64),
        new("bert", 12, 128),
        new("throughput", 128, 128),
        new("saturation", 512, 128),
    ];

    internal static void Run()
    {
        if (!DirectPtxRuntime.IsAvailable)
        {
            Console.WriteLine("NVIDIA CUDA Driver API is unavailable.");
            return;
        }
        GpuBenchmarkEnvironment.RequireIdleGpu("gpu-matrix-start");
        using (var probe = new DirectPtxRuntime())
        {
            if (!DirectPtxArchitecture.HasValidatedOnlineAttention(
                probe.ComputeCapabilityMajor, probe.ComputeCapabilityMinor))
            {
                Console.WriteLine(
                    $"Direct PTX online attention has no validated {probe.ArchitectureFamily} specialization.");
                return;
            }
        }
        GpuBenchmarkEnvironment.PrintSnapshot("start");
        Console.WriteLine(
            $"NVIDIA GPU-only steady-state matrix (E2E={LaunchesPerSample}, " +
            $"device={DeviceLaunchesPerSample} launches/sample; FP16 physical lane, FP32 output, D=64)");
        Console.WriteLine(
            "Direct PTX/cuBLAS/TorchSharp use FP16 Q/K/V. Current AiDotNet NVRTC rows use the same " +
            "half-rounded values in FP32 storage and are a separately labeled framework baseline.");
        Console.WriteLine(
            "TorchSharp rows are backend preferences; use --direct-ptx-external-gpu-baselines for forced PyTorch backends.");
        Console.WriteLine(
            $"Correctness max-absolute tolerance: attention={AttentionMaxAbsoluteTolerance:G4}; " +
            $"attention+epilogue={FusedMaxAbsoluteTolerance:G4}.");
        Console.WriteLine($"{"Shape",-11} {"Mode",-8} {"Lane",-10} {"Method",-52} {"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"GFLOPS",10} {"TFLOPS",9} {"managed B",9} {"tmp MiB",9} {"max abs",10} {"max rel",10} {"regs",6} {"static B",8} {"dynamic B",9} {"local B",8} {"occ",7}");
        Console.WriteLine(new string('-', 242));
        RunDirectAndCuBlas();
        GpuBenchmarkEnvironment.RequireIdleGpu("gpu-matrix-framework-baselines");
        RunAiDotNet();
        RunPyTorch();
        GpuBenchmarkEnvironment.RequireNoForeignCompute("gpu-matrix-end");
        GpuBenchmarkEnvironment.PrintSnapshot("end");
    }

    private static void RunDirectAndCuBlas()
    {
        using var runtime = new DirectPtxRuntime();
        using (runtime.Enter())
        {
            foreach (Shape shape in Shapes)
            foreach (bool causal in new[] { false, true })
            {
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"gpu-matrix-direct-{shape.Name}-{(causal ? "causal" : "plain")}");
                int elements = shape.BatchHeads * shape.Sequence * Dimension;
                ushort[] qHost = RandomHalf(new Random(1000 + elements), elements);
                ushort[] kHost = RandomHalf(new Random(2000 + elements), elements);
                ushort[] vHost = RandomHalf(new Random(3000 + elements), elements);
                foreach (bool fused in new[] { false, true })
                {
                    using var kernel = new PtxOnlineFusedAttention128x64Kernel(
                        runtime, shape.BatchHeads, causal, fused,
                        Scale, 1e-5f, shape.Sequence, emitSoftmaxStats: false);
                    using var q = runtime.AllocateBytes(kernel.QBytes);
                    using var k = runtime.AllocateBytes(kernel.KBytes);
                    using var v = runtime.AllocateBytes(kernel.VBytes);
                    using var gamma = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
                    using var beta = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
                    using var output = runtime.AllocateBytes(kernel.OutputBytes);
                    q.Upload<ushort>(qHost);
                    k.Upload<ushort>(kHost);
                    v.Upload<ushort>(vHost);
                    gamma.Upload<float>(GammaHost);
                    beta.Upload<float>(BetaHost);
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
                    var actual = new float[elements];
                    output.Download<float>(actual);
                    Error error = Validate(
                        actual, qHost, kHost, vHost, shape.BatchHeads, shape.Sequence, causal, fused);
                    Distribution deviceTime = Summarize(runtime.MeasureKernelSamples(
                        launch, warmup: 30, samples: 101,
                        launchesPerSample: DeviceLaunchesPerSample)
                        .Select(static value => (double)value).ToArray());
                    Distribution time = Measure(runtime.Synchronize, launch);
                    long allocation = Allocation(runtime.Synchronize, launch);
                    double occupancy = kernel.Audit.ActiveBlocksPerMultiprocessor *
                        kernel.Audit.BlockThreads / (double)runtime.MaxThreadsPerMultiprocessor;
                    string directMethod = $"Direct PTX online w{kernel.WarpsPerBlock}";
                    Print(shape, causal, fused, directMethod + " [device]", deviceTime,
                        Tflops(shape, deviceTime.Median), allocation, 0, error,
                        kernel.FunctionInfo.RegistersPerThread,
                        kernel.FunctionInfo.StaticSharedBytes,
                        0,
                        kernel.FunctionInfo.LocalBytesPerThread,
                        occupancy);
                    Print(shape, causal, fused, directMethod + " [E2E]", time,
                        Tflops(shape, time.Median), allocation, 0, error,
                        kernel.FunctionInfo.RegistersPerThread,
                        kernel.FunctionInfo.StaticSharedBytes,
                        0,
                        kernel.FunctionInfo.LocalBytesPerThread,
                        occupancy);

                    if (!fused)
                    {
                        using var cuBlas = new DirectPtxOnlineAttentionExperiment.CuBlasAttentionComparator(
                            runtime, shape.BatchHeads, shape.Sequence, Dimension, causal, Scale);
                        Action vendor = () => cuBlas.Launch(q, k, v, output);
                        vendor();
                        runtime.Synchronize();
                        output.Download<float>(actual);
                        Error vendorError = Validate(
                            actual, qHost, kHost, vHost,
                            shape.BatchHeads, shape.Sequence, causal, fused: false);
                        Distribution vendorDeviceTime = Summarize(runtime.MeasureKernelSamples(
                            vendor, warmup: 30, samples: 101,
                            launchesPerSample: DeviceLaunchesPerSample)
                            .Select(static value => (double)value).ToArray());
                        Distribution vendorTime = Measure(runtime.Synchronize, vendor);
                        long vendorAllocation = Allocation(runtime.Synchronize, vendor);
                        long temporary = checked((long)cuBlas.ScoreBytes + (long)cuBlas.ProbabilityBytes);
                        Print(shape, causal, false, "cuBLAS+PTX [device]", vendorDeviceTime,
                            Tflops(shape, vendorDeviceTime.Median), vendorAllocation, temporary,
                            vendorError);
                        Print(shape, causal, false, "cuBLAS+PTX [E2E]", vendorTime,
                            Tflops(shape, vendorTime.Median), vendorAllocation, temporary, vendorError);
                    }
                }
            }
        }
    }

    private static void RunAiDotNet()
    {
        using var backend = new CudaBackend();
        foreach (Shape shape in Shapes)
        foreach (bool causal in new[] { false, true })
        {
            GpuBenchmarkEnvironment.RequireNoForeignCompute(
                $"gpu-matrix-aidotnet-{shape.Name}-{(causal ? "causal" : "plain")}");
            int elements = shape.BatchHeads * shape.Sequence * Dimension;
            int rows = shape.BatchHeads * shape.Sequence;
            ushort[] qHost = RandomHalf(new Random(1000 + elements), elements);
            ushort[] kHost = RandomHalf(new Random(2000 + elements), elements);
            ushort[] vHost = RandomHalf(new Random(3000 + elements), elements);
            using var q = backend.AllocateBuffer(ToFloat(qHost));
            using var k = backend.AllocateBuffer(ToFloat(kHost));
            using var v = backend.AllocateBuffer(ToFloat(vHost));
            using var gamma = backend.AllocateBuffer(GammaHost);
            using var beta = backend.AllocateBuffer(BetaHost);
            using var attention = backend.AllocateBuffer(elements);
            using var normalized = backend.AllocateBuffer(elements);
            using var final = backend.AllocateBuffer(elements);
            using var stats = backend.AllocateBuffer(rows);
            using var means = backend.AllocateBuffer(rows);
            using var inverse = backend.AllocateBuffer(rows);
            Action attentionAction = () => backend.FlashAttentionV2(
                q, k, v, attention, stats, 1, shape.BatchHeads,
                shape.Sequence, shape.Sequence, Dimension, Scale, causal);
            attentionAction();
            backend.Synchronize();
            Error attentionError = Validate(
                backend.DownloadBuffer(attention), qHost, kHost, vHost,
                shape.BatchHeads, shape.Sequence, causal, fused: false);
            Distribution attentionDeviceTime = MeasureDevice(backend, attentionAction);
            Distribution attentionTime = Measure(backend.Synchronize, attentionAction);
            long attentionAllocation = Allocation(backend.Synchronize, attentionAction);
            Print(shape, causal, false, "AiDotNet NVRTC [device]", attentionDeviceTime,
                Tflops(shape, attentionDeviceTime.Median), attentionAllocation,
                rows * sizeof(float), attentionError);
            Print(shape, causal, false, "AiDotNet NVRTC [E2E]", attentionTime,
                Tflops(shape, attentionTime.Median), attentionAllocation,
                rows * sizeof(float), attentionError);

            Action fusedAction = () =>
            {
                attentionAction();
                backend.LayerNorm(attention, normalized, gamma, beta, means, inverse, rows, Dimension, 1e-5f);
                backend.Gelu(normalized, final, elements);
            };
            fusedAction();
            backend.Synchronize();
            Error fusedError = Validate(
                backend.DownloadBuffer(final), qHost, kHost, vHost,
                shape.BatchHeads, shape.Sequence, causal, fused: true);
            Distribution fusedDeviceTime = MeasureDevice(backend, fusedAction);
            Distribution fusedTime = Measure(backend.Synchronize, fusedAction);
            long temporary = (long)rows * sizeof(float) * 3 + (long)elements * sizeof(float) * 2;
            long fusedAllocation = Allocation(backend.Synchronize, fusedAction);
            Print(shape, causal, true, "AiDotNet fused [device]", fusedDeviceTime,
                Tflops(shape, fusedDeviceTime.Median), fusedAllocation, temporary, fusedError);
            Print(shape, causal, true, "AiDotNet fused [E2E]", fusedTime,
                Tflops(shape, fusedTime.Median), fusedAllocation, temporary,
                fusedError);
        }
    }

    private static void RunPyTorch()
    {
        if (!torch.cuda.is_available()) return;
        bool oldFlash = torch.backends.cuda.flash_sdp_enabled();
        bool oldMath = torch.backends.cuda.math_sdp_enabled();
        try
        {
            foreach ((string backendName, bool flash, bool math) in new[]
            {
                ("TorchSharp flash-preferred SDPA", true, false),
                ("TorchSharp flash-disabled SDPA", false, true),
            })
            {
                torch.backends.cuda.enable_flash_sdp(flash);
                torch.backends.cuda.enable_math_sdp(math);
                foreach (Shape shape in Shapes)
                foreach (bool causal in new[] { false, true })
                {
                    GpuBenchmarkEnvironment.RequireNoForeignCompute(
                        $"gpu-matrix-torchsharp-{shape.Name}-{(causal ? "causal" : "plain")}");
                    int elements = shape.BatchHeads * shape.Sequence * Dimension;
                    ushort[] qHost = RandomHalf(new Random(1000 + elements), elements);
                    ushort[] kHost = RandomHalf(new Random(2000 + elements), elements);
                    ushort[] vHost = RandomHalf(new Random(3000 + elements), elements);
                    using TorchTensor qFloat = torch.tensor(
                        ToFloat(qHost),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor kFloat = torch.tensor(
                        ToFloat(kHost),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor vFloat = torch.tensor(
                        ToFloat(vHost),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor q = qFloat.half();
                    using TorchTensor k = kFloat.half();
                    using TorchTensor v = vFloat.half();
                    using TorchTensor gamma = torch.tensor(GammaHost, device: torch.CUDA);
                    using TorchTensor beta = torch.tensor(BetaHost, device: torch.CUDA);
                    TorchTensor Attention()
                    {
                        using TorchTensor half = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, null, 0.0, causal);
                        return half.to_type(torch.ScalarType.Float32);
                    }
                    TorchTensor Fused()
                    {
                        using TorchTensor a = Attention();
                        using TorchTensor n = torch.nn.functional.layer_norm(
                            a, [Dimension], gamma, beta, 1e-5);
                        return TanhGelu(n);
                    }
                    Error attentionError;
                    using (TorchTensor check = Attention())
                    using (TorchTensor checkCpu = check.cpu())
                        attentionError = Validate(
                            checkCpu.data<float>().ToArray(), qHost, kHost, vHost,
                            shape.BatchHeads, shape.Sequence, causal, fused: false);
                    Distribution attentionTime = MeasureTorch(Attention);
                    Print(shape, causal, false, backendName + " [E2E]", attentionTime,
                        Tflops(shape, attentionTime.Median), AllocationTorch(Attention), -1,
                        attentionError);
                    Error fusedError;
                    using (TorchTensor check = Fused())
                    using (TorchTensor checkCpu = check.cpu())
                        fusedError = Validate(
                            checkCpu.data<float>().ToArray(), qHost, kHost, vHost,
                            shape.BatchHeads, shape.Sequence, causal, fused: true);
                    Distribution fusedTime = MeasureTorch(Fused);
                    Print(shape, causal, true, backendName + "+LN+GELU [E2E]", fusedTime,
                        Tflops(shape, fusedTime.Median), AllocationTorch(Fused), -1,
                        fusedError);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TorchSharp matrix stopped: {ex.GetType().Name}: {ex.Message}");
        }
        finally
        {
            torch.backends.cuda.enable_flash_sdp(oldFlash);
            torch.backends.cuda.enable_math_sdp(oldMath);
        }
    }

    // TorchSharp 0.106 exposes only exact GELU. Spell out PyTorch's
    // approximate="tanh" equation so this lane matches AiDotNet and PTX.
    private static TorchTensor TanhGelu(TorchTensor x)
    {
        using TorchTensor square = x.mul(x);
        using TorchTensor cube = square.mul(x);
        using TorchTensor corrected = cube.mul(0.044715).add(x);
        using TorchTensor scaled = corrected.mul(0.7978845608028654);
        using TorchTensor activated = scaled.tanh().add(1.0);
        return activated.mul(x).mul_(0.5);
    }

    private static Distribution Measure(Action synchronize, Action action)
    {
        for (int i = 0; i < 30; i++) action();
        synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int launch = 0; launch < LaunchesPerSample; launch++) action();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 30; i++) action();
        backend.Synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            using var start = backend.CreateEvent(enableTiming: true);
            using var stop = backend.CreateEvent(enableTiming: true);
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < DeviceLaunchesPerSample; launch++) action();
            backend.RecordEvent(stop, backend.DefaultStream);
            stop.Synchronize();
            samples[i] = backend.GetEventElapsedTime(start, stop) / DeviceLaunchesPerSample;
        }
        return Summarize(samples);
    }

    private static Distribution MeasureTorch(Func<TorchTensor> action)
    {
        for (int i = 0; i < 30; i++) using (TorchTensor t = action()) { }
        torch.cuda.synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int launch = 0; launch < LaunchesPerSample; launch++)
                using (TorchTensor t = action()) { }
            torch.cuda.synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds / LaunchesPerSample;
        }
        return Summarize(samples);
    }

    private static long Allocation(Action synchronize, Action action)
    {
        action(); synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) { action(); synchronize(); }
        long value = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        return value;
    }

    private static long AllocationTorch(Func<TorchTensor> action)
    {
        using (TorchTensor t = action()) torch.cuda.synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++)
        {
            using TorchTensor t = action();
            torch.cuda.synchronize();
        }
        long value = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        return value;
    }

    private static Distribution Summarize(double[] values)
    {
        double[] sorted = (double[])values.Clone();
        Array.Sort(sorted);
        return new Distribution(values.Average(), P(sorted, .5), P(sorted, .95), P(sorted, .99));
    }

    private static double P(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double Tflops(Shape shape, double milliseconds) =>
        4.0 * shape.BatchHeads * shape.Sequence * shape.Sequence * Dimension /
        (milliseconds * 1e-3) / 1e12;

    private static void Print(
        Shape shape, bool causal, bool fused, string method,
        Distribution d, double tflops, long allocation, long temporaryBytes,
        Error error, int registers = 0, int staticSharedBytes = 0,
        int dynamicSharedBytes = 0,
        int localBytes = 0, double occupancy = double.NaN)
    {
        string temporary = temporaryBytes < 0 ? "n/a" : (temporaryBytes / 1048576.0).ToString("F3");
        string registerText = registers == 0 ? "n/a" : registers.ToString();
        string staticSharedText = registers == 0 ? "n/a" : staticSharedBytes.ToString();
        string dynamicSharedText = registers == 0 ? "n/a" : dynamicSharedBytes.ToString();
        string localText = registers == 0 ? "n/a" : localBytes.ToString();
        string occupancyText = double.IsNaN(occupancy) ? "n/a" : occupancy.ToString("P0");
        Console.WriteLine(
            $"{shape.Name,-11} {(causal ? "causal" : "plain"),-8} {(fused ? "attn+epi" : "attention"),-10} " +
            $"{method,-52} {d.Median * 1000,10:F2} {d.P95 * 1000,10:F2} {d.P99 * 1000,10:F2} " +
            $"{d.Mean * 1000,10:F2} {tflops * 1000,10:F2} {tflops,9:F3} {allocation,9} {temporary,9} " +
            $"{error.MaxAbsolute,10:G4} {error.MaxRelative,10:G4} {registerText,6} " +
            $"{staticSharedText,8} {dynamicSharedText,9} {localText,8} {occupancyText,7}");
    }

    private static Error Validate(
        float[] actual, ushort[] q, ushort[] k, ushort[] v,
        int batchHeads, int sequence, bool causal, bool fused)
    {
        int[] heads = batchHeads <= 3
            ? Enumerable.Range(0, batchHeads).ToArray()
            : [0, batchHeads / 2, batchHeads - 1];
        var scores = new float[sequence];
        var expected = new float[Dimension];
        float maxAbsolute = 0;
        float maxRelative = 0;
        foreach (int head in heads)
        {
            int headOffset = head * sequence * Dimension;
            for (int row = 0; row < sequence; row++)
            {
                int lastKey = causal ? row : sequence - 1;
                float maximum = float.NegativeInfinity;
                for (int column = 0; column <= lastKey; column++)
                {
                    float score = 0;
                    for (int d = 0; d < Dimension; d++)
                    {
                        int queryIndex = headOffset + row * Dimension + d;
                        int keyIndex = headOffset + column * Dimension + d;
                        score += Half(q[queryIndex]) * Half(k[keyIndex]);
                    }
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
                    {
                        int valueIndex = headOffset + column * Dimension + d;
                        value += scores[column] / sum * Half(v[valueIndex]);
                    }
                    expected[d] = value;
                }

                if (fused)
                {
                    float mean = expected.Average();
                    float variance = 0;
                    for (int d = 0; d < Dimension; d++)
                        variance += (expected[d] - mean) * (expected[d] - mean);
                    float inverseStd = 1f / MathF.Sqrt(variance / Dimension + 1e-5f);
                    for (int d = 0; d < Dimension; d++)
                    {
                        float value = (expected[d] - mean) * inverseStd * GammaHost[d] + BetaHost[d];
                        expected[d] = 0.5f * value *
                            (1f + MathF.Tanh(0.7978845608f *
                                (value + 0.044715f * value * value * value)));
                    }
                }

                for (int d = 0; d < Dimension; d++)
                {
                    int actualIndex = headOffset + row * Dimension + d;
                    float absolute = MathF.Abs(actual[actualIndex] - expected[d]);
                    float relative = 2f * absolute /
                        (MathF.Abs(actual[actualIndex]) + MathF.Abs(expected[d]) + 1e-3f);
                    maxAbsolute = MathF.Max(maxAbsolute, absolute);
                    maxRelative = MathF.Max(maxRelative, relative);
                }
            }
        }

        float tolerance = fused ? FusedMaxAbsoluteTolerance : AttentionMaxAbsoluteTolerance;
        if (!float.IsFinite(maxAbsolute) || !float.IsFinite(maxRelative) || maxAbsolute > tolerance)
            throw new InvalidOperationException(
                $"GPU matrix validation failed: max abs error {maxAbsolute:G9}, " +
                $"max symmetric relative error {maxRelative:G9}, tolerance {tolerance:G9}.");
        return new Error(maxAbsolute, maxRelative);
    }

    private static float Half(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.HalfToUInt16Bits((Half)((random.NextSingle() - 0.5f) * 0.5f));
        return result;
    }

    private static float[] ToFloat(ushort[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = (float)BitConverter.UInt16BitsToHalf(values[i]);
        return result;
    }
}
