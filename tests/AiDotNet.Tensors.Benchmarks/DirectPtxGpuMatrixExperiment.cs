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
    private const int LaunchesPerSample = 1;
    private readonly record struct Shape(string Name, int BatchHeads, int Sequence);
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    private static readonly Shape[] Shapes =
    [
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
        using (var probe = new DirectPtxRuntime())
        {
            if (!DirectPtxArchitecture.HasValidatedOnlineAttention(probe.ArchitectureFamily))
            {
                Console.WriteLine(
                    $"Direct PTX online attention has no validated {probe.ArchitectureFamily} specialization.");
                return;
            }
        }

        GpuBenchmarkEnvironment.PrintSnapshot("start");
        Console.WriteLine(
            $"NVIDIA GPU-only steady-state matrix ({LaunchesPerSample} launches/sample, FP16 Q/K/V, FP32 output, D=64)");
        Console.WriteLine($"{"Shape",-11} {"Mode",-8} {"Lane",-10} {"Method",-27} {"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} {"TFLOPS",9} {"B/call",9} {"tmp MiB",9}");
        Console.WriteLine(new string('-', 139));
        RunDirectAndCuBlas();
        RunAiDotNet();
        RunPyTorch();
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
                    using var stats = runtime.AllocateBytes(kernel.StatsBytes);
                    q.Upload<ushort>(qHost);
                    k.Upload<ushort>(kHost);
                    v.Upload<ushort>(vHost);
                    gamma.Upload<float>(Enumerable.Repeat(1f, Dimension).ToArray());
                    beta.Upload<float>(new float[Dimension]);
                    Action launch = () => kernel.Launch(
                        DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]),
                        DirectPtxTensorView.CreateOwned(stats, kernel.Blueprint.Tensors[6]));
                    Distribution time = Measure(runtime.Synchronize, launch);
                    long allocation = Allocation(runtime.Synchronize, launch);
                    Print(shape, causal, fused, "Direct PTX online", time,
                        Tflops(shape, time.Median), allocation, 0);

                    if (!fused)
                    {
                        using var cuBlas = new DirectPtxOnlineAttentionExperiment.CuBlasAttentionComparator(
                            runtime, shape.BatchHeads, shape.Sequence, Dimension, causal, Scale);
                        Action vendor = () => cuBlas.Launch(q, k, v, output);
                        Distribution vendorTime = Measure(runtime.Synchronize, vendor);
                        long vendorAllocation = Allocation(runtime.Synchronize, vendor);
                        long temporary = checked((long)cuBlas.ScoreBytes + (long)cuBlas.ProbabilityBytes);
                        Print(shape, causal, false, "cuBLAS+PTX softmax", vendorTime,
                            Tflops(shape, vendorTime.Median), vendorAllocation, temporary);
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
            int elements = shape.BatchHeads * shape.Sequence * Dimension;
            int rows = shape.BatchHeads * shape.Sequence;
            using var q = backend.AllocateBuffer(ToFloat(RandomHalf(new Random(1000 + elements), elements)));
            using var k = backend.AllocateBuffer(ToFloat(RandomHalf(new Random(2000 + elements), elements)));
            using var v = backend.AllocateBuffer(ToFloat(RandomHalf(new Random(3000 + elements), elements)));
            using var gamma = backend.AllocateBuffer(Enumerable.Repeat(1f, Dimension).ToArray());
            using var beta = backend.AllocateBuffer(new float[Dimension]);
            using var attention = backend.AllocateBuffer(elements);
            using var normalized = backend.AllocateBuffer(elements);
            using var final = backend.AllocateBuffer(elements);
            using var stats = backend.AllocateBuffer(rows);
            using var means = backend.AllocateBuffer(rows);
            using var inverse = backend.AllocateBuffer(rows);
            Action attentionAction = () => backend.FlashAttentionV2(
                q, k, v, attention, stats, 1, shape.BatchHeads,
                shape.Sequence, shape.Sequence, Dimension, Scale, causal);
            Distribution attentionTime = Measure(backend.Synchronize, attentionAction);
            Print(shape, causal, false, "AiDotNet NVRTC Flash", attentionTime,
                Tflops(shape, attentionTime.Median), Allocation(backend.Synchronize, attentionAction),
                rows * sizeof(float));

            Action fusedAction = () =>
            {
                attentionAction();
                backend.LayerNorm(attention, normalized, gamma, beta, means, inverse, rows, Dimension, 1e-5f);
                backend.Gelu(normalized, final, elements);
            };
            Distribution fusedTime = Measure(backend.Synchronize, fusedAction);
            long temporary = (long)rows * sizeof(float) * 3 + (long)elements * sizeof(float) * 2;
            Print(shape, causal, true, "AiDotNet Flash+LN+GELU", fusedTime,
                Tflops(shape, fusedTime.Median), Allocation(backend.Synchronize, fusedAction), temporary);
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
                ("PyTorch Flash-SDPA", true, false),
                ("PyTorch Math-SDPA", false, true),
            })
            {
                torch.backends.cuda.enable_flash_sdp(flash);
                torch.backends.cuda.enable_math_sdp(math);
                foreach (Shape shape in Shapes)
                foreach (bool causal in new[] { false, true })
                {
                    int elements = shape.BatchHeads * shape.Sequence * Dimension;
                    using TorchTensor qFloat = torch.tensor(
                        ToFloat(RandomHalf(new Random(1000 + elements), elements)),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor kFloat = torch.tensor(
                        ToFloat(RandomHalf(new Random(2000 + elements), elements)),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor vFloat = torch.tensor(
                        ToFloat(RandomHalf(new Random(3000 + elements), elements)),
                        [1, shape.BatchHeads, shape.Sequence, Dimension], device: torch.CUDA);
                    using TorchTensor q = qFloat.half();
                    using TorchTensor k = kFloat.half();
                    using TorchTensor v = vFloat.half();
                    using TorchTensor gamma = torch.ones([Dimension], device: torch.CUDA);
                    using TorchTensor beta = torch.zeros([Dimension], device: torch.CUDA);
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
                    Distribution attentionTime = MeasureTorch(Attention);
                    Print(shape, causal, false, backendName, attentionTime,
                        Tflops(shape, attentionTime.Median), AllocationTorch(Attention), -1);
                    Distribution fusedTime = MeasureTorch(Fused);
                    Print(shape, causal, true, backendName + "+LN+GELU", fusedTime,
                        Tflops(shape, fusedTime.Median), AllocationTorch(Fused), -1);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"PyTorch matrix stopped: {ex.GetType().Name}: {ex.Message}");
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
        for (int i = 0; i < 50; i++) action();
        long value = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        synchronize();
        return value;
    }

    private static long AllocationTorch(Func<TorchTensor> action)
    {
        using (TorchTensor t = action()) torch.cuda.synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) using (TorchTensor t = action()) { }
        long value = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        torch.cuda.synchronize();
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
        Distribution d, double tflops, long allocation, long temporaryBytes)
    {
        string temporary = temporaryBytes < 0 ? "n/a" : (temporaryBytes / 1048576.0).ToString("F3");
        Console.WriteLine(
            $"{shape.Name,-11} {(causal ? "causal" : "plain"),-8} {(fused ? "attn+epi" : "attention"),-10} " +
            $"{method,-27} {d.Median * 1000,10:F2} {d.P95 * 1000,10:F2} {d.P99 * 1000,10:F2} " +
            $"{d.Mean * 1000,10:F2} {tflops,9:F3} {allocation,9} {temporary,9}");
    }

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
