using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#845 NVIDIA-only FP32-to-FP16 cast benchmark. Compares the hand-emitted
/// direct-PTX kernel against the current AiDotNet CUDA convert_fp32_to_fp16
/// kernel and the strongest resident PyTorch <c>.to(float16)</c> on identical
/// device tensors. CPU MKL/OpenBLAS are intentionally ineligible. In-process
/// repetitions are diagnostic only; production promotion consumes three
/// separately launched, clean captures.
/// </summary>
internal static class DirectPtxCastFp16Experiment
{
    private static readonly int[] Sizes = [65_536, 262_144, 1_048_576, 4_194_304];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Size, string Method, Distribution Time, double GigabytesPerSecond,
        long Allocation, long TemporaryBytes, float MaxError, int Registers, int LocalBytes);

    internal static void Run(int repetitions = 1)
    {
        if (repetitions <= 0) throw new ArgumentOutOfRangeException(nameof(repetitions));
        for (int repetition = 1; repetition <= repetitions; repetition++)
            RunRepetition(repetition, repetitions);
    }

    private static void RunRepetition(int repetition, int repetitions)
    {
        GpuBenchmarkEnvironment.PrintSnapshot($"start-{repetition}");
        var results = new List<Result>();
        RunDirect(results);
        RunAiDotNet(results);
        RunPyTorch(results);

        Console.WriteLine($"NVIDIA GPU-only FP32->FP16 cast (resident tensors), diagnostic repetition {repetition}/{repetitions}");
        Console.WriteLine($"{"Size",9} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max err",10} {"regs",6} {"local B",8}");
        Console.WriteLine(new string('-', 122));
        foreach (Result r in results.OrderBy(r => r.Size).ThenBy(r => r.Time.Median))
        {
            string registers = r.Registers < 0 ? "n/a" : r.Registers.ToString();
            string local = r.LocalBytes < 0 ? "n/a" : r.LocalBytes.ToString();
            Console.WriteLine(
                $"{r.Size,9} {r.Method,-27} {r.Time.Median * 1000,10:F2} {r.Time.P95 * 1000,10:F2} " +
                $"{r.Time.Mean * 1000,10:F2} {r.GigabytesPerSecond,9:F2} {r.Allocation,9} {r.MaxError,10:G4} {registers,6} {local,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Diagnostic gate: each in-process repetition counts as one diagnostic only; production still requires three clean, separately launched captures.");
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault with { RequiredIndependentRuns = 1 };
        foreach (int size in Sizes)
        {
            var direct = results.SingleOrDefault(r => r.Size == size && r.Method == "Direct PTX cast");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.Size == size && r.Method != "Direct PTX cast").ToList();
            if (competitors.Count == 0) continue;
            Result best = competitors.OrderBy(r => r.Time.Median).First();
            var directEvidence = new DirectPtxPerformanceEvidence(
                direct.Time.Median * 1000, direct.Time.P95 * 1000,
                direct.Allocation, direct.TemporaryBytes, direct.MaxError, direct.LocalBytes, IndependentRuns: 1);
            var competitorEvidence = new DirectPtxPerformanceEvidence(
                best.Time.Median * 1000, best.Time.P95 * 1000,
                best.Allocation, best.TemporaryBytes, best.MaxError, best.LocalBytes, IndependentRuns: 1);
            DirectPtxReleaseDecision decision = policy.Evaluate(directEvidence, competitorEvidence);
            Console.WriteLine(
                $"[{size}]: {(decision.Passed ? "PASS" : "HOLD"),-4} {decision.MedianSpeedup:F2}x vs {best.Method}; " +
                (decision.Passed ? "all gates passed" : string.Join("; ", decision.Failures)));
        }
        GpuBenchmarkEnvironment.PrintSnapshot($"end-{repetition}");
    }

    private static void RunDirect(List<Result> results)
    {
        using var runtime = new DirectPtxRuntime();
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere) return;
        using (runtime.Enter())
        foreach (int size in Sizes)
        {
            using var kernel = new PtxFusedCastF32ToF16Kernel(runtime, size);
            float[] input = Values(size, 100 + size);
            using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            inputBuffer.Upload<float>(input);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            launch(); runtime.Synchronize();
            var actual = new ushort[size];
            outputBuffer.Download<ushort>(actual);
            float error = Validate(actual, input);
            results.Add(new Result(size, "Direct PTX cast", distribution,
                Bandwidth(size, distribution.Median), allocation, 0, error,
                kernel.Audit.Function.RegistersPerThread, kernel.Audit.Function.LocalBytesPerThread));
        }
    }

    private static void RunAiDotNet(List<Result> results)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable) return;
        foreach (int size in Sizes)
        {
            float[] input = Values(size, 100 + size);
            IGpuBuffer? inputBuffer = null, outputBuffer = null;
            try
            {
                inputBuffer = backend.AllocateBuffer(input);
                outputBuffer = backend.AllocateBuffer(size / 2); // size fp16 = size*2 bytes = size/2 float slots
                IGpuBuffer inBuf = inputBuffer, outBuf = outputBuffer;
                Action launch = () => backend.ConvertToFp16(inBuf, outBuf, size);
                launch(); backend.Synchronize();
                Distribution distribution = Measure(backend.Synchronize, launch);
                long allocation = Allocation(backend.Synchronize, launch);
                results.Add(new Result(size, "AiDotNet convert_fp32_to_fp16", distribution,
                    Bandwidth(size, distribution.Median), allocation, 0, 0f, -1, -1));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"SKIP AiDotNet cast size={size}: {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                inputBuffer?.Dispose();
                outputBuffer?.Dispose();
            }
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach (int size in Sizes)
        {
            float[] input = Values(size, 100 + size);
            using TorchTensor source = torch.tensor(input, [size], device: torch.CUDA);
            void Launch()
            {
                using TorchTensor output = source.to(torch.float16);
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(size, "PyTorch to(float16)", distribution,
                Bandwidth(size, distribution.Median), allocation, -1, 0f, -1, -1));
        }
    }

    private static Distribution Measure(Action synchronize, Action launch)
    {
        for (int i = 0; i < 30; i++) launch();
        synchronize();
        var samples = new double[101];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        Array.Sort(samples);
        return new Distribution(samples.Average(), Percentile(samples, .5),
            Percentile(samples, .95), Percentile(samples, .99));
    }

    private static long Allocation(Action synchronize, Action launch)
    {
        launch(); synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) launch();
        long allocation = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        synchronize();
        return allocation;
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static double Bandwidth(int size, double milliseconds)
    {
        long bytes = checked((long)size * sizeof(float) + (long)size * 2);
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (random.NextSingle() * 2f - 1f) * 64f).ToArray();
    }

    private static float Validate(ushort[] actual, float[] input)
    {
        float maximum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            ushort expected = BitConverter.HalfToUInt16Bits((Half)input[i]);
            if (expected != actual[i])
            {
                float a = (float)BitConverter.UInt16BitsToHalf(actual[i]);
                maximum = MathF.Max(maximum, MathF.Abs(a - input[i]) + 1f);
            }
        }
        return maximum;
    }
}
