using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue-#848 NVIDIA-only fused FP32 SGD-with-momentum benchmark. Compares the
/// hand-emitted direct-PTX kernel against the current AiDotNet CUDA
/// sgd_momentum_update kernel and the strongest resident PyTorch update on
/// identical device tensors. Small learning rate keeps the 131-launch timing
/// loop numerically bounded. CPU MKL/OpenBLAS are intentionally ineligible.
/// In-process repetitions are diagnostic only; promotion consumes three
/// separately launched, clean captures.
/// </summary>
internal static class DirectPtxSgdMomentumExperiment
{
    private static readonly int[] Sizes = [65_536, 262_144, 1_048_576, 4_194_304];
    private const float LearningRate = 1e-6f;
    private const float Momentum = 0.9f;
    private const float WeightDecay = 1e-4f;

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

        Console.WriteLine($"NVIDIA GPU-only fused FP32 SGD-momentum step (resident tensors), diagnostic repetition {repetition}/{repetitions}");
        Console.WriteLine($"{"Size",9} {"Method",-27} {"median us",10} {"p95 us",10} {"mean us",10} {"GB/s",9} {"B/call",9} {"max rel",10} {"regs",6} {"local B",8}");
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
            var direct = results.SingleOrDefault(r => r.Size == size && r.Method == "Direct PTX SGD-momentum");
            if (direct.Method is null) continue;
            var competitors = results.Where(r => r.Size == size && r.Method != "Direct PTX SGD-momentum").ToList();
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
            using var kernel = new PtxFusedSgdMomentumF32Kernel(runtime, size, LearningRate, Momentum, WeightDecay);
            float[] param = Values(size, 100 + size);
            float[] grad = Values(size, 300 + size);
            float[] velocity = Values(size, 500 + size);
            using var paramBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var gradBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var velocityBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            gradBuffer.Upload<float>(grad);
            Action launch = () => kernel.Launch(
                DirectPtxTensorView.CreateOwned(paramBuffer, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(gradBuffer, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(velocityBuffer, kernel.Blueprint.Tensors[2]));
            paramBuffer.Upload<float>(param);
            velocityBuffer.Upload<float>(velocity);
            launch(); runtime.Synchronize();
            var actualParam = new float[size];
            var actualVelocity = new float[size];
            paramBuffer.Download<float>(actualParam);
            velocityBuffer.Download<float>(actualVelocity);
            float error = Validate(actualParam, actualVelocity, param, grad, velocity);
            Distribution distribution = Measure(runtime.Synchronize, launch);
            long allocation = Allocation(runtime.Synchronize, launch);
            results.Add(new Result(size, "Direct PTX SGD-momentum", distribution,
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
            float[] param = Values(size, 100 + size);
            float[] grad = Values(size, 300 + size);
            float[] velocity = Values(size, 500 + size);
            using var paramBuffer = backend.AllocateBuffer(param);
            using var gradBuffer = backend.AllocateBuffer(grad);
            using var velocityBuffer = backend.AllocateBuffer(velocity);
            Action launch = () => backend.SgdMomentumUpdate(
                paramBuffer, gradBuffer, velocityBuffer, LearningRate, Momentum, WeightDecay, size);
            Distribution distribution = Measure(backend.Synchronize, launch);
            long allocation = Allocation(backend.Synchronize, launch);
            results.Add(new Result(size, "AiDotNet sgd_momentum_update", distribution,
                Bandwidth(size, distribution.Median), allocation, 0, 0f, -1, -1));
        }
    }

    private static void RunPyTorch(List<Result> results)
    {
        if (!torch.cuda.is_available()) return;
        foreach (int size in Sizes)
        {
            float[] param = Values(size, 100 + size);
            float[] grad = Values(size, 300 + size);
            float[] velocity = Values(size, 500 + size);
            using TorchTensor p = torch.tensor(param, [size], device: torch.CUDA);
            using TorchTensor g = torch.tensor(grad, [size], device: torch.CUDA);
            using TorchTensor v = torch.tensor(velocity, [size], device: torch.CUDA);
            void Launch()
            {
                using (torch.no_grad())
                {
                    using TorchTensor decayed = g.add(p, alpha: WeightDecay); // materialized intermediate
                    v.mul_(Momentum).add_(decayed);
                    p.add_(v, alpha: -LearningRate);
                }
            }
            Distribution distribution = Measure(() => torch.cuda.synchronize(), Launch);
            long allocation = Allocation(() => torch.cuda.synchronize(), Launch);
            results.Add(new Result(size, "PyTorch SGD (3-op)", distribution,
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
        long bytes = checked((long)size * sizeof(float) * 5L); // read p,g,v; write p,v
        return bytes / (milliseconds * 1e-3) / 1e9;
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (random.NextSingle() * 2f - 1f)).ToArray();
    }

    private static float Validate(float[] actualParam, float[] actualVelocity, float[] param, float[] grad, float[] velocity)
    {
        float maximum = 0;
        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i] + WeightDecay * param[i];
            float v = Momentum * velocity[i] + g;
            float p = param[i] - LearningRate * v;
            maximum = MathF.Max(maximum, MathF.Abs(actualVelocity[i] - v) / (MathF.Abs(v) + 1f));
            maximum = MathF.Max(maximum, MathF.Abs(actualParam[i] - p) / (MathF.Abs(p) + 1f));
        }
        return maximum;
    }
}
