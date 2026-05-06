#if NET8_0_OR_GREATER
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

public static class Issue305GpuInitBenchmark
{
    private const float Min = -0.05f;
    private const float Max = 0.05f;
    private const float StdDev = 0.02f;
    private static readonly int[] ElementCounts = [1_000_000, 16_777_216];

    public static void Run()
    {
        using var engine = new DirectGpuEngine();
        if (!engine.IsAvailable || engine.Backend is null)
        {
            Console.WriteLine("DirectGPU is not available; cannot run Issue305 GPU initialization benchmark.");
            Console.WriteLine($"Backend: {engine.BackendName}");
            Console.WriteLine($"Device:  {engine.DeviceName}");
            return;
        }

        var backend = engine.Backend;
        Console.WriteLine("Issue #305 GPU initialization benchmark");
        Console.WriteLine($"Backend: {engine.BackendName}");
        Console.WriteLine($"Device:  {engine.DeviceName}");
        Console.WriteLine($"Vendor:  {engine.DeviceVendor}");
        Console.WriteLine($"Compute: {engine.ComputeUnits} CUs, {engine.GlobalMemoryGB:F2} GB global memory");
        Console.WriteLine("Timing includes kernel dispatch plus backend.Synchronize(); it excludes allocation and host download.");
        Console.WriteLine();
        Console.WriteLine($"{"Method",-32} {"Elements",12} {"Mean ms",10} {"Median ms",10} {"Min ms",10} {"Max ms",10} {"GB/s",10}");
        Console.WriteLine(new string('-', 98));

        foreach (int elements in ElementCounts)
        {
            using var output = backend.AllocateBuffer(elements);

            Warmup(backend, output, elements);

            var uniform = Measure(
                runs: 50,
                action: run => backend.GenerateRandomUniform(output, elements, Min, Max, (ulong)(305_000 + run)),
                synchronize: backend.Synchronize);
            Print("AiDotNet GPU uniform", elements, uniform);

            ValidateUniform(backend, output, elements);

            var normal = Measure(
                runs: 50,
                action: run => backend.GenerateRandomNormal(output, elements, 0f, StdDev, (ulong)(405_000 + run)),
                synchronize: backend.Synchronize);
            Print("AiDotNet GPU normal", elements, normal);

            ValidateNormal(backend, output, elements);
            Console.WriteLine();
        }
    }

    private static void Warmup(IDirectGpuBackend backend, IGpuBuffer output, int elements)
    {
        for (int i = 0; i < 10; i++)
        {
            backend.GenerateRandomUniform(output, elements, Min, Max, (ulong)(10_000 + i));
            backend.GenerateRandomNormal(output, elements, 0f, StdDev, (ulong)(20_000 + i));
        }

        backend.Synchronize();
    }

    private static Stats Measure(int runs, Action<int> action, Action synchronize)
    {
        var timings = new double[runs];
        for (int i = 0; i < runs; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action(i);
            synchronize();
            long stop = Stopwatch.GetTimestamp();
            timings[i] = (stop - start) * 1000.0 / Stopwatch.Frequency;
        }

        Array.Sort(timings);
        double sum = 0;
        foreach (double timing in timings)
            sum += timing;

        return new Stats(
            MeanMs: sum / timings.Length,
            MedianMs: timings[timings.Length / 2],
            MinMs: timings[0],
            MaxMs: timings[^1]);
    }

    private static void Print(string method, int elements, Stats stats)
    {
        double bytes = elements * sizeof(float);
        double gbPerSecond = bytes / (stats.MeanMs / 1000.0) / 1_000_000_000.0;
        Console.WriteLine($"{method,-32} {elements,12:N0} {stats.MeanMs,10:F4} {stats.MedianMs,10:F4} {stats.MinMs,10:F4} {stats.MaxMs,10:F4} {gbPerSecond,10:F2}");
    }

    private static void ValidateUniform(IDirectGpuBackend backend, IGpuBuffer output, int elements)
    {
        var sample = backend.DownloadBuffer(output);
        int checkedCount = Math.Min(elements, 4096);
        for (int i = 0; i < checkedCount; i++)
        {
            float value = sample[i];
            if (float.IsNaN(value) || value < Min || value > Max)
                throw new InvalidOperationException($"Uniform GPU initialization produced invalid value {value} at index {i}.");
        }
    }

    private static void ValidateNormal(IDirectGpuBackend backend, IGpuBuffer output, int elements)
    {
        var sample = backend.DownloadBuffer(output);
        int checkedCount = Math.Min(elements, 4096);
        bool sawNonZero = false;
        for (int i = 0; i < checkedCount; i++)
        {
            float value = sample[i];
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw new InvalidOperationException($"Normal GPU initialization produced invalid value {value} at index {i}.");
            sawNonZero |= value != 0f;
        }

        if (!sawNonZero)
            throw new InvalidOperationException("Normal GPU initialization produced only zeros in the validation sample.");
    }

    private readonly record struct Stats(double MeanMs, double MedianMs, double MinMs, double MaxMs);
}
#endif
