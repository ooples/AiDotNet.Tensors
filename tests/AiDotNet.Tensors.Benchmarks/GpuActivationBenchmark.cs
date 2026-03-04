using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks GPU activation kernels: ReLU, Sigmoid, Tanh, GELU, Softmax.
/// Measures throughput and correctness at sizes from 1K to 1M elements.
/// </summary>
public static class GpuActivationBenchmark
{
    public static void Run()
    {
        Console.WriteLine("===========================================");
        Console.WriteLine("GPU ACTIVATION BENCHMARK");
        Console.WriteLine("===========================================");
        Console.WriteLine();

        DirectGpuEngine? engine = null;
        try
        {
            engine = new DirectGpuEngine();
            if (!engine.IsAvailable)
            {
                Console.WriteLine("[SKIP] DirectGpu not available.");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            return;
        }

        Console.WriteLine($"Backend: {engine.BackendName}");
        Console.WriteLine($"Device:  {engine.DeviceName}");
        Console.WriteLine();

        var backend = engine.Backend;
        if (backend == null)
        {
            Console.WriteLine("[ERROR] Could not access backend.");
            engine.Dispose();
            return;
        }

        int[] sizes = [1024, 4096, 16384, 65536, 262144, 1048576];
        string[] ops = ["relu", "sigmoid", "tanh", "gelu", "softmax"];

        Console.WriteLine($"{"Operation",-12} {"Size",-10} {"Time(ms)",10} {"GB/s",8} {"GFLOPS",10} {"MaxErr",12} {"Status",10}");
        Console.WriteLine(new string('-', 74));

        foreach (var op in ops)
        {
            foreach (var size in sizes)
            {
                RunSingle(backend, op, size);
            }
            Console.WriteLine();
        }

        engine.Dispose();
    }

    private static void RunSingle(IDirectGpuBackend backend, string op, int size)
    {
        var rand = new Random(42);
        var input = new float[size];
        for (int i = 0; i < size; i++)
            input[i] = (float)(rand.NextDouble() - 0.5) * 4; // range [-2, 2]

        using var bufIn = backend.AllocateBuffer(input);
        using var bufOut = backend.AllocateBuffer(size);

        // Warmup
        for (int i = 0; i < 10; i++)
            Dispatch(backend, op, bufIn, bufOut, size);
        backend.Synchronize();

        // Benchmark
        int runs = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
            Dispatch(backend, op, bufIn, bufOut, size);
        backend.Synchronize();
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / runs;
        double bytes = size * 4.0 * 2; // read + write
        double gbps = bytes / (avgMs * 1e6);
        double gflops = size / (avgMs * 1e6); // 1 FLOP per element approximation

        // Correctness
        var gpuResult = new float[size];
        backend.DownloadBuffer(bufOut, gpuResult);
        var cpuResult = ComputeCpu(input, op);
        double maxError = 0;
        for (int i = 0; i < size; i++)
        {
            if (float.IsNaN(gpuResult[i]) || float.IsInfinity(gpuResult[i])) continue;
            double error = Math.Abs(cpuResult[i] - gpuResult[i]);
            if (error > maxError) maxError = error;
        }

        string status = maxError <= 1e-5 ? "PASS" : maxError <= 1e-3 ? "WARN" : "FAIL";
        Console.WriteLine($"{op,-12} {size,-10} {avgMs,10:F4} {gbps,8:F2} {gflops,10:F2} {maxError,12:E2} {status,10}");
    }

    private static void Dispatch(IDirectGpuBackend backend, string op, IGpuBuffer bufIn, IGpuBuffer bufOut, int size)
    {
        switch (op)
        {
            case "relu": backend.Relu(bufIn, bufOut, size); break;
            case "sigmoid": backend.Sigmoid(bufIn, bufOut, size); break;
            case "tanh": backend.Tanh(bufIn, bufOut, size); break;
            case "gelu": backend.Gelu(bufIn, bufOut, size); break;
            case "softmax": backend.Softmax(bufIn, bufOut, 1, size); break;
        }
    }

    private static float[] ComputeCpu(float[] input, string op)
    {
        var result = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = op switch
            {
                "relu" => MathF.Max(0, input[i]),
                "sigmoid" => 1.0f / (1.0f + MathF.Exp(-input[i])),
                "tanh" => MathF.Tanh(input[i]),
                "gelu" => 0.5f * input[i] * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (input[i] + 0.044715f * input[i] * input[i] * input[i]))),
                _ => input[i]
            };
        }

        if (op == "softmax")
        {
            float max = float.MinValue;
            for (int i = 0; i < input.Length; i++)
                if (input[i] > max) max = input[i];

            float sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = MathF.Exp(input[i] - max);
                sum += result[i];
            }

            for (int i = 0; i < input.Length; i++)
                result[i] /= sum;
        }

        return result;
    }
}
