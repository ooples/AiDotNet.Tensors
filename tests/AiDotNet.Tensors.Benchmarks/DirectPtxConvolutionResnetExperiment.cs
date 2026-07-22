using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Resident NVIDIA-only issue-#841 ResNet-class convolution evidence harness. For
/// each workhorse shape it measures the established AiDotNet CUDA/cuDNN path (and,
/// as they are verified, the direct-PTX specializations) and compares against the
/// strongest compiled PyTorch/cuDNN lane emitted by
/// <c>run_direct_ptx_convolution_resnet_competitors.py</c>, applying the #841
/// production gate (>=1.10x median speedup, candidate p95 &lt;= competitor p95 +10%).
/// Screening only: promotion still requires three clean processes plus Nsight
/// spill evidence.
/// </summary>
internal static class DirectPtxConvolutionResnetExperiment
{
    private readonly record struct Shape(
        string Scope, int N, int C, int H, int W, int K, int Kernel, int Stride, int Pad)
    {
        public int OutH => (H + 2 * Pad - Kernel) / Stride + 1;
        public int OutW => (W + 2 * Pad - Kernel) / Stride + 1;
        public long Flops => 2L * N * K * OutH * OutW * C * Kernel * Kernel;
    }

    // Matches SHAPES in run_direct_ptx_convolution_resnet_competitors.py exactly.
    private static readonly Shape[] Shapes =
    {
        new("resnet_3x3_c64_56", 32, 64, 56, 56, 64, 3, 1, 1),
        new("resnet_3x3_c128_28", 16, 128, 28, 28, 128, 3, 1, 1),
        new("resnet_3x3_c256_14", 8, 256, 14, 14, 256, 3, 1, 1),
        new("resnet_1x1_c64_56", 32, 64, 56, 56, 64, 1, 1, 0),
        new("resnet_1x1_c128_28", 16, 128, 28, 28, 128, 1, 1, 0),
        new("resnet_1x1_c256_14", 8, 256, 14, 14, 256, 1, 1, 0),
    };

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        string Method, Distribution TimeUs, double Gflops, long ManagedBytes, float MaximumAbsoluteError);

    internal static void Run(bool includeExternal)
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-convolution-resnet-start");
        GpuBenchmarkEnvironment.PrintSnapshot("direct-ptx-convolution-resnet-start");

        Dictionary<string, List<Dictionary<string, JsonElement>>> competitors =
            includeExternal ? RunAndParsePyTorchCompetitors() : new();

        Console.WriteLine("Issue #841 resident FP32 NCHW ResNet-class convolution evidence (screening).");
        Console.WriteLine("Gate: candidate median <= competitor median / 1.10 AND candidate p95 <= competitor p95 * 1.10.");
        Console.WriteLine("Promotion still requires 3 clean processes + Nsight zero-spill evidence.\n");

        foreach (Shape shape in Shapes)
        {
            Console.WriteLine($"== {shape.Scope}: N{shape.N} C{shape.C} H{shape.H} W{shape.W} K{shape.K} " +
                $"{shape.Kernel}x{shape.Kernel} s{shape.Stride} p{shape.Pad} ==");
            var rows = new List<Result> { RunEstablishedAiDotNet(shape) };

            Console.WriteLine($"{"Method",-40} {"median us",11} {"p95 us",11} {"p99 us",11} {"mean us",11} {"GFLOPS",11} {"managed B",11} {"max abs",10}");
            Console.WriteLine(new string('-', 120));
            foreach (Result r in rows.OrderBy(r => r.TimeUs.Median))
                Console.WriteLine(
                    $"{r.Method,-40} {r.TimeUs.Median,11:F2} {r.TimeUs.P95,11:F2} {r.TimeUs.P99,11:F2} " +
                    $"{r.TimeUs.Mean,11:F2} {r.Gflops,11:F2} {r.ManagedBytes,11} {r.MaximumAbsoluteError,10:G4}");

            if (competitors.TryGetValue(shape.Scope, out var lanes))
            {
                (string method, double median, double p95) best = ("(none)", double.MaxValue, double.MaxValue);
                foreach (var lane in lanes)
                {
                    if (!lane.TryGetValue("median_us", out JsonElement m) || m.ValueKind != JsonValueKind.Number)
                        continue;
                    double median = m.GetDouble();
                    double p95 = lane.TryGetValue("p95_us", out JsonElement p) && p.ValueKind == JsonValueKind.Number
                        ? p.GetDouble() : double.MaxValue;
                    string name = lane.TryGetValue("method", out JsonElement nm) ? nm.GetString() ?? "?" : "?";
                    Console.WriteLine($"{"[PyTorch] " + name,-40} {median,11:F2} {p95,11:F2}");
                    if (median < best.median) best = (name, median, p95);
                }
                Result candidate = rows.OrderBy(r => r.TimeUs.Median).First();
                bool medianPass = candidate.TimeUs.Median <= best.median / 1.10;
                bool tailPass = candidate.TimeUs.P95 <= best.p95 * 1.10;
                string verdict = medianPass && tailPass ? "PASS" : "HOLD";
                Console.WriteLine(
                    $"  strongest competitor: {best.method} @ {best.median:F2} us median / {best.p95:F2} us p95; " +
                    $"best candidate: {candidate.Method} @ {candidate.TimeUs.Median:F2}/{candidate.TimeUs.P95:F2}; " +
                    $"gate: {verdict} (median {(medianPass ? "ok" : "miss")}, p95 {(tailPass ? "ok" : "miss")})");
            }
            Console.WriteLine();
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-convolution-resnet-end");
        GpuBenchmarkEnvironment.PrintSnapshot("direct-ptx-convolution-resnet-end");
    }

    private static Result RunEstablishedAiDotNet(Shape shape)
    {
        int outElements = shape.N * shape.K * shape.OutH * shape.OutW;
        float[] inputHost = Values((long)shape.N * shape.C * shape.H * shape.W, 841);
        float[] weightHost = Values((long)shape.K * shape.C * shape.Kernel * shape.Kernel, 842);
        float[] biasHost = Values(shape.K, 843);
        float[] expected = Oracle(shape, inputHost, weightHost, biasHost);

        using var backend = new CudaBackend();
        using var input = backend.AllocateBuffer(inputHost);
        using var weights = backend.AllocateBuffer(weightHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var output = backend.AllocateBuffer(outElements);
        void Launch()
        {
            backend.Conv2D(input, weights, output, shape.N, shape.C, shape.H, shape.W,
                shape.K, shape.OutH, shape.OutW, shape.Kernel, shape.Kernel,
                shape.Stride, shape.Stride, shape.Pad, shape.Pad, 1, 1);
            backend.Conv2DBiasAdd(output, bias, shape.N, shape.K, shape.OutH * shape.OutW);
            backend.Relu(output, output, outElements);
        }
        Distribution distribution = Measure(backend.Synchronize, Launch);
        long allocation = Allocation(backend.Synchronize, Launch);
        Launch();
        backend.Synchronize();
        return new Result("AiDotNet established CUDA/cuDNN", distribution,
            shape.Flops / distribution.Median / 1e3, allocation,
            MaximumError(backend.DownloadBuffer(output), expected));
    }

    private static float[] Oracle(Shape s, float[] input, float[] weights, float[] bias)
    {
        var output = new float[s.N * s.K * s.OutH * s.OutW];
        for (int n = 0; n < s.N; n++)
        for (int k = 0; k < s.K; k++)
        for (int oy = 0; oy < s.OutH; oy++)
        for (int ox = 0; ox < s.OutW; ox++)
        {
            double sum = bias[k];
            for (int c = 0; c < s.C; c++)
            for (int ky = 0; ky < s.Kernel; ky++)
            for (int kx = 0; kx < s.Kernel; kx++)
            {
                int iy = oy * s.Stride - s.Pad + ky;
                int ix = ox * s.Stride - s.Pad + kx;
                if (iy < 0 || iy >= s.H || ix < 0 || ix >= s.W) continue;
                sum += (double)input[((n * s.C + c) * s.H + iy) * s.W + ix] *
                    weights[((k * s.C + c) * s.Kernel + ky) * s.Kernel + kx];
            }
            output[((n * s.K + k) * s.OutH + oy) * s.OutW + ox] = MathF.Max(0f, (float)sum);
        }
        return output;
    }

    private static Distribution Measure(Action synchronize, Action launch)
    {
        for (int warmup = 0; warmup < 30; warmup++) launch();
        synchronize();
        var samples = new double[101];
        for (int sample = 0; sample < samples.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            synchronize();
            samples[sample] = Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000.0; // us
        }
        Array.Sort(samples);
        return new Distribution(samples.Average(), Percentile(samples, 0.5),
            Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static long Allocation(Action synchronize, Action launch)
    {
        launch();
        synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int iteration = 0; iteration < 50; iteration++)
        {
            launch();
            synchronize();
        }
        return (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static float[] Values(long length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, checked((int)length))
            .Select(_ => (random.NextSingle() - 0.5f) * 0.25f).ToArray();
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        if (actual.Length != expected.Length)
            throw new InvalidOperationException("Convolution output extent mismatch.");
        float maximum = 0;
        for (int index = 0; index < actual.Length; index++)
            maximum = MathF.Max(maximum, MathF.Abs(actual[index] - expected[index]));
        return maximum;
    }

    private static Dictionary<string, List<Dictionary<string, JsonElement>>> RunAndParsePyTorchCompetitors()
    {
        string script = Path.Combine(AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_convolution_resnet_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("ResNet PyTorch convolution competitor runner is missing.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            Arguments = "\"" + script.Replace("\"", "\\\"") + "\"",
            UseShellExecute = false,
            RedirectStandardOutput = true
        };
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start ResNet PyTorch competitors.");
        string stdout = process.StandardOutput.ReadToEnd();
        if (!process.WaitForExit((int)TimeSpan.FromMinutes(30).TotalMilliseconds))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("ResNet PyTorch competitors exceeded 30 minutes.");
        }
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"ResNet PyTorch competitors exited with code {process.ExitCode}. Output:\n{stdout}");

        var byScope = new Dictionary<string, List<Dictionary<string, JsonElement>>>(StringComparer.Ordinal);
        using JsonDocument doc = JsonDocument.Parse(stdout);
        foreach (JsonElement entry in doc.RootElement.EnumerateArray())
        {
            string scope = entry.GetProperty("shape").GetProperty("scope").GetString() ?? "?";
            var lanes = new List<Dictionary<string, JsonElement>>();
            foreach (JsonElement lane in entry.GetProperty("results").EnumerateArray())
            {
                var map = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
                foreach (JsonProperty prop in lane.EnumerateObject())
                    map[prop.Name] = prop.Value.Clone();
                lanes.Add(map);
            }
            byScope[scope] = lanes;
        }
        return byScope;
    }
}
