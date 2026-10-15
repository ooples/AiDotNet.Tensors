using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Issue #436: the AiDotNet side of the paired, interleaved head-to-head against real PyTorch
/// (<c>tools/AisEvalH2H/driver.py</c>). The driver owns the data and the schedule. It writes the inputs and weights as
/// raw float32 files and alternates timing rounds between this process and a PyTorch process, so drift on the shared
/// machine hits both sides alike and each round yields its own ratio.
/// </summary>
/// <remarks>
/// One command per stdin line; one JSON line per command on stdout:
/// <list type="bullet">
/// <item><c>setup &lt;mlp|mha|lstm&gt; &lt;dir&gt;</c>: loads the model's tensors from <c>dir</c>.</item>
/// <item><c>dump &lt;path&gt;</c>: runs once and writes the output as raw float32, for the driver's correctness check.</item>
/// <item><c>round &lt;iters&gt;</c>: three untimed calls, then <c>iters</c> timed calls; reports median, p95 and min in ms.</item>
/// <item><c>quit</c>.</item>
/// </list>
/// Shapes (AIsEval CPU scaffold): MLP 784-512-128-10 at batch 128 with ReLU; MHA [128,32,64] with 4 heads and no
/// biases; LSTM [128,32,32] to hidden 64 with no biases, last-step output.
/// </remarks>
internal static class AisEvalH2HServer
{
    public static void Run()
    {
        var engine = new CpuEngine();
        Func<Tensor<float>>? forward = null;
        string? line;
        while ((line = Console.ReadLine()) is not null)
        {
            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;
            try
            {
                switch (parts[0])
                {
                    case "setup":
                        forward = Setup(engine, parts[1], parts[2]);
                        Reply("{\"ok\":true}");
                        break;
                    case "dump":
                        float[] output = Require(forward)().ToArray();
                        File.WriteAllBytes(parts[1], MemoryMarshal.AsBytes(output.AsSpan()).ToArray());
                        Reply($"{{\"ok\":true,\"count\":{output.Length}}}");
                        break;
                    case "round":
                        Reply(TimeRound(Require(forward), int.Parse(parts[1], CultureInfo.InvariantCulture)));
                        break;
                    case "quit":
                        return;
                    default:
                        Reply($"{{\"error\":\"unknown command {parts[0]}\"}}");
                        break;
                }
            }
            catch (Exception exception)
            {
                Reply("{\"error\":" + System.Text.Json.JsonSerializer.Serialize(exception.GetType().Name + ": " + exception.Message) + "}");
            }
        }
    }

    private static void Reply(string json)
    {
        Console.Out.WriteLine(json);
        Console.Out.Flush();
    }

    private static Func<Tensor<float>> Require(Func<Tensor<float>>? forward) =>
        forward ?? throw new InvalidOperationException("setup first");

    private static Tensor<float> Load(string dir, string name, params int[] shape)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] data = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        int expected = 1;
        foreach (int d in shape) expected *= d;
        if (data.Length != expected)
            throw new InvalidDataException($"{name}.bin holds {data.Length} floats, expected {expected}.");
        return new Tensor<float>(data, shape);
    }

    private static Func<Tensor<float>> Setup(CpuEngine engine, string model, string dir)
    {
        switch (model)
        {
            case "mlp":
            {
                var x = Load(dir, "x", 128, 784);
                var weights = new List<Tensor<float>>
                {
                    Load(dir, "w1", 784, 512), Load(dir, "w2", 512, 128), Load(dir, "w3", 128, 10),
                };
                var biases = new List<Tensor<float>?> { Load(dir, "b1", 512), Load(dir, "b2", 128), Load(dir, "b3", 10) };
                return () => engine.MlpForward(x, weights, biases, FusedActivationType.ReLU, FusedActivationType.None);
            }
            case "mha":
            {
                var x = Load(dir, "x", 128, 32, 64);
                var q = Load(dir, "wq", 64, 64);
                var k = Load(dir, "wk", 64, 64);
                var v = Load(dir, "wv", 64, 64);
                var o = Load(dir, "wo", 64, 64);
                return () => engine.MultiHeadAttentionForward(x, q, k, v, o, 4);
            }
            case "lstm":
            {
                var x = Load(dir, "x", 128, 32, 32);
                var wIh = Load(dir, "w_ih", 256, 32);
                var wHh = Load(dir, "w_hh", 256, 64);
                return () => engine.LstmSequenceForward(x, null, null, wIh, wHh, null, null);
            }
            default:
                throw new ArgumentException($"unknown model {model}");
        }
    }

    private static string TimeRound(Func<Tensor<float>> forward, int iters)
    {
        for (int i = 0; i < 3; i++) forward();
        var times = new double[iters];
        var watch = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            watch.Restart();
            forward();
            watch.Stop();
            times[i] = watch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        double median = iters % 2 == 1 ? times[iters / 2] : (times[iters / 2 - 1] + times[iters / 2]) / 2;
        double p95 = times[Math.Min(iters - 1, (int)Math.Ceiling(0.95 * iters) - 1)];
        return string.Format(CultureInfo.InvariantCulture, "{{\"median_ms\":{0:R},\"p95_ms\":{1:R},\"min_ms\":{2:R}}}",
            median, p95, times[0]);
    }
}
