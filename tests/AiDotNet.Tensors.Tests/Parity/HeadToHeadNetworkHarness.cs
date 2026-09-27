#if NET5_0_OR_GREATER
// Net-core only: it drives an external Python process (ProcessStartInfo.ArgumentList) to benchmark
// against PyTorch. The CI parity lanes run net10.0.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// Runs one head-to-head case (issue #1057): the PyTorch side through
/// <c>tools/parity/run_torch_network.py</c>, then the same network on Tensors from the bytes it wrote.
/// </summary>
internal static class HeadToHeadNetworkHarness
{
    internal sealed record PhaseStats(double MedianMs, double IqrMs, double MinMs, int Samples);

    internal sealed record SideResult(
        string Framework,
        string Version,
        IReadOnlyDictionary<string, PhaseStats> Phases,
        IReadOnlyList<double> Losses);

    internal sealed record CaseResult(string Network, string Device, string MachineKey, SideResult Torch, SideResult Tensors)
    {
        /// <summary>
        /// The gated ratio: our fastest step over theirs. The minimum is each side's cost with the least
        /// interference, and it is the stable statistic here. Across five process launches of the MLP case on one
        /// machine it moved 3.21-3.41x, while the ratio of medians moved 2.99-3.65x.
        /// </summary>
        public double StepRatio => Tensors.Phases["step"].MinMs / Torch.Phases["step"].MinMs;

        /// <summary>The ratio of median steps: reported alongside, not gated.</summary>
        public double MedianStepRatio => Tensors.Phases["step"].MedianMs / Torch.Phases["step"].MedianMs;

        /// <summary>Each side's IQR over its median, combined: how noisy this run was.</summary>
        public double NoiseBand
        {
            get
            {
                var a = Tensors.Phases["step"];
                var b = Torch.Phases["step"];
                return Math.Sqrt(Math.Pow(a.IqrMs / a.MedianMs, 2) + Math.Pow(b.IqrMs / b.MedianMs, 2));
            }
        }
    }

    /// <summary>Why the case could not run: no python, no torch, no CUDA. Null when it ran.</summary>
    internal sealed record Deferral(string Reason);

    internal static string PythonExecutable
        => Environment.GetEnvironmentVariable("PARITY_PYTHON") is { Length: > 0 } configured ? configured : "python";

    /// <summary>A coarse identity for comparing ratios: only numbers from the same machine class compare.</summary>
    internal static string MachineKey(string device)
        => $"{OsName()}-{RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()}-{Environment.ProcessorCount}cpu-{device}";

    private static string OsName()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "windows"
         : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux"
         : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "macos"
         : "other";

    internal static (CaseResult? Result, Deferral? Deferred) Run(string root, string network, string device)
    {
        string spec = Path.Combine(root, "parity", "networks", network + ".json");
        string work = Path.Combine(Path.GetTempPath(), "aidotnet-parity", network + "-" + device + "-" + Guid.NewGuid().ToString("N").Substring(0, 8));
        Directory.CreateDirectory(work);

        var psi = new ProcessStartInfo(PythonExecutable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in new[] { Path.Combine(root, "tools", "parity", "run_torch_network.py"), "--spec", spec, "--workdir", work, "--device", device })
            psi.ArgumentList.Add(arg);

        Process process;
        try
        {
            process = Process.Start(psi) ?? throw new InvalidOperationException("Process.Start returned null.");
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return (null, new Deferral($"'{PythonExecutable}' is not on PATH; set PARITY_PYTHON to a Python with torch installed."));
        }

        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode == 3) return (null, new Deferral(stderr.Trim()));
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"PyTorch runner failed ({process.ExitCode}):{Environment.NewLine}{stderr}{stdout}");

        var torch = LoadTorchResult(Path.Combine(work, "torch.json"));
        var tensors = RunTensors(ReadJson(spec), work);
        var result = new CaseResult(network, device, MachineKey(device), torch, tensors);
        WriteArtifact(result);
        return (result, null);
    }

    /// <summary>Latest result per case, for tooling and for the stamped evidence the GPU lane commits.</summary>
    internal static string ArtifactPath(string network, string device)
        => Path.Combine(Path.GetTempPath(), "aidotnet-parity", $"{network}-{device}-latest.json");

    private static void WriteArtifact(CaseResult r)
    {
        var doc = new
        {
            network = r.Network,
            device = r.Device,
            machineKey = r.MachineKey,
            stepRatio = r.StepRatio,
            medianStepRatio = r.MedianStepRatio,
            noiseBand = r.NoiseBand,
            torch = new { version = r.Torch.Version, phases = r.Torch.Phases, losses = r.Torch.Losses },
            tensors = new { version = r.Tensors.Version, phases = r.Tensors.Phases, losses = r.Tensors.Losses },
        };
        File.WriteAllText(ArtifactPath(r.Network, r.Device), JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static JsonElement ReadJson(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    }

    private static SideResult LoadTorchResult(string path)
    {
        var root = ReadJson(path);
        var phases = root.GetProperty("phases").EnumerateObject().ToDictionary(
            p => p.Name,
            p => new PhaseStats(
                p.Value.GetProperty("medianMs").GetDouble(),
                p.Value.GetProperty("iqrMs").GetDouble(),
                p.Value.GetProperty("minMs").GetDouble(),
                p.Value.GetProperty("samples").GetInt32()));
        var losses = root.GetProperty("losses").EnumerateArray().Select(l => l.GetDouble()).ToList();
        return new SideResult("torch", root.GetProperty("frameworkVersion").GetString() ?? "unknown", phases, losses);
    }

    private static float[] ReadFloats(BinaryReader reader, int count)
    {
        var bytes = reader.ReadBytes(count * sizeof(float));
        if (bytes.Length != count * sizeof(float))
            throw new InvalidDataException($"Expected {count} floats, found {bytes.Length / sizeof(float)}.");
        var values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static SideResult RunTensors(JsonElement spec, string work)
    {
        var engine = AiDotNetEngine.Current;
        int batch = spec.GetProperty("batch").GetInt32();
        int inputDim = spec.GetProperty("inputDim").GetInt32();
        var layerSpecs = spec.GetProperty("layers").EnumerateArray().ToList();
        var dims = new List<int> { inputDim };
        dims.AddRange(layerSpecs.Select(l => l.GetProperty("out").GetInt32()));

        var parameters = new List<(float[] Data, float[] Grad, Tensor<float> Tensor)>();
        var layers = new List<(Tensor<float> W, Tensor<float> B, FusedActivationType Act)>();
        using (var reader = new BinaryReader(File.OpenRead(Path.Combine(work, "weights.bin"))))
        {
            for (int i = 0; i < layerSpecs.Count; i++)
            {
                var w = ReadFloats(reader, dims[i] * dims[i + 1]);
                var b = ReadFloats(reader, dims[i + 1]);
                var wt = Tensor<float>.FromMemory(w, new[] { dims[i], dims[i + 1] });
                var bt = Tensor<float>.FromMemory(b, new[] { dims[i + 1] });
                parameters.Add((w, new float[w.Length], wt));
                parameters.Add((b, new float[b.Length], bt));
                var act = layerSpecs[i].GetProperty("activation").GetString() == "relu" ? FusedActivationType.ReLU : FusedActivationType.None;
                layers.Add((wt, bt, act));
            }
        }

        Tensor<float> x, y;
        using (var reader = new BinaryReader(File.OpenRead(Path.Combine(work, "data.bin"))))
        {
            x = Tensor<float>.FromMemory(ReadFloats(reader, batch * inputDim), new[] { batch, inputDim });
            y = Tensor<float>.FromMemory(ReadFloats(reader, batch * dims[dims.Count - 1]), new[] { batch, dims[dims.Count - 1] });
        }

        var optimizer = new SgdOptimizer();
        var group = optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = spec.GetProperty("optimizer").GetProperty("lr").GetDouble() });
        foreach (var p in parameters) group.AddParameter(p.Data, p.Grad);
        var sources = parameters.Select(p => p.Tensor).ToArray();
        var allAxes = new[] { 0, 1 };

        int warmup = spec.GetProperty("warmupSteps").GetInt32();
        int measured = spec.GetProperty("measuredSteps").GetInt32();
        int lossSteps = spec.GetProperty("lossAgreementSteps").GetInt32();
        var phases = new Dictionary<string, List<double>>
        {
            ["forward"] = new(), ["backward"] = new(), ["optimizer"] = new(), ["step"] = new(),
        };
        var losses = new List<double>();
        var sw = new Stopwatch();

        for (int step = 0; step < warmup + measured; step++)
        {
            sw.Restart();
            using var tape = new GradientTape<float>();
            var h = x;
            foreach (var (w, b, act) in layers) h = engine.FusedLinear(h, w, b, act);
            var loss = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(h, y)), allAxes, keepDims: false);
            double t1 = sw.Elapsed.TotalMilliseconds;

            var grads = tape.ComputeGradients(loss, sources);
            double t2 = sw.Elapsed.TotalMilliseconds;

            for (int i = 0; i < parameters.Count; i++)
            {
                if (!grads.TryGetValue(parameters[i].Tensor, out var g))
                    throw new InvalidOperationException($"Tape produced no gradient for parameter {i}.");
                g.AsSpan().CopyTo(parameters[i].Grad);
            }

            optimizer.Step();
            double t3 = sw.Elapsed.TotalMilliseconds;

            if (step < lossSteps) losses.Add(loss.GetFlat(0));
            if (step >= warmup)
            {
                phases["forward"].Add(t1);
                phases["backward"].Add(t2 - t1);
                phases["optimizer"].Add(t3 - t2);
                phases["step"].Add(t3);
            }
        }

        return new SideResult(
            "AiDotNet.Tensors",
            typeof(Tensor<>).Assembly.GetName().Version?.ToString() ?? "unknown",
            phases.ToDictionary(kv => kv.Key, kv => Stats(kv.Value)),
            losses);
    }

    private static PhaseStats Stats(List<double> samples)
    {
        var ordered = samples.OrderBy(s => s).ToList();
        double median = ordered.Count % 2 == 1
            ? ordered[ordered.Count / 2]
            : (ordered[ordered.Count / 2 - 1] + ordered[ordered.Count / 2]) / 2.0;
        return new PhaseStats(median, ordered[3 * ordered.Count / 4] - ordered[ordered.Count / 4], ordered[0], ordered.Count);
    }
}
#endif
