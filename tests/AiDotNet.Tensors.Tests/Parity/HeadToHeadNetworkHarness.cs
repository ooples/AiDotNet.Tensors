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
using AiDotNet.Tensors.Engines.Gpu;
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
    /// <summary>
    /// Ratios are only comparable on like hardware, so baselines are keyed by machine class, including the CPU
    /// model: the hosted pool behind one runner label mixes models, and on a faster one PyTorch's MLP step fell
    /// from 4.25 ms to 1.39 ms while ours fell from 9.74 to 7.62, moving the ratio from 2.3x to 5.5x with no code
    /// change. A GPU case adds the device model too. An unseen machine class defers; it never compares against
    /// another's baseline.
    /// </summary>
    internal static string MachineKey(string device, string? gpuName = null)
    {
        string key = $"{OsName()}-{RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()}-" +
                     $"{Environment.ProcessorCount}cpu-{Slug(CpuModel())}-{device}";
        return string.IsNullOrWhiteSpace(gpuName) ? key : key + "-" + Slug(gpuName);
    }

    private static string Slug(string text)
    {
        var slug = new string(text.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) ? c : '-').ToArray());
        while (slug.Contains("--")) slug = slug.Replace("--", "-");
        slug = slug.Trim('-');
        return slug.Length == 0 ? "unknown" : slug;
    }

    /// <summary>The CPU model string the OS reports, or "unknown".</summary>
    internal static string CpuModel()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && File.Exists("/proc/cpuinfo"))
            {
                var line = File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal));
                if (line is not null && line.IndexOf(':') is var colon and >= 0) return line.Substring(colon + 1).Trim();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                if (key?.GetValue("ProcessorNameString") is string name) return name.Trim();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                var psi = new ProcessStartInfo("sysctl", "-n machdep.cpu.brand_string") { RedirectStandardOutput = true, UseShellExecute = false };
                using var process = Process.Start(psi);
                if (process is not null)
                {
                    string output = process.StandardOutput.ReadToEnd().Trim();
                    process.WaitForExit();
                    if (output.Length > 0) return output;
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Security.SecurityException or System.ComponentModel.Win32Exception)
        {
        }

        return "unknown";
    }

    private static string OsName()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "windows"
         : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux"
         : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "macos"
         : "other";

    internal static (CaseResult? Result, Deferral? Deferred) Run(string root, string network, string device)
    {
        if (device != "cpu" && device != "cuda") throw new ArgumentException($"Unknown device '{device}'.", nameof(device));
        DirectGpuTensorEngine? gpu = null;
        if (device == "cuda")
        {
            try { gpu = new DirectGpuTensorEngine(); }
            catch (Exception ex) { return (null, new Deferral($"no GPU backend ({ex.GetType().Name}: {ex.Message}).")); }
            if (!gpu.IsGpuAvailable)
            {
                gpu.Dispose();
                return (null, new Deferral("no GPU available to AiDotNet.Tensors."));
            }
        }

        try
        {
            return RunWith(root, network, device, gpu);
        }
        finally
        {
            gpu?.Dispose();
        }
    }

    /// <summary>
    /// Runs the two sides <c>repeats</c> times (from the spec, default 1), alternating, and combines them. On a shared
    /// runner a burst of neighbour load lands on one side of one repetition; alternating spreads it, and the gated
    /// statistic (fastest step per side, over every repetition) is the least sensitive to it. Measured on the hosted
    /// 4-core runner: single-shot min/min ratios of 2.05, 2.12 and 2.82 for the unchanged MLP.
    /// </summary>
    private static (CaseResult? Result, Deferral? Deferred) RunWith(string root, string network, string device, DirectGpuTensorEngine? gpu)
    {
        var specJson = ReadJson(Path.Combine(root, "parity", "networks", network + ".json"));
        int repeats = specJson.TryGetProperty("repeats", out var repeatsElement) ? Math.Max(1, repeatsElement.GetInt32()) : 1;
        var torchRuns = new List<SideResult>();
        var tensorsRuns = new List<SideResult>();
        string machineKey = MachineKey(device);
        for (int r = 0; r < repeats; r++)
        {
            var (once, deferred) = RunOnce(root, network, device, gpu);
            if (deferred is not null) return (null, deferred);
            if (once is null) throw new InvalidOperationException("A repetition returned neither a result nor a deferral.");
            torchRuns.Add(once.Torch);
            tensorsRuns.Add(once.Tensors);
            machineKey = once.MachineKey;
        }

        var result = new CaseResult(network, device, machineKey, Combine(torchRuns), Combine(tensorsRuns));
        WriteArtifact(result);
        return (result, null);
    }

    /// <summary>Fastest step over every repetition (gated); median of medians and of IQRs; losses from the first.</summary>
    private static SideResult Combine(List<SideResult> runs)
    {
        if (runs.Count == 1) return runs[0];
        static double Median(IEnumerable<double> values)
        {
            var o = values.OrderBy(v => v).ToList();
            return o.Count % 2 == 1 ? o[o.Count / 2] : (o[o.Count / 2 - 1] + o[o.Count / 2]) / 2.0;
        }

        var phases = runs[0].Phases.Keys.ToDictionary(
            phase => phase,
            phase => new PhaseStats(
                Median(runs.Select(r => r.Phases[phase].MedianMs)),
                Median(runs.Select(r => r.Phases[phase].IqrMs)),
                runs.Min(r => r.Phases[phase].MinMs),
                runs.Sum(r => r.Phases[phase].Samples)));
        return new SideResult(runs[0].Framework, runs[0].Version, phases, runs[0].Losses);
    }

    private static (CaseResult? Result, Deferral? Deferred) RunOnce(string root, string network, string device, DirectGpuTensorEngine? gpu)
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

        var torchJson = ReadJson(Path.Combine(work, "torch.json"));
        var torch = LoadTorchResult(torchJson);
        string? gpuName = torchJson.GetProperty("machine").TryGetProperty("gpu", out var gpuElement)
                          && gpuElement.ValueKind == JsonValueKind.String
            ? gpuElement.GetString()
            : null;
        var tensors = RunTensors(ReadJson(spec), work, gpu);
        return (new CaseResult(network, device, MachineKey(device, gpuName), torch, tensors), null);
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

    private static SideResult LoadTorchResult(JsonElement root)
    {
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

    /// <summary>
    /// Trains the spec's network with Tensors. With <paramref name="gpu"/> the whole step runs on the device:
    /// data and parameters are uploaded once, the update is the device-side SGD kernel, and every phase
    /// boundary synchronises the stream, as the PyTorch side does, so a phase's time includes its kernels.
    /// </summary>
    private static SideResult RunTensors(JsonElement spec, string work, DirectGpuTensorEngine? gpu)
    {
        var previous = AiDotNetEngine.Current;
        if (gpu is not null) AiDotNetEngine.Current = gpu;
        try
        {
            return TrainTensors(spec, work, gpu);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static SideResult TrainTensors(JsonElement spec, string work, DirectGpuTensorEngine? gpu)
    {
        IEngine engine = gpu ?? AiDotNetEngine.Current;
        Tensor<float> Place(Tensor<float> tensor) => gpu is null ? tensor : gpu.UploadToGpu(tensor, GpuTensorRole.General);
        int batch = spec.GetProperty("batch").GetInt32();
        var inputShape = spec.TryGetProperty("inputShape", out var shapeElement)
            ? shapeElement.EnumerateArray().Select(d => d.GetInt32()).ToArray()
            : new[] { spec.GetProperty("inputDim").GetInt32() };

        // Mirrors tools/parity/run_torch_network.py layer for layer: the same shape arithmetic and the
        // same weights.bin order (each parameterised layer's weight, then its bias).
        var parameters = new List<(float[] Data, float[] Grad, Tensor<float> Tensor)>();
        var layers = new List<Func<Tensor<float>, Tensor<float>>>();
        var shape = inputShape.ToArray();
        using (var reader = new BinaryReader(File.OpenRead(Path.Combine(work, "weights.bin"))))
        {
            Tensor<float> Parameter(int[] parameterShape)
            {
                var data = ReadFloats(reader, parameterShape.Aggregate(1, (a, d) => a * d));
                var tensor = Place(Tensor<float>.FromMemory(data, parameterShape));
                parameters.Add((data, new float[data.Length], tensor));
                return tensor;
            }

            foreach (var layer in spec.GetProperty("layers").EnumerateArray())
            {
                string kind = layer.TryGetProperty("type", out var typeElement) ? typeElement.GetString() ?? "linear" : "linear";
                var act = layer.TryGetProperty("activation", out var actElement) && actElement.GetString() == "relu"
                    ? FusedActivationType.ReLU
                    : FusedActivationType.None;
                switch (kind)
                {
                    case "linear":
                    {
                        if (shape.Length != 1) throw new InvalidDataException($"linear layer needs a flat input, got [{string.Join(", ", shape)}].");
                        int outDim = layer.GetProperty("out").GetInt32();
                        var w = Parameter(new[] { shape[0], outDim });
                        var b = Parameter(new[] { outDim });
                        layers.Add(h => engine.FusedLinear(h, w, b, act));
                        shape = new[] { outDim };
                        break;
                    }
                    case "conv2d":
                    {
                        int outChannels = layer.GetProperty("out").GetInt32();
                        int k = layer.GetProperty("kernel").GetInt32();
                        int s = layer.TryGetProperty("stride", out var sElement) ? sElement.GetInt32() : 1;
                        int pad = layer.TryGetProperty("padding", out var pElement) ? pElement.GetInt32() : 0;
                        var w = Parameter(new[] { outChannels, shape[0], k, k });
                        var b = Parameter(new[] { outChannels });
                        layers.Add(h => engine.FusedConv2D(h, w, b, s, s, pad, pad, 1, 1, act));
                        shape = new[] { outChannels, (shape[1] + 2 * pad - k) / s + 1, (shape[2] + 2 * pad - k) / s + 1 };
                        break;
                    }
                    case "maxpool2d":
                    {
                        int k = layer.GetProperty("size").GetInt32();
                        layers.Add(h => engine.MaxPool2D(h, k));
                        shape = new[] { shape[0], shape[1] / k, shape[2] / k };
                        break;
                    }
                    case "flatten":
                    {
                        int flat = shape.Aggregate(1, (a, d) => a * d);
                        layers.Add(h => engine.Reshape(h, new[] { batch, flat }));
                        shape = new[] { flat };
                        break;
                    }
                    default:
                        throw new InvalidDataException($"unsupported layer type '{kind}'.");
                }
            }
        }

        if (shape.Length != 1) throw new InvalidDataException($"the network must end flat, ends at [{string.Join(", ", shape)}].");
        Tensor<float> x, y;
        using (var reader = new BinaryReader(File.OpenRead(Path.Combine(work, "data.bin"))))
        {
            var xShape = new[] { batch }.Concat(inputShape).ToArray();
            x = Place(Tensor<float>.FromMemory(ReadFloats(reader, xShape.Aggregate(1, (a, d) => a * d)), xShape));
            y = Place(Tensor<float>.FromMemory(ReadFloats(reader, batch * shape[0]), new[] { batch, shape[0] }));
        }

        double lr = spec.GetProperty("optimizer").GetProperty("lr").GetDouble();
        var optimizer = new SgdOptimizer();
        var group = optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = lr });
        foreach (var p in parameters) group.AddParameter(p.Data, p.Grad);
        void Sync() => gpu?.SynchronizeStream();
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
            Sync();
            sw.Restart();
            using var tape = new GradientTape<float>();
            var h = x;
            foreach (var layer in layers) h = layer(h);
            var loss = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(h, y)), allAxes, keepDims: false);
            Sync();
            double t1 = sw.Elapsed.TotalMilliseconds;

            var grads = tape.ComputeGradients(loss, sources);
            Sync();
            double t2 = sw.Elapsed.TotalMilliseconds;

            for (int i = 0; i < parameters.Count; i++)
            {
                if (!grads.TryGetValue(parameters[i].Tensor, out var g))
                    throw new InvalidOperationException($"Tape produced no gradient for parameter {i}.");
                if (gpu is null)
                    g.AsSpan().CopyTo(parameters[i].Grad);
                else if (!GpuOptimizer.TrySgdStep(parameters[i].Tensor, g, (float)lr))
                    throw new InvalidOperationException(
                        $"Parameter {i} [{string.Join(", ", parameters[i].Tensor.Shape.ToArray())}]: the device-side SGD step " +
                        $"was refused (parameter on device: {parameters[i].Tensor.TryGetGpuBuffer() is not null}, gradient on " +
                        $"device: {g.TryGetGpuBuffer() is not null}). That is a residency gap: the step would have to leave the device.");
            }

            if (gpu is null) optimizer.Step();
            Sync();
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
