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

    /// <summary>One PyTorch run of a spec; the CNN on a 4-core runner takes under two minutes.</summary>
    private static readonly TimeSpan RunnerTimeout = TimeSpan.FromMinutes(15);

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
        string specPath = Path.Combine(root, "parity", "networks", network + ".json");
        var specJson = ReadJson(specPath);
        int repeats = 1;
        if (specJson.TryGetProperty("repeats", out var repeatsElement))
        {
            if (repeatsElement.ValueKind != JsonValueKind.Number || !repeatsElement.TryGetInt32(out repeats) || repeats < 1)
                throw new InvalidDataException($"{specPath}: \"repeats\" must be a positive integer, got {repeatsElement.GetRawText()}.");
        }
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

        // Drain both pipes at once. Reading stdout to the end first deadlocks as soon as the runner fills the stderr
        // pipe (torch writes warnings there): the child blocks on its next write and never exits.
        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit((int)RunnerTimeout.TotalMilliseconds))
        {
            try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            process.WaitForExit();
            throw new TimeoutException($"PyTorch runner for {network} on {device} did not finish within {RunnerTimeout}.");
        }

        string stdout = stdoutTask.GetAwaiter().GetResult();
        string stderr = stderrTask.GetAwaiter().GetResult();
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
        // The CPU case must run on the CPU engine: on a machine with a GPU the default engine is the GPU engine, so the
        // "cpu" case trained and timed the GPU engine over host tensors.
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu is not null ? gpu : new CpuEngine();
        try
        {
            return TrainTensors(spec, work, gpu);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// The host/device crossings of one warmed-up training step of <paramref name="network"/> on the GPU engine,
    /// grouped by engine operation, with random weights and data (no PyTorch run needed). For finding the residency
    /// gaps of each network the head-to-head trains.
    /// </summary>
    internal static Dictionary<string, int> MeasureStepCrossings(string root, string network, DirectGpuTensorEngine gpu, int warmupSteps)
    {
        var spec = ReadJson(Path.Combine(root, "parity", "networks", network + ".json"));
        string work = Path.Combine(Path.GetTempPath(), "aidotnet-residency", network);
        Directory.CreateDirectory(work);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(spec.GetProperty("seed").GetInt32());
        void WriteRandom(string file, int count, float scale)
        {
            using var writer = new BinaryWriter(File.Create(Path.Combine(work, file)));
            for (int i = 0; i < count; i++) writer.Write((float)((rng.NextDouble() * 2 - 1) * scale));
        }
        // More than any spec's parameter count (ResNet-18: 11.2M) and batch; readers consume what they need.
        const int RandomParameterFloats = 12_000_000, RandomDataFloats = 2_000_000;
        const float ParameterScale = 0.05f;
        WriteRandom("weights.bin", RandomParameterFloats, ParameterScale);
        WriteRandom("data.bin", RandomDataFloats, 1f);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var capture = new StepCrossingCapture(warmupSteps);
            TrainTensors(spec, work, gpu, capture);
            return capture.Crossings ?? throw new InvalidOperationException("the measured step never ran.");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private sealed class StepCrossingCapture
    {
        internal StepCrossingCapture(int measuredStep) => MeasuredStep = measuredStep;
        internal int MeasuredStep { get; }
        internal Dictionary<string, int>? Crossings { get; set; }
    }

    private static SideResult TrainTensors(JsonElement spec, string work, DirectGpuTensorEngine? gpu, StepCrossingCapture? capture = null)
    {
        IEngine engine = AiDotNetEngine.Current;
        Tensor<float> Place(Tensor<float> tensor) => gpu is null ? tensor : gpu.UploadToGpu(tensor, GpuTensorRole.General);
        int batch = spec.GetProperty("batch").GetInt32();
        var inputShape = spec.TryGetProperty("inputShape", out var shapeElement)
            ? shapeElement.EnumerateArray().Select(d => d.GetInt32()).ToArray()
            : new[] { spec.GetProperty("inputDim").GetInt32() };

        // Mirrors tools/parity/run_torch_network.py layer for layer: the same shape arithmetic and the
        // same weights.bin order (each parameterised layer's weight, then its bias).
        var parameters = new List<(float[] Data, float[] Grad, Tensor<float> Tensor)>();
        // PyTorch's BatchNorm2d and LayerNorm default; the runner builds both with it.
        const double NormEpsilon = 1e-5;
        Tensor<float> BatchNorm(Tensor<float> h, Tensor<float> gamma, Tensor<float> beta)
            => engine.BatchNorm(h, gamma, beta, NormEpsilon, out _, out _);
        // A [rows, n] + [n] broadcast add. It is a CpuEngine operation (both engines derive from it), not an IEngine one.
        var broadcastEngine = (CpuEngine)engine;
        Tensor<float> AddBias(Tensor<float> a, Tensor<float> bias) => broadcastEngine.TensorBroadcastAdd(a, bias);
        Tensor<float> Activate(Tensor<float> h, FusedActivationType activation)
            => activation == FusedActivationType.ReLU ? engine.ReLU(h) : h;
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
                        bool hasBias = !layer.TryGetProperty("bias", out var biasElement) || biasElement.GetBoolean();
                        var w = Parameter(new[] { outChannels, shape[0], k, k });
                        var b = hasBias ? Parameter(new[] { outChannels }) : null;
                        layers.Add(h => engine.FusedConv2D(h, w, b, s, s, pad, pad, 1, 1, act));
                        shape = new[] { outChannels, (shape[1] + 2 * pad - k) / s + 1, (shape[2] + 2 * pad - k) / s + 1 };
                        break;
                    }
                    case "batchnorm2d":
                    {
                        var gamma = Parameter(new[] { shape[0] });
                        var beta = Parameter(new[] { shape[0] });
                        layers.Add(h => Activate(BatchNorm(h, gamma, beta), act));
                        break;
                    }
                    case "basicblock":
                    {
                        int inChannels = shape[0], outChannels = layer.GetProperty("out").GetInt32();
                        int s = layer.TryGetProperty("stride", out var sElement) ? sElement.GetInt32() : 1;
                        // The runner's order: conv1, bn1, conv2, bn2, then the projection shortcut when the shape changes.
                        var conv1 = Parameter(new[] { outChannels, inChannels, 3, 3 });
                        var gamma1 = Parameter(new[] { outChannels });
                        var beta1 = Parameter(new[] { outChannels });
                        var conv2 = Parameter(new[] { outChannels, outChannels, 3, 3 });
                        var gamma2 = Parameter(new[] { outChannels });
                        var beta2 = Parameter(new[] { outChannels });
                        bool project = s != 1 || inChannels != outChannels;
                        // Null when the block has no projection; the three are created together.
                        var projection = project
                            ? (Weight: Parameter(new[] { outChannels, inChannels, 1, 1 }),
                               Gamma: Parameter(new[] { outChannels }),
                               Beta: Parameter(new[] { outChannels }))
                            : ((Tensor<float> Weight, Tensor<float> Gamma, Tensor<float> Beta)?)null;
                        layers.Add(h =>
                        {
                            var o = engine.ReLU(BatchNorm(engine.FusedConv2D(h, conv1, null, s, s, 1, 1, 1, 1, FusedActivationType.None), gamma1, beta1));
                            o = BatchNorm(engine.FusedConv2D(o, conv2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None), gamma2, beta2);
                            var identity = projection is { } p
                                ? BatchNorm(engine.FusedConv2D(h, p.Weight, null, s, s, 0, 0, 1, 1, FusedActivationType.None), p.Gamma, p.Beta)
                                : h;
                            return engine.ReLU(engine.TensorAdd(o, identity));
                        });
                        shape = new[] { outChannels, (shape[1] + 2 - 3) / s + 1, (shape[2] + 2 - 3) / s + 1 };
                        break;
                    }
                    case "globalavgpool":
                    {
                        int channels = shape[0];
                        layers.Add(h => engine.Reshape(engine.GlobalAvgPool2D(h), new[] { batch, channels }));
                        shape = new[] { channels };
                        break;
                    }
                    case "lstm":
                    {
                        int features = shape[1], hidden = layer.GetProperty("hidden").GetInt32();
                        // PyTorch's layout and gate order (i, f, g, o): weight_ih, weight_hh, bias_ih, bias_hh.
                        var weightIh = Parameter(new[] { 4 * hidden, features });
                        var weightHh = Parameter(new[] { 4 * hidden, hidden });
                        var biasIh = Parameter(new[] { 4 * hidden });
                        var biasHh = Parameter(new[] { 4 * hidden });
                        // PyTorch's side runs the fused nn.LSTM, so ours runs the engine's fused sequence op: same
                        // weights and gate order, one tape node with the exact BPTT backward.
                        layers.Add(input => engine.LstmSequenceForward(input, null, null, weightIh, weightHh, biasIh, biasHh));
                        shape = new[] { hidden };
                        break;
                    }
                    case "transformer":
                    {
                        int seq = shape[0], model = shape[1];
                        int heads = layer.GetProperty("heads").GetInt32(), ffn = layer.GetProperty("ffn").GetInt32();
                        int headDim = model / heads;
                        // nn.TransformerEncoderLayer's order: in_proj, out_proj, linear1, linear2, norm1, norm2.
                        var inProj = Parameter(new[] { 3 * model, model });
                        var inBias = Parameter(new[] { 3 * model });
                        var outProj = Parameter(new[] { model, model });
                        var outBias = Parameter(new[] { model });
                        var linear1 = Parameter(new[] { ffn, model });
                        var bias1 = Parameter(new[] { ffn });
                        var linear2 = Parameter(new[] { model, ffn });
                        var bias2 = Parameter(new[] { model });
                        var norm1Gamma = Parameter(new[] { model });
                        var norm1Beta = Parameter(new[] { model });
                        var norm2Gamma = Parameter(new[] { model });
                        var norm2Beta = Parameter(new[] { model });
                        float scale = 1f / MathF.Sqrt(headDim);
                        layers.Add(input =>
                        {
                            var x2 = engine.Reshape(input, new[] { batch * seq, model });
                            var qkv = AddBias(engine.TensorMatMulTransposed(x2, inProj), inBias);
                            Tensor<float> Heads(int part) => engine.Reshape(
                                engine.TensorPermute(engine.Reshape(engine.TensorNarrow(qkv, 1, part * model, model), new[] { batch, seq, heads, headDim }), new[] { 0, 2, 1, 3 }),
                                new[] { batch * heads, seq, headDim });
                            var q = Heads(0);
                            var k = Heads(1);
                            var v = Heads(2);
                            var scores = engine.TensorMultiplyScalar(engine.BatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })), scale);
                            var context = engine.BatchMatMul(engine.Softmax(scores, -1), v);
                            context = engine.Reshape(
                                engine.TensorPermute(engine.Reshape(context, new[] { batch, heads, seq, headDim }), new[] { 0, 2, 1, 3 }),
                                new[] { batch * seq, model });
                            var attention = AddBias(engine.TensorMatMulTransposed(context, outProj), outBias);
                            var y1 = engine.LayerNorm(engine.TensorAdd(x2, attention), norm1Gamma, norm1Beta, NormEpsilon, out _, out _);
                            var hidden = engine.ReLU(AddBias(engine.TensorMatMulTransposed(y1, linear1), bias1));
                            var feedForward = AddBias(engine.TensorMatMulTransposed(hidden, linear2), bias2);
                            var y2 = engine.LayerNorm(engine.TensorAdd(y1, feedForward), norm2Gamma, norm2Beta, NormEpsilon, out _, out _);
                            return engine.Reshape(y2, new[] { batch, seq, model });
                        });
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

        int warmup = capture?.MeasuredStep ?? spec.GetProperty("warmupSteps").GetInt32();
        int measured = capture is null ? spec.GetProperty("measuredSteps").GetInt32() : 1;
        int lossSteps = spec.GetProperty("lossAgreementSteps").GetInt32();
        var phases = new Dictionary<string, List<double>>
        {
            ["forward"] = new(), ["backward"] = new(), ["optimizer"] = new(), ["step"] = new(),
        };
        var losses = new List<double>();
        var sw = new Stopwatch();

        // AiDotNet's training call sites (model bases, Optimize()) open a TensorArena around the loop and every
        // top-level tape resets it on dispose, so a step reuses the previous step's buffers. Without it every
        // step's activations and gradients are fresh large-object-heap arrays and the loop measures gen-2 GCs
        // (one every ~1.5 MLP steps) rather than the framework. CPU only: device buffers don't come from it.
        using var stepArena = gpu is null ? AiDotNet.Tensors.Helpers.TensorArena.Create() : null;

        for (int step = 0; step < warmup + measured; step++)
        {
            Sync();
            // Disposed on failure too, so a throwing step leaves no scope registered on this thread collecting transfers.
            using var crossingScope = capture is not null && step == capture.MeasuredStep
                ? AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true)
                : null;
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

            if (gpu is null)
            {
                // The optimizer updates the raw parameter arrays the tensors were built on; tell the tensors, or the
                // engine keeps using data derived from the previous weights.
                optimizer.Step();
                foreach (var parameter in parameters) parameter.Tensor.MarkModified();
            }
            Sync();
            double t3 = sw.Elapsed.TotalMilliseconds;
            if (crossingScope is not null && capture is not null)
            {
                crossingScope.Dispose();   // closed here, before the report is read; the using is then a no-op
                capture.Crossings = crossingScope.Events
                    .Where(e => e.Kind != AiDotNet.Tensors.Engines.Diagnostics.GpuTransferKind.Synchronize)
                    .GroupBy(e => $"{e.Kind} {e.Operation ?? "<outside the engine>"}")
                    .ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
            }

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
