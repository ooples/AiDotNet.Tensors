#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.Autodiff;
using AiDotNet.Tensors.Tests.Parity;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Device residency for every engine operation (issue #1058): each <see cref="IEngine"/> operation is run on the GPU
/// engine with device-resident inputs, and every host/device crossing it makes is counted, forward and through a
/// backward pass. A checked-in per-operation baseline is a ratchet: no operation may cross more often than recorded,
/// an operation that ran on the device may not start failing there, and an operation the baseline does not list must
/// make no crossing at all. When the baseline is empty the gate is strict: every operation is resident.
/// </summary>
/// <remarks>
/// <para>
/// The MLP ratchet in <see cref="GpuResidencyTests"/> covers the dozen operations one training step uses; a missing
/// kernel anywhere else fell back to the CPU engine unseen, round-tripping its inputs and output through the host,
/// until a model that used it was profiled. This sweep enumerates the engine interface by reflection, so a newly
/// added operation is measured the moment it exists, and builds arguments from the same per-operation tables
/// <see cref="DifferentiableOpsGradCheckSweep"/> uses for gradient checking.
/// </para>
/// <para>
/// Arguments are made in double by those tables and converted to float, the GPU engine's native type. Floating-point
/// tensors are uploaded before the measured call; other tensors (integer indices, masks) stay on the host, so an
/// operation that takes them is charged for their upload: the engine has no device-resident integer tensor yet.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public sealed class AllOpsGpuResidencySweep
{
    private const string BaselineFile = "parity/residency-ops-baseline.json";
    private const string SkippedKey = "_unrunnable";
    private readonly ITestOutputHelper _output;

    public AllOpsGpuResidencySweep(ITestOutputHelper output) => _output = output;

    /// <summary>One operation's measurement: crossings forward and in a forward-plus-backward step, or why it failed.</summary>
    private sealed record OpResult(string Key, int Forward, int Train, string? Error, string Detail);

    /// <summary>
    /// Ratchet (GPU lane): no engine operation crosses the host/device boundary more often than recorded for its
    /// backend, no operation that ran on the device starts failing there, and no new operation crosses at all.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "PyTorchParityGpu")]
    public void EveryEngineOp_CrossingsDoNotRegress()
    {
        var (backend, results, unrunnable) = RunSweep();
        string root = PyTorchParityInventory.FindRepositoryRoot()
            ?? throw new InvalidOperationException("parity/ is missing from this checkout.");
        string path = System.IO.Path.Combine(root, BaselineFile);
        string snapshot = Snapshot(results, unrunnable);
        string snapshotPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"residency-ops-{backend}.json");
        System.IO.File.WriteAllText(snapshotPath, snapshot);
        _output.WriteLine($"Snapshot for {backend} written to {snapshotPath}");

        using var doc = System.Text.Json.JsonDocument.Parse(System.IO.File.ReadAllText(path));
        bool recorded = doc.RootElement.TryGetProperty(backend, out var baseline);
        Assert.True(recorded,
            $"No per-operation residency baseline for {backend}. Measure it on that backend and add the snapshot in " +
            $"{snapshotPath} under \"{backend}\" in {BaselineFile}.");

        var failures = new List<string>();
        var improved = new List<string>();
        foreach (var r in results)
        {
            bool known = baseline.TryGetProperty(r.Key, out var b);
            string? allowedError = known && b.ValueKind == System.Text.Json.JsonValueKind.String ? b.GetString() : null;
            int allowedForward = known && b.ValueKind == System.Text.Json.JsonValueKind.Array ? b[0].GetInt32() : 0;
            int allowedTrain = known && b.ValueKind == System.Text.Json.JsonValueKind.Array ? b[1].GetInt32() : 0;

            if (r.Error is not null)
            {
                if (allowedError is null)
                    failures.Add($"  {r.Key}: now fails on the device ({r.Error}); recorded {(known ? $"[{allowedForward}, {allowedTrain}]" : "nothing")}");
                continue;
            }
            if (allowedError is not null)
            {
                improved.Add($"{r.Key} (now runs: [{r.Forward}, {r.Train}])");
                continue;
            }
            if (r.Forward > allowedForward || r.Train > allowedTrain)
                failures.Add($"  {r.Key}: [{r.Forward}, {r.Train}] crossings (recorded [{allowedForward}, {allowedTrain}]) {r.Detail}");
            else if (r.Forward < allowedForward || r.Train < allowedTrain)
                improved.Add($"{r.Key} ([{r.Forward}, {r.Train}] < [{allowedForward}, {allowedTrain}])");
        }

        int allowedUnrunnable = baseline.TryGetProperty(SkippedKey, out var s) ? s.GetArrayLength() : 0;
        var allowedUnrunnableKeys = baseline.TryGetProperty(SkippedKey, out var s2)
            ? s2.EnumerateArray().Select(e => e.GetString()!).ToHashSet(StringComparer.Ordinal)
            : new HashSet<string>(StringComparer.Ordinal);
        foreach (var u in unrunnable.Where(u => !allowedUnrunnableKeys.Contains(u.Key)))
            failures.Add($"  {u.Key}: the sweep cannot drive it ({u.Value}); add a per-op argument entry to " +
                         $"{nameof(DifferentiableOpsGradCheckSweep)}.{nameof(DifferentiableOpsGradCheckSweep.OpCases)}");

        if (improved.Count > 0)
            _output.WriteLine($"IMPROVED ({improved.Count}): {string.Join(", ", improved)}. Record {snapshotPath} in {BaselineFile} to lock it in.");
        Assert.True(failures.Count == 0,
            $"{backend}: {failures.Count} operation(s) regressed device residency:" + Environment.NewLine +
            string.Join(Environment.NewLine, failures));
    }

    private (string Backend, List<OpResult> Results, SortedDictionary<string, string> Unrunnable) RunSweep()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { Skip.If(true, $"DEFERRED: no GPU backend ({ex.GetType().Name})."); throw; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "DEFERRED: no GPU available."); }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var deviceBackend = gpu.GetBackend() ?? throw new InvalidOperationException("The GPU engine has no backend.");
            GpuBackendType backendType;
            using (var probe = GpuResidencyScope.Begin())
            {
                gpu.UploadToGpu(new Tensor<float>(new[] { 1f }, new[] { 1 }), GpuTensorRole.General);
                Assert.True(probe.Uploads >= 1, "The residency probe did not see a deliberate upload; its counts mean nothing.");
                backendType = probe.Events.First(e => e.Kind == GpuTransferKind.HostToDevice).Backend;
            }

            var cpu = new CpuEngine();
            var results = new List<OpResult>();
            var unrunnable = new SortedDictionary<string, string>(StringComparer.Ordinal);
            var candidates = typeof(IEngine).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(m => !m.IsSpecialName && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1)
                .Where(m => m.ReturnType.IsGenericType && m.ReturnType.GetGenericTypeDefinition() == typeof(Tensor<>)
                            && m.ReturnType.GetGenericArguments()[0].IsGenericParameter)
                .OrderBy(m => m.Name, StringComparer.Ordinal)
                .ToList();

            foreach (var def in candidates)
            {
                MethodInfo md, mf;
                try { md = def.MakeGenericMethod(typeof(double)); mf = def.MakeGenericMethod(typeof(float)); }
                catch (ArgumentException) { continue; }   // constrained away from float: not a GPU operation
                string name = def.Name;
                string key = $"{name}|{DifferentiableOpsGradCheckSweep.ParamFingerprint(md)}";

                object[]? dArgs = BuildDoubleArgs(name, key, md, out string why);
                if (dArgs is null) { unrunnable[key] = why; continue; }
                object?[]? hostArgs = ToFloat(mf, dArgs);
                if (hostArgs is null) { unrunnable[key] = "argument has no float counterpart"; continue; }

                // An operation the CPU engine rejects with these arguments is a harness gap, not a residency finding.
                try { _ = mf.Invoke(cpu, (object?[])hostArgs.Clone()); }
                catch (Exception ex) { unrunnable[key] = $"CPU float threw {Describe(ex)}"; continue; }

                bool differentiable = OpRegistry.DifferentiableOps.Contains(name)
                    && !(DifferentiableOpsGradCheckSweep.Exempt.TryGetValue(name, out var exemptWhy) && exemptWhy.StartsWith("stochastic", StringComparison.Ordinal));
                var result = Measure(gpu, deviceBackend, mf, key, differentiable, ToDevice(gpu, hostArgs));
                if (result.Error is null && differentiable)
                {
                    string? mismatch = CompareGradients(gpu, cpu, mf, hostArgs);
                    if (mismatch is not null) result = result with { Error = mismatch };
                }
                results.Add(result);
            }

            var withCrossings = results.Where(r => r.Error is null && (r.Forward > 0 || r.Train > 0)).ToList();
            var failed = results.Where(r => r.Error is not null).ToList();
            _output.WriteLine($"{backendType}: {results.Count} operations measured; {results.Count - withCrossings.Count - failed.Count} resident, " +
                              $"{withCrossings.Count} cross the boundary, {failed.Count} fail on the device; {unrunnable.Count} cannot be driven.");
            foreach (var r in withCrossings.OrderByDescending(r => r.Forward + r.Train))
                _output.WriteLine($"  CROSS  [{r.Forward}, {r.Train}] {r.Key} {r.Detail}");
            foreach (var r in failed) _output.WriteLine($"  FAIL   {r.Key}: {r.Error}");
            foreach (var u in unrunnable) _output.WriteLine($"  SKIP   {u.Key}: {u.Value}");
            return (backendType.ToString(), results, unrunnable);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    private static object[]? BuildDoubleArgs(string name, string key, MethodInfo md, out string why)
    {
        var cases = DifferentiableOpsGradCheckSweep.OpCases;
        why = "unsupported parameter types (NEEDS TABLE ENTRY)";
        if (cases.TryGetValue(key, out var factory)
            || cases.TryGetValue($"{name}/{md.GetParameters().Length}", out factory)
            || cases.TryGetValue(name, out factory))
        {
            try { return factory(new Random(1234)); }
            catch (Exception ex) { why = $"table entry threw {Describe(ex)}"; return null; }
        }
        foreach (var shape in new[] { new[] { 2, 3 }, new[] { 6 } })
        {
            var args = DifferentiableOpsGradCheckSweep.BuildArgs(md, shape, new Random(1234), out _);
            if (args is not null) return args;
        }
        return null;
    }

    /// <summary>One warmed-up forward, then one warmed-up forward-plus-backward, each counted in its own scope.</summary>
    private static OpResult Measure(DirectGpuTensorEngine gpu, AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend deviceBackend, MethodInfo m, string key,
        bool differentiable, object?[] args)
    {
        Tensor<float> Forward() => (Tensor<float>)m.Invoke(gpu, (object?[])args.Clone())!;
        var inputs = args.SelectMany(a => a switch
        {
            Tensor<float> t => new[] { t },
            Tensor<float>[] arr => arr,
            IEnumerable<Tensor<float>> seq => seq.ToArray(),
            _ => Array.Empty<Tensor<float>>(),
        }).ToArray();
        void Train()
        {
            using var tape = new GradientTape<float>();
            var loss = gpu.ReduceSum(Forward(), null);
            _ = tape.ComputeGradients(loss, inputs);
        }

        try
        {
            Forward(); Forward();
            deviceBackend.Synchronize();
        }
        catch (Exception ex) { return new OpResult(key, 0, 0, $"forward threw {Describe(ex)}", ""); }

        var detail = new StringBuilder();
        int forward;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true)) Forward();
        forward = scope.Events.Count;
        Summarize(detail, "fwd", scope);
        deviceBackend.Synchronize();
        if (forward > 0) AppendSites(detail, "fwd", () => Forward(), deviceBackend);

        int train = 0;
        if (differentiable && inputs.Length > 0)
        {
            try
            {
                Train(); Train();
                deviceBackend.Synchronize();
            }
            catch (Exception ex) { return new OpResult(key, forward, 0, $"backward threw {Describe(ex)}", detail.ToString()); }
            using (scope = GpuResidencyScope.Begin(captureOperations: true)) Train();
            train = scope.Events.Count;
            Summarize(detail, "train", scope);
            deviceBackend.Synchronize();
            if (train > 0) AppendSites(detail, "train", Train, deviceBackend);
        }
        return new OpResult(key, forward, train, null, detail.ToString());
    }

    /// <summary>
    /// Re-runs a phase that crossed with call-site capture on, so a failure names the file and line of every transfer
    /// and every silent CPU fallback instead of only the engine operation.
    /// </summary>
    private static void AppendSites(StringBuilder sb, string phase, Action run, AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend)
    {
        bool saved = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.CaptureReadbackSites = true;
            AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Reset();
            run();
            backend.Synchronize();
            var sites = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.ReadbackSites.Select(x => "down " + x)
                .Concat(AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.UploadSites.Select(x => "up " + x))
                .Concat(AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.Fallbacks.Select(x => "fallback " + x));
            sb.Append(phase).Append("-sites{").Append(string.Join(" | ", sites)).Append("} ");
        }
        catch (Exception ex) { sb.Append(phase).Append("-sites{rerun threw ").Append(Describe(ex)).Append("} "); }
        finally { AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.CaptureReadbackSites = saved; }
    }

    /// <summary>
    /// The device tape's gradients for every float input against the CPU engine's on the same float values. A GPU
    /// override that records its own tape node, rather than handing the op to the CPU engine under a tape, must
    /// produce the CPU engine's gradients; null when they agree, otherwise what differs.
    /// </summary>
    private static string? CompareGradients(DirectGpuTensorEngine gpu, CpuEngine cpu, MethodInfo m, object?[] hostArgs)
    {
        Tensor<float>[] Inputs(object?[] a) => a.SelectMany(x => x switch
        {
            Tensor<float> t => new[] { t },
            Tensor<float>[] arr => arr,
            _ => Array.Empty<Tensor<float>>(),
        }).ToArray();

        float[][] Grads(IEngine engine, object?[] args)
        {
            var inputs = Inputs(args);
            var previous = AiDotNetEngine.Current;
            AiDotNetEngine.Current = engine;
            try
            {
                using var tape = new GradientTape<float>();
                var output = (Tensor<float>)m.Invoke(engine, (object?[])args.Clone())!;
                var grads = tape.ComputeGradients(engine.ReduceSum(output, null), inputs);
                return inputs.Select(t => grads.TryGetValue(t, out var g) && g is not null ? g.ToArray() : Array.Empty<float>()).ToArray();
            }
            finally { AiDotNetEngine.Current = previous; }
        }

        float[][] expected, actual;
        try { expected = Grads(cpu, hostArgs); }
        catch (Exception) { return null; }   // the CPU engine cannot differentiate it here: nothing to compare against
        try { actual = Grads(gpu, ToDevice(gpu, hostArgs)); }
        catch (Exception ex) { return $"gradient threw on the device ({Describe(ex)})"; }

        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i].Length != actual[i].Length)
                return $"gradient of input {i}: {actual[i].Length} element(s) on the device, {expected[i].Length} on the CPU";
            double scale = 0, worst = 0; int at = -1;
            for (int j = 0; j < expected[i].Length; j++) scale = Math.Max(scale, Math.Abs(expected[i][j]));
            for (int j = 0; j < expected[i].Length; j++)
            {
                double diff = Math.Abs((double)expected[i][j] - actual[i][j]);
                if (double.IsNaN(actual[i][j]) && !double.IsNaN(expected[i][j])) diff = double.PositiveInfinity;
                if (diff > worst) { worst = diff; at = j; }
            }
            if (worst > 1e-3 * Math.Max(1.0, scale))
                return $"gradient of input {i} differs from the CPU engine's: [{at}] device {actual[i][at]:G6} vs CPU {expected[i][at]:G6} (max |g| {scale:G4})";
        }
        return null;
    }

    private static void Summarize(StringBuilder sb, string phase, GpuResidencyScope scope)
    {
        if (scope.Events.Count == 0) return;
        sb.Append(phase).Append('{');
        sb.Append(string.Join("; ", scope.Events
            .GroupBy(e => (e.Kind, Op: ShortOp(e.Operation)))
            .Select(g => $"{g.Key.Kind switch { GpuTransferKind.HostToDevice => "up", GpuTransferKind.DeviceToHost => "down", _ => "sync" }}x{g.Count()} {g.Key.Op}")));
        sb.Append("} ");
    }

    private static string ShortOp(string? op) => op is null ? "<outside the engine>"
        : op.StartsWith("DirectGpuTensorEngine.", StringComparison.Ordinal) ? op.Substring("DirectGpuTensorEngine.".Length) : op;

    /// <summary>Converts double arguments to the float overload's: tensors element-wise, generic scalars by cast.</summary>
    private static object?[]? ToFloat(MethodInfo mf, object[] dArgs)
    {
        var ps = mf.GetParameters();
        var result = new object?[dArgs.Length];
        for (int i = 0; i < dArgs.Length; i++)
        {
            var target = ps[i].ParameterType.IsByRef ? ps[i].ParameterType.GetElementType()! : ps[i].ParameterType;
            object? a = dArgs[i];
            switch (a)
            {
                case Tensor<double> t when target == typeof(Tensor<float>):
                    result[i] = ToFloat(t);
                    break;
                case IEnumerable<Tensor<double>> seq when target.IsAssignableFrom(typeof(Tensor<float>[])):
                    result[i] = seq.Select(ToFloat).ToArray();
                    break;
                case double d when target == typeof(float):
                    result[i] = (float)d;
                    break;
                default:
                    if (a is null || target.IsInstanceOfType(a)) result[i] = a;
                    else return null;
                    break;
            }
        }
        return result;
    }

    private static Tensor<float> ToFloat(Tensor<double> t)
    {
        var src = t.GetDataArray();
        var dst = new float[t.Length];
        for (int i = 0; i < dst.Length; i++) dst[i] = (float)src[i];
        return new Tensor<float>(dst, t.Shape.ToArray());
    }

    /// <summary>Uploads the floating-point tensors; everything else is left as the caller would pass it.</summary>
    private static object?[] ToDevice(DirectGpuTensorEngine gpu, object?[] args) => args.Select(a => a switch
    {
        Tensor<float> t => (object)gpu.UploadToGpu(t, GpuTensorRole.General),
        Tensor<float>[] arr => arr.Select(t => gpu.UploadToGpu(t, GpuTensorRole.General)).ToArray(),
        // Index and mask tensors live on the device too, as a training step's token ids and masks do; an op that
        // reads one back to the host (a host-side bounds check, say) is a crossing like any other.
        Tensor<int> t => TryUpload(gpu, t),
        Tensor<int>[] arr => arr.Select(t => TryUpload(gpu, t)).ToArray(),
        Tensor<Bit> t => TryUpload(gpu, t),
        _ => a,
    }).ToArray();

    /// <summary>The device copy of a non-float tensor, or the host tensor when the device cannot hold that type.</summary>
    private static Tensor<TElem> TryUpload<TElem>(DirectGpuTensorEngine gpu, Tensor<TElem> t)
    {
        try { return gpu.UploadToGpu(t, GpuTensorRole.General); }
        catch (Exception ex) when (ex is NotSupportedException or InvalidCastException or ArgumentException) { return t; }
    }

    private static string Describe(Exception ex)
    {
        var inner = DifferentiableOpsGradCheckSweep.Inner(ex);
        string msg = inner.Message.Replace('\r', ' ').Replace('\n', ' ');
        return $"{inner.GetType().Name}: {(msg.Length > 160 ? msg.Substring(0, 160) + "..." : msg)}";
    }

    /// <summary>The measured results in the baseline's format, one operation per line so a diff reads per operation.</summary>
    private static string Snapshot(List<OpResult> results, SortedDictionary<string, string> unrunnable)
    {
        var sb = new StringBuilder("{").AppendLine();
        foreach (var r in results.OrderBy(r => r.Key, StringComparer.Ordinal))
        {
            if (r.Error is null && r.Forward == 0 && r.Train == 0) continue;   // resident: absent means zero
            sb.Append("  ").Append(System.Text.Json.JsonSerializer.Serialize(r.Key)).Append(": ")
              .Append(r.Error is null ? $"[{r.Forward}, {r.Train}]" : System.Text.Json.JsonSerializer.Serialize(r.Error))
              .AppendLine(",");
        }
        sb.Append("  \"").Append(SkippedKey).Append("\": [");
        sb.Append(string.Join(", ", unrunnable.Keys.Select(k => System.Text.Json.JsonSerializer.Serialize(k))));
        sb.AppendLine("]").Append('}');
        return sb.ToString();
    }
}
#endif
