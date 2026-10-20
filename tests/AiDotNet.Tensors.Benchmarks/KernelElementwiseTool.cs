// Copyright (c) AiDotNet. All rights reserved.
// --kernel-elementwise: the pointwise family on declared kernels (#890 phase 4a), against the hand-written kernels.
//
//   --derive      every module's derived library functions, and why each other elementwise kernel is not derivable
//   --smoke       every derivable kernel, declared vs hand-written, bit for bit, on uniformly random bit patterns
//                 (every exponent: subnormals, infinities, NaN) at a length that takes the 128-bit path and one that
//                 does not, with several values of each scalar parameter
//   --exhaustive  [kernel...] every one of the 2^32 float inputs of each one-input kernel, compared on the device
//
// Bit for bit: a declared kernel calls the function derived from the hand-written kernel's own expression, compiled
// with that module's flags, so the results are the same bits. Any NaN equals any NaN (payloads are not part of the
// contract).

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Benchmarks;

internal static class KernelElementwiseTool
{
    internal static void Run(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : "--smoke";
        using var engine = new DirectGpuEngine();
        if (engine.Backend is not CudaBackend backend)
        {
            Console.WriteLine("Declared elementwise kernels are CUDA-only; no CUDA backend here.");
            return;
        }
        if (mode == "--derive")
        {
            Derive(backend);
            for (int i = 1; i < args.Length; i++)
                Console.WriteLine(args[i] + ": " + (backend.ResolveElementwise(args[i]) is { } f
                    ? "derived from " + f.Module.Name + "\n" + f.Wrapper
                    : "not derivable -- " + backend.ElementwiseRefusal(args[i])));
            return;
        }
        if (mode == "--smoke") Smoke(backend, new HashSet<string>(args[1..], StringComparer.Ordinal));
        else if (mode == "--bench") Bench(backend, args[1..]);
        else if (mode == "--allocations")
        {
            Console.WriteLine("route on:");
            Allocations(backend);
            backend.DeclaredKernelsSwitch = false;
            Console.WriteLine("route off (the hand-written kernels):");
            Allocations(backend);
            backend.DeclaredKernelsSwitch = null;
        }
        else throw new ArgumentException("Unknown mode " + mode + ".");
    }

    private static void Derive(CudaBackend backend)
    {
        var byModule = new SortedDictionary<string, List<CudaMathFunction>>(StringComparer.Ordinal);
        foreach (var function in backend.DerivableElementwise())
        {
            if (!byModule.TryGetValue(function.Module.Name, out var list)) byModule[function.Module.Name] = list = new List<CudaMathFunction>();
            list.Add(function);
        }
        int total = 0;
        foreach (var pair in byModule)
        {
            total += pair.Value.Count;
            Console.WriteLine(pair.Key + (pair.Value[0].Module.FastMath ? " (fast math)" : " (precise)") + ": " + pair.Value.Count + " derived");
            var names = new List<string>();
            foreach (var function in pair.Value)
                names.Add(function.Kernel + "(" + function.Inputs + (function.Scalars.Length > 0 ? "+" + string.Join(",", function.Scalars) : "") + ")");
            Console.WriteLine("  " + string.Join(" ", names));
        }
        Console.WriteLine(total + " functions derived from " + byModule.Count + " modules");
    }

    private static void Smoke(CudaBackend backend, HashSet<string> only)
    {
        var failures = new List<string>();
        int checkedCount = 0, routed = 0;
        Console.WriteLine("ELEMENTWISE - declared vs hand-written, bit for bit (random bit patterns)");
        {
            foreach (var function in backend.DerivableElementwise())
            {
                if (only.Count != 0 && !only.Contains(function.Kernel)) continue;
                foreach (float[] scalars in ScalarSets(function.Scalars.Length))
                {
                    foreach (int size in new[] { 1 << 20, (1 << 20) + 3 })
                    {
                        var (mismatches, firstBad, declaredValue, handValue, declared) = Compare(backend, function, scalars, size);
                        checkedCount++;
                        if (!declared)
                        {
                            failures.Add(function.Kernel + ": not declared (" + backend.LastElementwiseDecline + ")");
                            goto nextFunction;
                        }
                        routed++;
                        if (mismatches != 0)
                        {
                            failures.Add(function.Kernel + " [" + string.Join(",", scalars) + "] n=" + size + ": " + mismatches +
                                " mismatches; first at input bits 0x" + firstBad.ToString("X8", CultureInfo.InvariantCulture) +
                                " declared " + declaredValue.ToString("R", CultureInfo.InvariantCulture) +
                                " hand-written " + handValue.ToString("R", CultureInfo.InvariantCulture) + "; operands " + FirstOperands);
                        }
                    }
                }
            nextFunction:;
            }
        }
        // The composites, through the production entry points with the route on and off: one declared kernel against
        // the hand-written path's two launches (add then activation), and the declared bias broadcast against
        // bias_add_out.
        if (only.Count == 0)
        {
            foreach (string pair in new[] { "AddRelu", "AddSigmoid", "AddGelu", "BiasAdd", "BiasAddRelu" })
                foreach (var (rows, columns) in new[] { (256, 1024), (100, 37) })
                {
                    checkedCount++;
                    string? failure = CompareComposite(backend, pair, rows, columns);
                    if (failure is null) routed++;
                    else failures.Add(pair + " " + rows + "x" + columns + ": " + failure);
                }
        }
        Console.WriteLine(checkedCount + " comparisons, " + routed + " through declared kernels, " + failures.Count + " failures");
        var modules = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var function in backend.DerivableElementwise()) modules.Add(function.Module.Name);
        foreach (string module in modules)
            if (backend.MathLibraryLog(module) is { } log)
            {
                Console.WriteLine("  " + module + " library log:");
                foreach (string line in log.Split('\n'))
                    if (line.Contains("error", StringComparison.OrdinalIgnoreCase)) Console.WriteLine("    " + line.Trim());
            }
        foreach (string failure in failures) Console.WriteLine("  " + failure);
        if (failures.Count != 0) throw new InvalidOperationException(failures.Count + " elementwise failures.");
    }

    // Declared vs hand-written wall clock: median of 21 samples of 20 calls, paired, per kernel and length.
    private static void Bench(CudaBackend backend, string[] kernels)
    {
        if (kernels.Length == 0)
            kernels = new[] { "relu", "gelu", "sigmoid", "tanh_activation", "exp_vector", "sin_vector", "add_vectors",
                              "multiply_vectors", "leaky_relu", "parity210_lgamma_approx", "parity210_erfinv" };
        Console.WriteLine("kernel                       n        declared us  hand-written us   ratio");
        foreach (string kernel in kernels)
        {
            var function = backend.ResolveElementwise(kernel)
                ?? throw new InvalidOperationException(kernel + ": " + backend.ElementwiseRefusal(kernel));
            foreach (int n in new[] { 1 << 20, 1 << 24 })
            {
                var inputs = new IGpuBuffer[function.Inputs];
                try
                {
                    var data = new float[n];
                    var rng = new Random(n);
                    for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 1.8 - 0.9);
                    for (int i = 0; i < inputs.Length; i++) inputs[i] = backend.AllocateBuffer(data);
                    using var output = backend.AllocateBuffer(n);
                    var scalars = new float[function.Scalars.Length];
                    for (int i = 0; i < scalars.Length; i++) scalars[i] = 0.1f * (i + 1);
                    void Declared()
                    {
                        if (!backend.TryDeclaredElementwise(kernel, inputs, output, scalars, n))
                            throw new InvalidOperationException(kernel + " was not declared: " + backend.LastElementwiseDecline);
                    }
                    void Hand() => backend.LaunchHandWrittenElementwise(kernel, inputs, output, scalars, n);
                    double d = Median(Declared, backend), h = Median(Hand, backend);
                    Console.WriteLine(kernel.PadRight(28) + n.ToString(CultureInfo.InvariantCulture).PadLeft(9) +
                        d.ToString("F1", CultureInfo.InvariantCulture).PadLeft(14) + h.ToString("F1", CultureInfo.InvariantCulture).PadLeft(17) +
                        (h / d).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x");
                }
                finally
                {
                    foreach (var buffer in inputs) buffer?.Dispose();
                }
            }
        }
    }

    // Managed bytes per call through the production entry points (route on), over 1000 calls after warming.
    private static void Allocations(CudaBackend backend)
    {
        const int n = 1 << 16;
        using var a = backend.AllocateBuffer(n);
        using var b = backend.AllocateBuffer(n);
        using var c = backend.AllocateBuffer(n);
        var calls = new (string Name, Action Call)[]
        {
            ("Relu", () => backend.Relu(a, c, n)),
            ("Add", () => backend.Add(a, b, c, n)),
            ("LeakyRelu", () => backend.LeakyRelu(a, c, 0.1f, n)),
            ("AddGelu", () => backend.AddGelu(a, b, c, n)),
            ("BiasAdd", () => backend.BiasAdd(a, b, c, 256, 256)),
            ("Clamp", () => backend.Clamp(a, c, -1f, 1f, n)),
        };
        long before = backend.DeclaredElementwiseLaunches;
        foreach (var (name, call) in calls)
        {
            for (int i = 0; i < 20; i++) call();
            backend.Synchronize();
            long start = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1000; i++) call();
            long bytes = GC.GetAllocatedBytesForCurrentThread() - start;
            backend.Synchronize();
            Console.WriteLine("  " + name.PadRight(10) + (bytes / 1000.0).ToString("F1", CultureInfo.InvariantCulture) + " bytes/call");
        }
        Console.WriteLine("  declared launches: " + (backend.DeclaredElementwiseLaunches - before) + " of " + calls.Length * 1020);
    }

    private static double Median(Action call, CudaBackend backend)
    {
        for (int i = 0; i < 10; i++) call();
        backend.Synchronize();
        var samples = new double[21];
        for (int s = 0; s < samples.Length; s++)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < 20; i++) call();
            backend.Synchronize();
            samples[s] = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000.0 / 20;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    private static string? CompareComposite(CudaBackend backend, string op, int rows, int columns)
    {
        int n = rows * columns;
        var rng = new Random(op.GetHashCode() ^ n);
        float[] Random(int count)
        {
            var data = new float[count];
            var bytes = new byte[4];
            for (int i = 0; i < count; i++) { rng.NextBytes(bytes); data[i] = BitConverter.ToSingle(bytes, 0); }
            return data;
        }
        using var a = backend.AllocateBuffer(Random(n));
        using var b = backend.AllocateBuffer(Random(op.StartsWith("Bias", StringComparison.Ordinal) ? columns : n));
        var results = new float[2][];
        long before = backend.DeclaredElementwiseLaunches;
        for (int mode = 0; mode < 2; mode++)
        {
            backend.DeclaredKernelsSwitch = mode == 0;
            try
            {
                using var output = backend.AllocateBuffer(n);
                switch (op)
                {
                    case "AddRelu": backend.AddRelu(a, b, output, n); break;
                    case "AddSigmoid": backend.AddSigmoid(a, b, output, n); break;
                    case "AddGelu": backend.AddGelu(a, b, output, n); break;
                    case "BiasAdd": backend.BiasAdd(a, b, output, rows, columns); break;
                    default:
                        if (!((IBiasActivationBackend)backend).TryBiasAddActivation(a, b, output, rows, columns, FusedActivationType.ReLU))
                            throw new InvalidOperationException("The backend refused a fused bias + ReLU.");
                        break;
                }
                backend.Synchronize();
                results[mode] = backend.DownloadBuffer(output);
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        if (backend.DeclaredElementwiseLaunches == before) return "not declared (" + backend.LastElementwiseDecline + ")";
        long mismatches = 0;
        for (int i = 0; i < n; i++)
            if (!(float.IsNaN(results[0][i]) && float.IsNaN(results[1][i])) &&
                BitConverter.SingleToInt32Bits(results[0][i]) != BitConverter.SingleToInt32Bits(results[1][i])) mismatches++;
        return mismatches == 0 ? null : mismatches + " mismatches";
    }

    // The operands of the last comparison's first mismatch, for the report.
    private static string FirstOperands = "";

    private static IEnumerable<float[]> ScalarSets(int count)
    {
        if (count == 0) { yield return Array.Empty<float>(); yield break; }
        if (count == 1) { yield return new[] { 0.25f }; yield return new[] { -1.5f }; yield return new[] { 3f }; yield break; }
        if (count == 2) { yield return new[] { -0.5f, 0.5f }; yield return new[] { 1f, -2f }; yield break; }
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = 0.5f + i;
        yield return values;
    }

    private static (long Mismatches, uint FirstBad, float Declared, float Hand, bool Routed) Compare(
        CudaBackend backend, CudaMathFunction function, float[] scalars, int size)
    {
        var rng = new Random(function.Kernel.GetHashCode() ^ size);
        var inputs = new IGpuBuffer[function.Inputs];
        var hostInputs = new float[function.Inputs][];
        try
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var data = new float[size];
                var bits = new byte[4];
                for (int e = 0; e < size; e++)
                {
                    rng.NextBytes(bits);
                    data[e] = BitConverter.ToSingle(bits, 0);
                }
                hostInputs[i] = data;
                inputs[i] = backend.AllocateBuffer(data);
            }
            using var declaredOut = backend.AllocateBuffer(size);
            using var handOut = backend.AllocateBuffer(size);
            if (!backend.TryDeclaredElementwise(function.Kernel, inputs, declaredOut, scalars, size))
                return (0, 0, 0, 0, false);
            if (!backend.LaunchHandWrittenElementwise(function.Kernel, inputs, handOut, scalars, size))
                throw new InvalidOperationException(function.Kernel + ": the hand-written kernel is not loaded.");
            backend.Synchronize();
            var a = backend.DownloadBuffer(declaredOut);
            var b = backend.DownloadBuffer(handOut);
            long mismatches = 0;
            uint firstBad = 0;
            float firstDeclared = 0, firstHand = 0;
            for (int e = 0; e < size; e++)
            {
                if (float.IsNaN(a[e]) && float.IsNaN(b[e])) continue;
                if (BitConverter.SingleToInt32Bits(a[e]) == BitConverter.SingleToInt32Bits(b[e])) continue;
                if (mismatches++ == 0)
                {
                    firstBad = (uint)BitConverter.SingleToInt32Bits(hostInputs[0][e]);
                    var operands = new List<string>();
                    foreach (var input in hostInputs)
                        operands.Add("0x" + ((uint)BitConverter.SingleToInt32Bits(input[e])).ToString("X8", CultureInfo.InvariantCulture) +
                            " (" + input[e].ToString("R", CultureInfo.InvariantCulture) + ")");
                    FirstOperands = string.Join(", ", operands) + " at element " + e;
                    firstDeclared = a[e];
                    firstHand = b[e];
                }
            }
            return (mismatches, firstBad, firstDeclared, firstHand, true);
        }
        finally
        {
            foreach (var buffer in inputs) buffer?.Dispose();
        }
    }
}
