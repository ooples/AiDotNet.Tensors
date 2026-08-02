// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Scores mapped kernels from the shipped, hand-written CUDA source library against ceilings
/// derived from semantic specs.
/// </summary>
/// <remarks>
/// <para>
/// The generated-kernel oracle used to score synthetic specs only. That answers how good the
/// emitter is, but not how much headroom exists in the kernels a caller receives today. This
/// tool starts from the actual <c>GetKernelNames()</c> registries, maps the incumbents for which
/// an equivalent spec exists, and times the backend dispatch itself.
/// </para>
/// <para>
/// Coverage is printed and deliberately incomplete. A kernel without a semantic spec has no
/// defensible byte count, operation count, or ceiling; inventing one from its source text would
/// put a precise-looking percentage on an assumption. Unmapped kernels remain visible debt.
/// </para>
/// </remarks>
internal static class HandWrittenKernelOracleTool
{
    private const int AxisRows = 4096;
    private const int AxisInner = 1024;

    private static readonly string[] MappedKernelNames =
    {
        "embedding_forward",
        "sgd_momentum_update",
        "sum_axis",
        "mean_axis",
        "max_axis",
        "add_vectors_vec4",
        "multiply_vectors_vec4",
        "relu_vec4",
        "sigmoid_vec4",
        "tanh_activation_vec4",
        "gelu_vec4",
    };

    internal static void Run(string[] args)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("CUDA backend unavailable; hand-written library cannot be timed.");
            return;
        }

        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;
        var rates = DeviceCalibration.Measure(runtime, major, minor);
        var reference = CodegenMachineModel.Rtx3080Locked;
        var machine = DeviceCalibration.ToMachineModel(
            rates, reference.Multiprocessors, reference.ClockHz);

        var registered = RegisteredHandWrittenKernels();
        string[] staleMappings = MappedKernelNames
            .Where(name => !registered.ContainsKey(name))
            .ToArray();
        if (staleMappings.Length != 0)
        {
            throw new InvalidOperationException(
                "Semantic mappings no longer name registered CUDA kernels: " +
                string.Join(", ", staleMappings));
        }

        Console.WriteLine();
        Console.WriteLine("HAND-WRITTEN CUDA ORACLE - shipped kernels against spec-derived ceilings");
        Console.WriteLine("device: {0}", backend.DeviceName);
        Console.WriteLine("semantic coverage: {0} / {1} registered hand-written CUDA kernels",
            MappedKernelNames.Length, registered.Count);
        Console.WriteLine("unmapped kernels receive no invented ceiling");

        Console.WriteLine();
        Console.WriteLine("{0,-32} {1,14} {2,10} {3,9} {4,8}",
            "incumbent kernel", "measured", "ceiling", "% of max", "limiter");

        int failures = 0;
        void Record(bool succeeded)
        {
            if (!succeeded) failures++;
        }

        Record(ScoreBinary(backend, machine, major, minor,
            "add_vectors_vec4", multiply: false));
        Record(ScoreBinary(backend, machine, major, minor,
            "multiply_vectors_vec4", multiply: true));
        Record(ScoreUnary(backend, machine, major, minor,
            "relu_vec4", CodegenActivationKind.ReLU,
            static (b, input, output, count) => b.Relu(input, output, count)));
        Record(ScoreUnary(backend, machine, major, minor,
            "sigmoid_vec4", CodegenActivationKind.Sigmoid,
            static (b, input, output, count) => b.Sigmoid(input, output, count)));
        Record(ScoreUnary(backend, machine, major, minor,
            "tanh_activation_vec4", CodegenActivationKind.Tanh,
            static (b, input, output, count) => b.Tanh(input, output, count)));
        Record(ScoreUnary(backend, machine, major, minor,
            "gelu_vec4", CodegenActivationKind.Gelu,
            static (b, input, output, count) => b.Gelu(input, output, count)));
        Record(ScoreEmbeddingForward(backend, machine, major, minor));
        Record(ScoreSgdMomentum(backend, machine, major, minor));
        Record(ScoreAxis(backend, machine, major, minor,
            "sum_axis", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.SumAxis(input, output, rows, inner)));
        Record(ScoreAxis(backend, machine, major, minor,
            "mean_axis", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.MeanAxis(input, output, rows, inner),
            mean: true));
        Record(ScoreAxis(backend, machine, major, minor,
            "max_axis", CodegenReduceKind.Max,
            static (b, input, output, rows, inner) => b.MaxAxis(input, output, rows, inner)));

        if (failures != 0)
            throw new InvalidOperationException(
                failures + " mapped hand-written kernel(s) failed; coverage ledger not written.");

        string coveragePath = KernelToolArgs.ValueOf(args, "--coverage-out") ??
            Path.Combine(Directory.GetCurrentDirectory(), "artifacts",
                "handwritten-kernel-coverage.tsv");
        WriteCoverageLedger(coveragePath, registered);
        Console.WriteLine("coverage debt ledger: {0}", coveragePath);

        Console.WriteLine();
        Console.WriteLine("Adam/AdamW are registered incumbents but remain unmapped: the current spec");
        Console.WriteLine("cannot express grad^2, two independently-computed moments, and the parameter");
        Console.WriteLine("update in one iteration point. The oracle records that algebra gap instead");
        Console.WriteLine("of scoring a different operator under the Adam name.");
    }

    private static bool ScoreUnary(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor,
        string kernelName, CodegenActivationKind activation,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer, int> launch)
    {
        const int Count = 1 << 22;
        using var input = backend.AllocateBuffer(Values(Count, 61, 0.03125f));
        using var output = backend.AllocateBuffer(Count);
        var spec = IncumbentSemanticSpecs.Unary(
            kernelName + "_semantic", Count, activation);
        return Score(backend, machine, major, minor, kernelName, spec,
            () => launch(backend, input, output, Count));
    }

    private static bool ScoreBinary(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor,
        string kernelName, bool multiply)
    {
        const int Count = 1 << 22;
        using var left = backend.AllocateBuffer(Values(Count, 67, 0.03125f));
        using var right = backend.AllocateBuffer(Values(Count, 71, 0.015625f));
        using var output = backend.AllocateBuffer(Count);
        CodegenKernelSpec spec = multiply
            ? IncumbentSemanticSpecs.Multiply(kernelName + "_semantic", Count)
            : IncumbentSemanticSpecs.Add(kernelName + "_semantic", Count);
        return Score(backend, machine, major, minor, kernelName, spec,
            () =>
            {
                if (multiply) backend.Multiply(left, right, output, Count);
                else backend.Add(left, right, output, Count);
            });
    }

    private static bool ScoreEmbeddingForward(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor)
    {
        const int Tokens = 1 << 20, Vocabulary = 4096, Width = 64;
        var ids = new int[Tokens];
        // Keep the multiply wide so the benchmark never feeds negative, out-of-contract IDs.
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)((i * 7919L) % Vocabulary);
        var table = Values(Vocabulary * Width, 17, 0.03125f);

        using var idsBuffer = backend.AllocateIntBuffer(ids);
        using var tableBuffer = backend.AllocateBuffer(table);
        using var outputBuffer = backend.AllocateBuffer(Tokens * Width);

        var spec = IncumbentSemanticSpecs.Gather(
            "oracle_inc_embedding", Tokens, Vocabulary, Width);
        return Score(backend, machine, major, minor, "embedding_forward", spec,
            () => backend.Embedding(idsBuffer, tableBuffer, outputBuffer, Tokens, Width));
    }

    private static bool ScoreSgdMomentum(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor)
    {
        const int Count = 1 << 22;
        const float LearningRate = 0.01f, Momentum = 0.9f;
        using var parameter = backend.AllocateBuffer(Values(Count, 11, 0.25f));
        using var gradient = backend.AllocateBuffer(Values(Count, 23, 0.03125f));
        using var velocity = backend.AllocateBuffer(Values(Count, 37, 0.015625f));

        var spec = IncumbentSemanticSpecs.Momentum(
            "oracle_inc_momentum", Count, Momentum, LearningRate);
        return Score(backend, machine, major, minor, "sgd_momentum_update", spec,
            () => backend.SgdMomentumUpdate(
                parameter, gradient, velocity,
                LearningRate, Momentum, weightDecay: 0f, size: Count));
    }

    private static bool ScoreAxis(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor,
        string kernelName, CodegenReduceKind reduce,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer, int, int> launch,
        bool mean = false)
    {
        double reduceScale = mean ? 1.0 / AxisInner : 1.0;
        using var input = backend.AllocateBuffer(
            Values(AxisRows * AxisInner, 53, 0.015625f));
        using var output = backend.AllocateBuffer(AxisRows);
        var spec = IncumbentSemanticSpecs.RowReduction(
            kernelName + "_semantic", AxisRows, AxisInner, reduce, reduceScale);

        return Score(backend, machine, major, minor, kernelName, spec,
            () => launch(backend, input, output, AxisRows, AxisInner));
    }

    private static bool Score(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor,
        string kernelName, CodegenKernelSpec spec, Action launch)
    {
        try
        {
            // Emission is used only to obtain the spec-derived traffic and operation counts.
            // The schedule-dependent penalties are NOT part of the ceiling below.
            var emitter = new PtxAffineEmitter();
            _ = emitter.Emit(spec, major, minor);
            var prediction = CodegenPerformanceModel.Predict(
                spec, spec.Space.TotalThreads, emitter.DynamicLoadsPerThread,
                machine, emitter.LaunchBlockX);

            double? ceiling = prediction.HasComputeCeiling
                ? Math.Max(prediction.DramMicroseconds, prediction.ComputeMicroseconds)
                : null;
            long workUnits = (long)Math.Max(1.0,
                Math.Max(prediction.Macs, prediction.UniqueBytes));
            var timing = StableTimer.MeasureHost(launch, backend.Synchronize, workUnits);

            Console.WriteLine("{0,-32} {1,14} {2,10} {3,9} {4,8}",
                kernelName,
                timing.Describe(),
                ceiling?.ToString("0.0", CultureInfo.InvariantCulture) +
                    (ceiling.HasValue ? " us" : "-"),
                timing.Stable && ceiling is double ceilingValue
                    ? (ceilingValue / timing.Microseconds * 100.0)
                        .ToString("0.0", CultureInfo.InvariantCulture) + "%"
                    : "-",
                prediction.HasComputeCeiling
                    ? prediction.Limiter.ToString().ToLowerInvariant()
                    : "-");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-32} {1}", kernelName, ex.Message.Replace('\n', ' '));
            return false;
        }
    }

    /// <summary>All kernel names exposed by hand-written CUDA source registries and owners.</summary>
    private static Dictionary<string, HashSet<string>> RegisteredHandWrittenKernels()
    {
        var names = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        Assembly assembly = typeof(CudaOptimizerKernels).Assembly;

        foreach (Type type in assembly.GetTypes())
        {
            if (type.Namespace != "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels")
                continue;

            MethodInfo? method = type.GetMethod("GetKernelNames",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                binder: null, types: Type.EmptyTypes, modifiers: null);
            if (method is null || !typeof(IEnumerable<string>).IsAssignableFrom(method.ReturnType))
                continue;

            if (method.Invoke(null, null) is IEnumerable<string> typeNames)
            {
                foreach (string name in typeNames)
                {
                    if (!names.TryGetValue(name, out var owners))
                    {
                        owners = new HashSet<string>(StringComparer.Ordinal);
                        names.Add(name, owners);
                    }
                    owners.Add(type.Name);
                }
            }
        }

        return names;
    }

    private static void WriteCoverageLedger(
        string path, IReadOnlyDictionary<string, HashSet<string>> registered)
    {
        string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        var mapped = new HashSet<string>(MappedKernelNames, StringComparer.Ordinal);
        var lines = new List<string>
        {
            "kernel\towners\tsemantic_status\tsemantic_family\treason"
        };

        foreach (var pair in registered.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            bool isMapped = mapped.Contains(pair.Key);
            lines.Add(string.Join("\t",
                pair.Key,
                string.Join(",", pair.Value.OrderBy(v => v, StringComparer.Ordinal)),
                isMapped ? "mapped-and-timed" : "unmapped",
                isMapped ? MappedFamily(pair.Key) : "-",
                isMapped
                    ? "reviewed equivalent spec and backend launch"
                    : "no reviewed semantic spec plus launch binding; no ceiling assigned"));
        }
        File.WriteAllLines(path, lines);
    }

    private static string MappedFamily(string name) => name switch
    {
        "add_vectors_vec4" or "multiply_vectors_vec4" => "elementwise-binary",
        "relu_vec4" or "sigmoid_vec4" or "tanh_activation_vec4" or "gelu_vec4"
            => "elementwise-activation",
        "embedding_forward" => "gather",
        "sgd_momentum_update" => "optimizer",
        "sum_axis" or "mean_axis" or "max_axis" => "axis-reduction",
        _ => throw new ArgumentOutOfRangeException(
            nameof(name), name, "Mapped kernel has no semantic family."),
    };

    private static float[] Values(int count, int salt, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (((i * 37L + salt) % 97) - 48) * scale;
        return values;
    }

}
