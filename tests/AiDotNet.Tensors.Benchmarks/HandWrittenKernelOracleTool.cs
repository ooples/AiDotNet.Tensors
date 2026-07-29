// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
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
    private static readonly string[] MappedKernelNames =
    {
        "embedding_forward",
        "sgd_momentum_update",
        "sum_axis",
        "mean_axis",
        "max_axis",
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

        var registered = RegisteredHandWrittenKernelNames();
        string[] staleMappings = MappedKernelNames
            .Where(name => !registered.Contains(name))
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

        ScoreEmbeddingForward(backend, machine, major, minor);
        ScoreSgdMomentum(backend, machine, major, minor);
        ScoreAxis(backend, machine, major, minor,
            "sum_axis", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.SumAxis(input, output, rows, inner));
        ScoreAxis(backend, machine, major, minor,
            "mean_axis", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.MeanAxis(input, output, rows, inner),
            reduceScale: 1.0 / 1024.0);
        ScoreAxis(backend, machine, major, minor,
            "max_axis", CodegenReduceKind.Max,
            static (b, input, output, rows, inner) => b.MaxAxis(input, output, rows, inner));

        Console.WriteLine();
        Console.WriteLine("Adam/AdamW are registered incumbents but remain unmapped: the current spec");
        Console.WriteLine("cannot express grad^2, two independently-computed moments, and the parameter");
        Console.WriteLine("update in one iteration point. The oracle records that algebra gap instead");
        Console.WriteLine("of scoring a different operator under the Adam name.");
    }

    private static void ScoreEmbeddingForward(
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
        Score(backend, machine, major, minor, "embedding_forward", spec,
            () => backend.Embedding(idsBuffer, tableBuffer, outputBuffer, Tokens, Width));
    }

    private static void ScoreSgdMomentum(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor)
    {
        const int Count = 1 << 22;
        const float LearningRate = 0.01f, Momentum = 0.9f;
        using var parameter = backend.AllocateBuffer(Values(Count, 11, 0.25f));
        using var gradient = backend.AllocateBuffer(Values(Count, 23, 0.03125f));
        using var velocity = backend.AllocateBuffer(Values(Count, 37, 0.015625f));

        var spec = IncumbentSemanticSpecs.Momentum(
            "oracle_inc_momentum", Count, Momentum, LearningRate);
        Score(backend, machine, major, minor, "sgd_momentum_update", spec,
            () => backend.SgdMomentumUpdate(
                parameter, gradient, velocity,
                LearningRate, Momentum, weightDecay: 0f, size: Count));
    }

    private static void ScoreAxis(
        CudaBackend backend, CodegenMachineModel machine, int major, int minor,
        string kernelName, CodegenReduceKind reduce,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer, int, int> launch,
        double reduceScale = 1.0)
    {
        const int Rows = 4096, Inner = 1024;
        using var input = backend.AllocateBuffer(Values(Rows * Inner, 53, 0.015625f));
        using var output = backend.AllocateBuffer(Rows);
        var spec = IncumbentSemanticSpecs.RowReduction(
            kernelName + "_semantic", Rows, Inner, reduce, reduceScale);

        Score(backend, machine, major, minor, kernelName, spec,
            () => launch(backend, input, output, Rows, Inner));
    }

    private static void Score(
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

            double ceiling = Math.Max(
                prediction.DramMicroseconds, prediction.ComputeMicroseconds);
            long workUnits = (long)Math.Max(1.0,
                Math.Max(prediction.Macs, prediction.UniqueBytes));
            var timing = StableTimer.MeasureHost(launch, backend.Synchronize, workUnits);

            Console.WriteLine("{0,-32} {1,14} {2,10} {3,9} {4,8}",
                kernelName,
                timing.Describe(),
                ceiling.ToString("0.0", CultureInfo.InvariantCulture) + " us",
                timing.Stable
                    ? (ceiling / timing.Microseconds * 100.0)
                        .ToString("0.0", CultureInfo.InvariantCulture) + "%"
                    : "-",
                prediction.ComputeMicroseconds >= prediction.DramMicroseconds
                    ? "compute" : "memory");
        }
        catch (Exception ex)
        {
            Console.WriteLine("{0,-32} {1}", kernelName, ex.Message.Replace('\n', ' '));
        }
    }

    /// <summary>All kernel names exposed by hand-written CUDA source registries.</summary>
    private static HashSet<string> RegisteredHandWrittenKernelNames()
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
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
                foreach (string name in typeNames) names.Add(name);
        }

        return names;
    }

    private static float[] Values(int count, int salt, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (((i * 37L + salt) % 97) - 48) * scale;
        return values;
    }

}
