// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Compares every issue #840 direct-PTX softmax-family cell with the CUDA implementation
/// reached when the experiment gate is closed.
/// </summary>
/// <remarks>
/// Both sides launch on the same backend stream and are bracketed by CUDA events from a
/// borrowed <see cref="DirectPtxRuntime"/>. Each direct row also proves that its dispatch
/// counter advanced and compares one result with the incumbent before reporting a ratio.
/// This prevents an ineligible shape from being timed as a false direct-PTX win.
/// </remarks>
internal static class DirectPtxSoftmaxExperiment
{
    private const int Rows = 2048;
    private const int Columns = 1024;
    private const int Count = Rows * Columns;
    private static string[] _operationFilter = Array.Empty<string>();

    internal static void Run(string[] args)
    {
        _operationFilter = args ?? Array.Empty<string>();
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-softmax-start");
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("CUDA backend unavailable; softmax comparison skipped.");
            return;
        }

        using var timer = new DirectPtxRuntime(
            backend.CudaContextHandle, backend.DefaultStream.Handle);
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;

        try
        {
            float[] input = Values(Count, 17, 1.0f / 64.0f);
            float[] gradient = Values(Count, 31, 1.0f / 128.0f);
            float[] probabilities = SoftmaxReference(input);
            float[] lse = LogSumExpReference(input);
            float[] rowGradient = Values(Rows, 43, 1.0f / 32.0f);
            float[] mask = new float[Count];
            for (int i = 0; i < mask.Length; i++) mask[i] = i % 7 == 0 ? 1f : 0f;

            using var inputBuffer = backend.AllocateBuffer(input);
            using var gradientBuffer = backend.AllocateBuffer(gradient);
            using var probabilityBuffer = backend.AllocateBuffer(probabilities);
            using var lseBuffer = backend.AllocateBuffer(lse);
            using var rowGradientBuffer = backend.AllocateBuffer(rowGradient);
            using var maskBuffer = backend.AllocateBuffer(mask);

            Console.WriteLine();
            Console.WriteLine("DIRECT PTX SOFTMAX FAMILY - shipped CUDA incumbent vs issue #840 kernel");
            Console.WriteLine("device: {0}", backend.DeviceName);
            Console.WriteLine("shape: [{0},{1}] FP32; same stream; paired CUDA-event samples", Rows, Columns);
            Console.WriteLine();
            Console.WriteLine("{0,-30} {1,14} {2,14} {3,9} {4,11}  {5}",
                "operator", "incumbent", "direct PTX", "ratio", "max error", "verdict");

            CompareTwoOutput(
                backend, timer, "Softmax", Count, 2e-3,
                inputBuffer,
                static (b, i, o) => b.Softmax(i, o, Rows, Columns),
                () => backend.DirectPtxSoftmaxDispatchCount);
            CompareTwoOutput(
                backend, timer, "SoftmaxRows", Count, 2e-3,
                inputBuffer,
                static (b, i, o) => b.SoftmaxRows(i, o, Rows, Columns),
                () => backend.DirectPtxSoftmaxDispatchCount);
            CompareTwoOutput(
                backend, timer, "SoftmaxBackward", Count, 2e-3,
                probabilityBuffer,
                (b, _, o) => b.SoftmaxBackward(gradientBuffer, probabilityBuffer, o, Rows, Columns),
                () => backend.DirectPtxSoftmaxBackwardDispatchCount);
            CompareTwoOutput(
                backend, timer, "LogSoftmax", Count, 2e-3,
                inputBuffer,
                static (b, i, o) => b.LogSoftmax(i, o, Rows, Columns),
                () => backend.DirectPtxLogSoftmaxDispatchCount);
            CompareReduction(
                backend, timer, "LogSumExpAxis", Count, 2e-3,
                inputBuffer,
                static (b, i, o) => b.LogSumExpAxis(i, o, Rows, Columns),
                () => backend.DirectPtxLogSumExpDispatchCount);
            CompareTwoOutput(
                backend, timer, "LogSumExpBackward", Count, 2e-3,
                inputBuffer,
                (b, _, o) => b.LogSumExpBackward(
                    rowGradientBuffer, inputBuffer, lseBuffer, o, Rows, Columns),
                () => backend.DirectPtxLogSumExpBackwardDispatchCount);
            CompareTwoOutput(
                backend, timer, "MaskedFill", Count, 0.0,
                inputBuffer,
                (b, i, o) => b.MaskedFillKernel(i, maskBuffer, o, -10_000f, Count),
                () => backend.DirectPtxMaskedFillDispatchCount);
            CompareTwoOutput(
                backend, timer, "MaskedFillBackward", Count, 0.0,
                gradientBuffer,
                (b, i, o) => b.MaskedFillBackward(i, maskBuffer, o, Count),
                () => backend.DirectPtxMaskedFillBackwardDispatchCount);
            CompareTwoOutput(
                backend, timer, "Sparsemax", Count, 2e-3,
                inputBuffer,
                static (b, i, o) => b.Sparsemax(i, o, Rows, Columns),
                () => backend.DirectPtxSparsemaxDispatchCount,
                workMultiplier: PtxSparsemaxKernel.BisectionSteps);
            CompareTwoOutput(
                backend, timer, "TaylorSoftmax", Count, 2e-5,
                inputBuffer,
                static (b, i, o) => b.TaylorSoftmax(i, o, Rows, Columns),
                () => backend.DirectPtxTaylorSoftmaxDispatchCount);

            Console.WriteLine();
            Console.WriteLine("ratio > 1 means direct PTX is faster. A win must clear the measured");
            Console.WriteLine("paired-ratio spread; unstable rows and correctness mismatches are withheld.");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
            _operationFilter = Array.Empty<string>();
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-softmax-end", afterSuite: true);
    }

    private static void CompareTwoOutput(
        CudaBackend backend,
        DirectPtxRuntime timer,
        string label,
        int outputCount,
        double tolerance,
        IGpuBuffer input,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer> launch,
        Func<long> dispatchCount,
        int workMultiplier = 1)
    {
        if (!ShouldRun(label)) return;
        using var incumbentOutput = backend.AllocateBuffer(outputCount);
        using var directOutput = backend.AllocateBuffer(outputCount);
        Compare(
            backend, timer, label, outputCount, tolerance,
            () => launch(backend, input, incumbentOutput), incumbentOutput,
            () => launch(backend, input, directOutput), directOutput,
            dispatchCount, workMultiplier);
    }

    private static void CompareReduction(
        CudaBackend backend,
        DirectPtxRuntime timer,
        string label,
        int inputCount,
        double tolerance,
        IGpuBuffer input,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer> launch,
        Func<long> dispatchCount)
    {
        if (!ShouldRun(label)) return;
        using var incumbentOutput = backend.AllocateBuffer(Rows);
        using var directOutput = backend.AllocateBuffer(Rows);
        Compare(
            backend, timer, label, inputCount, tolerance,
            () => launch(backend, input, incumbentOutput), incumbentOutput,
            () => launch(backend, input, directOutput), directOutput,
            dispatchCount, workMultiplier: 2);
    }

    private static void Compare(
        CudaBackend backend,
        DirectPtxRuntime timer,
        string label,
        int workElements,
        double tolerance,
        Action launchIncumbent,
        IGpuBuffer incumbentOutput,
        Action launchDirect,
        IGpuBuffer directOutput,
        Func<long> dispatchCount,
        int workMultiplier)
    {
        SetIncumbent();
        launchIncumbent();
        backend.Synchronize();
        float[] expected = backend.DownloadBuffer(incumbentOutput);

        long before = dispatchCount();
        SetDirect();
        launchDirect();
        backend.Synchronize();
        if (dispatchCount() <= before)
            throw new InvalidOperationException(
                $"{label} did not dispatch direct PTX: {backend.DirectPtxLastError}");
        float[] actual = backend.DownloadBuffer(directOutput);
        double error = MaxAbsoluteError(expected, actual);

        long workUnits = checked((long)workElements * sizeof(float) * workMultiplier);
        StableTimer.PairResult timing = StableTimer.MeasurePair(
            timer,
            () => { SetIncumbent(); launchIncumbent(); },
            () => { SetDirect(); launchDirect(); },
            workUnits,
            workUnits);

        string verdict;
        if (error > tolerance)
        {
            verdict = "NOT EQUIVALENT -- withhold";
        }
        else if (!timing.Stable)
        {
            verdict = "NOT MEASURABLE";
        }
        else
        {
            double floor = 1.0 + timing.RelativeSpread;
            verdict = timing.Ratio > floor ? "DIRECT WINS"
                : timing.Ratio < 1.0 / floor ? "incumbent wins -- diagnose"
                : "TIE within noise -- diagnose";
        }

        Console.WriteLine("{0,-30} {1,14} {2,14} {3,9} {4,11}  {5}",
            label,
            timing.A.Describe(),
            timing.B.Describe(),
            timing.DescribeRatio(),
            error.ToString("0.00E+00", CultureInfo.InvariantCulture),
            verdict);
    }

    private static void SetIncumbent()
    {
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        DirectPtxFeatureGate.TestOverride = false;
    }

    private static bool ShouldRun(string label)
    {
        if (_operationFilter.Length == 0) return true;
        foreach (string requested in _operationFilter)
            if (string.Equals(requested, label, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    private static void SetDirect()
    {
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
    }

    private static float[] Values(int count, int salt, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (((i * 37L + salt) % 97) - 48) * scale;
        return values;
    }

    private static float[] SoftmaxReference(float[] input)
    {
        var output = new float[input.Length];
        for (int row = 0; row < Rows; row++)
        {
            int start = row * Columns;
            double max = double.NegativeInfinity;
            for (int column = 0; column < Columns; column++)
                max = Math.Max(max, input[start + column]);
            double sum = 0;
            for (int column = 0; column < Columns; column++)
                sum += Math.Exp(input[start + column] - max);
            for (int column = 0; column < Columns; column++)
                output[start + column] = (float)(Math.Exp(input[start + column] - max) / sum);
        }
        return output;
    }

    private static float[] LogSumExpReference(float[] input)
    {
        var output = new float[Rows];
        for (int row = 0; row < Rows; row++)
        {
            int start = row * Columns;
            double max = double.NegativeInfinity;
            for (int column = 0; column < Columns; column++)
                max = Math.Max(max, input[start + column]);
            double sum = 0;
            for (int column = 0; column < Columns; column++)
                sum += Math.Exp(input[start + column] - max);
            output[row] = (float)(max + Math.Log(sum));
        }
        return output;
    }

    private static double MaxAbsoluteError(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length) return double.PositiveInfinity;
        double max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]))
                return double.PositiveInfinity;
            max = Math.Max(max, Math.Abs((double)expected[i] - actual[i]));
        }
        return max;
    }
}
