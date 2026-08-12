// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Measures equivalent softmax-family launch geometries instead of selecting one from a
/// hand-written assumption. Every candidate is checked against the shipped CUDA result before
/// it is timed, and unstable candidates cannot become winners.
/// </summary>
internal static class DirectPtxSoftmaxAutotuneTool
{
    private const int Rows = 2048;
    private const int Columns = 1024;
    private const int Count = Rows * Columns;

    internal static void Run(string[] args)
    {
        string operation = args.Length == 0 ? "LogSumExpBackward" : args[0];
        bool tuneLogSumExpBackward = string.Equals(
            operation, "LogSumExpBackward", StringComparison.OrdinalIgnoreCase);
        bool tuneSoftmax = string.Equals(operation, "Softmax", StringComparison.OrdinalIgnoreCase);
        bool tuneMaskedFill = string.Equals(operation, "MaskedFill", StringComparison.OrdinalIgnoreCase);
        bool tuneMaskedFillBackward = string.Equals(
            operation, "MaskedFillBackward", StringComparison.OrdinalIgnoreCase);
        bool tuneMaskedSoftmax = string.Equals(
            operation, "MaskedSoftmax", StringComparison.OrdinalIgnoreCase);
        bool tuneMaskedSoftmaxBackward = string.Equals(
            operation, "MaskedSoftmaxBackward", StringComparison.OrdinalIgnoreCase);
        string? candidateFilter = args.Length > 1 ? args[1] : null;
        if (!tuneLogSumExpBackward && !tuneSoftmax &&
            !tuneMaskedFill && !tuneMaskedFillBackward &&
            !tuneMaskedSoftmax && !tuneMaskedSoftmaxBackward)
            throw new ArgumentException(
                "The softmax autotune search supports Softmax, LogSumExpBackward, " +
                "MaskedFill, MaskedFillBackward, MaskedSoftmax, and MaskedSoftmaxBackward.",
                nameof(args));

        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-softmax-autotune-start");
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("CUDA backend unavailable; softmax autotune skipped.");
            return;
        }

        using var runtime = new DirectPtxRuntime(
            backend.CudaContextHandle, backend.DefaultStream.Handle);
        float[] input = Values(Count, 17, 1.0f / 64.0f);
        float[] gradient = Values(Count, 31, 1.0f / 128.0f);
        float[] lse = LogSumExpReference(input);
        float[] rowGradient = Values(Rows, 43, 1.0f / 32.0f);
        var mask = new float[Count];
        for (int i = 0; i < mask.Length; i++) mask[i] = i % 7 == 0 ? 1f : 0f;
        float[] maskedProbabilities = MaskedSoftmaxReference(input, mask, -10_000f);
        using var inputBuffer = backend.AllocateBuffer(input);
        using var gradientBuffer = backend.AllocateBuffer(gradient);
        using var lseBuffer = backend.AllocateBuffer(lse);
        using var rowGradientBuffer = backend.AllocateBuffer(rowGradient);
        using var maskBuffer = backend.AllocateBuffer(mask);
        using var maskedProbabilityBuffer = backend.AllocateBuffer(maskedProbabilities);
        using var incumbentOutput = backend.AllocateBuffer(Count);
        using var incumbentIntermediate = backend.AllocateBuffer(Count);

        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        try
        {
            SetIncumbent();
            if (tuneLogSumExpBackward)
                backend.LogSumExpBackward(
                    rowGradientBuffer, inputBuffer, lseBuffer, incumbentOutput, Rows, Columns);
            else if (tuneMaskedFill)
                backend.MaskedFillKernel(inputBuffer, maskBuffer, incumbentOutput, -10_000f, Count);
            else if (tuneMaskedFillBackward)
                backend.MaskedFillBackward(gradientBuffer, maskBuffer, incumbentOutput, Count);
            else if (tuneMaskedSoftmax)
            {
                backend.MaskedFillKernel(
                    inputBuffer, maskBuffer, incumbentIntermediate, -10_000f, Count);
                backend.Softmax(incumbentIntermediate, incumbentOutput, Rows, Columns);
            }
            else if (tuneMaskedSoftmaxBackward)
            {
                backend.SoftmaxBackward(
                    gradientBuffer, maskedProbabilityBuffer, incumbentIntermediate,
                    Rows, Columns);
                backend.MaskedFillBackward(
                    incumbentIntermediate, maskBuffer, incumbentOutput, Count);
            }
            else
                backend.Softmax(inputBuffer, incumbentOutput, Rows, Columns);
            backend.Synchronize();
            float[] expected = backend.DownloadBuffer(incumbentOutput);

            Console.WriteLine();
            Console.WriteLine("DIRECT PTX SOFTMAX AUTOTUNE - correctness-gated measured variants");
            Console.WriteLine("device: {0}; shape: [{1},{2}] FP32; protocol: {3}",
                backend.DeviceName, Rows, Columns, CodegenMeasurementProtocol.Tag);
            Console.WriteLine();
            Console.WriteLine("{0,-24} {1,14} {2,14} {3,9} {4,11} {5,6} {6,7}  {7}",
                "candidate", "incumbent", "candidate", "ratio", "max error", "regs", "blocks", "verdict");

            if (tuneLogSumExpBackward)
            {
                foreach (PtxLogSumExpBackwardVariant variant in
                    PtxLogSumExpBackwardVariant.SearchSpace(Columns))
                {
                    if (!MatchesCandidate(candidateFilter, variant.Name)) continue;
                    try
                    {
                        MeasureLogSumExpBackward(
                            backend, runtime, variant, inputBuffer, lseBuffer, rowGradientBuffer,
                            incumbentOutput, expected);
                    }
                    catch (Exception ex)
                    {
                        ReportError(variant.Name, ex);
                    }
                }
            }
            else if (tuneSoftmax)
            {
                foreach (PtxSoftmaxVariant variant in PtxSoftmaxVariant.SearchSpace(Columns))
                {
                    if (!MatchesCandidate(candidateFilter, variant.Name)) continue;
                    try
                    {
                        MeasureSoftmax(
                            backend, runtime, variant, inputBuffer, incumbentOutput, expected);
                    }
                    catch (Exception ex)
                    {
                        ReportError(variant.Name, ex);
                    }
                }
            }
            else if (tuneMaskedFill || tuneMaskedFillBackward)
            {
                bool backward = tuneMaskedFillBackward;
                foreach (PtxElementwiseVariant variant in PtxElementwiseVariant.SearchSpace(backward))
                {
                    if (!MatchesCandidate(candidateFilter, variant.Name)) continue;
                    try
                    {
                        MeasureMaskedFill(
                            backend, runtime, variant, backward,
                            backward ? gradientBuffer : inputBuffer, maskBuffer,
                            incumbentOutput, expected);
                    }
                    catch (Exception ex)
                    {
                        ReportError(variant.Name, ex);
                    }
                }
            }
            else if (tuneMaskedSoftmax)
            {
                if (MatchesCandidate(candidateFilter, "t64-v4x4-register"))
                {
                    try
                    {
                        MeasureMaskedSoftmax(
                            backend, runtime, inputBuffer, maskBuffer, incumbentIntermediate,
                            incumbentOutput, expected);
                    }
                    catch (Exception ex)
                    {
                        ReportError("t64-v4x4-register", ex);
                    }
                }
            }
            else
            {
                if (MatchesCandidate(candidateFilter, "t256-shared-probability"))
                {
                    try
                    {
                        MeasureMaskedSoftmaxBackward(
                            backend, runtime, gradientBuffer, maskedProbabilityBuffer, maskBuffer,
                            incumbentIntermediate, incumbentOutput, expected);
                    }
                    catch (Exception ex)
                    {
                        ReportError("t256-shared-probability", ex);
                    }
                }
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute(
            "direct-ptx-softmax-autotune-end", afterSuite: true);
    }

    private static void MeasureMaskedSoftmaxBackward(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        IGpuBuffer gradient,
        IGpuBuffer softmax,
        IGpuBuffer mask,
        IGpuBuffer incumbentIntermediate,
        IGpuBuffer incumbentOutput,
        float[] expected)
    {
        using var candidateOutput = backend.AllocateBuffer(Count);
        using var kernel = new PtxMaskedSoftmaxBackwardKernel(runtime, Rows, Columns);
        DirectPtxTensorView softmaxView = DirectPtxTensorView.Create(
            softmax, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView gradientView = DirectPtxTensorView.Create(
            gradient, kernel.Blueprint.Tensors[1]);
        DirectPtxTensorView maskView = DirectPtxTensorView.Create(
            mask, kernel.Blueprint.Tensors[2]);
        DirectPtxTensorView outputView = DirectPtxTensorView.Create(
            candidateOutput, kernel.Blueprint.Tensors[3]);

        void LaunchIncumbent()
        {
            SetIncumbent();
            backend.SoftmaxBackward(
                gradient, softmax, incumbentIntermediate, Rows, Columns);
            backend.MaskedFillBackward(
                incumbentIntermediate, mask, incumbentOutput, Count);
        }

        void LaunchCandidate() =>
            kernel.Launch(softmaxView, gradientView, maskView, outputView);
        MeasureAndReport(
            backend, runtime, "t256-shared-probability", kernel.Audit,
            expected, candidateOutput, LaunchIncumbent, LaunchCandidate, tolerance: 2e-3);
    }

    private static void MeasureMaskedSoftmax(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        IGpuBuffer input,
        IGpuBuffer mask,
        IGpuBuffer incumbentIntermediate,
        IGpuBuffer incumbentOutput,
        float[] expected)
    {
        using var candidateOutput = backend.AllocateBuffer(Count);
        using var kernel = new PtxMaskedSoftmaxKernel(runtime, Rows, Columns, -10_000f);
        DirectPtxTensorView inputView = DirectPtxTensorView.Create(
            input, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView maskView = DirectPtxTensorView.Create(
            mask, kernel.Blueprint.Tensors[1]);
        DirectPtxTensorView outputView = DirectPtxTensorView.Create(
            candidateOutput, kernel.Blueprint.Tensors[2]);

        void LaunchIncumbent()
        {
            SetIncumbent();
            backend.MaskedFillKernel(
                input, mask, incumbentIntermediate, -10_000f, Count);
            backend.Softmax(incumbentIntermediate, incumbentOutput, Rows, Columns);
        }

        void LaunchCandidate() => kernel.Launch(inputView, maskView, outputView);
        MeasureAndReport(
            backend, runtime, "t64-v4x4-register", kernel.Audit, expected, candidateOutput,
            LaunchIncumbent, LaunchCandidate, tolerance: 2e-3);
    }

    private static void MeasureMaskedFill(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        PtxElementwiseVariant variant,
        bool backward,
        IGpuBuffer input,
        IGpuBuffer mask,
        IGpuBuffer incumbentOutput,
        float[] expected)
    {
        using var candidateOutput = backend.AllocateBuffer(Count);
        if (backward)
        {
            using var kernel = new PtxMaskedFillBackwardKernel(runtime, Count, variant);
            DirectPtxTensorView inputView = DirectPtxTensorView.Create(
                input, kernel.Blueprint.Tensors[0]);
            DirectPtxTensorView maskView = DirectPtxTensorView.Create(
                mask, kernel.Blueprint.Tensors[1]);
            DirectPtxTensorView outputView = DirectPtxTensorView.Create(
                candidateOutput, kernel.Blueprint.Tensors[2]);

            void LaunchIncumbent()
            {
                SetIncumbent();
                backend.MaskedFillBackward(input, mask, incumbentOutput, Count);
            }

            void LaunchCandidate() => kernel.Launch(inputView, maskView, outputView);
            MeasureAndReport(
                backend, runtime, variant.Name, kernel.Audit, expected, candidateOutput,
                LaunchIncumbent, LaunchCandidate, tolerance: 0.0);
        }
        else
        {
            using var kernel = new PtxMaskedFillKernel(runtime, Count, variant);
            DirectPtxTensorView inputView = DirectPtxTensorView.Create(
                input, kernel.Blueprint.Tensors[0]);
            DirectPtxTensorView maskView = DirectPtxTensorView.Create(
                mask, kernel.Blueprint.Tensors[1]);
            DirectPtxTensorView outputView = DirectPtxTensorView.Create(
                candidateOutput, kernel.Blueprint.Tensors[2]);

            void LaunchIncumbent()
            {
                SetIncumbent();
                backend.MaskedFillKernel(input, mask, incumbentOutput, -10_000f, Count);
            }

            void LaunchCandidate() => kernel.Launch(
                inputView, maskView, outputView, -10_000f);
            MeasureAndReport(
                backend, runtime, variant.Name, kernel.Audit, expected, candidateOutput,
                LaunchIncumbent, LaunchCandidate, tolerance: 0.0);
        }
    }

    private static void MeasureAndReport(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        string name,
        DirectPtxKernelAudit audit,
        float[] expected,
        IGpuBuffer candidateOutput,
        Action launchIncumbent,
        Action launchCandidate,
        double tolerance)
    {
        launchCandidate();
        backend.Synchronize();
        double error = MaxAbsoluteError(expected, backend.DownloadBuffer(candidateOutput));
        long workUnits = (long)Count * sizeof(float) * 3;
        StableTimer.PairResult timing = StableTimer.MeasurePair(
            runtime, launchIncumbent, launchCandidate, workUnits, workUnits);
        Report(name, audit, error, timing, tolerance);
    }

    private static void MeasureSoftmax(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        PtxSoftmaxVariant variant,
        IGpuBuffer input,
        IGpuBuffer incumbentOutput,
        float[] expected)
    {
        using var candidateOutput = backend.AllocateBuffer(Count);
        using var kernel = new PtxSoftmaxKernel(runtime, Rows, Columns, variant);
        DirectPtxTensorView inputView = DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView outputView = DirectPtxTensorView.Create(
            candidateOutput, kernel.Blueprint.Tensors[1]);

        void LaunchIncumbent()
        {
            SetIncumbent();
            backend.Softmax(input, incumbentOutput, Rows, Columns);
        }

        void LaunchCandidate() => kernel.Launch(inputView, outputView);

        LaunchCandidate();
        backend.Synchronize();
        double error = MaxAbsoluteError(expected, backend.DownloadBuffer(candidateOutput));
        long workUnits = (long)Count * sizeof(float) * 4;
        StableTimer.PairResult timing = StableTimer.MeasurePair(
            runtime, LaunchIncumbent, LaunchCandidate, workUnits, workUnits);
        Report(variant.Name, kernel.Audit, error, timing);
    }

    private static void MeasureLogSumExpBackward(
        CudaBackend backend,
        DirectPtxRuntime runtime,
        PtxLogSumExpBackwardVariant variant,
        IGpuBuffer input,
        IGpuBuffer lse,
        IGpuBuffer rowGradient,
        IGpuBuffer incumbentOutput,
        float[] expected)
    {
        using var candidateOutput = backend.AllocateBuffer(Count);
        using var kernel = new PtxLogSumExpBackwardKernel(runtime, Rows, Columns, variant);
        DirectPtxTensorView inputView = DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView lseView = DirectPtxTensorView.Create(lse, kernel.Blueprint.Tensors[1]);
        DirectPtxTensorView gradientView = DirectPtxTensorView.Create(
            rowGradient, kernel.Blueprint.Tensors[2]);
        DirectPtxTensorView outputView = DirectPtxTensorView.Create(
            candidateOutput, kernel.Blueprint.Tensors[3]);

        void LaunchIncumbent()
        {
            SetIncumbent();
            backend.LogSumExpBackward(
                rowGradient, input, lse, incumbentOutput, Rows, Columns);
        }

        void LaunchCandidate() => kernel.Launch(inputView, lseView, gradientView, outputView);

        LaunchCandidate();
        backend.Synchronize();
        double error = MaxAbsoluteError(expected, backend.DownloadBuffer(candidateOutput));
        long workUnits = (long)Count * sizeof(float) * 2;
        StableTimer.PairResult timing = StableTimer.MeasurePair(
            runtime, LaunchIncumbent, LaunchCandidate, workUnits, workUnits);

        Report(variant.Name, kernel.Audit, error, timing);
    }

    private static void Report(
        string name, DirectPtxKernelAudit audit, double error, StableTimer.PairResult timing,
        double tolerance = 2e-3)
    {
        double floor = Math.Max(
            DirectPtxReleaseGatePolicy.ProductionDefault.MinimumMedianSpeedup,
            1.0 + timing.RelativeSpread);
        string verdict = error > tolerance ? "NOT EQUIVALENT"
            : !timing.Stable ? "NOT MEASURABLE"
            : timing.Ratio >= floor ? "WIN -- release evidence pending"
            : "reject";

        Console.WriteLine("{0,-24} {1,14} {2,14} {3,9} {4,11} {5,6} {6,7}  {7}",
            name,
            timing.A.Describe(),
            timing.B.Describe(),
            timing.DescribeRatio(),
            error.ToString("0.00E+00", CultureInfo.InvariantCulture),
            audit.Function.RegistersPerThread,
            audit.ActiveBlocksPerMultiprocessor,
            verdict);
    }

    private static void ReportError(string name, Exception ex) =>
        Console.WriteLine("{0,-24} ERROR {1}", name, ex.Message.Replace('\n', ' '));

    private static bool MatchesCandidate(string? requested, string candidate) =>
        string.IsNullOrWhiteSpace(requested) ||
        string.Equals(requested, candidate, StringComparison.OrdinalIgnoreCase);

    private static void SetIncumbent()
    {
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        DirectPtxFeatureGate.TestOverride = false;
    }

    private static float[] Values(int count, int salt, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (((i * 37L + salt) % 97) - 48) * scale;
        return values;
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

    private static float[] MaskedSoftmaxReference(
        float[] input, float[] mask, float fillValue)
    {
        var output = new float[input.Length];
        for (int row = 0; row < Rows; row++)
        {
            int start = row * Columns;
            double max = double.NegativeInfinity;
            for (int column = 0; column < Columns; column++)
            {
                double selected = mask[start + column] != 0f
                    ? fillValue : input[start + column];
                max = Math.Max(max, selected);
            }
            double sum = 0;
            for (int column = 0; column < Columns; column++)
            {
                double selected = mask[start + column] != 0f
                    ? fillValue : input[start + column];
                sum += Math.Exp(selected - max);
            }
            for (int column = 0; column < Columns; column++)
            {
                double selected = mask[start + column] != 0f
                    ? fillValue : input[start + column];
                output[start + column] = (float)(Math.Exp(selected - max) / sum);
            }
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
