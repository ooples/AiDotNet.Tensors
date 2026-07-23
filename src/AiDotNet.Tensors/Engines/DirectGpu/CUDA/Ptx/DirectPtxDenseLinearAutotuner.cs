using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Bounded, persistent tuning domain for the exact M=16 FP16 Tensor-Core
/// fused-linear specialization. GPU UUID, SM, and driver are part of the
/// kernel identity; shape and semantics are part of the shape profile.
/// </summary>
internal static class DirectPtxDenseLinearAutotuner
{
    private const string VariantPrefix = "nblock-";
    internal const double NearTieTolerance = 0.03;

    internal static int[] Candidates(int inputFeatures, int outputFeatures)
    {
        if (!PtxFusedLinearGeluFp16M16Kernel.IsSupportedShape(
                inputFeatures, outputFeatures))
            throw new ArgumentOutOfRangeException(nameof(inputFeatures));
        return [64, 32];
    }

    internal static bool TryLoad(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures,
        out int outputsPerBlock)
    {
        KernelChoice? choice = AutotuneCache.Lookup(
            KernelId(runtime, inputFeatures, outputFeatures),
            Shape(inputFeatures, outputFeatures));
        if (choice?.Variant?.StartsWith(VariantPrefix, StringComparison.Ordinal) == true &&
            int.TryParse(choice.Variant.Substring(VariantPrefix.Length), NumberStyles.None,
                CultureInfo.InvariantCulture, out int parsed) &&
            Array.IndexOf(Candidates(inputFeatures, outputFeatures), parsed) >= 0)
        {
            outputsPerBlock = parsed;
            return true;
        }
        outputsPerBlock = 0;
        return false;
    }

    internal static void Store(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures,
        int outputsPerBlock,
        double milliseconds,
        double gflops,
        double n64MedianMilliseconds,
        double n32MedianMilliseconds)
    {
        if (Array.IndexOf(Candidates(inputFeatures, outputFeatures), outputsPerBlock) < 0)
            throw new ArgumentOutOfRangeException(nameof(outputsPerBlock));
        try
        {
            AutotuneCache.Store(
                KernelId(runtime, inputFeatures, outputFeatures),
                Shape(inputFeatures, outputFeatures),
                new KernelChoice
                {
                    Variant = VariantPrefix +
                        outputsPerBlock.ToString(CultureInfo.InvariantCulture),
                    Parameters = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["Rows"] = PtxFusedLinearGeluFp16M16Kernel.Rows.ToString(
                            CultureInfo.InvariantCulture),
                        ["OutputsPerBlock"] = outputsPerBlock.ToString(
                            CultureInfo.InvariantCulture),
                        ["WarpsPerBlock"] = (outputsPerBlock / 8).ToString(
                            CultureInfo.InvariantCulture),
                        ["GpuFingerprint"] = runtime.DeviceFingerprint,
                        ["SelectedCubinSourceKey"] =
                            DirectPtxCubinArtifactCache.ComputeSourceKey(
                                PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                                    runtime.ComputeCapabilityMajor,
                                    runtime.ComputeCapabilityMinor,
                                    inputFeatures, outputFeatures, outputsPerBlock),
                                runtime.ComputeCapabilityMajor,
                                runtime.ComputeCapabilityMinor),
                        ["N64MedianMicroseconds"] = (n64MedianMilliseconds * 1_000d)
                            .ToString("R", CultureInfo.InvariantCulture),
                        ["N32MedianMicroseconds"] = (n32MedianMilliseconds * 1_000d)
                            .ToString("R", CultureInfo.InvariantCulture),
                        ["NearTieTolerancePercent"] = (NearTieTolerance * 100d)
                            .ToString("R", CultureInfo.InvariantCulture)
                    },
                    MeasuredTimeMs = milliseconds,
                    MeasuredGflops = gflops,
                    RecordedAtUtc = DateTime.UtcNow
                });
        }
        catch
        {
            // Persistence is advisory. The in-memory winner remains valid when
            // the cache root is read-only or temporarily unavailable.
        }
    }

    /// <summary>
    /// Selects the measured winner while retaining N=64 when it is within the
    /// bounded noise band of the absolute minimum. That deterministic tie
    /// policy prevents identical hardware from oscillating between cache
    /// entries when launch timings differ only by normal measurement noise.
    /// </summary>
    internal static int SelectWinner(
        double n64MedianMilliseconds,
        double n32MedianMilliseconds)
    {
        ValidateMeasurement(n64MedianMilliseconds, nameof(n64MedianMilliseconds));
        ValidateMeasurement(n32MedianMilliseconds, nameof(n32MedianMilliseconds));
        return n64MedianMilliseconds <= n32MedianMilliseconds * (1d + NearTieTolerance)
            ? 64
            : 32;
    }

    internal static double MedianMilliseconds(float[] samples)
    {
        PtxCompat.ThrowIfNull(samples, nameof(samples));
        if (samples.Length == 0) throw new ArgumentException("At least one sample is required.", nameof(samples));
        var ordered = (float[])samples.Clone();
        Array.Sort(ordered);
        int middle = ordered.Length / 2;
        double median = (ordered.Length & 1) != 0
            ? ordered[middle]
            : ((double)ordered[middle - 1] + ordered[middle]) / 2d;
        ValidateMeasurement(median, nameof(samples));
        return median;
    }

    private static void ValidateMeasurement(double milliseconds, string parameterName)
    {
        if (double.IsNaN(milliseconds) || double.IsInfinity(milliseconds) || milliseconds <= 0d)
            throw new ArgumentOutOfRangeException(parameterName);
    }

    private static KernelId KernelId(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures)
    {
        string n64 = DirectPtxCubinArtifactCache.ComputeSourceKey(
            PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
                inputFeatures, outputFeatures, outputsPerBlock: 64),
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        string n32 = DirectPtxCubinArtifactCache.ComputeSourceKey(
            PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
                inputFeatures, outputFeatures, outputsPerBlock: 32),
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        string candidateSet = DirectPtxCubinArtifactCache.ComputePtxSha256(
            "n64=" + n64 + "\nn32=" + n32 + "\n");
        return new KernelId(
            "direct-ptx-dense-linear",
            "fused-gelu-fp16-m16-v5-tuner-v2-" + runtime.DeviceFingerprint +
            "-candidate-set-" + candidateSet);
    }

    private static ShapeProfile Shape(int inputFeatures, int outputFeatures) =>
        new(PtxFusedLinearGeluFp16M16Kernel.Rows, inputFeatures, outputFeatures,
            (int)DirectPtxPhysicalType.Float16,
            (int)DirectPtxPhysicalType.Float32,
            (int)DirectPtxPhysicalLayout.LinearWeightOutputMajor);
}
