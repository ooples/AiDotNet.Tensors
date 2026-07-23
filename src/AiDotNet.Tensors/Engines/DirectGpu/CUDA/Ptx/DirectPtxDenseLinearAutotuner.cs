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
        double gflops)
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
                                runtime.ComputeCapabilityMinor)
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
            "fused-gelu-fp16-m16-v4-" + runtime.DeviceFingerprint +
            "-candidate-set-" + candidateSet);
    }

    private static ShapeProfile Shape(int inputFeatures, int outputFeatures) =>
        new(PtxFusedLinearGeluFp16M16Kernel.Rows, inputFeatures, outputFeatures,
            (int)DirectPtxPhysicalType.Float16,
            (int)DirectPtxPhysicalType.Float32,
            (int)DirectPtxPhysicalLayout.LinearWeightOutputMajor);
}
