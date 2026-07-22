#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static class DirectPtxAttentionAutotuner
{
    private const string VariantPrefix = "query-warps-";

    internal static int[] Candidates(int sequenceLength) => sequenceLength switch
    {
        16 => [1],
        32 => [2, 1],
        64 => [4, 2],
        128 => [8, 4],
        _ => throw new ArgumentOutOfRangeException(nameof(sequenceLength))
    };

    internal static bool TryLoad(
        DirectPtxRuntime runtime,
        int batchHeads,
        int sequenceLength,
        bool isCausal,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon,
        out int warps)
    {
        KernelChoice? choice = AutotuneCache.Lookup(
            KernelId(runtime), Shape(batchHeads, sequenceLength, isCausal, fused, emitStats, scale, epsilon));
        if (choice?.Variant?.StartsWith(VariantPrefix, StringComparison.Ordinal) == true &&
            int.TryParse(choice.Variant.AsSpan(VariantPrefix.Length), NumberStyles.None,
                CultureInfo.InvariantCulture, out int parsed) &&
            Array.IndexOf(Candidates(sequenceLength), parsed) >= 0)
        {
            warps = parsed;
            return true;
        }
        warps = 0;
        return false;
    }

    internal static void Store(
        DirectPtxRuntime runtime,
        int batchHeads,
        int sequenceLength,
        bool isCausal,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon,
        int warps,
        double milliseconds,
        double tflops)
    {
        try
        {
            AutotuneCache.Store(
                KernelId(runtime),
                Shape(batchHeads, sequenceLength, isCausal, fused, emitStats, scale, epsilon),
                new KernelChoice
                {
                    Variant = VariantPrefix + warps.ToString(CultureInfo.InvariantCulture),
                    Parameters = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["QueryTileRows"] = PtxOnlineFusedAttention128x64Kernel.QueryTileRows.ToString(CultureInfo.InvariantCulture),
                        ["KeyTileRows"] = PtxOnlineFusedAttention128x64Kernel.KeyTileRows.ToString(CultureInfo.InvariantCulture),
                        ["WarpsPerBlock"] = warps.ToString(CultureInfo.InvariantCulture),
                        ["GpuFingerprint"] = runtime.DeviceFingerprint
                    },
                    MeasuredTimeMs = milliseconds,
                    MeasuredGflops = tflops * 1000.0,
                    RecordedAtUtc = DateTime.UtcNow
                });
        }
        catch
        {
            // Persistence is advisory. A read-only home directory must not
            // disable a numerically valid in-memory winner.
        }
    }

    private static KernelId KernelId(DirectPtxRuntime runtime) =>
        new("direct-ptx-sdpa", $"online-attention-v2-{runtime.DeviceFingerprint}");

    private static ShapeProfile Shape(
        int batchHeads,
        int sequenceLength,
        bool isCausal,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon) =>
        new(batchHeads, sequenceLength, PtxOnlineFusedAttention128x64Kernel.HeadDimension,
            isCausal ? 1 : 0, fused ? 1 : 0, emitStats ? 1 : 0,
            BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
}
#endif
