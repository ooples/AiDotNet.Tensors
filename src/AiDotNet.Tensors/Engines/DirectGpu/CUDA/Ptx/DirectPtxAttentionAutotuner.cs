using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static class DirectPtxAttentionAutotuner
{
    internal const string Category = "direct-ptx-sdpa";
    internal const string ShareableKernelName = "online-attention-v3";
    private const string VariantPrefix = "query-warps-";

    private static bool TrySupportedWarps(int sequenceLength, out int[] warps)
    {
        switch (sequenceLength)
        {
            case 16: warps = [1]; return true;
            case 32: warps = [2, 1]; return true;
            case 64: warps = [4, 2]; return true;
            case 128: warps = [8, 4]; return true;
            default: warps = []; return false;
        }
    }

    internal static int[] Candidates(int sequenceLength) =>
        TrySupportedWarps(sequenceLength, out int[] warps)
            ? warps
            : throw new ArgumentOutOfRangeException(nameof(sequenceLength));

    internal static IReadOnlyList<AutotuneCandidate> CandidateConfigurations(int sequenceLength)
    {
        int[] warps = Candidates(sequenceLength);
        var result = new AutotuneCandidate[warps.Length];
        for (int i = 0; i < warps.Length; i++) result[i] = CandidateFor(warps[i]);
        return result;
    }

    internal static bool TryGetWarps(
        AutotuneCandidate candidate, int sequenceLength, out int warps)
    {
        warps = 0;
        if (string.IsNullOrEmpty(candidate.Variant) ||
            !TrySupportedWarps(sequenceLength, out int[] supportedWarps) ||
            !candidate.Variant.StartsWith(VariantPrefix, StringComparison.Ordinal) ||
            !int.TryParse(candidate.Variant.Substring(VariantPrefix.Length), NumberStyles.None,
                CultureInfo.InvariantCulture, out int parsed) ||
            Array.IndexOf(supportedWarps, parsed) < 0)
            return false;
        warps = parsed;
        return true;
    }

    internal static int Resolve(
        DirectPtxRuntime runtime,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        int causalQueryOffset,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon,
        Func<int, double> benchmark,
        bool autotuneEnabled,
        IGpuTuningExchange? exchange = null)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (benchmark is null) throw new ArgumentNullException(nameof(benchmark));

        ShapeProfile shape = Shape(
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            isCausal, causalQueryOffset, fused, emitStats, scale, epsilon);
        MigrateLegacyCache(runtime, shape, querySequence);

        AutotuneResolution resolution = CommunityAutotune.Resolve(
            exchange ?? GpuTuningExchangeProvider.Current,
            Category,
            ShareableKernelName,
            runtime.Fingerprint,
            shape,
            CandidateConfigurations(querySequence),
            candidate =>
            {
                if (!TryGetWarps(candidate, querySequence, out int candidateWarps))
                    throw new InvalidOperationException(
                        $"Unsupported attention autotune candidate '{candidate.Variant}'.");
                return benchmark(candidateWarps);
            },
            autotuneEnabled);

        var selected = new AutotuneCandidate(resolution.Variant, resolution.Parameters);
        if (!TryGetWarps(selected, querySequence, out int warps))
            throw new InvalidOperationException(
                $"Attention autotune resolved unsupported variant '{resolution.Variant}'.");
        return warps;
    }

    internal static bool TryLoad(
        DirectPtxRuntime runtime,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        int causalQueryOffset,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon,
        out int warps)
    {
        ShapeProfile shape = Shape(
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            isCausal, causalQueryOffset, fused, emitStats, scale, epsilon);
        KernelId currentId = KernelId(runtime);
        KernelChoice? current = AutotuneCache.Lookup(currentId, shape);
        if (TryGetWarps(current, querySequence, out warps)) return true;

        KernelChoice? legacy = AutotuneCache.Lookup(LegacyKernelId(runtime), shape);
        if (!TryGetWarps(legacy, querySequence, out warps)) return false;

        // Preserve the old attention cache across the shared-autotuner key
        // migration. Persistence is advisory, so a read-only cache is harmless.
        AutotuneCache.TryStore(currentId, shape, legacy!);
        return true;
    }

    private static void MigrateLegacyCache(
        DirectPtxRuntime runtime, ShapeProfile shape, int querySequence)
    {
        if (AutotuneCache.Lookup(KernelId(runtime), shape) is not null) return;
        KernelChoice? legacy = AutotuneCache.Lookup(LegacyKernelId(runtime), shape);
        if (TryGetWarps(legacy, querySequence, out _))
            AutotuneCache.TryStore(KernelId(runtime), shape, legacy!);
    }

    private static bool TryGetWarps(
        KernelChoice? choice, int querySequence, out int warps)
    {
        warps = 0;
        if (choice is null) return false;
        return TryGetWarps(
            new AutotuneCandidate(choice.Variant, choice.Parameters), querySequence, out warps);
    }

    private static AutotuneCandidate CandidateFor(int warps)
    {
        string value = warps.ToString(CultureInfo.InvariantCulture);
        return new AutotuneCandidate(
            VariantPrefix + value,
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["QueryTileRows"] = PtxOnlineFusedAttention128x64Kernel.QueryTileRows
                    .ToString(CultureInfo.InvariantCulture),
                ["KeyTileRows"] = PtxOnlineFusedAttention128x64Kernel.KeyTileRows
                    .ToString(CultureInfo.InvariantCulture),
                ["WarpsPerBlock"] = value
            });
    }

    private static KernelId KernelId(DirectPtxRuntime runtime) =>
        GpuFirstRunAutotuner.GpuKernelId(Category, ShareableKernelName, runtime.Fingerprint);

    private static KernelId LegacyKernelId(DirectPtxRuntime runtime) =>
        new(Category, $"{ShareableKernelName}-{runtime.DeviceFingerprint}");

    private static ShapeProfile Shape(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        int causalQueryOffset,
        bool fused,
        bool emitStats,
        float scale,
        float epsilon) =>
        new(batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            PtxOnlineFusedAttention128x64Kernel.HeadDimension,
            isCausal ? 1 : 0, causalQueryOffset,
            fused ? 1 : 0, emitStats ? 1 : 0,
            PtxCompat.SingleToInt32Bits(scale), PtxCompat.SingleToInt32Bits(epsilon));
}
