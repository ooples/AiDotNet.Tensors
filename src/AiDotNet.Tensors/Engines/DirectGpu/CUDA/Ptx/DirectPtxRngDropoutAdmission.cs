using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxRngDropoutParameters(
    uint KeepThreshold,
    float InverseKeep);

/// <summary>
/// Allocation-free, fail-closed admission for the first stochastic PTX ABI.
/// Successful admission is the one place that proves exact extents, dense
/// vector layout, alignment, non-aliasing, numerical parameters, and exact SM.
/// </summary>
internal static class DirectPtxRngDropoutAdmission
{
    internal static bool TryValidate(
        IntPtr input,
        long inputBytes,
        IntPtr output,
        long outputBytes,
        IntPtr mask,
        long maskBytes,
        int elementCount,
        float dropoutRate,
        int computeMajor,
        int computeMinor,
        out DirectPtxRngDropoutParameters parameters,
        out string? rejection)
    {
        parameters = default;
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(computeMajor, computeMinor))
            return Reject("rng-dropout-exact-sm-not-supported", out rejection);
        if (!PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elementCount))
            return Reject("rng-dropout-exact-shape-not-supported", out rejection);
        if (!PtxCompat.IsFinite(dropoutRate) || dropoutRate <= 0.0f || dropoutRate >= 1.0f)
            return Reject("rng-dropout-rate-not-supported", out rejection);
        if (input == IntPtr.Zero || output == IntPtr.Zero || mask == IntPtr.Zero)
            return Reject("rng-dropout-invalid-device-pointer", out rejection);

        long requiredBytes;
        try { requiredBytes = checked((long)elementCount * sizeof(float)); }
        catch (OverflowException)
        {
            return Reject("rng-dropout-extent-overflow", out rejection);
        }
        if (inputBytes != requiredBytes || outputBytes != requiredBytes || maskBytes != requiredBytes)
            return Reject("rng-dropout-physical-extent-mismatch", out rejection);
        if (((PtxCompat.ToNuint(input) | PtxCompat.ToNuint(output) | PtxCompat.ToNuint(mask)) & 15u) != 0)
            return Reject("rng-dropout-alignment-mismatch", out rejection);
        try
        {
            if (Overlaps(input, requiredBytes, output, requiredBytes) ||
                Overlaps(input, requiredBytes, mask, requiredBytes) ||
                Overlaps(output, requiredBytes, mask, requiredBytes))
                return Reject("rng-dropout-alias-not-supported", out rejection);
        }
        catch (OverflowException)
        {
            return Reject("rng-dropout-invalid-pointer-range", out rejection);
        }

        float keepProbability = 1.0f - dropoutRate;
        ulong threshold64 = (ulong)Math.Floor((double)keepProbability * 4_294_967_296.0);
        if (threshold64 is 0 or > uint.MaxValue)
            return Reject("rng-dropout-keep-threshold-not-representable", out rejection);
        float inverseKeep = 1.0f / keepProbability;
        if (!PtxCompat.IsFinite(inverseKeep) || inverseKeep <= 1.0f)
            return Reject("rng-dropout-inverse-keep-not-representable", out rejection);

        parameters = new DirectPtxRngDropoutParameters((uint)threshold64, inverseKeep);
        rejection = null;
        return true;
    }

    private static bool Overlaps(IntPtr left, long leftBytes, IntPtr right, long rightBytes)
    {
        nuint leftStart = PtxCompat.ToNuint(left);
        nuint rightStart = PtxCompat.ToNuint(right);
        nuint leftEnd = checked(leftStart + (nuint)leftBytes);
        nuint rightEnd = checked(rightStart + (nuint)rightBytes);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static bool Reject(string reason, out string? rejection)
    {
        rejection = reason;
        return false;
    }
}
