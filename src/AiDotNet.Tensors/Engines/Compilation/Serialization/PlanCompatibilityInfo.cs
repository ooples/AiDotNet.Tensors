using Autotune = AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Version stamp embedded in every serialized plan. Fully qualifies the
/// environment the plan was compiled under so the loader can detect mismatches
/// and force a recompile rather than silently replaying a plan that would
/// produce wrong results.
/// </summary>
/// <remarks>
/// <para><b>Format version</b> tracks breaking layout changes in the binary
/// wire format. A format-version mismatch means the byte stream is
/// structurally unreadable.</para>
/// <para><b>Tensor-codec version</b> tracks semantic changes in the compiler
/// and optimization passes. Same format layout, but the optimized plan may
/// differ (different fusion decisions, different buffer shapes).</para>
/// <para><b>Hardware fingerprint</b> captures CPU vendor + architecture +
/// SIMD level + logical core count. Plans compiled on one CPU may rely on
/// SIMD paths (AVX2 vs SSE2) that don't exist on another. The loader
/// rejects cross-machine plans with a descriptive null return.</para>
/// </remarks>
public sealed class PlanCompatibilityInfo
{
    /// <summary>The binary wire-format version.</summary>
    public int FormatVersion { get; init; }

    /// <summary>The tensor-codec / compiler version.</summary>
    public int TensorCodecVersion { get; init; }

    /// <summary>Hardware fingerprint string (e.g. "x64-intel-avx2-cpu16").</summary>
    public string HardwareFingerprint { get; init; } = "";

    /// <summary>
    /// Element type name (e.g. "System.Single", "System.Double").
    /// </summary>
    public string ElementTypeName { get; init; } = "";

    /// <summary>
    /// Returns the compatibility info for the current process / runtime.
    /// </summary>
    public static PlanCompatibilityInfo Current<T>() => new()
    {
        FormatVersion = PlanFormatConstants.CurrentFormatVersion,
        TensorCodecVersion = PlanFormatConstants.TensorCodecVersion,
        HardwareFingerprint = Autotune.HardwareFingerprint.Current,
        ElementTypeName = typeof(T).FullName ?? typeof(T).Name,
    };

    /// <summary>
    /// Checks whether a loaded plan's compatibility info matches the
    /// current runtime. Returns null if compatible, or a human-readable
    /// reason string if not.
    /// </summary>
    public string? GetIncompatibilityReason<T>()
    {
        var current = Current<T>();

        if (FormatVersion != current.FormatVersion)
            return $"Format version mismatch: file has v{FormatVersion}, " +
                   $"this runtime expects v{current.FormatVersion}.";

        if (TensorCodecVersion != current.TensorCodecVersion)
            return $"Tensor-codec version mismatch: file has v{TensorCodecVersion}, " +
                   $"this runtime expects v{current.TensorCodecVersion}. " +
                   "The optimizer may produce different plans — recompile.";

        if (!string.Equals(HardwareFingerprint, current.HardwareFingerprint, StringComparison.Ordinal))
            return $"Hardware fingerprint mismatch: file was compiled on '{HardwareFingerprint}', " +
                   $"current machine is '{current.HardwareFingerprint}'. " +
                   "Cross-machine plan loading is not supported — recompile on this hardware.";

        if (!string.Equals(ElementTypeName, current.ElementTypeName, StringComparison.Ordinal))
            return $"Element type mismatch: file uses {ElementTypeName}, " +
                   $"but caller requested {current.ElementTypeName}.";

        return null; // compatible
    }
}
