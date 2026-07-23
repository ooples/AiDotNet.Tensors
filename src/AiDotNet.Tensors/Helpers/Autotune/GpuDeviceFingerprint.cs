using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// A structured identity for a GPU, unifying the previously ad-hoc fingerprint
/// strings so autotune configs are keyed consistently. It exposes two distinct
/// keys, because "tune once per machine" and "share across users" need
/// different granularity:
///
/// <list type="bullet">
/// <item><see cref="LocalKey"/> (a.k.a. <see cref="ToCacheToken"/>) is per
/// <b>physical card</b> — it includes the device UUID. This is what the on-disk
/// <see cref="AutotuneCache"/> keys on, so a machine tunes each of its cards
/// exactly once. The token is byte-compatible with the legacy CUDA fingerprint
/// (<c>gpu-{uuid}-sm{cc}-drv{driver}</c>), so existing caches stay valid.</item>
/// <item><see cref="ModelKey"/> is per <b>GPU model</b> — vendor, model, arch,
/// and driver, with <b>no</b> UUID. A config measured on one RTX 3080 applies to
/// every RTX 3080 on the same driver and architecture, so this is the key that
/// opt-in community sharing (Phase 2, Supabase <c>gpu_profiles</c>) rows use.</item>
/// </list>
/// </summary>
public readonly struct GpuDeviceFingerprint : IEquatable<GpuDeviceFingerprint>
{
    /// <summary>Normalized vendor: <c>nvidia</c>, <c>amd</c>, <c>intel</c>, <c>apple</c>, or <c>other</c>.</summary>
    public string Vendor { get; }

    /// <summary>Raw device model name as reported by the driver (e.g. "NVIDIA GeForce RTX 3080").</summary>
    public string Model { get; }

    /// <summary>Compute-capability / architecture major number (SM major on CUDA).</summary>
    public int ArchitectureMajor { get; }

    /// <summary>Compute-capability / architecture minor number (SM minor on CUDA).</summary>
    public int ArchitectureMinor { get; }

    /// <summary>Driver version integer (as CUDA reports it).</summary>
    public int DriverVersion { get; }

    /// <summary>Per-card unique id (device UUID, or <c>ordinal-N</c> when unavailable).</summary>
    public string UniqueId { get; }

    public GpuDeviceFingerprint(
        string vendor, string model, int architectureMajor, int architectureMinor,
        int driverVersion, string uniqueId)
    {
        if (string.IsNullOrWhiteSpace(uniqueId))
            throw new ArgumentException("A per-card unique id is required.", nameof(uniqueId));
        Vendor = string.IsNullOrWhiteSpace(vendor) ? "other" : vendor;
        Model = model ?? string.Empty;
        ArchitectureMajor = architectureMajor;
        ArchitectureMinor = architectureMinor;
        DriverVersion = driverVersion;
        UniqueId = uniqueId;
    }

    /// <summary>Compact architecture tag, e.g. <c>sm86</c>.</summary>
    public string Architecture => string.Concat(
        "sm",
        ArchitectureMajor.ToString(CultureInfo.InvariantCulture),
        ArchitectureMinor.ToString(CultureInfo.InvariantCulture));

    /// <summary>
    /// Per-physical-card cache token, byte-compatible with the legacy CUDA
    /// fingerprint <c>gpu-{uuid}-sm{cc}-drv{driver}</c>. Used as the local cache
    /// key so existing on-disk autotune winners are not invalidated.
    /// </summary>
    public string ToCacheToken() => string.Concat(
        "gpu-", UniqueId, "-", Architecture, "-drv",
        DriverVersion.ToString(CultureInfo.InvariantCulture));

    /// <summary>Alias for <see cref="ToCacheToken"/> — the per-card local autotune key.</summary>
    public string LocalKey => ToCacheToken();

    /// <summary>
    /// Per-GPU-model shareable key: <c>{vendor}|{model}|{arch}|drv{driver}</c>,
    /// with no UUID, so a config tuned on one card of a model applies to all
    /// cards of that model on the same driver/architecture. This is the Phase-2
    /// community-sharing key.
    /// </summary>
    public string ModelKey => string.Join("|",
        Vendor,
        NormalizeModel(Model),
        Architecture,
        "drv" + DriverVersion.ToString(CultureInfo.InvariantCulture));

    /// <summary>Builds a fingerprint from CUDA device attributes (all already queried at runtime init).</summary>
    public static GpuDeviceFingerprint FromCuda(
        string deviceName, string uniqueId, int computeCapabilityMajor,
        int computeCapabilityMinor, int driverVersion) =>
        new(DetectVendor(deviceName), deviceName ?? string.Empty,
            computeCapabilityMajor, computeCapabilityMinor, driverVersion,
            string.IsNullOrWhiteSpace(uniqueId) ? "unknown" : uniqueId);

    /// <summary>
    /// Parses a legacy cache token (<c>gpu-{uuid}-sm{maj}{min}-drv{driver}</c>).
    /// Vendor and model are unknown from the token alone, so the result is only
    /// suitable for <see cref="LocalKey"/> round-tripping, not <see cref="ModelKey"/>.
    /// </summary>
    public static bool TryParseCacheToken(string token, out GpuDeviceFingerprint fingerprint)
    {
        fingerprint = default;
        if (string.IsNullOrWhiteSpace(token) || !token.StartsWith("gpu-", StringComparison.Ordinal))
            return false;
        int smIndex = token.LastIndexOf("-sm", StringComparison.Ordinal);
        int drvIndex = token.LastIndexOf("-drv", StringComparison.Ordinal);
        if (smIndex <= 4 || drvIndex <= smIndex + 3) return false;

        string uuid = token.Substring(4, smIndex - 4);
        string sm = token.Substring(smIndex + 3, drvIndex - (smIndex + 3));
        string drv = token.Substring(drvIndex + 4);
        if (sm.Length < 2) return false;
        if (!int.TryParse(sm.Substring(0, sm.Length - 1), NumberStyles.None, CultureInfo.InvariantCulture, out int major) ||
            !int.TryParse(sm.Substring(sm.Length - 1), NumberStyles.None, CultureInfo.InvariantCulture, out int minor) ||
            !int.TryParse(drv, NumberStyles.None, CultureInfo.InvariantCulture, out int driver))
            return false;
        if (string.IsNullOrWhiteSpace(uuid)) return false;

        fingerprint = new GpuDeviceFingerprint("other", string.Empty, major, minor, driver, uuid);
        return true;
    }

    private static string DetectVendor(string model)
    {
        if (string.IsNullOrWhiteSpace(model)) return "other";
        string m = model.ToUpperInvariant();
        if (Contains(m, "NVIDIA") || Contains(m, "GEFORCE") || Contains(m, "TESLA") ||
            Contains(m, "QUADRO") || Contains(m, "RTX") || Contains(m, "GTX") || Contains(m, "TITAN"))
            return "nvidia";
        if (Contains(m, "AMD") || Contains(m, "RADEON") || Contains(m, "INSTINCT") || Contains(m, "FIREPRO"))
            return "amd";
        if (Contains(m, "INTEL") || Contains(m, "ARC ") || Contains(m, "IRIS") || Contains(m, "UHD GRAPHICS"))
            return "intel";
        if (Contains(m, "APPLE") || Contains(m, "M1") || Contains(m, "M2") || Contains(m, "M3"))
            return "apple";
        return "other";
    }

    private static bool Contains(string haystack, string needle) =>
        haystack.IndexOf(needle, StringComparison.Ordinal) >= 0;

    private static string NormalizeModel(string model)
    {
        if (string.IsNullOrWhiteSpace(model)) return "unknown";
        var sb = new StringBuilder(model.Length);
        bool prevSpace = false;
        foreach (char c in model.Trim())
        {
            // Collapse whitespace runs and drop the '|' separator so the model
            // segment can never break ModelKey's field structure.
            if (char.IsWhiteSpace(c))
            {
                if (!prevSpace) sb.Append(' ');
                prevSpace = true;
            }
            else if (c != '|')
            {
                sb.Append(c);
                prevSpace = false;
            }
        }
        string cleaned = sb.ToString().Trim();
        return cleaned.Length == 0 ? "unknown" : cleaned;
    }

    public bool Equals(GpuDeviceFingerprint other) =>
        string.Equals(UniqueId, other.UniqueId, StringComparison.Ordinal) &&
        ArchitectureMajor == other.ArchitectureMajor &&
        ArchitectureMinor == other.ArchitectureMinor &&
        DriverVersion == other.DriverVersion &&
        string.Equals(Vendor, other.Vendor, StringComparison.Ordinal) &&
        string.Equals(Model, other.Model, StringComparison.Ordinal);

    public override bool Equals(object? obj) => obj is GpuDeviceFingerprint other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + (UniqueId?.GetHashCode() ?? 0);
            hash = hash * 31 + ArchitectureMajor;
            hash = hash * 31 + ArchitectureMinor;
            hash = hash * 31 + DriverVersion;
            return hash;
        }
    }

    public override string ToString() => ToCacheToken();
}
