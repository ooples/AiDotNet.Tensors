// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The single source of truth for the <c>.version</c> / <c>.target</c> pair at the top of every
/// hand-written PTX module.
/// </summary>
/// <remarks>
/// <para>
/// A PTX module's ISA version must be able to <em>name</em> its target: ptxas rejects
/// <c>.version 7.1</c> + <c>.target sm_89</c> with "PTX .version 7.1 does not support .target sm_89",
/// because sm_89 was only introduced in ISA 7.8. The direct-PTX emitters used to hard-code
/// <c>.version 7.1</c> while parameterising the target with the device's compute capability, so every
/// module JIT-linked on anything newer than Ampere (sm_87, sm_89 Ada, sm_90 Hopper, Blackwell) failed
/// with <c>CUDA_ERROR_INVALID_PTX</c>.
/// </para>
/// <para>
/// The version chosen is the <em>minimum</em> ISA that names the target, floored at 7.1. That is always
/// JIT-able: a driver new enough to run a device of compute capability X necessarily ships a PTX JIT
/// that understands the ISA which introduced <c>sm_X</c>. The 7.1 floor keeps the text emitted for
/// sm_86 and older byte-identical to what the shipped sm_86 cubins were built from.
/// </para>
/// <para>
/// A compute capability this table does not know (a future GPU) is emitted for the highest known target
/// at or below it: plain <c>sm_XY</c> PTX is forward-compatible and JIT-compiles for every device whose
/// compute capability is &gt;= XY, whereas guessing an ISA version for an unknown target cannot be
/// validated.
/// </para>
/// </remarks>
internal static class PtxIsa
{
    /// <summary>The ISA every module is emitted at when its target does not require a newer one.</summary>
    internal const string FloorVersion = "7.1";

    /// <summary>
    /// Known targets (compute capability as major*10+minor) and the PTX ISA version that introduced each,
    /// ascending by capability. Source: PTX ISA "Release notes" / "PTX Module Directives: .target".
    /// </summary>
    private static readonly (int Capability, int IsaMajor, int IsaMinor)[] KnownTargets =
    {
        (30, 3, 1), (32, 4, 0), (35, 3, 1), (37, 4, 1),
        (50, 4, 0), (52, 4, 1), (53, 4, 2),
        (60, 5, 0), (61, 5, 0), (62, 5, 0),
        (70, 6, 0), (72, 6, 1), (75, 6, 3),
        (80, 7, 0), (86, 7, 1), (87, 7, 4), (88, 9, 0), (89, 7, 8),
        (90, 7, 8),
        (100, 8, 6), (101, 8, 6), (103, 8, 8),
        (110, 9, 0),
        (120, 8, 7), (121, 8, 8),
    };

    /// <summary>The <c>.target</c> name (e.g. <c>sm_89</c>) a module for this device is emitted for.</summary>
    internal static string TargetFor(int computeMajor, int computeMinor)
        => "sm_" + Resolve(computeMajor, computeMinor).Capability.ToString(CultureInfo.InvariantCulture);

    /// <summary>The <c>.version</c> (e.g. <c>7.8</c>) that can name <see cref="TargetFor"/>.</summary>
    internal static string VersionFor(int computeMajor, int computeMinor)
    {
        var target = Resolve(computeMajor, computeMinor);
        if (target.IsaMajor < 7 || (target.IsaMajor == 7 && target.IsaMinor <= 1))
            return FloorVersion;
        return target.IsaMajor.ToString(CultureInfo.InvariantCulture) + "." +
               target.IsaMinor.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// <see cref="VersionFor(int, int)"/>, raised to <paramref name="minimumVersion"/> when the module uses an
    /// instruction form introduced in a later ISA than the one that names the target.
    /// </summary>
    internal static string VersionFor(int computeMajor, int computeMinor, string minimumVersion)
    {
        string forTarget = VersionFor(computeMajor, computeMinor);
        return Compare(forTarget, minimumVersion) >= 0 ? forTarget : minimumVersion;
    }

    /// <summary>Orders two <c>major.minor</c> PTX ISA versions numerically.</summary>
    internal static int Compare(string left, string right)
    {
        var (lm, ln) = Parse(left);
        var (rm, rn) = Parse(right);
        return lm != rm ? lm.CompareTo(rm) : ln.CompareTo(rn);
    }

    internal static (int Major, int Minor) Parse(string version)
    {
        if (version is null) throw new ArgumentNullException(nameof(version));
        int dot = version.IndexOf('.');
        if (dot <= 0
            || !int.TryParse(version.Substring(0, dot), NumberStyles.None, CultureInfo.InvariantCulture, out int major)
            || !int.TryParse(version.Substring(dot + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int minor))
            throw new FormatException("'" + version + "' is not a PTX ISA version of the form major.minor.");
        return (major, minor);
    }

    /// <summary>
    /// The newest PTX ISA a CUDA driver's JIT accepts, from <c>cuDriverGetVersion</c>
    /// (1000*major + 10*minor). CUDA 11.x ships ISA 7.x and 13.x ships 9.x with the minor tracking the
    /// toolkit minor; 12.x does too up to 12.5 (ISA 8.5), after which 12.6 kept 8.5, 12.8 shipped 8.7 and
    /// 12.9 shipped 8.8 (there was no 12.7).
    /// </summary>
    internal static (int Major, int Minor) MaxIsaForDriver(int driverVersion)
    {
        int cudaMajor = driverVersion / 1000;
        int cudaMinor = driverVersion % 1000 / 10;
        if (cudaMajor < 11) throw new NotSupportedException("CUDA drivers older than 11.0 are not supported.");
        if (cudaMajor == 12 && cudaMinor >= 6) return (8, cudaMinor - 1);
        return (cudaMajor - 4, cudaMinor);
    }

    /// <summary>
    /// Appends <c>.version</c> and <c>.target</c> lines (each terminated by <see cref="StringBuilder.AppendLine()"/>)
    /// for the given device.
    /// </summary>
    internal static StringBuilder AppendVersionAndTarget(StringBuilder ptx, int computeMajor, int computeMinor)
    {
        if (ptx is null) throw new ArgumentNullException(nameof(ptx));
        ptx.Append(".version ").AppendLine(VersionFor(computeMajor, computeMinor));
        ptx.Append(".target ").AppendLine(TargetFor(computeMajor, computeMinor));
        return ptx;
    }

    private static (int Capability, int IsaMajor, int IsaMinor) Resolve(int computeMajor, int computeMinor)
    {
        if (computeMajor <= 0) throw new ArgumentOutOfRangeException(nameof(computeMajor));
        if (computeMinor < 0 || computeMinor > 9) throw new ArgumentOutOfRangeException(nameof(computeMinor));
        int capability = computeMajor * 10 + computeMinor;
        if (capability < KnownTargets[0].Capability)
            throw new NotSupportedException(
                "Compute capability " + computeMajor.ToString(CultureInfo.InvariantCulture) + "." +
                computeMinor.ToString(CultureInfo.InvariantCulture) + " predates every PTX target this library emits.");

        var best = KnownTargets[0];
        foreach (var known in KnownTargets)
        {
            if (known.Capability > capability) break;
            best = known;
        }
        return best;
    }
}
