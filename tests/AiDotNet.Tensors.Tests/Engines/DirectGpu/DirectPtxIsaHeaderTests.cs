using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Regression coverage for the PTX module header. Every direct-PTX emitter used to hard-code
/// <c>.version 7.1</c> next to <c>.target sm_{ccMajor}{ccMinor}</c>; on an sm_89 (Ada) device the driver JIT
/// rejected every module with "PTX .version 7.1 does not support .target sm_89" (CUDA_ERROR_INVALID_PTX), and the
/// unit tests could not see it because they only ever emitted for sm_86.
/// </summary>
public class DirectPtxIsaHeaderTests
{
    /// <summary>The compute capabilities the emitters are expected to serve, Volta through Blackwell.</summary>
    public static IEnumerable<object[]> SupportedCapabilities() => new[]
    {
        new object[] { 7, 0 }, new object[] { 7, 5 }, new object[] { 8, 0 }, new object[] { 8, 6 },
        new object[] { 8, 7 }, new object[] { 8, 8 }, new object[] { 8, 9 }, new object[] { 9, 0 },
        new object[] { 10, 0 }, new object[] { 10, 3 }, new object[] { 11, 0 }, new object[] { 12, 0 },
        new object[] { 12, 1 },
    };

    [Theory]
    [InlineData(7, 0, "7.1", "sm_70")]
    [InlineData(7, 5, "7.1", "sm_75")]
    [InlineData(8, 0, "7.1", "sm_80")]
    [InlineData(8, 6, "7.1", "sm_86")]
    [InlineData(8, 7, "7.4", "sm_87")]
    [InlineData(8, 8, "9.0", "sm_88")]
    [InlineData(8, 9, "7.8", "sm_89")]
    [InlineData(9, 0, "7.8", "sm_90")]
    [InlineData(10, 0, "8.6", "sm_100")]
    [InlineData(10, 3, "8.8", "sm_103")]
    [InlineData(11, 0, "9.0", "sm_110")]
    [InlineData(12, 0, "8.7", "sm_120")]
    [InlineData(12, 1, "8.8", "sm_121")]
    // Unknown capabilities fall back to the highest known target below them (plain sm_XY PTX is
    // forward-compatible), never to a guessed ISA for a target the table cannot name.
    [InlineData(8, 5, "7.1", "sm_80")]
    [InlineData(9, 5, "7.8", "sm_90")]
    [InlineData(13, 0, "8.8", "sm_121")]
    public void VersionAlwaysNamesTheTarget(int major, int minor, string version, string target)
    {
        Assert.Equal(version, PtxIsa.VersionFor(major, minor));
        Assert.Equal(target, PtxIsa.TargetFor(major, minor));
        string header = PtxIsa.AppendVersionAndTarget(new StringBuilder(), major, minor).ToString();
        Assert.Equal(".version " + version + Environment.NewLine + ".target " + target + Environment.NewLine, header);
    }

    [Fact]
    public void Sm86HeaderIsUnchanged_SoShippedCubinHashesStillMatch()
    {
        string header = PtxIsa.AppendVersionAndTarget(new StringBuilder(), 8, 6).ToString();
        Assert.Equal(".version 7.1" + Environment.NewLine + ".target sm_86" + Environment.NewLine, header);
    }

    [Theory]
    [InlineData(8, 6, "7.5", "7.5")]
    [InlineData(8, 9, "7.5", "7.8")]
    [InlineData(8, 8, "7.5", "9.0")]
    [InlineData(12, 0, "7.5", "8.7")]
    public void MinimumVersionNeverLowersBelowTheTargetsIsa(int major, int minor, string minimum, string expected)
        => Assert.Equal(expected, PtxIsa.VersionFor(major, minor, minimum));

    [Theory]
    [InlineData(11080, 7, 8)]
    [InlineData(12000, 8, 0)]
    [InlineData(12050, 8, 5)]
    [InlineData(12060, 8, 5)]
    [InlineData(12080, 8, 7)]
    [InlineData(12090, 8, 8)]
    [InlineData(13000, 9, 0)]
    public void DriverIsaCeilingMatchesToolkitReleases(int driverVersion, int isaMajor, int isaMinor)
        => Assert.Equal((isaMajor, isaMinor), PtxIsa.MaxIsaForDriver(driverVersion));

    /// <summary>
    /// Every parameter-free direct-PTX emitter (<c>EmitPtx(int major, int minor)</c> and friends), emitted for
    /// every supported architecture, must produce a header whose <c>.version</c> can name its <c>.target</c>.
    /// </summary>
    [Theory]
    [MemberData(nameof(SupportedCapabilities))]
    public void EveryParameterlessEmitter_EmitsAConsistentHeader(int major, int minor)
    {
        var emitters = ParameterlessEmitters();
        Assert.True(emitters.Count >= 40, "Expected to discover the direct-PTX emitters; found " + emitters.Count + ".");
        int emitted = 0;
        var failures = new List<string>();
        foreach (var emitter in emitters)
        {
            string? ptx = TryEmit(emitter, major, minor);
            if (ptx is null) continue;
            emitted++;
            string? problem = HeaderProblem(ptx, major, minor);
            if (problem is not null) failures.Add(Describe(emitter) + ": " + problem);
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        Assert.True(major < 8 || emitted > 0, "No emitter accepted sm_" + major + minor + ".");
    }

    /// <summary>
    /// The device check that would have caught the sm_89 failure on the first run: the header for the current
    /// device is within the driver's PTX JIT ceiling, and every parameter-free emitter's module links.
    /// </summary>
    [SkippableFact]
    public void EveryParameterlessEmitter_LinksOnTheCurrentDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        var ceiling = PtxIsa.MaxIsaForDriver(runtime.DriverVersion);
        string version = PtxIsa.VersionFor(major, minor);
        Assert.True(
            PtxIsa.Compare(version, ceiling.Major + "." + ceiling.Minor) <= 0,
            $"PTX .version {version} for sm_{major}{minor} exceeds driver {runtime.DriverVersion}'s JIT ceiling " +
            $"{ceiling.Major}.{ceiling.Minor}.");

        var failures = new List<string>();
        int linked = 0;
        foreach (var emitter in ParameterlessEmitters())
        {
            string? ptx = TryEmit(emitter, major, minor);
            if (ptx is null) continue;
            try
            {
                using var module = runtime.LoadModule(ptx);
                linked++;
            }
            catch (Exception ex)
            {
                failures.Add(Describe(emitter) + ": " + FirstLines(ex.Message, 3));
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        Assert.True(linked > 0, "No emitter produced a module for sm_" + major + minor + ".");
    }

    /// <summary>
    /// Source audit: a PTX header may only be written through <see cref="PtxIsa"/> (or the Codegen wrappers that
    /// delegate to it). A literal <c>.version</c> is allowed only beside <c>PtxIsa71Target.For</c>, which clamps the
    /// target to one ISA 7.1 can name.
    /// </summary>
    [Fact]
    public void NoEmitterHardCodesAVersionOrADeviceTarget()
    {
        string root = RepoRoot();
        var literalVersion = new Regex(@"""\s*\.version\s+\d+\.\d+", RegexOptions.Compiled);
        var rawTarget = new Regex(@"\.target sm_""|\.target sm_\{", RegexOptions.Compiled);
        var failures = new List<string>();
        foreach (string file in Directory.EnumerateFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories))
        {
            string text = Regex.Replace(File.ReadAllText(file), @"//[^\n]*", " ");
            bool clampedTo71 = text.Contains("PtxIsa71Target.For(");
            if (literalVersion.IsMatch(text) && !clampedTo71)
                failures.Add(Path.GetFileName(file) + ": literal .version");
            if (rawTarget.IsMatch(text))
                failures.Add(Path.GetFileName(file) + ": .target built from the raw compute capability");
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static List<MethodInfo> ParameterlessEmitters()
    {
        var assembly = typeof(PtxIsa).Assembly;
        Type[] types;
        try { types = assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t is not null).ToArray()!; }

        return types
            .Where(t => t.Namespace == typeof(PtxIsa).Namespace && !t.ContainsGenericParameters)
            .SelectMany(t => t.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            .Where(m => m.ReturnType == typeof(string) && m.Name.StartsWith("Emit", StringComparison.Ordinal))
            .Where(m =>
            {
                var p = m.GetParameters();
                return p.Length == 2 && p[0].ParameterType == typeof(int) && p[1].ParameterType == typeof(int)
                    && p[0].Name!.IndexOf("major", StringComparison.OrdinalIgnoreCase) >= 0
                    && p[1].Name!.IndexOf("minor", StringComparison.OrdinalIgnoreCase) >= 0;
            })
            .OrderBy(Describe, StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>Emits, or returns null when the emitter declines this architecture.</summary>
    private static string? TryEmit(MethodInfo emitter, int major, int minor)
    {
        try
        {
            return (string?)emitter.Invoke(null, new object[] { major, minor });
        }
        catch (TargetInvocationException ex) when (ex.InnerException is PlatformNotSupportedException
                                                   or NotSupportedException or ArgumentOutOfRangeException)
        {
            return null;
        }
    }

    private static string? HeaderProblem(string ptx, int major, int minor)
    {
        var version = Regex.Match(ptx, @"^\s*\.version\s+(\d+\.\d+)\s*$", RegexOptions.Multiline);
        var target = Regex.Match(ptx, @"^\s*\.target\s+sm_(\d+)\s*$", RegexOptions.Multiline);
        if (!version.Success || !target.Success) return "missing .version/.target";

        int targetCapability = int.Parse(target.Groups[1].Value, CultureInfo.InvariantCulture);
        int deviceCapability = major * 10 + minor;
        if (targetCapability > deviceCapability)
            return $".target sm_{targetCapability} cannot run on sm_{deviceCapability}";

        string required = PtxIsa.VersionFor(targetCapability / 10, targetCapability % 10);
        if (PtxIsa.Compare(version.Groups[1].Value, required) < 0)
            return $".version {version.Groups[1].Value} cannot name .target sm_{targetCapability} (needs {required})";
        return null;
    }

    private static string Describe(MethodInfo method) => method.DeclaringType!.Name + "." + method.Name;

    private static string FirstLines(string text, int count)
        => string.Join(" | ", text.Split('\n').Take(count).Select(l => l.Trim()));

    private static string RepoRoot()
    {
        var probe = Path.Combine("src", "AiDotNet.Tensors", "Engines", "CpuEngine.cs");
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, probe)))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException(
            $"could not locate repo root by walking up from {AppContext.BaseDirectory} looking for {probe}");
    }
}
