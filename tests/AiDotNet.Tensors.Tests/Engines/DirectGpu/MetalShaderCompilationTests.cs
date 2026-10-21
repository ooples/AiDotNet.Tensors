using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Compiles every Metal shader library the Metal backend builds, with Apple's own Metal compiler, on a machine that
/// has no Apple GPU (#915, #921). The backend compiles its sources at run time on a Mac, so until now a Metal source
/// that does not compile was only found by someone running on Apple hardware. This catches compile errors; it does
/// not run the kernels.
/// <para>
/// Needs Apple's Metal Developer Tools for Windows (the <c>metal</c> compiler). It is found on PATH, in
/// <c>AIDOTNET_METAL_COMPILER</c>, or at its default install location; the tests skip without it.
/// </para>
/// </summary>
public sealed class MetalShaderCompilationTests
{
    /// <summary>
    /// Every library MetalBackend compiles, with its source and whether it is compiled with fast math off. A test
    /// below checks this list against the backend's code, so a new library cannot be added without being compiled here.
    /// </summary>
    public static IEnumerable<object[]> Libraries() => Catalog().Select(entry => new object[] { entry.Name });

    private static IEnumerable<(string Name, Func<string> Source, bool Strict)> Catalog()
    {
        yield return ("ElementWise", () => MetalKernels.ElementWiseKernels, false);
        yield return ("Activation", () => MetalKernels.ActivationKernels, false);
        yield return ("Trig", () => MetalKernels.TrigKernels, false);
        yield return ("Reduction", () => MetalKernels.ReductionKernels, false);
        yield return ("Matrix", () => MetalKernels.MatrixKernels, false);
        yield return ("Normalization", () => MetalKernels.NormalizationKernels, false);
        yield return ("Convolution", () => MetalKernels.ConvolutionKernels, false);
        yield return ("Attention", () => MetalKernels.AttentionKernels, false);
        yield return ("Optimizer", () => MetalKernels.OptimizerKernels, false);
        yield return ("Loss", () => MetalKernels.LossKernels, false);
        yield return ("Comparison", () => MetalKernels.ComparisonKernels, false);
        yield return ("Random", () => MetalKernels.RandomKernels, false);
        yield return ("DotProduct", () => MetalKernels.DotProductKernels, false);
        yield return ("GatedActivation", () => MetalKernels.GatedActivationKernels, false);
        yield return ("ExtendedReduction", () => MetalKernels.ExtendedReductionKernels, false);
        yield return ("Hyperbolic", () => MetalKernels.HyperbolicKernels, false);
        yield return ("Octonion", () => MetalKernels.OctonionKernels, false);
        yield return ("SpectralPerf", () => MetalKernels.SpectralPerfKernels, false);
        yield return ("FusedLinear", () => MetalKernels.FusedLinearKernels, false);
        yield return ("QuantGemm", () => MetalKernels.QuantGemmKernels, false);
        yield return ("PagedAttention", () => MetalKernels.PagedAttentionKernels, false);
        yield return ("FlashDecode", () => MetalKernels.FlashDecodeKernels, false);
        yield return ("IoULoss", () => MetalKernels.IoULossKernels, false);
        yield return ("Recurrence", () => MetalRecurrenceKernels.Source, false);
        yield return ("FusedAdvancedKernels", () => MetalFusedAdvancedKernels.Source, false);
        yield return ("Ann", () => MetalAnnKernels.Source, false);
        yield return ("Audio", () => MetalAudioKernels.Source, false);
        yield return ("Complex", () => MetalComplexKernels.GetSource(), false);
        yield return ("CompressedOptimizer", () => MetalCompressedOptimizerKernels.Source, false);
        yield return ("ExtendedConv", () => MetalExtendedConvKernels.Source, false);
        yield return ("Fft212", () => MetalFftKernels.Source, false);
        yield return ("Resident", () => MetalResidentKernels.Source, false);
        yield return ("Roi", () => MetalRoiKernels.Source, false);
        yield return ("Sparse", () => MetalSparseKernels.Source, false);
        yield return ("Parity210", () => MetalParity210Kernels.Source, true);
        yield return ("Linalg", () => MetalLinalgKernels.Source, false);
        yield return ("Detection", () => MetalDetectionKernels.Source, false);
        yield return ("Geometry", () => MetalGeometryKernels.Source, false);
        yield return ("TunedRow", () => MetalTunedRowKernels.GetSource(), false);
        yield return ("Categorical", () => MetalCategoricalKernels.Source, true);
        yield return ("BatchExecution",() => MetalBatchExecutionKernels.Source, true);
        yield return ("TorchParityF32", () => MetalBatchExecutionKernels.TorchParitySource, true);
        // Two libraries, matching what MetalBackend compiles: the fused weight gradient is emitted apart from
        // the transforms so a driver that rejects it drops only the fused candidate.
        yield return ("Winograd", () => WinogradF23KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal,
            WinogradKernelGroup.Base), true);
        yield return ("WinogradFused", () => WinogradF23KernelTemplate.GetModuleSource(WinogradKernelDialect.Metal,
            WinogradKernelGroup.Fused), true);
        yield return ("ResidentProgram", () => ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.Metal), true);
        // Strict: false, as MetalBackend.BatchNormRegister compiles it (CompileLibrary's two-argument overload). The
        // mask the backward rebuilds rounds the same as the forward because both go through one explicit fma, not
        // because the compiler was asked to keep the operators apart.
        yield return ("BatchNormRegister", () => BatchNormRegisterKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal), false);
    }

    internal static string? FindCompiler()
    {
        string? configured = Environment.GetEnvironmentVariable("AIDOTNET_METAL_COMPILER");
        if (!string.IsNullOrEmpty(configured) && File.Exists(configured)) return configured;
        foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            if (string.IsNullOrWhiteSpace(dir)) continue;
            string candidate = Path.Combine(dir.Trim(), RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "metal.exe" : "metal");
            if (File.Exists(candidate)) return candidate;
        }
        string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        string installed = Path.Combine(programFiles, "Metal Developer Tools", "bin", "metal.exe");
        return File.Exists(installed) ? installed : null;
    }

    /// <summary>Compiles <paramref name="source"/> to AIR; null when it compiles, else the compiler's diagnostics.</summary>
    internal static string? Compile(string compiler, string name, string source, bool strict)
    {
        string dir = Path.Combine(Path.GetTempPath(), "aidotnet-metal-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string input = Path.Combine(dir, name + ".metal");
            File.WriteAllText(input, source);
            string output = Path.Combine(dir, name + ".air");
            // MetalShaderLibrary sets no language version, so at run time the OS compiles with the newest one it has.
            // The offline compiler instead defaults to macOS 10.15's (no atomic_float), so pin Metal 3.0: the
            // default on macOS 13, the oldest macOS the backend's MPS paths support. Strict libraries are compiled
            // with fast math off, as MetalShaderLibrary does at run time.
            string arguments = "-std=metal3.0 -c " + (strict ? "-fno-fast-math " : "") + Quote(input) + " -o " + Quote(output);
            var start = new ProcessStartInfo(compiler, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                WorkingDirectory = dir,
            };
            using var process = Process.Start(start)!;
            // Both streams are read asynchronously so a compiler that hangs with its pipes open cannot block the
            // test before the timeout is reached.
            var stderr = process.StandardError.ReadToEndAsync();
            var stdoutRead = process.StandardOutput.ReadToEndAsync();
            if (!process.WaitForExit(180_000))
            {
                try { process.Kill(); } catch (InvalidOperationException) { }
                return "the Metal compiler did not finish within 3 minutes";
            }
            process.WaitForExit();   // drains the redirected streams
            string stdout = stdoutRead.Result;
            if (process.ExitCode == 0) return null;
            // Keep the source so the line numbers in the diagnostics can be looked up.
            string kept = Path.Combine(Path.GetTempPath(), "aidotnet-metal-failed", name + ".metal");
            Directory.CreateDirectory(Path.GetDirectoryName(kept)!);
            File.Copy(input, kept, overwrite: true);
            return ("source kept at " + kept + Environment.NewLine + stdout + stderr.Result).Trim();
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Libraries))]
    public void Library_CompilesWithAppleMetalCompiler(string library)
    {
        string? compiler = FindCompiler();
        Skip.If(compiler is null, "Apple's Metal compiler is not installed (Metal Developer Tools for Windows, or set AIDOTNET_METAL_COMPILER).");
        var entry = Catalog().Single(e => e.Name == library);
        string? error = Compile(compiler!, library, entry.Source(), entry.Strict);
        Assert.True(error is null, $"Metal library '{library}' does not compile:{Environment.NewLine}{Truncate(error)}");
    }

    [SkippableFact]
    public void TheCatalog_CoversEveryLibraryTheBackendCompiles()
    {
        // Every CompileLibrary(name, ...) in the Metal backend's sources, whether the name is a string literal or a
        // string constant. MetalShaderLibrary.cs is skipped: it defines CompileLibrary and forwards its own parameter.
        string? root = RepositoryRoot();
        Skip.If(root is null, "The repository sources are not next to the test binaries.");
        string metal = Path.Combine(root!, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Metal");
        var timeout = TimeSpan.FromSeconds(5);
        var sources = Directory.GetFiles(metal, "*.cs")
            .Where(f => !string.Equals(Path.GetFileName(f), "MetalShaderLibrary.cs", StringComparison.Ordinal))
            .Select(File.ReadAllText).ToArray();
        var constants = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string text in sources)
            foreach (Match m in Regex.Matches(text, @"const\s+string\s+(\w+)\s*=\s*""(\w+)""\s*;", RegexOptions.None, timeout))
                constants[m.Groups[1].Value] = m.Groups[2].Value;
        var compiled = new HashSet<string>(StringComparer.Ordinal);
        var unresolved = new List<string>();
        foreach (string text in sources)
            foreach (Match m in Regex.Matches(text, @"CompileLibrary\(\s*(?:""(\w+)""|(\w+))\s*,", RegexOptions.None, timeout))
            {
                if (m.Groups[1].Success) compiled.Add(m.Groups[1].Value);
                else if (constants.TryGetValue(m.Groups[2].Value, out string? name)) compiled.Add(name);
                else unresolved.Add(m.Groups[2].Value);
            }
        Assert.True(unresolved.Count == 0,
            "CompileLibrary is called with a name this test cannot resolve to a constant: " + string.Join(", ", unresolved));
        var catalogued = new HashSet<string>(Catalog().Select(e => e.Name), StringComparer.Ordinal);
        var missing = compiled.Where(n => !catalogued.Contains(n)).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.True(missing.Length == 0, "Metal libraries compiled by the backend but not by this test: " + string.Join(", ", missing));
    }

    [SkippableFact]
    public void EveryKernelLookup_NamesAKernelDefinedInThatLibrary()
    {
        // GetPipeline("Name", _xLibrary, "kernel") and the Simple*/Dispatch* helpers that forward to it look a kernel
        // up by name in one compiled library; a name that library does not define throws "Function ... not found in
        // library" on a Mac, and only there. The first run on an Apple GPU found two such failures, and this test, which
        // needs no compiler, found the rest. Each library field is resolved through its CompileLibrary("Name", ...)
        // assignment to the catalogued source.
        string? root = RepositoryRoot();
        Skip.If(root is null, "The repository sources are not next to the test binaries.");
        string metal = Path.Combine(root!, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Metal");
        var timeout = TimeSpan.FromSeconds(5);
        var files = Directory.GetFiles(metal, "*.cs").ToDictionary(Path.GetFileName, File.ReadAllText, StringComparer.Ordinal);
        var catalog = Catalog().ToDictionary(e => e.Name, e => e.Source, StringComparer.Ordinal);

        // The library name is a string literal at some call sites and a string constant at others (the Winograd
        // libraries are named by WinogradLibraryName and WinogradFusedLibraryName). Matching only literals silently
        // dropped every lookup against those libraries, which is the blind spot this test exists to close, so the
        // constants are resolved here the way TheCatalog_CoversEveryLibraryTheBackendCompiles resolves them.
        var libraryConstants = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string text in files.Values)
            foreach (Match m in Regex.Matches(text, @"const\s+string\s+(\w+)\s*=\s*""(\w+)""\s*;", RegexOptions.None, timeout))
                libraryConstants[m.Groups[1].Value] = m.Groups[2].Value;

        var fieldToLibrary = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string text in files.Values)
            foreach (Match m in Regex.Matches(text, @"(_\w+Library)\s*=\s*_shaderLibrary\.CompileLibrary\(\s*(?:""(\w+)""|(\w+))", RegexOptions.None, timeout))
            {
                if (m.Groups[2].Success) fieldToLibrary[m.Groups[1].Value] = m.Groups[2].Value;
                else if (libraryConstants.TryGetValue(m.Groups[3].Value, out string? named))
                    fieldToLibrary[m.Groups[1].Value] = named;
            }

        // The control for the resolution above: both Winograd libraries are named by a constant, so a regression to
        // literal-only matching drops them from this map and silently stops checking their kernel lookups - which is
        // what it did until the fused library was split out and the omission became visible.
        Assert.Contains("Winograd", fieldToLibrary.Values);
        Assert.Contains("WinogradFused", fieldToLibrary.Values);

        var definedCache = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        HashSet<string> Defined(string library)
        {
            if (!definedCache.TryGetValue(library, out var set))
            {
                set = new HashSet<string>(Regex.Matches(catalog[library](), @"kernel\s+void\s+(\w+)\s*\(", RegexOptions.None, timeout)
                    .Cast<Match>().Select(k => k.Groups[1].Value), StringComparer.Ordinal);
                definedCache[library] = set;
            }
            return set;
        }

        var problems = new List<string>();
        int checkedLookups = 0;
        foreach (var file in files)
            foreach (Match m in Regex.Matches(file.Value, @"\(\s*""\w+""\s*,\s*(_\w+Library)\s*,\s*""(\w+)""", RegexOptions.None, timeout))
            {
                string field = m.Groups[1].Value, kernel = m.Groups[2].Value;
                if (!fieldToLibrary.TryGetValue(field, out string? library) || !catalog.ContainsKey(library)) continue;
                checkedLookups++;
                if (!Defined(library).Contains(kernel))
                {
                    int line = file.Value.Take(m.Index).Count(c => c == '\n') + 1;
                    problems.Add($"{file.Key}:{line} asks {library} for '{kernel}'");
                }
            }
        Assert.True(checkedLookups > 100, $"Only {checkedLookups} kernel lookups were recognised; the pattern no longer matches the backend.");
        Assert.True(problems.Count == 0, "Kernel lookups naming a kernel the library does not define:" + Environment.NewLine
            + string.Join(Environment.NewLine, problems));
    }

    private static string? RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            if (Directory.Exists(Path.Combine(dir.FullName, "src", "AiDotNet.Tensors"))) return dir.FullName;
        return null;
    }

    private static string Quote(string path) => "\"" + path + "\"";

    private static string? Truncate(string? text) => text is null || text.Length <= 6000 ? text : text.Substring(0, 6000) + " ...";
}
