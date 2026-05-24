using System;
using System.IO;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>
/// Sub-issue A (#369) task A.4: auto-wires <see cref="ShapeInstrumenter"/> to
/// <c>BlasProvider.ShapeLogHook</c> at assembly load when
/// <c>AIDOTNET_INSTRUMENT_SHAPES=1</c>, and dumps the harvest to
/// <c>artifacts/perf/instrumented-shapes.json</c> on process exit.
///
/// <para>
/// Runs at assembly load via <see cref="ModuleInitializerAttribute"/>. When the env
/// var is unset (the normal case), the bootstrap is a no-op and the hook stays null —
/// production code pays no cost.
/// </para>
///
/// <para>
/// Usage to harvest shapes from the full test suite:
/// <code>
///   AIDOTNET_INSTRUMENT_SHAPES=1 dotnet test
/// </code>
/// Output path is controlled by env var <c>AIDOTNET_INSTRUMENT_OUT</c>; if unset,
/// resolves to <c>&lt;repo&gt;/artifacts/perf/instrumented-shapes.json</c>.
/// </para>
/// </summary>
internal static class ShapeInstrumenterBootstrap
{
    [ModuleInitializer]
    internal static void Init()
    {
        if (!ShapeInstrumenter.Enabled) return;

        // Wire the hook. Idempotent: setting the same delegate twice is harmless.
        //
        // Merge resolution (#402 + main #412): `BlasProvider.ShapeLogHook` is now the
        // 5-arg `Action<int, int, int, bool, bool>?` from main (#412 phase A.1). The
        // dtype info that this harvest mode used to record per call is no longer
        // observable from the hook signature, so we record under DType.Single as a
        // sentinel and lose the FP32-vs-FP64 split in harvested catalogs. The split
        // can be reconstructed at analysis time by intersecting with the per-shape
        // call-site Type when needed; for the issue #369 catalog-build use case
        // (which dedupes by M/N/K/transA/transB anyway) the sentinel is sufficient.
        BlasProvider.ShapeLogHook = (m, n, k, transA, transB) =>
            ShapeInstrumenter.Record(m, n, k, transA, transB, DType.Single);

        // Dump on process exit. Errors are swallowed because the test process is
        // already exiting — surfacing an IOException at this point would mask
        // legitimate test failures.
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try
            {
                var outPath = ResolveOutputPath();
                ShapeInstrumenter.DumpToJson(outPath);
            }
            catch
            {
                // Swallowed deliberately: test runner is shutting down.
            }
        };
    }

    /// <summary>
    /// Resolves the JSON output path. Env var <c>AIDOTNET_INSTRUMENT_OUT</c> overrides;
    /// fallback is <c>&lt;repo&gt;/artifacts/perf/instrumented-shapes.json</c> located
    /// by walking up from the test binary directory.
    /// </summary>
    private static string ResolveOutputPath()
    {
        var envOverride = Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_OUT");
        if (!string.IsNullOrWhiteSpace(envOverride))
            return Path.GetFullPath(envOverride!);

        // Walk up from AppContext.BaseDirectory until we find a folder containing
        // "artifacts" or the repo root marker (a .git directory). bin/Debug/<tfm>/
        // is 3 levels below the test project; the repo root is typically 5+ levels up.
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir != null; i++)
        {
            var artifactsDir = Path.Combine(dir, "artifacts");
            var gitDir = Path.Combine(dir, ".git");
            if (Directory.Exists(artifactsDir) || Directory.Exists(gitDir))
                return Path.Combine(dir, "artifacts", "perf", "instrumented-shapes.json");
            dir = Path.GetDirectoryName(dir);
        }

        // Fallback: drop in the current working directory.
        return Path.GetFullPath(Path.Combine("artifacts", "perf", "instrumented-shapes.json"));
    }
}

