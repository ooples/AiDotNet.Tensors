using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Diagnostic: runs the catalog through the autotune cache and reports
/// per-shape decisions (managed vs native). Gated on
/// <c>AIDOTNET_RUN_AUTOTUNE_DIAG=1</c> because it's slow (measures 50+ shapes).
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class AutotuneCatalogDecisionsTest
{
    private readonly ITestOutputHelper _output;

    public AutotuneCatalogDecisionsTest(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Autotune_Decides_Per_Shape_Across_Catalog()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_AUTOTUNE_DIAG") != "1") return;
        if (!BlasProvider.IsAvailable) return;  // skip if no native available — autotune is pointless

        PrefersManagedCache.DiskPath = Path.Combine(
            Path.GetTempPath(),
            $"autotune-diag-{Guid.NewGuid():N}.json");
        PrefersManagedCache.Clear();
        try
        {
            int managedWins = 0, nativeWins = 0;
            foreach (var shape in ShapeCatalog.All)
            {
                Type dtype = shape.Dtype == DType.Single ? typeof(float) : typeof(double);
                bool prefersManaged = PrefersManagedCache.PrefersManaged(
                    shape.M, shape.N, shape.K, shape.TransA, shape.TransB, dtype);
                if (prefersManaged) managedWins++; else nativeWins++;
                _output.WriteLine($"{shape.Name,-45}  ratio-pick: {(prefersManaged ? "MANAGED" : "native")}");
            }
            _output.WriteLine("");
            _output.WriteLine($"Catalog: {ShapeCatalog.All.Count} shapes");
            _output.WriteLine($"  Managed wins: {managedWins} ({100.0 * managedWins / ShapeCatalog.All.Count:F0}%)");
            _output.WriteLine($"  Native wins:  {nativeWins} ({100.0 * nativeWins / ShapeCatalog.All.Count:F0}%)");
        }
        finally
        {
            try { File.Delete(PrefersManagedCache.DiskPath!); } catch { }
            PrefersManagedCache.Clear();
        }
    }
}
