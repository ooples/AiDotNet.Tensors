// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Globalization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #921: times Metal's fused two-vector Meshgrid2D (one <c>bx_meshgrid2d</c> dispatch writes both grids, taken by
/// <see cref="DirectGpuTensorEngine.TensorMeshgrid{T}(Tensor{T}[], string)"/>) against the composed route it replaced,
/// a reshape then a broadcast per grid, through the same engine so both sides pay the same allocation and dispatch
/// overheads. A shape is promoted only when the 95% confidence interval of the median paired speedup lies wholly at or
/// above 1.10x, kept composed when it lies wholly below, and reported as undecided otherwise. Exit 0 = every cell
/// promoted, 1 = some cell did not clear the gate, 2 = no Metal device.
/// </summary>
/// <remarks>
/// The hosted runner's GPU is virtualized: its timings swing 8-70%, so the three-sample spread gate never converged and
/// every cell was "no verdict" although each run's median was 1.34-1.73x. The verdict now comes from
/// <see cref="StableTimer.MeasurePairedRatioBound"/>, a distribution-free interval over 41 AB/BA brackets, which noise
/// widens but does not bias.
/// </remarks>
internal static class MetalMeshgrid2DExperiment
{
    private const double PromotionSpeedup = 1.10;

    public static int Run()
    {
        if (!OperatingSystem.IsMacOS())
        {
            Console.WriteLine("Metal Meshgrid2D: Metal runs only on macOS; nothing measured.");
            return 2;
        }
        MetalBackend backend;
        try
        {
            backend = new MetalBackend();
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            Console.WriteLine("Metal Meshgrid2D: Metal unavailable (" + ex.GetType().Name + "); nothing measured.");
            return 2;
        }
        using (backend)
        {
            if (!backend.IsAvailable)
            {
                Console.WriteLine("Metal Meshgrid2D: no Metal device; nothing measured.");
                return 2;
            }

            var engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
            Console.WriteLine("Metal Meshgrid2D fused vs composed on " + backend.DeviceName);
            Console.WriteLine("shape        mode  median   95% interval        verdict");
            bool allPromoted = true;
            foreach (var (n0, n1) in new[] { (256, 256), (1024, 256) })
                foreach (string indexing in new[] { "ij", "xy" })
                {
                    bool xy = indexing == "xy";
                    var x = Vector(n0, 1);
                    var y = Vector(n1, 2);
                    var inputs = new[] { x, y };
                    int[] grid = xy ? new[] { n1, n0 } : new[] { n0, n1 };
                    int[] axis0 = xy ? new[] { 1, n0 } : new[] { n0, 1 };
                    int[] axis1 = xy ? new[] { n1, 1 } : new[] { 1, n1 };
                    void Composed()
                    {
                        var g0 = engine.TensorBroadcastTo(engine.Reshape(x, axis0), grid);
                        var g1 = engine.TensorBroadcastTo(engine.Reshape(y, axis1), grid);
                        g0.Dispose();
                        g1.Dispose();
                    }
                    void Fused()
                    {
                        var grids = engine.TensorMeshgrid(inputs, indexing);
                        grids[0].Dispose();
                        grids[1].Dispose();
                    }
                    // Ratio = composed / fused: above 1 means the fused route is faster.
                    var bound = StableTimer.MeasurePairedRatioBound(Composed, backend.Synchronize, Fused, backend.Synchronize);
                    bool promoted = bound.Lower >= PromotionSpeedup;
                    bool rejected = bound.Upper < PromotionSpeedup;
                    allPromoted &= promoted;
                    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-12} {1,-5} {2,-8} [{3:0.00}x, {4:0.00}x]    {5}",
                        n0 + "x" + n1, indexing, bound.Median.ToString("0.00", CultureInfo.InvariantCulture) + "x",
                        bound.Lower, bound.Upper,
                        promoted ? "promote" : rejected ? "keep composed" : "undecided (interval spans " +
                            PromotionSpeedup.ToString("0.00", CultureInfo.InvariantCulture) + "x)"));
                }
            return allPromoted ? 0 : 1;
        }
    }

    private static Tensor<float> Vector(int n, int salt)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = (float)rng.NextDouble();
        return t;
    }
}
