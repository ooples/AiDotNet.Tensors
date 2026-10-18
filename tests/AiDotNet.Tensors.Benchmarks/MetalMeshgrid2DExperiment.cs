// Copyright (c) AiDotNet. All rights reserved.

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
/// overheads. Paired host timing; a shape counts as promoted only on a stable speedup of at least 1.10x, and an
/// unstable comparison is reported as unstable rather than as a number. Exit 0 = every cell promoted, 1 = some cell
/// did not clear the gate, 2 = no Metal device.
/// </summary>
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
            Console.WriteLine("shape        mode  composed       fused          speedup  spread  verdict");
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
                    var pair = StableTimer.MeasureCalibratedHostPair(Composed, backend.Synchronize, Fused, backend.Synchronize);
                    bool promoted = pair.Stable && pair.Ratio >= PromotionSpeedup;
                    allPromoted &= promoted;
                    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-12} {1,-5} {2,-14} {3,-14} {4,-8} {5,-7} {6}",
                        n0 + "x" + n1, indexing, pair.A.Describe(), pair.B.Describe(),
                        pair.Ratio.ToString("0.00", CultureInfo.InvariantCulture) + "x",
                        (pair.RelativeSpread * 100).ToString("0.0", CultureInfo.InvariantCulture) + "%",
                        !pair.Stable ? "UNSTABLE (no verdict)" : promoted ? "promote" : "keep composed"));
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
