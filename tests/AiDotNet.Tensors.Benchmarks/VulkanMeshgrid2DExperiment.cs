// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #923: times the Vulkan fused Meshgrid2D route against the composed reshape-plus-broadcast route it replaces, through
/// the engine (so both sides pay the same allocation and dispatch overheads), with paired host timing. A shape is
/// promoted only on a stable speedup of at least 1.10x; an unstable comparison is reported as unstable, not as a number.
/// </summary>
internal static class VulkanMeshgrid2DExperiment
{
    private const double PromotionSpeedup = 1.10;

    public static int Run()
    {
        VulkanBackend backend;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable)
            {
                Console.WriteLine("Vulkan Meshgrid2D: no Vulkan device with the runtime GLSL compiler; nothing measured.");
                return 2;
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            Console.WriteLine("Vulkan Meshgrid2D: Vulkan unavailable (" + ex.GetType().Name + "); nothing measured.");
            return 2;
        }

        // Non-owning: the backend is a process-wide singleton.
        var engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
        Console.WriteLine("Vulkan Meshgrid2D fused vs composed on " + backend.DeviceName);
        Console.WriteLine("shape        mode  composed       fused          speedup  spread  verdict");
        bool allPromoted = true;
        try
        {
            foreach (var (n0, n1) in new[] { (256, 256), (1024, 256) })
                foreach (string indexing in new[] { "ij", "xy" })
                {
                    var x = Vector(n0, 1);
                    var y = Vector(n1, 2);
                    var inputs = new[] { x, y };
                    void Composed()
                    {
                        VulkanBackend.Meshgrid2DFusedEnabled = false;
                        var grids = engine.TensorMeshgrid(inputs, indexing);
                        grids[0].Dispose();
                        grids[1].Dispose();
                    }
                    void Fused()
                    {
                        VulkanBackend.Meshgrid2DFusedEnabled = true;
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
        }
        finally
        {
            VulkanBackend.Meshgrid2DFusedEnabled = true;
        }
        return allPromoted ? 0 : 1;
    }

    private static Tensor<float> Vector(int n, int salt)
    {
        var rng = new Random(salt);
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = (float)rng.NextDouble();
        return t;
    }
}
