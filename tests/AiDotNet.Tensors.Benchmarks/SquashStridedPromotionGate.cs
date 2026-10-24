// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The promotion gate for the routes #1543 lands off by default: the capsule squash and its backward in direct PTX
/// (#833) and the declared strided gather and scatter (#1441), each against the NVRTC kernel it would replace, through
/// CudaBackend's own public entries. A cell is promoted when the 95% distribution-free interval of the median paired
/// speedup (StableTimer.MeasurePairedRatioBound, 41 AB/BA brackets) lies wholly at or above 1.10x. Exit 0 = every cell
/// promoted, 1 = some cell did not clear the gate, 2 = no CUDA device. Correctness is the driver tests' job
/// (DirectPtxSquashTests, DeclaredStridedRouteTests); run them first.
/// </summary>
internal static class SquashStridedPromotionGate
{
    private const double PromotionSpeedup = 1.10;

    public static int Run()
    {
        if (!CudaBackend.IsCudaAvailable)
        {
            Console.WriteLine("Squash/strided promotion gate: no CUDA device; nothing measured.");
            return 2;
        }
        using var backend = CudaBackend.CreateOrThrow();
        Console.WriteLine("Squash/strided promotion gate on " + backend.DeviceName);
        Console.WriteLine("route              shape            median   95% interval        verdict");
        bool allPromoted = true;
        var random = RandomHelper.CreateSeededRandom(1543);

        bool? savedTest = DirectPtxFeatureGate.TestOverride;
        bool savedExperiment = DirectPtxFeatureGate.CapsuleExperimentOverride;
        try
        {
            DirectPtxFeatureGate.CapsuleExperimentOverride = true;
            foreach ((int capsules, int width) in new[] { (1152, 8), (9216, 8), (1280, 16), (10240, 16) })
            {
                int elements = capsules * width;
                using var x = backend.AllocateBuffer(Values(elements, random));
                using var g = backend.AllocateBuffer(Values(elements, random));
                using var y = backend.AllocateBuffer(elements);
                float eps = PtxFusedSquashF32Kernel.Epsilon;
                // A = NVRTC (direct PTX off), B = direct PTX: a ratio above 1 means the PTX is faster.
                allPromoted &= Report(backend, "squash", capsules + "x" + width,
                    () => { DirectPtxFeatureGate.TestOverride = false; backend.Squash(x, y, capsules, width, eps); },
                    () => { DirectPtxFeatureGate.TestOverride = true; backend.Squash(x, y, capsules, width, eps); },
                    () => backend.DirectPtxSquashDispatchCount);
                allPromoted &= Report(backend, "squash-backward", capsules + "x" + width,
                    () => { DirectPtxFeatureGate.TestOverride = false; backend.SquashBackward(g, x, y, capsules, width, eps); },
                    () => { DirectPtxFeatureGate.TestOverride = true; backend.SquashBackward(g, x, y, capsules, width, eps); },
                    () => backend.DirectPtxSquashDispatchCount);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = savedTest;
            DirectPtxFeatureGate.CapsuleExperimentOverride = savedExperiment;
        }

        bool? savedStrided = backend.DeclaredStridedSwitch;
        try
        {
            foreach ((int count, int stride) in new[] { (65536, 3), (262144, 7), (1048576, 2) })
            {
                int length = (count - 1) * stride + 1;
                using var wide = backend.AllocateBuffer(Values(length, random));
                using var narrow = backend.AllocateBuffer(Values(count, random));
                using var dst = backend.AllocateBuffer(count);
                using var scattered = backend.AllocateBuffer(length);
                allPromoted &= Report(backend, "strided-gather", count + "/" + stride,
                    () => { backend.DeclaredStridedSwitch = false; backend.StridedGather(wide, dst, 0, stride, count); },
                    () => { backend.DeclaredStridedSwitch = true; backend.StridedGather(wide, dst, 0, stride, count); },
                    () => backend.DeclaredProgramLaunches);
                allPromoted &= Report(backend, "strided-scatter", count + "/" + stride,
                    () => { backend.DeclaredStridedSwitch = false; backend.StridedScatter(narrow, scattered, 0, stride, count); },
                    () => { backend.DeclaredStridedSwitch = true; backend.StridedScatter(narrow, scattered, 0, stride, count); },
                    () => backend.DeclaredProgramLaunches);
            }
        }
        finally
        {
            backend.DeclaredStridedSwitch = savedStrided;
        }
        return allPromoted ? 0 : 1;
    }

    // One cell: confirms the candidate route really dispatched (a silent fallback would time NVRTC against itself),
    // then bounds the median speedup.
    private static bool Report(CudaBackend backend, string route, string shape, Action established, Action candidate,
        Func<long> candidateDispatches)
    {
        long before = candidateDispatches();
        candidate();
        backend.Synchronize();
        if (candidateDispatches() == before)
        {
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-18} {1,-16} the candidate did not dispatch: {2}",
                route, shape, backend.DirectPtxLastError ?? "declared route declined"));
            return false;
        }
        var bound = StableTimer.MeasurePairedRatioBound(established, backend.Synchronize, candidate, backend.Synchronize);
        bool promoted = bound.Lower >= PromotionSpeedup;
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-18} {1,-16} {2,-8} [{3:0.00}x, {4:0.00}x]    {5}",
            route, shape, bound.Median.ToString("0.00", CultureInfo.InvariantCulture) + "x", bound.Lower, bound.Upper,
            promoted ? "promote" : bound.Upper < PromotionSpeedup ? "keep established" : "undecided"));
        return promoted;
    }

    private static float[] Values(int n, Random random)
    {
        var values = new float[n];
        for (int i = 0; i < n; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }
}
