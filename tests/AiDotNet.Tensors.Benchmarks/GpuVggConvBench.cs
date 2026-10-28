using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// VGG-16's thirteen 3x3 convolutions at batch 1, 224x224, on the GPU engine (#1432): each layer under the tuned slot's
/// own choice and under every forced forward candidate, warm, synchronised per call. Reports milliseconds and effective
/// TFLOP/s (2 * N * K * C * 9 * P * Q over the time), so a new route can be judged per layer against the others.
/// Usage: --gpu-vgg-convs [candidate-id ...]  (no ids: the tuned choice plus every OpenCL forward candidate)
/// </summary>
internal static class GpuVggConvBench
{
    private static readonly (int C, int K, int Hw)[] Layers =
    {
        (3, 64, 224), (64, 64, 224),
        (64, 128, 112), (128, 128, 112),
        (128, 256, 56), (256, 256, 56), (256, 256, 56),
        (256, 512, 28), (512, 512, 28), (512, 512, 28),
        (512, 512, 14), (512, 512, 14), (512, 512, 14),
    };

    private static readonly string[] DefaultCandidates =
    {
        "opencl.conv2d.winograd_gemm", "opencl.conv2d.winograd_f43_gemm", "opencl.conv2d.winograd_fused",
        "opencl.conv2d.im2col_rb_gemm", "opencl.conv2d.direct",
    };

    public static void Run(string[] candidates)
    {
        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable) { Console.WriteLine("No DirectGpu backend."); return; }
        var backend = gpu.GetBackend();
        Console.WriteLine($"backend {backend?.GetType().Name}");
        var routes = new string?[candidates.Length == 0 ? DefaultCandidates.Length + 1 : candidates.Length + 1];
        routes[0] = null;
        var ids = candidates.Length == 0 ? DefaultCandidates : candidates;
        for (int i = 0; i < ids.Length; i++) routes[i + 1] = ids[i];

        var rng = new Random(16);
        Tensor<float> Rand(params int[] shape)
        {
            int n = 1; foreach (var d in shape) n *= d;
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
            return new Tensor<float>(a, shape);
        }

        Console.WriteLine($"{"layer",-18}" + string.Concat(Array.ConvertAll(routes, r => $"{Short(r),16}")));
        double[] totals = new double[routes.Length];
        foreach (var (c, k, hw) in Layers)
        {
            var input = gpu.TensorAddScalar(Rand(1, c, hw, hw), 0f);   // resident input, as inside a forward
            var kernel = Rand(k, c, 3, 3);
            var bias = Rand(k);
            double flops = 2.0 * k * c * 9 * hw * hw;
            string row = $"{$"{c}->{k} @{hw}",-18}";
            for (int r = 0; r < routes.Length; r++)
            {
                TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DForward, routes[r]);
                double ms;
                try
                {
                    for (int w = 0; w < 5; w++) gpu.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
                    backend?.Synchronize();
                    var times = new double[15];
                    for (int i = 0; i < times.Length; i++)
                    {
                        var sw = Stopwatch.StartNew();
                        gpu.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
                        backend?.Synchronize();
                        times[i] = sw.Elapsed.TotalMilliseconds;
                    }
                    Array.Sort(times);
                    ms = times[times.Length / 2];
                }
                catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
                {
                    ms = double.NaN;
                }
                totals[r] += double.IsNaN(ms) ? 0 : ms;
                row += double.IsNaN(ms) ? $"{"n/a",16}" : $"{$"{ms:F3} ({flops / ms / 1e9:F1})",16}";
            }
            Console.WriteLine(row);
        }
        TunedKernelPolicy.SetOverride(TunedKernelOp.Conv2DForward, null);
        Console.WriteLine($"{"total ms",-18}" + string.Concat(Array.ConvertAll(totals, t => $"{t,16:F3}")));
        Console.WriteLine("cells: median ms (effective TFLOP/s); n/a = the route declined or failed the shape");
    }

    private static string Short(string? id) => id is null ? "tuned" : id.Substring(id.LastIndexOf('.') + 1);
}
