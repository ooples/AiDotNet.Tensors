using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="GpuOptimizer.TryClipGradNorm"/>: torch.nn.utils.clip_grad_norm_ on device-resident gradients, with no
/// device-to-host transfer until the norm is read.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeviceClipGradNormTests
{
    private static float[] Rand(int n, int seed, double scale)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(scale * (2 * r.NextDouble() - 1))).ToArray();
    }

    [SkippableTheory]
    [InlineData(0.5)]    // clips
    [InlineData(1e6)]    // under the bound: unchanged
    public void ClipsDeviceGradients_LikeClipGradNorm_WithoutReadback(double maxNorm)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            int[] sizes = { 1000, 37, 4096, 1 };
            var host = sizes.Select((n, i) => Rand(n, 10 + i, 0.3)).ToArray();
            // Device-resident gradients: GPU op results with a pending host download, like tape gradients.
            var grads = host.Select(h =>
            {
                var t = new Tensor<float>(new[] { h.Length }, new Vector<float>(h));
                return gpu!.TensorMultiplyScalar(t, 1f);
            }).ToList();

            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            DeviceGradientNorm? norm;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;
                GpuLaunchProbe.Reset();
                norm = GpuOptimizer.TryClipGradNorm(grads, maxNorm);
                Assert.True(GpuLaunchProbe.Readbacks == 0,
                    $"clip performed {GpuLaunchProbe.Readbacks} device-to-host transfers: " + string.Join("; ", GpuLaunchProbe.ReadbackSites));
            }
            finally
            {
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            }
            Assert.NotNull(norm);
            using var _n = norm;

            double sq = host.Sum(h => h.Sum(v => (double)v * v));
            double total = Math.Sqrt(sq);
            double coef = Math.Min(1.0, maxNorm / (total + 1e-6));
            Assert.Equal(total, norm!.TotalNorm, 3);
            Assert.True(norm.IsFinite);
            for (int t = 0; t < grads.Count; t++)
            {
                var got = grads[t].ToArray();
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - host[t][i] * coef) <= 1e-5 + 1e-5 * Math.Abs(host[t][i]),
                        $"tensor {t} element {i}: {got[i]} vs {host[t][i] * coef}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }

    [SkippableFact]
    public void NonFiniteGradient_LeavesGradientsUnscaled_AndReportsIt()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            var a = Rand(300, 1, 5);
            var b = Rand(200, 2, 5);
            b[17] = float.NaN;
            var grads = new[] { a, b }
                .Select(h => gpu!.TensorMultiplyScalar(new Tensor<float>(new[] { h.Length }, new Vector<float>(h)), 1f))
                .ToList();
            using var norm = GpuOptimizer.TryClipGradNorm(grads, 0.1);
            Assert.NotNull(norm);
            Assert.False(norm!.IsFinite);
            Assert.Equal(a, grads[0].ToArray());
            var gotB = grads[1].ToArray();
            for (int i = 0; i < b.Length; i++)
                if (i == 17) Assert.True(float.IsNaN(gotB[i]));
                else Assert.Equal(b[i], gotB[i]);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }

    /// <summary>
    /// A total norm past the float range (squares of 3e19 overflow a float sum) still clips. The float-accumulating
    /// backends narrowed the sum to an infinite float and left such gradients unscaled -- an exploding gradient passed
    /// the clip -- and Vulkan reported them as non-finite. Every element is finite, so the norm is reported finite.
    /// </summary>
    [SkippableFact]
    public void NormPastTheFloatRange_IsStillClipped_AndReportedFinite()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            var a = Rand(300, 3, 1);
            a[5] = 3e19f; a[200] = -2.5e19f;
            var b = Rand(64, 4, 1);
            b[0] = 3e19f;
            var grads = new[] { a, b }
                .Select(h => gpu!.TensorMultiplyScalar(new Tensor<float>(new[] { h.Length }, new Vector<float>(h)), 1f))
                .ToList();
            const double maxNorm = 1.0;
            using var norm = GpuOptimizer.TryClipGradNorm(grads, maxNorm);
            Assert.NotNull(norm);
            Assert.True(norm!.IsFinite, $"finite gradients reported non-finite (sum of squares {norm.TotalNormSquared})");
            double after = Math.Sqrt(grads.Sum(g => g.ToArray().Sum(v => (double)v * v)));
            Assert.True(grads.All(g => g.ToArray().All(v => !float.IsNaN(v) && !float.IsInfinity(v))), "clipped gradients not finite");
            // The clip scales to the bound, not merely under it: a coefficient that underflowed to zero erases the
            // gradients and would pass an upper bound alone.
            Assert.True(Math.Abs(after - maxNorm) <= maxNorm * 1e-3, $"norm after the clip {after:E3}, expected {maxNorm}");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }

    [SkippableFact]
    public void BoundPastTheFloatRange_ClipsOnTheHost()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu!;
        try
        {
            var g = gpu!.TensorMultiplyScalar(new Tensor<float>(new[] { 4 }, new Vector<float>(new float[] { 1, 2, 3, 4 })), 1f);
            // Narrowed to a float the bound is infinite and the coefficient 1, whatever the norm.
            Assert.Null(GpuOptimizer.TryClipGradNorm(new[] { g }, 1e39));
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu!.Dispose();
        }
    }

    [Fact]
    public void CpuEngine_ReturnsNull_SoTheCallerClipsOnTheHost()
    {
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var g = new Tensor<float>(new[] { 4 }, new Vector<float>(new float[] { 1, 2, 3, 4 }));
            Assert.Null(GpuOptimizer.TryClipGradNorm(new[] { g }, 1.0));
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
