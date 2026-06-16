// Parity tests for the fully GPU-resident global-L2 gradient clip used by CompiledTrainingPlan
// (TryClipGradientsGlobalL2Gpu). The clip composes ReduceSumOfSquares (accumulate) + Sqrt + AddScalar +
// Reciprocal + Scale + Min + the scale_by_device_scalar kernel — entirely on-device, no host read — so
// grad-norm clipping can stay ON without disabling CUDA-graph capture. This test replays that exact
// composition on the live CUDA backend and checks it matches a CPU clip_grad_norm reference, in both the
// clipping-active (norm>maxNorm) and no-op (norm<=maxNorm) regimes. Runs only when a GPU is present.

#if !NETFRAMEWORK
#nullable disable

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuGradientClipParityTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _gpuReady;

    public GpuGradientClipParityTests()
    {
        try { _gpu = new DirectGpuTensorEngine(); _gpuReady = _gpu.IsGpuAvailable; }
        catch { _gpuReady = false; }
    }

    public void Dispose() => _gpu?.Dispose();

    private CudaBackend GetCuda()
    {
        if (!_gpuReady) return null;
        return _gpu.GetBackend() as CudaBackend;   // null on non-CUDA GPUs → test no-ops
    }

    private static float[][] MakeGrads(int seed, int[] sizes, double mag)
    {
        var rng = new Random(seed);
        var g = new float[sizes.Length][];
        for (int p = 0; p < sizes.Length; p++)
        {
            g[p] = new float[sizes[p]];
            for (int i = 0; i < sizes[p]; i++) g[p][i] = (float)((rng.NextDouble() * 2.0 - 1.0) * mag);
        }
        return g;
    }

    // CPU reference: PyTorch clip_grad_norm_ — one global L2 norm across ALL grads, scale = min(1, maxNorm/(norm+1e-6)).
    private static float[][] CpuClip(float[][] grads, double maxNorm)
    {
        double sumSq = 0.0;
        foreach (var g in grads) foreach (var v in g) sumSq += (double)v * v;
        double norm = Math.Sqrt(sumSq);
        double scale = Math.Min(1.0, maxNorm / (norm + 1e-6));
        var outp = new float[grads.Length][];
        for (int p = 0; p < grads.Length; p++)
        {
            outp[p] = new float[grads[p].Length];
            for (int i = 0; i < grads[p].Length; i++) outp[p][i] = (float)(grads[p][i] * scale);
        }
        return outp;
    }

    // Replays TryClipGradientsGlobalL2Gpu's exact on-device composition on the CUDA backend.
    private static float[][] GpuClip(CudaBackend cb, float[][] grads, double maxNorm)
    {
        var bufs = new IGpuBuffer[grads.Length];
        for (int p = 0; p < grads.Length; p++) bufs[p] = cb.AllocateBuffer(grads[p]);
        var sumSq = cb.AllocateBuffer(1);
        var tmp = cb.AllocateBuffer(1);
        try
        {
            cb.Fill(sumSq, 0f, 1);
            for (int p = 0; p < grads.Length; p++)
            {
                cb.Fill(tmp, 0f, 1);
                cb.ReduceSumOfSquares(bufs[p], tmp, grads[p].Length);
                cb.Add(sumSq, tmp, sumSq, 1);
            }
            cb.Sqrt(sumSq, sumSq, 1);
            cb.AddScalar(sumSq, sumSq, 1e-6f, 1);
            cb.Reciprocal(sumSq, sumSq, 1);
            cb.Scale(sumSq, sumSq, (float)maxNorm, 1);
            cb.Fill(tmp, 1f, 1);
            cb.Min(sumSq, tmp, sumSq, 1);
            for (int p = 0; p < grads.Length; p++)
                cb.ScaleByDeviceScalar(bufs[p], sumSq, grads[p].Length);

            var outp = new float[grads.Length][];
            for (int p = 0; p < grads.Length; p++)
            {
                outp[p] = new float[grads[p].Length];
                cb.DownloadBuffer(bufs[p], outp[p]);
            }
            return outp;
        }
        finally
        {
            (sumSq as IDisposable)?.Dispose();
            (tmp as IDisposable)?.Dispose();
            foreach (var b in bufs) (b as IDisposable)?.Dispose();
        }
    }

    private static void AssertMatch(float[][] gpu, float[][] cpu)
    {
        double maxErr = 0;
        for (int p = 0; p < cpu.Length; p++)
            for (int i = 0; i < cpu[p].Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(gpu[p][i] - cpu[p][i]));
        Assert.True(maxErr < 1e-4, $"GPU vs CPU clip max_abs_err {maxErr:E3} exceeded 1e-4");
    }

    [Fact]
    public void GpuClip_Matches_Cpu_WhenClippingActive()
    {
        var cb = GetCuda();
        if (cb is null) return;
        var sizes = new[] { 589824, 768, 3072, 50 };   // mixed sizes incl. a non-pow2-block tail
        var grads = MakeGrads(11, sizes, mag: 2.0);     // big values → norm >> maxNorm → real clipping
        const double maxNorm = 1.0;
        AssertMatch(GpuClip(cb, grads, maxNorm), CpuClip(grads, maxNorm));
    }

    [Fact]
    public void GpuClip_IsNoOp_WhenNormBelowMax()
    {
        var cb = GetCuda();
        if (cb is null) return;
        var sizes = new[] { 1024, 256, 17 };
        var grads = MakeGrads(7, sizes, mag: 1e-3);     // tiny values → norm << maxNorm → scale==1 (no-op)
        const double maxNorm = 100.0;
        AssertMatch(GpuClip(cb, grads, maxNorm), CpuClip(grads, maxNorm));
    }
}
#endif
