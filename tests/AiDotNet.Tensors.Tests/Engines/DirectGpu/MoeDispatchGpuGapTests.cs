using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Two GPU gaps hit by mixture-of-experts dispatch: CUDA BatchedFFT rejecting a batch past gridDim.y's 65535, and
/// IndexSelect's backward running as a host loop on the GPU engine.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MoeDispatchGpuGapTests
{
    [SkippableFact]
    public void BatchedFft_BatchPastGridDimYLimit_MatchesHostDft()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend failed to initialize.");

        const int n = 8;
        const int batch = 65535 + 3;   // one full grid slice plus a partial one
        var rng = new Random(5);
        var re = Enumerable.Range(0, batch * n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var im = Enumerable.Range(0, batch * n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        using var inRe = backend.AllocateBuffer(re);
        using var inIm = backend.AllocateBuffer(im);
        using var outRe = backend.AllocateBuffer(batch * n);
        using var outIm = backend.AllocateBuffer(batch * n);
        using var backRe = backend.AllocateBuffer(batch * n);
        using var backIm = backend.AllocateBuffer(batch * n);

        backend.BatchedFFT(inRe, inIm, outRe, outIm, batch, n, inverse: false);
        backend.BatchedFFT(outRe, outIm, backRe, backIm, batch, n, inverse: true);
        backend.Synchronize();
        var fr = backend.DownloadBuffer(outRe);
        var fi = backend.DownloadBuffer(outIm);
        var br = backend.DownloadBuffer(backRe);

        // Rows on both sides of every slice boundary, and the last row.
        foreach (int row in new[] { 0, 65534, 65535, 65536, batch - 1 })
        {
            for (int k = 0; k < n; k++)
            {
                double sr = 0, si = 0;
                for (int t = 0; t < n; t++)
                {
                    double ang = -2 * Math.PI * k * t / n, xr = re[row * n + t], xi = im[row * n + t];
                    sr += xr * Math.Cos(ang) - xi * Math.Sin(ang);
                    si += xr * Math.Sin(ang) + xi * Math.Cos(ang);
                }

                Assert.True(Math.Abs(fr[row * n + k] - sr) < 1e-4, $"row {row} bin {k}: re {fr[row * n + k]} vs {sr}");
                Assert.True(Math.Abs(fi[row * n + k] - si) < 1e-4, $"row {row} bin {k}: im {fi[row * n + k]} vs {si}");
            }

            for (int t = 0; t < n; t++)
                Assert.True(Math.Abs(br[row * n + t] - re[row * n + t]) < 1e-5, $"row {row} round trip at {t}");
        }
    }

    [SkippableTheory]
    [InlineData(64, 16, 200)]    // repeated indices (scatter-add must accumulate)
    [InlineData(1000, 512, 300)]
    public void IndexSelectBackward_LeadingAxis_GpuMatchesCpu(int rows, int cols, int picks)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "GPU engine unavailable."); }
        using var _ = gpu;
        Skip.IfNot(gpu!.IsGpuAvailable, "GPU not available.");

        var rng = new Random(rows + picks);
        var xData = Enumerable.Range(0, rows * cols).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var wData = Enumerable.Range(0, picks * cols).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var idx = Enumerable.Range(0, picks).Select(_ => rng.Next(rows)).ToArray();

        float[] Grad(IEngine eng)
        {
            var x = new Tensor<float>(new[] { rows, cols }, new Vector<float>(xData));
            var w = new Tensor<float>(new[] { picks, cols }, new Vector<float>(wData));
            using var tape = new GradientTape<float>();
            var y = eng.TensorIndexSelectDiff(x, new Tensor<int>(idx, new[] { picks }), 0);
            var loss = eng.ReduceSum(eng.TensorMultiply(y, w), new[] { 0, 1 }, keepDims: false);
            return tape.ComputeGradients(loss, new[] { x })[x].ToArray();
        }

        // Residency: the backward of a leading-axis select must scatter on the device. The host loop it replaces
        // downloaded the upstream gradient (one readback per call) and walked it element by element.
        {
            var a = new Tensor<float>(new[] { picks, cols }, new Vector<float>(wData));
            var gradOutput = gpu.TensorMultiply(a, a);            // device-resident upstream gradient
            var source = new Tensor<float>(new[] { rows, cols }, new Vector<float>(xData));
            var saved = new object[] { new Tensor<int>(idx, new[] { picks }), 0 };
            bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
            try
            {
                GpuLaunchProbe.CaptureReadbackSites = true;
                var warm = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
                BackwardFunctions<float>.IndexSelectBackward(gradOutput, new[] { source }, gradOutput, saved, gpu, warm);
                // A fresh device-resident gradient: the warm-up may have made the first one host-current.
                var fresh = gpu.TensorMultiply(a, a);
                GpuLaunchProbe.Reset();
                var grads = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
                BackwardFunctions<float>.IndexSelectBackward(fresh, new[] { source }, fresh, saved, gpu, grads);
                long readbacks = GpuLaunchProbe.Readbacks;
                Assert.True(readbacks == 0, $"IndexSelect backward performed {readbacks} device-to-host transfers: " +
                    string.Join("; ", GpuLaunchProbe.ReadbackSites));
                var g = grads[source].ToArray();
                var sq = wData.Select(v => v * v).ToArray();
                var want = new float[rows * cols];
                for (int i = 0; i < picks; i++)
                    for (int c = 0; c < cols; c++) want[idx[i] * cols + c] += sq[i * cols + c];
                for (int j = 0; j < want.Length; j++)
                    Assert.True(Math.Abs(g[j] - want[j]) < 1e-4, $"resident grad {j}: {g[j]} vs {want[j]}");
            }
            finally
            {
                GpuLaunchProbe.CaptureReadbackSites = savedCapture;
            }
        }

        var expected = Grad(new CpuEngine());
        var actual = Grad(gpu);
        // Independent reference: d x[idx[i], :] += w[i, :].
        var manual = new float[rows * cols];
        for (int i = 0; i < picks; i++)
            for (int c = 0; c < cols; c++) manual[idx[i] * cols + c] += wData[i * cols + c];
        for (int j = 0; j < manual.Length; j++)
        {
            Assert.True(Math.Abs(expected[j] - manual[j]) < 1e-4, $"cpu grad {j}: {expected[j]} vs {manual[j]}");
            Assert.True(Math.Abs(actual[j] - manual[j]) < 1e-4, $"gpu grad {j}: {actual[j]} vs {manual[j]}");
        }
    }
}
