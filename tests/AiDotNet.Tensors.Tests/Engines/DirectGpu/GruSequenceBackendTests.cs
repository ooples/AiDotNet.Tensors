#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// IDirectGpuBackend.GruForwardSequence / GruBackwardSequence against a double-precision CPU GRU (PyTorch layout, gates
/// r, z, n) and its central-difference gradients. The kernels had been written for separate per-gate matrices the
/// backend never passed, so the packed arguments filled the wrong parameters; nothing exercised the API.
/// </summary>
[Collection("DirectGpuSerial")]
public class GruSequenceBackendTests
{
    private const int T = 4, B = 3, I = 5, H = 6;

    private sealed class Params
    {
        public double[] X = new double[T * B * I], H0 = new double[B * H];
        public double[] Wih = new double[3 * H * I], Whh = new double[3 * H * H], Bih = new double[3 * H], Bhh = new double[3 * H];
    }

    private static double Sig(double v) => 1 / (1 + Math.Exp(-v));

    // Returns output [T, B, H].
    private static double[] Forward(Params p)
    {
        var output = new double[T * B * H];
        var h = (double[])p.H0.Clone();
        for (int t = 0; t < T; t++)
        {
            var next = new double[B * H];
            for (int b = 0; b < B; b++)
                for (int j = 0; j < H; j++)
                {
                    double xr = p.Bih[j], xz = p.Bih[H + j], xn = p.Bih[2 * H + j];
                    for (int i = 0; i < I; i++)
                    {
                        double xi = p.X[(t * B + b) * I + i];
                        xr += p.Wih[j * I + i] * xi; xz += p.Wih[(H + j) * I + i] * xi; xn += p.Wih[(2 * H + j) * I + i] * xi;
                    }
                    double hr = p.Bhh[j], hz = p.Bhh[H + j], hn = p.Bhh[2 * H + j];
                    for (int k = 0; k < H; k++)
                    {
                        double hk = h[b * H + k];
                        hr += p.Whh[j * H + k] * hk; hz += p.Whh[(H + j) * H + k] * hk; hn += p.Whh[(2 * H + j) * H + k] * hk;
                    }
                    double r = Sig(xr + hr), z = Sig(xz + hz), n = Math.Tanh(xn + r * hn);
                    next[b * H + j] = (1 - z) * n + z * h[b * H + j];
                }
            h = next;
            Array.Copy(h, 0, output, t * B * H, B * H);
        }
        return output;
    }

    private static double Loss(Params p, double[] g)
    {
        var o = Forward(p);
        double l = 0;
        for (int i = 0; i < o.Length; i++) l += o[i] * g[i];
        return l;
    }

    private static double[] NumericGrad(Params p, double[] g, double[] target)
    {
        var grad = new double[target.Length];
        const double eps = 1e-5;
        for (int i = 0; i < target.Length; i++)
        {
            double keep = target[i];
            target[i] = keep + eps; double up = Loss(p, g);
            target[i] = keep - eps; double down = Loss(p, g);
            target[i] = keep;
            grad[i] = (up - down) / (2 * eps);
        }
        return grad;
    }

    private static float[] F(double[] d) => Array.ConvertAll(d, v => (float)v);

    private static void Close(double[] expected, float[] actual, double tol, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:G6}, got {actual[i]:G6}.");
    }

    [SkippableFact]
    public void ForwardAndBackwardSequence_MatchCpuGruAndNumericGradients()
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No DirectGpu backend available.");
        using (engine)
        {
            var backend = engine!.GetBackend()!;
            var rng = new Random(11);
            var p = new Params();
            foreach (var arr in new[] { p.X, p.H0, p.Wih, p.Whh, p.Bih, p.Bhh })
                for (int i = 0; i < arr.Length; i++) arr[i] = rng.NextDouble() - 0.5;
            var g = new double[T * B * H];
            for (int i = 0; i < g.Length; i++) g[i] = rng.NextDouble() * 2 - 1;

            using var x = backend.AllocateBuffer(F(p.X));
            using var h0 = backend.AllocateBuffer(F(p.H0));
            using var wih = backend.AllocateBuffer(F(p.Wih));
            using var whh = backend.AllocateBuffer(F(p.Whh));
            using var bih = backend.AllocateBuffer(F(p.Bih));
            using var bhh = backend.AllocateBuffer(F(p.Bhh));
            using var output = backend.AllocateBuffer(T * B * H);
            using var hFinal = backend.AllocateBuffer(B * H);
            using var allH = backend.AllocateBuffer((T + 1) * B * H);
            using var gates = backend.AllocateBuffer(T * B * H * 3);
            backend.GruForwardSequence(x, h0, wih, whh, bih, bhh, output, hFinal, allH, gates, T, B, I, H);

            var expectedOut = Forward(p);
            Close(expectedOut, backend.DownloadBuffer(output), 1e-4, "output");
            var expectedFinal = new double[B * H];
            Array.Copy(expectedOut, (T - 1) * B * H, expectedFinal, 0, B * H);
            Close(expectedFinal, backend.DownloadBuffer(hFinal), 1e-4, "hFinal");

            using var gOut = backend.AllocateBuffer(F(g));
            using var gIn = backend.AllocateBuffer(T * B * I);
            using var gH0 = backend.AllocateBuffer(B * H);
            using var dHBuf = backend.AllocateBuffer(B * H);
            using var gWih = backend.AllocateBuffer(3 * H * I);
            using var gWhh = backend.AllocateBuffer(3 * H * H);
            using var gBih = backend.AllocateBuffer(3 * H);
            using var gBhh = backend.AllocateBuffer(3 * H);
            backend.GruBackwardSequence(gOut, allH, gates, wih, whh, x, gIn, gH0, dHBuf, gWih, gWhh, gBih, gBhh, T, B, I, H);

            Close(NumericGrad(p, g, p.X), backend.DownloadBuffer(gIn), 2e-3, "gradInput");
            Close(NumericGrad(p, g, p.H0), backend.DownloadBuffer(gH0), 2e-3, "gradHInit");
            Close(NumericGrad(p, g, p.Wih), backend.DownloadBuffer(gWih), 2e-3, "gradWeightsIh");
            Close(NumericGrad(p, g, p.Whh), backend.DownloadBuffer(gWhh), 2e-3, "gradWeightsHh");
            Close(NumericGrad(p, g, p.Bih), backend.DownloadBuffer(gBih), 2e-3, "gradBiasIh");
            Close(NumericGrad(p, g, p.Bhh), backend.DownloadBuffer(gBhh), 2e-3, "gradBiasHh");
        }
    }
}
#endif
