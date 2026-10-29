#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The shared gru_p_ / gru_rb_ kernels (#1568) on each backend, through the launcher the engine uses
/// (<see cref="IFusedLstmSequenceTraining.StatefulLstmLauncher"/>), against a CPU forward and BPTT in double: nn.GRU's
/// gate order (r, z, n), a nonzero h0, the gradients of the input, h0, both weights and both biases, and both
/// placements of the reset gate (after the recurrent sum, PyTorch; before the recurrent product, Cho et al. / Keras
/// reset_after=False).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GruSequenceBackendParityTests
{
    private static readonly (int B, int T, int In, int H)[] Shapes =
    {
        (1, 1, 1, 1),
        (2, 3, 4, 5),
        (5, 3, 6, 64),    // B*H spans several work-groups
        (64, 8, 32, 64),  // the parity GRU's batch and widths
        (2, 3, 5, 300),   // H wider than a 256-wide work-group
    };

    private static float[] Values(int n, int seed, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((((i + seed) * 7919) % 41) - 20) * scale;
        return v;
    }

    private sealed record Expected(double[] Y, double[] GIn, double[] GH0, double[] GWih, double[] GWhh, double[] GBih, double[] GBhh);

    private static Expected Reference(float[] x, float[] h0, float[] wih, float[] whh, float[] bih, float[] bhh, float[] gy,
        int bN, int tN, int inN, int hN, bool resetAfter)
    {
        var y = new double[bN * tN * hN];
        var hs = new double[bN, tN + 1, hN];
        var r = new double[bN, tN, hN]; var z = new double[bN, tN, hN]; var n = new double[bN, tN, hN];
        var hn = new double[bN, tN, hN]; var rh = new double[bN, tN, hN];
        for (int b = 0; b < bN; b++)
        {
            for (int h = 0; h < hN; h++) hs[b, 0, h] = h0[b * hN + h];
            for (int t = 0; t < tN; t++)
            {
                double In(int row) { double s = bih[row]; for (int i = 0; i < inN; i++) s += (double)wih[row * inN + i] * x[(b * tN + t) * inN + i]; return s; }
                for (int h = 0; h < hN; h++)
                {
                    double sr = bhh[h], sz = bhh[hN + h];
                    for (int j = 0; j < hN; j++) { sr += (double)whh[h * hN + j] * hs[b, t, j]; sz += (double)whh[(hN + h) * hN + j] * hs[b, t, j]; }
                    r[b, t, h] = 1 / (1 + Math.Exp(-(In(h) + sr)));
                    z[b, t, h] = 1 / (1 + Math.Exp(-(In(hN + h) + sz)));
                    rh[b, t, h] = r[b, t, h] * hs[b, t, h];
                }
                for (int h = 0; h < hN; h++)
                {
                    double s = bhh[2 * hN + h];
                    for (int j = 0; j < hN; j++) s += (double)whh[(2 * hN + h) * hN + j] * (resetAfter ? hs[b, t, j] : rh[b, t, j]);
                    hn[b, t, h] = s;
                    n[b, t, h] = Math.Tanh(In(2 * hN + h) + (resetAfter ? r[b, t, h] * s : s));
                    y[(b * tN + t) * hN + h] = (1 - z[b, t, h]) * n[b, t, h] + z[b, t, h] * hs[b, t, h];
                }
                for (int h = 0; h < hN; h++) hs[b, t + 1, h] = y[(b * tN + t) * hN + h];
            }
        }

        var gIn = new double[bN * tN * inN]; var gH0 = new double[bN * hN];
        var gWih = new double[3 * hN * inN]; var gWhh = new double[3 * hN * hN];
        var gBih = new double[3 * hN]; var gBhh = new double[3 * hN];
        for (int b = 0; b < bN; b++)
        {
            var dhNext = new double[hN];
            for (int t = tN - 1; t >= 0; t--)
            {
                var dgi = new double[3 * hN]; var dgh = new double[3 * hN]; var dh = new double[hN];
                for (int h = 0; h < hN; h++)
                {
                    double dhv = gy[(b * tN + t) * hN + h] + dhNext[h];
                    double dnp = dhv * (1 - z[b, t, h]) * (1 - n[b, t, h] * n[b, t, h]);
                    double dzp = dhv * (hs[b, t, h] - n[b, t, h]) * z[b, t, h] * (1 - z[b, t, h]);
                    dgi[hN + h] = dzp; dgh[hN + h] = dzp; dgi[2 * hN + h] = dnp;
                    if (resetAfter)
                    {
                        double drp = dnp * hn[b, t, h] * r[b, t, h] * (1 - r[b, t, h]);
                        dgi[h] = drp; dgh[h] = drp; dgh[2 * hN + h] = dnp * r[b, t, h];
                    }
                    else
                    {
                        dgh[2 * hN + h] = dnp;
                    }
                    dh[h] = dhv * z[b, t, h];
                }
                if (!resetAfter)
                    for (int k = 0; k < hN; k++)
                    {
                        double drh = 0;
                        for (int j = 0; j < hN; j++) drh += dgh[2 * hN + j] * whh[(2 * hN + j) * hN + k];
                        double drp = drh * hs[b, t, k] * r[b, t, k] * (1 - r[b, t, k]);
                        dgi[k] = drp; dgh[k] = drp;
                        dh[k] += drh * r[b, t, k];
                    }
                int hRows = resetAfter ? 3 * hN : 2 * hN;
                for (int row = 0; row < 3 * hN; row++)
                {
                    for (int j = 0; j < hN; j++)
                    {
                        if (row < hRows) dh[j] += dgh[row] * whh[row * hN + j];
                        gWhh[row * hN + j] += dgh[row] * (row < hRows ? hs[b, t, j] : rh[b, t, j]);
                    }
                    for (int i = 0; i < inN; i++)
                    {
                        gIn[(b * tN + t) * inN + i] += dgi[row] * wih[row * inN + i];
                        gWih[row * inN + i] += dgi[row] * x[(b * tN + t) * inN + i];
                    }
                    gBih[row] += dgi[row]; gBhh[row] += dgh[row];
                }
                dhNext = dh;
            }
            for (int h = 0; h < hN; h++) gH0[b * hN + h] = dhNext[h];
        }
        return new Expected(y, gIn, gH0, gWih, gWhh, gBih, gBhh);
    }

    private static void AssertClose(double[] expected, float[] actual, string what)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} values for {expected.Length} expected.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-4 + 2e-3 * Math.Abs(expected[i]),
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]}.");
    }

    private static void Check(IDirectGpuBackend backend, PortableGenericFft.Launcher launch)
    {
        foreach (bool resetAfter in new[] { true, false })
        foreach (var (bN, tN, inN, hN) in Shapes)
        {
            var x = Values(bN * tN * inN, 1, 0.05f); var h0 = Values(bN * hN, 9, 0.04f);
            var wih = Values(3 * hN * inN, 2, 0.03f); var whh = Values(3 * hN * hN, 3, 0.03f);
            var bih = Values(3 * hN, 4, 0.02f); var bhh = Values(3 * hN, 6, 0.025f);
            var gy = Values(bN * tN * hN, 5, 0.1f);
            var e = Reference(x, h0, wih, whh, bih, bhh, gy, bN, tN, inN, hN, resetAfter);
            string shape = $"B={bN} T={tN} In={inN} H={hN} resetAfter={resetAfter}";
            var (allHSize, gatesSize, ghnSize, rhSize) = PortableGruSequence.CacheSizes(tN, bN, hN);

            using var bx = backend.AllocateBuffer(x); using var bh0 = backend.AllocateBuffer(h0);
            using var bwih = backend.AllocateBuffer(wih); using var bwhh = backend.AllocateBuffer(whh);
            using var bbih = backend.AllocateBuffer(bih); using var bbhh = backend.AllocateBuffer(bhh);
            using var y = backend.AllocateBuffer(bN * tN * hN);
            using var allH = backend.AllocateBuffer(allHSize); using var gates = backend.AllocateBuffer(gatesSize);
            using var ghn = backend.AllocateBuffer(ghnSize); using var rh = backend.AllocateBuffer(rhSize);
            PortableGruSequence.Forward(backend, launch, bx, bh0, bwih, bwhh, bbih, bbhh, y, allH, gates, ghn, rh,
                tN, bN, inN, hN, resetAfter);
            AssertClose(e.Y, backend.DownloadBuffer(y), $"{shape} output");

            using var bgy = backend.AllocateBuffer(gy);
            using var gIn = backend.AllocateBuffer(bN * tN * inN); using var gH0 = backend.AllocateBuffer(bN * hN);
            using var gWih = backend.AllocateBuffer(3 * hN * inN); using var gWhh = backend.AllocateBuffer(3 * hN * hN);
            using var gBih = backend.AllocateBuffer(3 * hN); using var gBhh = backend.AllocateBuffer(3 * hN);
            // Every gradient is written, never accumulated: start them at NaN so a read would show.
            foreach (var (buf, count) in new[] { (gIn, bN * tN * inN), (gH0, bN * hN), (gWih, 3 * hN * inN),
                         (gWhh, 3 * hN * hN), (gBih, 3 * hN), (gBhh, 3 * hN) })
                backend.Fill(buf, float.NaN, count);
            PortableGruSequence.Backward(backend, launch, bgy, allH, gates, ghn, rh, bwih, bwhh, bx,
                gIn, gH0, gWih, gWhh, gBih, gBhh, tN, bN, inN, hN, resetAfter: resetAfter);
            AssertClose(e.GIn, backend.DownloadBuffer(gIn), $"{shape} gradInput");
            AssertClose(e.GH0, backend.DownloadBuffer(gH0), $"{shape} gradH0");
            AssertClose(e.GWih, backend.DownloadBuffer(gWih), $"{shape} gradWih");
            AssertClose(e.GWhh, backend.DownloadBuffer(gWhh), $"{shape} gradWhh");
            AssertClose(e.GBih, backend.DownloadBuffer(gBih), $"{shape} gradBih");
            AssertClose(e.GBhh, backend.DownloadBuffer(gBhh), $"{shape} gradBhh");
        }
    }

    private static PortableGenericFft.Launcher EngineLauncher(IDirectGpuBackend backend)
        => ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;

    [SkippableFact]
    public void Cuda_GruSequence_MatchesCpu()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend, EngineLauncher(backend));
    }

    [SkippableFact]
    public void OpenCl_GruSequence_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend, EngineLauncher(backend));
    }

    [SkippableFact]
    public void Vulkan_GruSequence_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, EngineLauncher(backend));
    }

    [SkippableFact]
    public void Hip_GruSequence_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, EngineLauncher(backend));
    }

    [SkippableFact]
    public void WebGpuWgsl_GruSequence_MatchesCpu_OnDawn()
    {
        using var dawn = DawnWgslLauncher.TryCreate();
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        using var host = new OpenClBackend();
        Skip.IfNot(host.IsAvailable, "OpenCL hosts the buffers and is not available.");
        Check(host, dawn!.For(host));
    }
}
#endif
