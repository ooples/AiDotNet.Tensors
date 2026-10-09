#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The fused-LSTM training contract <c>DirectGpuTensorEngine.TryLstmSequenceTrain</c> relies on, checked per backend
/// against a CPU forward + BPTT: batch-major input [B, T, in] and output [B, T, H], PyTorch gate order (i, f, g, o),
/// h0 = c0 = 0, the two biases summed, and backward gradients for the input, both weights and the bias, called with
/// the same buffers the engine allocates, for every hidden size up to the backend's
/// <see cref="IFusedLstmSequenceTraining.MaxFusedLstmHidden"/>.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LstmSequenceBackendParityTests
{
    private static readonly (int B, int T, int In, int H)[] Shapes =
    {
        (1, 1, 1, 1),
        (2, 3, 4, 5),
        (3, 5, 7, 16),
        (5, 3, 6, 64),    // B*H spans several work-groups
        (5, 4, 6, 100),   // a generic 256-wide work-group would split a batch row
        (2, 3, 5, 300),   // H wider than a 256-wide work-group (the engine allows H <= 1024)
    };

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] Values(int n, int seed, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((((i + seed) * 7919) % 41) - 20) * scale;
        return v;
    }

    // CPU LSTM forward + BPTT in double; returns the output and the four gradients.
    private static (double[] Y, double[] GIn, double[] GWih, double[] GWhh, double[] GBias) Reference(
        float[] x, float[] wih, float[] whh, float[] bias, float[] gy, int bN, int tN, int inN, int hN)
    {
        var y = new double[bN * tN * hN];
        var hs = new double[bN, tN + 1, hN];
        var cs = new double[bN, tN + 1, hN];
        var gates = new double[bN, tN, 4 * hN]; // activated i, f, g, o
        for (int b = 0; b < bN; b++)
            for (int t = 0; t < tN; t++)
                for (int h = 0; h < hN; h++)
                {
                    var pre = new double[4];
                    for (int g = 0; g < 4; g++)
                    {
                        int row = g * hN + h;
                        double s = bias[row];
                        for (int i = 0; i < inN; i++) s += (double)wih[row * inN + i] * x[(b * tN + t) * inN + i];
                        for (int j = 0; j < hN; j++) s += (double)whh[row * hN + j] * hs[b, t, j];
                        pre[g] = s;
                    }
                    double ig = 1 / (1 + Math.Exp(-pre[0])), fg = 1 / (1 + Math.Exp(-pre[1]));
                    double gg = Math.Tanh(pre[2]), og = 1 / (1 + Math.Exp(-pre[3]));
                    double c = fg * cs[b, t, h] + ig * gg;
                    cs[b, t + 1, h] = c;
                    gates[b, t, h] = ig; gates[b, t, hN + h] = fg; gates[b, t, 2 * hN + h] = gg; gates[b, t, 3 * hN + h] = og;
                    // h for step t+1 is written after every unit of step t has read hs[b, t, *]: store into a staging row.
                    y[(b * tN + t) * hN + h] = og * Math.Tanh(c);
                    if (h == hN - 1)
                        for (int k = 0; k < hN; k++) hs[b, t + 1, k] = y[(b * tN + t) * hN + k];
                }

        var gIn = new double[bN * tN * inN];
        var gWih = new double[4 * hN * inN];
        var gWhh = new double[4 * hN * hN];
        var gBias = new double[4 * hN];
        for (int b = 0; b < bN; b++)
        {
            var dhNext = new double[hN];
            var dcNext = new double[hN];
            for (int t = tN - 1; t >= 0; t--)
            {
                var dPre = new double[4 * hN];
                for (int h = 0; h < hN; h++)
                {
                    double dh = gy[(b * tN + t) * hN + h] + dhNext[h];
                    double ig = gates[b, t, h], fg = gates[b, t, hN + h], gg = gates[b, t, 2 * hN + h], og = gates[b, t, 3 * hN + h];
                    double c = cs[b, t + 1, h], tc = Math.Tanh(c);
                    double dc = dh * og * (1 - tc * tc) + dcNext[h];
                    dPre[h] = dc * gg * ig * (1 - ig);
                    dPre[hN + h] = dc * cs[b, t, h] * fg * (1 - fg);
                    dPre[2 * hN + h] = dc * ig * (1 - gg * gg);
                    dPre[3 * hN + h] = dh * tc * og * (1 - og);
                    dcNext[h] = dc * fg;
                }
                Array.Clear(dhNext, 0, hN);
                for (int row = 0; row < 4 * hN; row++)
                {
                    gBias[row] += dPre[row];
                    for (int i = 0; i < inN; i++)
                    {
                        gWih[row * inN + i] += dPre[row] * x[(b * tN + t) * inN + i];
                        gIn[(b * tN + t) * inN + i] += dPre[row] * wih[row * inN + i];
                    }
                    for (int j = 0; j < hN; j++)
                    {
                        gWhh[row * hN + j] += dPre[row] * hs[b, t, j];
                        dhNext[j] += dPre[row] * whh[row * hN + j];
                    }
                }
            }
        }
        return (y, gIn, gWih, gWhh, gBias);
    }

    private static void AssertClose(double[] expected, float[] actual, string what)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * (1 + Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    private static void Check(IDirectGpuBackend backend)
    {
        int maxHidden = ((IFusedLstmSequenceTraining)backend).MaxFusedLstmHidden;
        Assert.True(maxHidden >= 64, $"MaxFusedLstmHidden {maxHidden} is too small to be useful.");
        foreach (var (bN, tN, inN, hN) in Shapes)
        {
            if (hN > maxHidden)
            {
                // Above the declared limit the forward must refuse, not compute a wrong sequence.
                using var x0 = backend.AllocateBuffer(new float[bN * tN * inN]);
                using var big = backend.AllocateBuffer(new float[(tN + 1) * bN * hN * 4 + 4 * hN * (inN + hN)]);
                Assert.Throws<InvalidOperationException>(() => backend.LstmForwardSequence(x0, big, big, big, big, big, big,
                    big, big, big, big, big, big, tN, bN, inN, hN));
                continue;
            }
            var x = Values(bN * tN * inN, 1, 0.05f);
            var wih = Values(4 * hN * inN, 2, 0.03f);
            var whh = Values(4 * hN * hN, 3, 0.03f);
            var bias = Values(4 * hN, 4, 0.02f);
            var gy = Values(bN * tN * hN, 5, 0.1f);
            var r = Reference(x, wih, whh, bias, gy, bN, tN, inN, hN);
            string shape = $"B={bN} T={tN} In={inN} H={hN}";

            using var bx = backend.AllocateBuffer(x);
            using var bwih = backend.AllocateBuffer(wih);
            using var bwhh = backend.AllocateBuffer(whh);
            using var bbias = backend.AllocateBuffer(bias);
            using var zeroBias = backend.AllocateBuffer(new float[4 * hN]);
            using var h0 = backend.AllocateBuffer(new float[bN * hN]);
            using var c0 = backend.AllocateBuffer(new float[bN * hN]);
            using var y = backend.AllocateBuffer(new float[bN * tN * hN]);
            using var hFinal = backend.AllocateBuffer(new float[bN * hN]);
            using var cFinal = backend.AllocateBuffer(new float[bN * hN]);
            using var allH = backend.AllocateBuffer(new float[(tN + 1) * bN * hN]);
            using var allC = backend.AllocateBuffer(new float[(tN + 1) * bN * hN]);
            using var gates = backend.AllocateBuffer(new float[tN * bN * hN * 4]);
            backend.LstmForwardSequence(bx, h0, c0, bwih, bwhh, bbias, zeroBias, y, hFinal, cFinal, allH, allC, gates,
                tN, bN, inN, hN);
            AssertClose(r.Y, backend.DownloadBuffer(y), $"{shape} output");

            using var bgy = backend.AllocateBuffer(gy);
            using var gIn = backend.AllocateBuffer(new float[bN * tN * inN]);
            using var gH0 = backend.AllocateBuffer(new float[bN * hN]);
            using var gC0 = backend.AllocateBuffer(new float[bN * hN]);
            using var gWih = backend.AllocateBuffer(new float[4 * hN * inN]);
            using var gWhh = backend.AllocateBuffer(new float[4 * hN * hN]);
            using var gBias = backend.AllocateBuffer(new float[4 * hN]);
            using var gBiasHh = backend.AllocateBuffer(new float[4 * hN]);
            backend.LstmBackwardSequence(bgy, allH, allC, gates, h0, c0, bwih, bwhh, bx,
                gIn, gH0, gC0, gWih, gWhh, gBias, gBiasHh, tN, bN, inN, hN);
            AssertClose(r.GIn, backend.DownloadBuffer(gIn), $"{shape} gradInput");
            AssertClose(r.GWih, backend.DownloadBuffer(gWih), $"{shape} gradWih");
            AssertClose(r.GWhh, backend.DownloadBuffer(gWhh), $"{shape} gradWhh");
            AssertClose(r.GBias, backend.DownloadBuffer(gBias), $"{shape} gradBias");
        }
    }

    [SkippableFact]
    public void OpenCl_LstmSequence_MatchesCpu()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_LstmSequence_MatchesCpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable,
            "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_LstmSequence_MatchesCpu()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif