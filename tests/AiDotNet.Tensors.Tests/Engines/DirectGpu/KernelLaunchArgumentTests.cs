#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Kernels whose launch passed a different argument list than the kernel declares. A CUDA launch does not
/// check the count: the kernel reads its parameters from the argument array by position, so a float landed in
/// an int `size` slot (~1e9 as an int), the bounds check never fired, and every thread of the last block wrote
/// past the output. SELU did this to the engine's cached ones vectors deep into the op-parity suite and turned
/// unrelated gradients wrong. Each test pads the output with a sentinel and requires it untouched.
/// </summary>
[Collection("DirectGpuSerial")]
public class KernelLaunchArgumentTests
{
    private const int N = 37;          // not a multiple of the 256-thread block: the overrun lands in the padding
    private const int Pad = 256;
    private const float Sentinel = -12345f;

    private static IDirectGpuBackend? Backend(out DirectGpuTensorEngine? engine)
    {
        engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or TypeInitializationException) { }
        return engine is not null && engine.IsGpuAvailable ? engine.GetBackend() : null;
    }

    private static float[] Inputs(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);
        return x;
    }

    private static float[] Padded(float[] data)
    {
        var p = new float[data.Length + Pad];
        Array.Fill(p, Sentinel);
        Array.Copy(data, p, data.Length);
        return p;
    }

    private static void AssertPaddingUntouched(float[] result, int n, string op)
    {
        for (int i = n; i < result.Length; i++)
            Assert.True(result[i] == Sentinel, $"{op} wrote past its {n} elements: [{i}] = {result[i]}.");
    }

    [SkippableTheory]
    [InlineData("selu")]
    [InlineData("hardtanh")]
    [InlineData("hardtanh-custom")]
    public void UnaryActivation_StaysInBoundsAndMatchesReference(string op)
    {
        var backend = Backend(out var engine);
        Skip.If(backend is null, "No DirectGpu backend available.");
        using (engine)
        {
            var x = Inputs(N, 3);
            using var input = backend!.AllocateBuffer(Padded(x));
            using var output = backend.AllocateBuffer(Padded(new float[N]));
            const float alpha = 1.6732632423543772f, scale = 1.0507009873554805f;
            float lo = op == "hardtanh-custom" ? -0.5f : -1f, hi = op == "hardtanh-custom" ? 2f : 1f;
            if (op == "selu") backend.Selu(input, output, alpha, scale, N);
            else backend.Hardtanh(input, output, lo, hi, N);
            var result = backend.DownloadBuffer(output);
            AssertPaddingUntouched(result, N, op);
            for (int i = 0; i < N; i++)
            {
                float expected = op == "selu"
                    ? scale * (x[i] > 0 ? x[i] : alpha * (MathF.Exp(x[i]) - 1))
                    : Math.Clamp(x[i], lo, hi);
                Assert.True(Math.Abs(expected - result[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected)),
                    $"{op}[{i}]: expected {expected}, got {result[i]} (x = {x[i]}).");
            }
        }
    }

    [SkippableTheory]
    [InlineData("selu")]
    [InlineData("hardtanh")]
    [InlineData("hardtanh-custom")]
    public void ActivationBackward_StaysInBoundsAndMatchesReference(string op)
    {
        var backend = Backend(out var engine);
        Skip.If(backend is null, "No DirectGpu backend available.");
        using (engine)
        {
            var x = Inputs(N, 5);
            var g = Inputs(N, 6);
            using var input = backend!.AllocateBuffer(Padded(x));
            using var grad = backend.AllocateBuffer(Padded(g));
            using var gradIn = backend.AllocateBuffer(Padded(new float[N]));
            const float alpha = 1.6732632423543772f, scale = 1.0507009873554805f;
            float lo = op == "hardtanh-custom" ? -0.5f : -1f, hi = op == "hardtanh-custom" ? 2f : 1f;
            if (op == "selu") backend.SeluBackward(grad, input, gradIn, alpha, scale, N);
            else backend.HardtanhBackward(grad, input, gradIn, lo, hi, N);
            var result = backend.DownloadBuffer(gradIn);
            AssertPaddingUntouched(result, N, op + " backward");
            for (int i = 0; i < N; i++)
            {
                float deriv = op == "selu"
                    ? (x[i] >= 0 ? scale : scale * alpha * MathF.Exp(x[i]))
                    : (x[i] > lo && x[i] < hi ? 1f : 0f);
                float expected = g[i] * deriv;
                Assert.True(Math.Abs(expected - result[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected)),
                    $"{op} backward[{i}]: expected {expected}, got {result[i]} (x = {x[i]}).");
            }
        }
    }

    [SkippableFact]
    public void ScalarMseAndHuberLoss_MatchReference()
    {
        // The elementwise mse_loss / huber_loss kernels shared their names with the per-sample kernels of the
        // loss_forward module, which loads later and replaced them: the scalar overloads ran the per-sample kernel
        // with the elementwise argument list.
        var backend = Backend(out var engine);
        Skip.If(backend is null, "No DirectGpu backend available.");
        using (engine)
        {
            const int n = 1000;
            var p = Inputs(n, 7);
            var t = Inputs(n, 8);
            using var pb = backend!.AllocateBuffer(p);
            using var tb = backend.AllocateBuffer(t);
            double mse = 0, huber = 0;
            const float delta = 0.75f;
            for (int i = 0; i < n; i++)
            {
                double d = p[i] - t[i], a = Math.Abs(d);
                mse += d * d;
                huber += a <= delta ? 0.5 * d * d : delta * (a - 0.5 * delta);
            }
            Assert.Equal(mse / n, backend.MseLoss(pb, tb, n), 3);
            Assert.Equal(huber / n, backend.HuberLoss(pb, tb, n, delta), 3);
        }
    }

    [SkippableFact]
    public void ContrastiveLossAndBackward_MatchPerSampleReference()
    {
        var backend = Backend(out var engine);
        Skip.If(backend is null, "No DirectGpu backend available.");
        using (engine)
        {
            // embeddingDim > batchSize: the old elementwise kernel bounded by embeddingDim wrote past the batch output.
            const int batch = 3, dim = 7;
            const float margin = 2.5f;
            var a = Inputs(batch * dim, 9);
            var b = Inputs(batch * dim, 10);
            var labels = new float[] { 0f, 1f, 1f };
            using var ab = backend!.AllocateBuffer(a);
            using var bb = backend.AllocateBuffer(b);
            using var lb = backend.AllocateBuffer(labels);

            double total = 0;
            var g1 = new double[batch * dim];
            for (int s = 0; s < batch; s++)
            {
                double distSq = 0;
                for (int d = 0; d < dim; d++) { double diff = a[s * dim + d] - b[s * dim + d]; distSq += diff * diff; }
                double dist = Math.Sqrt(distSq), md = Math.Max(0, margin - dist), y = labels[s];
                total += (1 - y) * 0.5 * distSq + y * 0.5 * md * md;
                double coeff = (1 - y) - (md > 0 && dist > 1e-7 ? y * (margin - dist) / dist : 0);
                for (int d = 0; d < dim; d++) g1[s * dim + d] = coeff * (a[s * dim + d] - b[s * dim + d]) / batch;
            }
            Assert.Equal(total / batch, backend.ContrastiveLoss(ab, bb, lb, batch, dim, margin), 4);

            using var grad1 = backend.AllocateBuffer(batch * dim);
            using var grad2 = backend.AllocateBuffer(batch * dim);
            backend.ContrastiveBackward(ab, bb, lb, grad1, grad2, batch, dim, margin);
            var r1 = backend.DownloadBuffer(grad1);
            var r2 = backend.DownloadBuffer(grad2);
            for (int i = 0; i < batch * dim; i++)
            {
                Assert.True(Math.Abs(g1[i] - r1[i]) < 1e-5, $"grad1[{i}]: expected {g1[i]}, got {r1[i]}.");
                Assert.True(Math.Abs(-g1[i] - r2[i]) < 1e-5, $"grad2[{i}]: expected {-g1[i]}, got {r2[i]}.");
            }
        }
    }
}
#endif
