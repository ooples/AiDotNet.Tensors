#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1165: WebGPU FusedLoRAForward used to refuse rank &gt; 256 (its workgroup projection buffer held 256 floats). The
/// kernel now walks the rank in chunks of 256, so any rank matches the reference
/// out = base + scaling * (input . A) . B -- including ranks that span several chunks and a partial last chunk.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuLoRARankTests
{
    private static float[] Values(int n, int salt)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((i * 7919 + salt * 131) % 97 - 48) / 64f;
        return v;
    }

    [SkippableTheory]
    [InlineData(3, 40, 8, 33)]      // one partial chunk
    [InlineData(2, 24, 256, 17)]    // exactly one chunk
    [InlineData(2, 16, 300, 513)]   // two chunks, partial last; more columns than threads
    [InlineData(1, 8, 600, 9)]      // three chunks
    public void FusedLoRAForward_AnyRank_MatchesReference(int batch, int inFeatures, int rank, int outFeatures)
    {
        using var backend = new WebGpuBackend();
        Skip.IfNot(backend.IsAvailable, "WebGPU is not available.");
        float[] input = Values(batch * inFeatures, 1), a = Values(inFeatures * rank, 2), b = Values(rank * outFeatures, 3);
        float[] baseOut = Values(batch * outFeatures, 4);
        const float scaling = 0.375f;

        using var bIn = backend.AllocateBuffer(input);
        using var bBase = backend.AllocateBuffer(baseOut);
        using var bA = backend.AllocateBuffer(a);
        using var bB = backend.AllocateBuffer(b);
        using var bOut = backend.AllocateBuffer(batch * outFeatures);
        backend.FusedLoRAForward(bIn, bBase, bA, bB, bOut, batch, inFeatures, rank, outFeatures, scaling);
        float[] got = backend.DownloadBuffer(bOut);

        for (int n = 0; n < batch; n++)
        {
            var proj = new double[rank];
            for (int r = 0; r < rank; r++)
                for (int i = 0; i < inFeatures; i++) proj[r] += input[n * inFeatures + i] * (double)a[i * rank + r];
            for (int o = 0; o < outFeatures; o++)
            {
                double delta = 0;
                for (int r = 0; r < rank; r++) delta += proj[r] * b[r * outFeatures + o];
                double expected = baseOut[n * outFeatures + o] + scaling * delta;
                double actual = got[n * outFeatures + o];
                Assert.True(Math.Abs(expected - actual) <= 1e-3 * (1 + Math.Abs(expected)),
                    $"[{n},{o}] rank {rank}: expected {expected}, got {actual}");
            }
        }
    }
}
#endif
