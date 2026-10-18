using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// ILstmLastStepTraining on CUDA: the BPTT of an LSTM whose output is its final hidden state alone, from that state's
/// gradient [batch, hidden], must equal LstmBackwardSequence fed a [batch, seq, hidden] gradient that is zero except at
/// the last step -- every gradient output, bit for bit (the recurrence sees the same values in the same order).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaLstmLastStepTests
{
    private static float[] Random(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    [SkippableTheory]
    [InlineData(32, 16, 32, 64)]   // the parity LSTM
    [InlineData(7, 4, 3, 33)]      // hidden padded to 64
    [InlineData(5, 9, 6, 20)]      // hidden padded to 32
    [InlineData(1, 1, 1, 1)]
    public void LastStepBackward_EqualsTheSequenceBackwardOfALastStepGradient(int b, int t, int inSize, int h)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        var last = (ILstmLastStepTraining)backend;
        Assert.True(last.SupportsLstmLastStepBackward(h));
        int g = 4 * h;
        using var x = backend.AllocateBuffer(Random(b * t * inSize, 1));
        using var wih = backend.AllocateBuffer(Random(g * inSize, 2, 0.4f));
        using var whh = backend.AllocateBuffer(Random(g * h, 3, 0.4f));
        using var bih = backend.AllocateBuffer(Random(g, 4, 0.1f));
        using var bhh = backend.AllocateBuffer(Random(g, 5, 0.1f));
        using var h0 = backend.AllocateBuffer(new float[b * h]);
        using var c0 = backend.AllocateBuffer(new float[b * h]);
        using var output = backend.AllocateBuffer(b * t * h);
        using var hFinal = backend.AllocateBuffer(b * h);
        using var cFinal = backend.AllocateBuffer(b * h);
        using var allH = backend.AllocateBuffer((t + 1) * b * h);
        using var allC = backend.AllocateBuffer((t + 1) * b * h);
        using var gates = backend.AllocateBuffer(t * b * g);
        backend.LstmForwardSequence(x, h0, c0, wih, whh, bih, bhh, output, hFinal, cFinal, allH, allC, gates, t, b, inSize, h);

        float[] dLast = Random(b * h, 6);
        var dSeq = new float[b * t * h];
        for (int r = 0; r < b; r++)
            for (int j = 0; j < h; j++) dSeq[(r * t + t - 1) * h + j] = dLast[r * h + j];
        using var dLastBuf = backend.AllocateBuffer(dLast);
        using var dSeqBuf = backend.AllocateBuffer(dSeq);

        float[][] Run(bool lastOnly)
        {
            var outs = new[] { b * t * inSize, b * h, b * h, g * inSize, g * h, g, g }
                .Select(n => backend.AllocateBuffer(Enumerable.Repeat(float.NaN, n).ToArray())).ToArray();
            try
            {
                if (lastOnly)
                    last.LstmBackwardLastStep(dLastBuf, allH, allC, gates, h0, c0, wih, whh, x,
                        outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], t, b, inSize, h);
                else
                    backend.LstmBackwardSequence(dSeqBuf, allH, allC, gates, h0, c0, wih, whh, x,
                        outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], t, b, inSize, h);
                return outs.Select(o => backend.DownloadBuffer(o)).ToArray();
            }
            finally
            {
                foreach (var o in outs) o.Dispose();
            }
        }

        var expected = Run(lastOnly: false);
        var actual = Run(lastOnly: true);
        string[] names = { "dX", "dH0", "dC0", "dWih", "dWhh", "dBih", "dBhh" };
        for (int k = 0; k < names.Length; k++)
        {
            Assert.All(actual[k], v => Assert.False(float.IsNaN(v), $"{names[k]} left an element unwritten"));
            Assert.Equal(expected[k], actual[k]);
        }
    }
}
