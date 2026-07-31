using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The IRFFT tape backward must run on the device its forward ran on, and must agree with the managed
/// adjoint to fp32.
/// </summary>
/// <remarks>
/// <para>
/// DirectGpuTensorEngine.IRFFT recorded BackwardFunctions.IRFFTAdjointBackward unconditionally, so every
/// IRFFT training step materialised host arrays and ran the transform on the CPU even under the GPU
/// engine — the sibling of the RFFT gap fixed earlier, and the second half of the CodeRabbit review
/// comment on PR #911.
/// </para>
/// <para>
/// IrfftAdjointGpu composes the adjoint from primitives every backend implements: zero-pad the real
/// gradient to nFft, take the UNNORMALIZED FORWARD transform, keep the one-sided bins, and weight them
/// by c_k/nFft with c_k = 1 at DC and Nyquist and 2 elsewhere. That per-bin weight is what a plain Scale
/// cannot express and is the easiest part to get wrong, so it is compared against the managed adjoint
/// element by element rather than spot-checked.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class IrfftAdjointGpuDispatchTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public IrfftAdjointGpuDispatchTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [SkippableTheory]
    [InlineData(1, 8)]
    [InlineData(2, 16)]
    [InlineData(3, 32)]
    public void IrfftAdjoint_GpuMatchesManaged(int batch, int nFft)
    {
        Skip.If(!_available, "GPU backend not available");
        int numFreqs = nFft / 2 + 1;
        int outputLength = nFft;

        // Spectrum laid out [.., numFreqs, 2] interleaved (re, im), matching IRFFT's input contract.
        var spectrum = Rand([batch, numFreqs, 2], 21);
        var gradOut = Rand([batch, outputLength], 23);

        // Managed adjoint — the reference.
        var managedGrads = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
        BackwardFunctions<float>.IRFFTAdjointBackward(
            gradOut, [spectrum], spectrum,
            new object[] { numFreqs, nFft, outputLength }, _cpu, managedGrads);
        var expected = managedGrads[spectrum];

        var actual = _gpu!.IrfftAdjointGpu(gradOut, numFreqs, nFft, outputLength, spectrum.Shape.ToArray());
        Assert.True(actual is not null,
            $"IrfftAdjointGpu returned null for batch={batch}, nFft={nFft}, so the IRFFT backward would " +
            $"silently run on the CPU under the GPU engine — the defect this test pins.");

        Assert.Equal(expected.Length, actual!.Length);
        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)expected[i] - actual[i]);
            if (d > worst) { worst = d; worstAt = i; }
        }
        _out.WriteLine($"batch={batch} nFft={nFft} numFreqs={numFreqs} maxAbsDiff={worst:E3} at={worstAt}");
        Assert.True(worst < 1e-5,
            $"GPU IRFFT adjoint differs from the managed adjoint by {worst:E3} at flat index {worstAt} " +
            $"(batch={batch}, nFft={nFft}). Index parity tells you where: an even index is a real part, " +
            $"an odd one imaginary; index {numFreqs * 2 - 1} within a row is the Nyquist imaginary slot, " +
            $"which must be exactly zero.");
    }

    /// <summary>The Nyquist imaginary slot cannot influence the forward, so its gradient must be zero.</summary>
    [SkippableFact]
    public void IrfftAdjoint_NyquistImaginaryGradientIsZero()
    {
        Skip.If(!_available, "GPU backend not available");
        const int Batch = 2, NFft = 16, OutputLength = 16;
        int numFreqs = NFft / 2 + 1;

        var spectrum = Rand([Batch, numFreqs, 2], 31);
        var gradOut = Rand([Batch, OutputLength], 37);

        var actual = _gpu!.IrfftAdjointGpu(gradOut, numFreqs, NFft, OutputLength, spectrum.Shape.ToArray());
        Skip.If(actual is null, "GPU adjoint unavailable");

        for (int b = 0; b < Batch; b++)
        {
            int nyquistImag = b * numFreqs * 2 + (numFreqs - 1) * 2 + 1;
            Assert.True(Math.Abs((double)actual![nyquistImag]) < 1e-12,
                $"Nyquist imaginary gradient for batch {b} is {actual[nyquistImag]:E3}, expected exactly 0 " +
                $"— that component does not affect the forward output.");
        }
    }
}
