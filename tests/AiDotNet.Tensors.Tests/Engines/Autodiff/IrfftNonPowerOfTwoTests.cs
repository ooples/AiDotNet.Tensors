using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// IRFFT at lengths whose transform size is not a power of two.
/// </summary>
/// <remarks>
/// <para>
/// The forward clamps <c>nFft</c> up to <c>outputLength</c> when the spectrum is narrower than the
/// requested output. That clamp could produce any length at all, while every native transform path
/// underneath was radix-2 only, so the whole family threw <see cref="IndexOutOfRangeException"/>
/// from inside a butterfly that indexed past its buffer.
/// </para>
/// <para>
/// Three further defects sat behind that crash, each invisible while it stood:
/// the reused scratch buffer was never cleared, so bins the spectrum did not cover kept the
/// previous call's values; the Hermitian mirror stopped at <c>numFreqs - 1</c>, which is the
/// self-conjugate bin only when <c>nFft</c> is even; and the adjoint reached for
/// <c>FFTCore</c>, which zero-pads to the next power of two — a transform whose bins sit at
/// <c>k/nPadded</c> instead of <c>k/n</c>, so it does not transpose the forward it is paired with.
/// </para>
/// </remarks>
public sealed class IrfftNonPowerOfTwoTests
{
    private readonly IEngine _engine = new CpuEngine();

    public static TheoryData<int, int> Lengths => new()
    {
        // numFreqs, outputLength.  Natural nFft is (numFreqs-1)*2; anything above it is clamped,
        // which is what reaches the non-power-of-two paths.
        { 5, 8 },   // power of two, the case that always worked - guards against regression
        { 5, 9 },   // odd, no Nyquist bin exists
        { 5, 10 },  // even, not a power of two
        { 5, 11 },  // odd, not a power of two
        { 5, 12 },  // even, spectrum narrower than the transform (zero-padded bins)
        { 9, 17 },  // odd, larger
        { 9, 18 },  // even, not a power of two
        { 9, 15 },  // outputLength odd but BELOW natural nFft, so nFft stays 16
    };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void Forward_DoesNotThrow_AndMatchesDirectInverseDft(int numFreqs, int outputLength)
    {
        var input = Spectrum(numFreqs);
        int nFft = TransformLength(numFreqs, outputLength);

        var actual = _engine.IRFFT(input, outputLength);
        var expected = ReferenceIrfft(input, numFreqs, nFft, outputLength);

        Assert.Equal(outputLength, actual.Length);
        for (int i = 0; i < outputLength; i++)
        {
            Assert.True(
                Math.Abs(actual[i] - expected[i]) < 1e-9,
                $"sample {i}: engine {actual[i]:G17} vs direct inverse DFT {expected[i]:G17} " +
                $"(numFreqs={numFreqs}, outputLength={outputLength}, nFft={nFft})");
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void Adjoint_MatchesFiniteDifferences(int numFreqs, int outputLength)
    {
        var input = Spectrum(numFreqs);

        using var tape = new GradientTape<double>();
        var output = _engine.IRFFT(input, outputLength);
        var loss = _engine.ReduceSum(output, null);
        var analytic = tape.ComputeGradients(loss, new[] { input })[input];

        const double eps = 1e-7;
        for (int i = 0; i < input.Length; i++)
        {
            double original = input[i];
            input[i] = original + eps;
            double plus = Total(_engine.IRFFT(input, outputLength));
            input[i] = original - eps;
            double minus = Total(_engine.IRFFT(input, outputLength));
            input[i] = original;

            double numeric = (plus - minus) / (2 * eps);
            Assert.True(
                Math.Abs(numeric - analytic[i]) <= 1e-4 * Math.Max(1.0, Math.Abs(numeric)),
                $"bin {i / 2} {(i % 2 == 0 ? "real" : "imaginary")}: analytic {analytic[i]:G10} vs " +
                $"finite difference {numeric:G10} (numFreqs={numFreqs}, outputLength={outputLength})");
        }
    }

    /// <summary>
    /// The scratch buffer is thread-static and reused. A narrow spectrum leaves some bins unwritten,
    /// so if they are not cleared the result depends on whatever transform ran before it.
    /// </summary>
    [Fact]
    public void RepeatedCalls_AreIndependentOfWhatRanBefore()
    {
        var wide = Spectrum(9);
        var narrow = Spectrum(5);

        // Same call, once cold and once after a differently-shaped transform has used the buffer.
        var cold = _engine.IRFFT(narrow, 12);
        _ = _engine.IRFFT(wide, 12);
        var afterOther = _engine.IRFFT(narrow, 12);

        for (int i = 0; i < cold.Length; i++)
        {
            Assert.True(
                Math.Abs(cold[i] - afterOther[i]) < 1e-12,
                $"sample {i} changed from {cold[i]:G17} to {afterOther[i]:G17} purely because another " +
                "transform ran in between — the shared scratch buffer is leaking state between calls.");
        }
    }

    private static int TransformLength(int numFreqs, int outputLength)
    {
        int nFft = (numFreqs - 1) * 2;
        if (nFft < outputLength) nFft = outputLength;
        return nFft < 1 ? 1 : nFft;
    }

    private static Tensor<double> Spectrum(int numFreqs)
    {
        var input = new Tensor<double>(new[] { numFreqs * 2 });
        var rng = new Random(11);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        return input;
    }

    /// <summary>
    /// Direct O(n^2) inverse DFT of the Hermitian-completed spectrum — the definition this op
    /// implements, computed independently of any FFT machinery.
    /// </summary>
    private static double[] ReferenceIrfft(Tensor<double> input, int numFreqs, int nFft, int outputLength)
    {
        var re = new double[nFft];
        var im = new double[nFft];
        for (int k = 0; k < numFreqs && k < nFft; k++)
        {
            re[k] = input[k * 2];
            im[k] = input[k * 2 + 1];
        }
        // Conjugate-symmetric completion for every bin whose mirror is a distinct slot.
        for (int k = 1; k < numFreqs && nFft - k > k; k++)
        {
            re[nFft - k] = re[k];
            im[nFft - k] = -im[k];
        }

        var result = new double[outputLength];
        for (int j = 0; j < outputLength; j++)
        {
            double acc = 0;
            for (int k = 0; k < nFft; k++)
            {
                double angle = 2.0 * Math.PI * k * j / nFft;
                acc += re[k] * Math.Cos(angle) - im[k] * Math.Sin(angle);
            }
            result[j] = acc / nFft;
        }
        return result;
    }

    private static double Total(Tensor<double> t)
    {
        double sum = 0;
        for (int i = 0; i < t.Length; i++) sum += t[i];
        return sum;
    }
}
