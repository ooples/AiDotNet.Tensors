using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Launch-time checks for kernels whose element count comes from the caller: a count past a buffer's capacity would
/// make the kernel read or write outside that buffer.
/// </summary>
internal static class GpuBufferCapacity
{
    /// <summary>
    /// True when a launch of <paramref name="count"/> elements has nothing to do; a negative count throws rather than
    /// returning with the output unwritten.
    /// </summary>
    internal static bool IsEmptyLaunch(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), $"The element count {count} is negative.");
        return count == 0;
    }

    /// <summary>Throws when <paramref name="buffer"/> holds fewer than <paramref name="count"/> 4-byte elements.</summary>
    internal static void Require(IGpuBuffer buffer, int count, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), $"The element count {count} is negative.");
        if (buffer.SizeInBytes < (long)count * sizeof(float))
            throw new ArgumentException(
                $"{name} holds {buffer.SizeInBytes / sizeof(float)} 4-byte elements; the launch needs {count}.", name);
    }

    /// <summary>The <see cref="IIndexConversionBackend.ScatterPositions"/> arguments a launch can trust.</summary>
    internal static void RequirePositions(IGpuBuffer output, int srcDim, int inner, int count)
    {
        if (srcDim <= 0) throw new ArgumentOutOfRangeException(nameof(srcDim), $"The scatter axis length {srcDim} must be positive.");
        if (inner <= 0) throw new ArgumentOutOfRangeException(nameof(inner), $"The inner size {inner} must be positive.");
        Require(output, count, nameof(output));
    }

    /// <summary>
    /// The <see cref="IDirectGpuBackend.StftBackwardFrames"/> arguments a launch can trust; returns its work-item count.
    /// The kernel reads <paramref name="numFreqs"/> bins of every frame (and the magnitude only for a phase target) and
    /// writes <paramref name="nFft"/> samples per frame.
    /// </summary>
    internal static int RequireStftFrames(IGpuBuffer grad, IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer frameGrad,
        int batch, int nFft, int numFrames, int numFreqs, int phaseTarget)
    {
        RequireStftShape(batch, nFft, numFrames);
        if (numFreqs != nFft / 2 + 1)
            throw new ArgumentOutOfRangeException(nameof(numFreqs), $"A one-sided {nFft}-point spectrum has {nFft / 2 + 1} bins, not {numFreqs}.");
        int bins = checked(batch * numFreqs * numFrames);
        Require(grad, bins, nameof(grad));
        Require(phase, bins, nameof(phase));
        if (phaseTarget != 0) Require(magnitude, bins, nameof(magnitude));
        else if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
        int samples = checked(batch * numFrames * nFft);
        Require(frameGrad, samples, nameof(frameGrad));
        return samples;
    }

    /// <summary>
    /// The <see cref="IDirectGpuBackend.StftBackwardSignal"/> arguments a launch can trust; returns its work-item count.
    /// </summary>
    internal static int RequireStftSignal(IGpuBuffer frameGrad, IGpuBuffer window, IGpuBuffer gradSignal,
        int batch, int length, int pad, int nFft, int hop, int numFrames)
    {
        RequireStftFraming(batch, length, pad, nFft, hop, numFrames);
        Require(frameGrad, checked(batch * numFrames * nFft), nameof(frameGrad));
        Require(window, nFft, nameof(window));
        int samples = checked(batch * length);
        Require(gradSignal, samples, nameof(gradSignal));
        return samples;
    }

    /// <summary>
    /// The <see cref="IDirectGpuBackend.StftBackwardWindow"/> arguments a launch can trust; returns its work-item count.
    /// </summary>
    internal static int RequireStftWindow(IGpuBuffer frameGrad, IGpuBuffer signal, IGpuBuffer gradWindow,
        int batch, int length, int pad, int nFft, int hop, int numFrames)
    {
        RequireStftFraming(batch, length, pad, nFft, hop, numFrames);
        Require(frameGrad, checked(batch * numFrames * nFft), nameof(frameGrad));
        Require(signal, checked(batch * length), nameof(signal));
        Require(gradWindow, nFft, nameof(gradWindow));
        return nFft;
    }

    private static void RequireStftShape(int batch, int nFft, int numFrames)
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), $"The batch size {batch} must be positive.");
        if (nFft <= 0) throw new ArgumentOutOfRangeException(nameof(nFft), $"The FFT size {nFft} must be positive.");
        if (numFrames <= 0) throw new ArgumentOutOfRangeException(nameof(numFrames), $"The frame count {numFrames} must be positive.");
    }

    // The frames the kernels index must be the ones the forward analysed: numFrames frames of nFft samples, hop apart,
    // inside the signal reflection-padded by pad on each side (a reflection needs pad < length).
    private static void RequireStftFraming(int batch, int length, int pad, int nFft, int hop, int numFrames)
    {
        RequireStftShape(batch, nFft, numFrames);
        if (hop <= 0) throw new ArgumentOutOfRangeException(nameof(hop), $"The hop length {hop} must be positive.");
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length), $"The signal length {length} must be positive.");
        if (pad < 0 || (pad > 0 && pad >= length))
            throw new ArgumentOutOfRangeException(nameof(pad), $"A reflection pad of {pad} needs a signal longer than it; got {length}.");
        long padded = (long)length + 2L * pad;
        // The kernels index in 32-bit ints: the overlap bound computes q - nFft + hop with q < padded, and the window
        // pass f*hop + i < padded. A hop past the padded length frames nothing a shorter one would not, so bounding
        // both keeps every intermediate under int.MaxValue.
        if (2L * padded > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(length), $"A padded signal of {padded} samples is too long for the 32-bit kernel indexing.");
        if (hop > padded)
            throw new ArgumentOutOfRangeException(nameof(hop), $"The hop length {hop} exceeds the padded signal length {padded}.");
        if (padded < nFft || (long)(numFrames - 1) * hop + nFft > padded)
            throw new ArgumentOutOfRangeException(nameof(numFrames),
                $"{numFrames} frames of {nFft} samples, {hop} apart, do not fit a padded signal of {padded} samples.");
    }
}
