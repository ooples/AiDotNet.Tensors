// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="ITorchModelOpsBackend"/> over the kernels of <see cref="TorchModelOpsKernels"/>: each member validates
/// its buffers and issues the launches through the backend's launcher (kernel name, threads, then the kernel's
/// arguments in order: buffers, int and float scalars). Every backend shares it, so the launch sequence is one code
/// path.
/// </summary>
internal static class PortableTorchModelOps
{
    private static void Require(IGpuBuffer buffer, long required, string parameter, string formula)
    {
        if (buffer is null) throw new System.ArgumentNullException(parameter);
        if (buffer.Size < required)
            throw new System.ArgumentException(
                $"{parameter} buffer too small: capacity={buffer.Size} elements, required={required} ({formula}).", parameter);
    }

    public static void ActivationBackward(PortableGenericFft.Launcher launch, int op, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gradInput, int n, float p0, float p1)
    {
        if (n <= 0) return;
        if (op < 0 || op > 7) throw new System.ArgumentOutOfRangeException(nameof(op));
        Require(gradOutput, n, nameof(gradOutput), "n");
        Require(input, n, nameof(input), "n");
        Require(gradInput, n, nameof(gradInput), "n");
        launch("mo_activation_backward", n, gradOutput, input, gradInput, n, op, p0, p1);
    }

    /// <summary>
    /// The quantization range [qMin, qMax] of <paramref name="numBits"/>-bit integers: [-2^(b-1), 2^(b-1) - 1] when
    /// <paramref name="symmetric"/>, [0, 2^b - 1] otherwise (the CPU's range).
    /// </summary>
    public static (long QMin, long QMax) FakeQuantizeRange(int numBits, bool symmetric)
        => symmetric ? (-(1L << (numBits - 1)), (1L << (numBits - 1)) - 1) : (0L, (1L << numBits) - 1);

    /// <summary>The largest magnitude of the clamp bounds qMin - zeroPoint and qMax - zeroPoint the kernel takes.</summary>
    public const int FakeQuantizeMaxBound = int.MaxValue - 1;

    /// <summary>
    /// Whether the kernel computes this quantization: always for a derived zero point, and for a given one when
    /// qMin - zeroPoint and qMax - zeroPoint both lie within +-<see cref="FakeQuantizeMaxBound"/> (the kernel clamps in
    /// 32-bit integers). Only zero points far outside the range miss it; the engine computes those on the host.
    /// </summary>
    public static bool FakeQuantizeFitsDevice(int numBits, bool symmetric, int? zeroPoint)
    {
        if (numBits < 1 || numBits > 30) return false;
        var (qMin, qMax) = FakeQuantizeRange(numBits, symmetric);
        long zp = zeroPoint ?? 0;
        return qMin - zp >= -FakeQuantizeMaxBound && qMax - zp <= FakeQuantizeMaxBound;
    }

    /// <summary>
    /// A positive finite double as the kernel reads it: a significand in [2^52, 2^53) split into its low and high
    /// words, and the exponent e with value = significand * 2^e (a subnormal normalized).
    /// </summary>
    public static (int Low, int High, int Exponent) SplitStep(double step)
    {
        if (!(step > 0) || double.IsInfinity(step))
            throw new System.ArgumentOutOfRangeException(nameof(step), "The step must be a positive finite number.");
        long bits = System.BitConverter.DoubleToInt64Bits(step);
        int biased = (int)((bits >> 52) & 0x7FF);
        long significand = bits & 0xFFFFFFFFFFFFFL;
        int exponent;
        if (biased != 0)
        {
            significand |= 1L << 52;
            exponent = biased - 1075;
        }
        else
        {
            int shift = 0;
            while ((significand & (1L << 52)) == 0)
            {
                significand <<= 1;
                shift++;
            }
            exponent = -1074 - shift;
        }
        return (unchecked((int)(uint)(significand & 0xFFFFFFFFL)), (int)(significand >> 32), exponent);
    }

    /// <summary>
    /// Fake quantization exactly as the CPU computes it (in double): <paramref name="scale"/> 0 derives the step (and,
    /// asymmetric without a zero point, the zero point) from the tensor's min / max on the device.
    /// </summary>
    public static void FakeQuantize(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax,
        IGpuBuffer output, IGpuBuffer mask, int n, int numBits, bool symmetric, double scale, int? zeroPoint)
    {
        if (n <= 0) return;
        if (numBits < 1 || numBits > 30) throw new System.ArgumentOutOfRangeException(nameof(numBits), "numBits must be between 1 and 30.");
        if (double.IsNaN(scale) || double.IsInfinity(scale) || scale < 0)
            throw new System.ArgumentOutOfRangeException(nameof(scale), "scale must be a finite non-negative number (0 derives it).");
        if (!FakeQuantizeFitsDevice(numBits, symmetric, zeroPoint))
            throw new System.ArgumentOutOfRangeException(nameof(zeroPoint),
                $"qMin - zeroPoint and qMax - zeroPoint must lie within +-{FakeQuantizeMaxBound} on the device.");
        Require(input, n, nameof(input), "n");
        Require(output, n, nameof(output), "n");
        Require(mask, n, nameof(mask), "n");
        bool derive = scale == 0;
        bool deriveZeroPoint = derive && !symmetric && !zeroPoint.HasValue;
        var (qMin, qMax) = FakeQuantizeRange(numBits, symmetric);
        long zp = zeroPoint ?? 0;
        var (stepLo, stepHi, stepExp) = derive ? (0, 0, 0) : SplitStep(scale);
        IGpuBuffer bounds = input;   // unread when the step is fixed; any bound buffer serves
        if (derive)
        {
            int slots = System.Math.Min(n, TorchModelOpsCodes.MinMaxSlots);
            Require(partial, 2L * slots, nameof(partial), "2 * slots");
            Require(minMax, 2, nameof(minMax), "2");
            launch("mo_minmax_partial", slots, input, partial, n, slots);
            launch("mo_minmax_final", 1, partial, minMax, slots);
            bounds = minMax;
        }
        launch("mo_fake_quantize", n, input, bounds, output, mask, n, derive ? 1 : 0, symmetric ? 1 : 0, deriveZeroPoint ? 1 : 0,
            (int)qMin, (int)qMax, (int)(qMin - zp), (int)(qMax - zp), stepLo, stepHi, stepExp);
    }

    public static void TopKSoftmax(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer scratchValues,
        IGpuBuffer scratchIndices, IGpuBuffer scratchWeights, IGpuBuffer output, int rows, int width, int k)
    {
        long total = (long)rows * width;
        if (total <= 0) return;
        if (total > int.MaxValue) throw new System.ArgumentOutOfRangeException(nameof(rows), "rows * width must fit an int.");
        if (k < 1 || k > width) throw new System.ArgumentOutOfRangeException(nameof(k));
        Require(input, total, nameof(input), "rows * width");
        Require(output, total, nameof(output), "rows * width");
        var plan = TorchModelOpsCodes.PlanTopK(rows, width, k);
        if (!plan.RowGroups)
        {
            // Short rows: every element ranks itself against its row (width threads per row, O(width) each).
            Require(scratchValues, plan.ScratchLength, nameof(scratchValues), "PlanTopK scratch");
            launch("mo_topk_select", (int)total, input, scratchValues, rows, width, k);
            launch("mo_topk_normalize", (int)total, scratchValues, output, rows, width);
            return;
        }
        // Long rows: one work group of TopKGroupSize lanes per row (radix selection, then the softmax), one launch.
        Require(scratchIndices, plan.ScratchLength, nameof(scratchIndices), "PlanTopK scratch");
        Require(scratchWeights, plan.ScratchLength, nameof(scratchWeights), "PlanTopK scratch");
        launch("mo_topk_row", rows * TorchModelOpsCodes.TopKGroupSize, input, scratchIndices, scratchWeights, output, rows, width, k);
    }

    public static void TimeEmbedding(PortableGenericFft.Launcher launch, IGpuBuffer timesteps, IGpuBuffer frequencies,
        IGpuBuffer output, int batch, int halfDim, int dim)
    {
        long total = (long)batch * dim;
        if (total <= 0) return;
        if (total > int.MaxValue) throw new System.ArgumentOutOfRangeException(nameof(batch), "batch * dim must fit an int.");
        Require(timesteps, batch, nameof(timesteps), "batch");
        Require(frequencies, System.Math.Max(1, 3L * halfDim), nameof(frequencies), "3 * halfDim");
        Require(output, total, nameof(output), "batch * dim");
        launch("mo_time_embedding", (int)total, timesteps, frequencies, output, batch, halfDim, dim);
    }

    public static void TimeEmbeddingBackward(PortableGenericFft.Launcher launch, IGpuBuffer timesteps, IGpuBuffer frequencies,
        IGpuBuffer gradOutput, IGpuBuffer gradTimesteps, int batch, int halfDim, int dim)
    {
        if (batch <= 0) return;
        Require(timesteps, batch, nameof(timesteps), "batch");
        Require(frequencies, System.Math.Max(1, 3L * halfDim), nameof(frequencies), "3 * halfDim");
        Require(gradOutput, (long)batch * dim, nameof(gradOutput), "batch * dim");
        Require(gradTimesteps, batch, nameof(gradTimesteps), "batch");
        launch("mo_time_embedding_backward", batch, timesteps, frequencies, gradOutput, gradTimesteps, batch, halfDim, dim);
    }

    private static int CrfVolume(int batch, int seqLen, int tags)
    {
        long volume = (long)batch * seqLen * tags;
        if (batch <= 0 || seqLen <= 0 || tags <= 0) throw new System.ArgumentOutOfRangeException(nameof(batch), "batch, seqLen and tags must be positive.");
        if (volume > int.MaxValue || (long)tags * tags > int.MaxValue)
            throw new System.ArgumentOutOfRangeException(nameof(seqLen), "batch * seqLen * tags must fit an int.");
        return (int)volume;
    }

    public static void CrfForward(PortableGenericFft.Launcher launch, IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start,
        IGpuBuffer end, IGpuBuffer alpha, IGpuBuffer logZ, int batch, int seqLen, int tags)
    {
        int volume = CrfVolume(batch, seqLen, tags);
        Require(emissions, volume, nameof(emissions), "batch * seqLen * tags");
        Require(transitions, (long)tags * tags, nameof(transitions), "tags * tags");
        Require(start, tags, nameof(start), "tags");
        Require(end, tags, nameof(end), "tags");
        Require(alpha, CrfAlphaLength(batch, seqLen, tags), nameof(alpha), "CrfAlphaLength");
        Require(logZ, batch, nameof(logZ), "batch");
        // One work group of CrfGroupSize lanes per sequence; the kernel walks the time steps itself.
        launch("mo_crf_forward", CrfThreads(batch), emissions, transitions, start, end, alpha, logZ, batch, seqLen, tags);
    }

    /// <summary>Lanes per CRF work group: the group size every backend launches these kernels with.</summary>
    public const int CrfGroupSize = 256;

    private static int CrfThreads(int batch)
    {
        if ((long)batch * CrfGroupSize > int.MaxValue)
            throw new System.ArgumentOutOfRangeException(nameof(batch), "batch * 256 must fit an int.");
        return batch * CrfGroupSize;
    }

    public static void CrfBackward(PortableGenericFft.Launcher launch, IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end,
        IGpuBuffer alpha, IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions,
        IGpuBuffer gradStart, IGpuBuffer gradEnd, int batch, int seqLen, int tags)
    {
        int volume = CrfVolume(batch, seqLen, tags);
        Require(emissions, volume, nameof(emissions), "batch * seqLen * tags");
        Require(transitions, (long)tags * tags, nameof(transitions), "tags * tags");
        Require(end, tags, nameof(end), "tags");
        Require(alpha, CrfAlphaLength(batch, seqLen, tags), nameof(alpha), "CrfAlphaLength");
        Require(gradLogZ, batch, nameof(gradLogZ), "batch");
        Require(beta, CrfBetaLength(batch, seqLen, tags), nameof(beta), "CrfBetaLength");
        Require(gradEmissions, volume, nameof(gradEmissions), "batch * seqLen * tags");
        Require(gradTransitions, (long)tags * tags, nameof(gradTransitions), "tags * tags");
        Require(gradStart, tags, nameof(gradStart), "tags");
        Require(gradEnd, tags, nameof(gradEnd), "tags");
        // The backward recursion and the emission gradient, one work group per sequence; then the two sums over the batch.
        launch("mo_crf_backward", CrfThreads(batch), emissions, transitions, end, alpha, gradLogZ, beta, gradEmissions,
            batch, seqLen, tags);
        launch("mo_crf_boundary_grad", tags, gradEmissions, gradStart, gradEnd, batch, seqLen, tags);
        launch("mo_crf_transition_grad", tags * tags, emissions, transitions, alpha, beta, gradLogZ, gradTransitions,
            batch, seqLen, tags);
    }

    /// <summary>The forward's alpha buffer: the stored rows, then per sequence (seqLen + 1) (hi, lo) offset pairs.</summary>
    /// <remarks>The engine rejects larger problems up front (CpuEngine.MaxCrfStateLength, the same bound), so this only
    /// throws for a direct backend call; it throws the backend's argument exception rather than an overflow.</remarks>
    public static int CrfAlphaLength(int batch, int seqLen, int tags)
        => CrfLength(CrfAlphaLengthLong(batch, seqLen, tags), "batch * seqLen * tags + 2 * batch * (seqLen + 1)");

    /// <summary><see cref="CrfAlphaLength"/> in 64 bits, never overflowing for int dimensions.</summary>
    public static long CrfAlphaLengthLong(int batch, int seqLen, int tags)
        => (long)batch * seqLen * tags + 2L * batch * (seqLen + 1L);

    /// <summary>The backward's beta scratch: the stored rows, then per sequence seqLen (G, H) exponent pairs.</summary>
    public static int CrfBetaLength(int batch, int seqLen, int tags)
        => CrfLength((long)batch * seqLen * tags + 2L * batch * seqLen, "batch * seqLen * tags + 2 * batch * seqLen");

    private static int CrfLength(long length, string formula)
    {
        if (length > int.MaxValue)
            throw new System.ArgumentOutOfRangeException("batch", $"The CRF state {formula} = {length} exceeds {int.MaxValue} elements.");
        return (int)length;
    }
}
