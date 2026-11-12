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

    public static void FakeQuantize(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax,
        IGpuBuffer output, IGpuBuffer mask, int n, bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax,
        float fixedStep, float zeroPoint)
    {
        if (n <= 0) return;
        Require(input, n, nameof(input), "n");
        Require(output, n, nameof(output), "n");
        Require(mask, n, nameof(mask), "n");
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
        launch("mo_fake_quantize", n, input, bounds, output, mask, n, derive ? 1 : 0, symmetric ? 1 : 0, hasZeroPoint ? 1 : 0,
            qMin, qMax, fixedStep, zeroPoint);
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
    public static int CrfAlphaLength(int batch, int seqLen, int tags)
        => checked(batch * seqLen * tags + batch * (seqLen + 1) * 2);

    /// <summary>The backward's beta scratch: the stored rows, then per sequence seqLen (G, H) exponent pairs.</summary>
    public static int CrfBetaLength(int batch, int seqLen, int tags)
        => checked(batch * seqLen * tags + batch * seqLen * 2);
}
