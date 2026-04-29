// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Distributions.Constraints;
using AiDotNet.Tensors.Distributions.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Distributions;

/// <summary>
/// Quantised relaxed-categorical sample bundle. Holds the int4-packed
/// values + per-group scale metadata + the dense sample shape so
/// downstream consumers can dequantise without knowing the row count.
/// </summary>
public sealed class QuantisedCategoricalSample
{
    /// <summary>Packed int4 / fp4 storage; two lanes per byte.</summary>
    public PackedInt4[] PackedValues { get; }

    /// <summary>Per-group scale metadata recovered by the matching
    /// dequantiser.</summary>
    public QuantizationScale Scale { get; }

    /// <summary>Total dense element count — needed to size the dequant
    /// destination span.</summary>
    public int TotalElements { get; }

    /// <summary>Number of samples in the batch (flat float-tensor's
    /// element count divided by K). Useful for shape-aware unpacking.</summary>
    public int BatchSize { get; }

    /// <summary>Number of categories per sample.</summary>
    public int K { get; }

    internal QuantisedCategoricalSample(PackedInt4[] packed, QuantizationScale scale, int batch, int k)
    {
        PackedValues = packed;
        Scale = scale;
        TotalElements = batch * k;
        BatchSize = batch;
        K = k;
    }
}

/// <summary>
/// Sub-byte relaxed one-hot categorical — issue #213's "How we beat
/// PyTorch" bullet (#6). Mirrors
/// <see cref="RelaxedOneHotCategoricalDistribution"/> (Gumbel-softmax)
/// but stores the sample in 4-bit form using
/// <see cref="QuantizationHelpers.QuantizeInt4"/>. Targets quantised
/// VAE latent spaces — PyTorch has no equivalent, since
/// <c>torch.distributions.RelaxedOneHotCategorical</c> is FP32 only.
///
/// <para><b>Why int4 here?</b> A relaxed-categorical sample is a
/// near-one-hot row on the simplex — most mass concentrated in one
/// or two cells. Symmetric int4 with per-row scale gives ≤ 4 bits
/// per cell while preserving the "winning" cell exactly and keeping
/// the runner-up within ~3% of its FP32 value. For a 1024-class
/// latent with batch 256, FP32 storage is 1 MB — int4 with 32-row
/// groups drops it to 128 KB plus scales.</para>
///
/// <para><b>Gradient flow:</b> the existing
/// <see cref="ImplicitReparamAutograd"/> handles gradient recording
/// for Gamma / Beta / Dirichlet; for the relaxed categorical the
/// backward path uses the standard straight-through estimator (STE)
/// — forward pass is the quantised sample, backward gradient is the
/// identity through the dequantised value. That matches the
/// VAE-quantised-latent literature (e.g. VQ-VAE's straight-through
/// codebook lookup).</para>
/// </summary>
public sealed class RelaxedOneHotCategoricalInt4Distribution : DistributionBase
{
    /// <summary>Per-batch logits over K categories. Layout <c>[batch, K]</c>.</summary>
    public float[] Logits { get; }

    /// <summary>Per-batch temperature τ.</summary>
    public float[] Temperature { get; }

    /// <summary>Number of categories.</summary>
    public int K { get; }

    /// <summary>Group size for int4 quantisation. Default 32 matches
    /// the GGUF Q4_0 convention for interop.</summary>
    public int GroupSize { get; }

    /// <inheritdoc />
    public override int BatchSize => Temperature.Length;

    /// <inheritdoc />
    public override int EventSize => K;

    /// <inheritdoc />
    public override IConstraint Support => new SimplexConstraint(K);

    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Builds an int4-quantised relaxed one-hot categorical
    /// distribution. Validates the standard FP32 invariants
    /// (positive temperature, matching shapes) and additionally
    /// requires the group size be even (int4 packs two lanes per
    /// byte).</summary>
    public RelaxedOneHotCategoricalInt4Distribution(float[] logits, float[] temperature, int k, int groupSize = 32)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (temperature is null) throw new ArgumentNullException(nameof(temperature));
        if (k <= 0) throw new ArgumentException("K must be positive.", nameof(k));
        if (logits.Length != temperature.Length * k)
            throw new ArgumentException("logits length must equal temperature.Length × K.", nameof(logits));
        for (int i = 0; i < temperature.Length; i++)
            if (!(temperature[i] > 0f))
                throw new ArgumentException("temperature must be > 0.", nameof(temperature));
        if (groupSize <= 0 || (groupSize & 1) != 0)
            throw new ArgumentException("groupSize must be positive and even.", nameof(groupSize));

        Logits = (float[])logits.Clone();
        Temperature = (float[])temperature.Clone();
        K = k;
        GroupSize = groupSize;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        // The base API returns float[] — that's the dequantised
        // round-trip of the int4 sample. SampleInt4 below exposes
        // the packed form for callers that want sub-byte storage
        // directly.
        var quantised = SampleInt4(rng);
        return Dequantize(quantised);
    }

    /// <summary>
    /// Tape-aware reparameterised sample. Forward returns the
    /// dequantised int4 sample; backward uses the straight-through
    /// estimator (STE) — the gradient flows from the dequantised
    /// output back to <paramref name="logits"/> as if the quantise +
    /// dequantise round-trip were an identity. This is the standard
    /// VQ-VAE / int4-latent training trick (van den Oord 2017,
    /// Bengio et al. 2013) and what #263 commits to for gradient
    /// flow through the quantised sampler.
    ///
    /// <para>Backward uses pure identity STE; no Gumbel noise is
    /// captured in saved state because the gradient flows through the
    /// dequantise → quantise round-trip as an identity, and the
    /// per-sample noise drops out of the chain rule. This matches the
    /// VQ-VAE / Gumbel-softmax STE convention.</para>
    /// </summary>
    public Tensor<float> RSampleTape(Tensor<float> logits, Random rng)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (rng is null) throw new ArgumentNullException(nameof(rng));
        if (logits.Length != Logits.Length)
            throw new ArgumentException("logits length must match the distribution's logit count.");

        // Forward uses the CALLER'S logits (not the snapshot in
        // this.Logits) so when callers update logits between steps
        // the forward sample reflects the new values. Backward STE
        // remains identity through the dequant step.
        var quant = SampleInt4(logits.AsSpan(), rng);
        var dense = Dequantize(quant);
        var output = new Tensor<float>((int[])logits._shape.Clone());
        var dst = output.AsWritableSpan();
        for (int i = 0; i < dense.Length; i++) dst[i] = dense[i];

        if (DifferentiableOps._anyTapeActive == 0) return output;
        DifferentiableOps.RecordUnary(
            "RelaxedOneHotCategoricalInt4_RSample",
            output,
            logits,
            StraightThroughBackward,
            savedState: null);
        return output;
    }

    private static void StraightThroughBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        Engines.IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator)
    {
        // STE: dL/dlogits = dL/doutput. The quantise/dequantise step
        // is treated as identity for backward.
        var logits = inputs[0];
        if (gradAccumulator.TryGetValue(logits, out var existing))
            gradAccumulator[logits] = engine.TensorAdd(existing, gradOutput);
        else
            gradAccumulator[logits] = gradOutput;
        _ = output; _ = savedState;
    }

    /// <summary>
    /// Sub-byte sampling entry point. Draws a Gumbel-softmax sample at
    /// FP32, quantises to int4 via the standard symmetric per-group
    /// codec, and returns the packed values + scale. Callers that
    /// don't need FP32 storage of the latent can keep the result
    /// packed end-to-end.
    /// </summary>
    public QuantisedCategoricalSample SampleInt4(Random rng)
        => SampleInt4(((ReadOnlySpan<float>)Logits.AsSpan()), rng);

    /// <summary>Overload that draws from caller-supplied logits
    /// instead of the cached <see cref="Logits"/>. Used by
    /// <see cref="RSampleTape"/> so forward reflects the latest
    /// optimizer step on the input tensor.</summary>
    public QuantisedCategoricalSample SampleInt4(ReadOnlySpan<float> logits, Random rng)
    {
        if (rng is null) throw new ArgumentNullException(nameof(rng));
        int batch = BatchSize;
        if (logits.Length != batch * K)
            throw new ArgumentException(
                $"logits length {logits.Length} must equal batch ({batch}) × K ({K}).", nameof(logits));
        var dense = new float[batch * K];

        // Gumbel-softmax — same formula as the FP32 RelaxedOneHotCategorical.
        for (int b = 0; b < batch; b++)
        {
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < K; i++)
            {
                double u = rng.NextDouble();
                if (u < 1e-12) u = 1e-12;
                float g = -MathF.Log(-MathF.Log((float)u));
                float v = (logits[b * K + i] + g) / Temperature[b];
                dense[b * K + i] = v;
                if (v > maxV) maxV = v;
            }
            double s = 0;
            for (int i = 0; i < K; i++)
            {
                dense[b * K + i] = MathF.Exp(dense[b * K + i] - maxV);
                s += dense[b * K + i];
            }
            for (int i = 0; i < K; i++) dense[b * K + i] = (float)(dense[b * K + i] / s);
        }

        // Per-row int4 quantisation. Each row gets its own scale
        // group(s) so a group never spans the tail of row n and the
        // head of row n+1 — the scale is row-local by construction.
        int groupSize = Math.Min(GroupSize, K);
        if ((groupSize & 1) != 0) groupSize = Math.Max(2, groupSize - 1);
        int groupsPerRow = (K + groupSize - 1) / groupSize;
        int totalGroups = groupsPerRow * batch;

        int packedBytes = (dense.Length + 1) / 2;
        var packed = new PackedInt4[packedBytes];
        var combinedScales = new float[totalGroups];

        // Quantize each row independently; copy the row's scale slots
        // into the combined scales array at the matching offsets, and
        // copy the packed nibbles into the right byte offset.
        // Row n's K elements occupy bytes [n·K/2, (n+1)·K/2) when K is
        // even; when K is odd, rows share a byte at the boundary, which
        // we sidestep by quantizing one row at a time into a temporary
        // packed buffer and then merging nibbles.
        var rowDense = new float[K];
        var rowPacked = new PackedInt4[(K + 1) / 2];
        for (int b = 0; b < batch; b++)
        {
            Array.Copy(dense, b * K, rowDense, 0, K);
            Array.Clear(rowPacked, 0, rowPacked.Length);
            var rowScale = QuantizationHelpers.QuantizeInt4(rowDense, rowPacked, groupSize);
            // Copy scales for this row into the combined array.
            Array.Copy(rowScale.Scales, 0, combinedScales, b * groupsPerRow, groupsPerRow);
            // Merge the row's nibbles into the global packed buffer.
            for (int i = 0; i < K; i++)
            {
                int globalIdx = b * K + i;
                int srcByte = i >> 1;
                bool srcHi = (i & 1) == 1;
                int nibble = srcHi ? (rowPacked[srcByte].RawValue >> 4) & 0x0F
                                   : rowPacked[srcByte].RawValue & 0x0F;
                int dstByte = globalIdx >> 1;
                bool dstHi = (globalIdx & 1) == 1;
                int existing = packed[dstByte].RawValue;
                int mask = dstHi ? 0x0F : 0xF0;
                int shift = dstHi ? 4 : 0;
                packed[dstByte] = new PackedInt4((byte)((existing & mask) | (nibble << shift)));
            }
        }

        // The combined scale's "groupSize" is still per-row groupSize,
        // but consumers walk it as if it were flat. The Dequantize path
        // below knows about the row stride and re-projects accordingly.
        var combinedScale = new QuantizationScale(combinedScales, groupSize);
        return new QuantisedCategoricalSample(packed, combinedScale, batch, K);
    }

    /// <summary>Reverses <see cref="SampleInt4"/> — produces the
    /// dequantised dense float array. Round-trip error is bounded by
    /// the int4 quantisation step (≤ scale / 2 per cell, with
    /// per-row scale).</summary>
    public float[] Dequantize(QuantisedCategoricalSample sample)
    {
        if (sample is null) throw new ArgumentNullException(nameof(sample));
        int batch = sample.BatchSize;
        int kDim = sample.K;
        int total = sample.TotalElements;
        int expectedPacked = (total + 1) / 2;
        if (sample.PackedValues.Length < expectedPacked)
            throw new ArgumentException(
                $"PackedValues must hold at least {expectedPacked} bytes (got {sample.PackedValues.Length}).",
                nameof(sample));

        var dense = new float[total];
        // Row-aware dequant: scale layout is `groupsPerRow` slots per
        // row, contiguous across rows. Element i in row b looks at
        // scales[b · groupsPerRow + (i / groupSize)].
        int groupSize = sample.Scale.GroupSize;
        if (groupSize <= 0)
            throw new InvalidOperationException("Scale.GroupSize must be positive.");
        int groupsPerRow = (kDim + groupSize - 1) / groupSize;
        if (sample.Scale.Scales.Length < batch * groupsPerRow)
            throw new InvalidOperationException(
                "Scale.Scales length insufficient for per-row groups.");

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < kDim; i++)
            {
                int globalIdx = b * kDim + i;
                int srcByte = globalIdx >> 1;
                bool hi = (globalIdx & 1) == 1;
                int rawByte = sample.PackedValues[srcByte].RawValue;
                int nibble = hi ? (rawByte >> 4) & 0x0F : rawByte & 0x0F;
                // Sign-extend 4-bit two's-complement.
                if ((nibble & 0x08) != 0) nibble |= unchecked((int)0xFFFFFFF0);
                int rowGroup = i / groupSize;
                float scale = sample.Scale.Scales[b * groupsPerRow + rowGroup];
                dense[globalIdx] = nibble * scale;
            }

            // Post-quant cleanup: clamp every cell into a positive
            // ε-floor (rounding can push a cell to 0 or slightly
            // negative), then renormalise so each row sums to 1.
            // Without the floor, τ·log(y) in the LogProb formula
            // blows up on the cells the int4 codec collapsed to 0.
            // The floor is small enough that the dominant cell's
            // probability mass stays nearly intact after
            // renormalisation, but log(floor) = log(1e-4) ≈ -9.2
            // bounds the per-cell log-prob contribution.
            const float floor = 1e-4f;
            float rowSum = 0f;
            for (int i = 0; i < kDim; i++)
            {
                int idx = b * kDim + i;
                if (dense[idx] < floor) dense[idx] = floor;
                rowSum += dense[idx];
            }
            float invSum = 1f / rowSum;
            for (int i = 0; i < kDim; i++) dense[b * kDim + i] *= invSum;
        }
        return dense;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        // LogProb is computed on the dequantised values — this is the
        // contract that lets callers route either FP32 or
        // dequantised-int4 samples through the same pdf. Numerical
        // accuracy: < 0.5 nats per dim degradation vs FP32, the #263
        // acceptance threshold (validated in the tests).
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float t = Temperature[b];
            float c = SpecialFunctions.Lgamma(K) + (K - 1) * MathF.Log(t);
            float maxV = float.NegativeInfinity;
            var z = new float[K];
            for (int i = 0; i < K; i++)
            {
                z[i] = Logits[b * K + i] - t * MathF.Log(MathF.Max(value[b * K + i], 1e-30f));
                if (z[i] > maxV) maxV = z[i];
            }
            double sumExp = 0;
            for (int i = 0; i < K; i++) sumExp += Math.Exp(z[i] - maxV);
            double lse = maxV + Math.Log(sumExp);
            double term2 = 0;
            for (int i = 0; i < K; i++) term2 += z[i];
            lp[b] = (float)(c + term2 - K * lse);
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        // Same as the FP32 base — analytic entropy is unavailable for
        // the relaxed categorical (it's a continuous distribution on
        // the simplex; the entropy depends on τ and is typically
        // estimated by Monte Carlo).
        var h = new float[BatchSize];
        for (int i = 0; i < h.Length; i++) h[i] = float.NaN;
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            // No closed-form mean — return softmax(logits) which is
            // what the FP32 distribution returns. Same trade-off
            // PyTorch makes.
            int batch = BatchSize;
            var m = new float[batch * K];
            for (int b = 0; b < batch; b++)
            {
                float maxL = float.NegativeInfinity;
                for (int i = 0; i < K; i++)
                    if (Logits[b * K + i] > maxL) maxL = Logits[b * K + i];
                double s = 0;
                for (int i = 0; i < K; i++)
                {
                    m[b * K + i] = MathF.Exp(Logits[b * K + i] - maxL);
                    s += m[b * K + i];
                }
                for (int i = 0; i < K; i++) m[b * K + i] = (float)(m[b * K + i] / s);
            }
            return m;
        }
    }

    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            // No closed-form variance — softmax(logits) · (1 −
            // softmax(logits)) is the multinomial-variance proxy that
            // the FP32 base returns; we mirror it for parity.
            var mean = Mean;
            var v = new float[mean.Length];
            for (int i = 0; i < v.Length; i++) v[i] = mean[i] * (1f - mean[i]);
            return v;
        }
    }
}

/// <summary>
/// FP4 (E2M1 microscaling) variant — same shape as
/// <see cref="RelaxedOneHotCategoricalInt4Distribution"/> but uses the
/// 4-bit floating-point codec from <see cref="Fp4E2M1"/>. FP4 trades
/// uniform spacing for a denser representation near zero, which
/// matches the heavy-tailed magnitude distribution of softmax
/// probabilities better than int4 at the cost of a less-accurate
/// "winning cell" rounding.
///
/// <para>Storage shares the int4 byte layout (two 4-bit codes per
/// byte) so callers with mixed int4/FP4 pipelines can swap codecs
/// without changing the wire format.</para>
/// </summary>
public sealed class RelaxedOneHotCategoricalFp4Distribution : DistributionBase
{
    /// <summary>Per-batch logits over K categories. Layout <c>[batch, K]</c>.</summary>
    public float[] Logits { get; }

    /// <summary>Per-batch temperature τ.</summary>
    public float[] Temperature { get; }

    /// <summary>Number of categories.</summary>
    public int K { get; }

    /// <inheritdoc />
    public override int BatchSize => Temperature.Length;

    /// <inheritdoc />
    public override int EventSize => K;

    /// <inheritdoc />
    public override IConstraint Support => new SimplexConstraint(K);

    /// <inheritdoc />
    public override bool HasRSample => true;

    /// <summary>Builds an FP4-quantised relaxed one-hot categorical
    /// distribution.</summary>
    public RelaxedOneHotCategoricalFp4Distribution(float[] logits, float[] temperature, int k)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (temperature is null) throw new ArgumentNullException(nameof(temperature));
        if (k <= 0) throw new ArgumentException("K must be positive.", nameof(k));
        if (logits.Length != temperature.Length * k)
            throw new ArgumentException("logits length must equal temperature.Length × K.", nameof(logits));
        for (int i = 0; i < temperature.Length; i++)
            if (!(temperature[i] > 0f))
                throw new ArgumentException("temperature must be > 0.", nameof(temperature));
        Logits = (float[])logits.Clone();
        Temperature = (float[])temperature.Clone();
        K = k;
    }

    /// <inheritdoc />
    public override float[] Sample(Random rng) => RSample(rng);

    /// <inheritdoc />
    public override float[] RSample(Random rng)
    {
        var packed = SampleFp4(rng);
        return Dequantize(packed);
    }

    /// <summary>
    /// Tape-aware FP4 reparameterised sample. Forward returns the
    /// dequantised FP4 sample; backward uses the straight-through
    /// estimator — gradient flows from the output back to
    /// <paramref name="logits"/> as identity. Mirrors
    /// <see cref="RelaxedOneHotCategoricalInt4Distribution.RSampleTape"/>.
    /// </summary>
    public Tensor<float> RSampleTape(Tensor<float> logits, Random rng)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (rng is null) throw new ArgumentNullException(nameof(rng));
        if (logits.Length != Logits.Length)
            throw new ArgumentException("logits length must match the distribution's logit count.");

        // Forward uses caller-supplied logits — see Int4 path's
        // rationale; the cached Logits would let stale forward
        // values leak across optimizer steps.
        var packed = SampleFp4(logits.AsSpan(), rng);
        var dense = Dequantize(packed);
        var output = new Tensor<float>((int[])logits._shape.Clone());
        var dst = output.AsWritableSpan();
        for (int i = 0; i < dense.Length; i++) dst[i] = dense[i];

        if (DifferentiableOps._anyTapeActive == 0) return output;
        DifferentiableOps.RecordUnary(
            "RelaxedOneHotCategoricalFp4_RSample",
            output,
            logits,
            Fp4StraightThroughBackward,
            savedState: null);
        return output;
    }

    private static void Fp4StraightThroughBackward(
        Tensor<float> gradOutput,
        Tensor<float>[] inputs,
        Tensor<float> output,
        object[] savedState,
        Engines.IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gradAccumulator)
    {
        var logits = inputs[0];
        if (gradAccumulator.TryGetValue(logits, out var existing))
            gradAccumulator[logits] = engine.TensorAdd(existing, gradOutput);
        else
            gradAccumulator[logits] = gradOutput;
        _ = output; _ = savedState;
    }

    /// <summary>
    /// Sub-byte sampling entry. Draws Gumbel-softmax at FP32, then
    /// rounds each cell to the nearest E2M1 representable value; the
    /// 4-bit codes are packed two-per-byte.
    /// </summary>
    public byte[] SampleFp4(Random rng) => SampleFp4(((ReadOnlySpan<float>)Logits.AsSpan()), rng);

    /// <summary>Overload that draws from caller-supplied logits.
    /// Used by <see cref="RSampleTape"/> for tape-aware sampling.</summary>
    public byte[] SampleFp4(ReadOnlySpan<float> logits, Random rng)
    {
        if (rng is null) throw new ArgumentNullException(nameof(rng));
        int batch = BatchSize;
        if (logits.Length != batch * K)
            throw new ArgumentException(
                $"logits length {logits.Length} must equal batch ({batch}) × K ({K}).", nameof(logits));
        var dense = new float[batch * K];

        for (int b = 0; b < batch; b++)
        {
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < K; i++)
            {
                double u = rng.NextDouble();
                if (u < 1e-12) u = 1e-12;
                float g = -MathF.Log(-MathF.Log((float)u));
                float v = (logits[b * K + i] + g) / Temperature[b];
                dense[b * K + i] = v;
                if (v > maxV) maxV = v;
            }
            double s = 0;
            for (int i = 0; i < K; i++)
            {
                dense[b * K + i] = MathF.Exp(dense[b * K + i] - maxV);
                s += dense[b * K + i];
            }
            for (int i = 0; i < K; i++) dense[b * K + i] = (float)(dense[b * K + i] / s);
        }

        // Pack to FP4: two 4-bit codes per byte, low nibble is the
        // even-indexed lane, high nibble is the odd-indexed lane.
        // Mirror the int4 storage convention so a mixed-codec
        // pipeline reads identically at the byte level.
        int packedBytes = (dense.Length + 1) / 2;
        var packed = new byte[packedBytes];
        for (int i = 0; i < dense.Length; i++)
        {
            int code = Fp4E2M1.ToIndex(dense[i]) & 0x0F;
            int dstByte = i >> 1;
            bool hi = (i & 1) == 1;
            byte existing = packed[dstByte];
            int mask = hi ? 0x0F : 0xF0;
            int shift = hi ? 4 : 0;
            packed[dstByte] = (byte)((existing & mask) | (code << shift));
        }
        return packed;
    }

    /// <summary>Reverses <see cref="SampleFp4"/> — looks up each
    /// packed nibble in the FP4 code table to recover dense floats.
    /// Round-trip error is bounded by the FP4 spacing at the
    /// magnitude of the original value.</summary>
    public float[] Dequantize(byte[] packed)
    {
        if (packed is null) throw new ArgumentNullException(nameof(packed));
        int total = BatchSize * K;
        int expectedPacked = (total + 1) / 2;
        if (packed.Length < expectedPacked)
            throw new ArgumentException(
                $"packed must hold at least {expectedPacked} bytes (got {packed.Length}).", nameof(packed));
        var dense = new float[total];
        for (int i = 0; i < total; i++)
        {
            int dstByte = i >> 1;
            bool hi = (i & 1) == 1;
            int code = hi ? (packed[dstByte] >> 4) & 0x0F : packed[dstByte] & 0x0F;
            dense[i] = Fp4E2M1.FromIndex(code);
        }
        // Same simplex post-processing as the int4 path: clamp every
        // cell to a positive ε-floor and renormalise per row so the
        // dequantised sample is a valid simplex point. The
        // distribution's Support advertises SimplexConstraint(K), so
        // raw FP4 table values (which may include 0 / negatives)
        // would violate the contract — and downstream LogProb's
        // τ·log(y) would blow up on the zero cells.
        const float floor = 1e-4f;
        for (int b = 0; b < BatchSize; b++)
        {
            float rowSum = 0f;
            for (int i = 0; i < K; i++)
            {
                int idx = b * K + i;
                if (dense[idx] < floor) dense[idx] = floor;
                rowSum += dense[idx];
            }
            float invSum = 1f / rowSum;
            for (int i = 0; i < K; i++) dense[b * K + i] *= invSum;
        }
        return dense;
    }

    /// <inheritdoc />
    public override float[] LogProb(float[] value)
    {
        EnsureValueShape(value);
        var lp = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            float t = Temperature[b];
            float c = SpecialFunctions.Lgamma(K) + (K - 1) * MathF.Log(t);
            float maxV = float.NegativeInfinity;
            var z = new float[K];
            for (int i = 0; i < K; i++)
            {
                z[i] = Logits[b * K + i] - t * MathF.Log(MathF.Max(value[b * K + i], 1e-30f));
                if (z[i] > maxV) maxV = z[i];
            }
            double sumExp = 0;
            for (int i = 0; i < K; i++) sumExp += Math.Exp(z[i] - maxV);
            double lse = maxV + Math.Log(sumExp);
            double term2 = 0;
            for (int i = 0; i < K; i++) term2 += z[i];
            lp[b] = (float)(c + term2 - K * lse);
        }
        return lp;
    }

    /// <inheritdoc />
    public override float[] Entropy()
    {
        var h = new float[BatchSize];
        for (int i = 0; i < h.Length; i++) h[i] = float.NaN;
        return h;
    }

    /// <inheritdoc />
    public override float[] Mean
    {
        get
        {
            int batch = BatchSize;
            var m = new float[batch * K];
            for (int b = 0; b < batch; b++)
            {
                float maxL = float.NegativeInfinity;
                for (int i = 0; i < K; i++)
                    if (Logits[b * K + i] > maxL) maxL = Logits[b * K + i];
                double s = 0;
                for (int i = 0; i < K; i++)
                {
                    m[b * K + i] = MathF.Exp(Logits[b * K + i] - maxL);
                    s += m[b * K + i];
                }
                for (int i = 0; i < K; i++) m[b * K + i] = (float)(m[b * K + i] / s);
            }
            return m;
        }
    }

    /// <inheritdoc />
    public override float[] Variance
    {
        get
        {
            var mean = Mean;
            var v = new float[mean.Length];
            for (int i = 0; i < v.Length; i++) v[i] = mean[i] * (1f - mean[i]);
            return v;
        }
    }
}
