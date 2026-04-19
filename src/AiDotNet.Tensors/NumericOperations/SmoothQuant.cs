namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Activation-to-weight scale migration — the pre-quantization transform
/// proposed by SmoothQuant (Xiao et al., 2023) that migrates the
/// activation range into the weight range so both tensors become
/// quantization-friendly. For a linear layer <c>Y = X · W</c> the
/// transform is:
/// <code>
/// s_j = max|X_j|^α / max|W_j|^(1-α)     per output feature j
/// X' = X · diag(1/s),  W' = diag(s) · W
/// Y = X' · W'                             (algebraically identical to X · W)
/// </code>
/// The scale vector <c>s</c> is chosen so that after the transform both
/// X and W have balanced per-channel magnitudes — so int8 / int4
/// quantization of either wastes fewer levels on outliers.
///
/// <para><b>α</b> is the migration strength in [0, 1]. α=0 leaves W
/// unchanged (pure activation-side quant); α=1 leaves X unchanged.
/// The paper recommends 0.5 for LLMs; 0.75–0.85 for models with sharper
/// activation outliers.</para>
///
/// <para>Integration path: compute <c>s</c> from calibration data, then
/// multiply weights by <c>diag(s)</c> once at build time and multiply
/// activations by <c>diag(1/s)</c> at runtime (or fold into the layer's
/// input normalization).</para>
/// </summary>
public static class SmoothQuant
{
    /// <summary>
    /// Compute the per-channel smoothing factor <c>s</c>.
    /// </summary>
    /// <param name="activationAbsMax">Per-channel max absolute value of
    /// the activation tensor, length C (input feature count).</param>
    /// <param name="weightAbsMax">Per-channel max absolute value of
    /// the weight tensor along the same axis, length C.</param>
    /// <param name="alpha">Migration strength in [0, 1]. 0.5 default.</param>
    /// <param name="eps">Numerical floor added to both magnitudes so
    /// zero-columns don't blow up the division.</param>
    public static float[] ComputeSmoothingFactor(
        ReadOnlySpan<float> activationAbsMax,
        ReadOnlySpan<float> weightAbsMax,
        float alpha = 0.5f,
        float eps = 1e-5f)
    {
        if (activationAbsMax.Length != weightAbsMax.Length)
            throw new ArgumentException(
                $"Length mismatch: activation {activationAbsMax.Length} vs weight {weightAbsMax.Length}.");
        if (alpha < 0f || alpha > 1f)
            throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be in [0, 1].");
        // eps is the numerical floor for both magnitudes and the
        // denominator of a 1/eps clamp. Zero, negative, NaN, or Inf all
        // poison the clamp range silently; reject at the boundary.
        // Using IsNaN || IsInfinity (not IsFinite) for net471 compat.
        if (eps <= 0f || float.IsNaN(eps) || float.IsInfinity(eps))
            throw new ArgumentOutOfRangeException(
                nameof(eps), "eps must be finite and > 0.");

        int c = activationAbsMax.Length;
        var s = new float[c];
        for (int j = 0; j < c; j++)
        {
            float ax = Math.Max(activationAbsMax[j], eps);
            float wx = Math.Max(weightAbsMax[j], eps);
            // Clamp result to [eps, 1/eps] so a degenerate column doesn't
            // produce NaN / Inf downstream.
            float v = (float)(Math.Pow(ax, alpha) / Math.Pow(wx, 1 - alpha));
            if (float.IsNaN(v) || float.IsInfinity(v)) v = 1f;
            // Manual clamp for net471 compatibility (Math.Clamp is net5+).
            float inv = 1f / eps;
            if (v < eps) v = eps;
            else if (v > inv) v = inv;
            s[j] = v;
        }
        return s;
    }

    /// <summary>
    /// Apply the smoothing factor to a weight tensor in place. Weight
    /// layout is row-major <c>[outC, inC]</c>; we multiply each column
    /// <c>j</c> by <c>s[j]</c>.
    /// </summary>
    public static void ApplyToWeights(
        Span<float> weights, int outC, int inC, ReadOnlySpan<float> smoothingFactor)
    {
        if (outC < 0)
            throw new ArgumentOutOfRangeException(nameof(outC), "must be non-negative.");
        if (inC < 0)
            throw new ArgumentOutOfRangeException(nameof(inC), "must be non-negative.");
        if (smoothingFactor.Length != inC)
            throw new ArgumentException(
                $"smoothingFactor length {smoothingFactor.Length} must match inC {inC}.");
        // int multiplication wraps silently on realistic LLM shapes
        // (e.g. 50k × 50k overflows int32), which would let an
        // undersized weights buffer slip past the length check. Do the
        // product in long.
        long required = (long)outC * inC;
        if (weights.Length < required)
            throw new ArgumentException(
                $"weights length {weights.Length} < outC*inC = {required}.", nameof(weights));
        for (int i = 0; i < outC; i++)
        {
            int rowBase = i * inC;
            for (int j = 0; j < inC; j++)
                weights[rowBase + j] *= smoothingFactor[j];
        }
    }

    /// <summary>
    /// Apply the inverse smoothing factor to an activation tensor in
    /// place. Activation layout is <c>[batch, inC]</c>; we divide each
    /// column <c>j</c> by <c>s[j]</c>.
    /// </summary>
    public static void ApplyToActivations(
        Span<float> activations, int batch, int inC, ReadOnlySpan<float> smoothingFactor)
    {
        if (batch < 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "must be non-negative.");
        if (inC < 0)
            throw new ArgumentOutOfRangeException(nameof(inC), "must be non-negative.");
        if (smoothingFactor.Length != inC)
            throw new ArgumentException(
                $"smoothingFactor length {smoothingFactor.Length} must match inC {inC}.");
        long required = (long)batch * inC;
        if (activations.Length < required)
            throw new ArgumentException(
                $"activations length {activations.Length} < batch*inC = {required}.",
                nameof(activations));
        for (int n = 0; n < batch; n++)
        {
            int rowBase = n * inC;
            for (int j = 0; j < inC; j++)
                activations[rowBase + j] /= smoothingFactor[j];
        }
    }

    /// <summary>
    /// Convenience helper: compute per-channel absmax of a
    /// <c>[rows, cols]</c> row-major float tensor along the column
    /// axis. Used to feed
    /// <see cref="ComputeSmoothingFactor"/>.
    /// </summary>
    public static float[] PerColumnAbsMax(ReadOnlySpan<float> data, int rows, int cols)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "must be non-negative.");
        if (cols < 0)
            throw new ArgumentOutOfRangeException(nameof(cols), "must be non-negative.");
        long required = (long)rows * cols;
        if (data.Length < required)
            throw new ArgumentException(
                $"data length {data.Length} < rows*cols = {required}.", nameof(data));
        var result = new float[cols];
        for (int i = 0; i < rows; i++)
        {
            int rowBase = i * cols;
            for (int j = 0; j < cols; j++)
            {
                float a = Math.Abs(data[rowBase + j]);
                if (a > result[j]) result[j] = a;
            }
        }
        return result;
    }
}
