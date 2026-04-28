using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// SparseAdam on NVIDIA 2:4 structured sparsity.
///
/// In 2:4 sparse format, every 4-element block has exactly 2 zeros at known
/// positions, encoded as a 4-bit nibble per block: bits 0-1 select the first
/// non-zero index, bits 2-3 select the second. The dense gradient buffer of
/// length <c>N</c> is therefore equivalently represented by:
///   * <c>values</c>      — packed array of length <c>N/2</c> with the two non-zero values per block
///   * <c>indices_nibble</c> — array of length <c>N/4</c> with one 4-bit nibble per block
///
/// SparseAdam-2:4 only touches the 2 of 4 positions per block that the sparsity
/// pattern marks live, mirroring the savings of NVIDIA's <c>cuSPARSELt</c> 2:4
/// matmul. Compared to dense Adam, this halves both the FLOPs and the moment-buffer
/// reads/writes; compared to <see cref="SparseAdamOptimizer"/> (which scans for
/// zeros at runtime) the structured 2:4 layout is branch-free and CUDA-Graph safe.
///
/// This is the implementation of the issue #224 bullet:
/// <i>"SparseAdam on sub-byte sparsity: when gradients come out of a 2:4 structured
/// path, SparseAdam update touches only the live channels in the packed
/// representation. Unique."</i>
/// </summary>
[CudaGraphSafe(Note = "sparsity pattern is metadata, not a runtime tensor decision")]
public sealed class SparseAdam24Optimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    private readonly Dictionary<(int gi, int pi), byte[]> _patterns = new();

    /// <summary>
    /// Add a 2:4-sparse parameter. <paramref name="parameter"/> length must be a
    /// multiple of 4. <paramref name="gradient"/> follows the same dense layout but
    /// only the two positions per block selected by <paramref name="patternNibbles"/>
    /// are read/updated. <paramref name="patternNibbles"/> packs two 2-bit indices per
    /// nibble, two nibbles per byte → length = <c>parameter.Length / 8</c>.
    /// </summary>
    public ParamGroup AddSparse24Parameter(
        float[] parameter, float[] gradient, byte[] patternNibbles,
        IDictionary<string, double>? overrides = null)
    {
        if (parameter == null) throw new ArgumentNullException(nameof(parameter));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (patternNibbles == null) throw new ArgumentNullException(nameof(patternNibbles));
        if (parameter.Length % 4 != 0)
            throw new ArgumentException("2:4 sparsity requires parameter length to be a multiple of 4.");
        int blocks = parameter.Length / 4;
        int expectedPatternBytes = (blocks + 1) / 2;
        if (patternNibbles.Length != expectedPatternBytes)
            throw new ArgumentException(
                $"patternNibbles length {patternNibbles.Length} != expected {expectedPatternBytes}.");

        var grp = AddParamGroup(overrides);
        grp.AddParameter(parameter, gradient);
        _patterns[(ParamGroups.Count - 1, grp.Parameters.Count - 1)] = patternNibbles;
        return grp;
    }

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr  = (float)g.LearningRate;
            float b1  = (float)g.GetOption("beta1", 0.9);
            float b2  = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!_patterns.TryGetValue((gi, pi), out var pattern))
                    throw new InvalidOperationException(
                        $"param[{gi},{pi}] was not added via AddSparse24Parameter.");

                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                float bc1 = 1f - MathF.Pow(b1, step);
                float bc2 = 1f - MathF.Pow(b2, step);
                float lrAdj = lr / bc1;
                float bc2Inv = 1f / bc2;

                int blocks = p.Length / 4;
                for (int blk = 0; blk < blocks; blk++)
                {
                    // Two 2-bit indices per nibble; two nibbles per byte.
                    byte b = pattern[blk >> 1];
                    byte nib = (blk & 1) == 0 ? (byte)(b & 0x0F) : (byte)((b >> 4) & 0x0F);
                    int idx0 = nib & 0x3;
                    int idx1 = (nib >> 2) & 0x3;
                    int blockBase = blk * 4;

                    UpdateOne(blockBase + idx0, grad, m, v, p, b1, b2, eps, lrAdj, bc2Inv);
                    if (idx1 != idx0)
                        UpdateOne(blockBase + idx1, grad, m, v, p, b1, b2, eps, lrAdj, bc2Inv);
                }
            }
        }
    }

    private static void UpdateOne(int i, float[] grad, float[] m, float[] v, float[] p,
                                  float b1, float b2, float eps, float lrAdj, float bc2Inv)
    {
        float gi = grad[i];
        float mNew = b1 * m[i] + (1f - b1) * gi;
        float vNew = b2 * v[i] + (1f - b2) * gi * gi;
        m[i] = mNew;
        v[i] = vNew;
        float vHat = vNew * bc2Inv;
        p[i] -= lrAdj * mNew / (MathF.Sqrt(vHat) + eps);
    }
}
