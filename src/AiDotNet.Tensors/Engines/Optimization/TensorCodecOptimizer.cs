using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// TensorCodec hybrid orchestrator: selects and applies the best combination of
/// computational transformations based on network structure analysis.
///
/// Phase A (Spectral): For weight matrices with fast singular value decay, decompose via SVD.
/// Phase B (Dataflow): For consecutive linear layers with small hidden dims, fuse across boundaries.
/// Phase C (Algebraic): For backward graphs with shared subexpressions, apply CSE + simplification.
///
/// Decision logic analyzes matrix dimensions, layer adjacency, and backward graph structure
/// to determine which optimizations to apply. All approaches are independently toggleable.
/// </summary>
internal static class TensorCodecOptimizer
{
    /// <summary>
    /// Analyzes a weight matrix and returns spectral factors if decomposition is beneficial.
    /// Returns null if the matrix is full-rank or compression ratio is insufficient.
    /// </summary>
    internal static SpectralFactors? TrySpectralDecompose(float[] weights, int m, int n)
    {
        var options = TensorCodecOptions.Current;
        if (!options.EnableSpectralDecomposition)
            return null;

        return SvdDecomposition.Decompose(weights, m, n,
            maxRank: 0,
            energyThreshold: 1.0 - options.SpectralErrorTolerance);
    }

    /// <summary>
    /// Determines if two consecutive layers can be fused via dataflow fusion.
    /// Requires hidden dimension to fit in L1 cache for the Mr-row tile.
    /// </summary>
    internal static bool CanFuseDataflow(int hiddenDim)
    {
        var options = TensorCodecOptions.Current;
        if (!options.EnableDataflowFusion)
            return false;

        return hiddenDim <= options.DataflowFusionMaxHidden;
    }

    /// <summary>
    /// Executes a 2-layer fused forward: input[M,K] → W1[K,H] → activation → W2[H,N] → output[M,N]
    /// using the best available path.
    /// </summary>
    internal static void FusedTwoLayerForward(
        float[] input, float[] w1, float[] w2,
        float[] output, float[] activated,
        int m, int k, int h, int n,
        Func<float, float> activation)
    {
        if (CanFuseDataflow(h))
        {
            FusedMultiLayerGemm.FusedGemmActivationGemm(
                input, w1, w2, output, activated, m, k, h, n, activation);
        }
        else
        {
            // Fallback: separate ops
            var hidden = new float[m * h];
            SimdGemm.Sgemm(input.AsSpan(0, m * k), w1.AsSpan(0, k * h), hidden.AsSpan(), m, k, h);
            for (int i = 0; i < m * h; i++)
            {
                float val = activation(hidden[i]);
                hidden[i] = val;
                activated[i] = val;
            }
            SimdGemm.Sgemm(hidden.AsSpan(0, m * h), w2.AsSpan(0, h * n), output.AsSpan(0, m * n), m, h, n);
        }
    }

    /// <summary>
    /// Executes a spectral matmul if the weight has been decomposed, otherwise direct matmul.
    /// </summary>
    internal static void SmartMatMul(
        float[] x, int xRows, int xCols,
        float[] weights, int wRows, int wCols,
        float[] output,
        SpectralFactors? spectralFactors = null)
    {
        if (spectralFactors.HasValue)
        {
            SvdDecomposition.SpectralMatMul(x, xRows, xCols, spectralFactors.Value, output);
        }
        else
        {
            Array.Clear(output, 0, xRows * wCols);
            if (!BlasProvider.TryGemm(xRows, wCols, xCols, x, 0, xCols, weights, 0, wCols, output, 0, wCols))
                SimdGemm.Sgemm(x.AsSpan(0, xRows * xCols), weights.AsSpan(0, wRows * wCols),
                    output.AsSpan(0, xRows * wCols), xRows, xCols, wCols);
        }
    }
}
