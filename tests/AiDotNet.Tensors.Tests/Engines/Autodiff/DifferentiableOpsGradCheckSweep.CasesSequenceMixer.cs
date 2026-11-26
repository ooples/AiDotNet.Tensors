using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Argument tables for the fused sequence-mixer ops (CpuEngine.SequenceMixer.cs). Each has a shape contract the
/// sweep's generic 2D/3D attempts cannot meet (even head dimensions, [B, S, H, scanDim] scans, per-expert tables), so
/// without these every gradient sweep reported them as "no valid invocation" and failed.
/// </summary>
public partial class DifferentiableOpsGradCheckSweep
{
    static partial void AddCasesSequenceMixer(Dictionary<string, Func<Random, object[]>> cases)
    {
        // input [..., D] with D even; one angle per pair, [D / 2].
        cases["SplitHalfGivensRotation"] = r => [SafeTensor([2, 3, 4], r), SafeTensor([2], r)];

        // input [B, S, C]; the two taps are per-channel, [C].
        cases["DepthwiseCausalConv1d2Tap"] = r => [SafeTensor([2, 3, 4], r), SafeTensor([4], r), SafeTensor([4], r)];

        // input [B, S, H * Dh] with Dh even; cos/sin [>= startPosition + S, Dh / 2]. Two heads of width 4, starting
        // at position 1 so the table offset is exercised.
        cases["RopeL2NormalizeHeads"] = r => [SafeTensor([1, 3, 8], r), SafeTensor([4, 2], r), SafeTensor([4, 2], r),
            2, 1e-6, 1.5, 1];

        // input [B, S, H * Mb] with Mb even; output head width >= Mb^2 (0 selects exactly Mb^2). cos/sin
        // [>= startPosition + S, Mb / 2]. Padding the head to 5 > 2^2 checks the zero columns too.
        cases["RopeQuadraticOuterFeatures"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([4, 1], r), SafeTensor([4, 1], r),
            2, 5, 1];

        // baseHidden and proposal share a shape; the parallel gate is one scalar logit.
        cases["ProjectiveGatedResidual"] = r => [SafeTensor([2, 3, 4], r), SafeTensor([2, 3, 4], r), SafeTensor([1], r), 1e-6];

        // diagScale [B, D] with D even; the half spectrum has D / 2 + 1 bins per batch row.
        cases["SpectralCirculantMatrix"] = r => [SafeTensor([2, 4], r), SafeTensor([2, 3], r), SafeTensor([2, 3], r)];

        // logits [N, G], expertTable [G, C], routingBias [G]; two of three experts per row. The bias update rate is 0,
        // so the op does not write its bias. Distinct logits keep the top-2 selection away from a tie that a finite
        // difference could cross.
        cases["SigmoidTopKExpertMix"] = r => [DistinctRows(3, 3, r), SafeTensor([3, 4], r), SafeTensor([3], r), 2, 1e-12, 0.0];

        // gateUpProj and its gain delta [N, 2H]; upBase [H].
        cases["DualModulatedSwiGlu"] = r => [SafeTensor([3, 4], r), SafeTensor([3, 4], r), SafeTensor([2], r)];

        // scanOutput [B, S, H, scanDim] whose first headDim columns are numerators and the next one the denominator;
        // sinkLogAlpha [H].
        cases["BornRatioNormalize"] = r => [SafeTensor([1, 3, 2, 3], r), SafeTensor([2], r), 2, 1e-4];
    }

    // Rows whose entries are spread at least 0.2 apart, in a shuffled order, so a top-k choice has clear margins.
    private static Tensor<double> DistinctRows(int rows, int cols, Random rng)
    {
        var t = new Tensor<double>([rows, cols]);
        for (int i = 0; i < rows; i++)
        {
            var order = new int[cols];
            for (int c = 0; c < cols; c++) order[c] = c;
            for (int c = cols - 1; c > 0; c--)
            {
                int j = rng.Next(c + 1);
                (order[c], order[j]) = (order[j], order[c]);
            }
            for (int c = 0; c < cols; c++)
                t[i * cols + order[c]] = -0.6 + 0.4 * c + 0.05 * rng.NextDouble();
        }
        return t;
    }
}
