using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Argument-table entries for the linear-algebra, recurrent-cell and box-loss ops.</summary>
public partial class DifferentiableOpsGradCheckSweep
{
    static partial void AddCasesLinearAlgebraAndLosses(Dictionary<string, Func<Random, object[]>> cases)
    {
        // input + alpha·Σ_b batch1[b]·batch2[b] (addbmm) / per batch (baddbmm); beta and alpha off 1 so both scales are checked.
        cases["TensorAddbmm"] = r => [SafeTensor([2, 4], r), SafeTensor([3, 2, 3], r), SafeTensor([3, 3, 4], r), 0.7, 1.3];
        cases["TensorBaddbmm"] = r => [SafeTensor([3, 2, 4], r), SafeTensor([3, 2, 3], r), SafeTensor([3, 3, 4], r), 0.7, 1.3];
        cases["TensorAddmv"] = r => [SafeTensor([3], r), SafeTensor([3, 4], r), SafeTensor([4], r), 0.7, 1.3];
        cases["TensorBatchMatMul"] = r => [SafeTensor([2, 2, 3], r), SafeTensor([2, 3, 2], r)];
        // x1 [N, in1], x2 [N, in2], weight [out, in1, in2], bias [out].
        cases["TensorBilinear"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 2], r), SafeTensor([4, 3, 2], r), SafeTensor([4], r)];

        // Boxes as (x1, y1, x2, y2) with x2 > x1, y2 > y1, overlapping but distinct, so every IoU term is smooth.
        cases["TensorIoULoss"] = r => [Boxes(3, r), Boxes(3, r)];
        cases["TensorGIoULoss"] = r => [Boxes(3, r), Boxes(3, r)];
        cases["TensorDIoULoss"] = r => [Boxes(3, r), Boxes(3, r)];
        cases["TensorCIoULoss"] = r => [Boxes(3, r), Boxes(3, r)];

        // The expert forward: rows [0, 2) use b[0], rows [2, 4) use b[1]; bias [N].
        cases["TensorGroupedMatMul"] = r => [SafeTensor([4, 3], r), SafeTensor([2, 3, 2], r),
            new Tensor<int>(new[] { 2, 4 }, new[] { 2 }), SafeTensor([2], r)];

        // Points (and the bias point) well inside the unit ball: ‖row‖ ≈ 0.2.
        cases["TensorHyperbolicLinear"] = r => [Scaled(SafeTensor([2, 3], r), 0.2), SafeTensor([4, 3], r),
            Scaled(SafeTensor([4], r), 0.1), 1.0];

        // PyTorch layouts: x [batch, in], h [batch, hidden], w_ih [gates·hidden, in], w_hh [gates·hidden, hidden].
        // The reflective default gave every tensor [2, 3], which is not a valid cell (hidden 3 against a 2-row w_hh).
        cases["TensorRnnCell"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 4], r), SafeTensor([4, 3], r), SafeTensor([4, 4], r),
            SafeTensor([4], r), SafeTensor([4], r), RnnCellType.RnnTanh];
        cases["TensorGruCell"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 4], r), SafeTensor([12, 3], r), SafeTensor([12, 4], r),
            SafeTensor([12], r), SafeTensor([12], r)];
    }

    private static Tensor<double> Scaled(Tensor<double> t, double factor)
    {
        for (int i = 0; i < t.Length; i++) t[i] *= factor;
        return t;
    }
}
