using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Argument tables for the ops that take a fused activation epilogue. Each is given Tanh here; the gradgrad sweep
/// enumerates every other <see cref="FusedActivationType"/> from that base, so no activation goes unchecked.
/// </summary>
public partial class DifferentiableOpsGradCheckSweep
{
    // A positive running variance (SafeTensor's values are already in [0.35, 0.95]).
    private static Tensor<double> Variance(int n, Random rng) => SafeTensor(new[] { n }, rng);

    static partial void AddCasesFusedActivations(Dictionary<string, Func<Random, object[]>> cases)
    {
        // NCHW input [1, 2, 4, 4], OIHW kernel [3, 2, 2, 2], bias [3]; stride 1, no padding, no dilation.
        cases["FusedConv2D"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([3, 2, 2, 2], r), SafeTensor([3], r),
            1, 1, 0, 0, 1, 1, FusedActivationType.Tanh];
        // NCDHW input [1, 2, 3, 3, 3], kernel [2, 2, 2, 2, 2].
        cases["FusedConv3D"] = r => [SafeTensor([1, 2, 3, 3, 3], r), SafeTensor([2, 2, 2, 2, 2], r), SafeTensor([2], r),
            1, 1, 1, 0, 0, 0, 1, 1, 1, FusedActivationType.Tanh];
        // Transposed convolution kernel is [Cin, Cout, kH, kW].
        cases["FusedConvTranspose2D"] = r => [SafeTensor([1, 2, 3, 3], r), SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r),
            1, 1, 0, 0, 0, 0, FusedActivationType.Tanh];
        cases["FusedConv2DBatchNorm"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([3, 2, 2, 2], r), SafeTensor([3], r),
            SafeTensor([3], r), SafeTensor([3], r), SafeTensor([3], r), Variance(3, r), 1e-5,
            1, 1, 0, 0, 1, 1, FusedActivationType.Tanh];
        // [N, C] batch norm in training mode (batch statistics); the trailing nulls are the saveMean / saveVar outs.
        cases["FusedBatchNorm"] = r => [SafeTensor([4, 3], r), SafeTensor([3], r), SafeTensor([3], r),
            SafeTensor([3], r), Variance(3, r), 1e-5, 0.1, true, FusedActivationType.Tanh, null!, null!];
    }
}
