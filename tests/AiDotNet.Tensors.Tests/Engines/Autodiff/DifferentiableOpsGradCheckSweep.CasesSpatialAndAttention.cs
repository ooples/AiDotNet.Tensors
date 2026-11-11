using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Argument tables for the spatial (pooling, convolution, mesh/graph) and attention delegators.</summary>
public partial class DifferentiableOpsGradCheckSweep
{
    // Fractional sampling offsets in (0.2, 0.4): deformable convolution samples bilinearly, which has a kink at every
    // integer position; these keep each sample (and its central-difference probes) strictly inside one cell.
    private static Tensor<double> FractionalOffsets(int[] shape, Random rng)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.2 + rng.NextDouble() * 0.2;
        return t;
    }

    private static Tensor<Complex<double>> ComplexFilter(int[] shape, Random rng)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var c = new Complex<double>[n];
        for (int i = 0; i < n; i++) c[i] = new Complex<double>(0.35 + rng.NextDouble() * 0.6, rng.NextDouble() - 0.5);
        return new Tensor<Complex<double>>(c, shape);
    }

    // A tetrahedron: four non-degenerate triangles, so every cotangent weight is finite.
    private static Tensor<double> TetrahedronVertices(Random rng)
    {
        var t = new Tensor<double>(new[] { 4, 3 });
        double[] baseVertices = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        for (int i = 0; i < 12; i++) t[i] = baseVertices[i] + (rng.NextDouble() - 0.5) * 0.1;
        return t;
    }

    static partial void AddCasesSpatialAndAttention(Dictionary<string, Func<Random, object[]>> cases)
    {
        // --- pooling (random values: every window has one strict winner far beyond a 1e-6 probe) ---
        cases["MaxPool3D|T,int,int,int"] = r => [SafeTensor([1, 1, 4, 4, 4], r), 2, 2, 0];
        cases["MaxPool3D|T,int[],int[],int[]"] = r => [SafeTensor([1, 1, 4, 4, 4], r), new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }];
        cases["TensorAdaptiveMaxPool2D"] = r => [SafeTensor([1, 2, 4, 4], r), new[] { 2, 2 }];
        cases["TensorAvgPool1D"] = r => [SafeTensor([1, 2, 6], r), 2, 2];
        cases["TensorAvgPool2D"] = r => [SafeTensor([1, 1, 4, 4], r), 2, 2, 0, false];
        cases["TensorMaxPool1D"] = r => [SafeTensor([1, 2, 6], r), 2, 2];
        cases["TensorMaxPool2D"] = r => [SafeTensor([1, 1, 4, 4], r), 2, 2, 0];
        cases["TensorPixelUnshuffle"] = r => [SafeTensor([1, 1, 4, 4], r), 2];
        cases["TensorLocalResponseNorm"] = r => [SafeTensor([1, 4, 3, 3], r), 3, 0.5, 0.75, 1.0];

        // --- convolutions ---
        cases["TensorConv2D"] = r => [SafeTensor([1, 2, 5, 5], r), SafeTensor([3, 2, 3, 3], r), 1, 1, 1];
        cases["TensorConvolution"] = r => [SafeTensor([1, 2, 5, 5], r), SafeTensor([3, 2, 3, 3], r), SafeTensor([3], r),
            new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, false, null!, 1];
        cases["TensorConvTbc"] = r => [SafeTensor([6, 2, 3], r), SafeTensor([3, 3, 4], r), SafeTensor([4], r), 1];
        cases["TensorConvTranspose1D"] = r => [SafeTensor([1, 2, 4], r), SafeTensor([2, 3, 3], r), 1, 0, 0];
        cases["DeformableConv2D"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([3, 2, 3, 3], r),
            FractionalOffsets([1, 18, 4, 4], r), SafeTensor([1, 9, 4, 4], r), new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }];
        cases["DeformableConv2DGrouped"] = r => [SafeTensor([1, 4, 4, 4], r), SafeTensor([4, 2, 3, 3], r),
            FractionalOffsets([1, 18, 4, 4], r), SafeTensor([1, 9, 4, 4], r), new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, 2, 1];
        cases["PartialCorrelationVolume"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([1, 2, 4, 4], r), 1];

        // --- spectral filtering (complex filter is a constant parameter here) ---
        cases["NativeSpectralFilter"] = r => [SafeTensor([4, 8], r), ComplexFilter([4, 8], r)];
        cases["NativeSpectralFilterBatch"] = r => [SafeTensor([1, 2, 4, 8], r), ComplexFilter([4, 8], r)];

        // --- mesh / graph ---
        cases["ComputeMeshLaplacian"] = r => [TetrahedronVertices(r),
            new Tensor<int>(new[] { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 }, new[] { 4, 3 }), LaplacianType.Cotangent];
        cases["SpiralConv"] = r => [SafeTensor([6, 4], r),
            new Tensor<int>(new[] { 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 0, 5, 0, 1 }, new[] { 6, 3 }),
            SafeTensor([5, 12], r), SafeTensor([5], r)];
        cases["DiffusionConv"] = r => [SafeTensor([6, 4], r), Shifted(SafeTensor([6, 6], r), -0.65), SafeTensor([4, 4], r),
            SafeTensor([4], r), 0.5];
        cases["GraphAttention"] = r => [SafeTensor([1, 4, 6], r),
            new Tensor<int>(new[] { 0, 1, 2, 3, 0, 1 }, new[] { 6 }), new Tensor<int>(new[] { 1, 2, 3, 0, 2, 3 }, new[] { 6 }),
            SafeTensor([6], r), SafeTensor([6], r), 0.2, null!];
        cases["MultiHeadGraphAttention"] = r => [SafeTensor([1, 4, 6], r),
            new Tensor<int>(new[] { 0, 1, 2, 3, 0, 1 }, new[] { 6 }), new Tensor<int>(new[] { 1, 2, 3, 0, 2, 3 }, new[] { 6 }),
            SafeTensor([2, 6, 3], r), SafeTensor([2, 3], r), SafeTensor([2, 3], r), 0.2, true, null!];

        // --- attention ---
        cases["FlashAttention"] = r => [SafeTensor([1, 2, 3, 4], r), SafeTensor([1, 2, 3, 4], r), SafeTensor([1, 2, 3, 4], r),
            null!, false, null!, null!];
        cases["GroupedQueryAttention"] = r => [SafeTensor([1, 4, 3, 4], r), SafeTensor([1, 2, 3, 4], r), SafeTensor([1, 2, 3, 4], r),
            2, null!, false, null!];
        cases["ScaledDotProductAttention|T,T,T,Tb,double?,T,double"] = r => [SafeTensor([1, 2, 3, 4], r),
            SafeTensor([1, 2, 3, 4], r), SafeTensor([1, 2, 3, 4], r), null!, null!, null!, 0.0];
        cases["ScaledDotProductAttention|T,T,T,Tb,double?,bool,double"] = r => [SafeTensor([1, 2, 3, 4], r),
            SafeTensor([1, 2, 3, 4], r), SafeTensor([1, 2, 3, 4], r), null!, null!, false, 0.0];
        cases["ScaledDotProductAttentionGqa"] = r => [SafeTensor([1, 4, 3, 4], r), SafeTensor([1, 2, 3, 4], r),
            SafeTensor([1, 2, 3, 4], r), 0.5, false, 0.0];
    }
}
