using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public partial class DifferentiableOpsGradCheckSweep
{
    /// <summary>
    /// Arguments for parameter variants (DifferentiableOpsGradGradCheckSweep.ParameterVariants) whose valid inputs differ
    /// in shape from the base table entry's: another axis needs index tensors and sources shaped for that axis, a 1-D or
    /// 3-D interpolation mode another rank, a transposed convolution another weight layout. Keyed
    /// "<c>Op|param=Value</c>" (or "<c>Op|Fingerprint|param=Value</c>" for one overload) - the variant's label.
    /// </summary>
    static partial void AddCasesVariants(Dictionary<string, Func<Random, object[]>> cases)
    {
        // gather along dim 0 / -2: indices index the 3 rows.
        foreach (var axis in new[] { 0, -2 })
        {
            int a = axis;
            cases[$"Gather|axis={a}"] = r => [SafeTensor([3, 2], r), IdxTensor([2, 2], 3, r), a];
            // Scatter / ScatterAdd(input, 1-D indices, values, axis): values carry one slice per index along the axis.
            // Distinct indices, so the overwriting scatter is a function of every source element.
            cases[$"Scatter|axis={a}"] = r => [SafeTensor([3, 2], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), SafeTensor([2, 2], r), a];
            cases[$"ScatterAdd|T,Ti,T,int|axis={a}"] = r => [SafeTensor([3, 2], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), SafeTensor([2, 2], r), a];
            // cross product along the size-3 axis 0.
            cases[$"TensorCross|dim={a}"] = r => [SafeTensor([3, 2], r), SafeTensor([3, 2], r), a];
            // take_along_dim along dim 0: indices into the 2 rows.
            cases[$"TensorTakeAlongDim|dim={a}"] = r => [SafeTensor([2, 3], r), IdxTensor([2, 3], 2, r), a];
            cases[$"TensorUnflatten|dim={a}"] = r => [SafeTensor([6, 2], r), a, new[] { 2, 3 }];
        }

        foreach (var axis in new[] { 1, -1 })
        {
            int a = axis;
            // index_add / index_copy along the columns: one source column per index (a permutation for index_copy).
            cases[$"TensorIndexAdd|axis={a}"] = r => [SafeTensor([2, 3], r), a, new Tensor<int>(new[] { 2, 0, 2 }, new[] { 3 }), SafeTensor([2, 3], r)];
            cases[$"TensorIndexCopy|axis={a}"] = r => [SafeTensor([2, 3], r), a, new Tensor<int>(new[] { 2, 0, 1 }, new[] { 3 }), SafeTensor([2, 3], r)];
            // index_reduce along the columns: 4 source columns into 3 destination columns.
            cases[$"TensorIndexReduce|dim={a}"] = r => [SafeTensor([2, 3], r), a, new Tensor<int>(new[] { 0, 2, 0, 1 }, new[] { 4 }),
                                                       SafeTensor([2, 4], r), ScatterReduceMode.Sum, true];
            cases[$"TensorIndexSelectDiff|axis={a}"] = r => [SafeTensor([2, 3], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), a];
            cases[$"TensorScatterAdd|axis={a}"] = r => [SafeTensor([2, 3], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), SafeTensor([2, 2], r), a];
            // scatter_reduce along dim 1: index (shaped like source) into the 2 destination columns.
            cases[$"TensorScatterReduce|dim={a}"] = r => [SafeTensor([3, 2], r), a, IdxTensor([3, 2], 2, r), SafeTensor([3, 2], r),
                                                         ScatterReduceMode.Sum, true];
            cases[$"TensorSelectScatter|dim={a}"] = r => [SafeTensor([3, 2], r), SafeTensor([3], r), a, 1];
            cases[$"TensorSliceScatter|dim={a}"] = r => [SafeTensor([2, 4], r), SafeTensor([2, 2], r), a, 1, 2];
        }

        // A transposed convolution's weight is [in, out/groups, kH, kW].
        cases["TensorConvolution|transposed=True"] = r => [SafeTensor([1, 2, 5, 5], r), SafeTensor([2, 3, 3, 3], r), SafeTensor([3], r),
            new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, true, null!, 1];

        // Linear interpolation is 1-D (N, C, L); trilinear is 3-D (N, C, D, H, W).
        cases["Interpolate|mode=Linear"] = r => [SafeTensor([1, 2, 3], r), new[] { 5 }, InterpolateMode.Linear, false];
        cases["Interpolate|mode=Trilinear"] = r => [SafeTensor([1, 1, 2, 2, 2], r), new[] { 3, 3, 3 }, InterpolateMode.Trilinear, false];
    }
}
