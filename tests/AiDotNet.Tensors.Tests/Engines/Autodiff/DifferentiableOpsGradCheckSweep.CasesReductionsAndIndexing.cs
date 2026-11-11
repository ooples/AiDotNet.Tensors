using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public partial class DifferentiableOpsGradCheckSweep
{
    /// <summary>Argument tables for the reduction, indexing and pooling delegators the reflective synthesis cannot drive.</summary>
    static partial void AddCasesReductionsAndIndexing(Dictionary<string, Func<Random, object[]>> cases)
    {
        cases["GlobalAvgPool2D"] = r => [SafeTensor([1, 2, 3, 3], r)];
        cases["GlobalMaxPool2D"] = r => [SafeTensor([1, 2, 3, 3], r)];
        cases["TensorDiff"] = r => [SafeTensor([2, 4], r), 1, -1];
        cases["TensorGather"] = r => [SafeTensor([3, 2], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), 0];
        // Class indices are values, one per row of the [2, 3] input; they are not differentiable (an integer target).
        cases["TensorGatherClassValues"] = r => [SafeTensor([2, 3], r), new Tensor<double>(new[] { 2.0, 0.0 }, new[] { 2 })];
        cases["TensorIndex|T,Ti[]"] = r => [SafeTensor([3, 2], r), new Tensor<int>?[] { new Tensor<int>(new[] { 2, 0 }, new[] { 2 }) }];
        cases["TensorIndex|T,Tb"] = r => [SafeTensor([3, 2], r), BoolMask([3, 2])];
        // index_reduce along dim 0: the index runs over source's dim-0 rows; every ScatterReduceMode is a sweep variant.
        cases["TensorIndexReduce"] = r => [SafeTensor([3, 2], r), 0, new Tensor<int>(new[] { 0, 2, 0, 1 }, new[] { 4 }),
                                           SafeTensor([4, 2], r), ScatterReduceMode.Sum, true];
        cases["TensorIndexSelectDiff"] = r => [SafeTensor([3, 2], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), 0];
        cases["TensorScatterAdd"] = r => [SafeTensor([3, 2], r), new Tensor<int>(new[] { 2, 0 }, new[] { 2 }), SafeTensor([2, 2], r), 0];
        cases["TensorStackDiff"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0];
        cases["TensorUnflatten"] = r => [SafeTensor([2, 6], r), 1, new[] { 2, 3 }];
        cases["TensorPadSequence"] = r => [new[] { SafeTensor([3, 2], r), SafeTensor([2, 2], r) }, false, 0.0];
        cases["TensorBroadcastTo"] = r => [SafeTensor([1, 3], r), new[] { 2, 3 }];
        cases["TensorEinsum"] = r => ["ij,jk->ik", new[] { SafeTensor([2, 3], r), SafeTensor([3, 2], r) }];

        // Random ops with a fixed seed: the draw (mask, noise) is the same on every call, so the finite differences of
        // the training-mode (and sampling) forwards are well defined and their recorded backwards are checked like any
        // other op's, instead of being exempted as stochastic.
        cases["TensorAlphaDropout"] = r => [SafeTensor([2, 4], r), 0.5, false, 7];
        cases["TensorFeatureAlphaDropout"] = r => [SafeTensor([2, 3, 2], r), 0.5, false, 7];
        cases["TensorChannelDropout"] = r => [SafeTensor([2, 3, 2], r), 0.5, false, 1, 7];
        cases["TensorNormal"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 3], r), 7];
    }
}
