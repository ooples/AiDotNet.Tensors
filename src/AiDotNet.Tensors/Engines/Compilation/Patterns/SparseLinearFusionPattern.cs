// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — sparse-linear chain detection for graph-mode auto-fusion.

using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the dense-with-mask sparse-linear pattern and rewrites it
/// to <see cref="LazyNodeType.FusedSparseLinear"/>. The CSR-format
/// kernel ships now (see
/// <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedSparseLinear"/>).
/// </summary>
internal sealed class SparseLinearFusionPattern : IFusionPattern
{
    public string Name => "SparseLinear";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> start)
            return false;

        LazyNode<float>? maskNode = null;
        LazyNode<float>? matmul = null;
        CsrWeights? weights = null;

        if (start.OpType == LazyNodeType.MatMul)
        {
            matmul = start;
            if (!TryBuildCsrFromStaticWeight(matmul, out weights))
                return false;
        }
        else if (PatternHelpers.IsMultiply(start.OpType))
        {
            maskNode = start;
            if (!PatternHelpers.FanOutAtMostOne(maskNode, consumerCounts))
                return false;

            matmul = PatternHelpers.FindNextFloat(
                nodes, index + 1, index + 5, alreadyRemoved,
                n => n.OpType == LazyNodeType.MatMul
                    && ReferenceEquals(n.Input1, maskNode.Output));
            if (matmul is null || !TryBuildCsrFromMask(maskNode, matmul, out weights))
                return false;
        }

        if (matmul is null || weights is null)
            return false;

        var input = matmul.Input0;
        if (input.Rank != 2 || matmul.Output.Rank != 2)
            return false;

        int batch = input._shape[0];
        int inFeat = input._shape[1];
        int outFeat = matmul.Output._shape[1];
        if (matmul.Output._shape[0] != batch
            || weights.InputFeatures != inFeat
            || weights.OutputFeatures != outFeat)
            return false;

        LazyNode<float>? addNode = null;
        LazyNode<float>? activationNode = null;
        Tensor<float>? bias = null;
        FusedActivationType activation = FusedActivationType.None;
        Tensor<float> finalOutput = matmul.Output;

        var matmulFanOutOk = PatternHelpers.FanOutAtMostOne(matmul, consumerCounts);
        if (matmulFanOutOk)
        {
            addNode = PatternHelpers.FindNextFloat(
                nodes, index + 1, index + 6, alreadyRemoved,
                n => PatternHelpers.IsAdd(n.OpType)
                    && PatternHelpers.ConsumesOutput(n, matmul)
                    && TryGetSparseBias(n, matmul.Output, out _));
            if (addNode is not null && TryGetSparseBias(addNode, matmul.Output, out bias))
                finalOutput = addNode.Output;
        }

        var activationBase = addNode ?? matmul;
        if (PatternHelpers.FanOutAtMostOne(activationBase, consumerCounts))
        {
            activationNode = PatternHelpers.FindNextFloat(
                nodes, index + 1, index + 8, alreadyRemoved,
                n => PatternHelpers.ConsumesOutput(n, activationBase)
                    && ActivationRegistry.TryGetActivationType(n.OpType, out var act)
                    && IsPointwiseActivation(act));

            if (activationNode is not null
                && ActivationRegistry.TryGetActivationType(activationNode.OpType, out activation))
            {
                finalOutput = activationNode.Output;
            }
        }

        fused = Build(input, bias, weights, activation, finalOutput);
        if (fused is null)
            return false;

        if (maskNode is not null)
            maskNode.Output.LazySource = null;
        matmul.Output.LazySource = null;
        addNode?.Output.LazySource = null;
        activationNode?.Output.LazySource = null;

        PatternHelpers.SetLazySource(fused);
        if (maskNode is not null)
            alreadyRemoved.Add(maskNode);
        alreadyRemoved.Add(matmul);
        if (addNode is not null)
            alreadyRemoved.Add(addNode);
        if (activationNode is not null)
            alreadyRemoved.Add(activationNode);
        return true;
    }

    private static LazyNode<float> Build(
        Tensor<float> input,
        Tensor<float>? bias,
        CsrWeights weights,
        FusedActivationType activation,
        Tensor<float> finalOutput)
    {
        var capturedInput = input;
        var capturedBias = bias;
        var rowOffsets = weights.RowOffsets;
        var colIndices = weights.ColIndices;
        var values = weights.Values;
        var capturedActivation = activation;

        var inputs = capturedBias is null
            ? new[] { capturedInput, values }
            : new[] { capturedInput, values, capturedBias };

        return new LazyNode<float>(
            LazyNodeType.FusedSparseLinear,
            "FusedSparseLinear",
            inputs,
            finalOutput,
            (eng, output) =>
            {
                if (eng is DirectGpuTensorEngine gpu)
                {
                    var eager = gpu.FusedSparseLinear(
                        capturedInput, rowOffsets, colIndices, values, capturedBias, capturedActivation);
                    eager.AsSpan().CopyTo(output.AsWritableSpan());
                }
                else
                {
                    CpuFusedOperations.FusedSparseLinear(
                        capturedInput, rowOffsets, colIndices, values, capturedBias, capturedActivation, output);
                }
            },
            backwardFn: null,
            savedState: new object[] { rowOffsets, colIndices, values, capturedActivation });
    }

    private static bool TryBuildCsrFromStaticWeight(LazyNode<float> matmul, out CsrWeights? weights)
    {
        weights = null;
        var weight = matmul.Input1;
        if (weight is null || weight.LazySource is not null)
            return false;
        if (matmul.Input0.Rank != 2 || weight.Rank != 2)
            return false;
        return TryDenseInOutToCsr(weight, matmul.Input0._shape[1], weight._shape[1], out weights);
    }

    private static bool TryBuildCsrFromMask(
        LazyNode<float> maskNode,
        LazyNode<float> matmul,
        out CsrWeights? weights)
    {
        weights = null;
        if (maskNode.Input1 is null || maskNode.Input0.LazySource is not null || maskNode.Input1.LazySource is not null)
            return false;
        if (matmul.Input0.Rank != 2 || maskNode.Input0.Rank != 2 || maskNode.Input1.Rank != 2)
            return false;
        if (!PatternHelpers.SameShape(maskNode.Input0, maskNode.Input1))
            return false;

        int inFeat = matmul.Input0._shape[1];
        int outFeat = maskNode.Input0._shape[1];
        if (maskNode.Input0._shape[0] != inFeat)
            return false;

        var left = maskNode.Input0.GetDataArray() as float[];
        var right = maskNode.Input1.GetDataArray() as float[];
        if (left is null || right is null)
            return false;

        return TryProductInOutToCsr(left, right, inFeat, outFeat, out weights);
    }

    private static bool TryDenseInOutToCsr(
        Tensor<float> dense,
        int inFeat,
        int outFeat,
        out CsrWeights? weights)
    {
        weights = null;
        if (dense._shape[0] != inFeat || dense._shape[1] != outFeat)
            return false;

        var data = dense.GetDataArray() as float[];
        if (data is null)
            return false;

        return TryDenseInOutToCsr(data, inFeat, outFeat, out weights);
    }

    private static bool TryProductInOutToCsr(
        float[] left,
        float[] right,
        int inFeat,
        int outFeat,
        out CsrWeights? weights)
    {
        var product = new float[inFeat * outFeat];
        for (int i = 0; i < product.Length; i++)
            product[i] = left[i] * right[i];
        return TryDenseInOutToCsr(product, inFeat, outFeat, out weights);
    }

    private static bool TryDenseInOutToCsr(
        float[] denseInOut,
        int inFeat,
        int outFeat,
        out CsrWeights? weights)
    {
        weights = null;
        int total = inFeat * outFeat;
        if (total == 0)
            return false;

        var rowOffsets = new int[outFeat + 1];
        var colIndices = new List<int>();
        var values = new List<float>();

        for (int j = 0; j < outFeat; j++)
        {
            for (int i = 0; i < inFeat; i++)
            {
                float value = denseInOut[i * outFeat + j];
                if (value == 0f)
                    continue;
                colIndices.Add(i);
                values.Add(value);
            }
            rowOffsets[j + 1] = colIndices.Count;
        }

        if (values.Count == total || values.Count > total / 2)
            return false;

        weights = new CsrWeights(
            rowOffsets,
            colIndices.ToArray(),
            new Tensor<float>(values.ToArray(), new[] { values.Count }),
            inFeat,
            outFeat);
        return true;
    }

    private static bool TryGetSparseBias(
        LazyNode<float> addNode,
        Tensor<float> matmulOutput,
        out Tensor<float>? bias)
    {
        bias = null;
        if (!PatternHelpers.TryGetOtherInput(addNode, matmulOutput, out var candidate))
            return false;
        int outFeat = matmulOutput._shape[matmulOutput._shape.Length - 1];
        if (candidate.Rank != 1 || candidate._shape[0] != outFeat)
            return false;
        bias = candidate;
        return true;
    }

    private static bool IsPointwiseActivation(FusedActivationType activation)
        => activation is FusedActivationType.None
            or FusedActivationType.ReLU
            or FusedActivationType.GELU
            or FusedActivationType.Sigmoid
            or FusedActivationType.Tanh
            or FusedActivationType.LeakyReLU
            or FusedActivationType.Swish;

    private sealed class CsrWeights
    {
        public CsrWeights(
            int[] rowOffsets,
            int[] colIndices,
            Tensor<float> values,
            int inputFeatures,
            int outputFeatures)
        {
            RowOffsets = rowOffsets;
            ColIndices = colIndices;
            Values = values;
            InputFeatures = inputFeatures;
            OutputFeatures = outputFeatures;
        }

        public int[] RowOffsets { get; }
        public int[] ColIndices { get; }
        public Tensor<float> Values { get; }
        public int InputFeatures { get; }
        public int OutputFeatures { get; }
    }
}
