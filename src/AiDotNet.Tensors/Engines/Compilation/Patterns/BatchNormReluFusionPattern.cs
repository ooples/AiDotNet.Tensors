// Copyright (c) AiDotNet. All rights reserved.
// Built-in fusion: training BatchNorm -> ReLU as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a training-mode <c>BatchNorm</c> node whose normalized output feeds exactly one <c>ReLU</c> (the conv -&gt; BN
/// -&gt; ReLU of every residual network) and emits one <c>BatchNormReLU</c> node: one kernel writes ReLU(BN(x)) and the
/// backward masks and differentiates in one kernel. Each fused pair saves two full passes over the activation in each
/// direction (the BN output written and re-read, the ReLU's own read and write).
/// </summary>
/// <remarks>
/// <para>Only graphs recorded on <see cref="DirectGpuTensorEngine"/> in float are fused (the CPU plan keeps its own
/// batch-norm backward specializations). The BatchNorm's batch-statistic nodes (<c>BatchNormBatchMean</c> /
/// <c>BatchNormBatchVariance</c>) read the normalized output only to order themselves after it; they are absorbed when
/// nothing consumes their outputs, and the fused node keeps writing those statistic tensors. A statistic that something
/// reads (a running-statistics EMA) leaves the pair unfused: the fused node could not be its producer in the plan.</para>
/// <para>The fused node refreshes the BatchNorm's saved mean and variance tensors every replay, exactly as the
/// BatchNorm node does, because its backward reads them.</para>
/// </remarks>
internal sealed class BatchNormReluFusionPattern : IFusionPattern
{
    public string Name => "BatchNormReLU";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records BatchNormReluBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> bn || bn.OpName != "BatchNorm"
            || bn.RecordingEngine is not DirectGpuTensorEngine
            || bn.BackwardFn is null || bn.SavedState is not { Length: 3 } saved
            || saved[0] is not Tensor<float> mean || saved[1] is not Tensor<float> variance || saved[2] is not double epsilon
            || bn.GetInputsArray() is not { Length: 3 } bnInputs)
            return false;

        LazyNode<float>? relu = null;
        var statistics = new List<LazyNode<float>>(2);
        int consumersSeen = 0;
        for (int j = index + 1; j < nodes.Count; j++)
        {
            var candidate = nodes[j];
            if (alreadyRemoved.Contains(candidate) || !PatternHelpers.IsConsumerOf(candidate, bn)) continue;
            consumersSeen++;
            if (candidate is LazyNode<float> { OpType: LazyNodeType.ReLU } r && relu is null) relu = r;
            else if (candidate is LazyNode<float> s && (s.OpName == "BatchNormBatchMean" || s.OpName == "BatchNormBatchVariance")
                     && PatternHelpers.FanOutAtMostOne(s, consumerCounts)
                     && !(consumerCounts.TryGetValue(s, out int statConsumers) && statConsumers > 0))
                statistics.Add(s);
            else return false;   // another reader of the normalized values, or a statistic something consumes
        }
        if (relu is null || !consumerCounts.TryGetValue(bn, out int bnConsumers) || bnConsumers != consumersSeen)
            return false;

        var x = bnInputs[0];
        var gamma = bnInputs[1];
        var beta = bnInputs[2];
        var statisticOutputs = new (Tensor<float> Output, Tensor<float> Source)[statistics.Count];
        for (int s = 0; s < statistics.Count; s++)
            statisticOutputs[s] = (statistics[s].Output, statistics[s].OpName == "BatchNormBatchMean" ? mean : variance);

        var node = new LazyNode<float>(LazyNodeType.Custom, "BatchNormReLU", new[] { x, gamma, beta }, relu.Output,
            (eng, output) =>
            {
                Tensor<float> result, freshMean, freshVar;
                Tensor<float>? fusedResult = null, fusedMean = null, fusedVar = null;
                if (eng is DirectGpuTensorEngine gpu)
                    fusedResult = gpu.TryBatchNormReluForward(x, gamma, beta, epsilon, out fusedMean, out fusedVar);
                if (fusedResult is not null && fusedMean is not null && fusedVar is not null)
                {
                    result = fusedResult;
                    freshMean = fusedMean;
                    freshVar = fusedVar;
                }
                else
                {
                    var normalized = eng.BatchNorm(x, gamma, beta, epsilon, out freshMean, out freshVar);
                    result = eng.ReLU(normalized);
                    Helpers.TensorAllocator.Return(normalized);
                }
                try
                {
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                    DirectGpuTensorEngine.CopyResultInto(eng, freshMean, mean);
                    DirectGpuTensorEngine.CopyResultInto(eng, freshVar, variance);
                    foreach (var (statisticOutput, source) in statisticOutputs)
                        DirectGpuTensorEngine.CopyResultInto(eng, source, statisticOutput);
                }
                finally
                {
                    Helpers.TensorAllocator.Return(result);
                    Helpers.TensorAllocator.Return(freshMean);
                    Helpers.TensorAllocator.Return(freshVar);
                }
            },
            BackwardFunctions<float>.BatchNormReluBackward, new object[] { mean, variance, epsilon })
        {
            RecordingEngine = bn.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(relu);
        foreach (var s in statistics) alreadyRemoved.Add(s);
        return true;
    }
}
