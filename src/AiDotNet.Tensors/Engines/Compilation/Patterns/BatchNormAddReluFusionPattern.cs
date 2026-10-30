// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Built-in fusion: training BatchNorm -> add(residual) -> ReLU as one op on the GPU engine.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches a training-mode <c>BatchNorm</c> whose normalized output feeds exactly one same-shape <c>TensorAdd</c> (the
/// residual connection) whose sum feeds exactly one <c>ReLU</c> -- the tail of every residual block -- and emits one
/// <c>BatchNormAddReLU</c> node: one kernel writes ReLU(BN(x) + residual), and one backward kernel masks the gradient,
/// writes it as the residual's gradient and runs the BatchNorm backward. The normalized output, the sum and the ReLU's
/// separate passes are gone in both directions.
/// </summary>
/// <remarks>
/// Same scope and statistic-node handling as <see cref="BatchNormReluFusionPattern"/>: float graphs recorded on
/// <see cref="DirectGpuTensorEngine"/>, and the batch-statistic nodes absorbed only when nothing consumes them.
/// </remarks>
internal sealed class BatchNormAddReluFusionPattern : IFusionPattern
{
    public string Name => "BatchNormAddReLU";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records BatchNormAddReluBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (!BatchNormFusionMatch.TryMatchBatchNorm(nodes, index, consumerCounts, alreadyRemoved,
                consumer => consumer is LazyNode<float> { OpType: LazyNodeType.Add },
                out var bn, out var add, out var statistics, out var mean, out var variance, out var epsilon))
            return false;
        // GPU only: the CPU has no fused BatchNorm+Add+ReLU kernel, and the composed fallback would allocate the
        // normalized output and the sum on every replay.
        if (BatchNormFusionMatch.IsCpuOnly(bn!.RecordingEngine)) return false;

        // The add's other operand is the residual: same shape (no broadcasting), and not the normalized output again.
        Tensor<float>? residual = ReferenceEquals(add!.Input0, bn!.Output) ? add.Input1
            : ReferenceEquals(add.Input1, bn.Output) ? add.Input0 : null;
        if (residual is null || ReferenceEquals(residual, bn.Output) || !SameShape(residual._shape, bn.Output._shape)
            || !SameShape(add.Output._shape, bn.Output._shape) || !PatternHelpers.FanOutAtMostOne(add, consumerCounts))
            return false;
        // The fused node takes the BatchNorm's place, so the residual must already exist there. A projection shortcut
        // recorded after this BatchNorm is produced later: decline, and the shortcut's own BatchNorm -- whose add reads
        // this one's (earlier) output as its residual -- fuses instead.
        if (residual.LazySource is ILazyNode residualProducer)
            for (int j = index + 1; j < nodes.Count; j++)
                if (ReferenceEquals(nodes[j], residualProducer)) return false;

        LazyNode<float>? relu = null;
        for (int j = 0; j < nodes.Count; j++)
        {
            var candidate = nodes[j];
            if (alreadyRemoved.Contains(candidate) || !PatternHelpers.IsConsumerOf(candidate, add)) continue;
            if (candidate is LazyNode<float> { OpType: LazyNodeType.ReLU } r && relu is null) relu = r;
            else return false;
        }
        if (relu is null) return false;

        var bnInputs = bn.GetInputsArray();
        var node = BatchNormFusionMatch.BuildFusedNode(bn, bnInputs[0], bnInputs[1], bnInputs[2], residual, relu.Output,
            statistics!, mean!, variance!, epsilon, "BatchNormAddReLU", BackwardFunctions<float>.BatchNormAddReluBackward);
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(bn);
        alreadyRemoved.Add(add);
        alreadyRemoved.Add(relu);
        foreach (var s in statistics!) alreadyRemoved.Add(s);
        return true;
    }

    private static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

/// <summary>Shared matching and node building for the BatchNorm fusion patterns.</summary>
internal static class BatchNormFusionMatch
{
    /// <summary>
    /// A float training BatchNorm recorded on the GPU or CPU engine at <paramref name="index"/> whose consumers are
    /// exactly one node accepted by <paramref name="isDataConsumer"/> plus its batch-statistic nodes. The statistics may be
    /// read (a layer's running-statistics update reads them): the fused node writes them every replay and becomes their
    /// producer (<see cref="BuildFusedNode"/>). Refusing them on the GPU left every AiDotNet BatchNorm layer unfused there,
    /// since each one updates its running statistics.
    /// </summary>
    internal static bool TryMatchBatchNorm(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        System.Func<ILazyNode, bool> isDataConsumer,
        out LazyNode<float>? bn, out LazyNode<float>? dataConsumer, out List<LazyNode<float>>? statistics,
        out Tensor<float>? mean, out Tensor<float>? variance, out double epsilon)
    {
        bn = null; dataConsumer = null; statistics = null; mean = null; variance = null; epsilon = 0;
        if (nodes[index] is not LazyNode<float> candidateBn || candidateBn.OpName != "BatchNorm"
            || !(candidateBn.RecordingEngine is DirectGpuTensorEngine || candidateBn.RecordingEngine is CpuEngine)
            || candidateBn.BackwardFn is null || candidateBn.SavedState is not { Length: 3 } saved
            || saved[0] is not Tensor<float> savedMean || saved[1] is not Tensor<float> savedVariance
            || saved[2] is not double savedEpsilon
            || candidateBn.GetInputsArray() is not { Length: 3 } bnInputs)
            return false;
        // The CPU's fused forward and backward kernels take 4D inputs only; a 2D or 3D BatchNorm stays unfused there
        // rather than reaching a backward that cannot run it.
        bool cpuOnly = IsCpuOnly(candidateBn.RecordingEngine);
        if (cpuOnly && bnInputs[0].Rank != 4) return false;

        var stats = new List<LazyNode<float>>(2);
        LazyNode<float>? consumer = null;
        int consumersSeen = 0;
        for (int j = index + 1; j < nodes.Count; j++)
        {
            var node = nodes[j];
            if (alreadyRemoved.Contains(node) || !PatternHelpers.IsConsumerOf(node, candidateBn)) continue;
            consumersSeen++;
            if (consumer is null && node is LazyNode<float> typed && isDataConsumer(node)) consumer = typed;
            else if (node is LazyNode<float> s && (s.OpName == "BatchNormBatchMean" || s.OpName == "BatchNormBatchVariance"))
                stats.Add(s);
            else return false;   // another reader of the normalized values, or a statistic something consumes
        }
        if (consumer is null || !consumerCounts.TryGetValue(candidateBn, out int bnConsumers) || bnConsumers != consumersSeen)
            return false;
        bn = candidateBn; dataConsumer = consumer; statistics = stats;
        mean = savedMean; variance = savedVariance; epsilon = savedEpsilon;
        return true;
    }

    /// <summary>
    /// The CPU engine itself, not <see cref="DirectGpuTensorEngine"/>, which derives from it: GPU-recorded graphs keep
    /// their own fusion rules.
    /// </summary>
    internal static bool IsCpuOnly(IEngine? engine) => engine is CpuEngine && engine is not DirectGpuTensorEngine;

    /// <summary>
    /// The fused node: ReLU(BN(x) [+ residual]) into <paramref name="output"/>, refreshing the BatchNorm's saved mean and
    /// variance (its backward reads them) and the absorbed statistic nodes' outputs every replay.
    /// </summary>
    internal static LazyNode<float> BuildFusedNode(LazyNode<float> bn, Tensor<float> x, Tensor<float> gamma,
        Tensor<float> beta, Tensor<float>? residual, Tensor<float> output, List<LazyNode<float>> statistics,
        Tensor<float> mean, Tensor<float> variance, double epsilon, string opName, BackwardFunction<float> backward)
    {
        var statisticOutputs = new (Tensor<float> Output, Tensor<float> Source)[statistics.Count];
        for (int s = 0; s < statistics.Count; s++)
            statisticOutputs[s] = (statistics[s].Output, statistics[s].OpName == "BatchNormBatchMean" ? mean : variance);
        var inputs = residual is null ? new[] { x, gamma, beta } : new[] { x, gamma, beta, residual };
        // Whether the last forward ran the register-resident kernel: only then may the backward rebuild the ReLU mask
        // from x (BackwardFunctions.MaskFromInput), since the composed fallback rounds the normalized value its own way.
        // A one-element bool[] rather than a box: SavedStateSerializer writes bool[] (training-plan save).
        var forwardUsedKernel = new bool[1];
        var fused = new LazyNode<float>(LazyNodeType.Custom, opName, inputs, output,
            (eng, result) =>
            {
                Tensor<float> fusedOutput, freshMean, freshVar;
                Tensor<float>? kernelOutput = null, kernelMean = null, kernelVar = null;
                // CPU: BatchNorm and ReLU written straight into the plan's output, the BatchNorm output never stored.
                if (residual is null && eng is CpuEngine cpu && !eng.SupportsGpu
                    && cpu.TryBatchNormReluTrainingInto(x, gamma, beta, epsilon, result, mean, variance))
                {
                    forwardUsedKernel[0] = false;
                    foreach (var (statisticOutput, source) in statisticOutputs)
                        source.AsSpan().CopyTo(statisticOutput.AsWritableSpan());
                    return;
                }
                if (eng is DirectGpuTensorEngine gpu)
                    kernelOutput = gpu.TryBatchNormReluForward(x, gamma, beta, epsilon, out kernelMean, out kernelVar, residual);
                forwardUsedKernel[0] = kernelOutput is not null && kernelMean is not null && kernelVar is not null;
                if (kernelOutput is not null && kernelMean is not null && kernelVar is not null)
                {
                    fusedOutput = kernelOutput;
                    freshMean = kernelMean;
                    freshVar = kernelVar;
                }
                else
                {
                    var normalized = eng.BatchNorm(x, gamma, beta, epsilon, out freshMean, out freshVar);
                    if (residual is not null)
                    {
                        var sum = eng.TensorAdd(normalized, residual);
                        Helpers.TensorAllocator.Return(normalized);
                        normalized = sum;
                    }
                    fusedOutput = eng.ReLU(normalized);
                    Helpers.TensorAllocator.Return(normalized);
                }
                try
                {
                    DirectGpuTensorEngine.CopyResultInto(eng, fusedOutput, result);
                    DirectGpuTensorEngine.CopyResultInto(eng, freshMean, mean);
                    DirectGpuTensorEngine.CopyResultInto(eng, freshVar, variance);
                    foreach (var (statisticOutput, source) in statisticOutputs)
                        DirectGpuTensorEngine.CopyResultInto(eng, source, statisticOutput);
                }
                finally
                {
                    Helpers.TensorAllocator.Return(fusedOutput);
                    Helpers.TensorAllocator.Return(freshMean);
                    Helpers.TensorAllocator.Return(freshVar);
                }
            },
            backward, new object[] { mean, variance, epsilon, forwardUsedKernel })
        {
            RecordingEngine = bn.RecordingEngine,
        };
        // The fused node writes the absorbed statistic outputs, so it is their producer: a reader of them (a CPU layer's
        // running-statistics update) is ordered after it.
        foreach (var statistic in statistics)
            statistic.Output.LazySource = fused;
        return fused;
    }
}
