// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// General pointwise chain fusion pass: lowers multi-op chains (<c>length &gt;= 2</c>) of elementwise and scalar
/// float32 operations into a single <see cref="CodegenGraph"/> and compiles them via <see cref="CodegenDispatcher"/>
/// on CPU and <see cref="INativeGpuCodegenExecutor"/> (PTX / NVRTC CUDA C) on GPU, eliminating intermediate
/// global memory allocations and kernel launches.
/// </summary>
internal sealed class ElementwiseChainFusionPattern : IFusionPattern
{
    private const int MaxChainNodes = 16;

    public string Name => "ElementwiseCodegenChain";

    // General multi-op codegen fusion runs on inference plans (training-specific multi-op patterns with exact
    // analytic adjoints are registered ahead of this pattern with IsDifferentiable = true).
    public bool IsDifferentiable => false;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> firstNode || !IsSupportedChainNode(firstNode))
            return false;
        if (!PatternHelpers.FanOutAtMostOne(firstNode, consumerCounts))
            return false;

        var chain = new List<LazyNode<float>>(8) { firstNode };
        var chainOutputs = new HashSet<Tensor<float>>(Autodiff.ReferenceEqualityComparer<Tensor<float>>.Instance)
        {
            firstNode.Output
        };

        int scan = index + 1;
        while (scan < nodes.Count && chain.Count < MaxChainNodes)
        {
            if (alreadyRemoved.Contains(nodes[scan]))
            {
                scan++;
                continue;
            }

            if (nodes[scan] is LazyNode<float> candidate
                && IsSupportedChainNode(candidate)
                && PatternHelpers.SameShape(candidate.Output, firstNode.Output)
                && ConsumesAnyChainOutput(candidate, chainOutputs))
            {
                // Verify any earlier chain node consumed by candidate had no outside consumers.
                var lastAdded = chain[chain.Count - 1];
                if (!PatternHelpers.FanOutAtMostOne(lastAdded, consumerCounts) && chain.Count >= 2)
                {
                    break;
                }

                chain.Add(candidate);
                chainOutputs.Add(candidate.Output);
                scan++;
                if (!PatternHelpers.FanOutAtMostOne(candidate, consumerCounts))
                {
                    // Terminal node of the fused chain may fan out to multiple downstream readers.
                    break;
                }
                continue;
            }

            break;
        }

        if (chain.Count < 2)
            return false;

        // Every interior node in chain[0 .. Count - 2] must have FanOutAtMostOne so no external reader is stranded.
        for (int i = 0; i < chain.Count - 1; i++)
        {
            if (!PatternHelpers.FanOutAtMostOne(chain[i], consumerCounts))
                return false;
        }

        if (!ElementwiseChainProgram.TryEncode(chain, out int[] program, out var externalInputs))
            return false;
        var codegenGraph = ElementwiseChainProgram.BuildGraph(program, externalInputs, (int[])firstNode.Output._shape.Clone());

        CodegenKernel? cpuKernel = CodegenDispatcher.TryEmitCpu(codegenGraph, CodegenElementType.Float32);
        var terminal = chain[chain.Count - 1];

        for (int i = 0; i < chain.Count - 1; i++)
        {
            chain[i].Output.LazySource = null;
        }

        var capturedChain = chain.ToArray();
        var capturedInputs = externalInputs.ToArray();

        var fusedNode = new LazyNode<float>(
            LazyNodeType.Custom,
            "FusedElementwiseChain",
            capturedInputs,
            terminal.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteCodegenGraphInto(codegenGraph, capturedInputs, output))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>() && cpuKernel is not null)
                {
                    if (TryExecuteCpuCodegenKernel(cpuKernel, capturedInputs, output))
                    {
                        return;
                    }
                }

                // Exact sequential fallback if inputs are strided views or codegen declined.
                for (int s = 0; s < capturedChain.Length - 1; s++)
                {
                    capturedChain[s].Execute(eng, capturedChain[s].Output);
                }
                capturedChain[capturedChain.Length - 1].Execute(eng, output);
            },
            null,
            // The program is the step's saved state, so a stored plan rebuilds this same fused kernel on load.
            new object[] { program })
        {
            RecordingEngine = terminal.RecordingEngine
        };

        fused = fusedNode;
        PatternHelpers.SetLazySource(fusedNode);
        for (int i = 0; i < chain.Count; i++)
        {
            alreadyRemoved.Add(chain[i]);
        }
        return true;
    }

    internal static bool TryExecuteCpuCodegenKernel(
        CodegenKernel kernel,
        Tensor<float>[] inputs,
        Tensor<float> output)
    {
        var outArr = output.GetCpuBackingForContiguousWrite(out int outOffset);
        if (outArr is null || outOffset != 0)
            return false;

        var inArrays = new float[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            var arr = inputs[i].GetCpuBackingForStridedRead(out int inOffset);
            if (arr is null || inOffset != 0 || !inputs[i].IsContiguous)
                return false;
            inArrays[i] = arr;
        }

        kernel.Execute(inArrays, new[] { outArr });
        output.IncrementVersion();
        return true;
    }

    private static bool ConsumesAnyChainOutput(LazyNode<float> node, HashSet<Tensor<float>> outputs)
    {
        if (outputs.Contains(node.Input0)) return true;
        if (node.Input1 is not null && outputs.Contains(node.Input1)) return true;
        if (node.Input2 is not null && outputs.Contains(node.Input2)) return true;
        return false;
    }

    private static bool IsSupportedChainNode(LazyNode<float> node)
    {
        switch (node.OpType)
        {
            case LazyNodeType.Negate:
            case LazyNodeType.ReLU:
            case LazyNodeType.GELU:
            case LazyNodeType.Sigmoid:
            case LazyNodeType.Tanh:
            case LazyNodeType.Swish:
                return true;
            case LazyNodeType.ELU:
                return PatternHelpers.TryGetFloatScalar(node, out _);
            case LazyNodeType.AddScalar:
            case LazyNodeType.MultiplyScalar:
            case LazyNodeType.DivideScalar:
                return PatternHelpers.TryGetFloatScalar(node, out _);
            case LazyNodeType.Add:
            case LazyNodeType.Subtract:
            case LazyNodeType.Multiply:
            case LazyNodeType.Divide:
                return node.Input1 is not null && PatternHelpers.SameShape(node.Input0, node.Input1);
            default:
                return false;
        }
    }
}
