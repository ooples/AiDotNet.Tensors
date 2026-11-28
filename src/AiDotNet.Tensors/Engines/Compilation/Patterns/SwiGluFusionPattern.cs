// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches gated linear activation units in Transformer FFNs:
/// <list type="bullet">
///   <item><c>Swish(gate) * up</c> (SwiGLU)</item>
///   <item><c>(gate * Sigmoid(gate)) * up</c> (decomposed SwiGLU)</item>
///   <item><c>GELU(gate) * up</c> (GeGLU)</item>
/// </list>
/// and fuses them into a single kernel launch without materializing intermediate <c>[B, S, D_ff]</c> tensors
/// in global memory, using Roslyn compile-time AVX2/FMA SIMD on CPU and <c>INativeGpuCodegenExecutor</c> NVRTC on GPU.
/// </summary>
internal sealed class SwiGluFusionPattern : IFusionPattern
{
    // One forward graph per (activation, shape): the GPU engine caches compiled kernels by graph identity, so every
    // rebuild of the same gated unit reuses the same graph and kernel instead of adding a new cache entry per build.
    // Bounded by the distinct FFN shapes a process traces.
    private static readonly ConcurrentDictionary<(bool IsGelu, string Shape), CodegenGraph> ForwardGraphs = new();

    public string Name => "SwiGluGatedActivation";

    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> startNode)
            return false;

        // Variant 1 & 3: Swish(gate) * up or GELU(gate) * up
        if ((startNode.OpType == LazyNodeType.Swish || startNode.OpType == LazyNodeType.GELU)
            && PatternHelpers.FanOutAtMostOne(startNode, consumerCounts))
        {
            var mulNode = PatternHelpers.FindNextFloat(
                nodes,
                index + 1,
                index + 6,
                alreadyRemoved,
                n => n.OpType == LazyNodeType.Multiply
                    && n.Input1 is not null
                    && PatternHelpers.ConsumesOutput(n, startNode));

            if (mulNode is null || !PatternHelpers.TryGetOtherInput(mulNode, startNode.Output, out var upTensor))
                return false;
            if (!PatternHelpers.SameShape(startNode.Input0, upTensor))
                return false;

            bool isGelu = startNode.OpType == LazyNodeType.GELU;
            startNode.Output.LazySource = null;

            var built = BuildFusedGatedNode(startNode.Input0, upTensor, mulNode.Output, isGelu, mulNode.RecordingEngine);
            fused = built;
            PatternHelpers.SetLazySource(built);
            alreadyRemoved.Add(startNode);
            alreadyRemoved.Add(mulNode);
            return true;
        }

        // Variant 2: Sigmoid(gate) -> Multiply(gate, sig) -> Multiply(swish, up)
        if (startNode.OpType == LazyNodeType.Sigmoid && PatternHelpers.FanOutAtMostOne(startNode, consumerCounts))
        {
            var gateTensor = startNode.Input0;
            var swishMul = PatternHelpers.FindNextFloat(
                nodes,
                index + 1,
                index + 5,
                alreadyRemoved,
                n => n.OpType == LazyNodeType.Multiply
                    && n.Input1 is not null
                    && PatternHelpers.FanOutAtMostOne(n, consumerCounts)
                    && PatternHelpers.ConsumesOutput(n, startNode)
                    && PatternHelpers.TryGetOtherInput(n, startNode.Output, out var other)
                    && ReferenceEquals(other, gateTensor));

            if (swishMul is null)
                return false;

            var gateUpMul = PatternHelpers.FindNextFloat(
                nodes,
                index + 2,
                index + 8,
                alreadyRemoved,
                n => n.OpType == LazyNodeType.Multiply
                    && n.Input1 is not null
                    && PatternHelpers.ConsumesOutput(n, swishMul));

            if (gateUpMul is null || !PatternHelpers.TryGetOtherInput(gateUpMul, swishMul.Output, out var upTensor))
                return false;
            if (!PatternHelpers.SameShape(gateTensor, upTensor))
                return false;

            startNode.Output.LazySource = null;
            swishMul.Output.LazySource = null;

            var built = BuildFusedGatedNode(gateTensor, upTensor, gateUpMul.Output, isGelu: false, gateUpMul.RecordingEngine);
            fused = built;
            PatternHelpers.SetLazySource(built);
            alreadyRemoved.Add(startNode);
            alreadyRemoved.Add(swishMul);
            alreadyRemoved.Add(gateUpMul);
            return true;
        }

        return false;
    }

    private static LazyNode<float> BuildFusedGatedNode(
        Tensor<float> gate,
        Tensor<float> up,
        Tensor<float> finalOutput,
        bool isGelu,
        IEngine recordingEngine)
    {
        var shapeClone = (int[])finalOutput._shape.Clone();
        var fwdGraph = GetForwardCodegenGraph(shapeClone, isGelu);
        string opName = isGelu ? "FusedGeGlu" : "FusedSwiGlu";

        return new LazyNode<float>(
            LazyNodeType.Custom,
            opName,
            new[] { gate, up },
            finalOutput,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteCodegenGraphInto(fwdGraph, new[] { gate, up }, output))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && gate.IsContiguous
                    && up.IsContiguous
                    && output.IsContiguous)
                {
                    if (isGelu)
                    {
                        CompiledFusedKernels.GeGluPair(gate.AsSpan(), up.AsSpan(), output.AsWritableSpan());
                    }
                    else
                    {
                        CompiledFusedKernels.SwiGluPair(gate.AsSpan(), up.AsSpan(), output.AsWritableSpan());
                    }
                    output.IncrementVersion();
                    return;
                }

                var activated = isGelu ? eng.GELU(gate) : eng.Swish(gate);
                var prod = eng.TensorMultiply(activated, up);
                DirectGpuTensorEngine.CopyResultInto(eng, prod, output);
            },
            (gradOutput, inputs, _, _, engine, grads) =>
            {
                var g = inputs[0];
                var u = inputs[1];
                bool needGate = DifferentiableOps.IsGradientRequired(g);
                bool needUp = DifferentiableOps.IsGradientRequired(u);
                if (!needGate && !needUp)
                    return;

                if (!isGelu
                    && !engine.Is<DirectGpuTensorEngine>()
                    && gradOutput.IsContiguous
                    && g.IsContiguous
                    && u.IsContiguous)
                {
                    if (needGate)
                    {
                        var dGate = TensorAllocator.Rent<float>(g._shape);
                        CompiledFusedKernels.SwiGluPairGradGate(
                            gradOutput.AsSpan(), g.AsSpan(), u.AsSpan(), dGate.AsWritableSpan());
                        dGate.IncrementVersion();
                        DifferentiableOps.AccumulateGrad(grads, g, dGate, engine);
                    }
                    if (needUp)
                    {
                        var dUp = TensorAllocator.Rent<float>(u._shape);
                        CompiledFusedKernels.SwiGluPairGradUp(
                            gradOutput.AsSpan(), g.AsSpan(), dUp.AsWritableSpan());
                        dUp.IncrementVersion();
                        DifferentiableOps.AccumulateGrad(grads, u, dUp, engine);
                    }
                    return;
                }

                var act = isGelu ? engine.GELU(g) : engine.Swish(g);
                if (needUp)
                {
                    DifferentiableOps.AccumulateGrad(grads, u, engine.TensorMultiply(gradOutput, act), engine);
                }
                if (needGate)
                {
                    var dAct = engine.TensorMultiply(gradOutput, u);
                    var dGate = isGelu ? engine.GeluBackward(dAct, g) : engine.SwishBackward(dAct, g);
                    DifferentiableOps.AccumulateGrad(grads, g, dGate, engine);
                }
            },
            null)
        {
            RecordingEngine = recordingEngine
        };
    }

    internal static CodegenGraph GetForwardCodegenGraph(int[] shape, bool isGelu)
    {
        var key = (isGelu, string.Join(",", shape));
        if (ForwardGraphs.TryGetValue(key, out var cached))
            return cached;
        var ownedShape = (int[])shape.Clone();
        return ForwardGraphs.GetOrAdd(key, _ => CreateForwardCodegenGraph(ownedShape, isGelu));
    }

    private static CodegenGraph CreateForwardCodegenGraph(int[] shape, bool isGelu)
    {
        var g = new CodegenGraph();
        int nGate = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, shape, 0));
        int nUp = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, shape, 1));
        int nAct = g.AddNode(new CodegenNode(isGelu ? CodegenOpKind.GELU : CodegenOpKind.Swish, new[] { nGate }, CodegenElementType.Float32, shape));
        int nMul = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { nAct, nUp }, CodegenElementType.Float32, shape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { nMul }, CodegenElementType.Float32, shape, 0));
        return g;
    }
}
