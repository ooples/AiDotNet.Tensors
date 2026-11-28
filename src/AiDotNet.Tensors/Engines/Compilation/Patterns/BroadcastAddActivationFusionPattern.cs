// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches standalone <c>Add(a, b) -&gt; [ReLU | GELU | Swish | Sigmoid | Tanh]</c> and trailing 1D
/// <c>BroadcastAdd(x, bias) -&gt; [ReLU | GELU | Swish | Sigmoid | Tanh]</c> (not preceded by <c>MatMul</c>,
/// which <see cref="LinearFusionPattern"/> claims earlier), fusing them into a single SIMD/NVRTC launch.
/// </summary>
internal sealed class BroadcastAddActivationFusionPattern : IFusionPattern
{
    public string Name => "AddAndBiasActivation";

    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> addNode
            || !PatternHelpers.IsAdd(addNode.OpType)
            || addNode.Input1 is null
            || !PatternHelpers.FanOutAtMostOne(addNode, consumerCounts))
        {
            return false;
        }

        var actNode = PatternHelpers.FindNextFloat(
            nodes,
            index + 1,
            index + 4,
            alreadyRemoved,
            n => IsSupportedActivation(n.OpType) && PatternHelpers.ConsumesOutput(n, addNode));

        if (actNode is null)
            return false;

        var lhs = addNode.Input0;
        var rhs = addNode.Input1;
        bool sameShape = PatternHelpers.SameShape(lhs, rhs);
        bool trailing1D = false;
        Tensor<float> mainInput = lhs;
        Tensor<float> biasInput = rhs;

        if (!sameShape)
        {
            int cols = addNode.Output._shape.Length > 0 ? addNode.Output._shape[addNode.Output._shape.Length - 1] : 0;
            if (cols <= 0 || addNode.Output.Length % cols != 0)
                return false;

            if (rhs._shape.Length == 1 && rhs._shape[0] == cols && PatternHelpers.SameShape(lhs, addNode.Output))
            {
                trailing1D = true;
                mainInput = lhs;
                biasInput = rhs;
            }
            else if (lhs._shape.Length == 1 && lhs._shape[0] == cols && PatternHelpers.SameShape(rhs, addNode.Output))
            {
                trailing1D = true;
                mainInput = rhs;
                biasInput = lhs;
            }
            else
            {
                return false;
            }
        }

        var actKind = MapActivationKind(actNode.OpType, out CodegenOpKind codegenOp);
        if (actKind == RowActivationKind.None)
            return false;

        var outShape = (int[])actNode.Output._shape.Clone();
        var fwdGraph = BuildCodegenGraph(mainInput._shape, biasInput._shape, outShape, codegenOp);
        int colsCount = outShape.Length > 0 ? outShape[outShape.Length - 1] : 1;
        int rowsCount = colsCount > 0 ? actNode.Output.Length / colsCount : 0;

        addNode.Output.LazySource = null;

        var node = new LazyNode<float>(
            LazyNodeType.Custom,
            trailing1D ? "FusedBiasActivation" : "FusedAddActivation",
            new[] { mainInput, biasInput },
            actNode.Output,
            (eng, output) =>
            {
                if (eng.TryGet<DirectGpuTensorEngine>(out var gpu)
                    && gpu.TryExecuteCodegenGraphInto(fwdGraph, new[] { mainInput, biasInput }, output))
                {
                    return;
                }

                if (!eng.Is<DirectGpuTensorEngine>()
                    && mainInput.IsContiguous
                    && biasInput.IsContiguous
                    && output.IsContiguous)
                {
                    if (trailing1D)
                    {
                        CompiledFusedKernels.BiasActivationRowForward(
                            mainInput.AsSpan(), biasInput.AsSpan(), output.AsWritableSpan(), rowsCount, colsCount, actKind);
                        output.IncrementVersion();
                        return;
                    }

                    if (actKind == RowActivationKind.Relu)
                    {
                        CompiledFusedKernels.AddRelu(mainInput.AsSpan(), biasInput.AsSpan(), output.AsWritableSpan());
                        output.IncrementVersion();
                        return;
                    }
                    if (actKind == RowActivationKind.Gelu)
                    {
                        CompiledFusedKernels.AddGelu(mainInput.AsSpan(), biasInput.AsSpan(), output.AsWritableSpan());
                        output.IncrementVersion();
                        return;
                    }
                    if (actKind == RowActivationKind.Swish)
                    {
                        CompiledFusedKernels.AddSwish(mainInput.AsSpan(), biasInput.AsSpan(), output.AsWritableSpan());
                        output.IncrementVersion();
                        return;
                    }
                }

                Tensor<float> preAct;
                if (trailing1D)
                {
                    preAct = new Tensor<float>(outShape);
                    eng.TensorBroadcastAddInto(preAct, mainInput, biasInput);
                }
                else
                {
                    preAct = eng.TensorAdd(mainInput, biasInput);
                }
                var res = ApplyActivationEager(eng, preAct, actKind);
                DirectGpuTensorEngine.CopyResultInto(eng, res, output);
            },
            (gradOutput, inputs, output, _, engine, grads) =>
            {
                var a = inputs[0];
                var b = inputs[1];
                bool needA = DifferentiableOps.IsGradientRequired(a);
                bool needB = DifferentiableOps.IsGradientRequired(b);
                if (!needA && !needB)
                    return;

                Tensor<float> dPre;
                if (actKind == RowActivationKind.Sigmoid)
                {
                    dPre = engine.SigmoidBackward(gradOutput, output);
                }
                else if (actKind == RowActivationKind.Tanh)
                {
                    dPre = engine.TanhBackward(gradOutput, output);
                }
                else
                {
                    Tensor<float> preAct;
                    if (trailing1D)
                    {
                        preAct = new Tensor<float>(outShape);
                        engine.TensorBroadcastAddInto(preAct, a, b);
                    }
                    else
                    {
                        preAct = engine.TensorAdd(a, b);
                    }
                    dPre = actKind switch
                    {
                        RowActivationKind.Relu => engine.ReluBackward(gradOutput, preAct),
                        RowActivationKind.Gelu => engine.GeluBackward(gradOutput, preAct),
                        RowActivationKind.Swish => engine.SwishBackward(gradOutput, preAct),
                        _ => gradOutput
                    };
                }

                if (needA)
                {
                    DifferentiableOps.AccumulateGrad(grads, a, dPre, engine);
                }
                if (needB)
                {
                    if (trailing1D)
                    {
                        var reduceAxes = new int[dPre.Rank - 1];
                        for (int axis = 0; axis < reduceAxes.Length; axis++) reduceAxes[axis] = axis;
                        var dBias = reduceAxes.Length > 0
                            ? engine.ReduceSum(dPre, reduceAxes, keepDims: false)
                            : dPre;
                        DifferentiableOps.AccumulateGrad(grads, b, dBias, engine);
                    }
                    else
                    {
                        DifferentiableOps.AccumulateGrad(grads, b, needA ? dPre.Clone() : dPre, engine);
                    }
                }
            },
            // The activation kind is the step's saved state, so a stored plan rebuilds the same fused op on load.
            // Stored as its integer value: that is the plan file format.
            new object[] { (int)actKind })
        {
            RecordingEngine = actNode.RecordingEngine
        };

        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(addNode);
        alreadyRemoved.Add(actNode);
        return true;
    }

    private static bool IsSupportedActivation(LazyNodeType op) =>
        op is LazyNodeType.ReLU or LazyNodeType.GELU or LazyNodeType.Swish or LazyNodeType.Sigmoid or LazyNodeType.Tanh;

    private static RowActivationKind MapActivationKind(LazyNodeType op, out CodegenOpKind codegenOp)
    {
        switch (op)
        {
            case LazyNodeType.ReLU: codegenOp = CodegenOpKind.ReLU; return RowActivationKind.Relu;
            case LazyNodeType.GELU: codegenOp = CodegenOpKind.GELU; return RowActivationKind.Gelu;
            case LazyNodeType.Swish: codegenOp = CodegenOpKind.Swish; return RowActivationKind.Swish;
            case LazyNodeType.Sigmoid: codegenOp = CodegenOpKind.Sigmoid; return RowActivationKind.Sigmoid;
            case LazyNodeType.Tanh: codegenOp = CodegenOpKind.Tanh; return RowActivationKind.Tanh;
            default: codegenOp = CodegenOpKind.ReLU; return RowActivationKind.None;
        }
    }

    /// <summary>The unfused activation for a kind; the plan codec rebuilds stored steps with the same mapping.</summary>
    internal static Tensor<T> ApplyActivationEager<T>(IEngine eng, Tensor<T> x, RowActivationKind actKind) => actKind switch
    {
        RowActivationKind.Relu => eng.ReLU(x),
        RowActivationKind.Gelu => eng.GELU(x),
        RowActivationKind.Swish => eng.Swish(x),
        RowActivationKind.Sigmoid => eng.Sigmoid(x),
        RowActivationKind.Tanh => eng.Tanh(x),
        _ => x
    };

    private static CodegenGraph BuildCodegenGraph(int[] in0Shape, int[] in1Shape, int[] outShape, CodegenOpKind actOp)
    {
        var g = new CodegenGraph();
        int n0 = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, (int[])in0Shape.Clone(), 0));
        int n1 = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, (int[])in1Shape.Clone(), 1));
        int nAdd = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { n0, n1 }, CodegenElementType.Float32, outShape));
        int nAct = g.AddNode(new CodegenNode(actOp, new[] { nAdd }, CodegenElementType.Float32, outShape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { nAct }, CodegenElementType.Float32, outShape, 0));
        return g;
    }
}
