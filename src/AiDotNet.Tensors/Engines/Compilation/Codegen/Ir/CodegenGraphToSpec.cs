// Copyright (c) AiDotNet. All rights reserved.
// CodegenGraph -> CodegenKernelSpec: the missing front end.
//
// The PTX emitter consumed CodegenKernelSpec, which nothing in the engine could
// produce. The only constructors were a hand-written catalog of ten kernels, the adjoint
// deriver, the cost model and the emitter itself; every other reference lived in a test
// or a benchmark. So the kernels were measured in isolation and could never run.
//
// Meanwhile CodegenGraph -- the IR every other emitter already consumes, and the one
// CodegenLowering builds from real CompiledStep chains -- was sitting in the same folder.
// This translates one into the other, so a graph produced by the ordinary lowering path
// reaches the PTX emitter through the ordinary IKernelEmitter contract.
//
// It translates the subset CodegenKernelSpec can express EXACTLY, and declines the rest
// with a reason. The spec's body is
//
//     out = activation( reduce(product of operands) + bias ) * scale
//
// so a pointwise graph maps onto it directly: multiplies become product operands, an add
// becomes the bias, a max-with-zero becomes the activation. Anything outside that shape
// is declined rather than approximated -- a translator that quietly mis-lowers is worse
// than one that refuses, which is the same rule the emitter follows for index maps.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Translates a <see cref="CodegenGraph"/> into a <see cref="CodegenKernelSpec"/>.</summary>
public static class CodegenGraphToSpec
{
    /// <summary>
    /// Attempts the translation.
    /// </summary>
    /// <param name="graph">Graph produced by the ordinary lowering path.</param>
    /// <param name="name">Kernel name to give the spec.</param>
    /// <param name="spec">The translated spec, when this returns true.</param>
    /// <param name="declineReason">Why the graph could not be expressed, otherwise.</param>
    public static bool TryTranslate(
        CodegenGraph graph, string name, out CodegenKernelSpec? spec, out string declineReason)
    {
        spec = null;
        declineReason = string.Empty;

        if (graph is null) { declineReason = "graph was null"; return false; }
        if (graph.OutputNodes.Count != 1)
        {
            declineReason = "the spec has a single output; graph produces " +
                            graph.OutputNodes.Count;
            return false;
        }

        // OutputNodes holds StoreOutput markers; the value being stored is the store's
        // operand, so the body walk starts one node earlier.
        int storeIndex = graph.OutputNodes[0];
        var storeNode = graph[storeIndex];
        int outputIndex = storeNode.Op == CodegenOpKind.StoreOutput && storeNode.Inputs.Length > 0
            ? storeNode.Inputs[0]
            : storeIndex;

        int[] shape = graph[outputIndex].Shape;
        if (shape is null || shape.Length == 0)
        {
            declineReason = "output has no shape";
            return false;
        }

        // Every input must be the same shape as the output. Broadcasting is a different
        // index map per operand and the graph does not carry one, so translating a
        // broadcast would mean guessing which axis was stretched.
        foreach (int inputIndex in graph.InputNodes)
        {
            int[] inputShape = graph[inputIndex].Shape;
            if (inputShape is null || !SameShape(inputShape, shape))
            {
                declineReason = "operand shape differs from the output; broadcasting is " +
                                "not expressible without a per-operand index map";
                return false;
            }
        }

        // Walk the graph from the output back through the body forms the spec supports.
        var productInputs = new List<int>();
        int? biasNode = null;
        var activation = CodegenActivationKind.None;

        int cursor = outputIndex;
        var guard = 0;
        while (guard++ < graph.Count + 1)
        {
            var node = graph[cursor];
            switch (node.Op)
            {
                case CodegenOpKind.ReLU when activation == CodegenActivationKind.None:
                    activation = CodegenActivationKind.ReLU;
                    cursor = node.Inputs[0];
                    continue;

                case CodegenOpKind.Add when biasNode is null && node.Inputs.Length == 2:
                {
                    // Whichever side is a plain input becomes the bias; the other side
                    // continues the chain.
                    int lhs = node.Inputs[0], rhs = node.Inputs[1];
                    if (graph[rhs].Op == CodegenOpKind.LoadInput) { biasNode = rhs; cursor = lhs; }
                    else if (graph[lhs].Op == CodegenOpKind.LoadInput) { biasNode = lhs; cursor = rhs; }
                    else { declineReason = "add of two computed values is not a bias"; return false; }
                    continue;
                }

                case CodegenOpKind.Mul when node.Inputs.Length == 2:
                {
                    // Collect both operands as product terms when both are inputs;
                    // otherwise keep walking the computed side.
                    int lhs = node.Inputs[0], rhs = node.Inputs[1];
                    bool lhsInput = graph[lhs].Op == CodegenOpKind.LoadInput;
                    bool rhsInput = graph[rhs].Op == CodegenOpKind.LoadInput;
                    if (lhsInput && rhsInput)
                    {
                        productInputs.Add(lhs);
                        productInputs.Add(rhs);
                        cursor = -1;
                        break;
                    }
                    if (rhsInput) { productInputs.Add(rhs); cursor = lhs; continue; }
                    if (lhsInput) { productInputs.Add(lhs); cursor = rhs; continue; }
                    declineReason = "product of two computed values is not expressible";
                    return false;
                }

                case CodegenOpKind.LoadInput:
                    productInputs.Add(cursor);
                    cursor = -1;
                    break;

                default:
                    declineReason = "op " + node.Op + " is outside the spec's body form " +
                                    "(activation(reduce(product) + bias) * scale)";
                    return false;
            }
            break;
        }

        if (productInputs.Count == 0)
        {
            declineReason = "no product operand found";
            return false;
        }

        // Build the spec: one parallel axis per output dimension, identity maps
        // throughout. This is a pointwise kernel, so there is no reduction axis.
        var axes = new CodegenAxis[shape.Length];
        for (int d = 0; d < shape.Length; d++)
            axes[d] = CodegenAxis.Parallel("d" + d.ToString(System.Globalization.CultureInfo.InvariantCulture), shape[d]);
        var space = new CodegenIterationSpace(axes);

        var identity = new CodegenAffineExpr[shape.Length];
        for (int d = 0; d < shape.Length; d++) identity[d] = CodegenAffineExpr.Axis(d);

        // Parameter order must be stable and must match what the launcher binds:
        // product operands first, then the bias, then the output.
        var ordered = new List<int>(productInputs);
        if (biasNode.HasValue) ordered.Add(biasNode.Value);

        var bindings = new CodegenTensorBinding[ordered.Count];
        for (int i = 0; i < ordered.Count; i++)
            bindings[i] = new CodegenTensorBinding(
                i, "in" + i.ToString(System.Globalization.CultureInfo.InvariantCulture),
                (int[])shape.Clone(), CloneMap(identity));

        var output = new CodegenTensorBinding(
            ordered.Count, "out", (int[])shape.Clone(), CloneMap(identity), isOutput: true);

        var productIndices = new int[productInputs.Count];
        for (int i = 0; i < productInputs.Count; i++) productIndices[i] = i;

        spec = new CodegenKernelSpec(
            name, space, bindings, output, productIndices,
            CodegenReduceKind.None,
            biasInput: biasNode.HasValue ? ordered.Count - 1 : null,
            activation: activation);
        return true;
    }

    /// <summary>Order the launcher must bind buffers in, matching the translated spec.</summary>
    public static bool TryGetParameterOrder(
        CodegenGraph graph, out IReadOnlyList<int> graphNodeOrder, out string declineReason)
    {
        graphNodeOrder = Array.Empty<int>();
        if (!TryTranslate(graph, "probe", out _, out declineReason)) return false;

        // Re-derive the same order the translation used, so a caller can bind device
        // buffers to graph inputs without reaching into the spec.
        var order = new List<int>();
        foreach (int inputIndex in graph.InputNodes) order.Add(inputIndex);
        graphNodeOrder = order;
        return true;
    }

    private static CodegenAffineExpr[] CloneMap(CodegenAffineExpr[] map)
    {
        var copy = new CodegenAffineExpr[map.Length];
        Array.Copy(map, copy, map.Length);
        return copy;
    }

    private static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}
