// Copyright (c) AiDotNet. All rights reserved.
// CodegenGraph -> CodegenKernelSpec: the front end.
//
// The PTX emitter consumed CodegenKernelSpec, which nothing in the engine could
// produce. The only constructors were a hand-written catalog, the adjoint deriver, the
// cost model and the emitter itself; every other reference lived in a test or a
// benchmark. So the kernels were measured in isolation and could never run.
//
// Meanwhile CodegenGraph -- the IR every other emitter already consumes, and the one
// CodegenLowering builds from real CompiledStep chains -- was sitting in the same folder.
// This translates one into the other, so a graph produced by the ordinary lowering path
// reaches the PTX emitter through the ordinary IKernelEmitter contract.
//
// The first version handled only elementwise chains, which meant every REDUCTION -- so
// every matmul, so every linear layer -- declined, and the PTX path was reachable only
// for fusion chains. This version translates the reduction forms the spec can express
// exactly. The spec's body is
//
//     out = activation( reduce(product of operands) + bias ) * scale
//
// and a matmul is exactly that: the product is A*B, the reduction is over k. What differs
// from a pointwise chain is only the INDEX MAPS, which is what the spec was built to
// carry.
//
// Everything outside that shape is declined with a reason rather than approximated. A
// translator that quietly mis-lowers is worse than one that refuses -- the same rule the
// emitter follows for index maps.

using System;
using System.Collections.Generic;
using System.Globalization;

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
        => TryTranslate(graph, name, out spec, out _, out declineReason);

    /// <summary>
    /// Attempts the translation and reports the graph nodes each kernel parameter binds
    /// to, in parameter order.
    /// </summary>
    /// <param name="graph">Graph produced by the ordinary lowering path.</param>
    /// <param name="name">Kernel name to give the spec.</param>
    /// <param name="spec">The translated spec, when this returns true.</param>
    /// <param name="graphNodeOrder">
    /// Graph node index per kernel input parameter, in order. A launcher binds device
    /// buffers by walking this, so it never has to re-derive the translator's ordering.
    /// </param>
    /// <param name="declineReason">Why the graph could not be expressed, otherwise.</param>
    public static bool TryTranslate(
        CodegenGraph graph, string name, out CodegenKernelSpec? spec,
        out IReadOnlyList<int> graphNodeOrder, out string declineReason)
    {
        spec = null;
        graphNodeOrder = Array.Empty<int>();
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

        int[] outShape = graph[outputIndex].Shape;
        if (outShape is null || outShape.Length == 0)
        {
            declineReason = "output has no shape; a rank-0 result would leave the kernel " +
                            "with no parallel axis at all";
            return false;
        }

        // ---- Strip the epilogue first, so the core op is whatever remains.
        int? biasNode = null;
        var activation = CodegenActivationKind.None;
        int cursor = outputIndex;

        for (int guard = 0; guard <= graph.Count; guard++)
        {
            var node = graph[cursor];
            if (node.Op == CodegenOpKind.ReLU && activation == CodegenActivationKind.None)
            {
                activation = CodegenActivationKind.ReLU;
                cursor = node.Inputs[0];
                continue;
            }
            if (node.Op == CodegenOpKind.Add && biasNode is null && node.Inputs.Length == 2)
            {
                // Whichever side is a plain input becomes the bias; the other continues
                // the chain. Two computed sides is an elementwise add of two subgraphs,
                // which is not the spec's bias.
                int lhs = node.Inputs[0], rhs = node.Inputs[1];
                if (graph[rhs].Op == CodegenOpKind.LoadInput) { biasNode = rhs; cursor = lhs; continue; }
                if (graph[lhs].Op == CodegenOpKind.LoadInput) { biasNode = lhs; cursor = rhs; continue; }
                declineReason = "add of two computed values is not a bias";
                return false;
            }
            break;
        }

        var core = graph[cursor];
        var operands = new List<int>();
        CodegenAxis[] axes;
        CodegenAffineExpr[][] operandMaps;
        CodegenAffineExpr[] outputMap;
        CodegenReduceKind reduce;

        switch (core.Op)
        {
            case CodegenOpKind.MatMul:
            case CodegenOpKind.MatMulTransposeA:
            case CodegenOpKind.MatMulTransposeB:
            case CodegenOpKind.BatchMatMul:
                if (!TryBuildMatMul(graph, core, outShape, operands,
                        out axes!, out operandMaps!, out outputMap!, out declineReason))
                    return false;
                reduce = CodegenReduceKind.Sum;
                break;

            case CodegenOpKind.ReduceSum:
            case CodegenOpKind.ReduceMax:
                if (!TryBuildReduce(graph, core, outShape, operands,
                        out axes!, out operandMaps!, out outputMap!, out reduce, out declineReason))
                    return false;
                break;

            case CodegenOpKind.Mul:
            case CodegenOpKind.LoadInput:
                if (!TryBuildPointwise(graph, cursor, outShape, operands,
                        out axes!, out operandMaps!, out outputMap!, out declineReason))
                    return false;
                reduce = CodegenReduceKind.None;
                break;

            default:
                declineReason = "op " + core.Op + " is outside the spec's body form " +
                                "(activation(reduce(product) + bias) * scale)";
                return false;
        }

        // ---- Bias binds after the product operands, and may broadcast over the output.
        var ordered = new List<int>(operands);
        var maps = new List<CodegenAffineExpr[]>(operandMaps);
        var shapes = new List<int[]>();
        foreach (int node in operands) shapes.Add(graph[node].Shape);

        if (biasNode.HasValue)
        {
            if (!TryBroadcastMap(graph[biasNode.Value].Shape, outShape, outputMap,
                    out var biasMap, out string biasReason))
            {
                declineReason = "bias " + biasReason;
                return false;
            }
            ordered.Add(biasNode.Value);
            maps.Add(biasMap!);
            shapes.Add(graph[biasNode.Value].Shape);
        }

        var bindings = new CodegenTensorBinding[ordered.Count];
        for (int i = 0; i < ordered.Count; i++)
            bindings[i] = new CodegenTensorBinding(
                i, "in" + i.ToString(CultureInfo.InvariantCulture),
                (int[])shapes[i].Clone(), maps[i]);

        var output = new CodegenTensorBinding(
            ordered.Count, "out", (int[])outShape.Clone(), outputMap, isOutput: true);

        var productIndices = new int[operands.Count];
        for (int i = 0; i < operands.Count; i++) productIndices[i] = i;

        spec = new CodegenKernelSpec(
            name, new CodegenIterationSpace(axes), bindings, output, productIndices, reduce,
            biasInput: biasNode.HasValue ? ordered.Count - 1 : null,
            activation: activation);
        graphNodeOrder = ordered;
        return true;
    }

    /// <summary>Order the launcher must bind buffers in, matching the translated spec.</summary>
    public static bool TryGetParameterOrder(
        CodegenGraph graph, out IReadOnlyList<int> graphNodeOrder, out string declineReason)
        => TryTranslate(graph, "probe", out _, out graphNodeOrder, out declineReason);

    // ---- Matmul.
    //
    // C[m,n] = sum_k A[m,k] * B[k,n] is the spec's body exactly: a product of two
    // operands summed over one axis. Only the index maps differ from a pointwise kernel,
    // and carrying those is what the spec exists for. The transpose variants change one
    // map and nothing else, which is the point of expressing operands as maps rather than
    // as strides baked into the emitter.
    private static bool TryBuildMatMul(
        CodegenGraph graph, CodegenNode core, int[] outShape, List<int> operands,
        out CodegenAxis[]? axes, out CodegenAffineExpr[][]? operandMaps,
        out CodegenAffineExpr[]? outputMap, out string reason)
    {
        axes = null; operandMaps = null; outputMap = null; reason = string.Empty;

        bool batched = core.Op == CodegenOpKind.BatchMatMul;
        bool transposeA = core.Op == CodegenOpKind.MatMulTransposeA;
        bool transposeB = core.Op == CodegenOpKind.MatMulTransposeB;
        int rank = batched ? 3 : 2;

        if (core.Inputs.Length != 2)
        {
            reason = core.Op + " needs exactly two operands, got " + core.Inputs.Length;
            return false;
        }

        int[] aShape = graph[core.Inputs[0]].Shape;
        int[] bShape = graph[core.Inputs[1]].Shape;
        if (outShape.Length != rank || aShape.Length != rank || bShape.Length != rank)
        {
            reason = core.Op + " expects rank-" + rank + " operands, got " +
                     aShape.Length + ", " + bShape.Length + " -> " + outShape.Length;
            return false;
        }

        // Axis numbering: [batch,] m, n, k -- with k last and the only reduction.
        int b = batched ? 0 : -1;
        int off = batched ? 1 : 0;
        int m = off + 0, n = off + 1, k = off + 2;

        int extentM = outShape[off];
        int extentN = outShape[off + 1];
        int extentK = transposeA ? aShape[off] : aShape[off + 1];

        int aRows = transposeA ? aShape[off + 1] : aShape[off];
        int bRows = transposeB ? bShape[off + 1] : bShape[off];
        int bCols = transposeB ? bShape[off] : bShape[off + 1];
        if (aRows != extentM || bRows != extentK || bCols != extentN ||
            (batched && (aShape[0] != outShape[0] || bShape[0] != outShape[0])))
        {
            reason = core.Op + " operand shapes do not contract: A and B must share k and " +
                     "match the output in m and n";
            return false;
        }

        var built = new List<CodegenAxis>();
        if (batched) built.Add(CodegenAxis.Parallel("b", outShape[0]));
        built.Add(CodegenAxis.Parallel("m", extentM));
        built.Add(CodegenAxis.Parallel("n", extentN));
        built.Add(CodegenAxis.Reduce("k", extentK));
        axes = built.ToArray();

        var aMap = new List<CodegenAffineExpr>();
        var bMap = new List<CodegenAffineExpr>();
        var oMap = new List<CodegenAffineExpr>();
        if (batched)
        {
            aMap.Add(CodegenAffineExpr.Axis(b));
            bMap.Add(CodegenAffineExpr.Axis(b));
            oMap.Add(CodegenAffineExpr.Axis(b));
        }
        aMap.Add(CodegenAffineExpr.Axis(transposeA ? k : m));
        aMap.Add(CodegenAffineExpr.Axis(transposeA ? m : k));
        bMap.Add(CodegenAffineExpr.Axis(transposeB ? n : k));
        bMap.Add(CodegenAffineExpr.Axis(transposeB ? k : n));
        oMap.Add(CodegenAffineExpr.Axis(m));
        oMap.Add(CodegenAffineExpr.Axis(n));

        operands.Add(core.Inputs[0]);
        operands.Add(core.Inputs[1]);
        operandMaps = new[] { aMap.ToArray(), bMap.ToArray() };
        outputMap = oMap.ToArray();
        return true;
    }

    // ---- Axis reductions.
    //
    // Reduced axes become Reduce axes and kept axes stay Parallel; the operand reads
    // through the identity map either way. Keepdims is a Const(0) in the output map,
    // because a kept dimension of extent one is addressed at index zero.
    private static bool TryBuildReduce(
        CodegenGraph graph, CodegenNode core, int[] outShape, List<int> operands,
        out CodegenAxis[]? axes, out CodegenAffineExpr[][]? operandMaps,
        out CodegenAffineExpr[]? outputMap, out CodegenReduceKind reduce, out string reason)
    {
        axes = null; operandMaps = null; outputMap = null; reason = string.Empty;
        reduce = core.Op == CodegenOpKind.ReduceMax ? CodegenReduceKind.Max : CodegenReduceKind.Sum;

        if (core.Inputs.Length != 1)
        {
            reason = core.Op + " needs exactly one operand, got " + core.Inputs.Length;
            return false;
        }
        if (core.Attribute is not int[] reducedAxes || reducedAxes.Length == 0)
        {
            reason = core.Op + " carries no reduction axes; a full reduction to a scalar " +
                     "would leave the kernel with no parallel axis";
            return false;
        }

        int[] inShape = graph[core.Inputs[0]].Shape;
        var isReduced = new bool[inShape.Length];
        foreach (int raw in reducedAxes)
        {
            int axis = raw < 0 ? inShape.Length + raw : raw;
            if ((uint)axis >= (uint)inShape.Length)
            {
                reason = "reduction axis " + raw + " is outside the operand's rank " + inShape.Length;
                return false;
            }
            isReduced[axis] = true;
        }

        int keptCount = 0;
        foreach (bool r in isReduced) if (!r) keptCount++;
        if (keptCount == 0)
        {
            reason = "every axis is reduced; the kernel would have no parallel axis";
            return false;
        }

        bool keepDims = outShape.Length == inShape.Length;
        if (!keepDims && outShape.Length != keptCount)
        {
            reason = "output rank " + outShape.Length + " matches neither the kept-axis " +
                     "count " + keptCount + " nor the operand rank " + inShape.Length;
            return false;
        }

        var built = new CodegenAxis[inShape.Length];
        var inMap = new CodegenAffineExpr[inShape.Length];
        for (int d = 0; d < inShape.Length; d++)
        {
            string label = "a" + d.ToString(CultureInfo.InvariantCulture);
            built[d] = isReduced[d]
                ? CodegenAxis.Reduce(label, inShape[d])
                : CodegenAxis.Parallel(label, inShape[d]);
            inMap[d] = CodegenAffineExpr.Axis(d);
        }

        var oMap = new CodegenAffineExpr[outShape.Length];
        if (keepDims)
        {
            for (int d = 0; d < inShape.Length; d++)
            {
                if (isReduced[d])
                {
                    if (outShape[d] != 1)
                    {
                        reason = "reduced dimension " + d + " has extent " + outShape[d] +
                                 " in the output; keepdims requires extent one";
                        return false;
                    }
                    oMap[d] = CodegenAffineExpr.Const(0);
                }
                else
                {
                    if (outShape[d] != inShape[d])
                    {
                        reason = "kept dimension " + d + " changes extent, " +
                                 inShape[d] + " -> " + outShape[d];
                        return false;
                    }
                    oMap[d] = CodegenAffineExpr.Axis(d);
                }
            }
        }
        else
        {
            int at = 0;
            for (int d = 0; d < inShape.Length; d++)
            {
                if (isReduced[d]) continue;
                if (outShape[at] != inShape[d])
                {
                    reason = "kept dimension " + d + " changes extent, " +
                             inShape[d] + " -> " + outShape[at];
                    return false;
                }
                oMap[at++] = CodegenAffineExpr.Axis(d);
            }
        }

        operands.Add(core.Inputs[0]);
        axes = built;
        operandMaps = new[] { inMap };
        outputMap = oMap;
        return true;
    }

    // ---- Pointwise product chain: the original front end, unchanged in behaviour.
    private static bool TryBuildPointwise(
        CodegenGraph graph, int cursor, int[] outShape, List<int> operands,
        out CodegenAxis[]? axes, out CodegenAffineExpr[][]? operandMaps,
        out CodegenAffineExpr[]? outputMap, out string reason)
    {
        axes = null; operandMaps = null; outputMap = null; reason = string.Empty;

        for (int guard = 0; guard <= graph.Count; guard++)
        {
            var node = graph[cursor];
            if (node.Op == CodegenOpKind.LoadInput) { operands.Add(cursor); break; }
            if (node.Op != CodegenOpKind.Mul || node.Inputs.Length != 2)
            {
                reason = "op " + node.Op + " is outside the spec's body form " +
                         "(activation(reduce(product) + bias) * scale)";
                return false;
            }

            int lhs = node.Inputs[0], rhs = node.Inputs[1];
            bool lhsInput = graph[lhs].Op == CodegenOpKind.LoadInput;
            bool rhsInput = graph[rhs].Op == CodegenOpKind.LoadInput;
            if (lhsInput && rhsInput) { operands.Add(lhs); operands.Add(rhs); break; }
            if (rhsInput) { operands.Add(rhs); cursor = lhs; continue; }
            if (lhsInput) { operands.Add(lhs); cursor = rhs; continue; }
            reason = "product of two computed values is not expressible";
            return false;
        }

        if (operands.Count == 0) { reason = "no product operand found"; return false; }

        var built = new CodegenAxis[outShape.Length];
        var identity = new CodegenAffineExpr[outShape.Length];
        for (int d = 0; d < outShape.Length; d++)
        {
            built[d] = CodegenAxis.Parallel("d" + d.ToString(CultureInfo.InvariantCulture), outShape[d]);
            identity[d] = CodegenAffineExpr.Axis(d);
        }

        var maps = new CodegenAffineExpr[operands.Count][];
        for (int i = 0; i < operands.Count; i++)
        {
            if (!TryBroadcastMap(graph[operands[i]].Shape, outShape, identity,
                    out var map, out string operandReason))
            {
                reason = "operand " + i + " " + operandReason;
                return false;
            }
            maps[i] = map!;
        }

        axes = built;
        operandMaps = maps;
        outputMap = identity;
        return true;
    }

    /// <summary>
    /// Maps an operand that is right-aligned against the output shape.
    /// </summary>
    /// <remarks>
    /// A broadcast is a per-operand index map, which is exactly what the spec carries, so
    /// it can be derived rather than guessed: a matching dimension reads the output's own
    /// axis, and a dimension of extent one reads index zero for every position of it.
    /// A dimension that is neither is declined, because stretching it would be an
    /// invention.
    /// </remarks>
    private static bool TryBroadcastMap(
        int[] shape, int[] outShape, CodegenAffineExpr[] outputMap,
        out CodegenAffineExpr[]? map, out string reason)
    {
        map = null; reason = string.Empty;
        if (shape.Length > outShape.Length)
        {
            reason = "has rank " + shape.Length + ", above the output's " + outShape.Length;
            return false;
        }

        int offset = outShape.Length - shape.Length;
        var built = new CodegenAffineExpr[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == outShape[offset + i]) built[i] = outputMap[offset + i];
            else if (shape[i] == 1) built[i] = CodegenAffineExpr.Const(0);
            else
            {
                reason = "dimension " + i + " has extent " + shape[i] +
                         " against the output's " + outShape[offset + i] +
                         "; only a matching extent or a broadcast one is expressible";
                return false;
            }
        }
        map = built;
        return true;
    }
}
