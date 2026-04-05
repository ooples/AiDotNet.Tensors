using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Symbolic representation of the backward computation graph.
/// Built from tape entries at compile time, then optimized via
/// CSE and algebraic simplification before generating executable code.
///
/// Each node represents either a tensor reference, a matrix operation,
/// or an elementwise operation. Nodes track their reference count
/// for common subexpression detection.
/// </summary>
internal abstract class SymbolicNode
{
    private static int _nextId;

    public int Id { get; } = _nextId++;
    public int[] Shape { get; }
    public int RefCount { get; set; }
    public bool IsTranspose { get; set; }

    protected SymbolicNode(int[] shape) => Shape = shape;
}

/// <summary>Reference to a concrete tensor (input, weight, or gradient).</summary>
internal sealed class TensorRefNode<T> : SymbolicNode
{
    public Tensor<T> Tensor { get; }
    public TensorRefNode(Tensor<T> tensor) : base(tensor._shape) => Tensor = tensor;
}

/// <summary>Matrix multiply: C = A @ B</summary>
internal sealed class MatMulNode : SymbolicNode
{
    public SymbolicNode Left { get; set; }
    public SymbolicNode Right { get; set; }
    public bool TransposeLeft { get; set; }
    public bool TransposeRight { get; set; }

    public MatMulNode(SymbolicNode left, SymbolicNode right, int[] outputShape,
        bool transLeft = false, bool transRight = false) : base(outputShape)
    {
        Left = left;
        Right = right;
        TransposeLeft = transLeft;
        TransposeRight = transRight;
        left.RefCount++;
        right.RefCount++;
    }
}

/// <summary>Elementwise operation: output = f(input) or output = a op b</summary>
internal sealed class ElementwiseNode : SymbolicNode
{
    public SymbolicNode Input { get; set; }
    public SymbolicNode? Input2 { get; set; }
    public string Operation { get; } // "relu_deriv", "multiply", "negate", "add"

    public ElementwiseNode(SymbolicNode input, string operation) : base(input.Shape)
    {
        Input = input;
        Operation = operation;
        input.RefCount++;
    }

    public ElementwiseNode(SymbolicNode input, SymbolicNode input2, string operation) : base(input.Shape)
    {
        Input = input;
        Input2 = input2;
        Operation = operation;
        input.RefCount++;
        input2.RefCount++;
    }
}

/// <summary>Reduction: output = sum(input, axis)</summary>
internal sealed class ReduceNode : SymbolicNode
{
    public SymbolicNode Input { get; set; }
    public int[] Axes { get; }

    public ReduceNode(SymbolicNode input, int[] axes, int[] outputShape) : base(outputShape)
    {
        Input = input;
        Axes = axes;
        input.RefCount++;
    }
}

/// <summary>
/// Builds a symbolic backward graph from tape entries.
/// The symbolic graph can then be optimized before execution.
/// </summary>
internal static class SymbolicBackwardGraphBuilder
{
    /// <summary>
    /// Analyzes backward operations and counts how many redundant transposes
    /// and shared subexpressions exist. Returns optimization potential.
    /// </summary>
    internal static BackwardAnalysis Analyze<T>(
        TapeEntryArena<T> entries,
        int[] reachableIndices)
    {
        var transposes = new HashSet<int>(); // tensor hash codes that get transposed
        var transposeCount = 0;
        var matmulCount = 0;

        foreach (int i in reachableIndices)
        {
            ref var entry = ref entries[i];
            var opName = entry.OperationName;

            if (opName == "TensorMatMul" || opName == "FusedLinear" ||
                opName == "FusedLinearReLU" || opName.StartsWith("FusedMatMulAdd"))
            {
                matmulCount++;
                // Each matmul backward produces 2 transposed GEMM calls
                transposeCount += 2;

                // Track which tensors get transposed
                if (entry.Input0 != null)
                    transposes.Add(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(entry.Input0));
                if (entry.InputCount >= 2 && entry.Input1 != null)
                    transposes.Add(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(entry.Input1));
            }
        }

        // Shared transposes: if the same tensor appears as input to multiple matmuls,
        // its transpose is computed multiple times in the naive backward
        int sharedTransposes = 0;
        var tensorUseCounts = new Dictionary<int, int>();
        foreach (int i in reachableIndices)
        {
            ref var entry = ref entries[i];
            void CountTensor(Tensor<T>? t)
            {
                if (t == null) return;
                int hash = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(t);
                if (tensorUseCounts.ContainsKey(hash))
                {
                    tensorUseCounts[hash]++;
                    if (transposes.Contains(hash))
                        sharedTransposes++;
                }
                else
                    tensorUseCounts[hash] = 1;
            }
            CountTensor(entry.Input0);
            if (entry.InputCount >= 2) CountTensor(entry.Input1);
            if (entry.InputCount >= 3) CountTensor(entry.Input2);
        }

        return new BackwardAnalysis
        {
            MatMulCount = matmulCount,
            TransposeCount = transposeCount,
            SharedTransposeCount = sharedTransposes,
            TotalEntries = reachableIndices.Length,
            CanBenefit = sharedTransposes >= 1 || matmulCount >= 2
        };
    }
}

/// <summary>Result of analyzing a backward graph for optimization potential.</summary>
internal struct BackwardAnalysis
{
    public int MatMulCount;
    public int TransposeCount;
    public int SharedTransposeCount;
    public int TotalEntries;
    public bool CanBenefit;
}
