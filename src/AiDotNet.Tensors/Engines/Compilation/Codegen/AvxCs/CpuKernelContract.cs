using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;

// Shared admission checks for the managed and intrinsic pointwise emitters.
// Invalid graphs decline before compilation; invalid buffers fail before writes.
internal static class CpuKernelContract
{
    /// <summary>
    /// Elements per chunk when a pointwise kernel is split across the CPU pool: the elementwise grain every CPU
    /// elementwise op uses, in whole vectors of <paramref name="width"/> so only the last chunk has a scalar tail.
    /// </summary>
    internal static int ChunkElements(int width)
        => Math.Max(width, Helpers.CpuParallelSettings.ElementwiseGrain / width * width);

    /// <summary>
    /// Runs a pointwise <paramref name="kernel"/>(inputs, outputs, start, end) over [0, <paramref name="count"/>) in
    /// fixed chunks of <paramref name="chunk"/> elements across the CPU pool; serially on the calling thread when that
    /// is one chunk. Each element is computed alone, so the split (fixed by the grain, not the thread count) cannot
    /// change a result. The degree of parallelism is read once, by ParallelForOrSerial, which also runs serially inside
    /// another parallel region.
    /// </summary>
    internal static void RunChunked<TElement>(TElement[][] inputs, TElement[][] outputs, int count, int chunk,
        Action<TElement[][], TElement[][], int, int> kernel)
    {
        int chunks = count <= 0 ? 0 : (count + chunk - 1) / chunk;
        if (chunks < 2)
        {
            kernel(inputs, outputs, 0, count);
            return;
        }
        long work = (long)count * (inputs.Length + outputs.Length);
        Helpers.CpuParallelSettings.ParallelForOrSerial(0, chunks, work, c =>
        {
            int from = c * chunk;
            kernel(inputs, outputs, from, Math.Min(count, from + chunk));
        }, deterministicSafe: true);
    }

    internal static bool TryValidateGraph(CodegenGraph graph, CodegenElementType dtype,
        out int elementCount, out string reason)
    {
        elementCount = 0;
        reason = string.Empty;
        for (int i = 0; i < graph.Count; i++)
        {
            var node = graph[i];
            if (node.Dtype != dtype)
            {
                reason = $"CPU pointwise kernels require every node to have dtype {dtype}; node {i} has {node.Dtype}.";
                return false;
            }
            if (node.Shape is null || node.Inputs is null)
            {
                reason = $"Node {i} has no shape or input list.";
                return false;
            }
            bool empty = false;
            foreach (int dimension in node.Shape)
            {
                if (dimension < 0)
                {
                    reason = $"Node {i} has a negative dimension.";
                    return false;
                }
                empty |= dimension == 0;
            }
            long count = empty ? 0 : 1;
            if (!empty)
                foreach (int dimension in node.Shape)
                {
                    // Each preceding product is <= int.MaxValue, so this
                    // multiplication cannot overflow long before the check.
                    count *= dimension;
                    if (count > int.MaxValue)
                    {
                        reason = $"CPU pointwise element count exceeds int.MaxValue at node {i}.";
                        return false;
                    }
                }
            if (i == 0) elementCount = (int)count;
            else if (count != elementCount)
            {
                reason = $"CPU pointwise kernels require uniform element counts; mismatch at node {i}.";
                return false;
            }

            int arity = node.Op switch
            {
                CodegenOpKind.LoadInput or CodegenOpKind.Constant => 0,
                CodegenOpKind.StoreOutput => 1,
                _ when CodegenOpKinds.IsUnaryPointwise(node.Op) => 1,
                _ when CodegenOpKinds.IsPointwise(node.Op) => 2,
                _ => -1
            };
            if (node.Inputs.Length != arity)
            {
                reason = $"CPU pointwise kernel cannot lower the operands of {node.Op} at node {i}.";
                return false;
            }
            foreach (int producer in node.Inputs)
                if (producer < 0 || producer >= i)
                {
                    reason = $"Node {i} must reference an earlier producer.";
                    return false;
                }
        }
        return true;
    }

    internal static void ValidateBuffers<T>(T[][] inputs, T[][] outputs,
        int inputCount, int outputCount, int elementCount)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Length != inputCount)
            throw new ArgumentException($"Expected {inputCount} input buffers, got {inputs.Length}.", nameof(inputs));
        if (outputs.Length != outputCount)
            throw new ArgumentException($"Expected {outputCount} output buffers, got {outputs.Length}.", nameof(outputs));
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is null || inputs[i].Length < elementCount)
                throw new ArgumentException($"Input buffer {i} must contain at least {elementCount} elements.", nameof(inputs));
        for (int i = 0; i < outputs.Length; i++)
            if (outputs[i] is null || outputs[i].Length < elementCount)
                throw new ArgumentException($"Output buffer {i} must contain at least {elementCount} elements.", nameof(outputs));
    }
}
