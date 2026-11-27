using System;
using System.Threading;
using AiDotNet.Tensors.Helpers;
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

/// <summary>
/// Runs a compiled pointwise kernel(inputs, outputs, start, end) over [0, count) in fixed chunks of whole vectors across
/// the CPU pool, serially on the calling thread when that is one chunk or the pool would not be used.
/// </summary>
/// <remarks>
/// <para>Each element is computed alone, so the split (fixed by the grain, not the thread count) cannot change a
/// result; the loop is deterministic-safe. The dispatch decision comes from
/// <see cref="CpuParallelSettings.TryPlanIndependentDispatch"/>, which reads the degree once and stays serial inside
/// another parallel region.</para>
/// <para>No per-call allocation here: the pool body is a delegate bound once to a cached call slot. A concurrent
/// second caller of the same kernel finds the slot taken and uses a slot of its own.</para>
/// </remarks>
internal sealed class ChunkedPointwiseRunner<TElement>
{
    private readonly Action<TElement[][], TElement[][], int, int> _kernel;
    private readonly int _count;
    private readonly int _chunk;
    private Slot? _idle;

    internal ChunkedPointwiseRunner(Action<TElement[][], TElement[][], int, int> kernel, int count, int width)
    {
        _kernel = kernel;
        _count = count;
        _chunk = CpuKernelContract.ChunkElements(width);
        ChunkCount = count <= 0 ? 0 : (count + _chunk - 1) / _chunk;
    }

    /// <summary>Fixed chunks the elements split into (1: always one serial pass).</summary>
    internal int ChunkCount { get; }

    /// <summary>Elements per chunk (a whole number of vectors).</summary>
    internal int ChunkElements => _chunk;

    internal void Run(TElement[][] inputs, TElement[][] outputs)
    {
        if (ChunkCount < 2
            || !CpuParallelSettings.TryPlanIndependentDispatch(ChunkCount,
                (long)_count * (inputs.Length + outputs.Length), int.MaxValue, deterministicSafe: true,
                out int tasks, out int maxDegree))
        {
            _kernel(inputs, outputs, 0, _count);
            return;
        }

        var slot = Interlocked.Exchange(ref _idle, null) ?? new Slot(this);
        slot.Inputs = inputs;
        slot.Outputs = outputs;
        slot.Tasks = tasks;
        try
        {
            PersistentParallelExecutor.Instance.Execute(tasks, maxDegree, slot.RunTask);
        }
        finally
        {
            slot.Inputs = null;
            slot.Outputs = null;
            Volatile.Write(ref _idle, slot);
        }
    }

    // The arguments of one call. Execute returns only after every task has finished, so the fields are stable while
    // tasks read them, and the slot is handed back for the next call only after that.
    private sealed class Slot
    {
        private readonly ChunkedPointwiseRunner<TElement> _owner;
        internal readonly Action<int> RunTask;
        internal TElement[][]? Inputs;
        internal TElement[][]? Outputs;
        internal int Tasks;

        internal Slot(ChunkedPointwiseRunner<TElement> owner)
        {
            _owner = owner;
            RunTask = Run;
        }

        // Task t of Tasks runs the contiguous chunks [t * ChunkCount / Tasks, (t + 1) * ChunkCount / Tasks).
        private void Run(int task)
        {
            var inputs = Inputs;
            var outputs = Outputs;
            if (inputs is null || outputs is null)
                throw new InvalidOperationException("Pointwise kernel task ran outside its call.");
            using var region = CpuParallelSettings.EnterParallelRegion();
            var owner = _owner;
            int chunks = owner.ChunkCount;
            int first = (int)((long)task * chunks / Tasks);
            int last = (int)((long)(task + 1) * chunks / Tasks);
            for (int c = first; c < last; c++)
            {
                int from = c * owner._chunk;
                owner._kernel(inputs, outputs, from, Math.Min(owner._count, from + owner._chunk));
            }
        }
    }
}
