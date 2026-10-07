// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// One float op that can produce any sub-range of its output. <see cref="Kernel"/> receives the buffers' BASE pointers
/// plus the output range, so a kernel whose arithmetic depends on where an element sits (a vector prefix and a scalar
/// tail, a length-specialized JIT kernel) can reproduce the whole-buffer partition exactly: running every range of an
/// output through <see cref="Kernel"/> writes the same bits as one whole-buffer call.
/// </summary>
internal sealed unsafe class PointwiseRangeOp
{
    /// <summary>Writes o[i] for i in [start, start + count); <paramref name="b"/> is null for a one-input op.</summary>
    internal delegate void RangeKernel(float* a, float* b, float* o, int start, int count);

    internal PointwiseRangeOp(OpType op, int arity, bool elementwise, RangeKernel kernel)
    {
        Op = op;
        Arity = arity;
        Elementwise = elementwise;
        Kernel = kernel;
    }

    internal OpType Op { get; }

    internal int Arity { get; }

    /// <summary>True when output element i reads input element i only (inputs have the output's shape). False for a
    /// gather (a slice), whose output element i reads an input element somewhere else: such an op can only read
    /// buffers that are complete before its group runs.</summary>
    internal bool Elementwise { get; }

    internal RangeKernel Kernel { get; }
}

/// <summary>
/// The single definition of each fusable op's float kernel: the same SIMD routine the op's standalone compiled forward
/// calls (or, for a slice, the same element copy the engine performs), exposed as a range kernel so a fused group can
/// run it tile by tile. Adding an op here makes it fusable in every model's compiled plan; nothing per-model is
/// involved.
/// </summary>
internal static class PointwiseKernelRegistry
{
    /// <summary>Elementwise fusion of compiled forward steps. On by default; <c>AIDOTNET_POINTWISE_FUSION=0</c>
    /// disables it (every step then replays as its own action, as before).</summary>
    internal static bool Enabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_POINTWISE_FUSION") != "0";

    /// <summary>
    /// The range kernel for <paramref name="op"/> producing an output of <paramref name="outShape"/> from inputs of
    /// <paramref name="inShapes"/>, or null when the op has none or the shapes/saved state are not the form it serves.
    /// Every elementwise kernel is bit-identical to the op's whole-buffer compiled forward over ranges that start at
    /// multiples of 64: add/subtract/multiply/divide/negate/ReLU are exact per element, tanh is position independent,
    /// sigmoid resolves its kernel from the whole length and replays that partition per range. A slice is a copy.
    /// </summary>
    internal static unsafe PointwiseRangeOp? TryGet(OpType op, int[] outShape, IReadOnlyList<int[]> inShapes, object[]? savedState)
    {
        long longLength = 1;
        for (int d = 0; d < outShape.Length; d++) longLength *= outShape[d];
        if (longLength <= 0 || longLength > int.MaxValue) return null;
        int length = (int)longLength;

        switch (op)
        {
            case OpType.TensorAdd when Elementwise(2, outShape, inShapes):
                return new PointwiseRangeOp(op, 2, true, (a, b, o, s, c) => SimdKernels.VectorAddUnsafe(a + s, b + s, o + s, c));
            case OpType.TensorSubtract when Elementwise(2, outShape, inShapes):
                return new PointwiseRangeOp(op, 2, true, (a, b, o, s, c) => SimdKernels.VectorSubtractUnsafe(a + s, b + s, o + s, c));
            case OpType.TensorMultiply when Elementwise(2, outShape, inShapes):
                return new PointwiseRangeOp(op, 2, true, (a, b, o, s, c) => SimdKernels.VectorMultiplyUnsafe(a + s, b + s, o + s, c));
            case OpType.TensorDivide when Elementwise(2, outShape, inShapes):
                return new PointwiseRangeOp(op, 2, true, (a, b, o, s, c) => SimdKernels.VectorDivideUnsafe(a + s, b + s, o + s, c));
            case OpType.TensorNegate when Elementwise(1, outShape, inShapes):
                return new PointwiseRangeOp(op, 1, true, (a, b, o, s, c) => SimdKernels.NegateUnsafe(a + s, o + s, c));
            case OpType.ReLU when Elementwise(1, outShape, inShapes):
                return new PointwiseRangeOp(op, 1, true, (a, b, o, s, c) => SimdKernels.ReLUUnsafe(a + s, o + s, c));
            case OpType.Tanh when Elementwise(1, outShape, inShapes):
                return new PointwiseRangeOp(op, 1, true, (a, b, o, s, c) => SimdKernels.TanhUnsafe(a + s, o + s, c));
            case OpType.Sigmoid when Elementwise(1, outShape, inShapes):
                var sigmoid = CpuSigmoidKernel.Resolve(length);
                return new PointwiseRangeOp(op, 1, true, (a, b, o, s, c) => sigmoid.InvokeRange(a, o, s, c));
            case OpType.TensorSliceAxis when inShapes.Count == 1 && TrySliceAxis(outShape, inShapes[0], savedState,
                out int stride, out int axisSize, out int index):
                return new PointwiseRangeOp(op, 1, false, (a, b, o, s, c) => GatherSliceAxis(a, o, s, c, stride, axisSize, index));
            default:
                return null;
        }
    }

    private static bool Elementwise(int arity, int[] outShape, IReadOnlyList<int[]> inShapes)
    {
        if (inShapes.Count != arity) return false;
        for (int k = 0; k < arity; k++)
            if (!SameShape(inShapes[k], outShape)) return false;
        return true;
    }

    /// <summary>TensorSliceAxis(x, axis, index): the output is x with <c>axis</c> removed, at position
    /// <c>index</c> along it. Saved state is <c>{ axis, index }</c>.</summary>
    private static bool TrySliceAxis(int[] outShape, int[] inShape, object[]? savedState,
        out int stride, out int axisSize, out int index)
    {
        stride = axisSize = index = 0;
        if (savedState is null || savedState.Length < 2 || savedState[0] is not int axis || savedState[1] is not int at)
            return false;
        if (inShape.Length != outShape.Length + 1 || axis < 0 || axis >= inShape.Length || at < 0 || at >= inShape[axis])
            return false;
        for (int d = 0, j = 0; d < inShape.Length; d++)
        {
            if (d == axis) continue;
            if (inShape[d] != outShape[j++]) return false;
        }
        stride = 1;
        for (int d = axis + 1; d < inShape.Length; d++) stride *= inShape[d];
        axisSize = inShape[axis];
        index = at;
        return stride > 0;
    }

    /// <summary>dst[i] = src[outer * axisSize * stride + index * stride + j] for i = outer * stride + j in the range:
    /// the engine's slice copy, run by run.</summary>
    private static unsafe void GatherSliceAxis(float* src, float* dst, int start, int count, int stride, int axisSize, int index)
    {
        int end = start + count;
        int i = start;
        while (i < end)
        {
            int outer = i / stride;
            int j = i - outer * stride;
            int run = Math.Min(stride - j, end - i);
            long from = ((long)outer * axisSize + index) * stride + j;
            Buffer.MemoryCopy(src + from, dst + i, (long)run * sizeof(float), (long)run * sizeof(float));
            i += run;
        }
    }

    internal static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

/// <summary>
/// Groups a compiled forward step list into connected sets of registered ops that can replay as ONE action: the
/// group runs at the position of its last member, tile by tile over the output index space, each member's kernel
/// reading the tile its producers just wrote (still in cache) instead of one dispatch and one full memory pass per
/// op. Graph-based, not pattern-based: any connected set of registered ops over float buffers of one element count
/// qualifies, in any model.
/// </summary>
internal static class PointwiseFusionPlanner
{
    /// <summary>How a step participates in grouping.</summary>
    internal enum StepRole : byte
    {
        /// <summary>An ordinary step: a candidate member when its op is registered, else a reader that seals the
        /// groups whose outputs it consumes.</summary>
        Normal = 0,

        /// <summary>Emits no action and computes nothing (a zero-copy reshape alias): ignored. Its output shares its
        /// input's storage, so its readers are seen as readers of that storage.</summary>
        Transparent = 1,

        /// <summary>Executes somewhere other than its own position (claimed by another fusion): seals every open
        /// group, so no group spans it.</summary>
        Barrier = 2,
    }

    /// <summary>
    /// Returns the groups (member step indices, ascending) to replay as one action each: every group of two or more
    /// members, plus single gather members (a slice has no other compiled forward, and its range kernel is a direct
    /// copy). A step is a member only when (1) its op has a registered range kernel for its shapes, (2) every input
    /// and the output are contiguous zero-offset float buffers, the output is not one of its inputs, and each of those
    /// buffers is written by at most one step, and (3) no step positioned before the group's last member reads a
    /// member's output from outside the group. (3) is what makes deferring every member to the last member's
    /// position safe. A gather never joins the group that produces its input (it reads outside its tile).
    /// </summary>
    internal static List<int[]> Plan<T>(IReadOnlyList<CompiledStep<T>> steps, Func<int, StepRole> role)
    {
        var groups = new List<int[]>();
        if (typeof(T) != typeof(float) || steps.Count == 0) return groups;

        // Writers per storage array, so a buffer rewritten by a second step (an in-place op) is never deferred past it.
        var writers = new Dictionary<float[], int>();
        for (int i = 0; i < steps.Count; i++)
        {
            if (role(i) == StepRole.Transparent) continue;
            var arr = Backing(steps[i].OutputBuffer);
            if (arr is not null) writers[arr] = writers.TryGetValue(arr, out int n) ? n + 1 : 1;
        }

        var members = new List<List<int>>();
        var hasGather = new List<bool>();
        var length = new List<int>();
        var open = new List<bool>();
        var parent = new List<int>();
        var groupOfStorage = new Dictionary<float[], int>();

        int Find(int g)
        {
            while (parent[g] != g) { parent[g] = parent[parent[g]]; g = parent[g]; }
            return g;
        }
        void SealAll()
        {
            for (int g = 0; g < open.Count; g++) open[g] = false;
        }
        void SealReadersOf(CompiledStep<T> step)
        {
            for (int k = 0; k < step.Inputs.Length; k++)
            {
                var inArr = step.Inputs[k] is null ? null : Backing(step.Inputs[k]);
                if (inArr is null)
                {
                    // A view (nonzero offset) or a non-host buffer: its storage cannot be matched, so assume the worst.
                    SealAll();
                    return;
                }
                if (groupOfStorage.TryGetValue(inArr, out int g)) open[Find(g)] = false;
            }
        }

        var producerGroups = new List<int>(2);
        for (int i = 0; i < steps.Count; i++)
        {
            var r = role(i);
            if (r == StepRole.Transparent) continue;
            if (r == StepRole.Barrier) { SealAll(); continue; }

            var step = steps[i];
            var op = Candidate(step, writers, out var outArr);
            if (op is null || outArr is null)
            {
                // A reader outside every group: the groups it reads from must run before it, so they take no more members.
                SealReadersOf(step);
                continue;
            }

            producerGroups.Clear();
            if (op.Elementwise)
            {
                for (int k = 0; k < step.Inputs.Length; k++)
                {
                    var inArr = Backing(step.Inputs[k]);
                    if (inArr is null || !groupOfStorage.TryGetValue(inArr, out int g)) continue;
                    g = Find(g);
                    if (!open[g]) continue;
                    if (length[g] != step.OutputBuffer.Length) { open[g] = false; continue; }
                    if (!producerGroups.Contains(g)) producerGroups.Add(g);
                }
            }
            else
            {
                // A gather reads its input outside the current tile: whatever produces that input must be complete
                // first, so the producing group is sealed and this op starts its own group.
                SealReadersOf(step);
            }

            int target;
            if (producerGroups.Count == 0)
            {
                target = members.Count;
                members.Add(new List<int>());
                hasGather.Add(false);
                length.Add(step.OutputBuffer.Length);
                open.Add(true);
                parent.Add(target);
            }
            else
            {
                target = producerGroups[0];
                for (int k = 1; k < producerGroups.Count; k++)
                {
                    int other = producerGroups[k];
                    members[target].AddRange(members[other]);
                    members[other].Clear();
                    hasGather[target] |= hasGather[other];
                    parent[other] = target;
                }
            }
            members[target].Add(i);
            hasGather[target] |= !op.Elementwise;
            groupOfStorage[outArr] = target;
        }

        for (int g = 0; g < members.Count; g++)
        {
            if (parent[g] != g || members[g].Count == 0) continue;
            if (members[g].Count < 2 && !hasGather[g]) continue;
            var m = members[g].ToArray();
            Array.Sort(m);
            groups.Add(m);
        }
        groups.Sort((x, y) => x[x.Length - 1].CompareTo(y[y.Length - 1]));
        return groups;
    }

    private static float[]? Backing<T>(Tensor<T> t)
        => t is Tensor<float> f && f.IsContiguous ? f.GetLiveBackingArrayAllowingPaddingOrNull() : null;

    /// <summary>The step's range op when the step can be a group member (its output storage in
    /// <paramref name="outArr"/>), else null.</summary>
    private static PointwiseRangeOp? Candidate<T>(CompiledStep<T> step, Dictionary<float[], int> writers, out float[]? outArr)
    {
        outArr = null;
        var output = step.OutputBuffer;
        if (output is null || output.Length == 0) return null;
        var inShapes = new int[step.Inputs.Length][];
        for (int k = 0; k < step.Inputs.Length; k++)
        {
            if (step.Inputs[k] is null) return null;
            inShapes[k] = step.Inputs[k]._shape;
        }
        var op = PointwiseKernelRegistry.TryGet(step.OpType, output._shape, inShapes, step.SavedState);
        if (op is null) return null;
        var oArr = Backing(output);
        if (oArr is null || !writers.TryGetValue(oArr, out int w) || w != 1) return null;
        for (int k = 0; k < step.Inputs.Length; k++)
        {
            var inArr = Backing(step.Inputs[k]);
            if (inArr is null || ReferenceEquals(inArr, oArr)) return null;
            if (writers.TryGetValue(inArr, out int iw) && iw != 1) return null;
        }
        outArr = oArr;
        return op;
    }
}

/// <summary>
/// Replays one fused group: every member's range kernel over a tile, tile after tile, so a member reads its producers'
/// tile from cache. Large groups split into chunks over the pool, chunk boundaries on 64-element multiples. Every
/// member still writes its own output buffer (the backward pass reads them), so the result is bit-identical to
/// replaying the members one by one.
/// </summary>
internal sealed unsafe class PointwiseGroupKernel
{
    /// <summary>Elements per tile: 16 KB per buffer, so a chain's working set stays in L1/L2.</summary>
    internal const int TileElements = 4096;

    private const int ParallelChunkElements = 16 * 1024;

    private readonly PointwiseRangeOp[] _ops;
    private readonly int[] _slotA, _slotB, _slotOut;
    private readonly GCHandle[] _handles;
    private readonly IntPtr[] _ptrs;
    private readonly Tensor<float>[] _outputs;
    private readonly int _length, _chunks, _chunkSize;
    private readonly Action<int>? _chunkBody;

    /// <summary>Builds the group's kernel. <paramref name="members"/> are in execution (step) order; every output has
    /// <paramref name="length"/> elements and every tensor is a contiguous zero-offset float buffer (the planner's
    /// guarantee).</summary>
    internal PointwiseGroupKernel(
        IReadOnlyList<(OpType Op, Tensor<float>[] Inputs, Tensor<float> Output, object[]? SavedState)> members,
        int length, List<GCHandle>? handleTracker)
    {
        _length = length;
        _ops = new PointwiseRangeOp[members.Count];
        _slotA = new int[members.Count];
        _slotB = new int[members.Count];
        _slotOut = new int[members.Count];
        _outputs = new Tensor<float>[members.Count];
        var slots = new Dictionary<float[], int>();
        var handles = new List<GCHandle>();
        int Slot(Tensor<float> t)
        {
            var arr = t.GetLiveBackingArrayAllowingPaddingOrNull()
                ?? throw new InvalidOperationException("A fused group was given a tensor without zero-offset live float storage.");
            if (slots.TryGetValue(arr, out int s)) return s;
            var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
            handleTracker?.Add(h);
            handles.Add(h);
            slots[arr] = handles.Count - 1;
            return handles.Count - 1;
        }
        for (int m = 0; m < members.Count; m++)
        {
            var (op, inputs, output, saved) = members[m];
            if (output.Length != length)
                throw new ArgumentException($"Member {m} ({op}) writes {output.Length} elements, the group {length}.", nameof(members));
            var inShapes = new int[inputs.Length][];
            for (int k = 0; k < inputs.Length; k++) inShapes[k] = inputs[k]._shape;
            _ops[m] = PointwiseKernelRegistry.TryGet(op, output._shape, inShapes, saved)
                ?? throw new InvalidOperationException($"No range kernel for {op} with these shapes.");
            _slotA[m] = Slot(inputs[0]);
            _slotB[m] = inputs.Length > 1 ? Slot(inputs[1]) : -1;
            _slotOut[m] = Slot(output);
            _outputs[m] = output;
        }
        _handles = handles.ToArray();
        _ptrs = new IntPtr[_handles.Length];

        int chunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / ParallelChunkElements));
        if (chunks >= 2)
        {
            _chunks = chunks;
            _chunkSize = (((length + chunks - 1) / chunks) + 63) & ~63;
            _chunkBody = RunChunk;
        }
        else
        {
            _chunks = 1;
            _chunkSize = length;
        }
    }

    /// <summary>Number of member ops.</summary>
    internal int MemberCount => _ops.Length;

    internal void Run()
    {
        for (int s = 0; s < _handles.Length; s++) _ptrs[s] = _handles[s].AddrOfPinnedObject();
        if (_chunkBody is null) RunRange(0, _length);
        else PersistentParallelExecutor.Instance.Execute(_chunks, _chunkBody);
        for (int m = 0; m < _outputs.Length; m++) _outputs[m].IncrementVersion();
    }

    private void RunChunk(int chunk)
    {
        int start = chunk * _chunkSize;
        int end = Math.Min(_length, start + _chunkSize);
        if (start < end) RunRange(start, end);
    }

    private void RunRange(int start, int end)
    {
        for (int t = start; t < end; t += TileElements)
        {
            int count = Math.Min(TileElements, end - t);
            for (int m = 0; m < _ops.Length; m++)
            {
                int b = _slotB[m];
                _ops[m].Kernel((float*)_ptrs[_slotA[m]], b < 0 ? null : (float*)_ptrs[b], (float*)_ptrs[_slotOut[m]], t, count);
            }
        }
    }
}