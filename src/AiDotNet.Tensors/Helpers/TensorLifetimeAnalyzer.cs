using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Analyzes tensor lifetimes in a computation graph to enable memory reuse.
/// Given a sequence of operations with their input/output tensor IDs,
/// computes the live range [firstUse, lastUse] for each tensor and determines
/// which tensors can share the same workspace slot (non-overlapping lifetimes).
/// </summary>
/// <remarks>
/// <para>
/// This is the foundation for beating PyTorch's memory efficiency. PyTorch's
/// caching allocator assigns a separate memory block per tensor. By analyzing
/// lifetimes, we can reuse memory: if tensor A is dead before tensor B is born,
/// they can share the same workspace slot.
/// </para>
/// <para>
/// The algorithm uses interval graph coloring (equivalent to register allocation):
/// 1. Compute live ranges for each tensor
/// 2. Sort by start time
/// 3. Greedily assign to slots, reusing slots from dead tensors
/// </para>
/// </remarks>
public sealed class TensorLifetimeAnalyzer
{
    /// <summary>
    /// Represents an operation in the computation graph.
    /// </summary>
    public readonly struct Operation
    {
        /// <summary>Tensor IDs consumed (read) by this operation.</summary>
        public readonly int[] Inputs;

        /// <summary>Tensor IDs produced (written) by this operation.</summary>
        public readonly int[] Outputs;

        /// <summary>Element count for each output tensor.</summary>
        public readonly int[] OutputSizes;

        /// <summary>
        /// Whether this operation can execute in-place (output overwrites first input).
        /// Only valid when Outputs.Length == 1 and Inputs.Length >= 1 and output size == first input size.
        /// </summary>
        public readonly bool CanExecuteInPlace;

        public Operation(int[] inputs, int[] outputs, int[] outputSizes, bool canExecuteInPlace = false)
        {
            Inputs = inputs;
            Outputs = outputs;
            OutputSizes = outputSizes;
            CanExecuteInPlace = canExecuteInPlace;
        }
    }

    /// <summary>
    /// Represents the live range of a tensor: [FirstUse, LastUse] in operation indices.
    /// </summary>
    public readonly struct LiveRange
    {
        /// <summary>Operation index where this tensor is first produced.</summary>
        public readonly int FirstUse;

        /// <summary>Operation index where this tensor is last consumed.</summary>
        public readonly int LastUse;

        /// <summary>Number of elements in this tensor.</summary>
        public readonly int Size;

        /// <summary>The workspace slot assigned to this tensor (-1 if not assigned).</summary>
        public readonly int SlotId;

        public LiveRange(int firstUse, int lastUse, int size, int slotId = -1)
        {
            FirstUse = firstUse;
            LastUse = lastUse;
            Size = size;
            SlotId = slotId;
        }

        /// <summary>Returns true if this tensor is alive at the given operation index.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsAliveAt(int opIndex) => opIndex >= FirstUse && opIndex <= LastUse;

        /// <summary>Returns true if this live range overlaps with another.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Overlaps(LiveRange other) => FirstUse <= other.LastUse && other.FirstUse <= LastUse;
    }

    /// <summary>
    /// Result of the workspace slot assignment, mapping tensor IDs to workspace slots.
    /// </summary>
    public sealed class AllocationPlan
    {
        /// <summary>Live range for each tensor ID.</summary>
        public LiveRange[] LiveRanges { get; }

        /// <summary>Workspace slot ID assigned to each tensor ID.</summary>
        public int[] SlotAssignments { get; }

        /// <summary>Size (element count) of each workspace slot.</summary>
        public int[] SlotSizes { get; }

        /// <summary>Total number of workspace slots needed.</summary>
        public int SlotCount { get; }

        /// <summary>Total elements without reuse (sum of all tensor sizes).</summary>
        public int TotalElementsWithoutReuse { get; }

        /// <summary>Total elements with reuse (sum of slot sizes).</summary>
        public int TotalElementsWithReuse { get; }

        /// <summary>Memory savings ratio: 1 - (withReuse / withoutReuse).</summary>
        public double SavingsRatio => TotalElementsWithoutReuse > 0
            ? 1.0 - (double)TotalElementsWithReuse / TotalElementsWithoutReuse
            : 0.0;

        /// <summary>
        /// Operations that can execute in-place (output overwrites input).
        /// Key: operation index, Value: (inputTensorId, outputTensorId) that share a slot.
        /// </summary>
        public Dictionary<int, (int InputTensorId, int OutputTensorId)> InPlaceOps { get; }

        internal AllocationPlan(LiveRange[] liveRanges, int[] slotAssignments, int[] slotSizes,
            Dictionary<int, (int, int)>? inPlaceOps = null)
        {
            LiveRanges = liveRanges;
            SlotAssignments = slotAssignments;
            SlotSizes = slotSizes;
            SlotCount = slotSizes.Length;
            TotalElementsWithoutReuse = 0;
            foreach (var lr in liveRanges)
                TotalElementsWithoutReuse += lr.Size;
            TotalElementsWithReuse = 0;
            foreach (int s in slotSizes)
                TotalElementsWithReuse += s;
            InPlaceOps = inPlaceOps ?? new Dictionary<int, (int, int)>();
        }
    }

    /// <summary>
    /// Analyzes a sequence of operations and computes an optimal allocation plan
    /// that minimizes workspace memory by reusing slots for non-overlapping tensors.
    /// </summary>
    /// <param name="operations">Ordered sequence of operations in the computation graph.</param>
    /// <param name="tensorCount">Total number of unique tensor IDs used across all operations.</param>
    /// <param name="inputTensorIds">Tensor IDs that are external inputs (not produced by any operation).
    /// These are excluded from workspace allocation since they're owned by the caller.</param>
    /// <returns>An allocation plan mapping tensor IDs to workspace slots.</returns>
    public static AllocationPlan Analyze(ReadOnlySpan<Operation> operations, int tensorCount, ReadOnlySpan<int> inputTensorIds)
    {
        // Step 1: Compute live ranges
        var liveRanges = ComputeLiveRanges(operations, tensorCount);

        // Step 2: Mark input tensors as excluded (they're not workspace-managed)
        var isInput = new bool[tensorCount];
        foreach (int id in inputTensorIds)
        {
            if (id >= 0 && id < tensorCount)
                isInput[id] = true;
        }

        // Step 3: Greedy interval coloring — assign slots to non-input tensors
        // Sort tensor IDs by first use (earliest first) for greedy assignment
        var sortedIds = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) sortedIds[i] = i;
        Array.Sort(sortedIds, (a, b) => liveRanges[a].FirstUse.CompareTo(liveRanges[b].FirstUse));

        var slotAssignments = new int[tensorCount];
        Array.Fill(slotAssignments, -1);

        // Each slot tracks: current size and last-use time of current occupant
        var slots = new List<(int size, int lastUse)>();

        foreach (int tensorId in sortedIds)
        {
            if (isInput[tensorId]) continue;

            var lr = liveRanges[tensorId];
            if (lr.Size == 0) continue; // Skip zero-size tensors

            // Try to find an existing slot that is free (last occupant dead before this tensor starts)
            // and large enough (or will be expanded)
            int bestSlot = -1;
            int bestWaste = int.MaxValue;

            for (int s = 0; s < slots.Count; s++)
            {
                var (slotSize, slotLastUse) = slots[s];
                if (slotLastUse < lr.FirstUse)
                {
                    // Slot is free — prefer one with minimal size waste
                    int waste = slotSize >= lr.Size ? slotSize - lr.Size : lr.Size - slotSize;
                    if (waste < bestWaste)
                    {
                        bestWaste = waste;
                        bestSlot = s;
                    }
                }
            }

            if (bestSlot >= 0)
            {
                // Reuse existing slot — expand if needed
                slotAssignments[tensorId] = bestSlot;
                var (oldSize, _) = slots[bestSlot];
                slots[bestSlot] = (Math.Max(oldSize, lr.Size), lr.LastUse);
            }
            else
            {
                // Allocate new slot
                int newSlot = slots.Count;
                slotAssignments[tensorId] = newSlot;
                slots.Add((lr.Size, lr.LastUse));
            }
        }

        // Build slot sizes array
        var slotSizes = new int[slots.Count];
        for (int i = 0; i < slots.Count; i++)
            slotSizes[i] = slots[i].size;

        // Step 4: Detect in-place opportunities
        // An operation can execute in-place when:
        // 1. CanExecuteInPlace is true
        // 2. The first input tensor dies at this operation (lastUse == opIndex)
        // 3. The output and input have the same size
        // 4. Both are assigned workspace slots
        var inPlaceOps = new Dictionary<int, (int, int)>();
        for (int opIdx = 0; opIdx < operations.Length; opIdx++)
        {
            ref readonly var op = ref operations[opIdx];
            if (!op.CanExecuteInPlace || op.Inputs.Length == 0 || op.Outputs.Length == 0)
                continue;

            int inputId = op.Inputs[0];
            int outputId = op.Outputs[0];

            if (inputId < 0 || inputId >= tensorCount || outputId < 0 || outputId >= tensorCount)
                continue;
            if (isInput[inputId]) continue; // Can't overwrite external inputs

            var inputRange = liveRanges[inputId];
            var outputRange = liveRanges[outputId];

            // Input must die at this operation and sizes must match
            if (inputRange.LastUse == opIdx && inputRange.Size == outputRange.Size &&
                slotAssignments[inputId] >= 0 && slotAssignments[outputId] >= 0)
            {
                // Assign output to input's slot (overwrite in place)
                int oldSlot = slotAssignments[outputId];
                slotAssignments[outputId] = slotAssignments[inputId];
                inPlaceOps[opIdx] = (inputId, outputId);

                // Update slot tracking: the old output slot may now be unused
                // (handled implicitly — slot sizes remain max of all occupants)
            }
        }

        return new AllocationPlan(liveRanges, slotAssignments, slotSizes, inPlaceOps);
    }

    /// <summary>
    /// Computes live ranges for all tensors given a sequence of operations.
    /// </summary>
    private static LiveRange[] ComputeLiveRanges(ReadOnlySpan<Operation> operations, int tensorCount)
    {
        var firstUse = new int[tensorCount];
        var lastUse = new int[tensorCount];
        var sizes = new int[tensorCount];
        Array.Fill(firstUse, int.MaxValue);
        Array.Fill(lastUse, -1);

        for (int opIdx = 0; opIdx < operations.Length; opIdx++)
        {
            ref readonly var op = ref operations[opIdx];

            // Outputs are produced at this operation
            for (int i = 0; i < op.Outputs.Length; i++)
            {
                int tid = op.Outputs[i];
                if (tid < 0 || tid >= tensorCount) continue;
                firstUse[tid] = Math.Min(firstUse[tid], opIdx);
                lastUse[tid] = Math.Max(lastUse[tid], opIdx);
                if (i < op.OutputSizes.Length)
                    sizes[tid] = op.OutputSizes[i];
            }

            // Inputs are consumed at this operation
            foreach (int tid in op.Inputs)
            {
                if (tid < 0 || tid >= tensorCount) continue;
                firstUse[tid] = Math.Min(firstUse[tid], opIdx);
                lastUse[tid] = Math.Max(lastUse[tid], opIdx);
            }
        }

        var ranges = new LiveRange[tensorCount];
        for (int i = 0; i < tensorCount; i++)
        {
            ranges[i] = new LiveRange(
                firstUse[i] == int.MaxValue ? 0 : firstUse[i],
                lastUse[i] == -1 ? 0 : lastUse[i],
                sizes[i]);
        }

        return ranges;
    }

    /// <summary>
    /// Convenience method: creates a TensorWorkspace from an allocation plan.
    /// </summary>
    public static TensorWorkspace<T> CreateWorkspace<T>(AllocationPlan plan, int[][] slotShapes)
    {
        var workspace = new TensorWorkspace<T>();
        for (int i = 0; i < plan.SlotCount; i++)
        {
            workspace.Register(slotShapes[i]);
        }
        workspace.Allocate();
        return workspace;
    }
}
