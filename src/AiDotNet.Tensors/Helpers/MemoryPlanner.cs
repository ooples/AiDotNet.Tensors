using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Computes the minimum memory layout for a computation graph by combining
/// tensor lifetime analysis with workspace slot assignment.
/// </summary>
/// <remarks>
/// <para>
/// Usage:
/// <code>
/// var planner = new MemoryPlanner();
///
/// // Register operations in execution order
/// int input = planner.AddExternalInput(new[] { 1, 3, 256, 256 });
/// int conv1 = planner.AddOp("Conv2D", inputs: [input], outputShape: new[] { 1, 64, 256, 256 });
/// int relu1 = planner.AddOp("ReLU", inputs: [conv1], outputShape: new[] { 1, 64, 256, 256 }, canInPlace: true);
/// int conv2 = planner.AddOp("Conv2D", inputs: [relu1], outputShape: new[] { 1, 128, 128, 128 });
///
/// // Compute optimal layout
/// var plan = planner.Plan();
///
/// // Create workspace from plan
/// using var workspace = plan.CreateWorkspace&lt;float&gt;();
/// </code>
/// </para>
/// </remarks>
public sealed class MemoryPlanner
{
    private readonly List<int[]> _tensorShapes = new();
    private readonly List<TensorLifetimeAnalyzer.Operation> _operations = new();
    private readonly HashSet<int> _externalInputs = new();
    private readonly HashSet<int> _outputTensorIds = new();
    private int _nextTensorId;

    /// <summary>
    /// Registers an external input tensor (not workspace-managed).
    /// </summary>
    /// <param name="shape">Shape of the input tensor.</param>
    /// <returns>Tensor ID for referencing in subsequent operations.</returns>
    public int AddExternalInput(int[] shape)
    {
        int id = _nextTensorId++;
        _tensorShapes.Add((int[])shape.Clone());
        _externalInputs.Add(id);
        return id;
    }

    /// <summary>
    /// Marks a tensor as a graph output. Output tensors have their lifetimes
    /// extended to the end of the graph so their slots are never recycled.
    /// </summary>
    public void MarkOutput(int tensorId)
    {
        _outputTensorIds.Add(tensorId);
    }

    /// <summary>
    /// Registers an operation that produces a single output tensor.
    /// </summary>
    /// <param name="name">Operation name (for debugging).</param>
    /// <param name="inputs">Tensor IDs consumed by this operation.</param>
    /// <param name="outputShape">Shape of the output tensor.</param>
    /// <param name="canInPlace">Whether the output can overwrite the first input.</param>
    /// <returns>Tensor ID of the output.</returns>
    public int AddOp(string name, int[] inputs, int[] outputShape, bool canInPlace = false)
    {
        int outputId = _nextTensorId++;
        _tensorShapes.Add((int[])outputShape.Clone());

        int outputSize = 1;
        foreach (int dim in outputShape)
            outputSize = checked(outputSize * dim);

        _operations.Add(new TensorLifetimeAnalyzer.Operation(
            inputs: inputs,
            outputs: [outputId],
            outputSizes: [outputSize],
            canExecuteInPlace: canInPlace));

        return outputId;
    }

    /// <summary>
    /// Registers an operation that produces multiple output tensors.
    /// </summary>
    /// <param name="name">Operation name (for debugging).</param>
    /// <param name="inputs">Tensor IDs consumed by this operation.</param>
    /// <param name="outputShapes">Shapes of the output tensors.</param>
    /// <returns>Tensor IDs of the outputs.</returns>
    public int[] AddMultiOutputOp(string name, int[] inputs, int[][] outputShapes)
    {
        var outputIds = new int[outputShapes.Length];
        var outputSizes = new int[outputShapes.Length];

        for (int i = 0; i < outputShapes.Length; i++)
        {
            outputIds[i] = _nextTensorId++;
            _tensorShapes.Add((int[])outputShapes[i].Clone());

            int size = 1;
            foreach (int dim in outputShapes[i])
                size = checked(size * dim);
            outputSizes[i] = size;
        }

        _operations.Add(new TensorLifetimeAnalyzer.Operation(
            inputs: inputs,
            outputs: outputIds,
            outputSizes: outputSizes));

        return outputIds;
    }

    /// <summary>
    /// Computes the optimal memory plan.
    /// </summary>
    /// <returns>A plan with slot assignments, sizes, and in-place operations.</returns>
    public MemoryPlan Plan()
    {
        var opsList = new List<TensorLifetimeAnalyzer.Operation>(_operations);

        // Pin output tensors by adding a dummy final operation that consumes them.
        // This extends their lifetimes to the end of the graph so their slots
        // are never recycled before execution completes.
        if (_outputTensorIds.Count > 0)
        {
            var outputIds = new int[_outputTensorIds.Count];
            int oi = 0;
            foreach (int id in _outputTensorIds)
                outputIds[oi++] = id;
            opsList.Add(new TensorLifetimeAnalyzer.Operation(
                inputs: outputIds, outputs: Array.Empty<int>(), outputSizes: Array.Empty<int>()));
        }

        var ops = opsList.ToArray();
        var inputIds = new int[_externalInputs.Count];
        int idx = 0;
        foreach (int id in _externalInputs)
            inputIds[idx++] = id;

        var allocPlan = TensorLifetimeAnalyzer.Analyze(ops, _nextTensorId, inputIds);

        // Build shape arrays for each slot (use the shape of the largest tensor assigned to each slot)
        var slotShapes = new int[allocPlan.SlotCount][];
        var slotMaxSize = new int[allocPlan.SlotCount];

        for (int tensorId = 0; tensorId < _nextTensorId; tensorId++)
        {
            int slot = allocPlan.SlotAssignments[tensorId];
            if (slot < 0) continue;

            int tensorSize = allocPlan.LiveRanges[tensorId].Size;
            if (tensorSize > slotMaxSize[slot])
            {
                slotMaxSize[slot] = tensorSize;
                slotShapes[slot] = _tensorShapes[tensorId];
            }
        }

        // Fill any null slots with a 1D shape matching the slot size
        for (int i = 0; i < slotShapes.Length; i++)
        {
            if (slotShapes[i] == null)
                slotShapes[i] = [allocPlan.SlotSizes[i]];
        }

        return new MemoryPlan(allocPlan, slotShapes, _tensorShapes.ToArray());
    }

    /// <summary>
    /// Result of memory planning — can create an optimized TensorWorkspace.
    /// </summary>
    public sealed class MemoryPlan
    {
        /// <summary>The underlying allocation plan from lifetime analysis.</summary>
        public TensorLifetimeAnalyzer.AllocationPlan AllocationPlan { get; }

        /// <summary>Shape for each workspace slot.</summary>
        public int[][] SlotShapes { get; }

        /// <summary>Original shapes for each tensor ID.</summary>
        public int[][] TensorShapes { get; }

        /// <summary>Number of workspace slots needed.</summary>
        public int SlotCount => AllocationPlan.SlotCount;

        /// <summary>Memory savings from slot reuse.</summary>
        public double SavingsRatio => AllocationPlan.SavingsRatio;

        /// <summary>Total elements without reuse.</summary>
        public int TotalElementsWithoutReuse => AllocationPlan.TotalElementsWithoutReuse;

        /// <summary>Total elements with reuse.</summary>
        public int TotalElementsWithReuse => AllocationPlan.TotalElementsWithReuse;

        /// <summary>Operations scheduled for in-place execution.</summary>
        public int InPlaceOpCount => AllocationPlan.InPlaceOps.Count;

        internal MemoryPlan(TensorLifetimeAnalyzer.AllocationPlan allocPlan,
            int[][] slotShapes, int[][] tensorShapes)
        {
            AllocationPlan = allocPlan;
            SlotShapes = slotShapes;
            TensorShapes = tensorShapes;
        }

        /// <summary>
        /// Creates a TensorWorkspace configured with the optimal slot layout.
        /// </summary>
        public TensorWorkspace<T> CreateWorkspace<T>()
        {
            var workspace = new TensorWorkspace<T>();
            for (int i = 0; i < SlotCount; i++)
            {
                // Skip zero-size slots (can occur from dummy pin operations)
                if (AllocationPlan.SlotSizes[i] <= 0) continue;
                // Ensure all dimensions are positive
                bool validShape = true;
                foreach (int dim in SlotShapes[i])
                {
                    if (dim <= 0) { validShape = false; break; }
                }
                if (validShape)
                    workspace.Register(SlotShapes[i]);
            }
            workspace.Allocate();
            return workspace;
        }

        /// <summary>
        /// Gets the workspace slot ID for a given tensor ID.
        /// Returns -1 for external inputs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetSlotForTensor(int tensorId) => AllocationPlan.SlotAssignments[tensorId];

        /// <summary>
        /// Returns true if the given operation index was scheduled for in-place execution.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsInPlace(int opIndex) => AllocationPlan.InPlaceOps.ContainsKey(opIndex);
    }
}
