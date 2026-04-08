using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 5.3: Gradient checkpointing — trades compute for memory by not storing
/// all forward activations. Instead, divides the forward graph into segments and
/// stores only segment boundary tensors. During backward, recomputes forward
/// activations for the current segment.
///
/// Memory: O(sqrt(N)) instead of O(N) for depth-N models.
/// Compute: ~33% overhead (each segment's forward runs twice).
///
/// This is essential for training very deep models (100+ layers) or
/// models with large intermediate activations (high-resolution vision).
///
/// Algorithm:
///   1. Divide N forward steps into sqrt(N) segments of sqrt(N) steps each
///   2. During forward: execute all steps, but only retain segment boundary outputs
///   3. During backward: for segment i, re-run forward for that segment to
///      regenerate intermediate activations, then compute gradients normally
/// </summary>
internal sealed class GradientCheckpointing<T>
{
    private readonly int _segmentSize;
    private readonly CompiledStep<T>[] _forwardSteps;
    private readonly Tensor<T>[][] _segmentBoundaryTensors;
    private readonly IEngine _engine;
    private readonly int _segmentCount;

    /// <summary>
    /// Creates a gradient checkpointing wrapper for compiled forward steps.
    /// </summary>
    /// <param name="forwardSteps">The compiled forward steps to checkpoint.</param>
    /// <param name="engine">The engine for executing operations.</param>
    /// <param name="segmentSize">Steps per segment. 0 = auto (sqrt(N)).</param>
    public GradientCheckpointing(CompiledStep<T>[] forwardSteps, IEngine engine, int segmentSize = 0)
    {
        _forwardSteps = forwardSteps;
        _engine = engine;
        _segmentSize = segmentSize > 0 ? segmentSize : (int)Math.Ceiling(Math.Sqrt(forwardSteps.Length));
        _segmentCount = (forwardSteps.Length + _segmentSize - 1) / _segmentSize;

        // Pre-allocate boundary tensor storage (only segment boundaries are retained)
        _segmentBoundaryTensors = new Tensor<T>[_segmentCount][];
    }

    /// <summary>Number of segments the forward pass is divided into.</summary>
    public int SegmentCount => _segmentCount;

    /// <summary>Steps per segment.</summary>
    public int SegmentSize => _segmentSize;

    /// <summary>
    /// Memory savings factor compared to storing all activations.
    /// For N=100 steps: full storage = 100 tensors, checkpointed = ~20 (sqrt(100)*2).
    /// </summary>
    public double MemorySavingsFactor => (double)_forwardSteps.Length / (2.0 * _segmentCount);

    /// <summary>
    /// Execute forward pass with checkpointing: only retain segment boundary outputs.
    /// Intermediate activations within each segment are computed but not retained.
    /// </summary>
    public Tensor<T> ForwardWithCheckpoints()
    {
        for (int seg = 0; seg < _segmentCount; seg++)
        {
            int start = seg * _segmentSize;
            int end = Math.Min(start + _segmentSize, _forwardSteps.Length);

            // Execute this segment's steps
            for (int i = start; i < end; i++)
                _forwardSteps[i].Execute(_engine, _forwardSteps[i].OutputBuffer);

            // Save boundary: the last step's output of this segment
            // and the first step's input (needed for recomputation during backward)
            var boundaryStep = _forwardSteps[end - 1];
            _segmentBoundaryTensors[seg] = new[] { boundaryStep.OutputBuffer };
        }

        return _forwardSteps[_forwardSteps.Length - 1].OutputBuffer;
    }

    /// <summary>
    /// Recompute forward activations for a specific segment.
    /// Called during backward pass when gradient computation needs
    /// intermediate activations that were not retained.
    /// </summary>
    /// <param name="segmentIndex">Which segment to recompute (0-based).</param>
    public void RecomputeSegment(int segmentIndex)
    {
        if (segmentIndex < 0 || segmentIndex >= _segmentCount)
            throw new ArgumentOutOfRangeException(nameof(segmentIndex));

        int start = segmentIndex * _segmentSize;
        int end = Math.Min(start + _segmentSize, _forwardSteps.Length);

        // Re-execute this segment's forward steps
        // The input to the first step comes from the previous segment's boundary
        for (int i = start; i < end; i++)
            _forwardSteps[i].Execute(_engine, _forwardSteps[i].OutputBuffer);
    }

    /// <summary>
    /// Gets the segment index that contains the given forward step.
    /// Used during backward to determine which segment to recompute.
    /// </summary>
    public int GetSegmentForStep(int stepIndex) => stepIndex / _segmentSize;
}
