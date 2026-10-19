// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A captured training step's host interface reduced to page-locked memory (<see cref="IGpuGraphCapture.CreateStepIo"/>):
/// the host writes each replay's inputs into a staging slot and reads the step's loss and progress from memory the
/// graph writes, so a replay costs the host one graph launch and no other driver call.
/// </summary>
/// <remarks>
/// Replay <c>r</c> (counted from 0 on both sides; the device keeps its own count) reads staging slot <c>r % 2</c>. The
/// host may rewrite a slot only once the replay that last read it has completed (<see cref="CompletedReplays"/>), and
/// reads replay <c>r</c>'s results (slot <c>r % 2</c>) once <see cref="CompletedReplays"/> exceeds <c>r</c>.
/// </remarks>
internal interface IGraphStepIo : IDisposable
{
    /// <summary>The number of step inputs.</summary>
    int InputCount { get; }

    /// <summary>The float count of input <paramref name="input"/>.</summary>
    int InputLength(int input);

    /// <summary>Host address of input <paramref name="input"/>'s page-locked staging for <paramref name="slot"/> (0 or 1).</summary>
    IntPtr InputStaging(int input, int slot);

    /// <summary>
    /// Whether the replay reading <paramref name="slot"/> copies input <paramref name="input"/> out of its staging
    /// (the default) or leaves the graph's device buffer as the previous copy left it, because the input has not
    /// changed since. Written under the same rule as the staging: only once the replay that last read the slot has
    /// completed.
    /// </summary>
    void SetInputCopied(int input, int slot, bool copy);

    /// <summary>Replays the graph has completed, as the graph last published it.</summary>
    uint CompletedReplays { get; }

    /// <summary>
    /// Waits until at least <paramref name="replays"/> replays have completed. Throws if the device stops making
    /// progress (a faulted or lost context) instead of waiting forever.
    /// </summary>
    void WaitForReplays(uint replays);

    /// <summary>
    /// Hands the launches queued so far to the device. A driver that batches submissions (Windows WDDM) otherwise holds
    /// a graph launch until its next call or a timer, and with no other call per step the host would wait on progress
    /// for the timer (measured ~7.4 ms per step).
    /// </summary>
    void Flush();

    /// <summary>Records the copy of every input out of the current slot. Call first inside the capture.</summary>
    void CapturePrologue();

    /// <summary>
    /// Records the step's publication: the first <paramref name="lossCount"/> floats of <paramref name="loss"/> (at
    /// most 64) and the progress word. Call last inside the capture.
    /// </summary>
    void CaptureEpilogue(IGpuBuffer? loss, int lossCount);

    /// <summary>Replay <paramref name="replay"/>'s loss values, valid once it has completed.</summary>
    ReadOnlySpan<float> Loss(uint replay, int count);

    // ---- captured optimizer --------------------------------------------------------------------------------------

    /// <summary>The most learning rates <see cref="WriteLearningRates"/> takes.</summary>
    int LearningRateCapacity { get; }

    /// <summary>
    /// Sets the device's count of applied optimizer steps (the steps taken before the graph). Call once, before the
    /// first replay.
    /// </summary>
    void InitializeAppliedSteps(uint applied);

    /// <summary>
    /// The applied optimizer steps as of the last completed replay (the initial count before any completes). A replay
    /// whose gradients were not finite does not advance it.
    /// </summary>
    uint AppliedSteps { get; }

    /// <summary>Replay <paramref name="replay"/>'s gradient sum of squares (non-finite: its update was discarded).</summary>
    double SumOfSquares(uint replay);

    /// <summary>
    /// Writes the learning rates replay <paramref name="replay"/> may need into its slot: <c>rates[k]</c> is the rate for
    /// step number <c>firstStep + k</c>, and the replay picks the one for its own step number (applied steps + 1).
    /// </summary>
    void WriteLearningRates(uint replay, int firstStep, ReadOnlySpan<float> rates);

    /// <summary>
    /// Builds, before the capture, the device tables the captured optimizer reads (the graph keeps them, so another
    /// tensor list replacing the backend's shared tables cannot invalidate it). <paramref name="firstMoments"/> and
    /// <paramref name="secondMoments"/> are null for SGD.
    /// </summary>
    void PrepareOptimizer(System.Collections.Generic.IReadOnlyList<IGpuBuffer> parameters,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> gradients,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer>? firstMoments,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer>? secondMoments, System.Collections.Generic.IReadOnlyList<int> sizes);

    /// <summary>
    /// Records, inside the capture, the gradients' finiteness reduction and a plain-SGD update of every tensor that
    /// reads this step's learning rate on the device and writes nothing when the gradients are not finite.
    /// </summary>
    void CaptureSgdUpdate(System.Collections.Generic.IReadOnlyList<IGpuBuffer> parameters,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> gradients, System.Collections.Generic.IReadOnlyList<int> sizes,
        float weightDecay);

    /// <summary>As <see cref="CaptureSgdUpdate"/> for Adam (or AdamW when <paramref name="decoupled"/>), fp32 moments.</summary>
    void CaptureAdamUpdate(bool decoupled, System.Collections.Generic.IReadOnlyList<IGpuBuffer> parameters,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> gradients, System.Collections.Generic.IReadOnlyList<IGpuBuffer> firstMoments,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> secondMoments, System.Collections.Generic.IReadOnlyList<int> sizes,
        float beta1, float beta2, float epsilon, float weightDecay);
}
