// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
    /// Which inputs replay <paramref name="replay"/> copies out of its staging slot (bit i: input i; all until set). An
    /// input whose host contents have not changed since the last replay that copied it is still in its device buffer,
    /// and its staging need not be written. Same timing as the slot's staging.
    /// </summary>
    void SetInputsToCopy(uint replay, uint mask);

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

    /// <summary>
    /// Sets a device copy of <paramref name="source"/>'s first <paramref name="count"/> floats into
    /// <paramref name="destination"/> for the prologue to make at the start of every replay (the compiled plan's
    /// loss-gradient reseed, which as a copy node of its own cost a launch and a gap per replay). Takes effect at the
    /// next <see cref="CapturePrologue"/>; null buffers clear it. False when the copy is not taken (the caller then
    /// records it itself).
    /// </summary>
    bool TrySetPrologueCopy(IGpuBuffer? source, IGpuBuffer? destination, int count);

    /// <summary>
    /// Records the copy of every input out of the current slot. Call first inside the capture. <paramref name="parity"/>
    /// is the replay parity the graph being captured runs at (0 or 1) when the plan alternates two graphs, else 0.
    /// </summary>
    void CapturePrologue(int parity);

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

    /// <summary>
    /// Builds, before the captures, the tables of a ping-pong SGD update: direction 0 reads the weights from
    /// <paramref name="first"/> and writes the updated weights to <paramref name="second"/>, direction 1 the reverse.
    /// </summary>
    void PrepareSgdPingPong(System.Collections.Generic.IReadOnlyList<IGpuBuffer> first,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> second,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> gradients, System.Collections.Generic.IReadOnlyList<int> sizes);

    /// <summary>
    /// Records, inside the capture, one direction of the ping-pong SGD update: the updated weights written to the other
    /// buffer in the same pass that takes the gradients' finiteness reduction, then (when that is not finite) the old
    /// weights copied over them, so the step is discarded as <see cref="CaptureSgdUpdate"/> discards it.
    /// </summary>
    void CaptureSgdPingPongUpdate(int direction, float weightDecay);

    /// <summary>
    /// True when the capture of <paramref name="graphExec"/> handed any memory of <paramref name="buffers"/> to captured
    /// work (directly or through an overlapping buffer).
    /// </summary>
    bool CapturedGraphReads(IntPtr graphExec, System.Collections.Generic.IReadOnlyList<IGpuBuffer> buffers);

    /// <summary>As <see cref="CaptureSgdUpdate"/> for Adam (or AdamW when <paramref name="decoupled"/>), fp32 moments.</summary>
    void CaptureAdamUpdate(bool decoupled, System.Collections.Generic.IReadOnlyList<IGpuBuffer> parameters,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> gradients, System.Collections.Generic.IReadOnlyList<IGpuBuffer> firstMoments,
        System.Collections.Generic.IReadOnlyList<IGpuBuffer> secondMoments, System.Collections.Generic.IReadOnlyList<int> sizes,
        float beta1, float beta2, float epsilon, float weightDecay);
}
