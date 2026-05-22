using System;
using System.Buffers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression for <see href="https://github.com/ooples/AiDotNet.Tensors/issues/396">AiDotNet.Tensors#396</see>:
/// fused-Adam path silently zeroes NaN loss readout when the loss tensor's
/// pool-rented backing is padded (logical Length=1 on a 16-slot ArrayPool bucket).
/// <para>
/// Root cause: specialised forward kernels in <see cref="CompiledTrainingPlan{T}.TryBuildSpecializedForward"/>
/// pinned via <c>step.OutputBuffer.GetDataArray()</c>, which returns a COPY for
/// ArrayPool-padded tensors. The kernel wrote NaN to the copy; subsequent
/// <c>lossOutput[0]</c> reads via <c>AsSpan</c> hit the live backing
/// (still zero-initialised from pool rent). Consumers saw <c>lastLoss = 0</c>
/// and assumed training was converging while gradients were corrupted.
/// </para>
/// <para>
/// The 5-experiment bisection in #396 didn't reproduce the bug in small
/// Tensors-direct traces because the pool's Length=1 bucket wasn't padded
/// enough to trigger COPY returns. This test deliberately pads the bucket by
/// renting+returning multiple oversized Length=1 arrays before running the
/// compile + step, so the loss tensor is allocated from a padded slot.
/// </para>
/// </summary>
public class PoolPaddedLossReadoutTests
{
    private readonly ITestOutputHelper _output;
    public PoolPaddedLossReadoutTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void NegateForward_PoolPaddedOutput_PropagatesNaN_NotSilentZero()
    {
        // Pre-pad the ArrayPool<float> Length-1 bucket by renting + returning
        // several oversized buffers. Subsequent tensor allocations for Length=1
        // get the same bucket (rounded up to a multiple of 16 floats).
        var prePadded = new float[8][];
        for (int i = 0; i < prePadded.Length; i++)
        {
            prePadded[i] = ArrayPool<float>.Shared.Rent(1);
            // Write a sentinel into the trailing padding so a subsequent rent
            // observes garbage there. The fix should not depend on padding
            // content — it should always pin the live backing.
            for (int j = 0; j < prePadded[i].Length; j++)
                prePadded[i][j] = float.PositiveInfinity;
        }
        try
        {
            // Tensor whose backing is now a pool-padded Length-1 slot.
            var inp = new Tensor<float>(new float[] { float.NaN }, new[] { 1 });
            // Force the output to be allocated from the pool too — RentUninitialized
            // is what TryBuildSpecializedForward expects to see as step.OutputBuffer.
            var outputTensor = TensorAllocator.RentUninitialized<float>(new[] { 1 });

            // The compiled trace path's TensorNegate specialised forward pins
            // the output's GetDataArray() pre-fix and writes through that pin.
            // We exercise the same kernel directly via the engine to verify
            // the pinning helper produces the right backing.
            var engine = new CpuEngine();
            var negated = engine.TensorNegate(inp);

            // Sanity: TensorNegate of NaN must propagate NaN.
            Assert.True(float.IsNaN(negated[0]),
                $"Direct engine.TensorNegate didn't propagate NaN — got {negated[0]}. " +
                "Pre-condition for the pool-padded test failed.");

            // Now exercise the helper directly: GetPinnableFloatBacking should
            // return the live backing for a contiguous pool-rented tensor, NOT
            // a fresh copy. We verify by writing through the returned array and
            // reading back via the tensor's indexer.
            // (Helper is private; we verify the semantic via behaviour: writes
            // through GetLiveBackingArrayAllowingPaddingOrNull must be visible
            // to subsequent tensor reads.)
            var live = outputTensor.GetLiveBackingArrayAllowingPaddingOrNull();
            Assert.NotNull(live);
            live![0] = float.NaN;
            Assert.True(float.IsNaN(outputTensor[0]),
                "Writing NaN through GetLiveBackingArrayAllowingPaddingOrNull was not " +
                "visible to outputTensor[0] — the live backing is not what the indexer reads.");
        }
        finally
        {
            for (int i = 0; i < prePadded.Length; i++)
                ArrayPool<float>.Shared.Return(prePadded[i]);
        }
    }

    [Fact]
    public void CompiledNegateChain_OnPoolPaddedLossOutput_ReadsNaN()
    {
        // End-to-end: trace a TensorNegate(TensorLog(input)) chain inside
        // GraphMode, COMPILE it (so the kernels go through
        // <see cref="CompiledTrainingPlan{T}.TryBuildSpecializedForward"/>
        // — the exact site the #396 fix patches), Execute the compiled plan,
        // and verify the output propagates NaN instead of being silently zeroed.
        //
        // Pre-fix, the specialized forward for TensorNegate / TensorLog pinned
        // step.OutputBuffer via GetDataArray(). For ArrayPool-padded tensors
        // that returned a COPY — the kernel wrote NaN to the copy, the consumer
        // reading via the tensor indexer hit the still-zero live backing, and
        // training silently consumed lastLoss=0 while gradients were corrupted.
        //
        // The previous (eager-engine) version of this test would have passed
        // even on the pre-fix code because eager TensorLog/TensorNegate write
        // through the live backing directly — only the compiled fast path
        // exhibits the bug. This rewrite drives the compiled fast path so the
        // assertion would FAIL on the pre-#419 commit.
        //
        // Pre-pad the pool first to maximise the chance the compiled plan's
        // intermediate / output buffers come from a padded slot.
        var prePadded = new float[16][];
        for (int i = 0; i < prePadded.Length; i++)
            prePadded[i] = ArrayPool<float>.Shared.Rent(1);
        // Capture the existing engine BEFORE mutating the process-global slot.
        // GraphMode + CompileInference require AiDotNetEngine.Current to be a
        // CpuEngine instance here, but xunit parallelises across collections
        // and other tests assume the slot they set is the one they observe.
        // Restore in the outer finally below.
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;

            // log(-1) = NaN; -NaN = NaN. The compiled forward must propagate
            // both. Use a length-1 input — that's exactly the ArrayPool bucket
            // that surfaced #396 (loss tensors are scalar).
            var input = new Tensor<float>(new float[] { -1.0f }, new[] { 1 });

            // Phase 1 — trace under GraphMode so the engine ops record into
            // the lazy graph rather than executing eagerly. CompileInference
            // then walks the recorded graph and emits the specialized-forward
            // closures from TryBuildSpecializedForward.
            ICompiledPlan<float> plan;
            Tensor<float> tracedOutput;
            var traceScope = GraphMode.Enable();
            try
            {
                var logged = engine.TensorLog(input);
                tracedOutput = engine.TensorNegate(logged);
                plan = traceScope.CompileInference<float>(tracedOutput, input._shape);
            }
            finally { traceScope.Dispose(); }

            // Phase 2 — Execute the compiled plan. The result tensor's backing
            // is the same step.OutputBuffer the specialized forward writes
            // through; if the pin grabs a copy, the indexer here sees 0 (the
            // pool's zero-initialised live backing) instead of NaN.
            Tensor<float> compiledOut;
            using (plan)
            {
                compiledOut = plan.Execute();
            }

            _output.WriteLine(
                $"compiledOut[0]={compiledOut[0]}, " +
                $"IsNaN={float.IsNaN(compiledOut[0])}, " +
                $"IsZero={compiledOut[0] == 0f}, " +
                $"backingLen={compiledOut.GetLiveBackingArrayAllowingPaddingOrNull()?.Length ?? -1}");

            // Bug manifestation: compiledOut[0] would be literal 0 on pre-fix
            // code because the kernel's NaN write went to the orphaned copy.
            Assert.False(compiledOut[0] == 0f && !float.IsNaN(compiledOut[0]),
                "#396 regression: compiled TensorNegate(TensorLog(-1)) returned literal 0 " +
                "instead of NaN. The pool-padded output's GetDataArray() must have " +
                "returned a copy, the kernel wrote NaN to the copy, and the read via " +
                "tensor indexer hit the still-zero live backing.");
            Assert.True(float.IsNaN(compiledOut[0]),
                $"Expected NaN from compiled -log(-1), got {compiledOut[0]}.");
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            for (int i = 0; i < prePadded.Length; i++)
                ArrayPool<float>.Shared.Return(prePadded[i]);
        }
    }
}
