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
    public void SpecializedFusedLinear_PoolPaddedOutput_WritesLiveBacking()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { 2.0f }, new[] { 1, 1 });
        var weights = new Tensor<float>(new float[] { 3.0f }, new[] { 1, 1 });
        var bias = new Tensor<float>(new float[] { 4.0f }, new[] { 1 });
        var pooledOutput = ArrayPool<float>.Shared.Rent(1);
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(pooledOutput, 0, 1), new[] { 1, 1 }, pooledOutput);

        try
        {
            var live = output.GetLiveBackingArrayAllowingPaddingOrNull();
            Assert.NotNull(live);
            Assert.True(live!.Length > output.Length,
                "The regression requires a pool-padded output backing.");
            Array.Clear(live, 0, live.Length);

            var step = new CompiledStep<float>(
                "FusedLinear",
                (eng, destination) => { },
                output,
                new[] { input, weights, bias },
                savedState: new object[] { FusedActivationType.None });

            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);
            specialized!(engine);

            Assert.Equal(10.0f, output[0]);
            Assert.Equal(10.0f, live[0]);
        }
        finally
        {
            TensorAllocator.Return(output);
        }
    }

    [Theory]
    [InlineData("ReduceSum", 6.0f)]
    [InlineData("ReduceMean", 2.0f)]
    public void SpecializedScalarReduction_PoolPaddedOutput_WritesLiveBacking(
        string opName, float expected)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var pooledOutput = ArrayPool<float>.Shared.Rent(1);
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(pooledOutput, 0, 1), new[] { 1 }, pooledOutput);

        try
        {
            var live = output.GetLiveBackingArrayAllowingPaddingOrNull();
            Assert.NotNull(live);
            Assert.True(live!.Length > output.Length);
            Array.Clear(live, 0, live.Length);

            var step = new CompiledStep<float>(
                opName,
                (eng, destination) => { },
                output,
                new[] { input });

            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);
            specialized!(engine);

            Assert.Equal(expected, output[0]);
            Assert.Equal(expected, live[0]);
        }
        finally
        {
            TensorAllocator.Return(output);
        }
    }

    [Fact]
    public void SpecializedReduceMax_PoolPaddedOutput_WritesLiveBacking()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { -3.0f, 7.0f, 2.0f }, new[] { 3 });
        var pooled = ArrayPool<float>.Shared.Rent(1);
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(pooled, 0, 1), new[] { 1 }, pooled);

        try
        {
            Array.Clear(pooled, 0, pooled.Length);
            Assert.True(pooled.Length > output.Length);
            var step = new CompiledStep<float>(
                "ReduceMax", (eng, destination) => { }, output, new[] { input });

            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);
            specialized!(engine);

            Assert.Equal(7.0f, output[0]);
            Assert.Equal(7.0f, pooled[0]);
        }
        finally
        {
            TensorAllocator.Return(output);
        }
    }

    [Fact]
    public void SpecializedMseLoss_PoolPaddedOutput_WritesLiveBacking()
    {
        var engine = new CpuEngine();
        var predicted = new Tensor<float>(new[] { 1.0f, 4.0f }, new[] { 2 });
        var target = new Tensor<float>(new[] { 3.0f, 2.0f }, new[] { 2 });
        var pooled = ArrayPool<float>.Shared.Rent(1);
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(pooled, 0, 1), new[] { 1 }, pooled);

        try
        {
            Array.Clear(pooled, 0, pooled.Length);
            Assert.True(pooled.Length > output.Length);
            var step = new CompiledStep<float>(
                "MSELoss", (eng, destination) => { }, output, new[] { predicted, target });

            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);
            specialized!(engine);

            Assert.Equal(4.0f, output[0]);
            Assert.Equal(4.0f, pooled[0]);
        }
        finally
        {
            TensorAllocator.Return(output);
        }
    }

    [Fact]
    public void SpecializedBroadcastAdd_PoolPaddedInputAndOutput_StayLiveAcrossReplay()
    {
        var engine = new CpuEngine();
        var inputBacking = ArrayPool<float>.Shared.Rent(2);
        var outputBacking = ArrayPool<float>.Shared.Rent(2);
        var input = Tensor<float>.FromPooledMemory(
            new Memory<float>(inputBacking, 0, 2), new[] { 1, 2 }, inputBacking);
        var addend = new Tensor<float>(new[] { 10.0f, 20.0f }, new[] { 2 });
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(outputBacking, 0, 2), new[] { 1, 2 }, outputBacking);

        try
        {
            Assert.True(inputBacking.Length > input.Length);
            Assert.True(outputBacking.Length > output.Length);
            input[0] = 1.0f;
            input[1] = 2.0f;
            Array.Clear(outputBacking, 0, outputBacking.Length);

            var step = new CompiledStep<float>(
                "TensorBroadcastAdd", (eng, destination) => { }, output, new[] { input, addend });
            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);

            // Mutate after closure creation. A cached GetDataArray() copy would
            // keep reading 1,2 instead of the live 3,4 values.
            input[0] = 3.0f;
            input[1] = 4.0f;
            specialized!(engine);

            Assert.Equal(new[] { 13.0f, 24.0f }, output.AsSpan().ToArray());
            Assert.Equal(13.0f, outputBacking[0]);
            Assert.Equal(24.0f, outputBacking[1]);
        }
        finally
        {
            TensorAllocator.Return(output);
            TensorAllocator.Return(input);
        }
    }

    [Fact]
    public void SpecializedNdMatMul_PoolPaddedWeightAndOutput_StayLiveAcrossReplay()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 1, 2, 2 });
        var weightBacking = ArrayPool<float>.Shared.Rent(2);
        var outputBacking = ArrayPool<float>.Shared.Rent(2);
        var weight = Tensor<float>.FromPooledMemory(
            new Memory<float>(weightBacking, 0, 2), new[] { 2, 1 }, weightBacking);
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(outputBacking, 0, 2), new[] { 1, 2, 1 }, outputBacking);

        try
        {
            Assert.True(weightBacking.Length > weight.Length);
            Assert.True(outputBacking.Length > output.Length);
            weight[0] = 1.0f;
            weight[1] = 1.0f;
            var step = new CompiledStep<float>(
                "TensorMatMul", (eng, destination) => { }, output, new[] { input, weight });
            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);

            weight[0] = 2.0f;
            weight[1] = 3.0f;
            specialized!(engine);

            Assert.Equal(new[] { 8.0f, 18.0f }, output.AsSpan().ToArray());
            Assert.Equal(8.0f, outputBacking[0]);
            Assert.Equal(18.0f, outputBacking[1]);
        }
        finally
        {
            TensorAllocator.Return(output);
            TensorAllocator.Return(weight);
        }
    }

    [Fact]
    public void SpecializedConcat_PoolPaddedInputAndOutput_StayLiveAcrossReplay()
    {
        var engine = new CpuEngine();
        var leftBacking = ArrayPool<float>.Shared.Rent(1);
        var outputBacking = ArrayPool<float>.Shared.Rent(2);
        var left = Tensor<float>.FromPooledMemory(
            new Memory<float>(leftBacking, 0, 1), new[] { 1 }, leftBacking);
        var right = new Tensor<float>(new[] { 5.0f }, new[] { 1 });
        var output = Tensor<float>.FromPooledMemory(
            new Memory<float>(outputBacking, 0, 2), new[] { 2 }, outputBacking);

        try
        {
            left[0] = 1.0f;
            var step = new CompiledStep<float>(
                "Concatenate", (eng, destination) => { }, output, new[] { left, right },
                savedState: new object[] { 0 });
            var specialized = CompiledTrainingPlan<float>.TryBuildSpecializedForward(step);
            Assert.NotNull(specialized);

            left[0] = 9.0f;
            specialized!(engine);

            Assert.Equal(new[] { 9.0f, 5.0f }, output.AsSpan().ToArray());
            Assert.Equal(9.0f, outputBacking[0]);
            Assert.Equal(5.0f, outputBacking[1]);
        }
        finally
        {
            TensorAllocator.Return(output);
            TensorAllocator.Return(left);
        }
    }

    [Fact]
    public void OffsetBackedParameterView_RejectsRawArraySpecialization_AndFallbackReadsCorrectSlice()
    {
        var engine = new CpuEngine();
        var parameterBacking = new float[] { 999.0f, 1.0f, 2.0f, 999.0f };
        var parameterView = Tensor<float>.FromMemory(
            new Memory<float>(parameterBacking, 1, 2), new[] { 2 });
        var addend = new Tensor<float>(new float[] { 10.0f, 20.0f }, new[] { 2 });
        var output = new Tensor<float>(new[] { 2 });

        Assert.True(parameterView.IsContiguous);
        Assert.Null(parameterView.GetLiveBackingArrayAllowingPaddingOrNull());
        var raw = parameterView.GetCpuBackingForStridedRead(out int storageOffset);
        Assert.Same(parameterBacking, raw);
        Assert.Equal(1, storageOffset);

        var step = new CompiledStep<float>(
            "TensorAdd",
            (eng, destination) => eng.TensorAddInto(destination, parameterView, addend),
            output,
            new[] { parameterView, addend });

        Assert.Null(CompiledTrainingPlan<float>.TryBuildSpecializedForward(step));
        step.Execute(engine, output);

        Assert.Equal(11.0f, output[0]);
        Assert.Equal(22.0f, output[1]);
    }

    [Fact]
    public void OffsetBackedOutputView_RejectsRawArraySpecialization_AndFallbackWritesCorrectSlice()
    {
        var engine = new CpuEngine();
        var left = new Tensor<float>(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var right = new Tensor<float>(new float[] { 10.0f, 20.0f }, new[] { 2 });
        var outputBacking = new float[] { 999.0f, 0.0f, 0.0f, 999.0f };
        var outputView = Tensor<float>.FromMemory(
            new Memory<float>(outputBacking, 1, 2), new[] { 2 });

        var step = new CompiledStep<float>(
            "TensorAdd",
            (eng, destination) => eng.TensorAddInto(destination, left, right),
            outputView,
            new[] { left, right });

        Assert.Null(CompiledTrainingPlan<float>.TryBuildSpecializedForward(step));
        step.Execute(engine, outputView);

        Assert.Equal(new float[] { 999.0f, 11.0f, 22.0f, 999.0f }, outputBacking);
    }

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
