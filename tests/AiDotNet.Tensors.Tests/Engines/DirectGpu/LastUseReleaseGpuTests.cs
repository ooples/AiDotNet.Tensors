using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On a tape that opts in (GradientTapeOptions.ReleaseDeviceActivationsAtLastUse), device intermediates are freed at
/// their last backward use (BackwardStorageReleasePlan), a graph's first backward run probes which of them code reads
/// afterwards (AutoRetainProbe), and a read the probe could not see throws an error that names the producing op and is
/// itself recorded. A default tape never invalidates a tensor its caller holds.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LastUseReleaseGpuTests
{
    // Odd shapes: the probe's keys are structural, so they must not collide with another test's graph.
    private static readonly int[] Shape = { 3, 7, 5 };

    private static Tensor<float> Input(int seed) =>
        new(Enumerable.Range(0, 105).Select(i => (float)Math.Sin(i * 0.37 + seed)).ToArray(), Shape);

    private static float[] Sigmoid(float[] x) => x.Select(v => 1f / (1f + (float)Math.Exp(-v))).ToArray();

    private static readonly GradientTapeOptions OptIn = new() { ReleaseDeviceActivationsAtLastUse = true };

    // y = sigmoid(x); loss = sum(y * w): y is an intermediate whose last backward use is the multiply's backward.
    // afterBackward runs once backward has finished, while the tape is still open: a tape's dispose frees the step's
    // intermediates anyway (Retain keeps one past it), so reads that test the last-use release happen before it.
    private static Tensor<float> Step(IEngine engine, Tensor<float> x, Tensor<float> w, bool readAfterBackward,
        out float[]? read, bool optIn = true, Action<Tensor<float>>? afterBackward = null)
    {
        using var tape = new GradientTape<float>(optIn ? OptIn : null);
        var y = engine.Sigmoid(x);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);
        tape.ComputeGradients(loss, new[] { x });
        read = readAfterBackward ? y.ToArray() : null;
        afterBackward?.Invoke(y);
        return y;
    }

    [SkippableFact]
    public void AnIntermediateReadAfterBackward_OnTheGraphsFirstRun_IsKeptOnEveryLaterRun()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(1);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(2), GpuTensorRole.General);
        var expected = Sigmoid(host.ToArray());

        // The first run is the probe: y is quarantined, not freed, so the read succeeds and marks y's producer.
        for (int run = 0; run < 3; run++)
        {
            Step(gpu, x, w, readAfterBackward: true, out var read);
            Assert.NotNull(read);
            if (read is not null)
                Assert.Equal(expected, read, new FloatTolerance(1e-5f));
        }
        Assert.True(AutoRetainProbe.RetainedCount >= 1, "the probe did not retain the intermediate it saw read");
    }

    [SkippableFact]
    public void AnIntermediateNothingReads_IsFreedAfterTheProbe_AndALateReadNamesItsProducer()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(3);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(4), GpuTensorRole.General);

        Step(gpu, x, w, readAfterBackward: false, out _);   // probe run: nothing is read
        var y = Step(gpu, x, w, readAfterBackward: false, out _);   // freed at its last backward use

        var error = Assert.Throws<InvalidOperationException>(() => y.ToArray());
        Assert.Contains("[3, 7, 5]", error.Message);
        Assert.Contains("GradientTape.Retain", error.Message);
        Assert.Contains("forward op", error.Message);

        // The failed read was recorded: the next run keeps y.
        Step(gpu, x, w, readAfterBackward: true, out var read);
        Assert.NotNull(read);
        if (read is not null)
            Assert.Equal(Sigmoid(host.ToArray()), read, new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void Retain_KeepsAnIntermediate_WithoutAProbe()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(5);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(6), GpuTensorRole.General);
        Step(gpu, x, w, readAfterBackward: false, out _);   // spend the probe

        using var tape = new GradientTape<float>(OptIn);
        var y = gpu.Sigmoid(x);
        tape.Retain(y);
        var loss = gpu.ReduceSum(gpu.TensorMultiply(y, w), null);
        tape.ComputeGradients(loss, new[] { x });
        Assert.Equal(Sigmoid(host.ToArray()), y.ToArray(), new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void AReshapeViewOfALeaf_NeverFreesTheLeafsStorage()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(7);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        for (int run = 0; run < 3; run++)   // past the probe run, where releases are real
        {
            using var tape = new GradientTape<float>(OptIn);
            var v = gpu.Reshape(x, new[] { 21, 5 });   // a recorded op whose output aliases the leaf's storage
            var loss = gpu.ReduceSum(gpu.TensorMultiply(v, v), null);
            var grads = tape.ComputeGradients(loss, new[] { x });
            Assert.Equal(host.ToArray().Select(a => 2f * a).ToArray(), grads[x].ToArray(), new FloatTolerance(1e-5f));
        }
        Assert.Equal(host.ToArray(), x.ToArray());
    }

    // loss = sum(tanh(x) * w) on another shape: a different graph from Step's.
    private static void OtherGraphStep(IEngine engine, Tensor<float> x, Tensor<float> w, bool optIn)
    {
        using var tape = new GradientTape<float>(optIn ? OptIn : null);
        var loss = engine.ReduceSum(engine.TensorMultiply(engine.Tanh(x), w), null);
        tape.ComputeGradients(loss, new[] { x });
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnOutputHeldAcrossAnotherGraphsBackward_StaysReadable(bool optIn)
    {
        // Keep one model's output, run a different model's backward, then compare: the first output must still be
        // readable (PyTorch keeps it while it is referenced). A probe window used to close at ANY backward run.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(9);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(10), GpuTensorRole.General);
        var x2 = gpu.UploadToGpu(new Tensor<float>(Enumerable.Range(0, 44).Select(i => (float)Math.Cos(i * 0.3)).ToArray(), new[] { 4, 11 }), GpuTensorRole.General);
        var w2 = gpu.UploadToGpu(new Tensor<float>(Enumerable.Range(0, 44).Select(i => (float)Math.Sin(i * 0.2)).ToArray(), new[] { 4, 11 }), GpuTensorRole.General);

        float[]? held = null;
        Step(gpu, x, w, readAfterBackward: false, out _, optIn, afterBackward: y =>
        {
            OtherGraphStep(gpu, x2, w2, optIn);
            OtherGraphStep(gpu, x2, w2, optIn);
            held = y.ToArray();
        });
        Assert.NotNull(held);
        if (held is not null) Assert.Equal(Sigmoid(host.ToArray()), held, new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void ADefaultTape_NeverFreesAnIntermediateItsCallerHolds()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(11);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(12), GpuTensorRole.General);
        Step(gpu, x, w, readAfterBackward: false, out _, optIn: false);
        float[]? held = null;
        // Past where an opted-in tape frees y, and across a later step of the same graph.
        Step(gpu, x, w, readAfterBackward: false, out _, optIn: false, afterBackward: y =>
        {
            Step(gpu, x, w, readAfterBackward: false, out _, optIn: false);
            held = y.ToArray();
        });
        Assert.NotNull(held);
        if (held is not null) Assert.Equal(Sigmoid(host.ToArray()), held, new FloatTolerance(1e-5f));
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);   // and a default tape never probes
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_tapes_dispose_frees_an_unretained_intermediate_and_Retain_keeps_one(bool retain)
    {
        // The step-end contract the last-use tests read around: dispose frees what the step produced, as PyTorch frees
        // an unreferenced CUDA tensor, unless Retain kept it.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(17);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(18), GpuTensorRole.General);
        Tensor<float> y;
        using (var tape = new GradientTape<float>())
        {
            y = gpu.Sigmoid(x);
            if (retain) tape.Retain(y);
            tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(y, w), null), new[] { x });
        }
        if (retain) Assert.Equal(Sigmoid(host.ToArray()), y.ToArray(), new FloatTolerance(1e-5f));
        else Assert.Throws<InvalidOperationException>(() => y.ToArray());
    }

    [SkippableFact]
    public void A_retained_graph_runs_backward_twice_with_the_same_gradients()
    {
        // StreamingGraphRetention = RetainUntilTapeDisposal keeps the graph for another backward. With the opt-in, the
        // second walk read activations the first walk had freed at their last use.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var x = gpu.UploadToGpu(Input(19), GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(20), GpuTensorRole.General);
        var options = new GradientTapeOptions
        {
            Persistent = true,
            ReleaseDeviceActivationsAtLastUse = true,
            StreamingGraphRetention = StreamingGraphRetentionMode.RetainUntilTapeDisposal,
        };
        for (int run = 0; run < 3; run++)   // the probe run and two after it
        {
            using var tape = new GradientTape<float>(options);
            var y = gpu.Sigmoid(x);
            var loss = gpu.ReduceSum(gpu.TensorMultiply(y, w), null);
            var first = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
            var second = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
            Assert.Equal(first, second, new FloatTolerance(1e-6f));
        }
    }

    [SkippableFact]
    public void A_second_backward_after_freeing_at_last_use_throws_a_clear_error()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var x = gpu.UploadToGpu(Input(21), GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(22), GpuTensorRole.General);
        for (int run = 0; run < 2; run++)   // on the probe run (quarantined) and after it (freed)
        {
            using var tape = new GradientTape<float>(
                new GradientTapeOptions { Persistent = true, ReleaseDeviceActivationsAtLastUse = true });
            var y = gpu.Sigmoid(x);
            var loss = gpu.ReduceSum(gpu.TensorMultiply(y, w), null);
            tape.ComputeGradients(loss, new[] { x });
            var error = Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { x }));
            Assert.Contains("RetainUntilTapeDisposal", error.Message);
        }
    }

    [SkippableTheory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void A_non_leaf_gradient_outlives_the_tape_unless_an_opted_in_tape_was_not_told_to_keep_it(bool optIn, bool retain)
    {
        // A backward with no sources hands back non-leaf gradients. An opted-in tape frees them at dispose unless Retain
        // kept one; a default tape never invalidates what its caller holds, so it keeps them (it freed them too).
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var hostW = Input(24);
        var x = gpu.UploadToGpu(Input(23), GpuTensorRole.General);
        var w = gpu.UploadToGpu(hostW, GpuTensorRole.General);
        Tensor<float>? gradY = null;
        using (var tape = new GradientTape<float>(optIn ? OptIn : null))
        {
            var y = gpu.Sigmoid(x);
            var grads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(y, w), null));
            if (grads.TryGetValue(y, out var g)) gradY = g;
            if (retain && gradY is not null) tape.Retain(gradY);
        }
        Assert.NotNull(gradY);
        if (gradY is null) return;
        if (optIn && !retain) Assert.Throws<InvalidOperationException>(() => gradY.ToArray());
        else Assert.Equal(hostW.ToArray(), gradY.ToArray(), new FloatTolerance(1e-6f));   // d/dy = w
    }

    [SkippableFact]
    public void A_backward_that_throws_after_freeing_leaves_the_tape_consumed()
    {
        // A step after the first free threw, so the consumed flag (set after a finished walk) was never set: a retry
        // walked the graph again over the freed intermediate.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var x = gpu.UploadToGpu(Input(33), GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(34), GpuTensorRole.General);
        bool throwNow = false;
        BackwardFunction<float> throwing = (gradOutput, inputs, output, saved, engine, accumulator) =>
        {
            if (throwNow) throw new NotSupportedException("injected backward failure");
        };
        for (int run = 0; run < 2; run++)   // the probe run, then one that frees
        {
            throwNow = run == 1;
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true, ReleaseDeviceActivationsAtLastUse = true });
            // Recorded first, so its backward runs last: after the sigmoid's backward freed y.
            var side = new Tensor<float>(new float[Shape[0] * Shape[1] * Shape[2]], Shape);
            DifferentiableOps.RecordUnary("InjectedFailure", side, x, throwing);
            var y = gpu.Sigmoid(x);
            var loss = gpu.TensorAdd(gpu.ReduceSum(gpu.TensorMultiply(y, w), null), gpu.ReduceSum(side, null));
            if (!throwNow)
            {
                tape.ComputeGradients(loss, new[] { x });
                continue;
            }
            Assert.Throws<NotSupportedException>(() => tape.ComputeGradients(loss, new[] { x }));
            var retry = Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { x }));
            Assert.Contains("RetainUntilTapeDisposal", retry.Message);
        }
    }

    [SkippableFact]
    public void A_reset_tape_records_and_runs_a_fresh_graph_after_freeing_the_last_one()
    {
        // Reset drops the consumed graph, but the consumed flag stayed set: the fresh graph's first backward threw.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var hostX = Input(35);
        var hostW = Input(36);
        var x = gpu.UploadToGpu(hostX, GpuTensorRole.General);
        var w = gpu.UploadToGpu(hostW, GpuTensorRole.General);
        var s = Sigmoid(hostX.ToArray());
        var wArr = hostW.ToArray();
        var expected = s.Select((si, i) => wArr[i] * si * (1f - si)).ToArray();   // d sum(w * sigmoid(x)) / dx
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true, ReleaseDeviceActivationsAtLastUse = true });
        for (int graph = 0; graph < 3; graph++)
        {
            if (graph > 0) tape.Reset();
            var y = gpu.Sigmoid(x);
            var grads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(y, w), null), new[] { x });
            Assert.Equal(expected, grads[x].ToArray(), new FloatTolerance(1e-5f));
        }
    }

    [SkippableFact]
    public void A_reduction_read_on_the_probe_run_keeps_the_intermediate_on_later_runs()
    {
        // ReduceSum on the innermost axis reads its input through UploadTensor, whose resident return skipped the probe:
        // the producer stayed unretained and the next run freed what this read still needed.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = new Tensor<float>(Enumerable.Range(0, 4 * 13).Select(i => (float)Math.Sin(i * 0.41)).ToArray(), new[] { 4, 13 });
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(new Tensor<float>(
            Enumerable.Range(0, 4 * 13).Select(i => (float)Math.Cos(i * 0.23)).ToArray(), new[] { 4, 13 }), GpuTensorRole.General);
        var s = Sigmoid(host.ToArray());
        var expected = Enumerable.Range(0, 4).Select(r => Enumerable.Range(0, 13).Sum(c => s[r * 13 + c])).ToArray();
        for (int run = 0; run < 3; run++)
        {
            float[]? rowSums = null;
            Step(gpu, x, w, readAfterBackward: false, out _, afterBackward: y => rowSums = gpu.ReduceSum(y, new[] { 1 }).ToArray());
            Assert.NotNull(rowSums);
            if (rowSums is not null) Assert.Equal(expected, rowSums, new FloatTolerance(1e-5f));
        }
    }

    [SkippableFact]
    public void Graphs_whose_shapes_hashed_alike_each_get_their_own_probe()
    {
        // [1, 32] and [2, 1] folded to one 31-based shape hash, so the second graph's first run counted as already
        // probed: it freed y at its last use, and a read after backward threw instead of being recorded.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        Tensor<float> Make(int[] shape, double phase) => gpu.UploadToGpu(new Tensor<float>(
            Enumerable.Range(0, shape[0] * shape[1]).Select(i => (float)Math.Sin(i * 0.3 + phase)).ToArray(), shape),
            GpuTensorRole.General);
        var xa = Make(new[] { 1, 32 }, 0);
        var wa = Make(new[] { 1, 32 }, 1);
        var xb = Make(new[] { 2, 1 }, 2);
        var wb = Make(new[] { 2, 1 }, 3);
        Step(gpu, xa, wa, readAfterBackward: false, out _);   // spend graph A's probe without a read
        Step(gpu, xa, wa, readAfterBackward: false, out _);
        // Graph B's first run is its own probe: the read after backward succeeds.
        Step(gpu, xb, wb, readAfterBackward: true, out var read);
        Assert.NotNull(read);
        if (read is not null) Assert.Equal(Sigmoid(xb.ToArray()), read, new FloatTolerance(1e-5f));
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_quarantined_intermediate_kept_after_backward_survives_the_graphs_next_run(bool overwrite)
    {
        // On the probe run y is quarantined, not read. The caller then keeps it past the tape (Retain after backward),
        // optionally writing a new value into it first (IEngine.TensorCopy reports no read). The graph's next run closed
        // the window and the old entry freed y anyway: the kept tensor, or the caller's new value in it.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(27);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(28), GpuTensorRole.General);
        var fresh = gpu.UploadToGpu(Input(29), GpuTensorRole.General);
        Tensor<float> kept;
        using (var tape = new GradientTape<float>(OptIn))
        {
            kept = gpu.Sigmoid(x);
            tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(kept, w), null), new[] { x });
            if (overwrite) gpu.TensorCopy(fresh, kept);
            tape.Retain(kept);
        }
        Step(gpu, x, w, readAfterBackward: false, out _);   // the same graph's next run closes the probe window
        var expected = overwrite ? Input(29).ToArray() : Sigmoid(host.ToArray());
        Assert.Equal(expected, kept.ToArray(), new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void A_quarantined_intermediate_overwritten_after_backward_keeps_the_new_value_through_the_next_run()
    {
        // The write reports no read, so the entry stayed armed; the graph's next run (inside this tape's life) released
        // the tensor that now holds the caller's value.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var x = gpu.UploadToGpu(Input(30), GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(31), GpuTensorRole.General);
        var fresh = gpu.UploadToGpu(Input(32), GpuTensorRole.General);
        float[]? read = null;
        Step(gpu, x, w, readAfterBackward: false, out _, afterBackward: y =>
        {
            gpu.TensorCopy(fresh, y);
            Step(gpu, x, w, readAfterBackward: false, out _);   // closes the probe window
            read = y.ToArray();
        });
        Assert.NotNull(read);
        if (read is not null) Assert.Equal(Input(32).ToArray(), read, new FloatTolerance(1e-6f));
    }

    [SkippableFact]
    public void A_checkpoint_segments_parameter_gradient_survives_its_step_scope()
    {
        // A checkpoint segment's backward writes the gradient of w, which is not among that step's inputs. The step
        // scope kept only the inputs' gradients and freed the rest, leaving a released buffer in grads[w].
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var hostX = Input(25);
        var hostW = Input(26);
        (float[] Dx, float[] Dw) Run(IEngine engine, Tensor<float> x, Tensor<float> w)
        {
            using var tape = new GradientTape<float>(engine.SupportsGpu ? OptIn : null);
            var stem = engine.Sigmoid(x);
            var segmentOut = GradientCheckpointing<float>.CheckpointSegments(
                new Func<Tensor<float>, Tensor<float>>[] { t => engine.TensorMultiply(t, w), t => engine.Sigmoid(t) },
                stem,
                segmentParameters: new System.Collections.Generic.IReadOnlyList<Tensor<float>>[] { new[] { w } },
                segmentSize: 2);
            var loss = engine.ReduceSum(engine.TensorMultiply(segmentOut, w), null);
            var grads = tape.ComputeGradients(loss, new[] { x, w });
            return (grads[x].ToArray(), grads[w].ToArray());
        }
        var truth = Run(new CpuEngine(), hostX, hostW);
        var gx = gpu.UploadToGpu(hostX, GpuTensorRole.General);
        var gw = gpu.UploadToGpu(hostW, GpuTensorRole.General);
        for (int run = 0; run < 3; run++)   // the probe run and two that free at last use
        {
            var got = Run(gpu, gx, gw);
            Assert.Equal(truth.Dw, got.Dw, new FloatTolerance(1e-4f));
            Assert.Equal(truth.Dx, got.Dx, new FloatTolerance(1e-4f));
        }
    }

    [SkippableFact]
    public void CreateGraph_True_KeepsSavedActivationsForHigherOrderBackward()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var hostX = Input(13);
        var hostW = Input(14);
        var s = Sigmoid(hostX.ToArray());
        var wArr = hostW.ToArray();
        // d²/dx² [w * sigmoid(x)] = w * s * (1 - s) * (1 - 2s)
        var expectedSecond = s.Select((si, idx) => wArr[idx] * si * (1f - si) * (1f - 2f * si)).ToArray();

        var x = gpu.UploadToGpu(hostX, GpuTensorRole.General);
        var w = gpu.UploadToGpu(hostW, GpuTensorRole.General);
        for (int run = 0; run < 3; run++)
        {
            using var outerTape = new GradientTape<float>(
                new GradientTapeOptions { Persistent = true, ReleaseDeviceActivationsAtLastUse = true });
            var y = gpu.Sigmoid(x);
            var loss = gpu.ReduceSum(gpu.TensorMultiply(y, w), null);
            var first = outerTape.ComputeGradients(loss, new[] { x }, createGraph: true)[x];
            var gradSum = gpu.ReduceSum(first, null);
            var second = outerTape.ComputeGradients(gradSum, new[] { x })[x];
            Assert.Equal(expectedSecond, second.ToArray(), new FloatTolerance(1e-4f));
        }
    }

    [SkippableFact]
    public void GradientCheckpointing_InsideOptedInOuterTape_PreservesOuterProbeAndRetention()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var hostX = Input(15);
        var hostW = Input(16);
        var x = gpu.UploadToGpu(hostX, GpuTensorRole.General);
        var w = gpu.UploadToGpu(hostW, GpuTensorRole.General);
        var expectedStem = Sigmoid(hostX.ToArray());

        for (int run = 0; run < 3; run++)
        {
            using var tape = new GradientTape<float>(OptIn);
            var stem = gpu.Sigmoid(x);
            var ckptOut = GradientCheckpointing<float>.CheckpointSegments(
                new Func<Tensor<float>, Tensor<float>>[]
                {
                    t => gpu.TensorMultiply(t, w),
                    t => gpu.Sigmoid(t)
                },
                stem,
                segmentParameters: new System.Collections.Generic.IReadOnlyList<Tensor<float>>[]
                {
                    new[] { w }
                },
                segmentSize: 2);
            var head = gpu.TensorMultiply(ckptOut, w);
            var loss = gpu.ReduceSum(head, null);
            tape.ComputeGradients(loss, new[] { x, w });
            Assert.Equal(expectedStem, stem.ToArray(), new FloatTolerance(1e-5f));
        }
        Assert.True(AutoRetainProbe.RetainedCount >= 1, "outer probe retention was lost across nested checkpoint backward");
    }

    private sealed class FloatTolerance : System.Collections.Generic.IEqualityComparer<float>
    {
        private readonly float _tolerance;
        public FloatTolerance(float tolerance) => _tolerance = tolerance;
        public bool Equals(float a, float b) => Math.Abs(a - b) <= _tolerance * Math.Max(1f, Math.Abs(a));
        public int GetHashCode(float value) => 0;
    }
}
