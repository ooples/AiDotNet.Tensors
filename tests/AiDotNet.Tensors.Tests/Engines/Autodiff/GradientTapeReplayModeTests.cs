using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for the _savedReplayMode field added to GradientTape in this PR.
///
/// The fix ensures that when a nested (inner) tape is disposed, it restores the
/// ReplayMode value that was active when it was created, rather than hard-resetting
/// it to false. This is critical for persistent outer tapes that have auto-compiled
/// a backward graph: an inner tape (e.g., used for a sub-computation) must not
/// silently disable the outer tape's replay optimization.
/// </summary>
public class GradientTapeReplayModeTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public GradientTapeReplayModeTests()
    {
        AutoTrainingCompiler.ResetState();
        AutoTrainingCompiler.Enabled = true;
    }

    public void Dispose()
    {
        AutoTrainingCompiler.ResetState();
        AutoTrainingCompiler.Enabled = true;
    }

    // ──────────────────────────────────────────────────────────────
    // _savedReplayMode: basic save/restore
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void GradientTape_Constructor_SavesCurrentReplayModeAsFalse()
    {
        // When no replay is active before tape creation, Dispose must restore false.
        AutoTrainingCompiler.ReplayMode = false;

        using (var tape = new GradientTape<float>())
        {
            // Change ReplayMode while tape is alive
            AutoTrainingCompiler.ReplayMode = true;
        }

        // Tape must have restored the pre-construction value (false)
        Assert.False(AutoTrainingCompiler.ReplayMode,
            "Dispose must restore ReplayMode to the value saved at construction (false)");
    }

    [Fact]
    public void GradientTape_Constructor_SavesCurrentReplayModeAsTrue()
    {
        // When replay is active before tape creation, Dispose must restore true.
        AutoTrainingCompiler.ReplayMode = true;

        using (var tape = new GradientTape<float>())
        {
            // Tape constructor should save true; even if something clears it inside,
            // Dispose must restore true.
            AutoTrainingCompiler.ReplayMode = false;
        }

        // Tape must have restored the pre-construction value (true)
        Assert.True(AutoTrainingCompiler.ReplayMode,
            "Dispose must restore ReplayMode to the value saved at construction (true)");
    }

    // ──────────────────────────────────────────────────────────────
    // Nested tapes: inner dispose must not clobber outer replay state
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void NestedTape_Dispose_DoesNotClearOuterReplayMode()
    {
        // Regression: before this PR fix, inner tape Dispose() hard-reset ReplayMode=false,
        // destroying the outer tape's compiled backward optimization.
        AutoTrainingCompiler.ReplayMode = true; // outer tape's compilation is active

        var outerTape = new GradientTape<float>();

        try
        {
            // Inner tape created while ReplayMode=true (set before outer tape)
            using (var innerTape = new GradientTape<float>())
            {
                // Inner tape should save the current ReplayMode (true from outer context)
                // Do some work that might change ReplayMode
                AutoTrainingCompiler.ReplayMode = false;
            }
            // Inner tape disposed: must restore what it saved (true)
            Assert.True(AutoTrainingCompiler.ReplayMode,
                "Inner tape dispose must restore ReplayMode to true (the value when inner tape was created)");
        }
        finally
        {
            outerTape.Dispose();
        }

        // After outer tape disposes: restores its own saved value (true, set before outer tape)
        Assert.True(AutoTrainingCompiler.ReplayMode,
            "Outer tape dispose must also restore ReplayMode to true");
    }

    [Fact]
    public void NestedTape_Dispose_RestoredCorrectly_WhenOuterHadFalse()
    {
        // Symmetric test: outer context has ReplayMode=false.
        // Inner tape must not leave ReplayMode in a different state.
        AutoTrainingCompiler.ReplayMode = false;

        var outerTape = new GradientTape<float>();
        try
        {
            using (var innerTape = new GradientTape<float>())
            {
                // Inner tape was created with ReplayMode=false, so saves false
                AutoTrainingCompiler.ReplayMode = true; // changes while inner alive
            }
            // Inner disposed: restores false (what it saved)
            Assert.False(AutoTrainingCompiler.ReplayMode,
                "Inner tape must restore ReplayMode=false (the value at inner tape creation)");
        }
        finally
        {
            outerTape.Dispose();
        }

        Assert.False(AutoTrainingCompiler.ReplayMode,
            "Outer tape must restore ReplayMode=false after dispose");
    }

    [Fact]
    public void ThreeLevelNestedTapes_ReplayModeRestoredAtEachLevel()
    {
        // Validate 3 levels of nesting restore correctly.
        // Level ordering:
        //   Before outermost: ReplayMode=true
        //   Outermost creates: saves true
        //   Middle creates:    saves true (current is still true)
        //   Inner creates:     saves true
        //   Inner disposes:    restores true
        //   Middle disposes:   restores true
        //   Outermost disposes: restores true (initial)

        AutoTrainingCompiler.ReplayMode = true;

        var outermost = new GradientTape<float>();
        try
        {
            var middle = new GradientTape<float>();
            try
            {
                using (var inner = new GradientTape<float>())
                {
                    AutoTrainingCompiler.ReplayMode = false; // cleared inside inner
                }
                // Inner disposed: restored true
                Assert.True(AutoTrainingCompiler.ReplayMode, "After inner dispose: must be true");

                AutoTrainingCompiler.ReplayMode = false; // cleared again at middle level
            }
            finally
            {
                middle.Dispose();
            }
            // Middle disposed: restored true
            Assert.True(AutoTrainingCompiler.ReplayMode, "After middle dispose: must be true");
        }
        finally
        {
            outermost.Dispose();
        }

        // Outermost disposed: restored true
        Assert.True(AutoTrainingCompiler.ReplayMode, "After outermost dispose: must be true");
    }

    [Fact]
    public void TapeDispose_Idempotent_DoesNotDoubleRestoreReplayMode()
    {
        // Calling Dispose() twice must not restore ReplayMode a second time.
        AutoTrainingCompiler.ReplayMode = false;

        var tape = new GradientTape<float>();
        AutoTrainingCompiler.ReplayMode = true; // change after construction

        tape.Dispose(); // First dispose: restores false
        Assert.False(AutoTrainingCompiler.ReplayMode, "First dispose must restore false");

        AutoTrainingCompiler.ReplayMode = true; // change again
        tape.Dispose(); // Second dispose: no-op (already disposed)
        Assert.True(AutoTrainingCompiler.ReplayMode, "Second dispose must be idempotent (no restore)");
    }

    // ──────────────────────────────────────────────────────────────
    // Integration: nested tape inside a compiled outer training loop
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void CompiledOuterTape_NestedTapeForSubComputation_PreservesReplayMode()
    {
        // Simulates a real use case: an outer persistent tape has compiled a backward,
        // and during the next training step an inner tape is created transiently for
        // a sub-computation (e.g., gradient penalty, validation loss). The inner tape
        // must not disable the outer tape's compiled replay optimization.
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        using var outerTape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        // Warm up outer tape to trigger auto-compilation (needs 2+ identical steps)
        for (int warmup = 0; warmup < 3; warmup++)
        {
            outerTape.Reset();
            var h = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(h, null);
            outerTape.ComputeGradients(loss, new[] { weight });
        }

        bool compiledBefore = AutoTrainingCompiler.ReplayMode;

        // Now create an inner tape for a sub-computation (e.g., validation pass)
        using (var innerTape = new GradientTape<float>())
        {
            var valOut = _engine.TensorAdd(input, weight);
            var valLoss = _engine.ReduceSum(valOut, null);
            innerTape.ComputeGradients(valLoss, new[] { weight });
        }

        // After inner tape is disposed, outer tape's ReplayMode must be preserved
        Assert.True(AutoTrainingCompiler.ReplayMode == compiledBefore,
            $"Inner tape dispose must not change outer tape's ReplayMode " +
            $"(was {compiledBefore}, now {AutoTrainingCompiler.ReplayMode})");
    }

    // ──────────────────────────────────────────────────────────────
    // createGraph compatibility: forward ops must still record GradFn
    // even after auto-compilation has triggered
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void CreateGraph_WorksAfterAutoCompilation()
    {
        // After auto-compilation triggers, forward ops must still record to tape
        // and set GradFn on outputs. Without this, createGraph:true would fail
        // because the tape would be empty and GradFn would be null.
        var x = new Tensor<float>(new float[] { 2f, 3f }, new[] { 2 });
        var w = new Tensor<float>(new float[] { 1f, 1f }, new[] { 2 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        // Warm up: 3 identical steps to trigger auto-compilation
        for (int i = 0; i < 3; i++)
        {
            tape.Reset();
            var y = _engine.TensorMultiply(x, w);
            var loss = _engine.ReduceSum(y, null);
            tape.ComputeGradients(loss, new[] { w });
        }

        // Now do a forward pass and use createGraph:true
        tape.Reset();
        var y2 = _engine.TensorMultiply(x, w);
        var loss2 = _engine.ReduceSum(y2, null);

        // This must NOT throw — the tape must have recorded ops and set GradFn
        // even though auto-compilation may be active
        var grads = tape.ComputeGradients(loss2, new[] { w }, createGraph: true);

        Assert.NotNull(grads);
        Assert.True(grads.Count > 0, "createGraph:true should produce gradients even after auto-compilation");
    }

    [Fact]
    public void ForwardPass_RecordsTapeEntries_EvenAfterCompilation()
    {
        // Verifies that DifferentiableOps.RecordIfActive always records (no replay mode skip).
        // This is the safety guarantee that makes createGraph:true possible.
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f }, new[] { 3 });
        var b = new Tensor<float>(new float[] { 4f, 5f, 6f }, new[] { 3 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        // Warm up to trigger compilation
        for (int i = 0; i < 3; i++)
        {
            tape.Reset();
            var r = _engine.TensorAdd(a, b);
            var l = _engine.ReduceSum(r, null);
            tape.ComputeGradients(l, new[] { a });
        }

        // After compilation, verify tape still records on next forward pass
        tape.Reset();
        var result = _engine.TensorAdd(a, b);
        var loss = _engine.ReduceSum(result, null);

        // Tape should have entries from the forward pass
        Assert.True(tape.EntryCount > 0,
            "Tape must record forward ops even after auto-compilation (needed for createGraph support)");

        tape.ComputeGradients(loss, new[] { a });
    }
}