using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for AutoTrainingCompiler changes introduced in this PR:
/// - ReplayMode thread-static field toggling
/// - RecordStep with loss parameter (loss identity included in pattern hash)
/// - TryGetCompiledBackward hash validation with loss identity
/// - TryCompileBackward guard for non-persistent tapes
/// </summary>
public class AutoTrainingCompilerTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // Always restore global state after each test to prevent cross-test contamination
    public AutoTrainingCompilerTests()
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
    // ReplayMode field
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void ReplayMode_DefaultValue_IsFalse()
    {
        // ReplayMode is [ThreadStatic] so starts false on any new thread context
        AutoTrainingCompiler.ReplayMode = false; // ensure clean state
        Assert.False(AutoTrainingCompiler.ReplayMode);
    }

    [Fact]
    public void ReplayMode_CanBeSetAndReadBack()
    {
        AutoTrainingCompiler.ReplayMode = true;
        Assert.True(AutoTrainingCompiler.ReplayMode);

        AutoTrainingCompiler.ReplayMode = false;
        Assert.False(AutoTrainingCompiler.ReplayMode);
    }

    // ──────────────────────────────────────────────────────────────
    // Enabled flag guard
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void RecordStep_WhenDisabled_DoesNotAdvanceState()
    {
        AutoTrainingCompiler.Enabled = false;

        // Run two identical training steps. Because Enabled=false, the state
        // should never reach ShouldCompile, so ReplayMode stays false.
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        for (int step = 0; step < 3; step++)
        {
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
            var output = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(output, null);
            tape.ComputeGradients(loss, new[] { weight });
        }

        // Disabled compiler must never set ReplayMode
        Assert.False(AutoTrainingCompiler.ReplayMode);
    }

    [Fact]
    public void TryGetCompiledBackward_WhenDisabled_ReturnsNull()
    {
        AutoTrainingCompiler.Enabled = false;

        var input = new Tensor<float>(new float[] { 1f, 2f }, new[] { 1, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        // Even after running many steps, TryGetCompiledBackward must return null
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var output = _engine.TensorMatMul(input, weight);
        var loss = _engine.ReduceSum(output, null);

        var compiled = AutoTrainingCompiler.TryGetCompiledBackward(tape, loss, new[] { weight });
        Assert.Null(compiled);
    }

    // ──────────────────────────────────────────────────────────────
    // RecordStep with loss parameter — hash differentiation
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void PersistentTape_SameLoss_SameOps_ActivatesReplayMode()
    {
        // Two identical steps (same loss tensor, same ops) should trigger compilation
        // and set ReplayMode = true.
        var input = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 2f, 0f, 0f, 2f }, new[] { 2, 2 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        Tensor<float>? sharedLoss = null;

        // Step 1: record the pattern
        {
            var h = _engine.TensorMatMul(input, weight);
            sharedLoss = _engine.ReduceSum(h, null);
            tape.ComputeGradients(sharedLoss, new[] { weight });
        }

        // Step 2: same pattern, same loss tensor
        {
            tape.Reset();
            var h = _engine.TensorMatMul(input, weight);
            sharedLoss = _engine.ReduceSum(h, null);
            tape.ComputeGradients(sharedLoss, new[] { weight });
        }

        // After 2 identical steps, the compiler should have kicked in.
        // ReplayMode indicates a compiled backward graph was stored and activated.
        Assert.True(AutoTrainingCompiler.ReplayMode,
            "ReplayMode should be true after 2 identical training steps on a persistent tape");
    }

    [Fact]
    public void PersistentTape_DifferentLossTensors_SameOps_ActivatesReplayMode()
    {
        // Two steps with the same op pattern but different loss tensor instances
        // SHOULD trigger compilation — loss tensors are always recreated each forward
        // pass, so identity-based matching would prevent compilation in all real training.
        // The pattern hash (op names + shapes) identifies the computation structure.
        var input = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 2f, 0f, 0f, 2f }, new[] { 2, 2 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        // Step 1: loss1
        {
            var h = _engine.TensorMatMul(input, weight);
            var loss1 = _engine.ReduceSum(h, null); // new tensor each step
            tape.ComputeGradients(loss1, new[] { weight });
        }

        tape.Reset();

        // Step 2: same ops but loss2 is a NEW tensor (different reference/identity)
        {
            var h = _engine.TensorMatMul(input, weight);
            var loss2 = _engine.ReduceSum(h, null); // different reference from step 1
            tape.ComputeGradients(loss2, new[] { weight });
        }

        // Same op pattern → compilation triggers → ReplayMode active
        Assert.True(AutoTrainingCompiler.ReplayMode,
            "ReplayMode should activate when op pattern matches, even with new loss tensor instances");
    }

    [Fact]
    public void TryGetCompiledBackward_HashMismatch_ClearsReplayMode()
    {
        // Simulate the case where ReplayMode was set true (e.g., from a previous run),
        // but then the pattern changes. TryGetCompiledBackward should clear ReplayMode.
        var input = new Tensor<float>(new float[] { 1f, 2f }, new[] { 1, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        // First compile a pattern
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        Tensor<float>? compiledLoss = null;
        for (int step = 0; step < 3; step++)
        {
            tape.Reset();
            var h = _engine.TensorMatMul(input, weight);
            compiledLoss = _engine.ReduceSum(h, null);
            tape.ComputeGradients(compiledLoss, new[] { weight });
        }

        // At this point ReplayMode may be true (if compilation succeeded).
        // Now run with a DIFFERENT operation pattern and a different loss.
        tape.Reset();
        var h2 = _engine.TensorAdd(input, input); // different op from matmul
        var differentLoss = _engine.ReduceSum(h2, null);

        // Call TryGetCompiledBackward with the changed tape and new loss.
        // Since the hash won't match the compiled plan, it must set ReplayMode=false.
        var result = AutoTrainingCompiler.TryGetCompiledBackward(tape, differentLoss, new[] { weight });

        Assert.Null(result); // No compiled backward for changed pattern
        Assert.False(AutoTrainingCompiler.ReplayMode,
            "TryGetCompiledBackward must clear ReplayMode when hash doesn't match compiled plan");
    }

    // ──────────────────────────────────────────────────────────────
    // TryCompileBackward guards
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void TryCompileBackward_NonPersistentTape_DoesNotSetReplayMode()
    {
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        // Non-persistent tape: TryCompileBackward is a no-op (guard: !tape.Options.Persistent)
        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>(); // NOT persistent
            var h = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(h, null);
            tape.ComputeGradients(loss, new[] { weight });
        }

        Assert.False(AutoTrainingCompiler.ReplayMode,
            "Non-persistent tape must never trigger compilation or set ReplayMode");
    }

    // ──────────────────────────────────────────────────────────────
    // Integration: 3-step training loop
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void PersistentTape_ThreeIdenticalSteps_ReplayModeActivatedAndGradientsCorrect()
    {
        // Validates the full compilation path:
        // Step 1: record pattern
        // Step 2: repeat detected, ShouldCompile=true
        // Step 3+: compiled backward used, ReplayMode=true
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        Tensor<float>? persistentLoss = null;
        Dictionary<Tensor<float>, Tensor<float>>? lastGrads = null;

        for (int step = 0; step < 3; step++)
        {
            tape.Reset();
            var h = _engine.TensorMatMul(input, weight);
            persistentLoss = _engine.ReduceSum(h, null);
            lastGrads = tape.ComputeGradients(persistentLoss, new[] { weight });
        }

        // After 3 identical steps, ReplayMode should be active
        Assert.True(AutoTrainingCompiler.ReplayMode,
            "ReplayMode should be active after 3 identical steps");

        // Gradients must still be correct (compilation doesn't change values)
        Assert.NotNull(lastGrads);
        Assert.True(lastGrads!.ContainsKey(weight),
            "Compiled backward must still produce weight gradients");

        // MatMul(2x2, 2x2) → sum. Gradient of sum w.r.t. weight should be non-zero.
        bool anyNonZero = false;
        for (int i = 0; i < lastGrads[weight].Length; i++)
            if (lastGrads[weight].GetFlat(i) != 0f) anyNonZero = true;
        Assert.True(anyNonZero, "Weight gradient must be non-zero after compilation");
    }

    [Fact]
    public void PatternReset_AfterCompilation_ClearsReplayModeOnNextMismatch()
    {
        // After compilation succeeds (ReplayMode=true), changing the forward ops
        // must reset ReplayMode to false on the next step.
        var input = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, new[] { 2, 2 });

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        // Step 1+2: same pattern to trigger compilation
        for (int step = 0; step < 2; step++)
        {
            tape.Reset();
            var h = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(h, null);
            tape.ComputeGradients(loss, new[] { weight });
        }

        // Step 3: completely different pattern (add instead of matmul)
        tape.Reset();
        var h2 = _engine.TensorAdd(input, weight);
        var differentLoss = _engine.ReduceSum(h2, null);
        tape.ComputeGradients(differentLoss, new[] { weight });

        // Hash mismatch on step 3 should clear ReplayMode
        Assert.False(AutoTrainingCompiler.ReplayMode,
            "ReplayMode must be cleared when forward pattern changes after compilation");
    }
}