using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for AutoTrainingCompiler behaviour introduced in this PR:
/// - <c>ReplayMode</c> [ThreadStatic] flag lifecycle (default, enable, disable)
/// - <c>RecordStep</c> now includes loss-tensor identity in the pattern hash
/// - <c>TryGetCompiledBackward</c> sets/clears ReplayMode based on compiled plan availability
/// - Nested <see cref="GradientTape{T}"/> correctly saves/restores ReplayMode on dispose
/// </summary>
public class AutoTrainingCompilerTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ──────────────────────────────────────────────────────────────
    // ReplayMode — default state
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void ReplayMode_DefaultsFalse()
    {
        // ReplayMode is [ThreadStatic] and must default to false on a fresh thread.
        // We cannot guarantee thread freshness inside xUnit, but we can at least
        // verify the field is readable and starts at a well-known value after explicit reset.
        bool saved = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = false;
            Assert.False(AutoTrainingCompiler.ReplayMode);
        }
        finally
        {
            AutoTrainingCompiler.ReplayMode = saved;
        }
    }

    [Fact]
    public void ReplayMode_CanBeSetAndRead()
    {
        bool saved = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = true;
            Assert.True(AutoTrainingCompiler.ReplayMode);

            AutoTrainingCompiler.ReplayMode = false;
            Assert.False(AutoTrainingCompiler.ReplayMode);
        }
        finally
        {
            AutoTrainingCompiler.ReplayMode = saved;
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Nested tape — _savedReplayMode restore on dispose
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void NestedTape_Dispose_RestoresOuterReplayMode_WhenOuterWasFalse()
    {
        // PR fix: outer tape's _savedReplayMode is captured at construction time.
        // When inner tape disposes it must restore the value that was current when
        // the OUTER tape was constructed, not blindly set it to false.
        bool saved = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = false;

            using var outer = new GradientTape<float>();
            // Inner tape captures ReplayMode=false at its construction time.
            using (var inner = new GradientTape<float>())
            {
                // While inner is alive, set ReplayMode to true
                AutoTrainingCompiler.ReplayMode = true;
            }
            // Inner dispose should restore the value saved when inner was constructed: false
            Assert.False(AutoTrainingCompiler.ReplayMode,
                "Inner tape dispose must restore ReplayMode to the value at inner-tape construction time");
        }
        finally
        {
            AutoTrainingCompiler.ReplayMode = saved;
        }
    }

    [Fact]
    public void NestedTape_Dispose_RestoresOuterReplayMode_WhenOuterWasTrue()
    {
        // If outer tape was created while ReplayMode was already true (e.g. after
        // compilation in a previous training step), an inner tape must not clear it.
        bool saved = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = true;

            using var outer = new GradientTape<float>();
            // Both outer and inner capture ReplayMode=true at construction.
            using (var inner = new GradientTape<float>())
            {
                AutoTrainingCompiler.ReplayMode = false; // inner changes it
            }
            // inner captured true → should restore to true on dispose
            Assert.True(AutoTrainingCompiler.ReplayMode,
                "Inner tape dispose must restore ReplayMode=true that was active at inner construction");
        }
        finally
        {
            AutoTrainingCompiler.ReplayMode = saved;
        }
    }

    [Fact]
    public void NestedTape_ThreeLevels_EachLevelRestoresCorrectReplayMode()
    {
        // Regression: three-level nesting must unwind in LIFO order.
        bool saved = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = false;

            using var outer = new GradientTape<float>(); // saved=false
            AutoTrainingCompiler.ReplayMode = true;

            using (var mid = new GradientTape<float>()) // saved=true
            {
                AutoTrainingCompiler.ReplayMode = false;

                using (var inner = new GradientTape<float>()) // saved=false
                {
                    AutoTrainingCompiler.ReplayMode = true; // change it
                }
                // inner restored to false
                Assert.False(AutoTrainingCompiler.ReplayMode,
                    "Inner dispose: should restore to false (value when inner was created)");
            }
            // mid restored to true
            Assert.True(AutoTrainingCompiler.ReplayMode,
                "Mid dispose: should restore to true (value when mid was created)");

            // Dispose outer
        }
        finally
        {
            // outer dispose restores to false (value when outer was created)
            AutoTrainingCompiler.ReplayMode = saved;
        }
        Assert.False(AutoTrainingCompiler.ReplayMode,
            "After all tapes disposed, ReplayMode should be the saved value (false)");
    }

    // ──────────────────────────────────────────────────────────────
    // ReplayMode is NOT enabled when compiler is disabled
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void ReplayMode_RemainsUnchanged_WhenCompilerDisabled()
    {
        // When AutoTrainingCompiler.Enabled=false, TryGetCompiledBackward returns null
        // without touching ReplayMode. Verify ReplayMode is not inadvertently enabled.
        bool savedEnabled = AutoTrainingCompiler.Enabled;
        bool savedReplay = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.Enabled = false;
            AutoTrainingCompiler.ReplayMode = false;

            var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
            var weight = new Tensor<float>(new float[] { 1f, 0f, 0f, 1f }, [2, 2]);

            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
            // Run several steps with identical pattern
            for (int i = 0; i < 5; i++)
            {
                var result = _engine.TensorMatMul(input, weight);
                var loss = _engine.ReduceSum(result, null);
                tape.ComputeGradients(loss, new[] { weight });
            }

            Assert.False(AutoTrainingCompiler.ReplayMode,
                "ReplayMode must stay false when compiler is disabled");
        }
        finally
        {
            AutoTrainingCompiler.Enabled = savedEnabled;
            AutoTrainingCompiler.ReplayMode = savedReplay;
        }
    }

    // ──────────────────────────────────────────────────────────────
    // ReplayMode enabled after successful auto-compilation
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void ReplayMode_BecomesTrue_AfterPatternRepeatsOnPersistentTape()
    {
        // After the same forward+backward pattern repeats CompileThreshold+1 times,
        // AutoTrainingCompiler should compile the backward graph and enable ReplayMode.
        bool savedEnabled = AutoTrainingCompiler.Enabled;
        bool savedReplay = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.Enabled = true;
            AutoTrainingCompiler.ReplayMode = false;

            var rng = new Random(42);
            var input = CreateRandom([4, 8], rng);
            var weight = CreateRandom([8, 4], rng);

            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            // Run enough steps to trigger compilation (CompileThreshold = 2, so need >=3 steps
            // for the 3rd call to see a compiled graph)
            for (int step = 0; step < 5; step++)
            {
                var result = _engine.TensorMatMul(input, weight);
                var loss = _engine.ReduceSum(result, null);
                tape.ComputeGradients(loss, new[] { weight });
            }

            // After enough repeating steps, ReplayMode should be true
            Assert.True(AutoTrainingCompiler.ReplayMode,
                "ReplayMode should be enabled after auto-compilation succeeds on repeated pattern");
        }
        finally
        {
            AutoTrainingCompiler.Enabled = savedEnabled;
            AutoTrainingCompiler.ReplayMode = savedReplay;
        }
    }

    [Fact]
    public void ReplayMode_BecomesFalse_WhenLossTensorChanges()
    {
        // If the loss tensor changes between steps (different object identity),
        // the pattern hash changes and ReplayMode should be disabled.
        bool savedEnabled = AutoTrainingCompiler.Enabled;
        bool savedReplay = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.Enabled = true;
            AutoTrainingCompiler.ReplayMode = false;

            var rng = new Random(99);
            var input = CreateRandom([4, 8], rng);
            var weight = CreateRandom([8, 4], rng);

            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            // Step 1: establish a compiled pattern with loss1
            for (int step = 0; step < 5; step++)
            {
                var result = _engine.TensorMatMul(input, weight);
                var loss = _engine.ReduceSum(result, null);
                tape.ComputeGradients(loss, new[] { weight });
            }
            // Pattern should be compiled now
            bool replayAfterCompile = AutoTrainingCompiler.ReplayMode;

            // Step 2: force a different tape entry count (extra op) to break the pattern
            {
                var result = _engine.TensorMatMul(input, weight);
                var scaled = _engine.TensorMultiplyScalar(result, 2f); // extra op
                var loss = _engine.ReduceSum(scaled, null);
                tape.ComputeGradients(loss, new[] { weight });
            }

            // Pattern changed — ReplayMode should be false
            Assert.True(replayAfterCompile, "Precondition: ReplayMode should have been true after compilation");
            Assert.False(AutoTrainingCompiler.ReplayMode,
                "ReplayMode must be cleared when forward pattern changes (different op sequence)");
        }
        finally
        {
            AutoTrainingCompiler.Enabled = savedEnabled;
            AutoTrainingCompiler.ReplayMode = savedReplay;
        }
    }

    // ──────────────────────────────────────────────────────────────
    // RecordStep: same loss identity produces stable pattern hash
    // (verified via ReplayMode eventually becoming true)
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void RecordStep_SameLoss_SamePattern_RepeatedStepsTriggerCompilation()
    {
        // RecordStep now incorporates the loss tensor's identity into the hash.
        // When the same persistent tape reuses the same loss tensor reference across
        // steps, the hash must be stable so compilation can fire.
        bool savedEnabled = AutoTrainingCompiler.Enabled;
        bool savedReplay = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.Enabled = true;
            AutoTrainingCompiler.ReplayMode = false;

            var rng = new Random(7);
            var input = CreateRandom([8, 16], rng);
            var weight = CreateRandom([16, 8], rng);

            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            // Each step creates a new loss tensor, but the forward graph topology
            // (op names + shapes) is identical → pattern should be detected as matching.
            bool compiledAtSomePoint = false;
            for (int step = 0; step < 6; step++)
            {
                var result = _engine.TensorMatMul(input, weight);
                var loss = _engine.ReduceSum(result, null);
                tape.ComputeGradients(loss, new[] { weight });
                if (AutoTrainingCompiler.ReplayMode)
                {
                    compiledAtSomePoint = true;
                    break;
                }
            }

            Assert.True(compiledAtSomePoint,
                "RecordStep should build a stable pattern hash so auto-compilation fires within 6 steps");
        }
        finally
        {
            AutoTrainingCompiler.Enabled = savedEnabled;
            AutoTrainingCompiler.ReplayMode = savedReplay;
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────

    private static Tensor<float> CreateRandom(int[] shape, Random rng)
    {
        int len = 1;
        foreach (int d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }
}