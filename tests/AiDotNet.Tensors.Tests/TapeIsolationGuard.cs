using System;
using System.Reflection;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit.Sdk;

// Applied assembly-wide: xUnit runs BeforeAfterTestAttributes found at the
// assembly level around every test in the assembly.
[assembly: AiDotNet.Tensors.Tests.TapeIsolationGuard]

namespace AiDotNet.Tensors.Tests;

/// <summary>
/// Enforces gradient-tape isolation between tests. A test that constructs a
/// <see cref="GradientTape{T}"/> and never disposes it (e.g. an assertion
/// throws before <c>Dispose</c>) leaves the process-wide
/// <c>DifferentiableOps._anyTapeActive</c> counter and the thread's
/// <c>GradientTape&lt;T&gt;.Current</c> set — which makes later tests on that
/// thread record onto a stale tape (wrong gradients) and flips tape-gated
/// BlasManaged GEMM dispatch (wrong numbers). That was the root cause of the
/// cross-suite, isolation-only flaky failures.
///
/// <para>After every test this checks for leaked tape state, logs the culprit
/// test by name (so the underlying missing-dispose can be fixed at source),
/// and resets the state so the next test starts clean. The reset is legitimate
/// test isolation, not assertion masking — combined with the GradientTape
/// finalizer it makes the suite deterministic.</para>
/// </summary>
[AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public sealed class TapeIsolationGuardAttribute : BeforeAfterTestAttribute
{
    public override void After(MethodInfo methodUnderTest)
    {
        // Reset ALL thread-static autodiff state for THIS test's thread after
        // every test. A test that leaves any of these set (e.g. an undisposed
        // GradientTape / NoGradScope, or ReplayMode toggled without restore on
        // an exception path) corrupts later tests on the thread — wrong/zeroed
        // gradients and flipped tape-gated GEMM dispatch. All four are
        // ThreadStatic, so resetting them is safe under xUnit's class-level
        // parallelism (it never touches another thread's state, nor the
        // process-wide _anyTapeActive counter — which the GradientTape
        // finalizer heals on GC).
        bool leaked = GradientTape<float>.Current is not null
            || GradientTape<double>.Current is not null
            || DifferentiableOps.ThreadTapeActive()
            || NoGradScope<float>.IsSuppressed
            || NoGradScope<double>.IsSuppressed
            || InferenceModeScope<float>.IsActive
            || InferenceModeScope<double>.IsActive
            || InferenceModeFlag.IsActive
            || AutoTrainingCompiler.ReplayMode;

        if (leaked)
        {
            Console.WriteLine(
                $"[TAPE LEAK] {methodUnderTest.DeclaringType?.FullName}.{methodUnderTest.Name} " +
                $"left thread-static autodiff state set (floatCurrent={GradientTape<float>.Current is not null}, " +
                $"doubleCurrent={GradientTape<double>.Current is not null}, threadDepth>0={DifferentiableOps.ThreadTapeActive()}, " +
                $"noGradFloat={NoGradScope<float>.IsSuppressed}, noGradDouble={NoGradScope<double>.IsSuppressed}, " +
                $"inferFloat={InferenceModeScope<float>.IsActive}, inferDouble={InferenceModeScope<double>.IsActive}, " +
                $"inferFlag={InferenceModeFlag.IsActive}, replayMode={AutoTrainingCompiler.ReplayMode}). " +
                $"The test should dispose its scopes — resetting to isolate the next test.");
        }

        GradientTape<float>.ResetCurrentForTests();
        GradientTape<double>.ResetCurrentForTests();
        DifferentiableOps.ResetThreadTapeStateForTests();
        NoGradScope<float>.ResetForTests();
        NoGradScope<double>.ResetForTests();
        InferenceModeScope<float>.ResetForTests();
        InferenceModeScope<double>.ResetForTests();
        InferenceModeFlag.ResetForTests();
        // ResetState() clears BOTH the [ThreadStatic] ReplayMode flag AND the
        // [ThreadStatic] cached AutoTrainingState (compiled-plan/step-hash
        // accumulator) — a leaked _state would otherwise let one test's
        // recorded step pattern bleed into the next test on the same thread.
        AutoTrainingCompiler.ResetState();
    }
}
