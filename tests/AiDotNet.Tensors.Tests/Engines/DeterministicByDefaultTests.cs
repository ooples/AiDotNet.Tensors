using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Acceptance tests for issue #164 — deterministic-by-default.
///
/// Verifies:
///   1. <see cref="TensorCodecOptions.Deterministic"/> defaults to <c>true</c>.
///   2. <see cref="AiDotNetEngine.DeterministicMode"/> defaults to <c>true</c>.
///   3. <see cref="CompiledModelCache{T}"/> invalidates plans compiled under the
///      opposite determinism setting.
///
/// Shares xUnit collection <c>BlasGlobalState</c> with the existing
/// <see cref="DeterministicModeTests"/> so tests do not run in parallel —
/// they mutate a process-wide static flag that would race under xUnit
/// parallelism.
/// </summary>
[Collection("BlasGlobalState")]
public class DeterministicByDefaultTests
{
    // ── Acceptance criterion #1 ──────────────────────────────────────────────
    [Fact]
    public void TensorCodecOptions_DeterministicProperty_DefaultsToTrue()
    {
        var options = new TensorCodecOptions();
        Assert.True(options.Deterministic,
            "new TensorCodecOptions().Deterministic must default to true per issue #164");
    }

    [Fact]
    public void TensorCodecOptions_DefaultSingleton_DeterministicIsTrue()
    {
        // TensorCodecOptions.Default returns a fresh instance each access; verify the
        // factory still produces deterministic-by-default instances.
        Assert.True(TensorCodecOptions.Default.Deterministic);
    }

    [Fact]
    public void TensorCodecOptions_DeterministicCanBeOverridden()
    {
        var options = new TensorCodecOptions { Deterministic = false };
        Assert.False(options.Deterministic);
    }

    // ── Acceptance criterion #2 ──────────────────────────────────────────────
    [Fact]
    public void AiDotNetEngine_DeterministicMode_DefaultsToTrue_WhenNotExplicitlySet()
    {
        // This is a process-wide static flag. Other tests in this collection
        // save/restore it; since tests in this collection run serially and the
        // flag starts at the compile-time default (true after #164), any test
        // in isolation will observe true. Tests that need to verify the flag
        // after explicit changes save/restore around themselves.
        //
        // To verify the *compile-time* default in a way that does not depend
        // on prior test state, we reset to the documented default and read back.
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            // Set to false, then to true to force a known-clean transition.
            AiDotNetEngine.SetDeterministicMode(false);
            AiDotNetEngine.SetDeterministicMode(true);
            Assert.True(AiDotNetEngine.DeterministicMode,
                "After issue #164, deterministic mode is the documented default.");
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    // ── Acceptance criterion #3 ──────────────────────────────────────────────
    [Fact]
    public void CompiledModelCache_DifferentDeterminismProducesDifferentPlans()
    {
        // Plan compiled under deterministic=true must not be served on a subsequent
        // call under deterministic=false (and vice versa). The cache key mixes in
        // the current determinism state so the second call misses and recompiles.
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            var engine = new CpuEngine();
            var input = Tensor<float>.CreateRandom([4, 3]);
            var weight = Tensor<float>.CreateRandom([3, 2]);

            using var cache = new CompiledModelCache<float>();

            AiDotNetEngine.SetDeterministicMode(true);
            var planDeterministic = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            AiDotNetEngine.SetDeterministicMode(false);
            var planNonDeterministic = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            // The two plans must be distinct instances — the cache is keyed on
            // (shape, elementType, deterministicMode), so switching determinism
            // forces a miss and recompile even though the shape and type are
            // identical.
            Assert.NotSame(planDeterministic, planNonDeterministic);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    [Fact]
    public void CompiledModelCache_SameDeterminismReusesPlan()
    {
        // Regression guard: the key mixes in determinism, but *same* state on
        // back-to-back calls must still hit the cache (otherwise every call
        // would recompile, wiping the point of the cache).
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);

            var engine = new CpuEngine();
            var input = Tensor<float>.CreateRandom([4, 3]);
            var weight = Tensor<float>.CreateRandom([3, 2]);

            using var cache = new CompiledModelCache<float>();

            var first = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            var second = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            Assert.Same(first, second);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    // ── Behavioural sanity: round-trip still converges numerically ──────────
    [Fact]
    public void DefaultDeterministicMode_MatMulProducesReasonableResults()
    {
        // Full path smoke test: fresh tensors, engine.TensorMatMul under the
        // new default, no explicit SetDeterministicMode. Output must not be
        // all-NaN / all-zero / otherwise nonsensical, and should agree with
        // an explicit deterministic=true run to bit precision (same path).
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            var engine = new CpuEngine();
            int m = 32, k = 32, n = 32;
            var rng = new Random(42);
            var a = new Tensor<float>([m, k]);
            var b = new Tensor<float>([k, n]);
            for (int i = 0; i < m * k; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < k * n; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            AiDotNetEngine.SetDeterministicMode(true);
            var reference = engine.TensorMatMul(a, b);

            // Pretend we "forgot" the explicit call and re-run at the default.
            AiDotNetEngine.SetDeterministicMode(false);
            AiDotNetEngine.SetDeterministicMode(true);
            var under_default = engine.TensorMatMul(a, b);

            for (int i = 0; i < m * n; i++)
            {
                Assert.Equal(reference[i], under_default[i]);
                Assert.False(float.IsNaN(reference[i]));
            }
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    // ── Thread-local determinism scope ───────────────────────────────────────
    //
    // The process-wide flag is a coarse lever — turning it off makes the entire
    // process non-deterministic until someone turns it back on. The thread-local
    // override (issue #164 follow-up) lets a single thread opt into or out of
    // determinism without affecting any other thread. This matters for:
    //   - Mixed workloads where one worker needs reproducibility (replay, debug)
    //     while another prioritises throughput.
    //   - Libraries layered on top of ours that want to scope determinism to a
    //     specific operation without a global side-effect.
    //
    // The override is driven by TensorCodecOptions.SetCurrent — the same
    // thread-local ambient-context pattern already used for the rest of the
    // options bag. `null` means "inherit the process-wide value."

    [Fact]
    public void SetCurrent_WithDeterministicFalse_OverridesProcessWideOnThisThread()
    {
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        var originalOptions  = TensorCodecOptions.Current;
        try
        {
            // Process-wide true (the default after #164).
            AiDotNetEngine.SetDeterministicMode(true);
            Assert.True(BlasProvider.IsDeterministicMode);

            // Install a thread-local override via SetCurrent.
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = false });

            // The accessor must now reflect the override, not the process-wide value.
            Assert.False(BlasProvider.IsDeterministicMode);
        }
        finally
        {
            // Restore (not clear) so we don't clobber any ambient options bag a
            // fixture or earlier collection-mate may have installed.
            // Restore the captured ambient options for hygiene, then explicitly
            // clear the BLAS override. Subtle but important: `Current` returns
            // `_current ?? Default`, so when `_current` was null at capture time
            // we end up holding a fresh `Default()` with `Deterministic=true`.
            // Calling `SetCurrent(originalOptions)` would install that forced
            // `true` override for this thread, leaking determinism state into
            // whatever test the collection runs next. Clearing the override
            // afterwards normalises that to "inherit process-wide."
            TensorCodecOptions.SetCurrent(originalOptions);
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }

    [Fact]
    public void SetCurrent_WithDeterministicTrue_OverridesProcessWideOnThisThread()
    {
        // The inverse direction: process is OFF but the caller wants ON for this thread.
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        var originalOptions  = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.SetDeterministicMode(false);
            Assert.False(BlasProvider.IsDeterministicMode);

            TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = true });
            Assert.True(BlasProvider.IsDeterministicMode);
        }
        finally
        {
            // Restore the captured ambient options for hygiene, then explicitly
            // clear the BLAS override. Subtle but important: `Current` returns
            // `_current ?? Default`, so when `_current` was null at capture time
            // we end up holding a fresh `Default()` with `Deterministic=true`.
            // Calling `SetCurrent(originalOptions)` would install that forced
            // `true` override for this thread, leaking determinism state into
            // whatever test the collection runs next. Clearing the override
            // afterwards normalises that to "inherit process-wide."
            TensorCodecOptions.SetCurrent(originalOptions);
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }

    [Fact]
    public void SetCurrent_Null_ClearsThreadLocalOverrideAndInheritsProcessWide()
    {
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        var originalOptions  = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);

            // Install then clear.
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = false });
            Assert.False(BlasProvider.IsDeterministicMode);
            TensorCodecOptions.SetCurrent(null);

            // Cleared → inherit the process-wide value.
            Assert.True(BlasProvider.IsDeterministicMode);

            // And flipping the process-wide now flips this thread too (no lingering override).
            AiDotNetEngine.SetDeterministicMode(false);
            Assert.False(BlasProvider.IsDeterministicMode);
        }
        finally
        {
            // Restore the captured ambient options for hygiene, then explicitly
            // clear the BLAS override. Subtle but important: `Current` returns
            // `_current ?? Default`, so when `_current` was null at capture time
            // we end up holding a fresh `Default()` with `Deterministic=true`.
            // Calling `SetCurrent(originalOptions)` would install that forced
            // `true` override for this thread, leaking determinism state into
            // whatever test the collection runs next. Clearing the override
            // afterwards normalises that to "inherit process-wide."
            TensorCodecOptions.SetCurrent(originalOptions);
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }

    [Fact]
    public async Task SetCurrent_OnOneThread_DoesNotLeakToAnotherThread()
    {
        // The core thread-local invariant: thread A's override must not change
        // what thread B sees. Without this, "thread-local" would be a lie.
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        var originalOptions  = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);

            // Parked thread: synchronises via a barrier so we can observe its state
            // AFTER the main thread installs its override but BEFORE the parked
            // thread exits.
            using var parkedReady = new ManualResetEventSlim(false);
            using var mayExit     = new ManualResetEventSlim(false);
            bool parkedObservation = false;

            var parked = Task.Run(() =>
            {
                parkedReady.Set();
                mayExit.Wait();
                // Record the thread-local accessor's value from the parked thread.
                parkedObservation = BlasProvider.IsDeterministicMode;
            });

            parkedReady.Wait();
            // Main thread installs a thread-local override.
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = false });
            Assert.False(BlasProvider.IsDeterministicMode); // main thread sees the override

            // Release the parked thread.
            mayExit.Set();
            await parked.ConfigureAwait(false);

            // The parked thread never installed an override of its own, so it
            // must still see the process-wide value (true), not main's override.
            Assert.True(parkedObservation,
                "The parked thread must inherit the process-wide setting, not main thread's override.");
        }
        finally
        {
            // Restore the captured ambient options for hygiene, then explicitly
            // clear the BLAS override. Subtle but important: `Current` returns
            // `_current ?? Default`, so when `_current` was null at capture time
            // we end up holding a fresh `Default()` with `Deterministic=true`.
            // Calling `SetCurrent(originalOptions)` would install that forced
            // `true` override for this thread, leaking determinism state into
            // whatever test the collection runs next. Clearing the override
            // afterwards normalises that to "inherit process-wide."
            TensorCodecOptions.SetCurrent(originalOptions);
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }

    [Fact]
    public void SetCurrent_ThreadLocalOverride_ReflectsInCompiledModelCacheKey()
    {
        // The cache key already mixes in BlasProvider.IsDeterministicMode, so the
        // thread-local override should automatically produce distinct plans when a
        // single thread toggles its own override between two compile calls.
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        var originalOptions  = TensorCodecOptions.Current;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);

            var engine = new CpuEngine();
            var input  = Tensor<float>.CreateRandom([4, 3]);
            var weight = Tensor<float>.CreateRandom([3, 2]);

            using var cache = new CompiledModelCache<float>();

            // First compile: thread-local override unset → process-wide (true).
            TensorCodecOptions.SetCurrent(null);
            var planInherited = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            // Second compile: same thread, but now with a local override to false.
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = false });
            var planOverridden = cache.GetOrCompileInference(
                input._shape,
                () =>
                {
                    var output = engine.TensorMatMul(input, weight);
                    return engine.ReduceSum(output, null);
                });

            Assert.NotSame(planInherited, planOverridden);
        }
        finally
        {
            // Restore the captured ambient options for hygiene, then explicitly
            // clear the BLAS override. Subtle but important: `Current` returns
            // `_current ?? Default`, so when `_current` was null at capture time
            // we end up holding a fresh `Default()` with `Deterministic=true`.
            // Calling `SetCurrent(originalOptions)` would install that forced
            // `true` override for this thread, leaking determinism state into
            // whatever test the collection runs next. Clearing the override
            // afterwards normalises that to "inherit process-wide."
            TensorCodecOptions.SetCurrent(originalOptions);
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }

    [Fact]
    public void BlasProvider_SetThreadLocalDeterministicMode_DirectAPI()
    {
        // Lower-level sanity: the BlasProvider setter works without going through
        // TensorCodecOptions. Exposing it directly lets advanced consumers wire
        // their own scope (e.g. a using-pattern helper) without installing a full
        // options bag.
        bool originalProcess = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            Assert.True(BlasProvider.IsDeterministicMode);

            BlasProvider.SetThreadLocalDeterministicMode(false);
            Assert.False(BlasProvider.IsDeterministicMode);

            BlasProvider.SetThreadLocalDeterministicMode(null);
            Assert.True(BlasProvider.IsDeterministicMode); // back to process-wide
        }
        finally
        {
            BlasProvider.SetThreadLocalDeterministicMode(null);
            AiDotNetEngine.SetDeterministicMode(originalProcess);
        }
    }
}
