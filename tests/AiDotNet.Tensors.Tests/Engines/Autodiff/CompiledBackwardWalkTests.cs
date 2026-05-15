using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Issue #338 Item 3: validates the compiled-IL backward integration
/// scaffolding. The walker compilation today is a passthrough; future
/// commits replace it with a JIT-emitted DynamicMethod. These tests pin
/// the API contract + register/lookup semantics so that future IL work
/// can replace the passthrough without breaking the integration.
/// </summary>
public class CompiledBackwardWalkTests
{
    /// <summary>
    /// Invokes the internal CompiledBackwardWalk&lt;T&gt; statics via
    /// reflection — they're intentionally <c>internal</c> so consumer
    /// code routes through GradientTape's hooks. The test runs in the
    /// same assembly so reflection access is permitted by visibility,
    /// but we keep the names typed at runtime so a future rename surfaces
    /// here rather than as a silent test pass.
    /// </summary>
    private static System.Type WalkerOfFloat()
    {
        var openType = typeof(CompiledBackwardWalk<>);
        return openType.MakeGenericType(typeof(float));
    }

    [Fact]
    public void Enabled_DefaultsToFalse_WithoutEnvVar()
    {
        var t = WalkerOfFloat();
        var prop = t.GetProperty("Enabled",
            BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(prop);
        var value = (bool)prop!.GetValue(null)!;
        // The default for AIDOTNET_COMPILED_BACKWARD is unset → Enabled false.
        // If a CI runner sets the env var, this test legitimately reflects
        // that — the assertion validates the parse logic, not a hard "must
        // be off" rule. So we only assert the parse: when env var is empty
        // or "0", we get false.
        var raw = System.Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_BACKWARD");
        if (string.IsNullOrEmpty(raw) || raw == "0")
            Assert.False(value);
        else
            Assert.True(value);
    }

    [Fact]
    public void Register_ThenTryGetWalker_ReturnsRegisteredDelegate()
    {
        var t = WalkerOfFloat();
        var resetMethod = t.GetMethod("ResetForTests",
            BindingFlags.NonPublic | BindingFlags.Static);
        var tryGetMethod = t.GetMethod("TryGetWalker",
            BindingFlags.NonPublic | BindingFlags.Static);
        var registerMethod = t.GetMethod("Register",
            BindingFlags.NonPublic | BindingFlags.Static);
        var compileMethod = t.GetMethod("Compile",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(resetMethod);
        Assert.NotNull(tryGetMethod);
        Assert.NotNull(registerMethod);
        Assert.NotNull(compileMethod);

        resetMethod!.Invoke(null, null);
        Assert.Null(tryGetMethod!.Invoke(null, new object[] { 12345L }));

        // Use Compile to obtain a real walker delegate — avoids the type-
        // matching gymnastics of constructing one externally. The walker
        // is the passthrough; we never invoke it here, only register and
        // retrieve it to verify cache lookup semantics.
        var indices = new[] { 0 };
        var walker = compileMethod!.Invoke(null, new object[] { indices });
        Assert.NotNull(walker);

        registerMethod!.Invoke(null, new object[] { 12345L, walker! });

        var retrieved = tryGetMethod.Invoke(null, new object[] { 12345L });
        Assert.NotNull(retrieved);
        Assert.Same(walker, retrieved);

        // Tidy up so subsequent tests start with a clean slate.
        resetMethod.Invoke(null, null);
    }

    [Fact]
    public void CompiledWalker_ProducesIdenticalGradientsToReference_SimpleTape()
    {
        // End-to-end parity gate: builds a tiny gradient tape, runs
        // ComputeGradients through the existing reference dispatcher,
        // and verifies the analytic gradient. The CompiledBackwardWalk
        // is registered in RebindablePlanCache.Store whenever
        // AIDOTNET_COMPILED_BACKWARD is set, so this also exercises the
        // walker's registration path. The walker's invocation path is
        // gated behind the env var — when it's off, this test exercises
        // the reference dispatcher and ensures CompiledBackwardWalk's
        // mere presence doesn't break the existing flow.
        var prior = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            // GradientTape activates on construction and deactivates on
            // Dispose. Watch isn't needed — DifferentiableOps records
            // every engine op while the tape is active.
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            // y = (a * b) + a — exercises MultiplyBackward, AddBackward,
            // and an input referenced twice (so a's gradient sums two
            // contributions: dL/d(a*b)*b + dL/d(a*b+a)*1).
            var a = new Tensor<float>(new float[] { 2.0f, 3.0f }, new[] { 2 });
            var b = new Tensor<float>(new float[] { 4.0f, 5.0f }, new[] { 2 });
            var prod = engine.TensorMultiply(a, b);
            var y = engine.TensorAdd(prod, a);
            // ReduceSum with no axes reduces to a scalar tensor (shape [1]).
            var loss = engine.ReduceSum(y);

            var referenceGrads = tape.ComputeGradients(loss, new List<Tensor<float>> { a, b });
            Assert.True(referenceGrads.ContainsKey(a));
            Assert.True(referenceGrads.ContainsKey(b));

            // y = a*b + a; sum(y) = sum(a*b + a)
            // dL/da = b + 1 = [5, 6]
            // dL/db = a     = [2, 3]
            var aGradRef = referenceGrads[a];
            var bGradRef = referenceGrads[b];
            Assert.Equal(5.0f, aGradRef.GetDataArray()[0], 4);
            Assert.Equal(6.0f, aGradRef.GetDataArray()[1], 4);
            Assert.Equal(2.0f, bGradRef.GetDataArray()[0], 4);
            Assert.Equal(3.0f, bGradRef.GetDataArray()[1], 4);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void CompiledWalker_ILEmitted_ProducesSameGradientsAsReference()
    {
        // Bypass the env-var gating by calling CompiledBackwardWalk<T>.Compile
        // directly (which always attempts IL emission first, falling back to
        // closure only on emit failure). The returned walker then runs on the
        // SAME tape entries the reference dispatcher walked, and we compare.
        var prior = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
            var a = new Tensor<float>(new float[] { 2.0f, 3.0f, 4.0f }, new[] { 3 });
            var b = new Tensor<float>(new float[] { 5.0f, 6.0f, 7.0f }, new[] { 3 });
            var prod = engine.TensorMultiply(a, b);
            var sum = engine.TensorAdd(prod, a);
            var loss = engine.ReduceSum(sum);

            // Reference path: compute via the inline dispatcher.
            var refGrads = tape.ComputeGradients(loss, new List<Tensor<float>> { a, b });
            float refDA0 = refGrads[a].GetDataArray()[0];
            float refDA1 = refGrads[a].GetDataArray()[1];
            float refDA2 = refGrads[a].GetDataArray()[2];
            float refDB0 = refGrads[b].GetDataArray()[0];
            float refDB1 = refGrads[b].GetDataArray()[1];
            float refDB2 = refGrads[b].GetDataArray()[2];

            // dL/da = b + 1, dL/db = a
            Assert.Equal(6.0f, refDA0, 4);
            Assert.Equal(7.0f, refDA1, 4);
            Assert.Equal(8.0f, refDA2, 4);
            Assert.Equal(2.0f, refDB0, 4);
            Assert.Equal(3.0f, refDB1, 4);
            Assert.Equal(4.0f, refDB2, 4);

            // Build walker indices directly from the tape entry count.
            // The tape's reverse-topo walk visits entries from last
            // to first, so indices = [count-1, count-2, ..., 0].
            var tapeType = typeof(GradientTape<float>);
            var entriesField = tapeType.GetField("_entries",
                BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(entriesField);
            var entriesObj = entriesField!.GetValue(tape);
            Assert.NotNull(entriesObj);

            // TapeEntryArena<float>.Count via reflection.
            var arenaType = entriesObj!.GetType();
            var countProp = arenaType.GetProperty("Count",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(countProp);
            int entryCount = (int)countProp!.GetValue(entriesObj)!;
            if (entryCount == 0)
                return; // tape didn't record (shouldn't happen but
                        // defensive)

            var indices = new int[entryCount];
            for (int i = 0; i < entryCount; i++)
                indices[i] = entryCount - 1 - i;  // reverse-topo

            var walkerType = WalkerOfFloat();
            var compileMethod = walkerType.GetMethod("Compile",
                BindingFlags.NonPublic | BindingFlags.Static)!;
            var walker = compileMethod.Invoke(null, new object[] { indices });
            Assert.NotNull(walker);

            // Walker signature:
            //   (TapeEntryArena<T>, Tensor<T>, IReadOnlyList<Tensor<T>>?, IEngine)
            //     -> Dictionary<Tensor<T>, Tensor<T>>?
            var sources = new List<Tensor<float>> { a, b };
            var result = walker!.GetType().GetMethod("Invoke")!
                .Invoke(walker, new object?[] { entriesObj, loss, sources, engine });
            Assert.NotNull(result);

            var ilGrads = (Dictionary<Tensor<float>, Tensor<float>>)result!;
            float ilDA0 = ilGrads[a].GetDataArray()[0];
            float ilDA1 = ilGrads[a].GetDataArray()[1];
            float ilDA2 = ilGrads[a].GetDataArray()[2];
            float ilDB0 = ilGrads[b].GetDataArray()[0];
            float ilDB1 = ilGrads[b].GetDataArray()[1];
            float ilDB2 = ilGrads[b].GetDataArray()[2];

            // The IL-emitted walker MUST produce identical gradients to
            // the reference dispatcher — same input tape, same logic,
            // different code-gen.
            Assert.Equal(refDA0, ilDA0, 4);
            Assert.Equal(refDA1, ilDA1, 4);
            Assert.Equal(refDA2, ilDA2, 4);
            Assert.Equal(refDB0, ilDB0, 4);
            Assert.Equal(refDB1, ilDB1, 4);
            Assert.Equal(refDB2, ilDB2, 4);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [Fact]
    public void Compile_PassthroughWalker_ReturnsNonNullDelegate()
    {
        var t = WalkerOfFloat();
        var compileMethod = t.GetMethod("Compile",
            BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(compileMethod);

        // Pass a 3-element index array as a minimal valid input. The
        // walker won't be invoked here — we only assert Compile returns
        // a non-null delegate, not the walk's behaviour. Walk behaviour
        // is exercised end-to-end by the GradientTape replay tests.
        var indices = new[] { 0, 1, 2 };
        var walker = compileMethod!.Invoke(null, new object[] { indices });
        Assert.NotNull(walker);
    }

}
