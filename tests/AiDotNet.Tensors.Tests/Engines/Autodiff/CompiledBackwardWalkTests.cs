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
