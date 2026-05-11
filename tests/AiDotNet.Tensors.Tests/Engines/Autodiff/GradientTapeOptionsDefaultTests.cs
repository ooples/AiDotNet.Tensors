// Copyright (c) AiDotNet. All rights reserved.
// PR #333 — GradientTapeOptions.Default now ships with Persistent=true,
// which gates the AutoTrainingCompiler fast path that cut 65-73% of
// ComputeGradients walltime on VGG / ResNet profile data. The change
// is silent at the type system, so a regression that flipped Default
// back to Persistent=false would not surface in compile errors; this
// test family is the explicit guard.
//
// AutoTrainingCompiler clone-then-train safety is a separate concern
// covered by AutoTrainingCompilerHashTests — this file only verifies
// the default-flag wiring, not the cache-key behavior.

using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class GradientTapeOptionsDefaultTests
{
    /// <summary>
    /// The Default singleton must report Persistent=true. This gates
    /// AutoTrainingCompiler eligibility — when false, the compiler skips
    /// the entire compiled-backward fast path and falls back to walking
    /// the tape entry list per step.
    /// </summary>
    [Fact]
    public void Default_IsPersistent()
    {
        Assert.True(GradientTapeOptions.Default.Persistent);
    }

    /// <summary>
    /// A parameterless `new GradientTapeOptions()` must also default to
    /// Persistent=true. Field initializer must match the Default singleton
    /// so user code that constructs options manually (not via the singleton)
    /// gets the same compiler-eligibility behavior.
    /// </summary>
    [Fact]
    public void NewInstance_DefaultsToPersistent()
    {
        var options = new GradientTapeOptions();
        Assert.True(options.Persistent);
    }

    /// <summary>
    /// A tape constructed without explicit options must report Persistent=true
    /// on its Options property. This is the end-to-end check that the
    /// new-default actually reaches the tape — earlier passes flipped the
    /// option but forgot to update the field initializer, masking the change.
    /// </summary>
    [Fact]
    public void NewTape_NoOptions_ExposesPersistentDefault()
    {
        using var tape = new GradientTape<float>();
        Assert.True(tape.Options.Persistent);
    }

    /// <summary>
    /// init-only setters must still allow callers to flip Persistent off
    /// when they explicitly want eager-disposal behavior (one-shot
    /// gradient compute, low-memory inference scoring with periodic grad
    /// probes). The Default singleton is not mutated by this — it's a
    /// fresh instance each time.
    /// </summary>
    [Fact]
    public void Persistent_CanBeOverriddenViaInit()
    {
        var options = new GradientTapeOptions { Persistent = false };
        Assert.False(options.Persistent);

        // Singleton Default is unaffected by the user's fresh instance.
        Assert.True(GradientTapeOptions.Default.Persistent);
    }

    /// <summary>
    /// All other defaults remain unchanged across this PR: RecordInPlace
    /// stays true, MaxEntries stays 0 (unlimited), EnableHooks stays false.
    /// Captures the full default surface so a future drift in any direction
    /// is caught — not just Persistent.
    /// </summary>
    [Fact]
    public void OtherDefaults_Unchanged()
    {
        var options = new GradientTapeOptions();
        Assert.True(options.RecordInPlace);
        Assert.Equal(0, options.MaxEntries);
        Assert.False(options.EnableHooks);
    }
}
