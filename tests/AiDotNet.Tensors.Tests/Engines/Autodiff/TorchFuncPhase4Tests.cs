// Copyright (c) AiDotNet. All rights reserved.
// Phase 4 of issue #214: SavedTensorHooks recipes, AnomalyMode,
// tensor backward hooks.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Phase 4 coverage — saved-tensor hooks (SaveOnCpu,
/// SaveQuantizedInt8), <see cref="AnomalyModeScope"/>, and the
/// existing tensor-hook API.
/// </summary>
public class TorchFuncPhase4Tests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ─── SavedTensorHooks ─────────────────────────────────────────────

    [Fact]
    public void SavedTensorHooks_IsActive_TracksScopeDepth()
    {
        Assert.False(SavedTensorHooks.IsActive);
        using (SavedTensorRecipes.SaveOnCpu<float>())
        {
            Assert.True(SavedTensorHooks.IsActive);
            using (SavedTensorRecipes.SaveOnCpu<float>())
            {
                Assert.True(SavedTensorHooks.IsActive);
            }
            Assert.True(SavedTensorHooks.IsActive);
        }
        Assert.False(SavedTensorHooks.IsActive);
    }

    [Fact]
    public void SavedTensorHooks_SaveOnCpu_RoundTripPreservesData()
    {
        using var _ = SavedTensorRecipes.SaveOnCpu<float>();
        var original = new Tensor<float>(new[] { 1.5f, -2.5f, 3.5f }, new[] { 3 });

        Assert.True(SavedTensorHooks.TryApplyPack(original, out var packed));
        Assert.NotNull(packed);

        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed!);
        Assert.Equal(new[] { 3 }, recovered._shape);
        Assert.Equal(1.5f, recovered[0]);
        Assert.Equal(-2.5f, recovered[1]);
        Assert.Equal(3.5f, recovered[2]);
    }

    [Fact]
    public void SavedTensorHooks_SaveOnCpu_IsDefensiveCopy()
    {
        using var _ = SavedTensorRecipes.SaveOnCpu<float>();
        var original = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });

        Assert.True(SavedTensorHooks.TryApplyPack(original, out var packed));
        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed!);

        // Mutating the recovered data must not touch the original.
        recovered.AsWritableSpan()[0] = 99f;
        Assert.Equal(1f, original[0]);
    }

    [Fact]
    public void SavedTensorHooks_SaveQuantizedInt8_ReducesBytes4x()
    {
        // 256 floats = 1024 bytes primal. Quantized sbyte = 256 bytes.
        // Round-trip introduces quantization error bounded by scale/128.
        var data = new float[256];
        for (int i = 0; i < 256; i++) data[i] = (i - 128) * 0.1f; // [-12.8, 12.7]
        var original = new Tensor<float>(data, new[] { 256 });

        using var _ = SavedTensorRecipes.SaveQuantizedInt8();
        Assert.True(SavedTensorHooks.TryApplyPack(original, out var packed));

        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed!);
        Assert.Equal(256, recovered.Length);

        // Every element within one quantization step of the original.
        float maxErr = 0f;
        for (int i = 0; i < 256; i++)
        {
            var err = Math.Abs(recovered[i] - original[i]);
            if (err > maxErr) maxErr = err;
        }
        // absMax ≈ 12.8, scale = 12.8/127 ≈ 0.1008. Max per-element
        // error ≈ 0.5·scale ≈ 0.05.
        Assert.True(maxErr < 0.1f,
            $"Quantization round-trip error {maxErr} exceeds half-scale tolerance.");
    }

    [Fact]
    public void SavedTensorHooks_TryApplyPack_ReturnsFalseWhenInactive()
    {
        var tensor = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.False(SavedTensorHooks.TryApplyPack(tensor, out var packed));
        Assert.Null(packed);
    }

    [Fact]
    public void SavedTensorHooks_ApplyUnpack_ThrowsWhenNoHookActive()
    {
        Assert.Throws<InvalidOperationException>(
            () => SavedTensorHooks.ApplyUnpack<float>(new object()));
    }

    [Fact]
    public void SavedTensorHooks_Pop_WithoutPush_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => SavedTensorHooks.Pop());
    }

    // ─── AnomalyMode ──────────────────────────────────────────────────

    [Fact]
    public void AnomalyMode_IsActive_TracksScope()
    {
        Assert.False(AnomalyModeScope.IsActive);
        using (var _ = new AnomalyModeScope())
        {
            Assert.True(AnomalyModeScope.IsActive);
        }
        Assert.False(AnomalyModeScope.IsActive);
    }

    [Fact]
    public void AnomalyMode_NestedScopes_BothRemainActive()
    {
        using (var outer = new AnomalyModeScope())
        {
            Assert.True(AnomalyModeScope.IsActive);
            using (var inner = new AnomalyModeScope())
            {
                Assert.True(AnomalyModeScope.IsActive);
            }
            Assert.True(AnomalyModeScope.IsActive);
        }
        Assert.False(AnomalyModeScope.IsActive);
    }

    [Fact]
    public void AnomalyMode_CapturesCallerSite()
    {
        using var scope = new AnomalyModeScope();
        // CallerFilePath / CallerLineNumber default args pick up this
        // test file and the line where the scope was constructed.
        Assert.NotNull(scope.EnteredAtFile);
        Assert.EndsWith("TorchFuncPhase4Tests.cs", scope.EnteredAtFile!);
        Assert.True(scope.EnteredAtLine > 0);
    }

    [Fact]
    public void AnomalyDetectedException_MessageIncludesFileAndLine()
    {
        var ex = new AnomalyDetectedException(
            operationName: "TensorDivide",
            forwardCallerFile: "Foo.cs",
            forwardCallerLine: 42);
        Assert.Equal("TensorDivide", ex.OperationName);
        Assert.Equal("Foo.cs", ex.ForwardCallerFile);
        Assert.Equal(42, ex.ForwardCallerLine);
        Assert.Contains("Foo.cs:42", ex.Message);
        Assert.Contains("TensorDivide", ex.Message);
    }

    // ─── Backward hooks ───────────────────────────────────────────────

    [Fact]
    public void TensorHook_ScalesGradientBeforeAccumulation()
    {
        // f(x) = x² at x=3: df/dx = 6. With a hook that halves the
        // output gradient, the final source gradient is 3.
        using var tape = new GradientTape<float>(new GradientTapeOptions { EnableHooks = true });
        var x = new Tensor<float>(new[] { 3f }, new[] { 1 });
        var y = _engine.TensorMultiply(x, x);

        tape.RegisterHook(y, g => _engine.TensorMultiplyScalar(g, 0.5f));

        var grads = tape.ComputeGradients(y, new[] { x });
        Assert.Equal(3f, grads[x][0], precision: 3);
    }
}
