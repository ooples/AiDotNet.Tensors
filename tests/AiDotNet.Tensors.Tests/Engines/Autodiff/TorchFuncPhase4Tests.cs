// Copyright (c) AiDotNet. All rights reserved.
// Phase 4 of issue #214: SavedTensorHooks recipes, AnomalyMode,
// tensor backward hooks.
//
// Nullable disabled: recovery path accesses out-parameter after a
// TryApplyPack that returned true — see CLAUDE.md ban on null-
// forgiving operators in production code.

#nullable disable

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

        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed);
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
        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed);

        // Mutating the recovered data must not touch the original.
        recovered.AsWritableSpan()[0] = 99f;
        Assert.Equal(1f, original[0]);
    }

    [Fact]
    public void SavedTensorHooks_SaveQuantizedInt8_ReducesBytes4x()
    {
        // 256 floats = 1024 bytes primal. Quantized sbyte = 256 bytes.
        // Round-trip introduces quantization error bounded by scale/2.
        var data = new float[256];
        for (int i = 0; i < 256; i++) data[i] = (i - 128) * 0.1f; // [-12.8, 12.7]
        var original = new Tensor<float>(data, new[] { 256 });

        using var _ = SavedTensorRecipes.SaveQuantizedInt8();
        Assert.True(SavedTensorHooks.TryApplyPack(original, out var packed));
        Assert.NotNull(packed);

        // Verify the actual compressed payload — the packed wrapper
        // exposes its sbyte[] body via reflection (PackedSavedTensor
        // is internal). 256 sbytes = 256 bytes vs 1024 float bytes →
        // exact 4× reduction. Without this assertion a regression
        // that stored the full-precision payload would still pass
        // the round-trip-error check below.
        var payloadField = packed.GetType().GetProperty("Payload",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(payloadField);
        var quantizedPayload = payloadField.GetValue(packed);
        Assert.NotNull(quantizedPayload);
        // The recipe's QuantizedSaved holds an sbyte[] Data field +
        // a float Scale + an int[] Shape. Locate the sbyte[] via
        // reflection so the test stays robust against renames.
        var dataMember = FindByteArrayField(quantizedPayload);
        Assert.NotNull(dataMember);
        var quantBytes = (sbyte[])dataMember!;
        Assert.Equal(256, quantBytes.Length); // 256 sbytes = 1/4 of 1024 float bytes.

        var recovered = SavedTensorHooks.ApplyUnpack<float>(packed);
        Assert.Equal(256, recovered.Length);

        // Every element within roughly half a quantization step of
        // the original. absMax ≈ 12.8 → scale = 12.8/127 ≈ 0.1008 →
        // max per-element error ≈ 0.5·scale ≈ 0.0504. Tightened from
        // the previous loose bound so a regression to nearly-a-full-
        // step error is caught.
        float maxErr = 0f;
        for (int i = 0; i < 256; i++)
        {
            var err = Math.Abs(recovered[i] - original[i]);
            if (err > maxErr) maxErr = err;
        }
        Assert.True(maxErr < 0.06f,
            $"Quantization round-trip error {maxErr} exceeds half-scale tolerance (~0.05).");
    }

    private static object? FindByteArrayField(object packed)
    {
        var t = packed.GetType();
        // Look for any sbyte[] property or field — the recipe's
        // private storage shape can evolve; the test just needs to
        // confirm an sbyte[] exists with the expected length.
        foreach (var prop in t.GetProperties(
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
          | System.Reflection.BindingFlags.Instance))
        {
            if (prop.PropertyType == typeof(sbyte[])) return prop.GetValue(packed);
        }
        foreach (var fld in t.GetFields(
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
          | System.Reflection.BindingFlags.Instance))
        {
            if (fld.FieldType == typeof(sbyte[])) return fld.GetValue(packed);
        }
        return null;
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

    [Fact]
    public void AnomalyMode_NanInBackward_ThrowsAnomalyDetectedException()
    {
        // log(0) sends the primal to -∞; log's backward is 1/x which
        // at x=0 produces +∞ → the gradient lands Inf on the input.
        // Under AnomalyModeScope the tape should throw
        // AnomalyDetectedException tagged with the op name.
        using var _ = new AnomalyModeScope();
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new[] { 0f }, new[] { 1 });
        var y = _engine.TensorLog(x); // primal = -∞, backward: 1/x = +∞

        var ex = Assert.Throws<AnomalyDetectedException>(
            () => tape.ComputeGradients(y, new[] { x }));
        // The exception must name a concrete op — any log-like name works.
        Assert.False(string.IsNullOrEmpty(ex.OperationName));
    }

    [Fact]
    public void AnomalyMode_NoScope_DoesNotFireOnNan()
    {
        // Without an active scope and without tape.DetectAnomaly set,
        // the tape must NOT throw — anomaly detection is strictly opt-in
        // and always off on the zero-cost hot path.
        using var tape = new GradientTape<float>();

        var x = new Tensor<float>(new[] { 0f }, new[] { 1 });
        var y = _engine.TensorLog(x);

        // Backward may or may not succeed numerically, but it must not
        // throw AnomalyDetectedException. Accept either a clean return
        // or a different exception (e.g. invalid-shape guard) — only
        // AnomalyDetectedException would indicate the scope fired.
        try
        {
            tape.ComputeGradients(y, new[] { x });
        }
        catch (AnomalyDetectedException)
        {
            Assert.Fail("Anomaly detection fired without a scope being active.");
        }
        catch
        {
            // Any other exception is fine for this test's contract.
        }
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
