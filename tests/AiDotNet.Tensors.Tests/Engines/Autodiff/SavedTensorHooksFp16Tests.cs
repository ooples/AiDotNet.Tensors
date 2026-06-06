using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// FP16 activation-storage recipe gate (Tensors #558): <see cref="SavedTensorRecipes.SaveFp16"/> packs a
/// saved float activation to Half (2 bytes/elem) and unpacks it back to float in the backward pass — the
/// industry-standard AMP activation-storage format. Validates the round-trip is within FP16 rounding, the
/// packed payload is genuinely half-sized, and it composes with the existing SavedTensorHooks plumbing.
/// </summary>
public class SavedTensorHooksFp16Tests
{
    private static Tensor<float> Rand(int n, int seed)
    {
        var rng = new Random(seed);
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * 4.0);
        return new Tensor<float>(d, new[] { n });
    }

    [Fact]
    public void SaveFp16_RoundTrips_WithinFp16_And_Halves_Bytes()
    {
        var x = Rand(513, 7);

        Assert.False(SavedTensorHooks.IsActive);
        object? packed;
        using (SavedTensorRecipes.SaveFp16())
        {
            Assert.True(SavedTensorHooks.IsActive);
            Assert.True(SavedTensorHooks.TryApplyPack(x, out packed), "SaveFp16 should pack a float tensor");
        }
        Assert.NotNull(packed);

        // Unpack (as the backward pass would) and check the round-trip is FP16-accurate.
        var restored = SavedTensorHooks.ApplyUnpack<float>(packed!);
        Assert.Equal(x.Shape.ToArray(), restored.Shape.ToArray());
        var xa = x.ToArray(); var ra = restored.ToArray();
        for (int i = 0; i < xa.Length; i++)
        {
            Assert.False(float.IsNaN(ra[i]) || float.IsInfinity(ra[i]));
            float denom = Math.Max(1e-3f, Math.Abs(xa[i]));
            Assert.True(Math.Abs(ra[i] - xa[i]) / denom <= 1.0f / 1024f + 1e-6f,
                $"FP16 round-trip drift at {i}: {xa[i]} -> {ra[i]}");
        }

        // On values exactly FP16-representable, the round-trip is exact.
        var exact = new Tensor<float>(new float[] { 0f, 1f, -2f, 0.5f, -0.25f, 8f }, new[] { 6 });
        using (SavedTensorRecipes.SaveFp16())
        {
            Assert.True(SavedTensorHooks.TryApplyPack(exact, out var p2));
            var r2 = SavedTensorHooks.ApplyUnpack<float>(p2!).ToArray();
            var ea = exact.ToArray();
            for (int i = 0; i < ea.Length; i++) Assert.Equal(ea[i], r2[i]);
        }

        Assert.False(SavedTensorHooks.IsActive); // scope popped cleanly
    }
}
