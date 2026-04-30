// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>Issue #276 sub-feature 3 follow-up: QAT (quantization-aware
/// training) — fake-quantize round-trip + optimizer step + export.</summary>
public class QatTrainingTests
{
    [Fact]
    public void Int8_FakeQuantize_RoundTrip_StaysWithinScale()
    {
        var hook = new QatTrainingHook(QuantizationBits.Int8, groupSize: 32);
        var w = new Tensor<float>(new[] { 32 });
        var rng = new Random(42);
        for (int i = 0; i < 32; i++) w[i] = (float)(rng.NextDouble() * 4 - 2);

        var fakeQ = hook.RegisterFloatMaster("w", w);
        // Fake-quantize: every value perturbed by at most scale/2.
        for (int i = 0; i < 32; i++)
        {
            float orig = w[i];
            float fq = fakeQ[i];
            // Group-symmetric int8: max scale ≈ absmax / 127. Loose bound.
            Assert.True(Math.Abs(fq - orig) < Math.Abs(orig) * 0.05f + 0.05f);
        }
    }

    [Fact]
    public void Int4_FakeQuantize_RoundTrip_StaysWithinScale()
    {
        var hook = new QatTrainingHook(QuantizationBits.Int4, groupSize: 32);
        var w = new Tensor<float>(new[] { 32 });
        var rng = new Random(7);
        for (int i = 0; i < 32; i++) w[i] = (float)(rng.NextDouble() * 4 - 2);

        var fakeQ = hook.RegisterFloatMaster("w", w);
        for (int i = 0; i < 32; i++)
        {
            float orig = w[i];
            float fq = fakeQ[i];
            // int4 = 4-bit symmetric range [-7,7] → looser tolerance.
            Assert.True(Math.Abs(fq - orig) < Math.Abs(orig) * 0.5f + 0.5f);
        }
    }

    [Fact]
    public void OptimizerStep_UpdatesFloatMaster()
    {
        var hook = new QatTrainingHook(QuantizationBits.Int8, groupSize: 32);
        var w = new Tensor<float>(new[] { 32 });
        for (int i = 0; i < 32; i++) w[i] = 1f;
        hook.RegisterFloatMaster("w", w);
        var grad = new float[32];
        for (int i = 0; i < 32; i++) grad[i] = 0.5f;
        hook.OptimizerStep("w", grad, learningRate: 0.1f);
        // Master should now be 1 - 0.1 * 0.5 = 0.95 everywhere.
        var fq = hook.FakeQuantize("w");
        for (int i = 0; i < 32; i++) Assert.Equal(0.95f, fq[i], 1);
    }

    [Fact]
    public void ExportInt8_AfterTraining_RoundTripsCleanly()
    {
        var hook = new QatTrainingHook(QuantizationBits.Int8, groupSize: 32);
        var w = new Tensor<float>(new[] { 32 });
        var rng = new Random(101);
        for (int i = 0; i < 32; i++) w[i] = (float)(rng.NextDouble() * 4 - 2);
        hook.RegisterFloatMaster("w", w);
        var qt = hook.ExportInt8("w");
        Assert.Equal(QuantizationBits.Int8, qt.Bits);
        Assert.Equal(32, qt.Length);
    }
}
