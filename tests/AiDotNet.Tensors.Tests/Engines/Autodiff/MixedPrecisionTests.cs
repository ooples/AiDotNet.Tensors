// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Issue #276 sub-feature 1 follow-up: mixed-precision training
/// — loss scaling + master weights round-trip.</summary>
public class MixedPrecisionTests
{
    [Fact]
    public void GradScaler_ScaleLoss_MultipliesByScale()
    {
        var scaler = new GradScaler(new MixedPrecisionConfig { LossScale = 65536f });
        var loss = new Tensor<float>(new[] { 3 });
        loss[0] = 1f; loss[1] = 2f; loss[2] = 3f;
        var scaled = scaler.ScaleLoss(loss);
        Assert.Equal(65536f, scaled[0]);
        Assert.Equal(131072f, scaled[1]);
        Assert.Equal(196608f, scaled[2]);
    }

    [Fact]
    public void GradScaler_UnscaleGradients_DividesByScale()
    {
        var scaler = new GradScaler(new MixedPrecisionConfig { LossScale = 1024f });
        var x = new Tensor<float>(new[] { 4 });
        var grad = new Tensor<float>(new[] { 4 });
        var grads = new Dictionary<Tensor<float>, Tensor<float>> { { x, grad } };
        for (int i = 0; i < 4; i++) grad[i] = 2048f * (i + 1); // scaled values
        bool overflow = scaler.UnscaleGradients(grads);
        Assert.False(overflow);
        for (int i = 0; i < 4; i++) Assert.Equal(2f * (i + 1), grad[i], 4);
    }

    [Fact]
    public void GradScaler_UnscaleGradients_DetectsInfNan()
    {
        var scaler = new GradScaler(new MixedPrecisionConfig { LossScale = 1f });
        var x = new Tensor<float>(new[] { 2 });
        var grad = new Tensor<float>(new[] { 2 });
        grad[0] = float.PositiveInfinity;
        grad[1] = 1f;
        bool overflow = scaler.UnscaleGradients(new Dictionary<Tensor<float>, Tensor<float>> { { x, grad } });
        Assert.True(overflow);
    }

    [Fact]
    public void GradScaler_DynamicSchedule_BackoffOnOverflow()
    {
        var scaler = new GradScaler(new MixedPrecisionConfig
        {
            LossScale = 1024f, GrowthInterval = 5, BackoffFactor = 0.5f, GrowthFactor = 2.0f,
        });
        Assert.Equal(1024f, scaler.Scale);
        scaler.Update(foundInfNan: true);
        Assert.Equal(512f, scaler.Scale);
    }

    [Fact]
    public void GradScaler_DynamicSchedule_GrowthAfterInterval()
    {
        var scaler = new GradScaler(new MixedPrecisionConfig
        {
            LossScale = 1024f, GrowthInterval = 3, BackoffFactor = 0.5f, GrowthFactor = 2.0f,
        });
        scaler.Update(false); scaler.Update(false); scaler.Update(false);
        Assert.Equal(2048f, scaler.Scale);
    }

    [Fact]
    public void MasterWeights_RoundTrip_FpMaster()
    {
        var mw = new MasterWeights();
        mw.Register("w", new float[] { 1f, 2f, 3f, 4f });
        var master = mw.GetMaster("w");
        Assert.Equal(4, master.Length);
        Assert.Equal(1f, master[0]);
        bool writeBackCalled = false;
        mw.UpdateMaster("w",
            opt => { for (int i = 0; i < opt.Length; i++) opt[i] -= 0.1f; },
            wb => { writeBackCalled = true; Assert.Equal(0.9f, wb[0], 4); });
        Assert.True(writeBackCalled);
    }
}
