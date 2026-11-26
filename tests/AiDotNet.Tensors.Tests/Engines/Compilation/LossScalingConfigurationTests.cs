// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The configuration loss scaling refuses: out-of-range options, a plan that is not float, and a non-finite gradient
/// policy that repairs or raises on the overflowed gradients a scaled step must skip (in either order of configuring).
/// </summary>
public class LossScalingConfigurationTests
{
    private sealed class PlanHandle : IDisposable
    {
        public PlanHandle(ICompiledTrainingPlanLossScaling scaling, INonFiniteGradientPolicyTarget policy, IDisposable owner)
        {
            Scaling = scaling;
            Policy = policy;
            Owner = owner;
        }

        public ICompiledTrainingPlanLossScaling Scaling { get; }
        public INonFiniteGradientPolicyTarget Policy { get; }
        public IDisposable Owner { get; }
        public void Dispose() => Owner.Dispose();
    }

    private static PlanHandle Plan<TT>()
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var x = new Tensor<TT>(new[] { 2, 3 });
            var w = new Tensor<TT>(new[] { 3, 2 });
            ICompiledTrainingPlan<TT> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorMatMul(x, w), null);
                plan = scope.CompileTraining(new[] { w });
            }
            return new PlanHandle((ICompiledTrainingPlanLossScaling)plan, (INonFiniteGradientPolicyTarget)plan, plan);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    public static TheoryData<string> BadOptions => new() { "scale0", "scaleInf", "growth", "backoff0", "backoff2", "interval", "min0", "maxBelowMin", "maxInf", "initialBelowMin", "initialAboveMax" };

    [Theory]
    [MemberData(nameof(BadOptions))]
    public void Out_of_range_options_are_refused(string which)
    {
        var options = new LossScalingOptions();
        switch (which)
        {
            case "scale0": options.InitialScale = 0f; break;
            case "scaleInf": options.InitialScale = float.PositiveInfinity; break;
            case "growth": options.GrowthFactor = 0.5f; break;
            case "backoff0": options.BackoffFactor = 0f; break;
            case "backoff2": options.BackoffFactor = 2f; break;
            case "interval": options.GrowthInterval = 0; break;
            case "min0": options.MinScale = 0f; break;
            case "maxBelowMin": options.MinScale = 4f; options.MaxScale = 2f; break;
            case "maxInf": options.MaxScale = float.PositiveInfinity; break;
            case "initialBelowMin": options.MinScale = 16f; options.InitialScale = 8f; break;
            case "initialAboveMax": options.MaxScale = 1024f; options.InitialScale = 2048f; break;
        }
        using var plan = Plan<float>();
        Assert.Throws<ArgumentOutOfRangeException>(() => plan.Scaling.ConfigureLossScaling(options));
        Assert.Null(plan.Scaling.ReadLossScaleState());
    }

    [Theory]
    [InlineData(0.5f)]
    [InlineData(1e9f)]
    public void A_resumed_scale_outside_the_bounds_is_refused(float scale)
    {
        using var plan = Plan<float>();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            plan.Scaling.ConfigureLossScaling(new LossScalingOptions(), new LossScaleState(scale, 0, 0, 0)));
        Assert.Null(plan.Scaling.ReadLossScaleState());
    }

    [Fact]
    public void A_double_plan_is_refused()
    {
        using var plan = Plan<double>();
        Assert.Throws<NotSupportedException>(() => plan.Scaling.ConfigureLossScaling(new LossScalingOptions()));
    }

    [Fact]
    public void A_repairing_non_finite_policy_is_refused_in_either_order()
    {
        using (var plan = Plan<float>())
        {
            plan.Policy.SetNonFiniteGradientPolicy(NonFiniteGradientPolicy.ZeroNonFinite);
            Assert.Throws<InvalidOperationException>(() => plan.Scaling.ConfigureLossScaling(new LossScalingOptions()));
        }
        using (var plan = Plan<float>())
        {
            plan.Scaling.ConfigureLossScaling(new LossScalingOptions { InitialScale = 8f });
            Assert.Throws<InvalidOperationException>(() => plan.Policy.SetNonFiniteGradientPolicy(NonFiniteGradientPolicy.Error));
            Assert.Equal((float?)8f, plan.Scaling.ReadLossScaleState()?.Scale);
            plan.Scaling.ConfigureLossScaling(null);
            Assert.Null(plan.Scaling.ReadLossScaleState());
        }
    }
}
