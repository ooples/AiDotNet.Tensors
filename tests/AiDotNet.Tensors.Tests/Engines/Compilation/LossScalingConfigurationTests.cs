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
        public ICompiledTrainingPlanLossScaling Scaling = null!;
        public INonFiniteGradientPolicyTarget Policy = null!;
        public IDisposable Owner = null!;
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
            return new PlanHandle
            {
                Scaling = (ICompiledTrainingPlanLossScaling)plan,
                Policy = (INonFiniteGradientPolicyTarget)plan,
                Owner = plan,
            };
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    public static TheoryData<string> BadOptions => new() { "scale0", "scaleInf", "growth", "backoff0", "backoff2", "interval", "min0", "maxBelowMin", "maxInf" };

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
        }
        using var plan = Plan<float>();
        Assert.Throws<ArgumentOutOfRangeException>(() => plan.Scaling.ConfigureLossScaling(options));
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
            Assert.Equal(8f, plan.Scaling.ReadLossScaleState()!.Value.Scale);
            plan.Scaling.ConfigureLossScaling(null);
            Assert.Null(plan.Scaling.ReadLossScaleState());
        }
    }
}
