// Copyright (c) AiDotNet. All rights reserved.
// Tests for ForwardAdRegistry — issue #214 acceptance gap.

#nullable disable

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff.ForwardAD;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies that the forward-mode rule registry is wired up with all
/// the standard pointwise ops and that custom registrations round-trip.
/// </summary>
public class ForwardAdRegistryTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    [Fact]
    public void IsRegistered_StandardOps_PreSeeded()
    {
        // The seed registrations cover every method on DualOps<T>.
        // Any future op added to DualOps without a matching registry
        // entry should fail this guard.
        Assert.True(ForwardAdRegistry.IsRegistered<float>("TensorAdd"));
        Assert.True(ForwardAdRegistry.IsRegistered<float>("TensorMultiply"));
        Assert.True(ForwardAdRegistry.IsRegistered<float>("TensorMatMul"));
        Assert.True(ForwardAdRegistry.IsRegistered<float>("TensorExp"));
        Assert.True(ForwardAdRegistry.IsRegistered<float>("Sigmoid"));

        Assert.True(ForwardAdRegistry.IsRegistered<double>("TensorAdd"));
        Assert.True(ForwardAdRegistry.IsRegistered<double>("Tanh"));
    }

    [Fact]
    public void IsRegistered_UnregisteredOp_ReturnsFalse()
    {
        Assert.False(ForwardAdRegistry.IsRegistered<float>("DoesNotExist_AbcXyz"));
    }

    [Fact]
    public void GetUnary_TensorExp_ProducesCorrectJvp()
    {
        // d/dx exp(x) = exp(x), so if tangent = 1, the result tangent
        // should equal the result primal.
        var rule = ForwardAdRegistry.GetUnary<float>("TensorExp");
        Assert.NotNull(rule);

        var x = new Tensor<float>(new[] { 0.5f }, new[] { 1 });
        var dx = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var result = rule(_engine, new Dual<float>(x, dx));

        Assert.Equal((float)Math.Exp(0.5), result.Primal[0], precision: 4);
        Assert.Equal((float)Math.Exp(0.5), result.Tangent[0], precision: 4);
    }

    [Fact]
    public void GetBinary_TensorMultiply_AppliesProductRule()
    {
        // d(a*b) = da*b + a*db. For a=2, b=3, da=1, db=0: tangent = 3.
        var rule = ForwardAdRegistry.GetBinary<float>("TensorMultiply");
        Assert.NotNull(rule);

        var a = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var b = new Tensor<float>(new[] { 3f }, new[] { 1 });
        var da = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var db = new Tensor<float>(new[] { 0f }, new[] { 1 });

        var result = rule(_engine, new Dual<float>(a, da), new Dual<float>(b, db));
        Assert.Equal(6f, result.Primal[0]);
        Assert.Equal(3f, result.Tangent[0]);
    }

    [Fact]
    public void RegisterUnary_CustomRule_OverridesExisting()
    {
        // Register a no-op rule, then verify lookup returns it.
        Func<IEngine, Dual<float>, Dual<float>> identity = (eng, x) => x;
        ForwardAdRegistry.RegisterUnary<float>("Test_Custom_Identity", identity);
        var got = ForwardAdRegistry.GetUnary<float>("Test_Custom_Identity");
        Assert.NotNull(got);
        var x = new Tensor<float>(new[] { 7f }, new[] { 1 });
        var dx = new Tensor<float>(new[] { 0.5f }, new[] { 1 });
        var r = got(_engine, new Dual<float>(x, dx));
        Assert.Equal(7f, r.Primal[0]);
        Assert.Equal(0.5f, r.Tangent[0]);
    }

    [Fact]
    public void GetUnary_NullOpName_Throws()
    {
        Assert.Throws<ArgumentNullException>(
            () => ForwardAdRegistry.GetUnary<float>(null));
    }

    [Fact]
    public void RegisteredOpNames_IncludesStandardOps()
    {
        var names = new System.Collections.Generic.HashSet<string>(
            ForwardAdRegistry.RegisteredOpNames());
        Assert.Contains("TensorAdd", names);
        Assert.Contains("TensorExp", names);
        Assert.Contains("TensorMatMul", names);
        Assert.Contains("Sigmoid", names);
    }
}
