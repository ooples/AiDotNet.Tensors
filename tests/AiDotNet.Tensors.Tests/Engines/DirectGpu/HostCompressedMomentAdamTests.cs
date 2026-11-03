using System;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;
namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;
public class HostCompressedMomentAdamTests
{
    private const float Lr = 1e-2f, Beta1 = 0.9f, Beta2 = 0.99f, Eps = 1e-8f, WeightDecay = 0.1f;
    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void HostSteps_RejectNonfiniteHyperparametersWithoutMutation(float invalid)
    {
        foreach (bool bf16 in new[] { true, false })
        foreach (string target in new[] { "lr", "beta1", "beta2", "epsilon", "weightDecay" })
        {
            var p = new[] { 3f };
            var m = new byte[] { 1, 2 };
            var v = new byte[] { 3, 4 };
            var ms = new[] { 5f };
            var vs = new[] { 6f };
            float lr = target == "lr" ? invalid : Lr;
            float b1 = target == "beta1" ? invalid : Beta1;
            float b2 = target == "beta2" ? invalid : Beta2;
            float eps = target == "epsilon" ? invalid : Eps;
            float wd = target == "weightDecay" ? invalid : WeightDecay;
            var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                if (bf16) HostCompressedMomentAdam.Bf16Step(p, new[] { 1f }, m, v, 1,
                    lr, b1, b2, eps, wd, 1, true);
                else HostCompressedMomentAdam.Int8Step(p, new[] { 1f }, m, v, ms, vs, 1, 64,
                    lr, b1, b2, eps, wd, 1, true);
            });
            Assert.Equal(target, error.ParamName);
            Assert.Equal(new[] { 3f }, p);
            Assert.Equal(new byte[] { 1, 2 }, m);
            Assert.Equal(new byte[] { 3, 4 }, v);
            Assert.Equal(new[] { 5f }, ms);
            Assert.Equal(new[] { 6f }, vs);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HostSteps_RejectNegativeDecayWithoutMutation(bool bf16)
    {
        var p = new[] { 3f }; var m = new byte[] { 1, 2 }; var v = new byte[] { 3, 4 };
        var ms = new[] { 5f }; var vs = new[] { 6f };
        var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            if (bf16) HostCompressedMomentAdam.Bf16Step(p, new[] { 1f }, m, v, 1,
                Lr, Beta1, Beta2, Eps, -0.1f, 1, true);
            else HostCompressedMomentAdam.Int8Step(p, new[] { 1f }, m, v, ms, vs, 1, 64,
                Lr, Beta1, Beta2, Eps, -0.1f, 1, true);
        });
        Assert.Equal("weightDecay", error.ParamName);
        Assert.Equal(new[] { 3f }, p);
        Assert.Equal(new byte[] { 1, 2 }, m); Assert.Equal(new byte[] { 3, 4 }, v);
        Assert.Equal(new[] { 5f }, ms); Assert.Equal(new[] { 6f }, vs);
    }

}
