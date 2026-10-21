using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// <see cref="GradientDeferral"/> carries a compiled GPU step's parked gradient contributions, and a contribution it
/// loses is a wrong gradient rather than a slower step: the consuming backward sums what it takes, so a take that
/// returns nothing silently drops that contribution. These run on the CPU and need no GPU, because what they hold is
/// the deferral's own bookkeeping and its scope - one engine serves every plan compiled on it, so the state is the
/// running thread's, not the engine's (PR #1490 review).
/// </summary>
public class GradientDeferralTests
{
    private static Tensor<float> Grad(int n) => new Tensor<float>([n]);

    [Fact]
    public void A_parked_contribution_is_taken_once()
    {
        var tensor = Grad(4);
        var contribution = Grad(4);
        var deferral = new GradientDeferral(new object[] { tensor });

        Assert.True(deferral.Accepts(tensor));
        Assert.True(deferral.TryPark(tensor, contribution));
        Assert.Same(contribution, deferral.Take(tensor));
        Assert.Null(deferral.Take(tensor));
    }

    [Fact]
    public void A_second_contribution_for_the_same_tensor_is_refused()
    {
        var tensor = Grad(4);
        var first = Grad(4);
        var deferral = new GradientDeferral(new object[] { tensor });

        Assert.True(deferral.TryPark(tensor, first));
        Assert.False(deferral.TryPark(tensor, Grad(4)));
        // The refusal must leave the first contribution takeable: the caller adds the one it was refused.
        Assert.Same(first, deferral.Take(tensor));
    }

    [Fact]
    public void Only_the_tensors_the_plan_named_are_accepted()
    {
        var named = Grad(4);
        var other = Grad(4);
        var deferral = new GradientDeferral(new object[] { named });

        Assert.Equal(1, deferral.DeferrableCount);
        Assert.True(deferral.Accepts(named));
        Assert.False(deferral.Accepts(other));
        // By reference, not by value: an equal-shaped tensor is a different gradient.
        Assert.Null(deferral.Take(other));
    }

    [Fact]
    public void Reset_drops_what_a_step_that_threw_left_parked()
    {
        var tensor = Grad(4);
        var deferral = new GradientDeferral(new object[] { tensor });
        Assert.True(deferral.TryPark(tensor, Grad(4)));

        deferral.Reset();

        Assert.Null(deferral.Take(tensor));
        Assert.True(deferral.TryPark(tensor, Grad(4)), "a reset deferral must take a contribution again");
    }

    /// <summary>
    /// The isolation between concurrently stepping plans: one engine serves every plan compiled on it, so a deferral
    /// kept on the engine would be overwritten - and then cleared - by a second plan's step while the first was still
    /// running, and the first plan's parked contribution would be taken by nothing. <see cref="GradientDeferral.Current"/>
    /// is the running thread's.
    /// </summary>
    [Fact]
    public void A_deferral_current_on_one_thread_is_invisible_to_another()
    {
        var mine = new GradientDeferral(new object[] { Grad(4) });
        var theirs = new GradientDeferral(new object[] { Grad(4) });
        GradientDeferral? seenByOther = null;
        GradientDeferral? setByOther = null;
        var prior = GradientDeferral.Current;
        try
        {
            GradientDeferral.Current = mine;
            var other = new Thread(() =>
            {
                seenByOther = GradientDeferral.Current;
                GradientDeferral.Current = theirs;
                setByOther = GradientDeferral.Current;
            });
            other.Start();
            Assert.True(other.Join(TimeSpan.FromSeconds(30)), "the second thread did not finish");

            Assert.Null(seenByOther);
            Assert.Same(theirs, setByOther);
            Assert.Same(mine, GradientDeferral.Current);
        }
        finally
        {
            GradientDeferral.Current = prior;
        }
    }
}
