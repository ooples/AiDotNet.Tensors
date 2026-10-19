using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The registry that keeps a parameter shared by two compiled plans on one authoritative device buffer: at most one
/// live plan's ping-pong update claims it, and every claim change moves the epoch the plans re-check against.
/// </summary>
public class PingPongParameterClaimsTests
{
    private sealed class Owner : IPingPongOwner
    {
        public int Yields;
        public void YieldPingPong() => Yields++;
    }

    [Fact]
    public void A_claim_is_exclusive_until_its_owner_releases_it()
    {
        var a = new Owner();
        var b = new Owner();
        object[] shared = { new float[4], new float[8] };
        object[] other = { new float[2] };

        Assert.True(PingPongParameterClaims.TryClaim(a, shared));
        Assert.True(PingPongParameterClaims.TryClaim(a, shared));   // the owner re-claiming its own keys
        Assert.False(PingPongParameterClaims.TryClaim(b, new[] { shared[1] }));
        Assert.True(PingPongParameterClaims.TryClaim(b, other));

        Assert.Equal(new IPingPongOwner[] { a }, PingPongParameterClaims.OthersClaiming(b, shared));
        Assert.Empty(PingPongParameterClaims.OthersClaiming(a, shared));
        Assert.Equal(new IPingPongOwner[] { b }, PingPongParameterClaims.OthersClaiming(a, new[] { shared[0], other[0] }));

        PingPongParameterClaims.Release(b, shared);   // not b's: a keeps them
        Assert.Equal(new IPingPongOwner[] { a }, PingPongParameterClaims.OthersClaiming(b, shared));
        PingPongParameterClaims.Release(a, shared);
        Assert.Empty(PingPongParameterClaims.OthersClaiming(b, shared));
        Assert.True(PingPongParameterClaims.TryClaim(b, shared));
        PingPongParameterClaims.Release(b, shared);
        PingPongParameterClaims.Release(b, other);
        Assert.Equal(0, a.Yields + b.Yields);   // the registry only reports owners; the plans make them yield
    }

    [Fact]
    public void Every_claim_refusal_and_release_moves_the_epoch()
    {
        var a = new Owner();
        var b = new Owner();
        object[] keys = { new float[3] };
        long e0 = PingPongParameterClaims.Epoch;
        Assert.True(PingPongParameterClaims.TryClaim(a, keys));
        long e1 = PingPongParameterClaims.Epoch;
        Assert.True(e1 > e0);
        // A refused claim moves it too: the owner's next step must re-check and yield to the plan that was refused.
        Assert.False(PingPongParameterClaims.TryClaim(b, keys));
        long e2 = PingPongParameterClaims.Epoch;
        Assert.True(e2 > e1);
        PingPongParameterClaims.Release(a, keys);
        Assert.True(PingPongParameterClaims.Epoch > e2);
    }

    [Fact]
    public void A_collected_owner_no_longer_holds_its_claims()
    {
        object[] keys = { new float[5] };
        ClaimWithAnOwnerThatDies(keys);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var b = new Owner();
        Assert.Empty(PingPongParameterClaims.OthersClaiming(b, keys));
        Assert.True(PingPongParameterClaims.TryClaim(b, keys));
        PingPongParameterClaims.Release(b, keys);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ClaimWithAnOwnerThatDies(object[] keys) => Assert.True(PingPongParameterClaims.TryClaim(new Owner(), keys));
}
