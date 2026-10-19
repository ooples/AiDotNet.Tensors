// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>A compiled plan that can hand its ping-pong parameters back (see <see cref="PingPongParameterClaims"/>).</summary>
internal interface IPingPongOwner
{
    /// <summary>Brings the current weights back into the shared buffers and updates in place from now on.</summary>
    void YieldPingPong();
}

/// <summary>
/// Which compiled plan's ping-pong update currently owns each parameter's storage. A ping-pong plan binds its
/// parameters to a second, plan-private buffer set on every other step. Another plan training the same parameters
/// (one per batch shape, or one continuing this plan's optimizer) resolves its own device buffers from those bindings,
/// so it could read or update a buffer the owner is about to overwrite or free. Parameters keep one authoritative
/// buffer instead: a plan stepping or configuring on parameters another plan claims makes that plan yield, and both
/// update in place from then on.
/// </summary>
internal static class PingPongParameterClaims
{
    private sealed class Claim
    {
        public Claim(IPingPongOwner owner) => Owner = new WeakReference<IPingPongOwner>(owner);
        public WeakReference<IPingPongOwner> Owner { get; }
    }

    private static readonly object s_lock = new();
    private static readonly ConditionalWeakTable<object, Claim> s_claims = new();
    private static long s_epoch;

    /// <summary>Advances on every claim, refused claim and release: a plan re-checks its parameters only after one.</summary>
    public static long Epoch => Interlocked.Read(ref s_epoch);

    /// <summary>Claims every key for <paramref name="owner"/>; false (nothing claimed) when another live owner holds one.</summary>
    public static bool TryClaim(IPingPongOwner owner, IReadOnlyList<object> keys)
    {
        lock (s_lock)
        {
            Interlocked.Increment(ref s_epoch);   // refused or not, every plan on these parameters re-checks
            foreach (var key in keys)
                if (s_claims.TryGetValue(key, out var claim) && claim.Owner.TryGetTarget(out var holder) && !ReferenceEquals(holder, owner))
                    return false;
            foreach (var key in keys)
            {
                s_claims.Remove(key);
                s_claims.Add(key, new Claim(owner));
            }
            return true;
        }
    }

    /// <summary>Drops every claim <paramref name="owner"/> (or an owner since collected) holds on <paramref name="keys"/>.</summary>
    public static void Release(IPingPongOwner owner, IReadOnlyList<object> keys)
    {
        lock (s_lock)
        {
            foreach (var key in keys)
                if (s_claims.TryGetValue(key, out var claim) && (!claim.Owner.TryGetTarget(out var holder) || ReferenceEquals(holder, owner)))
                    s_claims.Remove(key);
            Interlocked.Increment(ref s_epoch);
        }
    }

    /// <summary>The live owners other than <paramref name="self"/> that claim any of <paramref name="keys"/>.</summary>
    public static List<IPingPongOwner> OthersClaiming(IPingPongOwner self, IReadOnlyList<object> keys)
    {
        var others = new List<IPingPongOwner>();
        lock (s_lock)
        {
            foreach (var key in keys)
                if (s_claims.TryGetValue(key, out var claim) && claim.Owner.TryGetTarget(out var holder)
                    && !ReferenceEquals(holder, self) && !others.Contains(holder))
                    others.Add(holder);
        }
        return others;
    }
}
