// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Host arrays that the compiled CPU training plans of one group co-own: the arrays a plan's step only writes and then
/// reads within the step (its forward activations and intermediate gradients). Plans whose steps never overlap -- a
/// GAN's discriminator and generator -- then hold the larger of their working sets, not the sum, as one caching
/// allocator does for PyTorch. The host counterpart of the device's <c>SharedStepScratch</c>.
/// </summary>
/// <remarks>
/// <para>A plan compiled while a group is entered (<see cref="Enter"/>) takes each step-transient array from the group:
/// one of exactly the length it needs that it does not already own, or a new one the group then offers to the others.
/// Exact, because a kernel handed a tensor's backing array may work over the whole array: a longer one cost the work of
/// its length (a DCGAN generator activation bound to a discriminator array four times its size ran its step 1.7x
/// slower). The two networks' activations still coincide in size wherever batch times channels does. Every co-owner
/// treats the array as scratch: each step
/// writes a forward output before reading it and re-zeroes a pooled gradient before accumulating into it, and
/// parameters, gradients the caller reads, the loss, retained outputs, inputs and in-place state never come from here.
/// So nothing one plan leaves in an array is read by another.</para>
/// <para>The caller promises that the group's plans step one at a time, whatever the thread (a model's training is not
/// concurrent). An array stays with the group while any co-owner lives; once every owner has been collected it is
/// handed to the next plan that needs one.</para>
/// </remarks>
internal sealed class HostStepScratchGroup
{
    [ThreadStatic] private static HostStepScratchGroup? t_current;

    /// <summary>The group plans compiled on this thread now join, or null.</summary>
    internal static HostStepScratchGroup? Current => t_current;

    private sealed class Entry
    {
        internal Entry(Array array) => Array = array;
        internal Array Array { get; }
        internal List<WeakReference<object>> Owners { get; } = new(2);
    }

    private readonly object _lock = new();
    private readonly List<Entry> _entries = new();

    /// <summary>Arrays the group holds, their bytes, and how many requests an existing array served (tests).</summary>
    internal int Count { get { lock (_lock) return _entries.Count; } }
    internal long Bytes
    {
        get
        {
            lock (_lock)
            {
                long bytes = 0;
                foreach (var entry in _entries)
                    bytes += (long)entry.Array.Length * Helpers.TensorArena.ElementBytes(entry.Array.GetType().GetElementType()!);
                return bytes;
            }
        }
    }
    internal int Shares { get; private set; }

    /// <summary>Makes this the group plans compiled on this thread join until the returned scope is disposed.</summary>
    internal Scope Enter()
    {
        var previous = t_current;
        t_current = this;
        return new Scope(previous);
    }

    internal readonly struct Scope : IDisposable
    {
        private readonly HostStepScratchGroup? _previous;
        internal Scope(HostStepScratchGroup? previous) => _previous = previous;
        public void Dispose() => t_current = _previous;
    }

    /// <summary>
    /// An array of exactly <paramref name="length"/> elements for <paramref name="owner"/>'s step scratch: one the group
    /// holds that <paramref name="owner"/> does not, or a new one. Its contents are whatever its last user left.
    /// </summary>
    internal TElement[] Take<TElement>(int length, object owner)
    {
        if (TryTakeShared<TElement>(length, owner, out var shared)) return shared;
        var fresh = new TElement[length];
        Adopt(fresh, owner);
        return fresh;
    }

    /// <summary>
    /// One of the group's arrays of exactly <paramref name="length"/> elements that <paramref name="owner"/> does not
    /// hold yet, now co-owned by it; false when there is none. <paramref name="inUse"/> names arrays the owner's tensors
    /// already use outside the group, which it must not be handed a second time.
    /// </summary>
    internal bool TryTakeShared<TElement>(int length, object owner, out TElement[] array, ISet<object>? inUse = null)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        lock (_lock)
        {
            foreach (var entry in _entries)
            {
                if (entry.Array is not TElement[] candidate || candidate.Length != length) continue;
                if (OwnedBy(entry, owner) || inUse?.Contains(candidate) == true) continue;
                entry.Owners.Add(new WeakReference<object>(owner));
                Shares++;
                array = candidate;
                return true;
            }
        }
        array = null!;
        return false;
    }

    /// <summary>
    /// Offers <paramref name="array"/>, which <paramref name="owner"/> already uses as step scratch, to the group's other
    /// plans: a plan keeps the array it was traced with when no other plan's fits, rather than leaving it for the
    /// collector and taking a new one.
    /// </summary>
    internal void Adopt(Array array, object owner)
    {
        if (array is null) throw new ArgumentNullException(nameof(array));
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        lock (_lock)
        {
            foreach (var entry in _entries)
                if (ReferenceEquals(entry.Array, array))
                {
                    if (!OwnedBy(entry, owner)) entry.Owners.Add(new WeakReference<object>(owner));
                    return;
                }
            var created = new Entry(array);
            created.Owners.Add(new WeakReference<object>(owner));
            _entries.Add(created);
        }
    }

    // Whether owner co-owns the entry. Drops the references of collected owners on the way; an entry with none left is
    // free, and the next request takes it.
    private static bool OwnedBy(Entry entry, object owner)
    {
        bool owned = false;
        for (int i = entry.Owners.Count - 1; i >= 0; i--)
        {
            if (!entry.Owners[i].TryGetTarget(out var target)) { entry.Owners.RemoveAt(i); continue; }
            if (ReferenceEquals(target, owner)) owned = true;
        }
        return owned;
    }
}
