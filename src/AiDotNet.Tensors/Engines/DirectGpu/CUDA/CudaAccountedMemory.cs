// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines.DevicePrimitives;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Device memory this library allocates beside a backend's buffers -- the cuBLASLt autotune scratch, the cuRAND
/// output buffers, the direct-PTX runtime's buffers, the NCCL staging buffers, the step arena's backing blocks --
/// counted on the device's <see cref="DeviceMemoryAccount"/> and admitted under its <see cref="GpuMemoryBudget"/>
/// like a buffer. Without this they are memory the process holds that neither the statistics nor the budget see.
/// </summary>
/// <remarks>
/// The account is found from the context current on the calling thread, which the backend registered. A context no
/// backend registered (a caller's own) allocates uncounted, as before.
/// </remarks>
internal static class CudaAccountedMemory
{
    private sealed class Owner
    {
        public Owner(DeviceMemoryAccount account, Action reclaim, object token)
        {
            Account = account;
            Reclaim = reclaim;
            Token = token;
        }

        public DeviceMemoryAccount Account { get; }
        public Action Reclaim { get; }
        public object Token { get; }
    }

    // Every live registration of a context, oldest first: backends on one device share its primary context, and the
    // last one registered answers for it until it is disposed, when the one before it does again.
    private static readonly Dictionary<IntPtr, List<Owner>> s_contexts = new();
    private static readonly object s_contextsLock = new object();
    private static int s_registrations;
    private static readonly ConcurrentDictionary<IntPtr, (DeviceMemoryAccount Account, long Bytes)> s_live = new();

    /// <summary>
    /// Names the account of <paramref name="context"/> and the release its budget runs before refusing an allocation.
    /// <paramref name="reclaim"/> must not keep the backend alive (this table is static). <paramref name="token"/>
    /// identifies the registration to <see cref="Unregister"/>.
    /// </summary>
    internal static void Register(IntPtr context, DeviceMemoryAccount account, Action reclaim, object token)
    {
        if (context == IntPtr.Zero) return;
        lock (s_contextsLock)
        {
            if (!s_contexts.TryGetValue(context, out var owners))
            {
                owners = new List<Owner>(1);
                s_contexts[context] = owners;
            }

            owners.Add(new Owner(account, reclaim, token));
            s_registrations++;
        }
    }

    /// <summary>Removes the registration <paramref name="token"/> made. Another backend still registered on the
    /// context keeps answering for it.</summary>
    internal static void Unregister(IntPtr context, object token)
    {
        if (context == IntPtr.Zero) return;
        lock (s_contextsLock)
        {
            if (!s_contexts.TryGetValue(context, out var owners)) return;
            int removed = owners.RemoveAll(owner => ReferenceEquals(owner.Token, token));
            s_registrations -= removed;
            if (owners.Count == 0) s_contexts.Remove(context);
        }
    }

    /// <summary>Live registrations of <paramref name="context"/> (tests).</summary>
    internal static int RegistrationCount(IntPtr context)
    {
        lock (s_contextsLock) return s_contexts.TryGetValue(context, out var owners) ? owners.Count : 0;
    }

    private static Owner? Current()
    {
        if (Volatile.Read(ref s_registrations) == 0) return null;
        return CudaNativeBindings.cuCtxGetCurrent(out IntPtr context) == CudaResult.Success ? OwnerOf(context) : null;
    }

    private static Owner? OwnerOf(IntPtr context)
    {
        lock (s_contextsLock)
            return s_contexts.TryGetValue(context, out var owners) && owners.Count > 0 ? owners[owners.Count - 1] : null;
    }

    /// <summary>The account answering for <paramref name="context"/> now, or null (tests).</summary>
    internal static DeviceMemoryAccount? AccountOf(IntPtr context) => OwnerOf(context)?.Account;

    /// <summary>
    /// <c>cuMemAlloc</c> on the current context, counted on its device and refused with
    /// <see cref="CudaResult.OutOfMemory"/> past the device's budget (after the backend's release).
    /// </summary>
    internal static CudaResult Alloc(out IntPtr pointer, ulong bytes)
    {
        pointer = IntPtr.Zero;
        if (Current() is not { } owner || bytes > long.MaxValue)
            return CudaNativeBindings.cuMemAlloc(out pointer, bytes);
        if (!owner.Account.TryAdmit((long)bytes, owner.Reclaim, out _)) return CudaResult.OutOfMemory;
        var result = CudaNativeBindings.cuMemAlloc(out pointer, bytes);
        if (result != CudaResult.Success)
        {
            DeviceMemoryAccount.SettleAdmission();
            return result;
        }

        Track(pointer, owner.Account, (long)bytes);
        return result;
    }

    /// <summary><c>cuMemFree</c>, returning the bytes <see cref="Alloc"/> counted for the pointer.</summary>
    internal static CudaResult Free(IntPtr pointer)
    {
        // Uncounted before the free: once freed, the driver may hand the same address to another thread's allocation.
        Untrack(pointer);
        return CudaNativeBindings.cuMemFree(pointer);
    }

    /// <summary>Counts memory a backend allocated itself (and already admitted) as reserved and in use.</summary>
    internal static void Track(IntPtr pointer, DeviceMemoryAccount account, long bytes)
    {
        if (pointer == IntPtr.Zero || bytes <= 0) return;
        account.Reserve(bytes);
        account.Acquire(bytes);
        s_live[pointer] = (account, bytes);
    }

    /// <summary>Returns what <see cref="Track"/> counted for <paramref name="pointer"/>; nothing when it counted
    /// nothing.</summary>
    internal static void Untrack(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero || !s_live.TryRemove(pointer, out var held)) return;
        held.Account.Relinquish(held.Bytes);
        held.Account.Unreserve(held.Bytes);
    }
}
