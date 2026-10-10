using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// CPU last-level-cache (L3) topology detection + thread pinning for the CCX-aware GEMM. On Zen each L3
/// domain is a CCX; keeping a thread-group's packed B-panel in ITS OWN L3 (reused by that CCX's threads)
/// avoids the cross-CCX Infinity-Fabric / DRAM re-reads that cap the flat per-tile scheme. Windows-only for
/// now (returns no domains elsewhere ⇒ callers fall back to the non-pinned per-tile path, still correct).
/// </summary>
internal static class CpuTopology
{
    [StructLayout(LayoutKind.Sequential)]
    private struct GROUP_AFFINITY { public nuint Mask; public ushort Group; public ushort R0, R1, R2; }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int relationshipType, IntPtr buffer, ref uint returnedLength);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY ga, IntPtr prev);

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESSOR_NUMBER { public ushort Group; public byte Number; public byte Reserved; }
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetThreadIdealProcessorEx(IntPtr hThread, ref PROCESSOR_NUMBER ideal, IntPtr previous);
    [DllImport("kernel32.dll")]
    private static extern void GetCurrentProcessorNumberEx(out PROCESSOR_NUMBER number);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessAffinityMask(IntPtr hProcess, out nuint processMask, out nuint systemMask);
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")]
    private static extern uint GetActiveProcessorCount(ushort groupNumber);

    /// <summary>Logical processors of the whole machine, across every processor group (null off Windows).</summary>
    internal static int? MachineLogicalProcessorCount()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return null;
        try
        {
            const ushort AllProcessorGroups = 0xFFFF;
            uint count = GetActiveProcessorCount(AllProcessorGroups);
            return count == 0 ? null : (int)count;
        }
        catch { return null; }
    }

    /// <summary>An L3 cache domain (CCX): the logical cores sharing one last-level cache.</summary>
    internal readonly struct Domain
    {
        public readonly ulong Mask;   // affinity mask within Group
        public readonly ushort Group; // Windows processor group
        public readonly int Cores;    // popcount(Mask)
        public Domain(ulong mask, ushort group, int cores) { Mask = mask; Group = group; Cores = cores; }
    }

    /// <summary>Enumerate L3 domains. Empty on non-Windows or on failure (caller uses the per-tile path).</summary>
    [MethodImpl(Hot)]
    internal static Domain[] DetectL3Domains()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return Array.Empty<Domain>();
        try
        {
            const int RelationCache = 2;
            uint len = 0;
            GetLogicalProcessorInformationEx(RelationCache, IntPtr.Zero, ref len);
            if (len == 0) return Array.Empty<Domain>();
            IntPtr buf = Marshal.AllocHGlobal((int)len);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationCache, buf, ref len)) return Array.Empty<Domain>();
                var list = new List<Domain>();
                long ptr = (long)buf, end = ptr + len;
                while (ptr < end)
                {
                    int rel = Marshal.ReadInt32((IntPtr)ptr);
                    int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                    if (size <= 0) break;
                    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: CACHE_RELATIONSHIP at +8; record offsets
                    // Level@8, CacheSize@12, GROUP_AFFINITY{ Mask@40 (8B), Group@48 }.
                    if (rel == RelationCache && Marshal.ReadByte((IntPtr)(ptr + 8)) == 3)
                    {
                        ulong mask = (ulong)Marshal.ReadInt64((IntPtr)(ptr + 40));
                        ushort group = (ushort)Marshal.ReadInt16((IntPtr)(ptr + 48));
                        int cores = System.Numerics.BitOperations.PopCount(mask);
                        if (cores > 0) list.Add(new Domain(mask, group, cores));
                    }
                    ptr += size;
                }
                return list.ToArray();
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        catch { return Array.Empty<Domain>(); }
    }

    /// <summary>Enumerate PHYSICAL cores — one <see cref="Domain"/> per core, Mask = that core's logical
    /// procs (its SMT siblings). Lets a pool pin exactly one thread per physical core (no SMT contention),
    /// matching OpenBLAS's 1-thread-per-core decomposition. Empty on non-Windows / failure.</summary>
    [MethodImpl(Hot)]
    internal static Domain[] DetectPhysicalCores()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return Array.Empty<Domain>();
        try
        {
            const int RelationProcessorCore = 0;
            uint len = 0;
            GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref len);
            if (len == 0) return Array.Empty<Domain>();
            IntPtr buf = Marshal.AllocHGlobal((int)len);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buf, ref len)) return Array.Empty<Domain>();
                var list = new List<Domain>();
                long ptr = (long)buf, end = ptr + len;
                while (ptr < end)
                {
                    int rel = Marshal.ReadInt32((IntPtr)ptr);
                    int size = Marshal.ReadInt32((IntPtr)(ptr + 4));
                    if (size <= 0) break;
                    // PROCESSOR_RELATIONSHIP at +8: Flags@8, EfficiencyClass@9, Reserved[20]@10,
                    // GroupCount@30 (WORD), GroupMask[0] (GROUP_AFFINITY) @32 { Mask@32 (8B), Group@40 (2B) }.
                    if (rel == RelationProcessorCore)
                    {
                        ushort groupCount = (ushort)Marshal.ReadInt16((IntPtr)(ptr + 30));
                        if (groupCount >= 1)
                        {
                            ulong mask = (ulong)Marshal.ReadInt64((IntPtr)(ptr + 32));
                            ushort group = (ushort)Marshal.ReadInt16((IntPtr)(ptr + 40));
                            int cores = System.Numerics.BitOperations.PopCount(mask);
                            if (cores > 0) list.Add(new Domain(mask, group, cores));
                        }
                    }
                    ptr += size;
                }
                return list.ToArray();
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        catch { return Array.Empty<Domain>(); }
    }

    /// <summary>
    /// The first logical processor of each physical core in the calling thread's processor group that the
    /// process may run on, in core order. Empty on non-Windows or on failure.
    /// </summary>
    [MethodImpl(Hot)]
    internal static (ushort Group, byte Number)[] UsableCoresInCurrentGroup()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return Array.Empty<(ushort, byte)>();
        try
        {
            GetCurrentProcessorNumberEx(out var here);
            ulong allowed = ulong.MaxValue;
            // The process mask describes the primary group only; elsewhere every processor is allowed.
            if (GetProcessAffinityMask(GetCurrentProcess(), out var processMask, out _) && processMask != 0)
                allowed = (ulong)processMask;
            var list = new List<(ushort, byte)>();
            foreach (var core in DetectPhysicalCores())
            {
                if (core.Group != here.Group) continue;
                ulong usable = core.Mask & allowed;
                if (usable == 0) continue;
                list.Add((core.Group, (byte)System.Numerics.BitOperations.TrailingZeroCount(usable)));
            }
            return list.ToArray();
        }
        catch { return Array.Empty<(ushort, byte)>(); }
    }

    /// <summary>
    /// Sets the calling thread's ideal processor: a scheduling hint, not a restriction. The scheduler
    /// prefers that processor whenever the thread becomes ready, so a thread that is suspended and
    /// resumed comes back to the core whose caches hold its working set.
    /// </summary>
    internal static bool TrySetCurrentThreadIdealProcessor(ushort group, byte number)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return false;
        try
        {
            var pn = new PROCESSOR_NUMBER { Group = group, Number = number };
            return SetThreadIdealProcessorEx(GetCurrentThread(), ref pn, IntPtr.Zero);
        }
        catch { return false; }
    }

    /// <summary>Pin the calling thread to a domain's cores (best-effort; correctness is independent of it).</summary>
    internal static bool TryPinCurrentThread(in Domain d)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return false;
        try
        {
            var ga = new GROUP_AFFINITY { Mask = (nuint)d.Mask, Group = d.Group };
            return SetThreadGroupAffinity(GetCurrentThread(), ref ga, IntPtr.Zero);
        }
        catch { return false; }
    }
}
#endif
