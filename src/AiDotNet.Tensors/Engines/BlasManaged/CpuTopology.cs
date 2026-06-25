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

    /// <summary>An L3 cache domain (CCX): the logical cores sharing one last-level cache.</summary>
    internal readonly struct Domain
    {
        public readonly ulong Mask;   // affinity mask within Group
        public readonly ushort Group; // Windows processor group
        public readonly int Cores;    // popcount(Mask)
        public Domain(ulong mask, ushort group, int cores) { Mask = mask; Group = group; Cores = cores; }
    }

    /// <summary>Enumerate L3 domains. Empty on non-Windows or on failure (caller uses the per-tile path).</summary>
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
