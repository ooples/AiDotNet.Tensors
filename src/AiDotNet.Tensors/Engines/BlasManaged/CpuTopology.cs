using System.Runtime.CompilerServices;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
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
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern nuint SetThreadAffinityMask(IntPtr hThread, nuint mask);

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESSOR_NUMBER { public ushort Group; public byte Number; public byte Reserved; }
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetThreadIdealProcessorEx(IntPtr hThread, ref PROCESSOR_NUMBER ideal, IntPtr previous);
    [DllImport("kernel32.dll")]
    private static extern bool GetThreadGroupAffinity(IntPtr hThread, out GROUP_AFFINITY affinity);
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
    /// The first logical processor of each physical core in the calling thread's primary processor group that the
    /// process may run on, in core order. Empty on non-Windows or on failure.
    /// </summary>
    [MethodImpl(Hot)]
    internal static (ushort Group, byte Number)[] UsableCoresInCurrentGroup()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return Array.Empty<(ushort, byte)>();
        try
        {
            if (!GetThreadGroupAffinity(GetCurrentThread(), out var primary)) return Array.Empty<(ushort, byte)>();
            ulong allowed = ulong.MaxValue;
            // With the current-process pseudo-handle, the mask describes the calling thread's
            // primary group, which can differ from the group where it is currently running.
            if (GetProcessAffinityMask(GetCurrentProcess(), out var processMask, out _) && processMask != 0)
                allowed = (ulong)processMask;
            var list = new List<(ushort, byte)>();
            foreach (var core in DetectPhysicalCores())
            {
                if (core.Group != primary.Group) continue;
                ulong usable = core.Mask & allowed;
                if (usable == 0) continue;
                list.Add((core.Group, (byte)System.Numerics.BitOperations.TrailingZeroCount(usable)));
            }
            return list.ToArray();
        }
        catch { return Array.Empty<(ushort, byte)>(); }
    }

    /// <summary>
    /// <see cref="UsableCoresInCurrentGroup"/> followed by the physical cores of every other processor group, when
    /// the process's affinity is unrestricted in its own group. A Windows host with more than 64 logical processors
    /// splits them into groups (a 64-core / 128-thread Threadripper 3990X: two groups of 32 cores), and a process
    /// starts in one of them; binding workers only there capped the pool at 32 cores, so a 64-thread budget stacked
    /// two FMA-bound workers on each core's SMT pair (#653: 256x1152x1152 GEMM 897 us at 32 threads, 1926 us at 64).
    /// A process restricted by affinity (start /affinity, a job object) keeps its own group's allowed cores only.
    /// </summary>
    [MethodImpl(Hot)]
    internal static (ushort Group, byte Number)[] UsableCores()
    {
        if (RequestedCores is { } requested) return ConfineToRequestedCores(requested);
        var own = UsableCoresInCurrentGroup();
        // AIDOTNET_CPU_SPAN_GROUPS=0 keeps the pool in the process's own group (the pre-#653 placement).
        if (own.Length == 0 || Environment.GetEnvironmentVariable("AIDOTNET_CPU_SPAN_GROUPS") == "0"
            || !AffinityAllowsOtherGroups)
            return own;
        ushort home = own[0].Group;
        var list = new List<(ushort, byte)>(own);
        foreach (var core in DetectPhysicalCores().Where(c => c.Group != home && c.Mask != 0))
            list.Add((core.Group, (byte)System.Numerics.BitOperations.TrailingZeroCount(core.Mask)));
        return list.ToArray();
    }

    /// <summary>The environment variable that confines this process's CPU work to a set of physical cores.</summary>
    internal const string CoresEnvironmentVariable = "AIDOTNET_CPU_CORES";

    /// <summary>
    /// The physical cores <c>AIDOTNET_CPU_CORES</c> confines this process to, as indexes into
    /// <see cref="DetectPhysicalCores"/> (machine order, across processor groups), or null when it is not set. Two
    /// processes sharing a machine otherwise both bind their pool's first worker to core 0, the second to core 1 and so
    /// on: each takes the other's cores. Giving each its own set (<c>0-31</c> and <c>32-63</c>) keeps them apart.
    /// The list takes indexes and ranges: <c>0-15,32,40-47</c>. A malformed value throws on first use.
    /// </summary>
    internal static int[]? RequestedCores => s_requestedCores.Value;

    private static readonly Lazy<int[]?> s_requestedCores = new(() =>
    {
        string? text = Environment.GetEnvironmentVariable(CoresEnvironmentVariable);
        return string.IsNullOrWhiteSpace(text) ? null : ParseCoreList(text!);
    });

    /// <summary>Parses a core list (<c>0-15,32,40-47</c>) into distinct indexes in the order given.</summary>
    internal static int[] ParseCoreList(string text)
    {
        var cores = new List<int>();
        var seen = new HashSet<int>();
        foreach (string rawPart in text.Split(','))
        {
            string part = rawPart.Trim();
            int dash = part.IndexOf('-');
            string first = dash < 0 ? part : part.Substring(0, dash).Trim();
            string last = dash < 0 ? part : part.Substring(dash + 1).Trim();
            if (!int.TryParse(first, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int lo)
                || !int.TryParse(last, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int hi)
                || hi < lo)
                throw new InvalidOperationException(
                    $"{CoresEnvironmentVariable}={text}: '{part}' is not a core index or an ascending range such as 0-15.");
            for (int c = lo; c <= hi; c++)
                if (seen.Add(c)) cores.Add(c);
        }
        return cores.ToArray();
    }

    // The requested cores as (group, first logical processor), and -- when they all lie in the process's primary group,
    // the only group a process affinity mask can name -- the process confined to their logical processors (SMT siblings
    // included), so the threads that dispatch work stay there too, not only the pool's workers.
    // On Linux the process is confined the same way and no worker is bound one by one: every thread then stays on the
    // requested cores. A platform with neither (macOS has no affinity) is told once that only the thread count follows.
    private static (ushort Group, byte Number)[] ConfineToRequestedCores(int[] requested)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            SetProcessAffinity(MapLinuxCores(requested, ReadLinuxPhysicalCores("/sys/devices/system/cpu")));
            return Array.Empty<(ushort, byte)>();
        }

        var all = DetectPhysicalCores();
        if (all.Length == 0)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"{CoresEnvironmentVariable} is set, but this platform has no processor affinity: threads are not bound to "
                + "the requested cores. Only the thread count follows the setting.");
            return Array.Empty<(ushort, byte)>();
        }

        var (cores, group, mask) = MapRequestedCores(requested, all);
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            && group is { } only && GetThreadGroupAffinity(GetCurrentThread(), out var primary) && primary.Group == only)
            SetProcessAffinity(mask);
        return cores;
    }

    // Confines the process to the logical processors of `mask`, or throws naming the setting: a process left unconfined
    // while its pool is sized for the requested cores is the collision the setting exists to prevent.
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static void SetProcessAffinity(ulong mask)
    {
        // The affinity mask is pointer-sized: a 32-bit process cannot name a logical processor above 31, and a
        // truncated mask would confine it to other cores than the ones asked for.
        if (IntPtr.Size == 4 && mask > uint.MaxValue)
            throw new InvalidOperationException(
                $"{CoresEnvironmentVariable} names logical processors above 31, which a 32-bit process cannot be confined to.");
        IntPtr native = IntPtr.Size == 4 ? new IntPtr(unchecked((int)(uint)mask)) : new IntPtr(unchecked((long)mask));
        try
        {
            using var process = System.Diagnostics.Process.GetCurrentProcess();
            process.ProcessorAffinity = native;
        }
        catch (Exception refused) when (refused is System.ComponentModel.Win32Exception or InvalidOperationException
                                            or NotSupportedException or PlatformNotSupportedException)
        {
            throw new InvalidOperationException(
                $"{CoresEnvironmentVariable}: the operating system refused to confine the process to the requested cores "
                + $"(affinity mask 0x{mask:X}).", refused);
        }
    }

    /// <summary>
    /// The logical processors of each physical core under a Linux sysfs CPU directory
    /// (<c>/sys/devices/system/cpu</c>), cores in order of their first logical processor: each
    /// <c>cpuN/topology/thread_siblings_list</c> names the SMT siblings of one core. Empty when the directory has no
    /// topology.
    /// </summary>
    internal static int[][] ReadLinuxPhysicalCores(string sysfsCpuDirectory)
    {
        var cores = new SortedDictionary<int, int[]>();
        if (!System.IO.Directory.Exists(sysfsCpuDirectory)) return Array.Empty<int[]>();
        foreach (string directory in System.IO.Directory.GetDirectories(sysfsCpuDirectory, "cpu*"))
        {
            string name = System.IO.Path.GetFileName(directory);
            if (!int.TryParse(name.Substring(3), System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out _))
                continue;   // cpufreq, cpuidle
            // Joined by hand: the parts are fixed relative names under a directory the enumeration returned.
            string siblingsFile = directory + System.IO.Path.DirectorySeparatorChar + "topology"
                + System.IO.Path.DirectorySeparatorChar + "thread_siblings_list";
            if (!System.IO.File.Exists(siblingsFile)) continue;   // an offline processor has no topology
            int[] siblings;
            try
            {
                siblings = ParseCoreList(System.IO.File.ReadAllText(siblingsFile).Trim());
            }
            catch (Exception unreadable) when (unreadable is InvalidOperationException or System.IO.IOException or UnauthorizedAccessException)
            {
                System.Diagnostics.Trace.TraceWarning($"CPU topology: {siblingsFile} could not be read ({unreadable.GetType().Name}).");
                continue;
            }

            if (siblings.Length == 0) continue;
            Array.Sort(siblings);
            cores[siblings[0]] = siblings;
        }

        var ordered = new int[cores.Count][];
        cores.Values.CopyTo(ordered, 0);
        return ordered;
    }

    /// <summary>The affinity mask of the requested physical cores of <paramref name="cores"/> (as
    /// <see cref="ReadLinuxPhysicalCores"/> lists them), SMT siblings included.</summary>
    internal static ulong MapLinuxCores(int[] requested, int[][] cores)
    {
        if (cores.Length == 0)
            throw new InvalidOperationException(
                $"{CoresEnvironmentVariable} is set, but this machine's CPU topology could not be read from sysfs, so the "
                + "requested cores cannot be found. Confine the process with taskset or numactl instead.");
        ulong mask = 0;
        foreach (int index in requested)
        {
            if (index >= cores.Length)
                throw new InvalidOperationException(
                    $"{CoresEnvironmentVariable} names core {index}, but this machine has {cores.Length} physical cores (0-{cores.Length - 1}).");
            foreach (int processor in cores[index])
            {
                if (processor >= IntPtr.Size * 8)
                    throw new InvalidOperationException(
                        $"{CoresEnvironmentVariable} names core {index}, whose logical processor {processor} is beyond the "
                        + $"{IntPtr.Size * 8} a process affinity mask can name. Confine the process with taskset or numactl instead.");
                mask |= 1UL << processor;
            }
        }

        return mask;
    }

    /// <summary>The requested cores of <paramref name="all"/> as (group, first logical processor); their single group
    /// (null when they span several) and the mask of their logical processors in it.</summary>
    internal static ((ushort Group, byte Number)[] Cores, ushort? Group, ulong Mask) MapRequestedCores(int[] requested, Domain[] all)
    {
        var list = new List<(ushort, byte)>(requested.Length);
        ulong mask = 0;
        ushort? group = requested.Length > 0 && requested[0] < all.Length ? all[requested[0]].Group : null;
        foreach (int index in requested.Where(index => index >= all.Length))
            throw new InvalidOperationException(
                $"{CoresEnvironmentVariable} names core {index}, but this machine has {all.Length} physical cores (0-{all.Length - 1}).");
        foreach (var core in requested.Select(index => all[index]))
        {
            list.Add((core.Group, (byte)System.Numerics.BitOperations.TrailingZeroCount(core.Mask)));
            if (core.Group != group) group = null;
            else mask |= core.Mask;
        }
        return (list.ToArray(), group, group is null ? 0 : mask);
    }

    private static readonly Lazy<bool> s_affinityAllowsOtherGroups = new(ComputeAffinityAllowsOtherGroups);

    /// <summary>
    /// Whether the process may run on other processor groups: its affinity covers every processor of its own group
    /// (nothing restricted it), or it already runs threads in several groups. Decided once: binding a worker to another
    /// group turns the process multi-group, after which <c>GetProcessAffinityMask</c> succeeds with a zero mask.
    /// </summary>
    internal static bool AffinityAllowsOtherGroups => s_affinityAllowsOtherGroups.Value;

    private static bool ComputeAffinityAllowsOtherGroups()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return false;
        try
        {
            if (!GetThreadGroupAffinity(GetCurrentThread(), out var primary)) return false;
            ulong groupMask = 0;
            foreach (var core in DetectPhysicalCores().Where(c => c.Group == primary.Group))
                groupMask |= core.Mask;
            if (groupMask == 0 || !GetProcessAffinityMask(GetCurrentProcess(), out var processMask, out var systemMask)) return false;
            // Zero: the process already has threads in more than one group (documented for GetProcessAffinityMask).
            return IsUnrestrictedAffinity((ulong)processMask, (ulong)systemMask, groupMask);
        }
        catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException)
        {
            return false;
        }
    }

    // Both zero masks mean the process already spans groups; a failed native call is rejected above.
    internal static bool IsUnrestrictedAffinity(ulong processMask, ulong systemMask, ulong groupMask) =>
        (processMask == 0 && systemMask == 0) || (groupMask != 0 && (processMask & groupMask) == groupMask);

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
            var thread = GetCurrentThread();
            if (!GetThreadGroupAffinity(thread, out var current)) return false;
            if (current.Group == d.Group)
            {
                if (!GetProcessAffinityMask(GetCurrentProcess(), out var allowed, out var system)) return false;
                nuint mask = (nuint)d.Mask;
                if (allowed != 0) mask &= allowed;
                else if (system != 0) return false;
                if (mask == 0) return false;
                // A group's affinity can include SMT siblings excluded by the caller's process mask.
                // SetThreadGroupAffinity can lift that restriction on Windows; the same-group API
                // preserves it and rejects any processor outside the allowed process allocation.
                return SetThreadAffinityMask(thread, mask) != 0;
            }
            if (!ComputeAffinityAllowsOtherGroups()) return false;
            var ga = new GROUP_AFFINITY { Mask = (nuint)d.Mask, Group = d.Group };
            return SetThreadGroupAffinity(thread, ref ga, IntPtr.Zero);
        }
        catch { return false; }
    }
}
#endif
