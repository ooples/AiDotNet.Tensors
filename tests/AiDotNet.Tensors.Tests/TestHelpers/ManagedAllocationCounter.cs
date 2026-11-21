using System;
using System.Reflection;

namespace AiDotNet.Tensors.Tests.TestHelpers;

internal static class ManagedAllocationCounter
{
#if NET5_0_OR_GREATER
    internal static bool SupportsCurrentThread => true;
    internal static long CurrentThread() => GC.GetAllocatedBytesForCurrentThread();
    internal static long AllThreads() => GC.GetTotalAllocatedBytes(precise: true);
#else
    // A .NET Framework 4.8 runtime has this counter even when the project targets older reference assemblies.
    private static readonly Func<long>? ReadCurrentThread =
        typeof(GC).GetMethod("GetAllocatedBytesForCurrentThread", BindingFlags.Public | BindingFlags.Static)
            ?.CreateDelegate(typeof(Func<long>)) as Func<long>;
    internal static bool SupportsCurrentThread => ReadCurrentThread is not null;
    internal static long CurrentThread() => ReadCurrentThread!();
    internal static long AllThreads()
    {
        AppDomain.MonitoringIsEnabled = true;
        return AppDomain.CurrentDomain.MonitoringTotalAllocatedMemorySize;
    }
#endif
}
