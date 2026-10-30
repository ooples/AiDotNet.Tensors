using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The CUDA driver keeps cuLinkCreate's option-value array and writes the info/error logs' actual sizes back into it
/// at cuLinkComplete. When that array was a marshalled managed IntPtr[] it was pinned only for the cuLinkCreate call,
/// so a GC in between moved it and the write-back overwrote the low 32 bits of whatever object took its place: the
/// CUDA lane's intermittent "Internal CLR error 0x80131506" (heap corruption at a 16-byte stride, crash dumps showed
/// cuLinkComplete on the faulting thread).
/// </summary>
[Collection("DirectGpuSerial")]
public class DirectPtxLinkOptionLifetimeTests
{
    [Fact]
    public void CuLinkCreate_TakesNoManagedArrays()
    {
        var methods = typeof(CudaNativeBindings)
            .GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(m => m.GetCustomAttribute<DllImportAttribute>()?.EntryPoint?.StartsWith("cuLinkCreate", StringComparison.Ordinal) == true)
            .ToArray();
        Assert.NotEmpty(methods);
        foreach (var method in methods)
            foreach (var parameter in method.GetParameters())
                Assert.False(parameter.ParameterType.IsArray,
                    $"{method.Name}({parameter.Name}) marshals a managed array the link state writes into after the call returns.");
    }

    [SkippableFact]
    public void DriverLink_UnderCompactingGcs_LeavesTheManagedHeapIntact()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor > 8 || (runtime.ComputeCapabilityMajor == 8 && runtime.ComputeCapabilityMinor >= 6),
            "Requires an SM86+ device for the sm_86 test kernel.");
        string ptx = PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, 65_536);

        const long Sentinel = 0x5A5A5A5A5A5A5A5A;
        var live = new Queue<long[]>();
        int corrupted = 0;
        void Check(long[] block)
        {
            for (int i = 0; i < block.Length; i++)
                if (block[i] != Sentinel) { Interlocked.Increment(ref corrupted); return; }
        }

        // Small sentinel blocks churned through gen0 with forced compacting collections: whatever slides into a moved
        // option array's old address is a sentinel block, so a stray write-back shows up as a changed word.
        using var stop = new ManualResetEventSlim();
        var churn = new Thread(() =>
        {
            for (int n = 0; !stop.IsSet; n++)
            {
                var block = new long[6];
                block.AsSpan().Fill(Sentinel);
                live.Enqueue(block);
                if (live.Count > 50_000) Check(live.Dequeue());
                if ((n & 1023) == 0) GC.Collect(0, GCCollectionMode.Forced, blocking: true, compacting: true);
            }
        }) { IsBackground = true };
        churn.Start();
        try
        {
            for (int i = 0; i < 150; i++)
                Assert.NotEmpty(DirectPtxCubinArtifactCache.CompileExact(runtime, ptx).Image);
        }
        finally
        {
            stop.Set();
            churn.Join();
        }
        while (live.Count > 0) Check(live.Dequeue());
        Assert.Equal(0, corrupted);
    }
}
