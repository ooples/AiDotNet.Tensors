// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A host thread waiting on a stream whose work does not finish (here a kernel spinning on a host flag; in the field a
/// device starved of memory by other processes) must not keep a CPU core busy for the whole wait. Under the context's
/// spin flag cuStreamSynchronize did exactly that: a training process stalled for hours at 100% of a core.
/// </summary>
[Collection("DirectGpuSerial")]
public class HostWaitPolicyTests
{
    // One thread polls a flag in host-mapped memory until the host writes a non-zero value.
    private const string SpinUntilFlagPtx = @"
.version 6.0
.target sm_52
.address_size 64
.visible .entry spin_until_flag(.param .u64 flag)
{
  .reg .u64 %rd<2>; .reg .u32 %r<2>; .reg .pred %p<2>;
  ld.param.u64 %rd1, [flag];
WAIT:
  ld.volatile.global.u32 %r1, [%rd1];
  setp.eq.u32 %p1, %r1, 0;
  @%p1 bra WAIT;
  ret;
}
";

    private static readonly TimeSpan Stall = TimeSpan.FromSeconds(1.5);

    [SkippableFact]
    public void StreamWait_OnStalledDevice_DoesNotSpinACore_UnderTheDefaultPolicy()
    {
        Assert.Equal(GpuHostWaitPolicy.PollThenBlock, GpuHostWaitOptions.Policy);
        var (cpu, wall) = MeasureStalledWait(GpuHostWaitPolicy.PollThenBlock);
        // Spinning reads ~100% of the wall time (measured 98% before the fix); a blocked wait ~0%.
        Assert.True(cpu.TotalMilliseconds < 0.25 * wall.TotalMilliseconds,
            $"the waiting thread used {cpu.TotalMilliseconds:F0} ms of CPU in a {wall.TotalMilliseconds:F0} ms stall");
    }

    // The measurement can see a spin: the explicit Spin policy (the previous default) keeps the core busy.
    [SkippableFact]
    public void StreamWait_OnStalledDevice_SpinsUnderTheSpinPolicy()
    {
        var (cpu, wall) = MeasureStalledWait(GpuHostWaitPolicy.Spin);
        Assert.True(cpu.TotalMilliseconds > 0.6 * wall.TotalMilliseconds,
            $"the waiting thread used {cpu.TotalMilliseconds:F0} ms of CPU in a {wall.TotalMilliseconds:F0} ms stall");
    }

    // One thread spins until its SM clock has advanced by the given number of cycles.
    private const string SpinCyclesPtx = @"
.version 6.0
.target sm_52
.address_size 64
.visible .entry spin_cycles(.param .u64 cycles)
{
  .reg .u64 %rd<5>; .reg .pred %p<2>;
  ld.param.u64 %rd1, [cycles];
  mov.u64 %rd2, %clock64;
LOOP:
  mov.u64 %rd3, %clock64;
  sub.u64 %rd4, %rd3, %rd2;
  setp.lt.u64 %p1, %rd4, %rd1;
  @%p1 bra LOOP;
  ret;
}
";

    /// <summary>
    /// A wait shorter than the poll window ends inside the poll, so PollThenBlock costs a short wait what Spin does.
    /// Paired: each round times one launch-and-wait under each policy, in alternating order, on one context (both
    /// policies create it with the spin flag, so only the stream wait differs). Raw medians of Stopwatch timestamps.
    /// </summary>
    [SkippableFact]
    public void ShortStreamWait_UnderPollThenBlock_IsAsFastAsSpin()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend is unavailable.");
        var prior = GpuHostWaitOptions.Policy; bool priorSet = GpuHostWaitOptions.IsPolicySet;
        IntPtr module = IntPtr.Zero;
        try
        {
            IntPtr kernel = LoadKernel(backend, SpinCyclesPtx, "spin_cycles", out module);
            const long cycles = 300_000;   // ~0.2 ms at 1.5-2 GHz: well inside the 2 ms poll window
            const int warmup = 20, rounds = 200;
            var spin = new List<long>(rounds);
            var pollThenBlock = new List<long>(rounds);
            for (int r = 0; r < warmup + rounds; r++)
            {
                bool spinFirst = (r & 1) == 0;
                for (int k = 0; k < 2; k++)
                {
                    var policy = (k == 0) == spinFirst ? GpuHostWaitPolicy.Spin : GpuHostWaitPolicy.PollThenBlock;
                    GpuHostWaitOptions.Policy = policy;
                    long ticks = TimeLaunchAndWait(backend, kernel, cycles);
                    if (r < warmup) continue;
                    (policy == GpuHostWaitPolicy.Spin ? spin : pollThenBlock).Add(ticks);
                }
            }
            double Median(List<long> v) { var o = v.OrderBy(x => x).ToArray(); return o[o.Length / 2] * 1e6 / Stopwatch.Frequency; }
            double P95(List<long> v) { var o = v.OrderBy(x => x).ToArray(); return o[(int)(o.Length * 0.95)] * 1e6 / Stopwatch.Frequency; }
            double spinMedian = Median(spin), pollMedian = Median(pollThenBlock);
            string report = $"launch+wait median: Spin {spinMedian:F1} us (p95 {P95(spin):F1}), PollThenBlock {pollMedian:F1} us (p95 {P95(pollThenBlock):F1}), {rounds} paired rounds";
            Console.WriteLine(report);
            // A wait that blocked instead (an OS wake-up) adds tens of microseconds; the poll adds a query loop.
            Assert.True(pollMedian <= spinMedian * 1.05 + 10.0, report);
        }
        finally
        {
            GpuHostWaitOptions.RestoreForTests(prior, priorSet);
            if (module != IntPtr.Zero)
            {
                CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(backend.CudaContextHandle), "cuCtxPushCurrent");
                CudaNativeBindings.cuModuleUnload(module);
                CuBlasNative.cuCtxPopCurrent(out _);
            }
        }
    }

    private static IntPtr LoadKernel(CudaBackend backend, string ptx, string name, out IntPtr module)
    {
        CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(backend.CudaContextHandle), "cuCtxPushCurrent");
        try
        {
            var image = GCHandle.Alloc(Encoding.ASCII.GetBytes(ptx + "\0"), GCHandleType.Pinned);
            try
            {
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out module, image.AddrOfPinnedObject()), "cuModuleLoadData");
            }
            finally { image.Free(); }
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, module, name), "cuModuleGetFunction");
            return kernel;
        }
        finally { CuBlasNative.cuCtxPopCurrent(out _); }
    }

    private static unsafe long TimeLaunchAndWait(CudaBackend backend, IntPtr kernel, long cycles)
    {
        long start;
        CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(backend.CudaContextHandle), "cuCtxPushCurrent");
        try
        {
            long c = cycles;
            IntPtr* args = stackalloc IntPtr[1];
            args[0] = (IntPtr)(&c);
            start = Stopwatch.GetTimestamp();
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0,
                backend.DefaultStream.Handle, (IntPtr)args, IntPtr.Zero), "cuLaunchKernel");
        }
        finally { CuBlasNative.cuCtxPopCurrent(out _); }
        backend.Synchronize();
        return Stopwatch.GetTimestamp() - start;
    }

    /// <summary>
    /// The typed option is what takes effect: once <see cref="GpuHostWaitOptions.Policy"/> is set it decides both the
    /// stream waits and a new context's scheduling flag, whatever the deprecated environment variable says. The
    /// variable is read, with a deprecation warning, only while the typed policy has never been set.
    /// </summary>
    [Fact]
    public void TheTypedPolicyWinsOverTheDeprecatedEnvironmentVariable()
    {
        var prior = GpuHostWaitOptions.Policy; bool priorSet = GpuHostWaitOptions.IsPolicySet;
        string? priorEnvironment = Environment.GetEnvironmentVariable(CudaContextScheduling.EnvironmentVariable);
        var warnings = new WarningListener();
        System.Diagnostics.Trace.Listeners.Add(warnings);
        try
        {
            Environment.SetEnvironmentVariable(CudaContextScheduling.EnvironmentVariable, "blocking");
            CudaContextScheduling.ResetEnvironmentCacheForTests();

            // Never set: the deprecated variable still decides, and says it is deprecated.
            GpuHostWaitOptions.RestoreForTests(GpuHostWaitPolicy.PollThenBlock, set: false);
            Assert.Equal(GpuHostWaitPolicy.BlockingSync, CudaContextScheduling.EffectivePolicy);
            Assert.Equal(CudaContextScheduling.Blocking, CudaContextScheduling.ResolveContextFlags());
            Assert.Contains(warnings.Messages, m => m.Contains("deprecated") && m.Contains(nameof(GpuHostWaitOptions)));

            // Set: the typed policy wins over the variable, for the waits and for a new context's flag.
            GpuHostWaitOptions.Policy = GpuHostWaitPolicy.Spin;
            Assert.Equal(GpuHostWaitPolicy.Spin, CudaContextScheduling.EffectivePolicy);
            Assert.Equal(CudaContextScheduling.Spin, CudaContextScheduling.ResolveContextFlags());
            GpuHostWaitOptions.Policy = GpuHostWaitPolicy.PollThenBlock;
            Assert.Equal(GpuHostWaitPolicy.PollThenBlock, CudaContextScheduling.EffectivePolicy);

            // A misspelled variable is ignored once the typed policy is set, and an error while it decides.
            Environment.SetEnvironmentVariable(CudaContextScheduling.EnvironmentVariable, "park-sometimes");
            CudaContextScheduling.ResetEnvironmentCacheForTests();
            Assert.Equal(CudaContextScheduling.Spin, CudaContextScheduling.ResolveContextFlags());
            GpuHostWaitOptions.RestoreForTests(GpuHostWaitPolicy.PollThenBlock, set: false);
            Assert.Throws<InvalidOperationException>(() => CudaContextScheduling.ResolveContextFlags());
        }
        finally
        {
            System.Diagnostics.Trace.Listeners.Remove(warnings);
            Environment.SetEnvironmentVariable(CudaContextScheduling.EnvironmentVariable, priorEnvironment);
            CudaContextScheduling.ResetEnvironmentCacheForTests();
            GpuHostWaitOptions.RestoreForTests(prior, priorSet);
        }
    }

    private sealed class WarningListener : System.Diagnostics.TraceListener
    {
        internal readonly List<string> Messages = new();
        public override void Write(string? message) { lock (Messages) Messages.Add(message ?? ""); }
        public override void WriteLine(string? message) => Write(message);
    }

    [Fact]
    public void Options_RejectAnUnknownPolicyAndANegativeWindow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuHostWaitOptions.Policy = (GpuHostWaitPolicy)99);
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuHostWaitOptions.PollWindow = TimeSpan.FromTicks(-1));
        Assert.Equal(CudaContextScheduling.Spin, CudaContextScheduling.Flags(GpuHostWaitPolicy.PollThenBlock));
        Assert.Equal(CudaContextScheduling.Blocking, CudaContextScheduling.Flags(GpuHostWaitPolicy.BlockingSync));
        Assert.Equal(GpuHostWaitPolicy.PollThenBlock, CudaContextScheduling.ResolvePolicy("poll-then-block"));
    }

    private static unsafe (TimeSpan Cpu, TimeSpan Wall) MeasureStalledWait(GpuHostWaitPolicy policy)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        Skip.If(!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && !RuntimeInformation.IsOSPlatform(OSPlatform.Linux),
            "Thread CPU time is read on Windows and Linux.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend is unavailable.");
        var prior = GpuHostWaitOptions.Policy; bool priorSet = GpuHostWaitOptions.IsPolicySet;
        IntPtr host = IntPtr.Zero, module = IntPtr.Zero;
        Thread? releaser = null;
        try
        {
            GpuHostWaitOptions.Policy = policy;
            CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(backend.CudaContextHandle), "cuCtxPushCurrent");
            try
            {
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemAllocHost(out host, sizeof(int)), "cuMemAllocHost");
                *(int*)host = 0;
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemHostGetDevicePointer(out IntPtr deviceFlag, host, 0), "cuMemHostGetDevicePointer");
                var image = GCHandle.Alloc(Encoding.ASCII.GetBytes(SpinUntilFlagPtx + "\0"), GCHandleType.Pinned);
                try
                {
                    CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out module, image.AddrOfPinnedObject()), "cuModuleLoadData");
                }
                finally { image.Free(); }
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleGetFunction(out IntPtr kernel, module, "spin_until_flag"), "cuModuleGetFunction");
                IntPtr* args = stackalloc IntPtr[1];
                args[0] = (IntPtr)(&deviceFlag);
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0,
                    backend.DefaultStream.Handle, (IntPtr)args, IntPtr.Zero), "cuLaunchKernel");
            }
            finally { CuBlasNative.cuCtxPopCurrent(out _); }

            IntPtr flag = host;
            releaser = new Thread(() => { Thread.Sleep(Stall); *(int*)flag = 1; }) { IsBackground = true };
            var wall = Stopwatch.StartNew();
            TimeSpan cpu0 = CurrentThreadCpu();
            releaser.Start();
            backend.Synchronize();   // waits for the stalled kernel
            TimeSpan cpu = CurrentThreadCpu() - cpu0;
            wall.Stop();
            Assert.True(wall.Elapsed >= Stall - TimeSpan.FromMilliseconds(100), $"the wait returned after {wall.Elapsed.TotalMilliseconds:F0} ms");
            return (cpu, wall.Elapsed);
        }
        finally
        {
            if (host != IntPtr.Zero) *(int*)host = 1;   // never leave the kernel spinning
            releaser?.Join();
            backend.Synchronize();
            GpuHostWaitOptions.RestoreForTests(prior, priorSet);
            CuBlasNative.CheckCudaResult(CuBlasNative.cuCtxPushCurrent(backend.CudaContextHandle), "cuCtxPushCurrent");
            if (module != IntPtr.Zero) CudaNativeBindings.cuModuleUnload(module);
            if (host != IntPtr.Zero) CudaNativeBindings.cuMemFreeHost(host);
            CuBlasNative.cuCtxPopCurrent(out _);
        }
    }

    private static TimeSpan CurrentThreadCpu()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (!GetThreadTimes(GetCurrentThread(), out _, out _, out long kernel, out long user))
                throw new InvalidOperationException("GetThreadTimes failed.");
            return TimeSpan.FromTicks(kernel + user);   // 100 ns units, as TimeSpan ticks
        }
        if (clock_gettime(3 /* CLOCK_THREAD_CPUTIME_ID */, out Timespec ts) != 0)
            throw new InvalidOperationException("clock_gettime failed.");
        return TimeSpan.FromTicks(ts.Seconds * TimeSpan.TicksPerSecond + ts.Nanoseconds / 100);
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetThreadTimes(IntPtr thread, out long creation, out long exit, out long kernel, out long user);

    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }

    [DllImport("libc", SetLastError = true)]
    private static extern int clock_gettime(int clock, out Timespec time);
}
