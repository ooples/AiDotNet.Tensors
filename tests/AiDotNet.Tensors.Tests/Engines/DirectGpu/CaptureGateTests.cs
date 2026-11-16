using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A CUDA call failing while a capture scope opens (stream/event creation, the ordering record/wait) must leave the
/// backend as it found it. It used to leave the context's capture gate held (every other capture and every
/// SynchronizeContextOutsideCapture on the context blocked forever), this thread's capture depth raised (later captures
/// ran as "nested", frees deferred forever) and the DirectPtx pin set active (every later capture threw).
/// </summary>
[Collection("DirectGpuSerial")]
public class CaptureGateTests
{
    [SkippableFact]
    public void First_use_solver_tuning_takes_the_capture_gate_before_the_cache_lock()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (gpu?.GetBackend() is not CudaBackend)
        {
            gpu?.Dispose();
            Skip.If(true, "CUDA backend did not resolve.");
        }
        using (gpu)
        {
            var cuda = (CudaBackend)gpu!.GetBackend()!;
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            int major = (int)typeof(CudaBackend).GetField("_ccMajor", flags)!.GetValue(cuda)!;
            int minor = (int)typeof(CudaBackend).GetField("_ccMinor", flags)!.GetValue(cuda)!;
            Skip.If(major != 8 || minor != 6, "Direct PTX Cholesky 4x4 requires SM86.");
            const int batch = 1024;
            var matrices = new float[batch * 16];
            for (int i = 0; i < batch; i++)
                for (int d = 0; d < 4; d++) matrices[i * 16 + d * 5] = 4f;
            using var input = cuda.AllocateBuffer(matrices);
            using var output = cuda.AllocateBuffer(matrices.Length);
            using var info = cuda.AllocateBuffer(batch);
            bool? prior = DirectPtxFeatureGate.Cholesky4x4ExperimentOverride;
            var priorHook = CudaBackend.TestHookInsideDirectPtxSolverLock;
            bool? gateFree = null;
            try
            {
                DirectPtxFeatureGate.Cholesky4x4ExperimentOverride = true;
                CudaBackend.TestHookInsideDirectPtxSolverLock = () =>
                    gateFree = GateFreeFromAnotherThread(CudaBackend.CaptureGateFor(cuda.CudaContextHandle));
                Assert.True(cuda.TryDirectPtxCholesky4x4(input, output, info, batch, 4, upper: false), cuda.DirectPtxLastError);
                Assert.Equal(false, gateFree);
                var factors = cuda.DownloadBuffer(output);
                for (int i = 0; i < batch; i++)
                    for (int d = 0; d < 4; d++) Assert.Equal(2f, factors[i * 16 + d * 5], 4);
            }
            finally
            {
                DirectPtxFeatureGate.Cholesky4x4ExperimentOverride = prior;
                CudaBackend.TestHookInsideDirectPtxSolverLock = priorHook;
            }
        }
    }

    [SkippableFact]
    public void A_failed_capture_setup_releases_the_gate_the_depth_and_the_pins()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        if (!(gpu is not null && gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend))
        {
            gpu?.Dispose();
            Skip.If(true, "CUDA backend did not resolve.");
        }
        using (gpu)
        {
            var cuda = (CudaBackend)gpu!.GetBackend()!;
            // Ordinary cache access must stay independent: another thread can run an op on the main
            // stream while this thread captures on its private stream and waits for that op.
            var ptxGate = typeof(CudaBackend).GetField("_directPtxLock",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(cuda);
            Assert.NotNull(ptxGate);
            lock (ptxGate!)
                Assert.True(GateFreeFromAnotherThread(CudaBackend.CaptureGateFor(cuda.CudaContextHandle)),
                    "Ordinary PTX cache access must not hold the capture gate.");
            CudaBackend.TestHookCaptureSetup = () => throw new InvalidOperationException("injected capture setup failure");
            try
            {
                var thrown = Assert.Throws<InvalidOperationException>(() => cuda.CaptureGraph(() => { }));
                Assert.Equal("injected capture setup failure", thrown.Message);
                thrown = Assert.Throws<InvalidOperationException>(() => cuda.TryUpdateCapturedGraph(new IntPtr(1), () => { }));
                Assert.Equal("injected capture setup failure", thrown.Message);
            }
            finally { CudaBackend.TestHookCaptureSetup = null; }

            Assert.False(CudaBackend.IsCapturingOnThisThread, "the failed scope left this thread's capture depth raised");

            var gate = CudaBackend.CaptureGateFor(cuda.CudaContextHandle);
            Assert.True(GateFreeFromAnotherThread(gate, 5000), "the failed scope left the context's capture gate held");

            // Pins released and the backend still captures.
            var exec = cuda.CaptureGraph(() => { });
            Assert.NotEqual(IntPtr.Zero, exec);
            cuda.DestroyCapturedGraph(exec);
        }
    }

    // A dedicated thread: Task.Run(..).Result may inline the task on the waiting thread, and the monitor is reentrant.
    internal static bool GateFreeFromAnotherThread(object gate, int timeoutMs = 200)
    {
        bool free = false;
        var probe = new Thread(() =>
        {
            if (!Monitor.TryEnter(gate, timeoutMs)) return;
            Monitor.Exit(gate);
            free = true;
        });
        probe.Start();
        probe.Join();
        return free;
    }

    /// <summary>
    /// The DirectPtx runtime's own captures took no gate, so another thread's Upload/Download could cuCtxSynchronize
    /// the context mid-capture and the driver rejected it. Both capture paths must hold the gate exactly while the
    /// capture is open.
    /// </summary>
    [SkippableFact]
    public void DirectPtx_captures_hold_the_context_gate_only_while_open()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "CUDA driver not available.");
        DirectPtxRuntime? runtime = null;
        // A driver without a usable device fails cuInit / cuDeviceGet; anything else is a real failure.
        try { runtime = new DirectPtxRuntime(0); }
        catch (InvalidOperationException ex) when (ex.Message.StartsWith("cuInit") || ex.Message.StartsWith("cuDeviceGet")) { }
        Skip.If(runtime is null, "CUDA device did not initialize.");
        using (runtime)
        {
            var gate = CudaBackend.CaptureGateFor(runtime!.Context);
            bool? freeDuringCapture = null;
            using (runtime.CaptureGraph(() => freeDuringCapture = GateFreeFromAnotherThread(gate))) { }
            Assert.True(freeDuringCapture == false, $"CaptureGraph left the gate open while its capture was open ({freeDuringCapture?.ToString() ?? "launch never ran"})");
            Assert.True(GateFreeFromAnotherThread(gate), "CaptureGraph kept the gate after the capture ended");

            freeDuringCapture = null;
            runtime.MeasureCapturedKernelSamples(() => freeDuringCapture ??= GateFreeFromAnotherThread(gate), 0, 1, 1);
            Assert.True(freeDuringCapture == false, $"the tuner capture left the gate open while its capture was open ({freeDuringCapture?.ToString() ?? "launch never ran"})");
            Assert.True(GateFreeFromAnotherThread(gate), "the tuner capture kept the gate after the capture ended");

            Assert.Throws<InvalidOperationException>(() => runtime.CaptureGraph(() => throw new InvalidOperationException("boom")));
            Assert.True(GateFreeFromAnotherThread(gate), "a failed capture kept the gate");
        }
    }
}

/// <summary>
/// Two engines may wrap one DirectGpuEngine. The owner registry kept only the newest and nothing unregistered on
/// Dispose, so a tape over that backend's data ran its backward on a disposed engine.
/// </summary>
[Collection("DirectGpuSerial")]
public class BackendOwnerRegistryTests
{
    [SkippableFact]
    public void Disposing_the_newest_engine_hands_the_backend_back_to_a_live_one()
    {
        AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine? direct = null;
        try { direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.IfNot(direct is not null && direct.IsAvailable, "GPU backend did not resolve.");
        using (direct)
        {
            var older = new DirectGpuTensorEngine(direct!);
            var newer = new DirectGpuTensorEngine(direct!);
            var backend = older.GetBackend();
            Assert.Same(newer, DirectGpuTensorEngine.EngineOwning(backend));
            newer.Dispose();
            Assert.Same(older, DirectGpuTensorEngine.EngineOwning(backend));
            older.Dispose();
            Assert.Null(DirectGpuTensorEngine.EngineOwning(backend));
        }
    }
}
