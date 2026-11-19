// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly object _realFftLock = new();
    private readonly Dictionary<(int N, int Rows, bool Inverse, IntPtr Stream), RealFftPlan> _realFftPlans = new();
    private bool _realFftUnavailable;
    internal long RealFftExecutions { get; private set; }
    internal int RealFftVersion => CuFftNative.Version;

    // The backend/context owns native plans. Each execution borrows its caller
    // workspace from the plan's action pool, which owns captured pointer lifetime.
    internal RealFftPlan? GetRealFftPlan(int n, int rows, bool inverse)
    {
        using var context = PushContext();
        lock (_realFftLock)
        {
            var key = (n, rows, inverse, _stream);
            if (_realFftPlans.TryGetValue(key, out var ready)) return ready;
            if (_realFftUnavailable) return null;
            if (IsStreamCapturing())
                throw new InvalidOperationException("Real FFT native plans must be warmed before CUDA graph capture.");
            if (!CuFftNative.Available) { _realFftUnavailable = true; return null; }
            var plan = CreateRealFftPlan(n, rows, inverse, _stream);
            _realFftPlans.Add(key, plan);
            return plan;
        }
    }

    // Capture uses a private stream. Prepare native handles before capture,
    // following the existing cuDNN stream-preparation contract. Only shapes
    // already warmed by the real eager body are prepared.
    private void PrepareRealFftStream(IntPtr stream)
    {
        lock (_realFftLock)
        {
            foreach (var key in _realFftPlans.Keys.Where(key => key.Stream == _mainStream).ToArray())
            {
                var captureKey = (key.N, key.Rows, key.Inverse, stream);
                if (!_realFftPlans.ContainsKey(captureKey))
                    _realFftPlans.Add(captureKey, CreateRealFftPlan(key.N, key.Rows, key.Inverse, stream));
            }
        }
    }

    private RealFftPlan CreateRealFftPlan(int n, int rows, bool inverse, IntPtr stream)
    {
        CuFftNative.Check(CuFftNative.Create(out int handle), "cufftCreate");
        try
        {
            CuFftNative.Check(CuFftNative.SetAutoAllocation(handle, 0), "cufftSetAutoAllocation(false)");
            CuFftNative.Check(CuFftNative.MakePlan1d(handle, n, inverse ? 0x29 : 0x2a, rows, out _), "cufftMakePlan1d");
            CuFftNative.Check(CuFftNative.SetStream(handle, stream), "cufftSetStream");
            CuFftNative.Check(CuFftNative.GetSize(handle, out var bytes), "cufftGetSize");
            return new RealFftPlan(this, handle, checked((int)((bytes.ToUInt64() + 3) / 4)), inverse, stream);
        }
        catch { CuFftNative.Destroy(handle); throw; }
    }

    internal void ExecuteRealFft(RealFftPlan plan, IGpuBuffer input, IGpuBuffer output, IGpuBuffer? workspace)
    {
        using var context = PushContext();
        lock (_realFftLock)
        {
            if (plan.Stream != _stream || plan.Handle == 0)
                throw new InvalidOperationException("Real FFT plan belongs to another stream or has been disposed.");
            if (plan.WorkspaceFloats > 0)
            {
                if (workspace is null || workspace.Size < plan.WorkspaceFloats)
                    throw new ArgumentException("The real FFT caller workspace is too small.", nameof(workspace));
                CuFftNative.Check(CuFftNative.SetWorkArea(plan.Handle, workspace.Handle), "cufftSetWorkArea");
            }
            CuFftNative.Check(plan.Inverse
                ? CuFftNative.ExecC2C(plan.Handle, input.Handle, output.Handle, 1)
                : CuFftNative.ExecR2C(plan.Handle, input.Handle, output.Handle), plan.Inverse ? "cufftExecC2C" : "cufftExecR2C");
            GpuLaunchProbe.OnLaunch(); // One successful native vendor dispatch, not its internal kernel count.
            RealFftExecutions++;
        }
    }

    // Called during the backend's existing synchronized teardown, before its context dies.
    private void DisposeRealFftPlans()
    {
        lock (_realFftLock)
        {
            foreach (IntPtr stream in _realFftPlans.Values.Select(plan => plan.Stream).Distinct())
            {
                var status = CudaNativeBindings.cuStreamSynchronize(stream);
                if (status != CudaResult.Success)
                    System.Diagnostics.Trace.TraceWarning($"CUDA stream synchronization during FFT teardown returned {status}.");
            }
            foreach (var plan in _realFftPlans.Values)
            {
                if (plan.Handle == 0) continue;
                int status = CuFftNative.Destroy(plan.Handle);
                if (status != 0)
                    System.Diagnostics.Trace.TraceWarning($"cufftDestroy({plan.Handle}) returned {status} during teardown.");
                plan.Handle = 0;
            }
            _realFftPlans.Clear();
        }
    }

    internal sealed class RealFftPlan : IRfftNativePlan
    {
        private readonly CudaBackend _owner;
        internal int Handle;
        internal int WorkspaceFloats { get; }
        internal bool Inverse { get; }
        internal IntPtr Stream { get; }
        int IRfftNativePlan.WorkspaceFloats => WorkspaceFloats;
        void IRfftNativePlan.Execute(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? workspace)
            => _owner.ExecuteRealFft(this, input, output, workspace);
        internal RealFftPlan(CudaBackend owner, int handle, int workspaceFloats, bool inverse, IntPtr stream)
        { _owner = owner; Handle = handle; WorkspaceFloats = workspaceFloats; Inverse = inverse; Stream = stream; }
    }
}
