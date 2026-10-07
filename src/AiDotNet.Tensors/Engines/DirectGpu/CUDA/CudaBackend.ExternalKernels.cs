using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Consumption point for externally evolved kernels (<see cref="ExternalKernelArtifact"/>, produced for example by
/// <c>AiDotNet.Evolution.Ptx</c>): the PTX is JIT-loaded through the driver and registered as an
/// <see cref="TunedKernelOrigin.External"/> candidate in the slot of the op family its ABI names. It then competes
/// under the same local evidence gate as every built-in and generated candidate; the artifact's own evidence is
/// never trusted.
/// </summary>
public sealed partial class CudaBackend
{
    /// <summary>Directory of <c>*.kernel.json</c> artifacts loaded when the first tuned-kernel slot is created.</summary>
    public const string ExternalKernelArtifactsEnvironmentVariable = "AIDOTNET_KERNEL_ARTIFACTS_DIR";

    private readonly List<IntPtr> _externalKernelModules = new();
    private int _externalArtifactsLoaded;

    /// <summary>Rejections from the environment artifact directory, for diagnostics.</summary>
    internal List<string> ExternalKernelArtifactErrors { get; } = new();

    /// <summary>
    /// Loads an artifact's PTX and registers it as a candidate for its op family. Returns false, with the reason,
    /// when the artifact targets a newer SM than this device, fails to JIT, or the backend is capturing.
    /// </summary>
    public bool RegisterExternalKernel(ExternalKernelArtifact artifact, out string? rejection)
    {
        if (artifact is null) throw new ArgumentNullException(nameof(artifact));
        artifact.Validate();
        int deviceSm = _ccMajor * 10 + _ccMinor;
        if (artifact.TargetSm > deviceSm)
        {
            rejection = $"{artifact.CandidateId}: targets sm_{artifact.TargetSm}, device is sm_{deviceSm}";
            return false;
        }
        if (!IsAvailable || IsStreamCapturing())
        {
            rejection = $"{artifact.CandidateId}: backend unavailable or capturing";
            return false;
        }

        IntPtr function;
        try
        {
            using var _ = PushContext();
            IntPtr image = Marshal.StringToHGlobalAnsi(artifact.Ptx);
            IntPtr module;
            try
            {
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuModuleLoadData(out module, image),
                    "cuModuleLoadData(external " + artifact.CandidateId + ")");
            }
            finally
            {
                Marshal.FreeHGlobal(image);
            }
            var status = CudaNativeBindings.cuModuleGetFunction(out function, module, artifact.EntryPoint);
            if (status != CudaResult.Success)
            {
                CudaNativeBindings.cuModuleUnload(module);
                rejection = $"{artifact.CandidateId}: entry point {artifact.EntryPoint} not found ({status})";
                return false;
            }
            lock (_externalKernelModules) _externalKernelModules.Add(module);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ExternalException)
        {
            rejection = $"{artifact.CandidateId}: {ex.Message}";
            return false;
        }

        var launch = artifact.Launch;
        Func<TunedShape, bool> applicable = shape => artifact.Supports(shape);
        string id = artifact.CandidateId;
        bool deterministic = artifact.Deterministic;
        switch (artifact.Op)
        {
            case TunedKernelOp.Softmax:
                SoftmaxSlot.AddCandidate(new CudaTunedKernelCandidate<CudaSoftmaxArgs>(id, TunedKernelOrigin.External,
                    deterministic, applicable,
                    (in CudaSoftmaxArgs a) => LaunchExternal(function, launch, a.Rows, a.N, 2,
                        a.Input.Handle, a.Output.Handle)));
                break;
            case TunedKernelOp.SoftmaxBackward:
                SoftmaxBackwardSlot.AddCandidate(new CudaTunedKernelCandidate<CudaSoftmaxBackwardArgs>(id,
                    TunedKernelOrigin.External, deterministic, applicable,
                    (in CudaSoftmaxBackwardArgs a) => LaunchExternal(function, launch, a.Rows, a.N, 3,
                        a.GradOutput.Handle, a.Output.Handle, a.GradInput.Handle)));
                break;
            case TunedKernelOp.LayerNorm:
                LayerNormSlot.AddCandidate(new CudaTunedKernelCandidate<CudaLayerNormArgs>(id, TunedKernelOrigin.External,
                    deterministic, applicable,
                    (in CudaLayerNormArgs a) => LaunchExternalLayerNorm(function, launch, a)));
                break;
            case TunedKernelOp.LayerNormBackward:
                LayerNormBackwardSlot.AddCandidate(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(id,
                    TunedKernelOrigin.External, deterministic, applicable,
                    (in CudaNormBackwardArgs a) => LaunchExternal(function, launch, a.Rows, a.N, 6,
                        a.GradOutput.Handle, a.Input.Handle, Required(a.Gamma).Handle, a.Stat0.Handle,
                        Required(a.Stat1).Handle, a.Out0.Handle)));
                break;
            case TunedKernelOp.LayerNormGradParameters:
                LayerNormGradParametersSlot.AddCandidate(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(id,
                    TunedKernelOrigin.External, deterministic, applicable,
                    (in CudaNormBackwardArgs a) => LaunchExternal(function, launch, a.Rows, a.N, 6,
                        a.GradOutput.Handle, a.Input.Handle, a.Stat0.Handle, Required(a.Stat1).Handle,
                        a.Out0.Handle, Required(a.Out1).Handle)));
                break;
            case TunedKernelOp.RmsNormGradGamma:
                RmsNormGradGammaSlot.AddCandidate(new CudaTunedKernelCandidate<CudaNormBackwardArgs>(id,
                    TunedKernelOrigin.External, deterministic, applicable,
                    (in CudaNormBackwardArgs a) => LaunchExternal(function, launch, a.Rows, a.N, 4,
                        a.GradOutput.Handle, a.Input.Handle, a.Stat0.Handle, a.Out0.Handle)));
                break;
            default:
                rejection = $"{id}: op {artifact.Op} has no CUDA consumption point";
                return false;
        }
        rejection = null;
        return true;
    }

    /// <summary>Loads the environment's artifact directory once, when the first slot is created.</summary>
    private void EnsureExternalKernelArtifactsLoaded()
    {
        if (System.Threading.Interlocked.Exchange(ref _externalArtifactsLoaded, 1) != 0) return;
        string? dir = Environment.GetEnvironmentVariable(ExternalKernelArtifactsEnvironmentVariable);
        if (dir is null || dir.Length == 0) return;
        foreach (var artifact in ExternalKernelArtifact.LoadDirectory(dir, ExternalKernelArtifactErrors))
        {
            if (!RegisterExternalKernel(artifact, out string? rejection) && rejection is not null)
                ExternalKernelArtifactErrors.Add(rejection);
        }
        foreach (string error in ExternalKernelArtifactErrors)
            System.Diagnostics.Trace.TraceWarning("[kernel-registry] external artifact rejected: " + error);
    }

    private static uint ExternalGrid(ExternalKernelLaunch launch, int rows, int n)
    {
        long units = launch.GridRule == ExternalKernelGridRule.ColumnsPerBlock ? n : rows;
        return (uint)((units + launch.UnitsPerBlock - 1) / launch.UnitsPerBlock);
    }

    // The ABI fixes the pointer count; (rows, n) always follow the pointers.
    private unsafe void LaunchExternal(IntPtr function, ExternalKernelLaunch launch, int rows, int n, int pointers,
        IntPtr p0, IntPtr p1, IntPtr p2 = default, IntPtr p3 = default, IntPtr p4 = default, IntPtr p5 = default)
    {
        if (pointers < 2 || pointers > 6) throw new ArgumentOutOfRangeException(nameof(pointers));
        using var _ = PushContext();
        void** args = stackalloc void*[8];
        IntPtr* ptrs = stackalloc IntPtr[6];
        ptrs[0] = p0; ptrs[1] = p1; ptrs[2] = p2; ptrs[3] = p3; ptrs[4] = p4; ptrs[5] = p5;
        for (int i = 0; i < pointers; i++) args[i] = &ptrs[i];
        args[pointers] = &rows;
        args[pointers + 1] = &n;
        lock (GpuDispatchLock)
            LaunchKernel3D(function, ExternalGrid(launch, rows, n), 1, 1, (uint)launch.BlockX, (uint)launch.BlockY, 1,
                args, (uint)launch.SharedMemoryBytes);
    }

    private unsafe void LaunchExternalLayerNorm(IntPtr function, ExternalKernelLaunch launch, in CudaLayerNormArgs a)
    {
        using var _ = PushContext();
        IntPtr x = a.Input.Handle, y = a.Output.Handle, gamma = a.Gamma.Handle, beta = a.Beta.Handle;
        IntPtr mean = a.SaveMean.Handle, invStd = a.SaveInvVar.Handle;
        int rows = a.Rows, n = a.N;
        float eps = a.Epsilon;
        void** args = stackalloc void*[9];
        args[0] = &x; args[1] = &y; args[2] = &gamma; args[3] = &beta; args[4] = &mean; args[5] = &invStd;
        args[6] = &rows; args[7] = &n; args[8] = &eps;
        lock (GpuDispatchLock)
            LaunchKernel3D(function, ExternalGrid(launch, rows, n), 1, 1, (uint)launch.BlockX, (uint)launch.BlockY, 1,
                args, (uint)launch.SharedMemoryBytes);
    }

    private void DisposeExternalKernelModules()
    {
        lock (_externalKernelModules)
        {
            foreach (var module in _externalKernelModules)
            {
                try { CudaNativeBindings.cuModuleUnload(module); } catch { }
            }
            _externalKernelModules.Clear();
        }
    }
}
