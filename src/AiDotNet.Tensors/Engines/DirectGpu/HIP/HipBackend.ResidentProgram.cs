using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1448: the resident program on HIP, a port of CUDA's (CudaBackend.ResidentProgram.cs): a chain of declared stages
// run by ONE persistent cooperative launch, every block walking the stage list with a grid barrier between stages.
// HIP's cooperative launch guarantees the co-residency the spin-wait barrier needs. The kernels come from
// ResidentProgramKernelTemplate; the parameter block is CUDA's ResidentProgramArgs, since the HIP kernel declares the
// same struct.
public sealed partial class HipBackend
{
    private IntPtr _residentProgramModule;
    private string? _residentProgramError;
    private bool _residentProgramTried;
    private int _residentProgramBlocksPerCu;
    private readonly object _residentProgramLock = new object();

    /// <summary>Whether this device can run a resident program (it needs a cooperative launch).</summary>
    internal bool SupportsResidentProgram => IsAvailable && _supportsCooperativeLaunch;

    /// <summary>
    /// Allocates the two-word grid barrier a resident program synchronizes on, zeroed; it stays valid across launches
    /// (the last arrival at every barrier resets the count).
    /// </summary>
    internal IGpuBuffer AllocateResidentBarrier() => AllocateBuffer(2);

    /// <summary>
    /// Runs <paramref name="stages"/> in order as ONE persistent cooperative launch, a grid barrier between stages.
    /// Returns the grid it used.
    /// </summary>
    internal unsafe int RunResidentProgram(ReadOnlySpan<ResidentStage> stages, IGpuBuffer barrier)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        if (!SupportsResidentProgram) throw new NotSupportedException("A resident program needs a cooperative launch, which this device does not support.");
        if (barrier is null || barrier.Size < 2)
            throw new ArgumentException("The barrier must be the two-word buffer from AllocateResidentBarrier.", nameof(barrier));
        IntPtr kernel = ResidentKernel(ResidentProgramKernelTemplate.HipProgramKernel);

        // Occupancy depends only on the kernel, its block size and its (static) shared memory: queried once rather than
        // on every launch of a chain that exists to cut launch cost.
        int perCu = _residentProgramBlocksPerCu;
        if (perCu == 0)
        {
            HipNativeBindings.CheckError(
                HipNativeBindings.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out perCu, kernel,
                    ResidentProgramKernelTemplate.GroupThreads, 0, 0),
                "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(resident_program)");
            _residentProgramBlocksPerCu = perCu;
        }
        int needed = 1;
        foreach (var s in stages) needed = Math.Max(needed, ResidentProgramKernelTemplate.GroupsFor(s));
        int grid = Math.Max(1, Math.Min(perCu * _deviceProps.MultiProcessorCount, needed));

        var args = new ResidentProgramArgs { Count = stages.Length, Bar = barrier.Handle };
        var slots = (ResidentStageArgs*)args.Stages;
        for (int i = 0; i < stages.Length; i++) slots[i] = ToResidentArgs(stages[i]);
        void** argv = stackalloc void*[1];
        argv[0] = &args;
        LaunchCooperativeKernel(kernel, (uint)grid, ResidentProgramKernelTemplate.GroupThreads, 0, argv);
        return grid;
    }

    /// <summary>The same stages, one ordinary launch each (the comparison a resident program is measured against).</summary>
    internal unsafe void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        IntPtr kernel = ResidentKernel(ResidentProgramKernelTemplate.StageKernel);
        void** argv = stackalloc void*[7];
        foreach (var stage in stages)
        {
            // An unread operand still gets a valid pointer: the input stands in.
            IntPtr a = stage.A.Handle, b = (stage.B ?? stage.A).Handle, c = (stage.C ?? stage.A).Handle, o = stage.Output.Handle;
            int op = (int)stage.Op, rows = stage.Rows, cols = stage.Cols;
            argv[0] = &a; argv[1] = &b; argv[2] = &c; argv[3] = &o; argv[4] = &op; argv[5] = &rows; argv[6] = &cols;
            LaunchKernel(kernel, (uint)ResidentProgramKernelTemplate.GroupsFor(stage), ResidentProgramKernelTemplate.GroupThreads, argv);
        }
    }

    private static ResidentStageArgs ToResidentArgs(in ResidentStage s) => new()
    {
        Op = (int)s.Op, Rows = s.Rows, Cols = s.Cols,
        A = s.A.Handle, B = s.B?.Handle ?? s.A.Handle, C = s.C?.Handle ?? s.A.Handle, Out = s.Output.Handle,
    };

    private IntPtr ResidentKernel(string name)
    {
        lock (_residentProgramLock)
        {
            if (!_residentProgramTried)
            {
                _residentProgramTried = true;
                // Strict math: the softmax's exp and division must round like the other backends' instances.
                var failure = CompileKernelModule(ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.Hip),
                    "resident_program", ref _residentProgramModule,
                    ResidentProgramKernelTemplate.GetKernelNames(ResidentKernelDialect.Hip), useFastMath: false);
                if (failure is not null)
                {
                    _residentProgramError = failure.Message;
                    System.Diagnostics.Trace.TraceWarning($"HIP resident program kernels unavailable: {failure.Message}");
                }
            }
            if (_residentProgramModule == IntPtr.Zero || !_kernelCache.TryGetValue(name, out var kernel))
                throw new GpuCapabilityUnavailableException("HIP resident program kernels are unavailable"
                    + (_residentProgramError is null ? "." : ": " + _residentProgramError));
            return kernel;
        }
    }
}
