using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1448: the resident program on HIP. A row-local chain is ONE launch of resident_rows, each block taking a row through
// every stage with only block barriers, so it needs no cooperative launch. Any other chain is a port of CUDA's
// persistent kernel (CudaBackend.ResidentProgram.cs): every block walks the stage list with a grid barrier between
// stages, which HIP's cooperative launch makes safe by guaranteeing co-residency. The kernels come from
// ResidentProgramKernelTemplate; the parameter block is CUDA's ResidentProgramArgs, since the HIP kernel declares the
// same struct.
public sealed partial class HipBackend
{
    private IntPtr _residentProgramModule;
    private string? _residentProgramError;
    private bool _residentProgramTried;
    private int _residentProgramBlocksPerCu;
    private readonly object _residentProgramLock = new object();

    /// <summary>Whether this device can run a resident program. A row-local chain needs nothing more than HIP; any other
    /// chain uses the persistent kernel when <see cref="SupportsCooperativeResidentProgram"/>, else ordinary launches.</summary>
    internal bool SupportsResidentProgram => IsAvailable;

    /// <summary>Whether a chain that is not row-local can run as one persistent cooperative launch.</summary>
    internal bool SupportsCooperativeResidentProgram => IsAvailable && _supportsCooperativeLaunch;

    /// <summary>
    /// Allocates the two-word grid barrier a resident program synchronizes on, zeroed; it stays valid across launches
    /// (the last arrival at every barrier resets the count).
    /// </summary>
    internal IGpuBuffer AllocateResidentBarrier() => AllocateBuffer(2);

    /// <summary>
    /// Runs <paramref name="stages"/> in order as ONE launch: a row-local chain
    /// (<see cref="ResidentProgramKernelTemplate.TryPlanRowProgram"/>) as resident_rows, each block taking a row
    /// through every stage; any other chain as the persistent cooperative kernel, a grid barrier between stages, on
    /// <paramref name="barrier"/>. Without a cooperative launch such a chain runs as one launch per stage. Returns the
    /// number of launches.
    /// </summary>
    internal unsafe int RunResidentProgram(ReadOnlySpan<ResidentStage> stages, IGpuBuffer barrier)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        if (!SupportsResidentProgram) throw new GpuCapabilityUnavailableException("HIP is not available.");
        if (ResidentProgramKernelTemplate.TryPlanRowProgram(stages, out var plan) && plan is not null)
        {
            LaunchRowProgram(plan);
            return 1;
        }
        if (!SupportsCooperativeResidentProgram)
        {
            RunStagesSeparately(stages);
            return stages.Length;
        }
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
        return 1;
    }

    private unsafe void LaunchRowProgram(ResidentProgramKernelTemplate.RowProgramPlan plan)
    {
        IntPtr kernel = ResidentKernel(ResidentProgramKernelTemplate.RowProgramKernel);
        const int slots = ResidentProgramKernelTemplate.MaxRowProgramBuffers;
        IntPtr* buffers = stackalloc IntPtr[slots];
        for (int i = 0; i < slots; i++) buffers[i] = (i < plan.Buffers.Length ? plan.Buffers[i] : plan.Buffers[0]).Handle;
        int* p = stackalloc int[4 + ResidentProgramKernelTemplate.MaxStages];
        var parameters = plan.Parameters();
        for (int i = 0; i < parameters.Length; i++) p[i] = parameters[i];
        void** argv = stackalloc void*[slots + 4 + ResidentProgramKernelTemplate.MaxStages];
        for (int i = 0; i < slots; i++) argv[i] = &buffers[i];
        for (int i = 0; i < parameters.Length; i++) argv[slots + i] = &p[i];
        LaunchKernel(kernel, (uint)plan.Groups, ResidentProgramKernelTemplate.GroupThreads, argv);
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
