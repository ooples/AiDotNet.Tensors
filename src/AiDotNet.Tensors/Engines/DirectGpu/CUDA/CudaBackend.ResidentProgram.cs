using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

[StructLayout(LayoutKind.Sequential)]
internal struct ResidentStageArgs
{
    public int Op, Rows, Cols, Pad;
    public IntPtr A, B, C, Out;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct ResidentProgramArgs
{
    public fixed byte Stages[CudaResidentProgramKernels.MaxStages * 48];   // ResidentStageArgs: 4 ints + 4 pointers
    public int Count, Pad;
    public IntPtr Bar;
}

public sealed partial class CudaBackend
{
    private IntPtr _residentProgramModule;
    private string? _residentProgramError;
    private int _residentProgramBlocksPerSm;
    private readonly object _residentProgramLock = new object();

    /// <summary>Whether this device can run a resident program (it needs a cooperative launch).</summary>
    internal bool SupportsResidentProgram => IsAvailable && _supportsCooperativeLaunch;

    /// <summary>
    /// Allocates the two-word grid barrier a resident program synchronizes on, zeroed once; it stays valid across
    /// launches (the last arrival at every barrier resets the count).
    /// </summary>
    internal IGpuBuffer AllocateResidentBarrier()
    {
        var bar = AllocateBuffer(2);
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemsetD32Async(bar.Handle, 0u, 2, _stream), "cuMemsetD32Async(resident barrier)");
        return bar;
    }

    /// <summary>
    /// Runs <paramref name="stages"/> in order as ONE persistent cooperative launch, a grid barrier between stages.
    /// Returns the grid it used.
    /// </summary>
    internal unsafe int RunResidentProgram(ReadOnlySpan<ResidentStage> stages, IGpuBuffer barrier)
    {
        if (stages.Length == 0) throw new ArgumentException("A resident program needs at least one stage.", nameof(stages));
        if (stages.Length > CudaResidentProgramKernels.MaxStages)
            throw new ArgumentException($"A resident program holds at most {CudaResidentProgramKernels.MaxStages} stages.", nameof(stages));
        if (!SupportsResidentProgram) throw new NotSupportedException("A resident program needs a cooperative launch, which this device does not support.");
        if (barrier is null || barrier.Size < 2)
            throw new ArgumentException("The barrier must be the two-word buffer from AllocateResidentBarrier.", nameof(barrier));
        IntPtr kernel = ResidentKernel(CudaResidentProgramKernels.ProgramKernel);
        using var _ = PushContext();

        // Occupancy depends only on the kernel, its block size and its (zero) dynamic shared memory, all fixed once
        // the module is compiled: queried once rather than on every launch of a chain that exists to cut launch cost.
        int perSm = _residentProgramBlocksPerSm;
        if (perSm == 0)
        {
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out perSm, kernel,
                    CudaResidentProgramKernels.BlockThreads, 0, 0),
                "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(resident_program)");
            _residentProgramBlocksPerSm = perSm;
        }
        int needed = 1;
        foreach (var s in stages) needed = Math.Max(needed, BlocksFor(s));
        int grid = Math.Max(1, Math.Min(perSm * _multiProcessorCount, needed));

        var args = new ResidentProgramArgs { Count = stages.Length, Bar = barrier.Handle };
        var slots = (ResidentStageArgs*)args.Stages;
        for (int i = 0; i < stages.Length; i++) slots[i] = ToArgs(stages[i]);
        void** argv = stackalloc void*[1];
        argv[0] = &args;
        LaunchCooperativeKernel(kernel, (uint)grid, CudaResidentProgramKernels.BlockThreads, 0, argv);
        return grid;
    }

    /// <summary>The same stages, one ordinary launch each (the comparison a resident program is measured against).</summary>
    internal unsafe void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
    {
        IntPtr kernel = ResidentKernel(CudaResidentProgramKernels.StageKernel);
        using var _ = PushContext();
        ResidentStageArgs a;
        void** argv = stackalloc void*[1];
        argv[0] = &a;
        foreach (var stage in stages)
        {
            a = ToArgs(stage);
            LaunchKernelWithSharedMem(kernel, (uint)BlocksFor(stage), CudaResidentProgramKernels.BlockThreads, 0, argv);
        }
    }

    private static int BlocksFor(in ResidentStage s)
        => s.Op == ResidentOp.RowSoftmax
            ? s.Rows
            : (int)Math.Min(int.MaxValue, ((long)s.Rows * s.Cols + CudaResidentProgramKernels.BlockThreads - 1) / CudaResidentProgramKernels.BlockThreads);

    private static ResidentStageArgs ToArgs(in ResidentStage s) => new()
    {
        Op = (int)s.Op, Rows = s.Rows, Cols = s.Cols,
        A = s.A.Handle, B = s.B?.Handle ?? IntPtr.Zero, C = s.C?.Handle ?? IntPtr.Zero, Out = s.Output.Handle,
    };

    private IntPtr ResidentKernel(string name)
    {
        lock (_residentProgramLock)
        {
            if (_residentProgramModule == IntPtr.Zero && _residentProgramError is null)
            {
                try
                {
                    using var _ = PushContext();
                    _residentProgramModule = CompileKernelModule(_cuDevice, CudaResidentProgramKernels.GetSource(),
                        "resident_program", CudaResidentProgramKernels.GetKernelNames(), useFastMath: false);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    _residentProgramError = ex.Message;
                }
            }
            if (_residentProgramModule == IntPtr.Zero || !_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException("CUDA resident program kernels are unavailable"
                    + (_residentProgramError is null ? "." : ": " + _residentProgramError));
            return kernel;
        }
    }
}
