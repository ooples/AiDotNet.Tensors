using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>A stage of a resident program (<see cref="CudaResidentProgramKernels"/>).</summary>
internal enum ResidentOp
{
    /// <summary><c>out[r,c] = max(0, a[r,c] * b[c] + c[c])</c>: a per-column scale and bias, then ReLU.</summary>
    AffineRelu = 0,

    /// <summary><c>out[r,c] = a[r,c] * b[c] + c[c]</c>.</summary>
    Affine = 1,

    /// <summary><c>out[r,:] = softmax(a[r,:])</c>, one row per block.</summary>
    RowSoftmax = 2,

    /// <summary><c>out = a + b</c>, elementwise.</summary>
    Add = 3,
}

/// <summary>One declared stage: the op and its operands over a [rows, cols] tensor (unused operands may be null).</summary>
internal readonly struct ResidentStage
{
    public ResidentStage(ResidentOp op, int rows, int cols, IGpuBuffer a, IGpuBuffer? b, IGpuBuffer? c, IGpuBuffer output)
    {
        if (rows <= 0 || cols <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "A stage needs a non-empty [rows, cols] shape.");
        if ((long)rows * cols > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(rows), "A stage is limited to int.MaxValue elements.");
        bool needsB = op != ResidentOp.RowSoftmax;
        bool needsC = op == ResidentOp.Affine || op == ResidentOp.AffineRelu;
        if (needsB && b is null) throw new ArgumentNullException(nameof(b), $"{op} reads a second operand.");
        if (needsC && c is null) throw new ArgumentNullException(nameof(c), $"{op} reads a bias operand.");
        Op = op; Rows = rows; Cols = cols; A = a; B = b; C = c; Output = output;
    }

    public ResidentOp Op { get; }
    public int Rows { get; }
    public int Cols { get; }
    public IGpuBuffer A { get; }
    public IGpuBuffer? B { get; }
    public IGpuBuffer? C { get; }
    public IGpuBuffer Output { get; }
}

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
        IntPtr kernel = ResidentKernel(CudaResidentProgramKernels.ProgramKernel);
        using var _ = PushContext();

        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out int perSm, kernel,
                CudaResidentProgramKernels.BlockThreads, 0, 0),
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(resident_program)");
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
