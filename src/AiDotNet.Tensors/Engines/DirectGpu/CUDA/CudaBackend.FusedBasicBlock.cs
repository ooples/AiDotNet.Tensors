using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>The device arguments of <see cref="CudaFusedBasicBlockKernels"/>, laid out as its <c>FbbParams</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct FusedBasicBlockParams
{
    public IntPtr X, W1, G1, B1, W2, G2, B2;
    public IntPtr C1, C2, Stats, Y, Bar;
    public int N, C, H, W;
    public float Eps;
}

public sealed partial class CudaBackend
{
    private IntPtr _fusedBasicBlockModule;
    private string? _fusedBasicBlockError;
    private readonly object _fusedBasicBlockLock = new object();

    /// <summary>
    /// #1353 prototype: the shapes the persistent basic-block kernel runs. Channels and N * H * W must be whole
    /// 64-wide tiles; the device must support cooperative launch.
    /// </summary>
    internal bool SupportsFusedBasicBlock(int n, int c, int h, int w)
        => IsAvailable && _supportsCooperativeLaunch && n > 0 && c > 0 && h > 0 && w > 0
           && c % CudaFusedBasicBlockKernels.Tile == 0 && (long)n * h * w % CudaFusedBasicBlockKernels.Tile == 0
           && (long)n * c * h * w <= int.MaxValue;

    /// <summary>
    /// One launch of a ResNet basic block's training forward (see <see cref="CudaFusedBasicBlockKernels"/>):
    /// y = relu(bn2(conv2(relu(bn1(conv1(x))))) + x) with batch statistics. <paramref name="conv1Out"/> and
    /// <paramref name="conv2Out"/> hold N*C*H*W floats each, <paramref name="stats"/> 4*C floats (BN1 mean, BN1
    /// 1/std, BN2 mean, BN2 1/std, written), and <paramref name="barrier"/> two 32-bit words (zeroed here).
    /// Returns the grid it ran with.
    /// </summary>
    internal unsafe int FusedBasicBlockForward(IGpuBuffer x, IGpuBuffer w1, IGpuBuffer gamma1, IGpuBuffer beta1,
        IGpuBuffer w2, IGpuBuffer gamma2, IGpuBuffer beta2, IGpuBuffer conv1Out, IGpuBuffer conv2Out, IGpuBuffer stats,
        IGpuBuffer y, IGpuBuffer barrier, int n, int c, int h, int w, float eps, int gridOverride = 0)
    {
        if (!SupportsFusedBasicBlock(n, c, h, w))
            throw new NotSupportedException($"fused basic block: shape [{n},{c},{h},{w}] or this device is not supported.");
        IntPtr kernel = FusedBasicBlockKernel();
        using var _ = PushContext();

        // Every block must be resident at once; the work loops cover any grid, so take the device's full width,
        // capped where more blocks than tiles or channels would only wait at the barriers.
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(out int perSm, kernel,
                CudaFusedBasicBlockKernels.BlockThreads, 0, 0),
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(fbb_forward)");
        int tiles = c / CudaFusedBasicBlockKernels.Tile * (n * h * w / CudaFusedBasicBlockKernels.Tile);
        int grid = Math.Max(1, Math.Min(perSm * _multiProcessorCount, Math.Max(tiles, c)));
        if (gridOverride > 0) grid = Math.Min(gridOverride, perSm * _multiProcessorCount);

        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemsetD32Async(barrier.Handle, 0u, 2, _stream), "cuMemsetD32Async(fbb barrier)");
        var p = new FusedBasicBlockParams
        {
            X = x.Handle, W1 = w1.Handle, G1 = gamma1.Handle, B1 = beta1.Handle,
            W2 = w2.Handle, G2 = gamma2.Handle, B2 = beta2.Handle,
            C1 = conv1Out.Handle, C2 = conv2Out.Handle, Stats = stats.Handle, Y = y.Handle, Bar = barrier.Handle,
            N = n, C = c, H = h, W = w, Eps = eps,
        };
        void** args = stackalloc void*[1];
        args[0] = &p;
        LaunchCooperativeKernel(kernel, (uint)grid, CudaFusedBasicBlockKernels.BlockThreads, 0, args);
        return grid;
    }

    // Compiled on first use: a prototype route nothing else dispatches, so it costs a backend nothing until asked for.
    // Fast math is off so the statistics divide and take square roots exactly.
    private IntPtr FusedBasicBlockKernel()
    {
        lock (_fusedBasicBlockLock)
        {
            if (_fusedBasicBlockModule == IntPtr.Zero && _fusedBasicBlockError is null)
            {
                try
                {
                    using var _ = PushContext();
                    _fusedBasicBlockModule = CompileKernelModule(_cuDevice, CudaFusedBasicBlockKernels.GetSource(),
                        "fused_basic_block", CudaFusedBasicBlockKernels.GetKernelNames(), useFastMath: false);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    _fusedBasicBlockError = ex.Message;
                }
            }
            if (_fusedBasicBlockModule == IntPtr.Zero || !_kernelCache.TryGetValue(CudaFusedBasicBlockKernels.ForwardKernel, out var kernel))
                throw new InvalidOperationException("CUDA fused basic block kernel is unavailable"
                    + (_fusedBasicBlockError is null ? "." : ": " + _fusedBasicBlockError));
            return kernel;
        }
    }
}
