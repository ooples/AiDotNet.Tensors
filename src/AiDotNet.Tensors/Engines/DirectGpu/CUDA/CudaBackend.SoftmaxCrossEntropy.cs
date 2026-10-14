using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Fused soft-target softmax cross-entropy (see ISoftmaxCrossEntropyBackend).
public sealed partial class CudaBackend : ISoftmaxCrossEntropyBackend
{
    unsafe bool ISoftmaxCrossEntropyBackend.TrySoftmaxCrossEntropyForward(IGpuBuffer logits, IGpuBuffer target,
        IGpuBuffer rowStats, IGpuBuffer loss, IGpuBuffer count, int rows, int cols)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || logits.Size < n || target.Size < n || rowStats.Size < 4L * rows
            || loss.Size < 1 || count.Size < 1
            || !_kernelCache.TryGetValue("softmax_xent_rows", out var rowsKernel)
            || !_kernelCache.TryGetValue("softmax_xent_total", out var totalKernel))
            return false;
        using var _ = PushContext();
        IntPtr z = logits.Handle, t = target.Handle, stats = rowStats.Handle, l = loss.Handle, c = count.Handle;
        int r = rows, k = cols;
        void** rargs = stackalloc void*[5];
        rargs[0] = &z; rargs[1] = &t; rargs[2] = &stats; rargs[3] = &r; rargs[4] = &k;
        // One warp per row, eight rows per block.
        LaunchKernel(rowsKernel, (uint)((rows + 7) / 8), DefaultBlockSize, rargs);
        void** targs = stackalloc void*[4];
        targs[0] = &stats; targs[1] = &l; targs[2] = &c; targs[3] = &r;
        LaunchKernel(totalKernel, 1, DefaultBlockSize, targs);
        return true;
    }

    unsafe bool ISoftmaxCrossEntropyBackend.TrySoftmaxCrossEntropyBackward(IGpuBuffer logits, IGpuBuffer target,
        IGpuBuffer rowStats, IGpuBuffer count, IGpuBuffer gradOutput, IGpuBuffer gradLogits, int rows, int cols, bool add)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || logits.Size < n || target.Size < n || gradLogits.Size < n
            || rowStats.Size < 4L * rows || count.Size < 1 || gradOutput.Size < 1
            || !_kernelCache.TryGetValue("softmax_xent_backward", out var kernel))
            return false;
        using var _ = PushContext();
        IntPtr z = logits.Handle, t = target.Handle, stats = rowStats.Handle, c = count.Handle, g = gradOutput.Handle,
            d = gradLogits.Handle;
        int r = rows, k = cols, a = add ? 1 : 0;
        void** args = stackalloc void*[9];
        args[0] = &z; args[1] = &t; args[2] = &stats; args[3] = &c; args[4] = &g; args[5] = &d;
        args[6] = &r; args[7] = &k; args[8] = &a;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        return true;
    }
}
