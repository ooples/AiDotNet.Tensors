// Copyright (c) AiDotNet. All rights reserved.
// CUDA IGpuBatchExecution (#1142). The fused operations already existed; this adds the batch plumbing and the
// launcher for the batch kernels shared with HIP.
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IMeshgridBackend, IGpuBatchExecution
{
    private int _batchDepth;
    private int _secondaryStreamOwnerThreadId;

    private unsafe void LaunchBatchKernel(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        ulong* slots = stackalloc ulong[args.Length];
        void** pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case IGpuBuffer buffer: *(IntPtr*)(slots + i) = buffer.Handle; break;
                case int value: *(int*)(slots + i) = value; break;
                case float value: *(float*)(slots + i) = value; break;
                case ulong value: slots[i] = value; break;
                default:
                    throw new ArgumentException($"Unsupported argument type {args[i]?.GetType().Name ?? "null"} for {name}.");
            }
            pointers[i] = slots + i;
        }
        using var _ = PushContext();
        LaunchKernel(kernel, (uint)((threads + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, pointers);
    }

    /// <summary>
    /// Whole-tensor reduction in one block: deterministic in both determinism modes (#382), and it overwrites
    /// <paramref name="output"/> instead of accumulating into it.
    /// </summary>
    private unsafe void BatchFullReduce(IGpuBuffer input, IGpuBuffer output, int size, int mode, float maxVal)
    {
        if (!_kernelCache.TryGetValue("bx_full_reduce", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: bx_full_reduce");
        using var _ = PushContext();
        IntPtr inPtr = input.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &size; args[3] = &mode; args[4] = &maxVal;
        LaunchKernel(kernel, 1, K.FullReduceBlockSize, args);
    }

    public bool SupportsBatchExecution => true;

    /// <inheritdoc/>
    public IGpuBuffer AllocateWorkspaceBuffer(int totalElements) => AllocateBuffer(totalElements);

    /// <summary>
    /// Launches go to one in-order stream and never wait on the host, so a batch is already one uninterrupted
    /// submission; <see cref="EndBatch"/> waits for it like Vulkan's submit-and-wait.
    /// </summary>
    public void BeginBatch()
    {
        if (Interlocked.CompareExchange(ref _batchDepth, 1, 0) != 0)
            throw new InvalidOperationException("Already in batch mode.");
    }

    public void EndBatch()
    {
        if (Interlocked.CompareExchange(ref _batchDepth, 0, 1) != 1)
            throw new InvalidOperationException("Not in batch mode.");
        Synchronize();
    }

    /// <summary>In-order stream: every launch already observes the writes of the launches before it.</summary>
    public void InsertBarrier(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
    }

    /// <summary>
    /// Work recorded between the calls stays on the in-order stream, so it is ordered after earlier launches;
    /// <see cref="EndSecondaryStream"/> waits for it, matching Vulkan's secondary submit.
    /// </summary>
    public void BeginSecondaryStream()
    {
        int thread = Environment.CurrentManagedThreadId;
        if (Interlocked.CompareExchange(ref _secondaryStreamOwnerThreadId, thread, 0) != 0)
            throw new InvalidOperationException("Secondary stream is already owned by another thread.");
    }

    public void EndSecondaryStream()
    {
        int thread = Environment.CurrentManagedThreadId;
        if (Interlocked.CompareExchange(ref _secondaryStreamOwnerThreadId, 0, thread) != thread)
            throw new InvalidOperationException("Not in a secondary stream on this thread.");
        Synchronize();
    }

    public void Meshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0, IGpuBuffer output1,
        int n0, int n1, bool xy)
    {
        if (n0 <= 0) throw new ArgumentOutOfRangeException(nameof(n0));
        if (n1 <= 0) throw new ArgumentOutOfRangeException(nameof(n1));
        LaunchBatchKernel("bx_meshgrid2d", checked(n0 * n1), source0, source1, output0, output1, n0, n1, xy ? 1 : 0);
    }
}
