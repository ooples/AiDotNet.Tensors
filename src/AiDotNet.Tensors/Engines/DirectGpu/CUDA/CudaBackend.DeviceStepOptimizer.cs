using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Device-tables for the multi-tensor optimizer update of one compiled training plan: the parameter, gradient and
/// moment pointer arrays plus the chunk table, uploaded once and reused every step.
/// </summary>
/// <remarks>
/// Owned by the plan, not cached on the backend: a captured step graph bakes these device addresses into its kernel
/// nodes, so they must live exactly as long as the plan's buffers do. (The backend's single-slot gradient table is
/// shared by every plan and rebuilt whenever another plan uses it, which would free a table a graph still reads.)
/// </remarks>
internal sealed class MultiTensorOptimizerBinding : IDisposable
{
    internal IntPtr[] Handles = Array.Empty<IntPtr>();
    internal int[] Sizes = Array.Empty<int>();
    internal IGpuBuffer? ParamPtrs, GradPtrs, MPtrs, VPtrs, SizesBuffer, ChunkTensor, ChunkStart;
    internal int TotalChunks;

    /// <summary>True when this binding was built for exactly these buffers and sizes.</summary>
    internal bool Matches(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments, IReadOnlyList<int> sizes)
    {
        int n = Sizes.Length;
        if (parameters.Count != n || gradients.Count != n || firstMoments.Count != n || secondMoments.Count != n
            || sizes.Count != n)
            return false;
        for (int i = 0; i < n; i++)
        {
            if (Sizes[i] != sizes[i]
                || Handles[4 * i] != parameters[i].Handle || Handles[4 * i + 1] != gradients[i].Handle
                || Handles[4 * i + 2] != firstMoments[i].Handle || Handles[4 * i + 3] != secondMoments[i].Handle)
                return false;
        }
        return true;
    }

    public void Dispose()
    {
        ParamPtrs?.Dispose(); GradPtrs?.Dispose(); MPtrs?.Dispose(); VPtrs?.Dispose();
        SizesBuffer?.Dispose(); ChunkTensor?.Dispose(); ChunkStart?.Dispose();
        ParamPtrs = GradPtrs = MPtrs = VPtrs = SizesBuffer = ChunkTensor = ChunkStart = null;
    }
}

public sealed partial class CudaBackend
{
    /// <summary>
    /// Builds the device tables for <see cref="AdamMultiTensorUpdateDeviceStep"/> and
    /// <see cref="MultiTensorSumOfSquares(MultiTensorOptimizerBinding, IGpuBuffer)"/>. The caller owns the result.
    /// </summary>
    internal MultiTensorOptimizerBinding CreateMultiTensorOptimizerBinding(
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments, IReadOnlyList<int> sizes)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (firstMoments is null) throw new ArgumentNullException(nameof(firstMoments));
        if (secondMoments is null) throw new ArgumentNullException(nameof(secondMoments));
        if (sizes is null) throw new ArgumentNullException(nameof(sizes));
        int n = parameters.Count;
        if (n == 0 || gradients.Count != n || firstMoments.Count != n || secondMoments.Count != n || sizes.Count != n)
            throw new ArgumentException("Multi-tensor optimizer buffer lists must be non-empty and the same length.");

        var handles = new IntPtr[4 * n];
        var sizeArray = new int[n];
        var paramAddr = new byte[n * sizeof(ulong)];
        var gradAddr = new byte[n * sizeof(ulong)];
        var mAddr = new byte[n * sizeof(ulong)];
        var vAddr = new byte[n * sizeof(ulong)];
        int totalChunks = 0;
        for (int t = 0; t < n; t++)
        {
            int size = sizes[t];
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(sizes), "Every tensor size must be positive.");
            if (parameters[t].Size < size || gradients[t].Size < size || firstMoments[t].Size < size
                || secondMoments[t].Size < size)
                throw new ArgumentException("A buffer is smaller than its tensor size.", nameof(sizes));
            handles[4 * t] = parameters[t].Handle;
            handles[4 * t + 1] = gradients[t].Handle;
            handles[4 * t + 2] = firstMoments[t].Handle;
            handles[4 * t + 3] = secondMoments[t].Handle;
            WriteAddress(paramAddr, t, handles[4 * t]);
            WriteAddress(gradAddr, t, handles[4 * t + 1]);
            WriteAddress(mAddr, t, handles[4 * t + 2]);
            WriteAddress(vAddr, t, handles[4 * t + 3]);
            sizeArray[t] = size;
            totalChunks += (size + MultiTensorChunk - 1) / MultiTensorChunk;
        }
        var chunkTensor = new int[totalChunks];
        var chunkStart = new int[totalChunks];
        int c = 0;
        for (int t = 0; t < n; t++)
        {
            int chunks = (sizeArray[t] + MultiTensorChunk - 1) / MultiTensorChunk;
            for (int k = 0; k < chunks; k++) { chunkTensor[c] = t; chunkStart[c] = k * MultiTensorChunk; c++; }
        }

        var binding = new MultiTensorOptimizerBinding { Handles = handles, Sizes = sizeArray, TotalChunks = totalChunks };
        try
        {
            using (PushContext())
            {
                binding.ParamPtrs = AllocateByteBuffer(paramAddr.Length); UploadByteBuffer(binding.ParamPtrs, paramAddr);
                binding.GradPtrs = AllocateByteBuffer(gradAddr.Length); UploadByteBuffer(binding.GradPtrs, gradAddr);
                binding.MPtrs = AllocateByteBuffer(mAddr.Length); UploadByteBuffer(binding.MPtrs, mAddr);
                binding.VPtrs = AllocateByteBuffer(vAddr.Length); UploadByteBuffer(binding.VPtrs, vAddr);
                binding.SizesBuffer = AllocateIntBuffer(sizeArray);
                binding.ChunkTensor = AllocateIntBuffer(chunkTensor);
                binding.ChunkStart = AllocateIntBuffer(chunkStart);
            }
        }
        catch
        {
            binding.Dispose();
            throw;
        }
        return binding;
    }

    /// <summary>
    /// Sum of squares of every gradient in <paramref name="binding"/>, accumulated in double into
    /// <paramref name="sumOfSquares"/> (two float slots). The same reduction as
    /// <see cref="MultiTensorSumOfSquares(IReadOnlyList{IGpuBuffer}, IReadOnlyList{int}, IGpuBuffer)"/>, over the
    /// plan's own tables.
    /// </summary>
    internal unsafe void MultiTensorSumOfSquares(MultiTensorOptimizerBinding binding, IGpuBuffer sumOfSquares)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (sumOfSquares is null) throw new ArgumentNullException(nameof(sumOfSquares));
        if (sumOfSquares.Size < 2) throw new ArgumentException("The double result needs two float slots.", nameof(sumOfSquares));
        if (binding.GradPtrs is null) throw new ObjectDisposedException(nameof(binding));
        Fill(sumOfSquares, 0f, 2);
        if (!_kernelCache.TryGetValue("multi_tensor_sum_squares", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: multi_tensor_sum_squares");
        using var _ = PushContext();
        IntPtr pH = binding.GradPtrs.Handle, sH = binding.SizesBuffer!.Handle;
        IntPtr ctH = binding.ChunkTensor!.Handle, csH = binding.ChunkStart!.Handle;
        IntPtr outH = sumOfSquares.Handle;
        int totalChunks = binding.TotalChunks;
        void** args = stackalloc void*[6];
        args[0] = &pH; args[1] = &sH; args[2] = &ctH; args[3] = &csH; args[4] = &totalChunks; args[5] = &outH;
        LaunchKernel(kernel, (uint)Math.Min(totalChunks, MultiTensorReductionMaxBlocks), MultiTensorChunk, args);
    }

    /// <summary>
    /// Decides on the device whether this step's gradients are finite (<paramref name="sumOfSquares"/> from the
    /// gradient reduction) and advances the device step state (see <c>optimizer_step_prepare</c>).
    /// </summary>
    internal unsafe void OptimizerStepPrepare(IGpuBuffer sumOfSquares, IGpuBuffer stepState)
    {
        if (sumOfSquares is null) throw new ArgumentNullException(nameof(sumOfSquares));
        if (stepState is null) throw new ArgumentNullException(nameof(stepState));
        if (!_kernelCache.TryGetValue("optimizer_step_prepare", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: optimizer_step_prepare");
        using var _ = PushContext();
        IntPtr sH = sumOfSquares.Handle, stH = stepState.Handle;
        void** args = stackalloc void*[2];
        args[0] = &sH; args[1] = &stH;
        LaunchKernel(kernel, 1, 1, args);
    }

    /// <summary>
    /// Multi-tensor Adam (<paramref name="decoupledWeightDecay"/> false) or AdamW (true) whose step, learning rate
    /// and discard decision come from device memory (<paramref name="stepState"/>, <paramref name="learningRates"/>),
    /// so it can run without a host read and be captured into a step graph. Per-element math is identical to
    /// <see cref="AdamMultiTensorUpdate"/> / <see cref="AdamWMultiTensorUpdate"/>.
    /// </summary>
    internal unsafe void AdamMultiTensorUpdateDeviceStep(MultiTensorOptimizerBinding binding, IGpuBuffer stepState,
        IGpuBuffer learningRates, float beta1, float beta2, float epsilon, float weightDecay, bool decoupledWeightDecay)
    {
        if (binding is null) throw new ArgumentNullException(nameof(binding));
        if (stepState is null) throw new ArgumentNullException(nameof(stepState));
        if (learningRates is null) throw new ArgumentNullException(nameof(learningRates));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        if (binding.ParamPtrs is null) throw new ObjectDisposedException(nameof(binding));
        if (!_kernelCache.TryGetValue("adam_multi_tensor_update_device_step", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: adam_multi_tensor_update_device_step");
        using var _ = PushContext();
        IntPtr pH = binding.ParamPtrs.Handle, gH = binding.GradPtrs!.Handle, mH = binding.MPtrs!.Handle;
        IntPtr vH = binding.VPtrs!.Handle, szH = binding.SizesBuffer!.Handle;
        IntPtr ctH = binding.ChunkTensor!.Handle, csH = binding.ChunkStart!.Handle;
        IntPtr stH = stepState.Handle, lrH = learningRates.Handle;
        int decoupled = decoupledWeightDecay ? 1 : 0;
        void** args = stackalloc void*[14];
        args[0] = &pH; args[1] = &gH; args[2] = &mH; args[3] = &vH;
        args[4] = &szH; args[5] = &ctH; args[6] = &csH; args[7] = &stH; args[8] = &lrH;
        args[9] = &beta1; args[10] = &beta2; args[11] = &epsilon; args[12] = &weightDecay; args[13] = &decoupled;
        LaunchKernel(kernel, (uint)binding.TotalChunks, MultiTensorChunk, args);
    }
}
