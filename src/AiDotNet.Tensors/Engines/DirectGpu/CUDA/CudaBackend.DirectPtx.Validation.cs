using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private static bool DirectPtxBufferIsInvalid(IGpuBuffer? buffer) =>
        buffer is null || buffer.Handle == IntPtr.Zero || buffer.SizeInBytes <= 0;

    private static bool DirectPtxBuffersOverlap(IGpuBuffer left, IGpuBuffer right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Handle);
        nuint rightStart = PtxCompat.ToNuint(right.Handle);
        try
        {
            nuint leftEnd = checked(leftStart + (nuint)left.SizeInBytes);
            nuint rightEnd = checked(rightStart + (nuint)right.SizeInBytes);
            return leftStart < rightEnd && rightStart < leftEnd;
        }
        catch (OverflowException)
        {
            // A malformed address range is never admissible to a raw-pointer ABI.
            return true;
        }
    }
}
