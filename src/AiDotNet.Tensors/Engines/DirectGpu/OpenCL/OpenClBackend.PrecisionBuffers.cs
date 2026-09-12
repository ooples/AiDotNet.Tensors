using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend
{
    // Precision describes kernel storage, not the CLR wrapper used to allocate its cl_mem.
    // Both the legacy float-sized allocation and an exact byte-sized allocation are valid.
    private IntPtr GetPrecisionBufferHandle(IGpuBuffer buffer, long requiredBytes)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (requiredBytes < 0 || buffer.SizeInBytes < requiredBytes)
            throw new ArgumentException($"Precision kernel requires {requiredBytes} bytes, but the buffer has {buffer.SizeInBytes}.", nameof(buffer));
        IDirectOpenClMemoryObject memory = buffer switch
        {
            DirectOpenClGpuBuffer floats => floats.Buffer,
            DirectOpenClGpuByteBuffer bytes => bytes.Buffer,
            _ => throw new ArgumentException("The buffer must be owned by the OpenCL backend.", nameof(buffer)),
        };
        if (_context is null || memory.OwningContext.Context != _context.Context)
            throw new ArgumentException("The buffer must belong to this OpenCL backend's context.", nameof(buffer));
        IntPtr handle = memory.NativeHandle;
        if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(buffer));
        return handle;
    }
}
