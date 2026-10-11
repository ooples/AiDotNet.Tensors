// Copyright (c) AiDotNet. All rights reserved.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGpuBufferViews
    {
        /// <inheritdoc/>
        /// <remarks>
        /// An OpenCL sub-buffer: kernels address it like any buffer. The device fixes the alignment a sub-buffer
        /// origin needs (CL_DEVICE_MEM_BASE_ADDR_ALIGN), so an offset off that alignment yields null, as the contract
        /// allows.
        /// </remarks>
        IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
        {
            if (parent is not DirectOpenClGpuBuffer owner) return null;
            if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
            var view = DirectOpenClBuffer.TryCreateView(owner.Buffer, elementOffset, elementCount);
            return view is null ? null : new DirectOpenClGpuBuffer(view);
        }
    }
}
#endif
