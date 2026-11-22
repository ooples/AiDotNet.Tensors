// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGpuBufferViews
    {
        /// <inheritdoc/>
        // A view is a sub-buffer object, created per view.
        bool IGpuBufferViews.ViewsAreFreeAtAnyOffset => false;

        /// <inheritdoc/>
        /// <remarks>
        /// An OpenCL sub-buffer: kernels address it like any buffer. The device fixes the alignment a sub-buffer
        /// origin needs (CL_DEVICE_MEM_BASE_ADDR_ALIGN), so an offset off that alignment yields null, as the contract
        /// allows.
        /// </remarks>
        IGpuBuffer? IGpuBufferViews.TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount)
        {
            if (parent is not DirectOpenClGpuBuffer owner) return null;
            // Offsets below are counted in 4-byte words: a parent of another element width has no such view.
            if (!GpuElementTypes.IsWordAddressed(owner.ElementType)) return null;
            if (elementOffset < 0 || elementCount <= 0 || (long)elementOffset + elementCount > parent.Size) return null;
            var view = DirectOpenClBuffer.TryCreateView(owner.Buffer, elementOffset, elementCount);
            if (view is null) return null;
            var typed = new DirectOpenClGpuBuffer(view, viewParent: owner);
            // The view is the parent's memory, so it is the parent's elements (an Int32 buffer's view is Int32).
            ((IRetypableGpuBuffer)typed).SetElementType(owner.ElementType);
            return typed;
        }
    }
}
