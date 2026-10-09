using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // lstm_forward_sequence orders each row's timesteps with a work-group barrier, so a batch row must be one
    // work-group: H is bounded by the kernel's work-group limit on this device.
    public sealed partial class OpenClBackend : IFusedLstmSequenceTraining
    {
        int IFusedLstmSequenceTraining.MaxFusedLstmHidden => MaxLstmSequenceHidden();

        private int MaxLstmSequenceHidden()
        {
            if (_context == null || !_kernelCache.TryGetValue("lstm_forward_sequence", out var kernel)) return 0;
            int limit = (int)Math.Min(_context.MaxWorkGroupSize, 1024UL);
            var kernelMax = OpenClNativeBindings.GetKernelWorkGroupInfoSizeT(
                kernel.Handle, _context.Device, OpenClNativeBindings.CL_KERNEL_WORK_GROUP_SIZE);
            if (kernelMax != UIntPtr.Zero) limit = (int)Math.Min((ulong)limit, kernelMax.ToUInt64());
            return limit;
        }
    }
}