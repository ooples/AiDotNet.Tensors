using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The global-norm clip as one work-group reduction per tensor (no device address tables on OpenCL), the OpenCL
    // port of CudaBackend's IMultiTensorKernels. The per-tensor launches accumulate a float sum and a non-finite mark;
    // a last launch finishes them as the double CUDA writes (NaN when any element was not finite), so the same buffer
    // also serves IDeviceScalarOps and the device-guarded optimizer updates (#1502).
    public sealed partial class OpenClBackend : IMultiTensorKernels
    {
        public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
        {
            MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
            MultiTensorArgs.Validate(tensors, sizes);
            Fill(sumOfSquares, 0f, 2);
            var kernel = _kernelCache["tensor_sum_squares_accumulate"];
            var acc = ((DirectOpenClGpuBuffer)sumOfSquares).Buffer.Handle;
            for (int t = 0; t < tensors.Count; t++)
            {
                kernel.SetArg(0, ((DirectOpenClGpuBuffer)tensors[t]).Buffer.Handle);
                kernel.SetArg(1, sizes[t]);
                kernel.SetArg(2, acc);
                kernel.Execute1D(MultiTensorArgs.ReductionGroupSize, MultiTensorArgs.ReductionGroupSize);
            }
            FinishSumOfSquaresDouble(sumOfSquares);
        }

        // Turns the float sum and non-finite mark in the buffer's two slots into the double they stand for.
        internal void FinishSumOfSquaresDouble(IGpuBuffer accumulator)
        {
            var finish = _kernelCache["sum_squares_finish_double"];
            finish.SetArg(0, ((DirectOpenClGpuBuffer)accumulator).Buffer.Handle);
            finish.Execute1D(1, 1);
        }

        public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
        {
            MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
            if (scale is null) throw new System.ArgumentNullException(nameof(scale));
            var kernel = _kernelCache["clip_scale_from_sum_squares"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)sumOfSquares).Buffer.Handle);
            kernel.SetArg(1, maxNorm);
            kernel.SetArg(2, ((DirectOpenClGpuBuffer)scale).Buffer.Handle);
            kernel.Execute1D(1, 1);
        }

        public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
        {
            if (scale is null) throw new System.ArgumentNullException(nameof(scale));
            MultiTensorArgs.Validate(tensors, sizes);
            var kernel = _kernelCache["scale_by_device_scalar_inplace"];
            var s = ((DirectOpenClGpuBuffer)scale).Buffer.Handle;
            for (int t = 0; t < tensors.Count; t++)
            {
                kernel.SetArg(0, ((DirectOpenClGpuBuffer)tensors[t]).Buffer.Handle);
                kernel.SetArg(1, sizes[t]);
                kernel.SetArg(2, s);
                kernel.Execute1D(sizes[t], CalculateOptimalWorkGroupSize1D(sizes[t]));
            }
        }
    }
}