using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The global-norm clip as chunk reductions per tensor (no device address tables on OpenCL), the OpenCL port of
    // CudaBackend's IMultiTensorKernels: the deterministic record (Gpu.DeterministicGradientNorm). FinishSumOfSquaresDouble
    // still serves the float-and-scaled-sum accumulator of IDeviceScalarOps and the device-guarded updates (#1502).
    public sealed partial class OpenClBackend : IMultiTensorKernels
    {
        // The finiteness reduction IDeviceScalarOps promises (a double in two slots): float and 2^-80-scaled sums.
        internal void FinitenessSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
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

        private IGpuBuffer? _normPartials;

        // The deterministic norm record (Gpu.DeterministicGradientNorm): one chunk-reduction launch per tensor writes
        // that tensor's partials, then one group combines them all.
        public void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
        {
            MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
            MultiTensorArgs.Validate(tensors, sizes);
            int total = 0;
            for (int t = 0; t < sizes.Count; t++) total += Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
            int words = System.Math.Max(1, total) * Gpu.DeterministicGradientNorm.PartialWords;
            if (_normPartials is not { } partials || partials.Size < words)
            {
                _normPartials?.Dispose();
                partials = _normPartials = AllocateBuffer(System.Math.Max(words, 64 * Gpu.DeterministicGradientNorm.PartialWords));
            }
            var p = ((DirectOpenClGpuBuffer)partials).Buffer.Handle;
            var chunk = _kernelCache["dgn_chunk_partials_tensor"];
            int first = 0;
            for (int t = 0; t < tensors.Count; t++)
            {
                int chunks = Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
                chunk.SetArg(0, ((DirectOpenClGpuBuffer)tensors[t]).Buffer.Handle);
                chunk.SetArg(1, sizes[t]);
                chunk.SetArg(2, (uint)t);
                chunk.SetArg(3, first);
                chunk.SetArg(4, p);
                chunk.Execute1D(chunks * DeterministicNormKernels.GroupSize, DeterministicNormKernels.GroupSize);
                first += chunks;
            }
            var finish = _kernelCache["dgn_finish"];
            finish.SetArg(0, p);
            finish.SetArg(1, total);
            finish.SetArg(2, Gpu.DeterministicGradientNorm.Headroom(total));
            finish.SetArg(3, ((DirectOpenClGpuBuffer)sumOfSquares).Buffer.Handle);
            finish.Execute1D(DeterministicNormKernels.GroupSize, DeterministicNormKernels.GroupSize);
        }

        // Turns the float sum and scaled sum in the buffer's two slots into the double they stand for.
        internal void FinishSumOfSquaresDouble(IGpuBuffer accumulator)
        {
            var finish = _kernelCache["sum_squares_finish_double"];
            finish.SetArg(0, ((DirectOpenClGpuBuffer)accumulator).Buffer.Handle);
            finish.Execute1D(1, 1);
        }

        public void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
            => ((IMultiTensorKernels)this).ClipScaleFromRecord(sumOfSquares, maxNorm, scale, Gpu.NonFiniteGradientPolicy.LeaveUnscaled);

        void IMultiTensorKernels.ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale,
            Gpu.NonFiniteGradientPolicy policy)
        {
            MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
            if (scale is null) throw new System.ArgumentNullException(nameof(scale));
            var kernel = _kernelCache["dgn_clip_scale"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)sumOfSquares).Buffer.Handle);
            kernel.SetArg(1, Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm));
            kernel.SetArg(2, (uint)policy);
            kernel.SetArg(3, ((DirectOpenClGpuBuffer)scale).Buffer.Handle);
            kernel.Execute1D(1, 1);
        }

        public void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
        {
            if (scale is null) throw new System.ArgumentNullException(nameof(scale));
            MultiTensorArgs.Validate(tensors, sizes);
            // dgn_mul: the IEEE product with subnormals kept, so a flush-to-zero device still matches the others.
            var kernel = _kernelCache["dgn_scale_tensor"];
            var s = ((DirectOpenClGpuBuffer)scale).Buffer.Handle;
            for (int t = 0; t < tensors.Count; t++)
            {
                kernel.SetArg(0, ((DirectOpenClGpuBuffer)tensors[t]).Buffer.Handle);
                kernel.SetArg(1, sizes[t]);
                kernel.SetArg(2, s);
                kernel.Execute1D(sizes[t], CalculateOptimalWorkGroupSize1D(sizes[t]));
            }
        }
    

        /// <inheritdoc />
        double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
        {
            MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
            return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
        }
}
}