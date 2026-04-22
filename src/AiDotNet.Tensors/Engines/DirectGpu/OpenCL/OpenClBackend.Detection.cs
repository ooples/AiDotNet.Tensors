// Copyright (c) AiDotNet. All rights reserved.
// OpenCL launcher shims for the vision-detection kernels (Issue #217).
// Mirrors OpenClBackend.Parity210.cs's pattern: each method pulls the
// compiled DirectOpenClKernel from _kernelCache and dispatches via
// kernel.Execute1D with 256-thread workgroups.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IDetectionBackend
    {
        private const int DetectionLocalSize = 256;

        private DirectOpenClKernel GetDetectionKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException(
                    $"OpenCL detection kernel not found: {name}. Module may have failed to compile.");
            return kernel;
        }

        private static int RoundUpToDetectionGroup(int v) =>
            ((v + DetectionLocalSize - 1) / DetectionLocalSize) * DetectionLocalSize;

        private static IntPtr DetectionBufHandle(IGpuBuffer b) =>
            ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        // --------------------------------------------------------------
        // Pairwise IoU family — one thread per (i, j).
        // --------------------------------------------------------------

        public void BoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
            => DispatchPairwiseIou("detection_box_iou", boxesA, boxesB, output, n, m);

        public void GeneralizedBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
            => DispatchPairwiseIou("detection_generalized_box_iou", boxesA, boxesB, output, n, m);

        public void DistanceBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
            => DispatchPairwiseIou("detection_distance_box_iou", boxesA, boxesB, output, n, m);

        public void CompleteBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
            => DispatchPairwiseIou("detection_complete_box_iou", boxesA, boxesB, output, n, m);

        private void DispatchPairwiseIou(string kernelName,
            IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        {
            int total = n * m;
            if (total <= 0) return;
            var k = GetDetectionKernel(kernelName);
            k.SetArg(0, DetectionBufHandle(boxesA));
            k.SetArg(1, DetectionBufHandle(boxesB));
            k.SetArg(2, DetectionBufHandle(output));
            k.SetArg(3, n);
            k.SetArg(4, m);
            k.Execute1D(RoundUpToDetectionGroup(total), DetectionLocalSize);
        }

        // --------------------------------------------------------------
        // BoxArea — one thread per box.
        // --------------------------------------------------------------

        public void BoxArea(IGpuBuffer boxes, IGpuBuffer output, int n)
        {
            if (n <= 0) return;
            var k = GetDetectionKernel("detection_box_area");
            k.SetArg(0, DetectionBufHandle(boxes));
            k.SetArg(1, DetectionBufHandle(output));
            k.SetArg(2, n);
            k.Execute1D(RoundUpToDetectionGroup(n), DetectionLocalSize);
        }

        // --------------------------------------------------------------
        // BoxConvert — one thread per box. Format codes match the
        // BoxFormat enum: 0=XYXY, 1=XYWH, 2=CXCYWH.
        // --------------------------------------------------------------

        public void BoxConvert(IGpuBuffer boxes, IGpuBuffer output, int n, int fromFormat, int toFormat)
        {
            if (n <= 0) return;
            if ((uint)fromFormat > 2 || (uint)toFormat > 2)
                throw new ArgumentException(
                    $"fromFormat/toFormat must be 0/1/2; got {fromFormat}, {toFormat}.");
            var k = GetDetectionKernel("detection_box_convert");
            k.SetArg(0, DetectionBufHandle(boxes));
            k.SetArg(1, DetectionBufHandle(output));
            k.SetArg(2, n);
            k.SetArg(3, fromFormat);
            k.SetArg(4, toFormat);
            k.Execute1D(RoundUpToDetectionGroup(n), DetectionLocalSize);
        }

        // --------------------------------------------------------------
        // IoU family backward — Issue #217. Two kernels: A-side owns rows
        // of N (each thread iterates M columns), B-side owns rows of M.
        // Atomics-free; portable.
        // --------------------------------------------------------------

        public void IouFamilyBackward(
            IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
            IGpuBuffer gradA, IGpuBuffer gradB,
            int n, int m, int variant)
        {
            // See CudaBackend.Detection for rationale — don't leak pooled
            // buffer contents when one of n/m is zero.
            if (n <= 0 && m <= 0) return;

            if (n > 0)
            {
                var kA = GetDetectionKernel("detection_iou_backward_a");
                kA.SetArg(0, DetectionBufHandle(gradOutput));
                kA.SetArg(1, DetectionBufHandle(boxesA));
                kA.SetArg(2, DetectionBufHandle(boxesB));
                kA.SetArg(3, DetectionBufHandle(gradA));
                kA.SetArg(4, n);
                kA.SetArg(5, m);
                kA.SetArg(6, variant);
                kA.Execute1D(RoundUpToDetectionGroup(n), DetectionLocalSize);
            }

            if (m > 0)
            {
                var kB = GetDetectionKernel("detection_iou_backward_b");
                kB.SetArg(0, DetectionBufHandle(gradOutput));
                kB.SetArg(1, DetectionBufHandle(boxesA));
                kB.SetArg(2, DetectionBufHandle(boxesB));
                kB.SetArg(3, DetectionBufHandle(gradB));
                kB.SetArg(4, n);
                kB.SetArg(5, m);
                kB.SetArg(6, variant);
                kB.Execute1D(RoundUpToDetectionGroup(m), DetectionLocalSize);
            }
        }
    }
}
#endif
