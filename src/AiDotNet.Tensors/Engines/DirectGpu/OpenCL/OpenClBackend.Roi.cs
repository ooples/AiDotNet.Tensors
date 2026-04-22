// Copyright (c) AiDotNet. All rights reserved.
// OpenCL RoI launcher shims (Issue #217 tail).
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IRoiBackend
    {
        private const int RoiLocalSize = 256;
        private DirectOpenClKernel GetRoiKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var k))
                throw new InvalidOperationException($"OpenCL RoI kernel not found: {name}.");
            return k;
        }
        private static int RoundUpRoi(int v) => ((v + RoiLocalSize - 1) / RoiLocalSize) * RoiLocalSize;
        private static IntPtr RoiHandle(IGpuBuffer b) => ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        public void RoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW,
            float spatialScale, int samplingRatio, bool aligned)
        {
            int total = K * C * outH * outW;
            if (total <= 0) return;
            var k = GetRoiKernel("roi_align");
            k.SetArg(0, RoiHandle(input));
            k.SetArg(1, RoiHandle(boxes));
            k.SetArg(2, RoiHandle(output));
            k.SetArg(3, N); k.SetArg(4, C); k.SetArg(5, H); k.SetArg(6, W);
            k.SetArg(7, K); k.SetArg(8, outH); k.SetArg(9, outW);
            k.SetArg(10, spatialScale); k.SetArg(11, samplingRatio);
            k.SetArg(12, aligned ? 1 : 0);
            k.Execute1D(RoundUpRoi(total), RoiLocalSize);
        }

        public void RoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW, float spatialScale)
        {
            int total = K * C * outH * outW;
            if (total <= 0) return;
            var k = GetRoiKernel("roi_pool");
            k.SetArg(0, RoiHandle(input));
            k.SetArg(1, RoiHandle(boxes));
            k.SetArg(2, RoiHandle(output));
            k.SetArg(3, N); k.SetArg(4, C); k.SetArg(5, H); k.SetArg(6, W);
            k.SetArg(7, K); k.SetArg(8, outH); k.SetArg(9, outW);
            k.SetArg(10, spatialScale);
            k.Execute1D(RoundUpRoi(total), RoiLocalSize);
        }

        public void PsRoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
            float spatialScale, int samplingRatio)
        {
            int total = K * outputChannels * outH * outW;
            if (total <= 0) return;
            var k = GetRoiKernel("ps_roi_align");
            k.SetArg(0, RoiHandle(input));
            k.SetArg(1, RoiHandle(boxes));
            k.SetArg(2, RoiHandle(output));
            k.SetArg(3, N); k.SetArg(4, C); k.SetArg(5, H); k.SetArg(6, W);
            k.SetArg(7, K); k.SetArg(8, outH); k.SetArg(9, outW); k.SetArg(10, outputChannels);
            k.SetArg(11, spatialScale); k.SetArg(12, samplingRatio);
            k.Execute1D(RoundUpRoi(total), RoiLocalSize);
        }

        public void PsRoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
            float spatialScale)
        {
            int total = K * outputChannels * outH * outW;
            if (total <= 0) return;
            var k = GetRoiKernel("ps_roi_pool");
            k.SetArg(0, RoiHandle(input));
            k.SetArg(1, RoiHandle(boxes));
            k.SetArg(2, RoiHandle(output));
            k.SetArg(3, N); k.SetArg(4, C); k.SetArg(5, H); k.SetArg(6, W);
            k.SetArg(7, K); k.SetArg(8, outH); k.SetArg(9, outW); k.SetArg(10, outputChannels);
            k.SetArg(11, spatialScale);
            k.Execute1D(RoundUpRoi(total), RoiLocalSize);
        }
    }
}
#endif
