using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The gradients of a 1 x 1, stride-1, unpadded convolution are plain products over the whole batch, which the direct
    // backward kernels (one work item per output element, a serial dot product each) ran tens of times slower than GEMM:
    // they were 80% of a ResNet-50 training step on this backend, whose convolutions are mostly 1 x 1. With HW the
    // spatial size and B the batch, and dY' the output gradient laid out [Co, B * HW]:
    //   input gradient   dX' [Ci, B * HW] = W^T [Ci, Co] . dY',  then dX' laid back out as [B, Ci, HW];
    //   filter gradient  dW [Co, Ci]      = dY' . X'' [B * HW, Ci], X'' the input laid out [B, HW, Ci].
    // The layout passes are memory-bound and small next to the product.
    public sealed partial class OpenClBackend
    {
        /// <summary>A 1 x 1, stride-1, unpadded, undilated convolution, whose product fits an int index.</summary>
        internal static bool AdmitsPointwiseGemm(TunedShape s)
            => s[5] == 1 && s[6] == 1 && s[7] == 1 && s[8] == 1 && s[9] == 0 && s[10] == 0 && s[11] == 1 && s[12] == 1
               && s[13] == s[2] && s[14] == s[3]
               && (long)s[0] * s[2] * s[3] * System.Math.Max(s[1], s[4]) <= int.MaxValue;

        internal void Conv2DBackwardInputPointwiseGemm(in TunedConv2DBackwardArgs a)
        {
            int batch = a.Batch, ci = a.InChannels, co = a.OutChannels, hw = a.InHeight * a.InWidth, n = batch * hw;
            using var weightT = AllocateBuffer(ci * co);
            using var gradOutT = AllocateBuffer(co * n);
            using var gradInT = AllocateBuffer(ci * n);
            Transpose(a.Second, weightT, co, ci);                                   // W [Co, Ci] -> W^T [Ci, Co]
            Permute(a.First, gradOutT, new[] { batch, co, hw }, new[] { 1, 0, 2 }); // dY -> [Co, B, HW]
            Gemm(weightT, gradOutT, gradInT, ci, n, co);
            Permute(gradInT, a.Output, new[] { ci, batch, hw }, new[] { 1, 0, 2 }); // [Ci, B, HW] -> [B, Ci, HW]
        }

        internal void Conv2DBackwardFilterPointwiseGemm(in TunedConv2DBackwardArgs a)
        {
            int batch = a.Batch, ci = a.InChannels, co = a.OutChannels, hw = a.InHeight * a.InWidth, n = batch * hw;
            using var gradOutT = AllocateBuffer(co * n);
            using var inputT = AllocateBuffer(n * ci);
            Permute(a.Second, gradOutT, new[] { batch, co, hw }, new[] { 1, 0, 2 }); // dY -> [Co, B, HW]
            Permute(a.First, inputT, new[] { batch, ci, hw }, new[] { 0, 2, 1 });    // X -> [B, HW, Ci]
            Gemm(gradOutT, inputT, a.Output, co, ci, n);
        }

        private static readonly string[] s_conv2DBackwardDataUntunedPreference =
            { "opencl.conv2d_dx.pointwise_gemm", "opencl.conv2d_dx.s2k1", "opencl.conv2d_dx.s2k3", "opencl.conv2d_dx.winograd_f23" };

        private static readonly string[] s_conv2DBackwardFilterUntunedPreference =
            { "opencl.conv2d_dw.pointwise_gemm", "opencl.conv2d_dw.winograd_fused", "opencl.conv2d_dw.winograd_f32" };
    }
}
