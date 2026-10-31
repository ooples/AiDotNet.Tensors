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

        // Any other convolution's gradients go through the column matrix, a batch chunk at a time so it stays bounded:
        //   dX: G [p, Bc * L] = W^T [p, Co] . dY' [Co, Bc * L], laid out [Bc, p, L], then col2im (a gather, deterministic);
        //   dW += dY' [Co, Bc * L] . col^T [Bc * L, p], col = im2col(X).
        // p = Ci * kH * kW and L the output pixels. These replace the direct kernels (one work item per element and a
        // serial sum each), which were 58% of a ResNet-50 GPU step on its 3 x 3 and stride-2 convolutions.
        private const long GemmBackwardChunkElements = 16L * 1024 * 1024;

        /// <summary>A convolution the column-matrix backward handles: square indices that fit an int.</summary>
        internal static bool AdmitsColumnGemm(TunedShape s)
        {
            long p = (long)s[1] * s[5] * s[6], l = (long)s[13] * s[14];
            return s[0] > 0 && p > 0 && l > 0 && p * l <= GemmBackwardChunkElements
                   && (long)s[0] * System.Math.Max(s[1], s[4]) * System.Math.Max((long)s[2] * s[3], l) <= int.MaxValue;
        }

        private static int ColumnChunk(int batch, long perImage) => (int)System.Math.Max(1, System.Math.Min(batch, GemmBackwardChunkElements / perImage));

        internal void Conv2DBackwardInputColumnGemm(in TunedConv2DBackwardArgs a)
        {
            int ci = a.InChannels, co = a.OutChannels, l = a.OutHeight * a.OutWidth, p = ci * a.KernelH * a.KernelW;
            int inImage = ci * a.InHeight * a.InWidth, outImage = co * l;
            int chunk = ColumnChunk(a.Batch, (long)p * l);
            using var weightT = AllocateBuffer(p * co);
            Transpose(a.Second, weightT, co, p);                                   // W [Co, p] -> W^T [p, Co]
            using var gradOutT = AllocateBuffer(co * chunk * l);
            using var columns = AllocateBuffer(p * chunk * l);
            using var columnsByImage = AllocateBuffer(p * chunk * l);
            using var gradOutChunk = chunk < a.Batch ? AllocateBuffer(outImage * chunk) : null;
            using var gradInChunk = chunk < a.Batch ? AllocateBuffer(inImage * chunk) : null;
            for (int b0 = 0; b0 < a.Batch; b0 += chunk)
            {
                int bc = System.Math.Min(chunk, a.Batch - b0);
                var gradOut = a.First;
                if (gradOutChunk is not null) { Copy(a.First, b0 * outImage, gradOutChunk, 0, bc * outImage); gradOut = gradOutChunk; }
                Permute(gradOut, gradOutT, new[] { bc, co, l }, new[] { 1, 0, 2 });      // [Co, Bc, L]
                Gemm(weightT, gradOutT, columns, p, bc * l, co);                           // [p, Bc, L]
                Permute(columns, columnsByImage, new[] { p, bc, l }, new[] { 1, 0, 2 });  // [Bc, p, L]
                var gradIn = gradInChunk ?? a.Output;
                Col2Im(columnsByImage, gradIn, bc, a);
                if (gradInChunk is not null) Copy(gradInChunk, 0, a.Output, b0 * inImage, bc * inImage);
            }
        }

        internal void Conv2DBackwardFilterColumnGemm(in TunedConv2DBackwardArgs a)
        {
            int ci = a.InChannels, co = a.OutChannels, l = a.OutHeight * a.OutWidth, p = ci * a.KernelH * a.KernelW;
            int inImage = ci * a.InHeight * a.InWidth, outImage = co * l;
            int chunk = ColumnChunk(a.Batch, (long)p * l);
            using var columns = AllocateBuffer(p * chunk * l);
            using var columnsT = AllocateBuffer(p * chunk * l);
            using var gradOutT = AllocateBuffer(co * chunk * l);
            using var inputChunk = chunk < a.Batch ? AllocateBuffer(inImage * chunk) : null;
            using var gradOutChunk = chunk < a.Batch ? AllocateBuffer(outImage * chunk) : null;
            for (int b0 = 0; b0 < a.Batch; b0 += chunk)
            {
                int bc = System.Math.Min(chunk, a.Batch - b0);
                var input = a.First;
                var gradOut = a.Second;
                if (inputChunk is not null) { Copy(a.First, b0 * inImage, inputChunk, 0, bc * inImage); input = inputChunk; }
                if (gradOutChunk is not null) { Copy(a.Second, b0 * outImage, gradOutChunk, 0, bc * outImage); gradOut = gradOutChunk; }
                Im2Col(input, columns, bc, a);                                              // [Bc, p, L]
                Permute(columns, columnsT, new[] { bc, p, l }, new[] { 0, 2, 1 });         // [Bc, L, p]
                Permute(gradOut, gradOutT, new[] { bc, co, l }, new[] { 1, 0, 2 });        // [Co, Bc, L]
                Gemm(gradOutT, columnsT, a.Output, co, p, bc * l, 1f, b0 == 0 ? 0f : 1f);    // dW (+)= dY' . col^T
            }
        }

        private void Im2Col(IGpuBuffer input, IGpuBuffer columns, int batch, in TunedConv2DBackwardArgs a)
        {
            var k = _kernelCache["im2col"];
            ColumnKernelArgs(k, input, columns, batch, a);
            k.Execute1D(batch * a.OutHeight * a.OutWidth, System.Math.Min(256, (int)_maxWorkGroupSize));
        }

        private void Col2Im(IGpuBuffer columns, IGpuBuffer output, int batch, in TunedConv2DBackwardArgs a)
        {
            var k = _kernelCache["col2im"];
            ColumnKernelArgs(k, columns, output, batch, a);
            k.Execute1D(batch * a.InChannels * a.InHeight * a.InWidth, System.Math.Min(256, (int)_maxWorkGroupSize));
        }

        private static void ColumnKernelArgs(DirectOpenClKernel k, IGpuBuffer source, IGpuBuffer destination, int batch,
            in TunedConv2DBackwardArgs a)
        {
            uint arg = 0;
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)source).Buffer.Handle);
            k.SetArg(arg++, ((DirectOpenClGpuBuffer)destination).Buffer.Handle);
            k.SetArg(arg++, batch); k.SetArg(arg++, a.InChannels);
            k.SetArg(arg++, a.InHeight); k.SetArg(arg++, a.InWidth);
            k.SetArg(arg++, a.KernelH); k.SetArg(arg++, a.KernelW);
            k.SetArg(arg++, a.StrideH); k.SetArg(arg++, a.StrideW);
            k.SetArg(arg++, a.PadH); k.SetArg(arg++, a.PadW);
            k.SetArg(arg++, a.DilationH); k.SetArg(arg++, a.DilationW);
            k.SetArg(arg++, a.OutHeight); k.SetArg(arg, a.OutWidth);
        }
        private static readonly string[] s_conv2DBackwardDataUntunedPreference =
            { "opencl.conv2d_dx.pointwise_gemm", "opencl.conv2d_dx.s2k1", "opencl.conv2d_dx.s2k3", "opencl.conv2d_dx.winograd_f23", "opencl.conv2d_dx.column_gemm" };

        private static readonly string[] s_conv2DBackwardFilterUntunedPreference =
            { "opencl.conv2d_dw.pointwise_gemm", "opencl.conv2d_dw.winograd_fused", "opencl.conv2d_dw.winograd_f32", "opencl.conv2d_dw.column_gemm" };
    }
}
