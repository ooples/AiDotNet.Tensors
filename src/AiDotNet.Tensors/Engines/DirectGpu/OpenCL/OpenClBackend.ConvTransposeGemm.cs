// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on OpenCL, through
    // PortableConvTransposeGemm and ConvTransposeGemmKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _convTransposeGemmReady;

        private bool ConvTransposeGemmReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _convTransposeGemmReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_convTransposeGemmReady == 0)
                        System.Threading.Volatile.Write(ref _convTransposeGemmReady, CompileConvTransposeGemmModule() ? 1 : 2);
                    return _convTransposeGemmReady == 1;
                }
            }
        }

        // The module, built once, and both kernels able to run 256 work-items in one group; otherwise the
        // conv_transpose2d kernel serves every shape, and the trace says why.
        private bool CompileConvTransposeGemmModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(ConvTransposeGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "transposed-convolution GEMM kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames())
                        built.Add((name, new DirectOpenClKernel(context, program, name)));
                }
                catch
                {
                    foreach (var (_, kernel) in built) kernel.Dispose();
                    program.Dispose();
                    throw;
                }
                _programs.Add(program);
                foreach (var (name, kernel) in built) _kernelCache[name] = kernel;
                foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < ConvTransposeGemmKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"OpenCL transposed-convolution GEMM kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL transposed-convolution GEMM kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchConvTransposeGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * ConvTransposeGemmKernelTemplate.GroupSize, ConvTransposeGemmKernelTemplate.GroupSize);
        }

        /// <summary>The transposed forward through one GEMM per batch chunk. False: nothing launched.</summary>
        internal bool TryConvTranspose2DGemm(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, int batch,
            int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
            int kernelW, int strideH, int strideW, int padH, int padW)
            => PortableConvTransposeGemm.Enabled && ConvTransposeGemmReady
               && PortableConvTransposeGemm.TryForward(this, LaunchConvTransposeGemm, input, kernel, output, batch,
                   inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW,
                   padH, padW);
    }
}
