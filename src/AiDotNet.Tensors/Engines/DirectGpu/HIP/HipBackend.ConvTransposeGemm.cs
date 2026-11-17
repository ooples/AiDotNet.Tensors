// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on HIP (CUDA's
// CudaBackend.TryConvTranspose2DGemm, with the backend's GEMM), through PortableConvTransposeGemm and
// ConvTransposeGemmKernelTemplate's kernels, compiled on first need.
public sealed partial class HipBackend
{
    private IntPtr _convTransposeGemmModule;
    private bool? _convTransposeGemmReady;

    // The module, built once; a compile the device rejects leaves the conv_transpose2d kernel serving every shape, and
    // says why.
    private bool ConvTransposeGemmReady
    {
        get
        {
            if (_convTransposeGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_convTransposeGemmReady is null)
                {
                    var failure = CompileKernelModule(
                        ConvTransposeGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip), "conv_transpose_gemm",
                        ref _convTransposeGemmModule, ConvTransposeGemmKernelTemplate.GetKernelNames(), useFastMath: false);
                    if (failure is not null)
                        System.Diagnostics.Trace.TraceWarning(
                            $"HIP transposed-convolution GEMM kernels unavailable: {failure.Message}");
                    bool ok = failure is null;
                    foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    _convTransposeGemmReady = ok;
                }
                return _convTransposeGemmReady.Value;
            }
        }
    }

    private unsafe void LaunchConvTransposeGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)ConvTransposeGemmKernelTemplate.GroupSize, args);
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
