#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on WebGPU, through
// PortableConvTransposeGemm and the WGSL restatement of ConvTransposeGemmKernelTemplate (WebGpuConvTransposeGemmKernels),
// built on first need.
public sealed partial class WebGpuBackend
{
    private const string ConvTransposeGemmModulePrefix = "ConvTransposeGemm_";
    private bool? _convTransposeGemmReady;

    // Both pipelines, built once; a module the implementation rejects leaves the conv_transpose2d shader serving every
    // shape, and says why.
    private bool ConvTransposeGemmReady
    {
        get
        {
            if (_convTransposeGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_convTransposeGemmReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(ConvTransposeGemmModulePrefix + name,
                                    WebGpuConvTransposeGemmKernels.SourceFor(name), name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning(
                                $"WebGPU transposed-convolution GEMM kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _convTransposeGemmReady = ok;
                }
                return _convTransposeGemmReady.Value;
            }
        }
    }

    private void LaunchConvTransposeGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds the ints (bit-cast) and its padding words.
        var uniforms = new float[WebGpuConvTransposeGemmKernels.UniformWords(kernelName)];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(ConvTransposeGemmModulePrefix + kernelName, WebGpuConvTransposeGemmKernels.SourceFor(kernelName),
            kernelName, buffers, uniforms, groups * ConvTransposeGemmKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif
