// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL instance, whose ConvTransposeGemmPortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The transposed-convolution forward as one GEMM per batch chunk and an overlap-add on Metal, through
// PortableConvTransposeGemm and ConvTransposeGemmKernelTemplate's kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IConvTranspose2DBiasBackend
{
    private const string ConvTransposeGemmLibraryName = "ConvTransposeGemm";
    private IntPtr _convTransposeGemmLibrary;
    private bool? _convTransposeGemmReady;

    // The library, compiled once, and both kernels able to run 256 threads in one threadgroup; otherwise the
    // conv_transpose2d kernel serves every shape, and the trace says why.
    private bool ConvTransposeGemmReady
    {
        get
        {
            if (_convTransposeGemmReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_convTransposeGemmReady is null)
                {
                    bool ok;
                    try
                    {
                        _convTransposeGemmLibrary = _shaderLibrary.CompileLibrary(ConvTransposeGemmLibraryName,
                            ConvTransposeGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _convTransposeGemmLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in ConvTransposeGemmKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(ConvTransposeGemmLibraryName, _convTransposeGemmLibrary, name)
                                    .MaxTotalThreadsPerThreadgroup >= ConvTransposeGemmKernelTemplate.GroupSize;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal transposed-convolution GEMM kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _convTransposeGemmReady = ok;
                }
                return _convTransposeGemmReady.Value;
            }
        }
    }

    private void LaunchConvTransposeGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var pipeline = GetPipeline(ConvTransposeGemmLibraryName, _convTransposeGemmLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)ConvTransposeGemmKernelTemplate.GroupSize, 1, 1));
    }

    /// <summary>The transposed forward through one GEMM per batch chunk. False: nothing launched.</summary>
    internal bool TryConvTranspose2DGemm(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, int batch,
        int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH,
        int kernelW, int strideH, int strideW, int padH, int padW, IGpuBuffer? bias = null)
        => PortableConvTransposeGemm.Enabled && ConvTransposeGemmReady
           && PortableConvTransposeGemm.TryForward(this, LaunchConvTransposeGemm, input, kernel, output, batch,
               inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW,
               padH, padW, bias);

    /// <summary>IConvTranspose2DBiasBackend: the bias added in the single-GEMM route's overlap-add; where that route does
    /// not serve the shape, the conv_transpose2d kernel and the bias kernel.</summary>
    void IConvTranspose2DBiasBackend.ConvTranspose2DBias(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer bias,
        IGpuBuffer output, int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight,
        int outWidth, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH,
        int outputPadW)
    {
        if (TryConvTranspose2DGemm(input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight,
                outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias))
            return;
        ConvTranspose2D(input, kernel, output, batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, outputPadH, outputPadW);
        Conv2DBiasAdd(output, bias, batch, outChannels, outHeight * outWidth);
    }
}
