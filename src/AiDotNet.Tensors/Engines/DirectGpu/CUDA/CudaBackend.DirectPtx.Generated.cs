// Copyright (c) AiDotNet. All rights reserved.
// Dispatch to GENERATED kernels, for the families whose measurements earned it.
//
// PROMO-1 recorded which families beat cuDNN and which did not, but only the 1x1
// convolution had a call site, so depthwise's 2.08x-2.99x and max-pool's 1.41x existed
// only inside the benchmark harness. This is the missing half: the engine can now reach
// them.
//
// Unlike the baked 1x1 kernel, which carries hand-written PTX for one exact shape, these
// are EMITTED from a CodegenKernelSpec at first use and cached. That is the point of the
// index-map layer -- a shape it can express does not need a hand-written kernel -- and it
// is why the dispatch covers a family rather than a single extent.
//
// Every path fails closed. Feature flag off, wrong architecture, family not promoted,
// shape outside what the spec expresses, emission refused: all return false and the caller
// runs the established kernel.

#nullable enable

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly Dictionary<string, GeneratedKernel> _generatedKernels = new(StringComparer.Ordinal);
    private long _generatedDispatchCount;

    /// <summary>How many times a generated kernel has been dispatched to.</summary>
    internal long GeneratedDispatchCount =>
        System.Threading.Interlocked.Read(ref _generatedDispatchCount);

    private sealed class GeneratedKernel
    {
        internal DirectPtxModule Module = null!;
        internal IntPtr Function;
        internal uint Blocks;
        internal uint BlockX;
        internal uint BlockY;
    }

    /// <summary>
    /// Attempts the generated depthwise 3x3 convolution.
    /// </summary>
    /// <remarks>
    /// Measured at 2.08x (plain) and 2.99x (with bias and ReLU) against cuDNN, and at
    /// L1 93% — a roofline, so no further code-generator change improves it. The backend's
    /// depthwise entry point has no bias or activation, so the spec built here has neither.
    /// </remarks>
    internal unsafe bool TryDirectPtxDepthwiseConv2D(
        IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (!_directPtxConvolutionOptedIn || !IsAvailable) return false;
        if (!DirectPtxArchitecture.HasExperimentalConvolution(_ccMajor, _ccMinor)) return false;
        if (!DirectPtxConvolutionPromotion.IsPromoted(
                DirectPtxConvolutionFamily.Depthwise3x3, out _)) return false;

        // Only the geometry the measurement covered. A 5x5 or a strided depthwise is a
        // different kernel with no evidence behind it, so it takes the established path.
        if (kernelH != 3 || kernelW != 3) return false;
        if (strideH != 1 || strideW != 1 || padH != 1 || padW != 1) return false;
        if (outHeight != inHeight || outWidth != inWidth) return false;
        if (batch <= 0 || channels <= 0 || inHeight <= 0 || inWidth <= 0) return false;

        if (input is null || kernel is null || output is null) return false;
        if (input.Handle == IntPtr.Zero || kernel.Handle == IntPtr.Zero ||
            output.Handle == IntPtr.Zero) return false;

        long inputElements = (long)batch * channels * inHeight * inWidth;
        if (input.SizeInBytes != inputElements * sizeof(float)) return false;
        if (output.SizeInBytes != inputElements * sizeof(float)) return false;
        if (kernel.SizeInBytes != (long)channels * 9 * sizeof(float)) return false;

        string key = "dw3x3:" + batch + "x" + channels + "x" + inHeight + "x" + inWidth;
        GeneratedKernel? entry;
        lock (_directPtxLock)
        {
            if (!_generatedKernels.TryGetValue(key, out entry))
            {
                entry = BuildDepthwise(batch, channels, inHeight, inWidth);
                if (entry is null) return false;
                _generatedKernels[key] = entry;
            }
        }

        IntPtr inputPtr = input.Handle, kernelPtr = kernel.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr;
        args[1] = &kernelPtr;
        args[2] = &outputPtr;

        using var scope = PushContext();
        entry.Module.Launch(entry.Function, entry.Blocks, 1, 1, entry.BlockX, entry.BlockY, 1, 0, args);
        System.Threading.Interlocked.Increment(ref _generatedDispatchCount);
        return true;
    }

    /// <summary>Emits and loads the depthwise kernel, or null when the emitter refuses.</summary>
    private GeneratedKernel? BuildDepthwise(int batch, int channels, int height, int width)
    {
        try
        {
            var spec = DepthwiseSpec(batch, channels, height, width);
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, _ccMajor, _ccMinor);

            _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
            var module = _directPtxRuntime.LoadModule(ptx, allowExperimentalJitFallback: true);

            return new GeneratedKernel
            {
                Module = module,
                Function = module.GetFunction(spec.Name, out _),
                Blocks = emitter.LaunchBlocks,
                BlockX = (uint)emitter.LaunchBlockX,
                BlockY = (uint)emitter.LaunchBlockY,
            };
        }
        catch (NotSupportedException)
        {
            // The emitter refuses specs it cannot lower correctly. That is a decline, not
            // a failure: the caller runs the established kernel.
            return null;
        }
    }

    /// <summary>
    /// The depthwise 3x3 operator, as an index-map spec.
    /// </summary>
    /// <remarks>
    /// Identical in shape to the catalog entry that verifies at 0.000E+000 against the
    /// fp64 oracle, so what dispatches here is the kernel the evidence describes.
    /// </remarks>
    private static CodegenKernelSpec DepthwiseSpec(int batch, int channels, int height, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("n", batch), CodegenAxis.Parallel("c", channels),
            CodegenAxis.Parallel("oh", height), CodegenAxis.Parallel("ow", width),
            CodegenAxis.Reduce("kh", 3), CodegenAxis.Reduce("kw", 3));
        const int N = 0, C = 1, OH = 2, OW = 3, KH = 4, KW = 5;

        var input = new CodegenTensorBinding(0, "input", new[] { batch, channels, height, width },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Window(OH, KH, 1, 1), CodegenAffineExpr.Window(OW, KW, 1, 1)
            });
        var weights = new CodegenTensorBinding(1, "weights", new[] { channels, 3, 3 },
            new[] { CodegenAffineExpr.Axis(C), CodegenAffineExpr.Axis(KH), CodegenAffineExpr.Axis(KW) });
        var output = new CodegenTensorBinding(2, "output", new[] { batch, channels, height, width },
            new[]
            {
                CodegenAffineExpr.Axis(N), CodegenAffineExpr.Axis(C),
                CodegenAffineExpr.Axis(OH), CodegenAffineExpr.Axis(OW)
            }, isOutput: true);

        return new CodegenKernelSpec("generated_dwconv2d_3x3", space,
            new[] { input, weights }, output, new[] { 0, 1 }, CodegenReduceKind.Sum);
    }
}
