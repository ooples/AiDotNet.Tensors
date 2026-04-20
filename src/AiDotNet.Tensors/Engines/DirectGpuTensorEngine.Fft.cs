// Copyright (c) AiDotNet. All rights reserved.
// DirectGpuTensorEngine entry point for the public Fft module. Prefers the
// new IFftBackend dispatch (Metal / Vulkan — supply-chain-clean, custom
// radix-2 kernels) and falls back to the long-standing split-real/imag
// engine.FFT path (CUDA / HIP / OpenCL) before giving up and letting the
// Fft module route through CPU.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// Attempt a GPU FFT on an interleaved real/imag float tensor. Returns
    /// <c>null</c> when the current backend doesn't ship a kernel we can
    /// dispatch; callers route to CPU in that case.
    /// </summary>
    /// <param name="interleaved">Input with last axis = <c>2·n</c> (re/im pairs).</param>
    /// <param name="inverse">Inverse transform when true.</param>
    internal Tensor<float>? TryBackendFft(Tensor<float> interleaved, bool inverse)
    {
        if (!TryGetBackend(out var backend)) return null;

        int rank = interleaved.Rank;
        int last = interleaved.Shape[rank - 1];
        if (last % 2 != 0) return null;
        int n = last / 2;
        if ((n & (n - 1)) != 0) return null; // pow2 only on every GPU path today
        int batch = interleaved.Length / last;

        // Path A: IFftBackend (Metal / Vulkan — interleaved layout, single buffer).
        if (backend is IFftBackend fftBackend)
        {
            using var buf = GetOrAllocateBuffer(backend, interleaved);
            // Kernel is in-place; allocate a separate output buffer so we can
            // preserve the input (some call paths expect the input tensor to
            // still contain its original data — GetOrAllocateBuffer returns
            // a buffer that may be cached).
            var outBuf = AllocateOutputBuffer(backend, interleaved.Length);
            try
            {
                // Copy input → output by letting the kernel do the initial copy
                // on Vulkan (2-binding shader), or by calling an explicit copy
                // for Metal. The safe-for-all-backends approach is to upload
                // the input directly into outBuf first, then dispatch in place.
                backend.CopyBuffer(buf.Buffer, outBuf.Buffer, interleaved.Length);
                fftBackend.LaunchFft(outBuf.Buffer, batch, n, inverse);
                var outArr = FinishGpuOp<float>(backend, outBuf, interleaved.Length);
                return new Tensor<float>(outArr, (int[])interleaved._shape.Clone());
            }
            catch
            {
                outBuf.Dispose();
                return null;
            }
        }

        // Path B: split real/imag engine.FFT (CUDA / HIP / OpenCL — existing
        // surface). Decompose the interleaved buffer CPU-side (cheap for the
        // sizes that actually matter), dispatch the native kernel, reassemble.
        var realShape = (int[])interleaved._shape.Clone();
        realShape[realShape.Length - 1] = n;
        var realIn = new Tensor<float>(realShape);
        var imagIn = new Tensor<float>(realShape);
        var inD = interleaved.GetDataArray();
        var reD = realIn.GetDataArray();
        var imD = imagIn.GetDataArray();
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < n; i++)
            {
                reD[b * n + i] = inD[b * last + 2 * i];
                imD[b * n + i] = inD[b * last + 2 * i + 1];
            }
        }
        Tensor<float> reOut, imOut;
        if (inverse)
            IFFT(realIn, imagIn, out reOut, out imOut);
        else
            FFT(realIn, imagIn, out reOut, out imOut);
        var output = new Tensor<float>((int[])interleaved._shape.Clone());
        var oD = output.GetDataArray();
        var outReD = reOut.GetDataArray();
        var outImD = imOut.GetDataArray();
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < n; i++)
            {
                oD[b * last + 2 * i] = outReD[b * n + i];
                oD[b * last + 2 * i + 1] = outImD[b * n + i];
            }
        }
        return output;
    }
}
