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
        if (rank == 0) return null;
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
            using var outBuf = AllocateOutputBuffer(backend, interleaved.Length);

            // Pre-handoff phase: outBuf still owned by us. Narrow the catch
            // to the exceptions GPU dispatch is expected to throw when a
            // kernel isn't supported on this hardware (InvalidOperation for
            // library-not-compiled, NotSupported for out-of-contract shapes,
            // ArgumentOutOfRange for dimension guards). Other exceptions are
            // real bugs and should propagate.
            try
            {
                backend.CopyBuffer(buf.Buffer, outBuf.Buffer, interleaved.Length);
                fftBackend.LaunchFft(outBuf.Buffer, batch, n, inverse);
                var result = DeferTensorResult<float>(backend, outBuf.Buffer,
                    interleaved.Length, (int[])interleaved._shape.Clone());
                outBuf.RelinquishOwnership();
                return result;
            }
            catch (Exception ex) when (
                ex is InvalidOperationException
                   or NotSupportedException
                   or ArgumentException)
            {
                if (ThrowOnGpuKernelFallback) throw;
            }
        }

        // Path B: split real/imag FFT. Deinterleave and reassemble with strided
        // device copies so the public FFT module never materializes an intermediate.
        IGpuBuffer? realInput = null;
        IGpuBuffer? imaginaryInput = null;
        IGpuBuffer? realOutput = null;
        IGpuBuffer? imaginaryOutput = null;
        IGpuBuffer? interleavedOutput = null;
        bool outputHandedOff = false;
        try
        {
            using var input = GetOrAllocateBuffer(backend, interleaved);
            int complexCount = checked(batch * n);
            realInput = backend.AllocateBuffer(complexCount);
            imaginaryInput = backend.AllocateBuffer(complexCount);
            realOutput = backend.AllocateBuffer(complexCount);
            imaginaryOutput = backend.AllocateBuffer(complexCount);
            interleavedOutput = backend.AllocateBuffer(interleaved.Length);

            backend.StridedGather(input.Buffer, realInput, 0, 2, complexCount);
            backend.StridedGather(input.Buffer, imaginaryInput, 1, 2, complexCount);
            backend.BatchedFFT(realInput, imaginaryInput, realOutput, imaginaryOutput,
                batch, n, inverse);
            backend.StridedScatter(realOutput, interleavedOutput, 0, 2, complexCount);
            backend.StridedScatter(imaginaryOutput, interleavedOutput, 1, 2, complexCount);
            backend.Synchronize();

            var result = DeferTensorResult<float>(backend, interleavedOutput,
                interleaved.Length, (int[])interleaved._shape.Clone());
            outputHandedOff = true;
            return result;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            if (ThrowOnGpuKernelFallback) throw;
            System.Diagnostics.Trace.TraceWarning(
                $"GPU interleaved FFT fallback: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
        finally
        {
            realInput?.Dispose();
            imaginaryInput?.Dispose();
            realOutput?.Dispose();
            imaginaryOutput?.Dispose();
            if (!outputHandedOff) interleavedOutput?.Dispose();
        }
    }
}
