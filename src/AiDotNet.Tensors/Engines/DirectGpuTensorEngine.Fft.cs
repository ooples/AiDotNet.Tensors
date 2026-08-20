// Copyright (c) AiDotNet. All rights reserved.
// DirectGpuTensorEngine entry point for the public Fft module. CUDA exposes
// generic-precision radix-2 and Bluestein transforms through IEngine. Legacy
// power-of-two backends remain available to the public Fft module, but an
// explicit IEngine.FftGeneric request never performs a hidden host fallback.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    private const string GenericFftBackendRoadmap =
        "HIP #961, OpenCL #962, Metal #963, Vulkan #964, and WebGPU #965";

    /// <inheritdoc/>
    public override bool SupportsFftElementType(FftElementType type)
    {
        _ = type.ByteSize();
        return TryGetBackend(out var backend)
            && backend is CudaBackend cuda
            && cuda.SupportsFftElementType(type);
    }

    /// <inheritdoc/>
    public override Tensor<float> FftGeneric(
        Tensor<float> input,
        bool inverse = false,
        FftElementType elementType = FftElementType.Float32)
    {
        ValidateGenericFftInput(input);
        _ = elementType.ByteSize();

        if (!TryGetBackend(out var backend) || backend is not CudaBackend cuda)
        {
            string backendName = backend?.BackendName ?? "unavailable DirectGpu backend";
            throw new NotSupportedException(
                $"{backendName} does not implement generic-precision arbitrary-length FFT. " +
                $"Tracked backend work: {GenericFftBackendRoadmap}. No host fallback was performed.");
        }

        if (!cuda.SupportsFftElementType(elementType))
        {
            throw new NotSupportedException(
                $"CUDA cannot execute {elementType} FFT storage on this device/toolkit. " +
                "The capability result is authoritative; no host fallback was performed.");
        }

        Tensor<float> result = ExecuteCudaFftGeneric(cuda, input, inverse, elementType);
        int n = input._shape[^1] / 2;
        if (inverse)
            FftAutograd.RecordIFft1(result, input, n, FftNorm.Backward);
        else
            FftAutograd.RecordFft1(result, input, n, FftNorm.Backward);
        return result;
    }

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
        if (last < 2 || last % 2 != 0) return null;
        int n = last / 2;
        int batch = interleaved.Length / last;

        // Preferred CUDA path: arbitrary length, batched, and fully device-resident.
        // The public Fft module records the backward operation after this method returns.
        if (backend is CudaBackend cuda && cuda.SupportsFftElementType(FftElementType.Float32))
        {
            try
            {
                return ExecuteCudaFftGeneric(cuda, interleaved, inverse, FftElementType.Float32);
            }
            catch (Exception ex) when (
                ex is not CudaGenericFftRecoveryException &&
                (ex is InvalidOperationException or NotSupportedException or ArgumentException))
            {
                if (ThrowOnGpuKernelFallback) throw;
                System.Diagnostics.Trace.TraceWarning(
                    $"CUDA generic FFT fallback: {ex.GetType().Name}: {ex.Message}");
            }
        }

        if ((n & (n - 1)) != 0) return null; // legacy backends remain radix-2-only

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

    private static void ValidateGenericFftInput(Tensor<float> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0)
            throw new ArgumentException("Generic FFT input must have at least one dimension.", nameof(input));
        int last = input._shape[^1];
        if (last < 2 || (last & 1) != 0)
        {
            throw new ArgumentException(
                "The final input dimension must contain one or more interleaved real/imaginary pairs.",
                nameof(input));
        }
    }

    private Tensor<float> ExecuteCudaFftGeneric(
        CudaBackend cuda,
        Tensor<float> interleaved,
        bool inverse,
        FftElementType elementType)
    {
        int last = interleaved._shape[^1];
        int n = last / 2;
        int batch = interleaved.Length / last;
        int complexCount = checked(batch * n);

        IGpuBuffer? realFloat = null;
        IGpuBuffer? imaginaryFloat = null;
        IGpuBuffer? realStorage = null;
        IGpuBuffer? imaginaryStorage = null;
        IGpuBuffer? interleavedOutput = null;
        bool outputHandedOff = false;
        bool cudaWorkMayBePending = false;
        Exception? dispatchFailure = null;
        try
        {
            using var input = GetOrAllocateBuffer(cuda, interleaved);
            realFloat = cuda.AllocateBuffer(complexCount);
            imaginaryFloat = cuda.AllocateBuffer(complexCount);
            cudaWorkMayBePending = true;
            cuda.StridedGather(input.Buffer, realFloat, 0, 2, complexCount);
            cuda.StridedGather(input.Buffer, imaginaryFloat, 1, 2, complexCount);

            if (elementType == FftElementType.Float32)
            {
                realStorage = realFloat;
                imaginaryStorage = imaginaryFloat;
            }
            else
            {
                int storageBytes = checked(complexCount * elementType.ByteSize());
                realStorage = cuda.AllocateByteBuffer(storageBytes);
                imaginaryStorage = cuda.AllocateByteBuffer(storageBytes);
                cuda.ConvertFloatToFftStorage(realFloat, realStorage, complexCount, elementType);
                cuda.ConvertFloatToFftStorage(imaginaryFloat, imaginaryStorage, complexCount, elementType);
            }

            cuda.LaunchFftGeneric(realStorage, imaginaryStorage, batch, n, inverse, elementType);

            if (elementType != FftElementType.Float32)
            {
                cuda.ConvertFftStorageToFloat(realStorage, realFloat, complexCount, elementType);
                cuda.ConvertFftStorageToFloat(imaginaryStorage, imaginaryFloat, complexCount, elementType);
            }

            interleavedOutput = cuda.AllocateBuffer(interleaved.Length);
            cuda.StridedScatter(realFloat, interleavedOutput, 0, 2, complexCount);
            cuda.StridedScatter(imaginaryFloat, interleavedOutput, 1, 2, complexCount);
            cuda.Synchronize();

            var result = DeferTensorResult<float>(cuda, interleavedOutput,
                interleaved.Length, (int[])interleaved._shape.Clone());
            outputHandedOff = true;
            return result;
        }
        catch (Exception ex)
        {
            dispatchFailure = ex;
            throw;
        }
        finally
        {
            // A generic FFT error may occur after one or more asynchronous launches. Do not
            // release or reuse any device buffer, or enter a legacy CUDA fallback, until the
            // stream proves those launches completed. If synchronization itself reports a
            // device fault, surface a non-fallback exception and leave cleanup to backend
            // teardown rather than risk a use-after-free on a poisoned context.
            if (!outputHandedOff && cudaWorkMayBePending && dispatchFailure is not null)
            {
                try
                {
                    cuda.Synchronize();
                }
                catch (Exception synchronizationFailure)
                {
                    throw new CudaGenericFftRecoveryException(dispatchFailure, synchronizationFailure);
                }
            }

            if (!ReferenceEquals(realStorage, realFloat)) realStorage?.Dispose();
            if (!ReferenceEquals(imaginaryStorage, imaginaryFloat)) imaginaryStorage?.Dispose();
            realFloat?.Dispose();
            imaginaryFloat?.Dispose();
            if (!outputHandedOff) interleavedOutput?.Dispose();
        }
    }

    private sealed class CudaGenericFftRecoveryException : InvalidOperationException
    {
        internal CudaGenericFftRecoveryException(Exception dispatchFailure, Exception synchronizationFailure)
            : base(
                "CUDA generic FFT failed and the stream could not be synchronized safely; " +
                "legacy GPU fallback was blocked to preserve the original device fault.",
                new AggregateException(dispatchFailure, synchronizationFailure))
        {
        }
    }
}
