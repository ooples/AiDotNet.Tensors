// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorRfftMagnitude<T>(Tensor<T> input, double epsilon = 1e-9)
    {
        var (rows, n, bins) = ValidateRfftMagnitude(input, null, epsilon);
        if (GraphMode.IsActive) return base.TensorRfftMagnitude(input, epsilon);
        using var operation = EnterGpuOp();
        var backend = RequireRfftBackend<T>();
        if (rows != 0 && !HasDeviceRfft(backend, rows, n))
        {
            ReportHostRfft(n);
            return base.TensorRfftMagnitude(input, epsilon);
        }
        int[] shape = (int[])input._shape.Clone();
        shape[shape.Length - 1] = bins;
        var output = rows == 0 ? TensorAllocator.RentUninitialized<T>(shape) : CreateResidentOutput<T>(backend, shape);
        TensorRfftMagnitudeInto(output, input, epsilon);
        DifferentiableOps.RecordUnary("TensorRfftMagnitude", output, input,
            BackwardFunctions<T>.RfftMagnitudeBackward, new object[] { epsilon });
        return output;
    }

    /// <inheritdoc/>
    public override void TensorRfftMagnitudeInto<T>(Tensor<T> output, Tensor<T> input, double epsilon = 1e-9)
    {
        var (rows, n, _) = ValidateRfftMagnitude(input, output, epsilon);
        using var operation = EnterGpuOp();
        var backend = RequireRfftBackend<T>();
        if (rows != 0 && !HasDeviceRfft(backend, rows, n))
        {
            ReportHostRfft(n);
            base.TensorRfftMagnitudeInto(output, input, epsilon);
            return;
        }
        RequireRfftDestination(output);
        if (rows == 0) return;
        using var source = GetRfftInputBuffer(backend, input);
        // All input reads finish before the output is written, including actual
        // overlapping input/output views. The spectrum is action-local scratch.
        using var spectrum = RfftSpectrum(backend, source.Buffer, rows, n, out int stride);
        var destination = GetOrCreateResidentBuffer(backend, output, output.Length, fullyWritten: true);
        ((IRfftMagnitudeKernels)backend).RfftMagnitude(spectrum.Buffer, destination, rows, n, stride, (float)epsilon);
        ResidentSyncCheck("RfftMagnitude");
        output.IncrementVersion();
        BindResidentBuffer(output, destination, backend);
        MarkCallerOwnedDestination(output);
    }

    /// <inheritdoc/>
    public override void TensorRfftMagnitudeBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gradInput,
        double epsilon = 1e-9, bool accumulate = false)
    {
        var (rows, n, _) = ValidateRfftMagnitudeBackward(gradOutput, input, gradInput, epsilon);
        using var operation = EnterGpuOp();
        var backend = RequireRfftBackend<T>();
        if (rows != 0 && !HasDeviceRfft(backend, rows, n))
        {
            ReportHostRfft(n);
            base.TensorRfftMagnitudeBackwardInto(gradOutput, input, gradInput, epsilon, accumulate);
            return;
        }
        RequireRfftDestination(gradInput);
        if (rows == 0) return;
        using var source = GetRfftInputBuffer(backend, input);
        using var upstream = GetRfftInputBuffer(backend, gradOutput);
        using var spectrum = RfftSpectrum(backend, source.Buffer, rows, n, out int stride);
        using var seed = AllocateTransientScratch(backend, checked(rows * n * 2), fullyWritten: true);
        var kernels = (IRfftMagnitudeKernels)backend;
        kernels.RfftMagnitudeSeed(spectrum.Buffer, upstream.Buffer, seed.Buffer, rows, n, stride, (float)epsilon);
        // Real-input one-sided FFT adjoint: zero-pad the negative bins and
        // take Re(IFFT(seed)). cuFFT is unnormalized; portable inverse is 1/N.
        using var transformed = AllocateTransientScratch(backend, checked(rows * n * 2), fullyWritten: true);
        float scale;
        var plan = kernels.GetOptimizedRealFftPlan(n, rows, inverse: true);
        if (plan is not null)
        {
            using var workspace = plan.WorkspaceFloats == 0 ? default : AllocateTransientScratch(backend, plan.WorkspaceFloats, fullyWritten: true);
            plan.Execute(seed.Buffer, transformed.Buffer, plan.WorkspaceFloats == 0 ? null : workspace.Buffer);
            scale = 1;
        }
        else
        {
            PortableRfftTransform(backend, seed.Buffer, transformed.Buffer, rows, n, inverse: true);
            scale = n;
        }
        // Resolve a read-modify-write destination only after every primal and
        // upstream read, so in-place VJPs preserve alias semantics.
        using var old = accumulate ? GetResidentOrPersistentInputBuffer(backend, gradInput) : default;
        var destination = accumulate ? old.Buffer : GetOrCreateResidentBuffer(backend, gradInput, gradInput.Length, fullyWritten: true);
        kernels.RfftRealGradient(transformed.Buffer, destination, gradInput.Length, scale, accumulate);
        ResidentSyncCheck("RfftMagnitudeBackward");
        gradInput.IncrementVersion();
        BindResidentBuffer(gradInput, destination, backend);
        MarkCallerOwnedDestination(gradInput);
    }

    private IDirectGpuBackend RequireRfftBackend<T>()
    {
        if (typeof(T) != typeof(float))
            throw new NotSupportedException("DirectGpu real FFT magnitude currently requires Float32 storage. No host fallback was performed.");
        if (!TryGetBackend(out var backend) || backend is not IRfftMagnitudeKernels)
            throw new NotSupportedException("The current backend has no native real FFT magnitude route. No host fallback was performed.");
        return backend;
    }

    private static void RequireRfftDestination<T>(Tensor<T> destination)
    {
        if (!IsCanonicalDenseAllocation(destination))
            throw new NotSupportedException("DirectGpu real FFT magnitude destinations must be dense contiguous zero-offset allocations. No host fallback was performed.");
    }

    private OwnedBuffer GetRfftInputBuffer<T>(IDirectGpuBackend backend, Tensor<T> input)
    {
        if (!IsPartialView(input)) return GetResidentOrPersistentInputBuffer(backend, input);
        // The general resolver may eventually call Contiguous() on the host.
        // Native FFT never silently downloads an unsupported resident layout.
        if (!IsFp16Resident(input))
        {
            if (TryResolveDeviceParameterView(backend, input, out var parameter, out bool owned))
                return new OwnedBuffer(parameter, ownsBuffer: owned);
            if (!input.IsContiguous && TryPermuteViewOnDevice(backend, input, out var permutation)) return permutation;
            if (TrySliceViewOnDevice(backend, input, out var slice)) return slice;
            if (TrySliceResidentMemberOnDevice(backend, input, out var member)) return member;
        }
        throw new NotSupportedException("The real FFT input view has no native layout route on this backend. No host fallback was performed.");
    }

    private OwnedBuffer RfftSpectrum(IDirectGpuBackend backend, IGpuBuffer input, int rows, int n, out int stride)
    {
        var plan = ((IRfftMagnitudeKernels)backend).GetOptimizedRealFftPlan(n, rows, inverse: false);
        stride = plan is null ? n : n / 2 + 1;
        var spectrum = AllocateTransientScratch(backend, checked(rows * stride * 2), fullyWritten: true);
        try
        {
            if (plan is not null)
            {
                using var workspace = plan.WorkspaceFloats == 0 ? default : AllocateTransientScratch(backend, plan.WorkspaceFloats, fullyWritten: true);
                plan.Execute(input, spectrum.Buffer, plan.WorkspaceFloats == 0 ? null : workspace.Buffer);
            }
            else
            {
                int complexLength = checked(rows * n * 2);
                using var complex = AllocateTransientScratch(backend, complexLength, fullyWritten: true);
                // StridedScatter fills only the real lanes. Reused native storage can hold a prior spectrum.
                if (backend is IDeviceZeroFill zero) zero.ZeroBuffer(complex.Buffer, complexLength);
                else backend.Fill(complex.Buffer, 0f, complexLength);
                backend.StridedScatter(input, complex.Buffer, 0, 2, checked(rows * n));
                PortableRfftTransform(backend, complex.Buffer, spectrum.Buffer, rows, n, inverse: false);
            }
            return spectrum;
        }
        catch { spectrum.Dispose(); throw; }
    }

    // The device serves a length through the vendor plan (any length) or the portable radix-2 transform (powers of two).
    // Any other length runs on the host engine, which takes every length: a real FFT of 129 samples is a valid call.
    private static bool HasDeviceRfft(IDirectGpuBackend backend, int rows, int n)
        => (n & (n - 1)) == 0 || ((IRfftMagnitudeKernels)backend).GetOptimizedRealFftPlan(n, rows, inverse: false) is not null;

    private static void ReportHostRfft(int n)
        => GpuLaunchProbe.OnFallback($"TensorRfftMagnitude: length {n} is not a power of two and the backend has no vendor FFT plan", null);

    private static void PortableRfftTransform(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int rows, int n, bool inverse)
    {
        if ((n & (n - 1)) != 0)
            throw new NotSupportedException($"{backend.BackendName} requires a power-of-two real FFT length without a supported vendor plan. No host fallback was performed.");
        int count = checked(rows * n);
        using var re = AllocateTransientScratch(backend, count, fullyWritten: true);
        using var im = AllocateTransientScratch(backend, count, fullyWritten: true);
        using var outRe = AllocateTransientScratch(backend, count, fullyWritten: true);
        using var outIm = AllocateTransientScratch(backend, count, fullyWritten: true);
        backend.StridedGather(input, re.Buffer, 0, 2, count);
        backend.StridedGather(input, im.Buffer, 1, 2, count);
        backend.BatchedFFT(re.Buffer, im.Buffer, outRe.Buffer, outIm.Buffer, rows, n, inverse);
        backend.StridedScatter(outRe.Buffer, output, 0, 2, count);
        backend.StridedScatter(outIm.Buffer, output, 1, 2, count);
    }

    internal Tensor<T> CreateRfftMagnitudeGradient<T>(Tensor<T> input)
    {
        using var operation = EnterGpuOp();
        var backend = RequireRfftBackend<T>();
        return input.Length == 0 ? TensorAllocator.RentUninitialized<T>(input._shape) : CreateResidentOutput<T>(backend, input._shape);
    }
}
