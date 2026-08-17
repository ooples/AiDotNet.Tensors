// Copyright (c) AiDotNet. All rights reserved.
// Wires CudaFFTGenericKernels into the CUDA backend, so the generic-precision, arbitrary-length FFT path is
// reachable from the engine rather than existing only as compiled kernel source.
//
// WHY A PRIVATE FUNCTION CACHE INSTEAD OF _kernelCache.
// Every element-type module emits the SAME type-invariant entry points (fftg_build_twiddles, fftg_build_chirp,
// the float32 scratch kernels) alongside its suffixed ones. _kernelCache is keyed by bare kernel name, so
// compiling the fp32 module and then the bf16 module would have the second overwrite the first's shared
// entries and leave callers holding a function handle from a different module than the one they think they are
// launching. Keying on (element type, name) here keeps each module's handles with that module, which also means
// a backend that cannot compile bf16 loses nothing else.
//
// WHY LAZY COMPILATION.
// Two reasons, and the second is a correctness one. NVRTC compilation of three modules would be paid by every
// backend construction whether or not anyone asks for an FFT; and bfloat16 cannot be compiled at all below
// compute capability 8.0, so compiling eagerly would either throw during construction on older hardware or
// force the constructor to swallow a failure it cannot describe. Compiling on first use lets
// SupportsFftElementType answer honestly up front and keeps the failure attributable.

using System;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IFftGenericBackend
{
    private readonly ConcurrentDictionary<FftElementType, IntPtr> _fftGenericModules =
        new ConcurrentDictionary<FftElementType, IntPtr>();

    private readonly ConcurrentDictionary<(FftElementType Type, string Name), IntPtr> _fftGenericKernels =
        new ConcurrentDictionary<(FftElementType, string), IntPtr>();

    private readonly object _fftGenericCompileLock = new object();

    /// <inheritdoc />
    public bool SupportsFftElementType(FftElementType type)
    {
        if (!IsAvailable) return false;
        if (type != FftElementType.Float32 && type != FftElementType.Float16 && type != FftElementType.BFloat16)
            return false;
        // _ccMajor/_ccMinor are populated during construction from cuDeviceGetAttribute.
        return (_ccMajor * 10 + _ccMinor) >= type.MinComputeCapabilityX10();
    }

    /// <inheritdoc />
    public unsafe void LaunchFftGeneric(
        IGpuBuffer real, IGpuBuffer imag, int batchCount, int n, bool inverse, FftElementType type)
    {
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imag is null) throw new ArgumentNullException(nameof(imag));
        if (batchCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchCount), batchCount, "batchCount must be positive.");
        if (n <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), n, "Transform length must be positive.");
        if (!SupportsFftElementType(type))
        {
            throw new NotSupportedException(
                $"FFT element type {type} needs compute capability {type.MinComputeCapabilityX10() / 10}." +
                $"{type.MinComputeCapabilityX10() % 10}; this device reports {_ccMajor}.{_ccMinor}. " +
                "Callers must check SupportsFftElementType first.");
        }

        // A length-1 transform is the identity under the Backward convention (no forward scale, and the inverse
        // scale is 1/1). Returning early also keeps log2(1) = 0 out of the radix-2 stage loop.
        if (n == 1) return;

        EnsureFftGenericModule(type);
        using var _ = PushContext();

        if (CudaFFTGenericKernels.IsPowerOfTwo(n))
            FftGenericRadix2(real, imag, batchCount, n, inverse, type);
        else
            FftGenericBluestein(real, imag, batchCount, n, inverse, type);
    }

    private IntPtr EnsureFftGenericModule(FftElementType type)
    {
        if (_fftGenericModules.TryGetValue(type, out var existing) && existing != IntPtr.Zero)
            return existing;

        lock (_fftGenericCompileLock)
        {
            if (_fftGenericModules.TryGetValue(type, out existing) && existing != IntPtr.Zero)
                return existing;

            string[] names = CudaFFTGenericKernels.GetKernelNames(type);
            IntPtr module = CompileKernelModule(
                _fftGenericDevice,
                CudaFFTGenericKernels.GetSource(type),
                "fftg_" + type.KernelSuffix().TrimStart('_'),
                names);

            // Re-resolve every entry point against THIS module and key it by element type, so the shared
            // entry names emitted into each module cannot alias one another (see the file header).
            foreach (string name in names)
            {
                CuBlasNative.CheckCudaResult(
                    CudaNativeBindings.cuModuleGetFunction(out IntPtr fn, module, name),
                    $"cuModuleGetFunction({name}) for FFT element type {type}");
                _fftGenericKernels[(type, name)] = fn;
            }

            _fftGenericModules[type] = module;
            return module;
        }
    }

    private IntPtr ResolveFftGenericKernel(FftElementType type, string name)
    {
        if (_fftGenericKernels.TryGetValue((type, name), out var fn) && fn != IntPtr.Zero)
            return fn;
        throw new InvalidOperationException($"Generic FFT kernel '{name}' was not compiled for {type}.");
    }

    private static int Log2Exact(int n)
    {
        int bits = 0;
        while ((1 << bits) < n) bits++;
        return bits;
    }

    /// <summary>
    /// In-place batched radix-2 transform on buffers of the caller's element type.
    /// </summary>
    /// <remarks>
    /// The bit-reversal kernel is out-of-place by construction - it reads <c>src[rev]</c> and writes
    /// <c>dst[idx]</c>, which aliases destructively if the two are the same buffer - so the permutation lands in
    /// scratch, the stages run there, and the result is copied back to honour the in-place contract.
    /// </remarks>
    private unsafe void FftGenericRadix2(
        IGpuBuffer re, IGpuBuffer im, int batch, int n, bool inverse, FftElementType type)
    {
        int log2n = Log2Exact(n);
        long elemsLong = (long)batch * n;
        if (elemsLong > int.MaxValue)
            throw new OverflowException($"FFT batch {batch} x length {n} exceeds Int32.MaxValue elements.");
        int elems = (int)elemsLong;
        int bytes = checked(elems * type.ByteSize());
        int half = n / 2;

        string sfx = type.KernelSuffix();
        IGpuBuffer? scratchRe = null, scratchIm = null, twRe = null, twIm = null;
        try
        {
            scratchRe = AllocateByteBuffer(bytes);
            scratchIm = AllocateByteBuffer(bytes);
            twRe = AllocateBuffer(half);
            twIm = AllocateBuffer(half);

            BuildTwiddles(type, twRe, twIm, half, n, inverse);

            // Permute into scratch.
            {
                var k = ResolveFftGenericKernel(type, "fftg_batched_bit_reverse" + sfx);
                IntPtr sR = re.Handle, sI = im.Handle, dR = scratchRe.Handle, dI = scratchIm.Handle;
                int b = batch, len = n, lg = log2n;
                void** args = stackalloc void*[7];
                args[0] = &sR; args[1] = &sI; args[2] = &dR; args[3] = &dI;
                args[4] = &b; args[5] = &len; args[6] = &lg;
                LaunchKernel2D(k, Grid(n), (uint)batch, DefaultBlockSize, 1, args);
            }

            // log2(n) butterfly stages. Each stage touches n/2 butterflies per batch element.
            // The argument buffer is allocated ONCE outside the loop (CA2014): a stackalloc per iteration
            // would grow the frame by log2(n) allocations that are only released when the method returns.
            // args[6] holds the address of `st`, so advancing the stage is a write to `st`.
            {
                var k = ResolveFftGenericKernel(type, "fftg_batched_butterfly" + sfx);
                IntPtr rP = scratchRe.Handle, iP = scratchIm.Handle;
                IntPtr tR = twRe.Handle, tI = twIm.Handle;
                int b = batch, len = n, st = 0;
                void** args = stackalloc void*[7];
                args[0] = &rP; args[1] = &iP; args[2] = &tR; args[3] = &tI;
                args[4] = &b; args[5] = &len; args[6] = &st;
                for (int stride = 2; stride <= n; stride <<= 1)
                {
                    st = stride;
                    LaunchKernel2D(k, Grid(half), (uint)batch, DefaultBlockSize, 1, args);
                }
            }

            // Backward convention: no forward scaling, 1/n on the inverse.
            if (inverse)
            {
                var k = ResolveFftGenericKernel(type, "fftg_scale" + sfx);
                IntPtr rP = scratchRe.Handle, iP = scratchIm.Handle;
                int count = elems;
                float s = 1.0f / n;
                void** args = stackalloc void*[4];
                args[0] = &rP; args[1] = &iP; args[2] = &count; args[3] = &s;
                LaunchKernel(k, Grid(elems), DefaultBlockSize, args);
            }

            CopyBufferDtoD(scratchRe, re, bytes);
            CopyBufferDtoD(scratchIm, im, bytes);
        }
        finally
        {
            scratchRe?.Dispose();
            scratchIm?.Dispose();
            twRe?.Dispose();
            twIm?.Dispose();
        }
    }

    /// <summary>
    /// Arbitrary-length transform by Bluestein's chirp-z algorithm, entirely on device.
    /// </summary>
    /// <remarks>
    /// The workspace is float32 whatever the caller's element type is. It is transient scratch that the caller
    /// never sees, so narrowing it would compound rounding through two extra transforms while saving no traffic
    /// anyone observes; the narrow type is applied where it pays, on the caller's own buffers, by the premul and
    /// postmul kernels.
    /// </remarks>
    private unsafe void FftGenericBluestein(
        IGpuBuffer re, IGpuBuffer im, int batch, int n, bool inverse, FftElementType type)
    {
        int m = CudaFFTGenericKernels.BluesteinLength(n);
        long wsLong = (long)batch * m;
        if (wsLong > int.MaxValue)
            throw new OverflowException($"Bluestein workspace {batch} x {m} exceeds Int32.MaxValue elements.");
        int ws = (int)wsLong;

        string sfx = type.KernelSuffix();
        IGpuBuffer? chRe = null, chIm = null, bRe = null, bIm = null, wRe = null, wIm = null;
        try
        {
            chRe = AllocateBuffer(n);
            chIm = AllocateBuffer(n);
            bRe = AllocateBuffer(m);
            bIm = AllocateBuffer(m);
            wRe = AllocateBuffer(ws);
            wIm = AllocateBuffer(ws);

            // chirp c[k] = exp(i*pi*k^2/n), and its symmetrically extended, zero-padded length-m form.
            {
                var k = ResolveFftGenericKernel(type, "fftg_build_chirp");
                IntPtr cR = chRe.Handle, cI = chIm.Handle;
                int len = n, inv = inverse ? 1 : 0;
                void** args = stackalloc void*[4];
                args[0] = &cR; args[1] = &cI; args[2] = &len; args[3] = &inv;
                LaunchKernel(k, Grid(n), DefaultBlockSize, args);
            }
            {
                var k = ResolveFftGenericKernel(type, "fftg_build_chirp_padded");
                IntPtr pR = bRe.Handle, pI = bIm.Handle;
                int len = n, mm = m, inv = inverse ? 1 : 0;
                void** args = stackalloc void*[5];
                args[0] = &pR; args[1] = &pI; args[2] = &len; args[3] = &mm; args[4] = &inv;
                LaunchKernel(k, Grid(m), DefaultBlockSize, args);
            }

            // B = FFT(b), a single length-m transform of the convolution kernel.
            Fp32Radix2(type, bRe, bIm, 1, m, false);

            // w = zero-padded x * conj(chirp), in the caller's element type on the way in.
            {
                var k = ResolveFftGenericKernel(type, "fftg_bluestein_premul" + sfx);
                IntPtr xR = re.Handle, xI = im.Handle, cR = chRe.Handle, cI = chIm.Handle;
                IntPtr oR = wRe.Handle, oI = wIm.Handle;
                int b = batch, len = n, mm = m;
                void** args = stackalloc void*[9];
                args[0] = &xR; args[1] = &xI; args[2] = &cR; args[3] = &cI;
                args[4] = &oR; args[5] = &oI; args[6] = &b; args[7] = &len; args[8] = &mm;
                LaunchKernel2D(k, Grid(m), (uint)batch, DefaultBlockSize, 1, args);
            }

            Fp32Radix2(type, wRe, wIm, batch, m, false);

            // Cyclic convolution in the transform domain. The symmetric extension above is what makes this
            // equal the LINEAR convolution Bluestein requires.
            {
                var k = ResolveFftGenericKernel(type, "fftg_bluestein_pointwise");
                IntPtr oR = wRe.Handle, oI = wIm.Handle, kR = bRe.Handle, kI = bIm.Handle;
                int b = batch, mm = m;
                void** args = stackalloc void*[6];
                args[0] = &oR; args[1] = &oI; args[2] = &kR; args[3] = &kI; args[4] = &b; args[5] = &mm;
                LaunchKernel2D(k, Grid(m), (uint)batch, DefaultBlockSize, 1, args);
            }

            Fp32Radix2(type, wRe, wIm, batch, m, true);

            // First n outputs * conj(chirp), narrowed back to the caller's type. The 1/n of the Backward
            // convention is folded in here rather than launched as a separate pass.
            {
                var k = ResolveFftGenericKernel(type, "fftg_bluestein_postmul" + sfx);
                IntPtr iR = wRe.Handle, iI = wIm.Handle, cR = chRe.Handle, cI = chIm.Handle;
                IntPtr yR = re.Handle, yI = im.Handle;
                int b = batch, len = n, mm = m;
                float scale = inverse ? 1.0f / n : 1.0f;
                void** args = stackalloc void*[10];
                args[0] = &iR; args[1] = &iI; args[2] = &cR; args[3] = &cI;
                args[4] = &yR; args[5] = &yI; args[6] = &b; args[7] = &len; args[8] = &mm; args[9] = &scale;
                LaunchKernel2D(k, Grid(n), (uint)batch, DefaultBlockSize, 1, args);
            }
        }
        finally
        {
            chRe?.Dispose();
            chIm?.Dispose();
            bRe?.Dispose();
            bIm?.Dispose();
            wRe?.Dispose();
            wIm?.Dispose();
        }
    }

    /// <summary>float32 in-place batched radix-2 transform over the Bluestein workspace.</summary>
    private unsafe void Fp32Radix2(FftElementType type, IGpuBuffer re, IGpuBuffer im, int batch, int n, bool inverse)
    {
        int log2n = Log2Exact(n);
        int elems = checked(batch * n);
        int half = n / 2;

        IGpuBuffer? scratchRe = null, scratchIm = null, twRe = null, twIm = null;
        try
        {
            scratchRe = AllocateBuffer(elems);
            scratchIm = AllocateBuffer(elems);
            twRe = AllocateBuffer(half);
            twIm = AllocateBuffer(half);

            BuildTwiddles(type, twRe, twIm, half, n, inverse);

            {
                var k = ResolveFftGenericKernel(type, "fftg_f32_bit_reverse");
                IntPtr sR = re.Handle, sI = im.Handle, dR = scratchRe.Handle, dI = scratchIm.Handle;
                int b = batch, len = n, lg = log2n;
                void** args = stackalloc void*[7];
                args[0] = &sR; args[1] = &sI; args[2] = &dR; args[3] = &dI;
                args[4] = &b; args[5] = &len; args[6] = &lg;
                LaunchKernel2D(k, Grid(n), (uint)batch, DefaultBlockSize, 1, args);
            }

            {
                // stackalloc hoisted out of the loop (CA2014); args[6] holds &st, so the stage advances by
                // writing st rather than by rebuilding the argument buffer.
                var k = ResolveFftGenericKernel(type, "fftg_f32_butterfly");
                IntPtr rP = scratchRe.Handle, iP = scratchIm.Handle;
                IntPtr tR = twRe.Handle, tI = twIm.Handle;
                int b = batch, len = n, st = 0;
                void** args = stackalloc void*[7];
                args[0] = &rP; args[1] = &iP; args[2] = &tR; args[3] = &tI;
                args[4] = &b; args[5] = &len; args[6] = &st;
                for (int stride = 2; stride <= n; stride <<= 1)
                {
                    st = stride;
                    LaunchKernel2D(k, Grid(half), (uint)batch, DefaultBlockSize, 1, args);
                }
            }

            if (inverse)
            {
                var k = ResolveFftGenericKernel(type, "fftg_f32_scale");
                IntPtr rP = scratchRe.Handle, iP = scratchIm.Handle;
                int count = elems;
                float s = 1.0f / n;
                void** args = stackalloc void*[4];
                args[0] = &rP; args[1] = &iP; args[2] = &count; args[3] = &s;
                LaunchKernel(k, Grid(elems), DefaultBlockSize, args);
            }

            CopyBufferDtoD(scratchRe, re, (long)elems * sizeof(float));
            CopyBufferDtoD(scratchIm, im, (long)elems * sizeof(float));
        }
        finally
        {
            scratchRe?.Dispose();
            scratchIm?.Dispose();
            twRe?.Dispose();
            twIm?.Dispose();
        }
    }

    private unsafe void BuildTwiddles(FftElementType type, IGpuBuffer twRe, IGpuBuffer twIm, int half, int n, bool inverse)
    {
        var k = ResolveFftGenericKernel(type, "fftg_build_twiddles");
        IntPtr tR = twRe.Handle, tI = twIm.Handle;
        int h = half, len = n, inv = inverse ? 1 : 0;
        void** args = stackalloc void*[5];
        args[0] = &tR; args[1] = &tI; args[2] = &h; args[3] = &len; args[4] = &inv;
        LaunchKernel(k, Grid(half), DefaultBlockSize, args);
    }

    private static uint Grid(int threads) => (uint)((threads + DefaultBlockSize - 1) / DefaultBlockSize);
}
