using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    // GPU buffer APIs never emulate a missing kernel on the host. A registration or
    // compilation gap must remain observable instead of silently breaking residency.
    public unsafe void ReduceMean(IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (size <= 0) return;
        string kernelName = GpuDeterminism.IsActive ? "reduce_mean_deterministic" : "reduce_mean";
        IntPtr kernel = RequireResidentKernel(kernelName);
        Fill(output, 0f, 1);
        IntPtr inputPtr = input.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &size;
        LaunchKernel(kernel, GpuDeterminism.IsActive ? 1u : GridFor(size), DefaultBlockSize, args);
    }

    public unsafe void ClipKernel(IGpuBuffer input, IGpuBuffer output, float min, float max, int size)
    {
        if (size <= 0) return;
        IntPtr kernel = RequireResidentKernel("clip_kernel");
        IntPtr inputPtr = input.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &min; args[3] = &max; args[4] = &size;
        LaunchKernel(kernel, GridFor(size), DefaultBlockSize, args);
    }

    public void PowScalar(IGpuBuffer input, IGpuBuffer output, float exponent, int size)
        => LaunchResidentScalar("pow_scalar", input, output, exponent, size);

    public void FracKernel(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchResidentUnary("frac_kernel", input, output, size, size);

    public unsafe void EyeKernel(IGpuBuffer output, int n)
    {
        int total = n * n;
        if (total <= 0) return;
        IntPtr kernel = RequireResidentKernel("eye_kernel");
        IntPtr outputPtr = output.Handle;
        void** args = stackalloc void*[2];
        args[0] = &outputPtr; args[1] = &n;
        LaunchKernel(kernel, GridFor(total), DefaultBlockSize, args);
    }

    public unsafe void OneHotKernel(IGpuBuffer indices, IGpuBuffer output, int batchSize, int numClasses)
    {
        int total = batchSize * numClasses;
        if (total <= 0) return;
        IntPtr kernel = RequireResidentKernel("one_hot_kernel");
        IntPtr indicesPtr = indices.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &indicesPtr; args[1] = &outputPtr; args[2] = &batchSize; args[3] = &numClasses;
        LaunchKernel(kernel, GridFor(total), DefaultBlockSize, args);
    }

    public unsafe void MaskedFillKernel(IGpuBuffer input, IGpuBuffer mask, IGpuBuffer output, float fillValue, int size)
    {
        if (size <= 0) return;
        IntPtr kernel = RequireResidentKernel("masked_fill_kernel");
        IntPtr inputPtr = input.Handle, maskPtr = mask.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &inputPtr; args[1] = &maskPtr; args[2] = &outputPtr; args[3] = &fillValue; args[4] = &size;
        LaunchKernel(kernel, GridFor(size), DefaultBlockSize, args);
    }

    public void EqualsKernel(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size)
        => LaunchResidentBinary("equals_kernel", left, right, output, size, size);

    public void NotEqualsKernel(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size)
        => LaunchResidentBinary("not_equals_kernel", left, right, output, size, size);

    public unsafe void OuterProduct(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int m, int n)
    {
        int total = m * n;
        if (total <= 0) return;
        IntPtr kernel = RequireResidentKernel("outer_product");
        IntPtr leftPtr = left.Handle, rightPtr = right.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &leftPtr; args[1] = &rightPtr; args[2] = &outputPtr; args[3] = &m; args[4] = &n;
        LaunchKernel(kernel, GridFor(total), DefaultBlockSize, args);
    }

    public void BatchDotProduct(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int batchSize, int dim)
        => LaunchResidentAxis("batch_dot_product", left, right, output, batchSize, dim, batchSize);

    public void GluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim)
        => LaunchResidentUnary("glu_forward", input, output, outerSize, halfDim, outerSize * halfDim);

    public void GeGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim)
        => LaunchResidentUnary("geglu_forward", input, output, outerSize, halfDim, outerSize * halfDim);

    public void ReGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim)
        => LaunchResidentUnary("reglu_forward", input, output, outerSize, halfDim, outerSize * halfDim);

    public void SwiGluForward(IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim)
        => LaunchResidentUnary("swiglu_forward", input, output, outerSize, halfDim, outerSize * halfDim);

    public void BceLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int size)
        => LaunchResidentBinary("bce_loss", predictions, targets, loss, size, size);

    public void AddScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => LaunchResidentScalar("add_scalar", input, output, scalar, size);

    public void SubScalar(IGpuBuffer input, IGpuBuffer output, float scalar, int size)
        => LaunchResidentScalar("sub_scalar", input, output, scalar, size);

    public void BroadcastAddLast(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchResidentAxis("broadcast_add_last", left, right, output, outerSize, innerSize, outerSize * innerSize);

    public void BroadcastSubLast(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchResidentAxis("broadcast_sub_last", left, right, output, outerSize, innerSize, outerSize * innerSize);

    public void BroadcastMulLast(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchResidentAxis("broadcast_mul_last", left, right, output, outerSize, innerSize, outerSize * innerSize);

    public void BroadcastDivLast(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int outerSize, int innerSize)
        => LaunchResidentAxis("broadcast_div_last", left, right, output, outerSize, innerSize, outerSize * innerSize);
    public unsafe void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        // Issue #382: deterministic variant fixes accumulation order by
        // launching a single block (kernel itself ignores blockIdx.x — only
        // threadIdx.x stride is used) and writing `*result = sum` instead of
        // atomicAdd.
        string kname = GpuDeterminism.IsActive ? "dot_product_deterministic" : "dot_product";
        if (size <= 0) return;
        IntPtr kernel = RequireResidentKernel(kname);
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outputPtr; args[3] = &size;
        uint grid = GpuDeterminism.IsActive ? 1u : GridFor(size);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
    public unsafe void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size, int strideA, int strideB, int count)
    {
        // Issue #382: deterministic variant launched single-block (kernel uses
        // only threadIdx.x stride). Note: the kernel signature is
        // (a, b, result, aSize, bSize, bOffset, bStride) — preserve the existing
        // wire-up; argument-name mismatch is a separate concern.
        string kname = GpuDeterminism.IsActive ? "strided_dot_product_deterministic" : "strided_dot_product";
        if (size <= 0 || count <= 0) return;
        IntPtr kernel = RequireResidentKernel(kname);
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[7];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outputPtr; args[3] = &size;
        args[4] = &strideA; args[5] = &strideB; args[6] = &count;
        uint grid = GpuDeterminism.IsActive ? 1u : GridFor(count);
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public void BatchedDotProduct(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int batchSize, int dim)
        => LaunchResidentAxis("batched_dot_product", left, right, output, batchSize, dim, batchSize);

    private IntPtr RequireResidentKernel(string kernelName)
    {
        if (!_kernelCache.TryGetValue(kernelName, out IntPtr kernel))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        return kernel;
    }

    private static uint GridFor(int workSize)
        => (uint)((workSize + DefaultBlockSize - 1) / DefaultBlockSize);

    private unsafe void LaunchResidentUnary(string kernelName, IGpuBuffer input, IGpuBuffer output, int size, int workSize)
    {
        if (workSize <= 0) return;
        IntPtr kernel = RequireResidentKernel(kernelName);
        IntPtr inputPtr = input.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &size;
        LaunchKernel(kernel, GridFor(workSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchResidentUnary(
        string kernelName, IGpuBuffer input, IGpuBuffer output,
        int outerSize, int innerSize, int workSize)
    {
        if (workSize <= 0) return;
        IntPtr kernel = RequireResidentKernel(kernelName);
        IntPtr inputPtr = input.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &outerSize; args[3] = &innerSize;
        LaunchKernel(kernel, GridFor(workSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchResidentScalar(
        string kernelName, IGpuBuffer input, IGpuBuffer output, float scalar, int size)
    {
        if (size <= 0) return;
        IntPtr kernel = RequireResidentKernel(kernelName);
        IntPtr inputPtr = input.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &inputPtr; args[1] = &outputPtr; args[2] = &scalar; args[3] = &size;
        LaunchKernel(kernel, GridFor(size), DefaultBlockSize, args);
    }

    private unsafe void LaunchResidentBinary(
        string kernelName, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size, int workSize)
    {
        if (workSize <= 0) return;
        IntPtr kernel = RequireResidentKernel(kernelName);
        IntPtr leftPtr = left.Handle, rightPtr = right.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &leftPtr; args[1] = &rightPtr; args[2] = &outputPtr; args[3] = &size;
        LaunchKernel(kernel, GridFor(workSize), DefaultBlockSize, args);
    }

    private unsafe void LaunchResidentAxis(
        string kernelName, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output,
        int outerSize, int innerSize, int workSize)
    {
        if (workSize <= 0) return;
        IntPtr kernel = RequireResidentKernel(kernelName);
        IntPtr leftPtr = left.Handle, rightPtr = right.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &leftPtr; args[1] = &rightPtr; args[2] = &outputPtr;
        args[3] = &outerSize; args[4] = &innerSize;
        LaunchKernel(kernel, GridFor(workSize), DefaultBlockSize, args);
    }

    // --- Split-buffer native Complex<T> operations (HIP dispatch) ---

    private static void ValidateHipSplitBuffers(int n, string opName, params IGpuBuffer[] buffers)
    {
        foreach (var buf in buffers)
        {
            if (buf is null) throw new ArgumentNullException(nameof(buffers), $"{opName}: GPU buffer cannot be null.");
            if (n > buf.Size) throw new ArgumentException($"{opName}: n ({n}) exceeds buffer size ({buf.Size}).");
        }
    }

    public unsafe void SplitComplexMultiply(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMultiply), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_multiply", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_multiply. Register HipComplexKernels.");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void InterleaveComplex(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer interleaved, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(InterleaveComplex), real, imag);
        if (interleaved is null) throw new ArgumentNullException(nameof(interleaved));
        long requiredInterleaved = checked((long)n * 2);
        if (interleaved.Size < requiredInterleaved)
            throw new ArgumentException($"{nameof(InterleaveComplex)}: n ({n}) requires {requiredInterleaved} interleaved elements but buffer size is {interleaved.Size}.", nameof(interleaved));
        if (!_kernelCache.TryGetValue("interleave_complex", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: interleave_complex");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pR = real.Handle, pI = imag.Handle, pO = interleaved.Handle;
        void** args = stackalloc void*[4]; args[0] = &pR; args[1] = &pI; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void DeinterleaveComplex(IGpuBuffer interleaved, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(DeinterleaveComplex), real, imag);
        if (interleaved is null) throw new ArgumentNullException(nameof(interleaved));
        long requiredInterleaved = checked((long)n * 2);
        if (interleaved.Size < requiredInterleaved)
            throw new ArgumentException($"{nameof(DeinterleaveComplex)}: n ({n}) requires {requiredInterleaved} interleaved elements but buffer size is {interleaved.Size}.", nameof(interleaved));
        if (!_kernelCache.TryGetValue("deinterleave_complex", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: deinterleave_complex");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIn = interleaved.Handle, pR = real.Handle, pI = imag.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIn; args[1] = &pR; args[2] = &pI; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexConjugate(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexConjugate), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_conjugate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_conjugate");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5]; args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitude(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMagnitude), inReal, inImag, outMag);
        if (!_kernelCache.TryGetValue("split_complex_magnitude", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_magnitude");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMag.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitudeSquared(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMagSq, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMagnitudeSquared), inReal, inImag, outMagSq);
        if (!_kernelCache.TryGetValue("split_complex_magnitude_squared", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_magnitude_squared");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMagSq.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexPhase(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outPhase, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexPhase), inReal, inImag, outPhase);
        if (!_kernelCache.TryGetValue("split_complex_phase", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_phase");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outPhase.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexFromPolar(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexFromPolar), mag, phase, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_from_polar", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_from_polar");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pM = mag.Handle, pP = phase.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5]; args[0] = &pM; args[1] = &pP; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexScale(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, float scalar, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexScale), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_scale", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_scale");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[6]; args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &scalar; args[5] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexAdd(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexAdd), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_add");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexCrossSpectral(IGpuBuffer xReal, IGpuBuffer xImag, IGpuBuffer yReal, IGpuBuffer yImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexCrossSpectral), xReal, xImag, yReal, yImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_cross_spectral", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_cross_spectral");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pXR = xReal.Handle, pXI = xImag.Handle, pYR = yReal.Handle, pYI = yImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pXR; args[1] = &pXI; args[2] = &pYR; args[3] = &pYI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexTopK(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n, int k)
    {
        if (n <= 0 || k <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexTopK), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_topk", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_topk");
        k = Math.Min(k, n);
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &k; args[5] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        if (!_kernelCache.TryGetValue("softmax_rows", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: softmax_rows");
        uint grid = (uint)rows;
        int blockSize = Math.Min(256, cols);
        int sharedMem = blockSize * sizeof(float);
        IntPtr pI = input.Handle, pO = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pI; args[1] = &pO; args[2] = &rows; args[3] = &cols;
        HipNativeBindings.hipModuleLaunchKernel(kernel, grid, 1, 1, (uint)blockSize, 1, 1,
            (uint)sharedMem, IntPtr.Zero, (IntPtr)args, IntPtr.Zero);
    }

    // ─── HRR binding primitives (issue #248) ────────────────────────

    public unsafe void SplitComplexUnitPhaseCodebook(
        IGpuBuffer outReal, IGpuBuffer outImag, int seed, int V, int D, bool kPsk, int k)
    {
        // Reject negative dims up front — if both are negative, their
        // product is positive and would slip past a naive "total <= 0"
        // check, then launch the kernel with invalid shape.
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (V == 0 || D == 0) return;
        long total = (long)V * D;
        if (total > int.MaxValue) throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        if (kPsk && k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        int n = (int)total;
        ValidateHipSplitBuffers(n, nameof(SplitComplexUnitPhaseCodebook), outReal, outImag);
        if (!_kernelCache.TryGetValue("hrr_unit_phase_codebook", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_unit_phase_codebook");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        int kPskI = kPsk ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &pOR; args[1] = &pOI; args[2] = &seed;
        args[3] = &V; args[4] = &D; args[5] = &kPskI; args[6] = &k;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexPhaseCoherenceDecode(
        IGpuBuffer codesReal, IGpuBuffer codesImag,
        IGpuBuffer queryReal, IGpuBuffer queryImag,
        IGpuBuffer outScores, int V, int D)
    {
        if (V <= 0 || D <= 0) return;
        long codeCount = (long)V * D;
        if (codesReal is null || codesImag is null || queryReal is null || queryImag is null || outScores is null)
            throw new ArgumentNullException(nameof(codesReal),
                "All five GPU buffers must be non-null for SplitComplexPhaseCoherenceDecode.");
        if (codesReal.Size < codeCount || codesImag.Size < codeCount)
            throw new ArgumentException(
                $"codes buffers must each hold at least V*D = {codeCount} elements " +
                $"(got {codesReal.Size}, {codesImag.Size}).");
        if (queryReal.Size < D || queryImag.Size < D)
            throw new ArgumentException(
                $"query buffers must each hold at least D = {D} elements " +
                $"(got {queryReal.Size}, {queryImag.Size}).");
        if (outScores.Size < V)
            throw new ArgumentException(
                $"outScores must hold at least V = {V} elements (got {outScores.Size}).");
        if (!_kernelCache.TryGetValue("hrr_phase_coherence_decode", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_phase_coherence_decode");
        uint grid = (uint)V;
        int blockSize = 1;
        while (blockSize < D && blockSize < 1024) blockSize <<= 1;
        blockSize = Math.Max(32, blockSize);
        uint sharedMem = (uint)blockSize * sizeof(float);
        IntPtr pCR = codesReal.Handle, pCI = codesImag.Handle;
        IntPtr pQR = queryReal.Handle, pQI = queryImag.Handle;
        IntPtr pOS = outScores.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pCR; args[1] = &pCI;
        args[2] = &pQR; args[3] = &pQI;
        args[4] = &pOS; args[5] = &V; args[6] = &D;
        // Route through the shared helper so this runs on _stream with
        // the same error-surfacing as the rest of the backend.
        LaunchKernelWithSharedMem(kernel, grid, (uint)blockSize, sharedMem, args);
    }

    public unsafe void SplitComplexHrrBindAccumulate(
        IGpuBuffer keyCodeReal, IGpuBuffer keyCodeImag,
        IGpuBuffer valPermCodeReal, IGpuBuffer valPermCodeImag,
        IGpuBuffer keyIds, IGpuBuffer valIds,
        IGpuBuffer memoryReal, IGpuBuffer memoryImag,
        int N, int D)
    {
        if (N <= 0 || D <= 0) return;
        if (keyCodeReal is null || keyCodeImag is null || valPermCodeReal is null || valPermCodeImag is null
            || keyIds is null || valIds is null || memoryReal is null || memoryImag is null)
            throw new ArgumentNullException(nameof(keyCodeReal),
                "All eight GPU buffers must be non-null for SplitComplexHrrBindAccumulate.");
        if (keyIds.Size < N || valIds.Size < N)
            throw new ArgumentException(
                $"ID buffers must each hold at least N = {N} elements " +
                $"(got keyIds={keyIds.Size}, valIds={valIds.Size}).");
        if (memoryReal.Size < D || memoryImag.Size < D)
            throw new ArgumentException(
                $"Memory buffers must each hold at least D = {D} elements " +
                $"(got memoryReal={memoryReal.Size}, memoryImag={memoryImag.Size}).");
        if (keyCodeReal.Size < D || keyCodeImag.Size < D
            || valPermCodeReal.Size < D || valPermCodeImag.Size < D)
            throw new ArgumentException(
                $"Each codebook buffer must hold at least one full row of D = {D} elements.");
        // Derive vocabulary sizes from codebook capacities so the
        // kernel can reject out-of-range ids without OOB reads —
        // mirrors the CUDA backend's nKeys/nVals guard and the CPU
        // path's range check in NativeHRRBindAccumulate.
        long nKeysL = keyCodeReal.Size / D;
        long nValsL = valPermCodeReal.Size / D;
        if (nKeysL <= 0 || nValsL <= 0)
            throw new ArgumentException(
                $"Codebook buffers must hold at least one full row of D = {D} elements.");
        if (nKeysL > int.MaxValue || nValsL > int.MaxValue)
            throw new ArgumentException("Codebook vocabulary size exceeds int.MaxValue.");
        int nKeys = (int)nKeysL;
        int nVals = (int)nValsL;
        if (!_kernelCache.TryGetValue("hrr_bind_accumulate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_bind_accumulate");
        uint grid = (uint)((D + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pKR = keyCodeReal.Handle, pKI = keyCodeImag.Handle;
        IntPtr pVR = valPermCodeReal.Handle, pVI = valPermCodeImag.Handle;
        IntPtr pKId = keyIds.Handle, pVId = valIds.Handle;
        IntPtr pMR = memoryReal.Handle, pMI = memoryImag.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pKR; args[1] = &pKI;
        args[2] = &pVR; args[3] = &pVI;
        args[4] = &pKId; args[5] = &pVId;
        args[6] = &pMR; args[7] = &pMI;
        args[8] = &N; args[9] = &D;
        args[10] = &nKeys; args[11] = &nVals;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    // ── Fused GLA (Gated Linear Attention) scan (#1464) ────────────────────────────────────
    // q/k/v: [batch, seqLen, modelDim]; gate: [batch, seqLen, numHeads]; output: [batch, seqLen, modelDim].
    public unsafe void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.HipGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.HipGlaKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gla_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gla_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pg = gate.Handle, po = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pg; args[4] = &po;
        args[5] = &batch; args[6] = &seqLen; args[7] = &modelDim; args[8] = &numHeads; args[9] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // dQ/dK/dV (each [batch, seqLen, modelDim]) and dG ([batch, seqLen, numHeads]); dQ/dK/dG are
    // the atomicAdd accumulators and are zeroed internally below (no caller pre-zeroing required).
    public unsafe void GlaScanBackward(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.HipGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.HipGlaKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gla_scan_recompute", out var recompute) ||
            !_kernelCache.TryGetValue("gla_scan_backward", out var backward))
            throw new InvalidOperationException("HIP kernel not found: gla_scan_recompute / gla_scan_backward");

        // State trajectory scratch: [batch * numHeads * seqLen * headDim * headDim].
        long trajLenLong = (long)batch * numHeads * seqLen * headDim * headDim;
        if (trajLenLong > int.MaxValue)
            throw new InvalidOperationException(
                $"GLA trajectory buffer size ({trajLenLong}) exceeds int.MaxValue.");
        int trajLen = (int)trajLenLong;
        using var trajBuf = AllocateBuffer(trajLen);
        // dQ/dK/dG are the atomicAdd accumulators — zero them here so the method is
        // self-contained and correct even if the caller reuses dirty buffers.
        Fill(dQ, 0f, batch * seqLen * modelDim);
        Fill(dK, 0f, batch * seqLen * modelDim);
        Fill(dG, 0f, batch * seqLen * numHeads);
        uint total = (uint)(batch * numHeads * headDim);
        uint grid = (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize;

        IntPtr pk = k.Handle, pv = v.Handle, pg = gate.Handle, ptr = trajBuf.Handle;
        void** ra = stackalloc void*[9];
        ra[0] = &pk; ra[1] = &pv; ra[2] = &pg; ra[3] = &ptr;
        ra[4] = &batch; ra[5] = &seqLen; ra[6] = &modelDim; ra[7] = &numHeads; ra[8] = &headDim;
        LaunchKernel(recompute, grid, (uint)DefaultBlockSize, ra);

        IntPtr pdo = dOut.Handle, pq = q.Handle;
        IntPtr pdq = dQ.Handle, pdk = dK.Handle, pdv = dV.Handle, pdg = dG.Handle;
        void** ba = stackalloc void*[15];
        ba[0] = &pdo; ba[1] = &pq; ba[2] = &pk; ba[3] = &pv; ba[4] = &pg; ba[5] = &ptr;
        ba[6] = &pdq; ba[7] = &pdk; ba[8] = &pdv; ba[9] = &pdg;
        ba[10] = &batch; ba[11] = &seqLen; ba[12] = &modelDim; ba[13] = &numHeads; ba[14] = &headDim;
        LaunchKernel(backward, grid, (uint)DefaultBlockSize, ba);
    }

    // ── Fused xLSTM (mLSTM) scan forward (#1464) ───────────────────────────────────────────
    public unsafe void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "xLSTM dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.HipXLstmKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"xLSTM headDim ({headDim}) exceeds max ({Kernels.HipXLstmKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("xlstm_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: xlstm_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle;
        IntPtr pi = iGate.Handle, pf = fGate.Handle, po = oGate.Handle, pout = output.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pi; args[4] = &pf; args[5] = &po; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &modelDim; args[10] = &numHeads; args[11] = &headDim;
        uint total = (uint)(batch * numHeads);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Gated DeltaNet scan forward (#1464) ──────────────────────────────────────────
    public unsafe void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GatedDeltaNet dimensions must be positive.");
        if (modelDim != numHeads * headDim)
            throw new ArgumentException($"modelDim ({modelDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (headDim > Kernels.HipGatedDeltaNetKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GatedDeltaNet headDim ({headDim}) exceeds max ({Kernels.HipGatedDeltaNetKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gated_delta_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gated_delta_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pa = alpha.Handle, pb = beta.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pa; args[4] = &pb; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim; args[9] = &numHeads; args[10] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RG-LRU scan forward (#1464) ──────────────────────────────────────────────────
    public unsafe void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
    {
        if (batch <= 0 || seqLen <= 0 || recDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "RG-LRU dimensions must be positive.");
        if (!_kernelCache.TryGetValue("rglru_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rglru_scan_forward");

        IntPtr pv = value.Handle, pr = recGate.Handle, pi = inpGate.Handle, pd = decay.Handle, pout = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pv; args[1] = &pr; args[2] = &pi; args[3] = &pd; args[4] = &pout;
        args[5] = &batch; args[6] = &seqLen; args[7] = &recDim;
        uint total = (uint)(batch * recDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RWKV-4 WKV scan forward (#1464) ──────────────────────────────────────────────
    public unsafe void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "RWKV-4 dimensions must be positive.");
        if (!_kernelCache.TryGetValue("rwkv4_wkv_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rwkv4_wkv_forward");

        IntPtr pr = r.Handle, pk = k.Handle, pv = v.Handle, pd = timeDecay.Handle, pf = timeFirst.Handle, pout = output.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pr; args[1] = &pk; args[2] = &pv; args[3] = &pd; args[4] = &pf; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim;
        uint total = (uint)(batch * modelDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba selective scan forward (#1464) ─────────────────────────────────────────
    public unsafe void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
    {
        if (batch <= 0 || seqLen <= 0 || innerDim <= 0 || stateDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Mamba dimensions must be positive.");
        if (stateDim > Kernels.HipMambaKernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba stateDim ({stateDim}) exceeds max ({Kernels.HipMambaKernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba_selective_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mamba_selective_scan_forward");

        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba-2 SSD scan forward (#1464) ─────────────────────────────────────────────
    public unsafe void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
    {
        if (batch <= 0 || seqLen <= 0 || innerDim <= 0 || numHeads <= 0 || headDim <= 0 || stateDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Mamba-2 dimensions must be positive.");
        if (innerDim != numHeads * headDim)
            throw new ArgumentException($"innerDim ({innerDim}) must equal numHeads * headDim ({numHeads * headDim}).");
        if (stateDim > Kernels.HipMamba2Kernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba-2 stateDim ({stateDim}) exceeds max ({Kernels.HipMamba2Kernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba2_ssd_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mamba2_ssd_scan_forward");

        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[13];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &numHeads; args[11] = &headDim; args[12] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused linear (LM head) + cross-entropy (#1464) ─────────────────────────────────────
    // Returns the mean cross-entropy over the N rows. targetIds holds the class ids as floats.
    public unsafe float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_index", hidden, weight, bias, targetIds, n, d, vocab);

    public unsafe float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_dense", hidden, weight, bias, target, n, d, vocab);

    public unsafe void FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeLaunchResident("fused_linear_ce_index", hidden, weight, bias, targetIds, meanLoss, n, d, vocab);

    public unsafe void FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeLaunchResident("fused_linear_ce_dense", hidden, weight, bias, target, meanLoss, n, d, vocab);

    private unsafe float FusedCeLaunch(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        using var lossBuf = AllocateBuffer(1);
        FusedCeLaunchResident(kernelName, hidden, weight, bias, tgt, lossBuf, n, d, vocab);
        return DownloadBuffer(lossBuf)[0];
    }

    private unsafe void FusedCeLaunchResident(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt,
        IGpuBuffer meanLoss, int n, int d, int vocab)
    {
        if (n <= 0 || d <= 0 || vocab <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Fused CE dimensions (n, d, vocab) must be positive.");
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        Fill(meanLoss, 0f, 1);
        IntPtr ph = hidden.Handle, pw = weight.Handle, pb = bias.Handle, pt = tgt.Handle, pl = meanLoss.Handle;
        void** args = stackalloc void*[8];
        args[0] = &ph; args[1] = &pw; args[2] = &pb; args[3] = &pt; args[4] = &pl;
        args[5] = &n; args[6] = &d; args[7] = &vocab;
        uint total = (uint)n;
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
        Scale(meanLoss, meanLoss, 1f / n, 1);
    }
}
