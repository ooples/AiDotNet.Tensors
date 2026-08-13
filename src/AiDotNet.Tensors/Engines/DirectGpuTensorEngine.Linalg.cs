// Copyright (c) AiDotNet. All rights reserved.
// DirectGpuTensorEngine GPU routing for torch.linalg decompositions (#211).
// Each method tries the GPU path (via ILinalgBackend on the active backend)
// and falls back to the managed CPU implementation when:
//   - the active backend doesn't advertise ILinalgBackend, or
//   - the problem size exceeds the per-backend kernel limit, or
//   - the kernel launch throws (e.g. driver rejected the shader).
//
// The GPU kernels we ship today are self-contained: one workgroup per batch
// slice, with the matrix dimension bounded by the workgroup-size limit
// (typically n ≤ 256 on Vulkan/WebGPU/OpenCL, ≤ 1024 on CUDA/HIP/Metal).
// Beyond that we transparently route through the managed tier.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Chosen conservatively to cover every backend's workgroup-size cap.
    // Matrices larger than this route to CPU. Individual backends support more
    // (CUDA/HIP up to 1024); this cap is the common-denominator safety net.
    private const int GpuLinalgMaxN = 256;

    /// <summary>
    /// Backends that can report a compile-failure capability flag (currently
    /// only OpenCL) pass through this probe. Returns true when linalg kernels
    /// are known-good, false when they failed to compile and the caller must
    /// fall back. Backends without a flag default to true.
    /// </summary>
    private static bool BackendLinalgReady(IDirectGpuBackend backend)
    {
#if !NET462
        if (backend is AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend ocl)
            return ocl.LinalgAvailable;
#endif
        return true;
    }

    /// <summary>
    /// Per-backend hard cap on Eigh's matrix dimension. Each backend's
    /// shader statically provisions its Ash/Vsh workgroup arrays, and
    /// exceeding those bounds silently reads/writes past the workgroup
    /// memory region. Keep this table in sync with the kernel sources.
    /// </summary>
    private static int BackendEighMaxN(IDirectGpuBackend backend)
    {
#if NET7_0_OR_GREATER
        if (backend is AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend)
            return 32; // WebGpuLinalgKernels.Eigh: Ash[1024] = 32×32.
#endif
        // CUDA/HIP/Metal/OpenCL/Vulkan all provision 64×64.
        return 64;
    }

    /// <summary>
    /// GPU-accelerated Cholesky. Returns null on unsupported backends or
    /// oversized matrices so callers fall back to the CPU path.
    /// </summary>
    internal (Tensor<float>? Factor, Tensor<int>? Info) TryGpuCholesky(
        Tensor<float> input, bool upper)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) return (null, null);
        if (n == 0) return (null, null); // Empty matrix — fall back to CPU.
        if (n > GpuLinalgMaxN) return (null, null);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        if (batch == 0) return (null, null);

        // Split the try block at the ownership handoff to FinishGpuOp: any throw
        // BEFORE the handoff disposes outBuf (we still own it); throws AFTER
        // never run outBuf.Dispose (the activation cache owns it, disposing
        // would double-free). infoBuf is sync-downloaded and `using`-scoped.
        using var inBuf = GetOrAllocateBuffer(backend, input);
        using var infoBuf = AllocateInt32Buffer(backend, batch);
        var outBuf = AllocateOutputBuffer(backend, batch * n * n);
        float[] factorArr;
        try
        {
            lin.LinalgCholesky(inBuf.Buffer, outBuf.Buffer, infoBuf.Buffer, batch, n, upper);
            factorArr = FinishGpuOp<float>(backend, outBuf, batch * n * n);
            // Ownership of outBuf is now in the activation/deferred-download
            // caches — do NOT Dispose it below.
        }
        catch
        {
            outBuf.Dispose();
            return (null, null);
        }
        try
        {
            var infoArr = DownloadInt32(backend, infoBuf.Buffer, batch);
            var factor = new Tensor<float>(factorArr, (int[])input._shape.Clone());
            var infoTensor = BuildInfoTensor(input._shape, infoArr);
            return (factor, infoTensor);
        }
        catch
        {
            return (null, null); // outBuf already owned by caches; don't touch it.
        }
    }

    /// <summary>GPU-accelerated LU factor. See <see cref="TryGpuCholesky"/> for semantics.</summary>
    internal (Tensor<float>? LU, Tensor<int>? Pivots) TryGpuLuFactor(Tensor<float> input)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        if (m == 0 || n == 0) return (null, null); // Empty matrix — fall back to CPU.
        if (Math.Max(m, n) > GpuLinalgMaxN) return (null, null);
        int k = Math.Min(m, n);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        if (batch == 0) return (null, null);

        using var inBuf = GetOrAllocateBuffer(backend, input);
        using var pivBuf = AllocateInt32Buffer(backend, batch * k);
        var outBuf = AllocateOutputBuffer(backend, batch * m * n);
        float[] luArr;
        try
        {
            lin.LinalgLuFactor(inBuf.Buffer, outBuf.Buffer, pivBuf.Buffer, batch, m, n);
            luArr = FinishGpuOp<float>(backend, outBuf, batch * m * n);
            // Ownership transferred — do not dispose outBuf below.
        }
        catch
        {
            outBuf.Dispose();
            return (null, null);
        }
        try
        {
            var pivArr = DownloadInt32(backend, pivBuf.Buffer, batch * k);
            var lu = new Tensor<float>(luArr, (int[])input._shape.Clone());
            var pivShape = new int[rank - 1];
            for (int i = 0; i < rank - 2; i++) pivShape[i] = input._shape[i];
            pivShape[rank - 2] = k;
            var pivots = new Tensor<int>(pivShape);
            Array.Copy(pivArr, pivots.GetDataArray(), pivArr.Length);
            return (lu, pivots);
        }
        catch
        {
            return (null, null); // outBuf owned by caches; don't touch it.
        }
    }

    /// <summary>GPU-accelerated reduced QR via Modified Gram–Schmidt.</summary>
    internal (Tensor<float>? Q, Tensor<float>? R) TryGpuQrReduced(
        Tensor<float> input, bool directOnly = false)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        if (m == 0 || n == 0) return (null, null); // Empty matrix — fall back to CPU.
        if (Math.Max(m, n) > GpuLinalgMaxN) return (null, null);
        int k = Math.Min(m, n);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        if (batch == 0) return (null, null);

        using var inBuf = GetOrAllocateBuffer(backend, input);
        var qBuf = AllocateOutputBuffer(backend, batch * m * k);
        var rBuf = AllocateOutputBuffer(backend, batch * k * n);

        // Pre-handoff phase: both qBuf and rBuf are owned by us, dispose on throw.
        float[] qArr;
        try
        {
            if (directOnly)
            {
                if (backend is not IExtendedLinalgBackend extended)
                    throw new NotSupportedException("The active backend has no direct-PTX QR route.");
                extended.LinalgQrReducedDirect(inBuf.Buffer, qBuf.Buffer, rBuf.Buffer, batch, m, n);
            }
            else lin.LinalgQrReduced(inBuf.Buffer, qBuf.Buffer, rBuf.Buffer, batch, m, n);
            qArr = FinishGpuOp<float>(backend, qBuf, batch * m * k);
            // qBuf ownership now in caches. rBuf still owned by us.
        }
        catch
        {
            qBuf.Dispose();
            rBuf.Dispose();
            return (null, null);
        }

        // Partial-handoff phase: qBuf is cache-owned, rBuf is ours.
        float[] rArr;
        try
        {
            rArr = FinishGpuOp<float>(backend, rBuf, batch * k * n);
            // Now both qBuf and rBuf ownership lives in the caches.
        }
        catch
        {
            rBuf.Dispose(); // qBuf is already in the cache — don't touch it.
            return (null, null);
        }

        // Post-handoff phase: neither qBuf nor rBuf belongs to us anymore.
        try
        {
            var qShape = (int[])input._shape.Clone();
            qShape[rank - 1] = k;
            var rShape = (int[])input._shape.Clone();
            rShape[rank - 2] = k;
            return (new Tensor<float>(qArr, qShape), new Tensor<float>(rArr, rShape));
        }
        catch
        {
            return (null, null);
        }
    }

    // NOTE on buffer lifetimes in this file: each "output" buffer passed to
    // FinishGpuOp has its IGpuBuffer reference registered in the activation
    // cache and the deferred-download map — ownership transfers, so the
    // OwnedBuffer struct going out of scope without Dispose() is intentional
    // (calling Dispose on success would double-free the GPU handle). Buffers
    // that we download synchronously (info/pivots) use `using` and are
    // released right after the download.

    /// <summary>GPU-accelerated symmetric eigendecomposition.</summary>
    internal (Tensor<float>? Eigenvalues, Tensor<float>? Eigenvectors) TryGpuEigh(
        Tensor<float> input, bool upper)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) return (null, null);
        if (n == 0) return (null, null); // Empty matrix — fall back to CPU.
        // Per-backend cap: WebGPU's shader ships with 32×32 workgroup arrays,
        // everyone else provisions 64×64. Above the backend's cap we fall
        // back to the CPU Jacobi path instead of overrunning workgroup memory.
        if (n > BackendEighMaxN(backend)) return (null, null);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];
        if (batch == 0) return (null, null);

        using var inBuf = GetOrAllocateBuffer(backend, input);
        var wBuf = AllocateOutputBuffer(backend, batch * n);
        var vBuf = AllocateOutputBuffer(backend, batch * n * n);

        // Pre-handoff: both buffers owned by us.
        float[] wArr;
        try
        {
            if (backend is IExtendedLinalgBackend extended)
                extended.LinalgEighSymmetric(inBuf.Buffer, wBuf.Buffer, vBuf.Buffer, batch, n, upper);
            else if (upper)
                lin.LinalgEigh(inBuf.Buffer, wBuf.Buffer, vBuf.Buffer, batch, n);
            else
                throw new NotSupportedException("The established GPU Eigh backend is upper-triangle only.");
            wArr = FinishGpuOp<float>(backend, wBuf, batch * n);
            // wBuf ownership handed off; vBuf still ours.
        }
        catch
        {
            wBuf.Dispose();
            vBuf.Dispose();
            return (null, null);
        }

        float[] vArr;
        try
        {
            vArr = FinishGpuOp<float>(backend, vBuf, batch * n * n);
            // Both buffers now cache-owned.
        }
        catch
        {
            vBuf.Dispose(); // wBuf is cache-owned — don't touch.
            return (null, null);
        }

        try
        {
            var wShape = new int[rank - 1];
            for (int i = 0; i < rank - 2; i++) wShape[i] = input._shape[i];
            wShape[rank - 2] = n;
            return (new Tensor<float>(wArr, wShape), new Tensor<float>(vArr, (int[])input._shape.Clone()));
        }
        catch
        {
            return (null, null);
        }
    }

    /// <summary>GPU exact-shape reduced SVD through the optional extended solver backend.</summary>
    internal (Tensor<float>? U, Tensor<float>? S, Tensor<float>? Vh) TryGpuSvdReduced(
        Tensor<float> input,
        bool fullMatrices)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return (null, null, null);
        int rank = input.Rank;
        if (rank < 2) return (null, null, null);
        int m = input.Shape[rank - 2], n = input.Shape[rank - 1];
        // The first direct specialization is square 4x4; reduced and complete
        // SVD have identical output extents in this regime.
        if (m != 4 || n != 4) return (null, null, null);
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch = checked(batch * input._shape[i]);
        if (batch == 0) return (null, null, null);

        using var inBuf = GetOrAllocateBuffer(backend, input);
        var uBuf = AllocateOutputBuffer(backend, batch * 16);
        var sBuf = AllocateOutputBuffer(backend, batch * 4);
        var vhBuf = AllocateOutputBuffer(backend, batch * 16);
        float[] u;
        try
        {
            lin.LinalgSvdReduced(inBuf.Buffer, uBuf.Buffer, sBuf.Buffer, vhBuf.Buffer, batch, m, n);
            u = FinishGpuOp<float>(backend, uBuf, batch * 16);
        }
        catch
        {
            uBuf.Dispose(); sBuf.Dispose(); vhBuf.Dispose();
            return (null, null, null);
        }
        float[] s;
        try { s = FinishGpuOp<float>(backend, sBuf, batch * 4); }
        catch { sBuf.Dispose(); vhBuf.Dispose(); return (null, null, null); }
        float[] vh;
        try { vh = FinishGpuOp<float>(backend, vhBuf, batch * 16); }
        catch { vhBuf.Dispose(); return (null, null, null); }

        try
        {
            int[] matrixShape = (int[])input._shape.Clone();
            int[] singularShape = new int[rank - 1];
            for (int i = 0; i < rank - 2; i++) singularShape[i] = input._shape[i];
            singularShape[rank - 2] = 4;
            return (new Tensor<float>(u, matrixShape), new Tensor<float>(s, singularShape),
                new Tensor<float>(vh, matrixShape));
        }
        catch { return (null, null, null); }
    }

    /// <summary>GPU exact-shape LU solve for one vector RHS per 4x4 factor.</summary>
    internal Tensor<float>? TryGpuLuSolve(
        Tensor<float> lu,
        Tensor<int> pivots,
        Tensor<float> rhs)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return null;
        int rank = lu.Rank;
        if (rank < 2 || lu.Shape[rank - 2] != 4 || lu.Shape[rank - 1] != 4)
            return null;
        bool vector = rhs.Rank == rank - 1;
        if (!vector || rhs.Shape[rhs.Rank - 1] != 4) return null;
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch = checked(batch * lu._shape[i]);
        if (batch == 0 || pivots.Length != batch * 4 || rhs.Length != batch * 4) return null;

        using var luBuf = GetOrAllocateBuffer(backend, lu);
        using var rhsBuf = GetOrAllocateBuffer(backend, rhs);
        float[] pivotBits = new float[pivots.Length];
        Buffer.BlockCopy(pivots.GetDataArray(), 0, pivotBits, 0, pivots.Length * sizeof(int));
        using var pivBuf = new OwnedBuffer(backend.AllocateBuffer(pivotBits), ownsBuffer: true);
        var outBuf = AllocateOutputBuffer(backend, rhs.Length);
        float[] result;
        try
        {
            lin.LinalgLuSolve(luBuf.Buffer, pivBuf.Buffer, rhsBuf.Buffer, outBuf.Buffer,
                batch, 4, 1, rhsIsVector: true);
            result = FinishGpuOp<float>(backend, outBuf, rhs.Length);
        }
        catch
        {
            outBuf.Dispose();
            return null;
        }

        // FinishGpuOp transfers the output buffer to the backend cache. Keep
        // tensor materialization outside the buffer-owned failure region so a
        // constructor failure cannot return the same buffer twice.
        try { return new Tensor<float>(result, (int[])rhs._shape.Clone()); }
        catch { return null; }
    }

    // ── Small helpers specific to linalg buffer plumbing ────────────────────

    /// <summary>Exact-shape lower LDL factorization through direct PTX.</summary>
    internal (Tensor<float>? LD, Tensor<int>? Pivots) TryGpuLdlFactor(
        Tensor<float> input, bool upper)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return (null, null);
        int rank = input.Rank;
        if (rank < 2 || input.Shape[rank - 2] != 4 || input.Shape[rank - 1] != 4)
            return (null, null);
        int batch = Batch4x4(input);
        if (batch == 0) return (null, null);
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var pivotBuffer = AllocateInt32Buffer(backend, batch * 4);
        var outputBuffer = AllocateOutputBuffer(backend, batch * 16);
        float[] ld;
        try
        {
            lin.LinalgLdlFactor(inputBuffer.Buffer, outputBuffer.Buffer, pivotBuffer.Buffer,
                batch, 4, upper);
            ld = FinishGpuOp<float>(backend, outputBuffer, batch * 16);
        }
        catch
        {
            outputBuffer.Dispose();
            return (null, null);
        }
        try
        {
            int[] pivots = DownloadInt32(backend, pivotBuffer.Buffer, batch * 4);
            int[] pivotShape = new int[rank - 1];
            for (int i = 0; i < rank - 2; i++) pivotShape[i] = input._shape[i];
            pivotShape[rank - 2] = 4;
            return (new Tensor<float>(ld, (int[])input._shape.Clone()),
                new Tensor<int>(pivots, pivotShape));
        }
        catch { return (null, null); }
    }

    /// <summary>Exact-shape lower LDL vector solve through direct PTX.</summary>
    internal Tensor<float>? TryGpuLdlSolve(
        Tensor<float> ld, Tensor<int> pivots, Tensor<float> rhs, bool upper)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return null;
        int rank = ld.Rank;
        int batch = Batch4x4(ld);
        if (batch == 0 || rhs.Rank != rank - 1 || rhs.Shape[rhs.Rank - 1] != 4 ||
            pivots.Length != batch * 4 || rhs.Length != batch * 4)
            return null;
        using var ldBuffer = GetOrAllocateBuffer(backend, ld);
        using var rhsBuffer = GetOrAllocateBuffer(backend, rhs);
        using var pivotBuffer = UploadInt32(backend, pivots);
        var outputBuffer = AllocateOutputBuffer(backend, batch * 4);
        float[] output;
        try
        {
            lin.LinalgLdlSolve(ldBuffer.Buffer, pivotBuffer.Buffer, rhsBuffer.Buffer,
                outputBuffer.Buffer, batch, 4, upper, rhsIsVector: true);
            output = FinishGpuOp<float>(backend, outputBuffer, batch * 4);
        }
        catch
        {
            outputBuffer.Dispose();
            return null;
        }
        try { return new Tensor<float>(output, (int[])rhs._shape.Clone()); }
        catch { return null; }
    }

    /// <summary>Fused exact-shape general vector solve, optionally returning info.</summary>
    internal (Tensor<float>? Solution, Tensor<int>? Info) TryGpuSolveVector(
        Tensor<float> input, Tensor<float> rhs)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return (null, null);
        int rank = input.Rank;
        int batch = Batch4x4(input);
        if (batch == 0 || rhs.Rank != rank - 1 || rhs.Shape[rhs.Rank - 1] != 4 ||
            rhs.Length != batch * 4)
            return (null, null);
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var rhsBuffer = GetOrAllocateBuffer(backend, rhs);
        using var infoBuffer = AllocateInt32Buffer(backend, batch);
        var outputBuffer = AllocateOutputBuffer(backend, batch * 4);
        float[] output;
        try
        {
            lin.LinalgSolveVector(inputBuffer.Buffer, rhsBuffer.Buffer, outputBuffer.Buffer,
                infoBuffer.Buffer, batch, 4);
            output = FinishGpuOp<float>(backend, outputBuffer, batch * 4);
        }
        catch
        {
            outputBuffer.Dispose();
            return (null, null);
        }
        try
        {
            return (new Tensor<float>(output, (int[])rhs._shape.Clone()),
                BuildInfoTensor(input._shape, DownloadInt32(backend, infoBuffer.Buffer, batch)));
        }
        catch { return (null, null); }
    }

    /// <summary>Exact-shape non-unit triangular vector solve through direct PTX.</summary>
    internal Tensor<float>? TryGpuTriangularSolveVector(
        Tensor<float> input, Tensor<float> rhs, bool upper, bool unitDiagonal)
    {
        if (!TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return null;
        int rank = input.Rank;
        int batch = Batch4x4(input);
        if (batch == 0 || rhs.Rank != rank - 1 || rhs.Shape[rhs.Rank - 1] != 4 ||
            rhs.Length != batch * 4)
            return null;
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var rhsBuffer = GetOrAllocateBuffer(backend, rhs);
        var outputBuffer = AllocateOutputBuffer(backend, batch * 4);
        float[] output;
        try
        {
            lin.LinalgTriangularSolveVector(inputBuffer.Buffer, rhsBuffer.Buffer,
                outputBuffer.Buffer, batch, 4, upper, unitDiagonal);
            output = FinishGpuOp<float>(backend, outputBuffer, batch * 4);
        }
        catch
        {
            outputBuffer.Dispose();
            return null;
        }
        try { return new Tensor<float>(output, (int[])rhs._shape.Clone()); }
        catch { return null; }
    }

    /// <summary>Fused lower-Cholesky gradient specialization.</summary>
    internal Tensor<float>? TryGpuCholeskyBackwardLower(
        Tensor<float> factor, Tensor<float> gradOutput)
    {
        if (!SameExactMatrix4x4Shape(factor, gradOutput, out int batch) ||
            !TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return null;
        using var factorBuffer = GetOrAllocateBuffer(backend, factor);
        using var gradBuffer = GetOrAllocateBuffer(backend, gradOutput);
        var outputBuffer = AllocateOutputBuffer(backend, batch * 16);
        float[] output;
        try
        {
            lin.LinalgCholeskyBackwardLower(factorBuffer.Buffer, gradBuffer.Buffer,
                outputBuffer.Buffer, batch, 4);
            output = FinishGpuOp<float>(backend, outputBuffer, batch * 16);
        }
        catch
        {
            outputBuffer.Dispose();
            return null;
        }
        try { return new Tensor<float>(output, (int[])factor._shape.Clone()); }
        catch { return null; }
    }

    /// <summary>Fused exact-shape vector-solve gradient specialization.</summary>
    internal (Tensor<float>? GradInput, Tensor<float>? GradRhs) TryGpuSolveBackwardVector(
        Tensor<float> input, Tensor<float> solution, Tensor<float> gradOutput)
    {
        int rank = input.Rank;
        int batch = Batch4x4(input);
        if (batch == 0 || solution.Rank != rank - 1 || gradOutput.Rank != rank - 1 ||
            solution.Length != batch * 4 || gradOutput.Length != batch * 4 ||
            !TryGetBackend(out var backend) || backend is not IExtendedLinalgBackend lin)
            return (null, null);
        using var inputBuffer = GetOrAllocateBuffer(backend, input);
        using var solutionBuffer = GetOrAllocateBuffer(backend, solution);
        using var gradBuffer = GetOrAllocateBuffer(backend, gradOutput);
        var gradInputBuffer = AllocateOutputBuffer(backend, batch * 16);
        var gradRhsBuffer = AllocateOutputBuffer(backend, batch * 4);
        float[] gradInput;
        try
        {
            lin.LinalgSolveBackwardVector(inputBuffer.Buffer, solutionBuffer.Buffer,
                gradBuffer.Buffer, gradInputBuffer.Buffer, gradRhsBuffer.Buffer, batch, 4);
            gradInput = FinishGpuOp<float>(backend, gradInputBuffer, batch * 16);
        }
        catch
        {
            gradInputBuffer.Dispose(); gradRhsBuffer.Dispose();
            return (null, null);
        }
        float[] gradRhs;
        try { gradRhs = FinishGpuOp<float>(backend, gradRhsBuffer, batch * 4); }
        catch { gradRhsBuffer.Dispose(); return (null, null); }
        try
        {
            return (new Tensor<float>(gradInput, (int[])input._shape.Clone()),
                new Tensor<float>(gradRhs, (int[])solution._shape.Clone()));
        }
        catch { return (null, null); }
    }

    private static int Batch4x4(Tensor<float> input)
    {
        if (input.Rank < 2 || input.Shape[input.Rank - 2] != 4 || input.Shape[input.Rank - 1] != 4)
            return 0;
        int batch = 1;
        for (int i = 0; i < input.Rank - 2; i++) batch = checked(batch * input._shape[i]);
        return batch;
    }

    private static bool SameExactMatrix4x4Shape(
        Tensor<float> left, Tensor<float> right, out int batch)
    {
        batch = Batch4x4(left);
        if (batch == 0 || left.Rank != right.Rank || left.Length != right.Length) return false;
        for (int i = 0; i < left.Rank; i++) if (left.Shape[i] != right.Shape[i]) return false;
        return true;
    }

    private static OwnedBuffer UploadInt32(IDirectGpuBackend backend, Tensor<int> values)
    {
        float[] bits = new float[values.Length];
        Buffer.BlockCopy(values.GetDataArray(), 0, bits, 0, values.Length * sizeof(int));
        return new OwnedBuffer(backend.AllocateBuffer(bits), ownsBuffer: true);
    }

    private static OwnedBuffer AllocateInt32Buffer(IDirectGpuBackend backend, int count)
    {
        // Int32 buffers use 4 bytes per element — same as float, so we reuse
        // AllocateBuffer with a float[]-sized allocation and interpret the
        // handle as int32 on the device side. This avoids an int-specific
        // allocator on every backend.
        return new OwnedBuffer(backend.AllocateBuffer(new float[Math.Max(1, count)]), ownsBuffer: true);
    }

    private static int[] DownloadInt32(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        if (count <= 0) return Array.Empty<int>();
        // Read as float[] and bit-cast to int32. Guaranteed safe since we
        // allocated the buffer as 4-byte slots and kernels wrote int32.
        var asFloat = backend.DownloadBuffer(buffer);
        var result = new int[count];
        Buffer.BlockCopy(asFloat, 0, result, 0, count * sizeof(int));
        return result;
    }

    private static Tensor<int> BuildInfoTensor(int[] inputShape, int[] infoArr)
    {
        int rank = inputShape.Length;
        var shape = rank > 2 ? new int[rank - 2] : new[] { 1 };
        if (rank > 2) for (int i = 0; i < rank - 2; i++) shape[i] = inputShape[i];
        var t = new Tensor<int>(shape);
        Array.Copy(infoArr, t.GetDataArray(), Math.Min(infoArr.Length, t.Length));
        return t;
    }
}
