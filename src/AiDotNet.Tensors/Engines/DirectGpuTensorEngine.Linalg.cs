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
        if (n > GpuLinalgMaxN) return (null, null);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        // outBuf ownership transfers into the activation cache via FinishGpuOp
        // on the success path (so it lives past this scope). On failure we must
        // dispose it ourselves. infoBuf is downloaded synchronously and always
        // disposed here — no deferred lifetime to preserve.
        using var inBuf = GetOrAllocateBuffer(backend, input);
        using var infoBuf = AllocateInt32Buffer(backend, batch);
        var outBuf = AllocateOutputBuffer(backend, batch * n * n);
        try
        {
            lin.LinalgCholesky(inBuf.Buffer, outBuf.Buffer, infoBuf.Buffer, batch, n, upper);
            var factorArr = FinishGpuOp<float>(backend, outBuf, batch * n * n);
            var infoArr = DownloadInt32(backend, infoBuf.Buffer, batch);
            var factor = new Tensor<float>(factorArr, (int[])input._shape.Clone());
            var infoTensor = BuildInfoTensor(input._shape, infoArr);
            return (factor, infoTensor);
        }
        catch
        {
            outBuf.Dispose();
            return (null, null);
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
        if (Math.Max(m, n) > GpuLinalgMaxN) return (null, null);
        int k = Math.Min(m, n);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        // outBuf lifetime transfers into the activation cache on success.
        using var inBuf = GetOrAllocateBuffer(backend, input);
        using var pivBuf = AllocateInt32Buffer(backend, batch * k);
        var outBuf = AllocateOutputBuffer(backend, batch * m * n);
        try
        {
            lin.LinalgLuFactor(inBuf.Buffer, outBuf.Buffer, pivBuf.Buffer, batch, m, n);
            var luArr = FinishGpuOp<float>(backend, outBuf, batch * m * n);
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
            outBuf.Dispose();
            return (null, null);
        }
    }

    /// <summary>GPU-accelerated reduced QR.</summary>
    internal (Tensor<float>? Q, Tensor<float>? R) TryGpuQrReduced(Tensor<float> input)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int m = input.Shape[rank - 2];
        int n = input.Shape[rank - 1];
        if (Math.Max(m, n) > GpuLinalgMaxN) return (null, null);
        int k = Math.Min(m, n);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        // qBuf/rBuf lifetime transfers into the activation cache on success.
        using var inBuf = GetOrAllocateBuffer(backend, input);
        var qBuf = AllocateOutputBuffer(backend, batch * m * k);
        var rBuf = AllocateOutputBuffer(backend, batch * k * n);
        try
        {
            lin.LinalgQrReduced(inBuf.Buffer, qBuf.Buffer, rBuf.Buffer, batch, m, n);
            var qArr = FinishGpuOp<float>(backend, qBuf, batch * m * k);
            var rArr = FinishGpuOp<float>(backend, rBuf, batch * k * n);
            var qShape = (int[])input._shape.Clone();
            qShape[rank - 1] = k;
            var rShape = (int[])input._shape.Clone();
            rShape[rank - 2] = k;
            return (new Tensor<float>(qArr, qShape), new Tensor<float>(rArr, rShape));
        }
        catch
        {
            qBuf.Dispose();
            rBuf.Dispose();
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
    internal (Tensor<float>? Eigenvalues, Tensor<float>? Eigenvectors) TryGpuEigh(Tensor<float> input)
    {
        if (!TryGetBackend(out var backend)) return (null, null);
        if (backend is not ILinalgBackend lin) return (null, null);
        if (!BackendLinalgReady(backend)) return (null, null);

        int rank = input.Rank;
        if (rank < 2) return (null, null);
        int n = input.Shape[rank - 1];
        if (input.Shape[rank - 2] != n) return (null, null);
        // Eigh's shared-memory budget is tighter (2·n² floats).
        if (n > 64) return (null, null);

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input._shape[i];

        using var inBuf = GetOrAllocateBuffer(backend, input);
        var wBuf = AllocateOutputBuffer(backend, batch * n);
        var vBuf = AllocateOutputBuffer(backend, batch * n * n);
        try
        {
            lin.LinalgEigh(inBuf.Buffer, wBuf.Buffer, vBuf.Buffer, batch, n);
            var wArr = FinishGpuOp<float>(backend, wBuf, batch * n);
            var vArr = FinishGpuOp<float>(backend, vBuf, batch * n * n);
            var wShape = new int[rank - 1];
            for (int i = 0; i < rank - 2; i++) wShape[i] = input._shape[i];
            wShape[rank - 2] = n;
            return (new Tensor<float>(wArr, wShape), new Tensor<float>(vArr, (int[])input._shape.Clone()));
        }
        catch
        {
            wBuf.Dispose();
            vBuf.Dispose();
            return (null, null);
        }
    }

    // ── Small helpers specific to linalg buffer plumbing ────────────────────

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
