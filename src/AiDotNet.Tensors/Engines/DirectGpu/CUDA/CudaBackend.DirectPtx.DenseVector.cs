using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxDenseVectorKey, PtxDenseVectorKernel>
        _directPtxDenseVectorKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxBatchedVectorKey, PtxBatchedVectorKernel>
        _directPtxBatchedVectorKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxStridedDotKey, PtxStridedDotKernel>
        _directPtxStridedDotKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly record struct DirectPtxDenseVectorKey(int Operation, int M, int N);
    private readonly record struct DirectPtxBatchedVectorKey(int Operation, int Batch, int M, int N);
    private readonly record struct DirectPtxStridedDotKey(
        int ASize, int BSize, int BOffset, int BStep);

    internal bool TryDirectPtxOuterProduct(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int m,
        int n) =>
        TryDirectPtxDenseVector(
            left, right, output, DirectPtxDenseVectorOperation.Outer, m, n);

    internal bool TryDirectPtxDotProduct(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int length) =>
        TryDirectPtxDenseVector(
            left, right, output, DirectPtxDenseVectorOperation.Dot, length, 1);

    internal bool TryDirectPtxBatchDotProduct(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int batch,
        int dimension) =>
        TryDirectPtxBatchedVector(
            left, right, output, DirectPtxBatchedVectorOperation.Dot,
            batch, dimension, 1);

    internal bool TryDirectPtxBatchOuterProduct(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int batch,
        int m,
        int n) =>
        TryDirectPtxBatchedVector(
            left, right, output, DirectPtxBatchedVectorOperation.Outer,
            batch, m, n);

    internal bool TryDirectPtxStridedDotProduct(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxStridedDotSemantics(aSize, bSize))
            return false;
        if (DirectPtxBufferIsInvalid(left) || DirectPtxBufferIsInvalid(right) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "strided-dot-null-or-invalid-buffer";
            return false;
        }
        if (left.SizeInBytes != checked((long)aSize * sizeof(float)) ||
            right.SizeInBytes != checked((long)bSize * sizeof(float)) ||
            output.SizeInBytes != sizeof(float))
        {
            DirectPtxLastError = "strided-dot-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, left) ||
            DirectPtxBuffersOverlap(output, right))
        {
            DirectPtxLastError = "strided-dot-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxStridedDotKey(aSize, bSize, bOffset, bStep);
            lock (_directPtxLock)
            {
                if (!_directPtxStridedDotKernels.TryGetValue(
                    key, out PtxStridedDotKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX strided-dot kernel must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxStridedDotKernelSlow(key);
                }
                if (capturing && !_directPtxStridedDotKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX strided-dot module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(left, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(right, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool TryDirectPtxDenseVector(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        DirectPtxDenseVectorOperation operation,
        int m,
        int n)
    {
        // No dense-vector specialization is performance-qualified yet. Keep
        // the implementation reachable only from the explicit benchmark/test
        // scope until its exact shape wins the production gate.
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxDenseVectorSemantics(operation, m, n))
            return false;
        if (DirectPtxBufferIsInvalid(left) || DirectPtxBufferIsInvalid(right) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "dense-vector-null-or-invalid-buffer";
            return false;
        }
        long leftBytes = checked((long)m * sizeof(float));
        long rightBytes = checked((long)(operation == DirectPtxDenseVectorOperation.Dot ? m : n) * sizeof(float));
        long outputBytes = checked((long)(operation == DirectPtxDenseVectorOperation.Dot ? 1 : checked(m * n)) * sizeof(float));
        if (left.SizeInBytes != leftBytes || right.SizeInBytes != rightBytes ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "dense-vector-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, left) ||
            DirectPtxBuffersOverlap(output, right))
        {
            DirectPtxLastError = "dense-vector-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxDenseVectorKey((int)operation, m, n);
            lock (_directPtxLock)
            {
                if (!_directPtxDenseVectorKernels.TryGetValue(
                    key, out PtxDenseVectorKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX dense-vector kernel must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxDenseVectorKernelSlow(key);
                }
                if (capturing && !_directPtxDenseVectorKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX dense-vector module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(left, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(right, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool TryDirectPtxBatchedVector(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxBatchedVectorSemantics(operation, batch, m, n))
            return false;
        if (DirectPtxBufferIsInvalid(left) || DirectPtxBufferIsInvalid(right) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "batched-vector-null-or-invalid-buffer";
            return false;
        }
        long leftBytes = checked((long)batch * m * sizeof(float));
        long rightBytes = checked((long)batch *
            (operation == DirectPtxBatchedVectorOperation.Dot ? m : n) * sizeof(float));
        long outputBytes = checked((long)(operation == DirectPtxBatchedVectorOperation.Dot
            ? batch : checked(batch * checked(m * n))) * sizeof(float));
        if (left.SizeInBytes != leftBytes || right.SizeInBytes != rightBytes ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "batched-vector-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, left) ||
            DirectPtxBuffersOverlap(output, right))
        {
            DirectPtxLastError = "batched-vector-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxBatchedVectorKey((int)operation, batch, m, n);
            lock (_directPtxLock)
            {
                if (!_directPtxBatchedVectorKernels.TryGetValue(
                    key, out PtxBatchedVectorKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX batched-vector kernel must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxBatchedVectorKernelSlow(key);
                }
                if (capturing && !_directPtxBatchedVectorKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX batched-vector module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(left, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(right, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxDenseVector(
        DirectPtxDenseVectorOperation operation,
        int m,
        int n = 1)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxDenseVectorSemantics(operation, m, n))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX dense-vector prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxDenseVectorKey((int)operation, m, n);
                if (!_directPtxDenseVectorKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxDenseVectorKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxBatchedVector(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n = 1)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxBatchedVectorSemantics(operation, batch, m, n))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX batched-vector prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxBatchedVectorKey((int)operation, batch, m, n);
                if (!_directPtxBatchedVectorKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxBatchedVectorKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxStridedDot(
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxStridedDotSemantics(aSize, bSize))
            return false;
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "Direct PTX strided-dot prewarm is not capture-safe.";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxStridedDotKey(aSize, bSize, bOffset, bStep);
                if (!_directPtxStridedDotKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxStridedDotKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxDenseVectorKernel CreateAndCacheDirectPtxDenseVectorKernelSlow(
        DirectPtxDenseVectorKey key) =>
        _directPtxDenseVectorKernels.GetOrAdd(
            key, () => new PtxDenseVectorKernel(
                _directPtxRuntime!, (DirectPtxDenseVectorOperation)key.Operation,
                key.M, key.N));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxBatchedVectorKernel CreateAndCacheDirectPtxBatchedVectorKernelSlow(
        DirectPtxBatchedVectorKey key) =>
        _directPtxBatchedVectorKernels.GetOrAdd(
            key, () => new PtxBatchedVectorKernel(
                _directPtxRuntime!, (DirectPtxBatchedVectorOperation)key.Operation,
                key.Batch, key.M, key.N));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxStridedDotKernel CreateAndCacheDirectPtxStridedDotKernelSlow(
        DirectPtxStridedDotKey key) =>
        _directPtxStridedDotKernels.GetOrAdd(
            key, () => new PtxStridedDotKernel(
                _directPtxRuntime!, key.ASize, key.BSize, key.BOffset, key.BStep));

    internal bool TryGetDirectPtxDenseVectorAudit(
        DirectPtxDenseVectorOperation operation,
        int m,
        int n,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxDenseVectorKey((int)operation, m, n);
            if (_directPtxDenseVectorKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxBatchedVectorAudit(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxBatchedVectorKey((int)operation, batch, m, n);
            if (_directPtxBatchedVectorKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxStridedDotAudit(
        int aSize,
        int bSize,
        int bOffset,
        int bStep,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxStridedDotKey(aSize, bSize, bOffset, bStep);
            if (_directPtxStridedDotKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool ValidateDirectPtxDenseVectorSemantics(
        DirectPtxDenseVectorOperation operation,
        int m,
        int n)
    {
        if (operation != DirectPtxDenseVectorOperation.Dot &&
            operation != DirectPtxDenseVectorOperation.Outer)
        {
            DirectPtxLastError = "dense-vector-operation-not-implemented";
            return false;
        }
        if (m <= 0 || m > 1_048_576 ||
            (operation == DirectPtxDenseVectorOperation.Dot && n != 1) ||
            (operation == DirectPtxDenseVectorOperation.Outer &&
             (n <= 0 || n > 65_536 || (long)m * n > int.MaxValue)))
        {
            DirectPtxLastError = "dense-vector-shape-not-implemented";
            return false;
        }
        return true;
    }

    private bool ValidateDirectPtxBatchedVectorSemantics(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n)
    {
        if (operation != DirectPtxBatchedVectorOperation.Dot &&
            operation != DirectPtxBatchedVectorOperation.Outer)
        {
            DirectPtxLastError = "batched-vector-operation-not-implemented";
            return false;
        }
        long perBatch = operation == DirectPtxBatchedVectorOperation.Dot
            ? m : (long)m * n;
        if (batch <= 0 || batch > 65_535 || m <= 0 || m > 1_048_576 ||
            (operation == DirectPtxBatchedVectorOperation.Dot && n != 1) ||
            (operation == DirectPtxBatchedVectorOperation.Outer &&
             (n <= 0 || n > 65_536)) ||
            perBatch <= 0 || perBatch * batch > int.MaxValue)
        {
            DirectPtxLastError = "batched-vector-shape-not-implemented";
            return false;
        }
        return true;
    }

    private bool ValidateDirectPtxStridedDotSemantics(int aSize, int bSize)
    {
        if (aSize <= 0 || aSize > 1_048_576 ||
            bSize <= 0 || bSize > 1_048_576)
        {
            DirectPtxLastError = "strided-dot-shape-not-implemented";
            return false;
        }
        return true;
    }
}
