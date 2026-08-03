using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxFp16GemmKey, PtxFp16GemmKernel>
        _directPtxFp16GemmKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity));

    private readonly record struct DirectPtxFp16GemmKey(
        int M, int N, int K, int Batch,
        bool TransposeA, bool TransposeB,
        int InputType, int OutputType, bool HalfAccumulate);

    internal bool TryDirectPtxFp16Gemm(
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer output,
        int m,
        int n,
        int k,
        int batch = 1,
        bool transposeA = false,
        bool transposeB = false,
        DirectPtx16BitInputType inputType = DirectPtx16BitInputType.Float16,
        DirectPtxGemmOutputType outputType = DirectPtxGemmOutputType.Float32,
        bool halfAccumulate = false)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxFp16GemmSemantics(
            m, n, k, batch, inputType, outputType, halfAccumulate))
            return false;
        if (DirectPtxBufferIsInvalid(left) || DirectPtxBufferIsInvalid(right) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "16-bit-gemm-null-or-invalid-buffer";
            return false;
        }

        long leftBytes = checked((long)batch * m * k * sizeof(ushort));
        long rightBytes = checked((long)batch * k * n * sizeof(ushort));
        int outputElementBytes = outputType == DirectPtxGemmOutputType.Float32
            ? sizeof(float) : sizeof(ushort);
        long outputBytes = checked((long)batch * m * n * outputElementBytes);
        if (left.SizeInBytes != leftBytes || right.SizeInBytes != rightBytes ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "16-bit-gemm-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, left) ||
            DirectPtxBuffersOverlap(output, right))
        {
            DirectPtxLastError = "16-bit-gemm-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFp16GemmKey(
                m, n, k, batch, transposeA, transposeB,
                (int)inputType, (int)outputType, halfAccumulate);
            lock (_directPtxLock)
            {
                PtxFp16GemmKernel kernel = GetOrCreateDirectPtxFp16GemmKernel(
                    key, capturing, inputType, outputType);
                if (capturing && !_directPtxFp16GemmKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX 16-bit GEMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(left, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(right, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (DirectPtxPrewarmRequiredException ex)
        {
            DirectPtxLastError = ex.Message;
            return false;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxFp16Backward(
        IGpuBuffer gradC,
        IGpuBuffer left,
        IGpuBuffer right,
        IGpuBuffer gradLeft,
        IGpuBuffer gradRight,
        int m,
        int n,
        int k,
        bool halfOutput)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxFp16GemmSemantics(
            m, n, k, 1, DirectPtx16BitInputType.Float16,
            halfOutput ? DirectPtxGemmOutputType.Float16 : DirectPtxGemmOutputType.Float32,
            halfAccumulate: false))
            return false;
        if (DirectPtxBufferIsInvalid(gradC) || DirectPtxBufferIsInvalid(left) ||
            DirectPtxBufferIsInvalid(right) || DirectPtxBufferIsInvalid(gradLeft) ||
            DirectPtxBufferIsInvalid(gradRight))
        {
            DirectPtxLastError = "16-bit-backward-null-or-invalid-buffer";
            return false;
        }
        int outputBytes = halfOutput ? sizeof(ushort) : sizeof(float);
        if (gradC.SizeInBytes != checked((long)m * n * sizeof(ushort)) ||
            left.SizeInBytes != checked((long)m * k * sizeof(ushort)) ||
            right.SizeInBytes != checked((long)k * n * sizeof(ushort)) ||
            gradLeft.SizeInBytes != checked((long)m * k * outputBytes) ||
            gradRight.SizeInBytes != checked((long)k * n * outputBytes))
        {
            DirectPtxLastError = "16-bit-backward-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(gradLeft, gradC) ||
            DirectPtxBuffersOverlap(gradLeft, left) ||
            DirectPtxBuffersOverlap(gradLeft, right) ||
            DirectPtxBuffersOverlap(gradRight, gradC) ||
            DirectPtxBuffersOverlap(gradRight, left) ||
            DirectPtxBuffersOverlap(gradRight, right) ||
            DirectPtxBuffersOverlap(gradLeft, gradRight))
        {
            DirectPtxLastError = "16-bit-backward-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            DirectPtxGemmOutputType outputType = halfOutput
                ? DirectPtxGemmOutputType.Float16 : DirectPtxGemmOutputType.Float32;
            var gradLeftKey = new DirectPtxFp16GemmKey(
                m, k, n, 1, false, true,
                (int)DirectPtx16BitInputType.Float16, (int)outputType, false);
            var gradRightKey = new DirectPtxFp16GemmKey(
                k, n, m, 1, true, false,
                (int)DirectPtx16BitInputType.Float16, (int)outputType, false);
            lock (_directPtxLock)
            {
                PtxFp16GemmKernel gradLeftKernel = GetOrCreateDirectPtxFp16GemmKernel(
                    gradLeftKey, capturing, DirectPtx16BitInputType.Float16, outputType);
                PtxFp16GemmKernel gradRightKernel = GetOrCreateDirectPtxFp16GemmKernel(
                    gradRightKey, capturing, DirectPtx16BitInputType.Float16, outputType);
                if (capturing &&
                    (!_directPtxFp16GemmKernels.Pin(gradLeftKey) ||
                     !_directPtxFp16GemmKernels.Pin(gradRightKey)))
                    throw new InvalidOperationException(
                        "Could not pin both direct-PTX 16-bit backward modules for CUDA graph capture.");
                lock (GpuDispatchLock)
                {
                    gradLeftKernel.Launch(
                        DirectPtxTensorView.Create(gradC, gradLeftKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(right, gradLeftKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradLeft, gradLeftKernel.Blueprint.Tensors[2]));
                    gradRightKernel.Launch(
                        DirectPtxTensorView.Create(left, gradRightKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(gradC, gradRightKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradRight, gradRightKernel.Blueprint.Tensors[2]));
                }
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (DirectPtxPrewarmRequiredException ex)
        {
            DirectPtxLastError = ex.Message;
            return false;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxFp16GemmKernel GetOrCreateDirectPtxFp16GemmKernel(
        DirectPtxFp16GemmKey key,
        bool capturing,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType)
    {
        if (_directPtxFp16GemmKernels.TryGetValue(key, out PtxFp16GemmKernel? kernel))
            return kernel;
        if (capturing)
            throw new DirectPtxPrewarmRequiredException(
                "Direct PTX 16-bit GEMM must be prewarmed before CUDA graph capture.");
        _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
        return CreateAndCacheDirectPtxFp16GemmKernelSlow(
            key, inputType, outputType);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFp16GemmKernel CreateAndCacheDirectPtxFp16GemmKernelSlow(
        DirectPtxFp16GemmKey key,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType)
    {
        return _directPtxFp16GemmKernels.GetOrAdd(
            key, () => new PtxFp16GemmKernel(
                _directPtxRuntime!, key.M, key.N, key.K, key.Batch,
                key.TransposeA, key.TransposeB, inputType, outputType,
                key.HalfAccumulate));
    }

    internal bool PrewarmDirectPtxFp16Gemm(
        int m,
        int n,
        int k,
        int batch = 1,
        bool transposeA = false,
        bool transposeB = false,
        DirectPtx16BitInputType inputType = DirectPtx16BitInputType.Float16,
        DirectPtxGemmOutputType outputType = DirectPtxGemmOutputType.Float32,
        bool halfAccumulate = false)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "Direct PTX 16-bit GEMM prewarm is not capture-safe.";
            return false;
        }
        if (!ValidateDirectPtxFp16GemmSemantics(
            m, n, k, batch, inputType, outputType, halfAccumulate))
            return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var key = new DirectPtxFp16GemmKey(
                    m, n, k, batch, transposeA, transposeB,
                    (int)inputType, (int)outputType, halfAccumulate);
                _ = GetOrCreateDirectPtxFp16GemmKernel(
                    key, false, inputType, outputType);
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

    internal bool TryGetDirectPtxFp16GemmAudit(
        int m,
        int n,
        int k,
        int batch,
        bool transposeA,
        bool transposeB,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType,
        bool halfAccumulate,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFp16GemmKey(
                m, n, k, batch, transposeA, transposeB,
                (int)inputType, (int)outputType, halfAccumulate);
            if (_directPtxFp16GemmKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool ValidateDirectPtxFp16GemmSemantics(
        int m,
        int n,
        int k,
        int batch,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType,
        bool halfAccumulate)
    {
        if (m <= 0 || m > 65_536 || n <= 0 || n > 65_536 ||
            k <= 0 || k > 65_536 || batch <= 0 || batch > 65_535)
        {
            DirectPtxLastError = "16-bit-gemm-shape-not-implemented";
            return false;
        }
        if ((inputType != DirectPtx16BitInputType.Float16 &&
             inputType != DirectPtx16BitInputType.BFloat16) ||
            (outputType != DirectPtxGemmOutputType.Float16 &&
             outputType != DirectPtxGemmOutputType.Float32) ||
            (halfAccumulate && (inputType != DirectPtx16BitInputType.Float16 ||
                                outputType != DirectPtxGemmOutputType.Float16)))
        {
            DirectPtxLastError = "16-bit-gemm-semantics-not-implemented";
            return false;
        }
        return true;
    }

    private sealed class DirectPtxPrewarmRequiredException : InvalidOperationException
    {
        internal DirectPtxPrewarmRequiredException(string message) : base(message) { }
    }
}
