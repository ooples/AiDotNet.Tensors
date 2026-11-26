// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The fused linear layers (<c>FusedLinear*</c>, <c>FusedLinear*Backward</c>) under matmul autocast. Their own kernels
/// compute the products in FP32, so under an open scope a layer runs as torch.autocast runs it: the products on the
/// GEMM entry points, which cast their operands to 16 bits and accumulate in FP32, and the bias, the activation and
/// the bias gradient in FP32 on those products.
/// </summary>
public sealed partial class CudaBackend
{
    /// <summary>
    /// Diagnostics: told of every GEMM that runs with its operands cast to 16 bits (the cuBLAS type, the column-major
    /// problem and the batch), on the thread that issues it. Null when nobody listens.
    /// </summary>
    internal System.Action<CastGemmRecord>? CastGemmObserver;

    /// <summary>One GEMM run with its operands cast: cuBLAS's 16-bit type and the column-major problem.</summary>
    internal readonly record struct CastGemmRecord(int CastType, bool TransA, bool TransB, int M, int N, int K, int Batch);

    // The activation codes of FusedLinear*Backward (the switch of the fused_linear_bias_grad kernel).
    private const int LinearReluCode = 0, LinearSigmoidCode = 1, LinearTanhCode = 2, LinearGeluCode = 3, LinearSwishCode = 4;

    /// <summary>
    /// A GEMM operand as the open matmul autocast scope's products read it: rounded (to nearest even) to the scope's
    /// 16-bit type and widened back to FP32, in stream-ordered scratch the caller disposes. For an FP32 kernel that
    /// computes a product itself (the fused linear + cross-entropy head): on the rounded operands its FP32 dot products
    /// are those of a cast GEMM. Null when no scope is open; throws when the device cannot round to the scope's type.
    /// </summary>
    private unsafe IGpuBuffer? RoundAutocastOperand(IGpuBuffer source, int count, string what)
    {
        if (!GemmAutocastActive || count <= 0) return null;
        int state = GemmAutocastState;
        int castType = AutocastCastTypeOf(state);
        bool bf16 = castType != 0
            ? castType == CuBlasNative.CUDA_R_16BF
            : AutocastComputeOf(state) == (int)CublasComputeType.Float32FastBF16;
        var storage = bf16 ? FftElementType.BFloat16 : FftElementType.Float16;
        if (!TryGetCastKernel(storage, out System.IntPtr narrow, out string? why))
            throw new GpuCapabilityUnavailableException(
                $"{what} under matmul autocast cannot round its operands to {storage}: {why}.");
        System.IntPtr widen = RequireGenericFftKernel("fftg_to_f32" + storage.KernelSuffix(), storage);
        using var _ = PushContext();
        using var half = AllocateStreamOrderedScratchBufferUninitialized(HalfWords(count, what));
        var rounded = AllocateStreamOrderedScratchBufferUninitialized(count);
        try
        {
            LaunchCast(narrow, source.Handle, half.Handle, count, _stream);
            System.IntPtr src = half.Handle, dst = rounded.Handle;
            int n = count;
            void** args = stackalloc void*[3];
            args[0] = &src; args[1] = &dst; args[2] = &n;
            LaunchKernel(widen, GridFor(n), DefaultBlockSize, args);
            return rounded;
        }
        catch
        {
            rounded.Dispose();
            throw;
        }
    }

    /// <summary>
    /// <see cref="GemmAsync"/> under matmul autocast: row-major <c>C = alpha·A·B + beta·C</c> on the GEMM entry point,
    /// so it takes the open scope's policy (operands cast, FP32 accumulation and result), issued on
    /// <paramref name="stream"/>. False, with nothing launched, when no scope is open.
    /// </summary>
    private bool TryAutocastGemmOnStream(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha, float beta, IGpuStream stream)
    {
        if (!GemmAutocastActive) return false;
        using var context = PushContext();
        System.IntPtr previous = _gemmFanoutStream.Value;
        // The caller's stream stands in for a fan-out lease: the cast scratch, the casts and the GEMM run in its order.
        _gemmFanoutStream.Value = stream.Handle == _stream ? System.IntPtr.Zero : stream.Handle;
        try
        {
            float a = alpha, b = beta;
            // Row-major C = A·B is column-major Cᵀ = Bᵀ·Aᵀ.
            BlasSgemm(CublasOperation.None, CublasOperation.None, N, M, K, ref a, B.Handle, N, A.Handle, K, ref b,
                C.Handle, N, "GemmAsync");
        }
        finally
        {
            _gemmFanoutStream.Value = previous;
            if (IsCublasAvailable) _ = _cublasHandle;   // the thread's handle back on its own stream
        }
        return true;
    }

    /// <summary>
    /// <see cref="FusedGemmBiasActivationAsync"/> under matmul autocast: the product through
    /// <see cref="TryAutocastGemmOnStream"/> on <paramref name="stream"/>, then the bias and the activation in FP32. As
    /// the FP32 path, nothing blocks the host: on a caller's stream the backend's stream waits for the product on the
    /// device, runs the epilogue, and the caller's stream waits for the epilogue, so work queued on
    /// <paramref name="stream"/> afterwards reads the finished output. False, with nothing launched, when no scope is open.
    /// </summary>
    private bool TryAutocastGemmBiasActivationOnStream(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, IGpuBuffer output,
        int M, int N, int K, FusedActivationType activation, IGpuStream stream)
    {
        if (!GemmAutocastActive) return false;
        if (activation is not (FusedActivationType.None or FusedActivationType.ReLU or FusedActivationType.Sigmoid
            or FusedActivationType.Tanh))
            throw new System.NotSupportedException($"Activation type {activation} not supported for fused GEMM");
        TryAutocastGemmOnStream(A, B, output, M, N, K, 1f, 0f, stream);
        bool callerStream = stream.Handle != _stream;
        if (callerStream) OrderStreamAfter(_stream, stream.Handle);
        int outputs = checked(M * N);
        BiasAdd(output, bias, output, M, N);
        switch (activation)
        {
            case FusedActivationType.ReLU: Relu(output, output, outputs); break;
            case FusedActivationType.Sigmoid: Sigmoid(output, output, outputs); break;
            case FusedActivationType.Tanh: Tanh(output, output, outputs); break;
        }
        if (callerStream) OrderStreamAfter(stream.Handle, _stream);
        return true;
    }

    // Makes <paramref name="consumer"/> wait on the device for everything queued so far on <paramref name="producer"/>:
    // an event recorded on the producer, waited on by the consumer, no host block (capturable: a fork or a join).
    private void OrderStreamAfter(System.IntPtr consumer, System.IntPtr producer)
    {
        if (consumer == producer) return;
        using var _ = PushContext();
        CuBlasNative.CheckCudaResult(
            CudaNativeBindings.cuEventCreate(out System.IntPtr ready, CudaNativeBindings.CU_EVENT_DISABLE_TIMING),
            "cuEventCreate(autocast epilogue order)");
        try
        {
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuEventRecord(ready, producer), "cuEventRecord(autocast epilogue order)");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamWaitEvent(consumer, ready, 0),
                "cuStreamWaitEvent(autocast epilogue order)");
        }
        finally
        {
            // Legal once recorded: the wait binds to the record, not to the event object.
            CudaNativeBindings.cuEventDestroy(ready);
        }
    }

    /// <summary>
    /// <c>output = activation(input · weight + bias)</c> under matmul autocast: input [batch, in], weight [in, out].
    /// False, with nothing launched, when no scope is open.
    /// </summary>
    private bool TryAutocastFusedLinear(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batchSize, int inFeatures, int outFeatures, int activationCode)
    {
        if (!GemmAutocastActive) return false;
        int outputs = checked(batchSize * outFeatures);
        Gemm(input, weight, output, batchSize, outFeatures, inFeatures, 1f, 0f);
        BiasAdd(output, bias, output, batchSize, outFeatures);
        switch (activationCode)
        {
            case LinearReluCode: Relu(output, output, outputs); break;
            case LinearSigmoidCode: Sigmoid(output, output, outputs); break;
            case LinearTanhCode: Tanh(output, output, outputs); break;
            case LinearGeluCode: Gelu(output, output, outputs); break;
            case LinearSwishCode: Swish(output, output, outputs); break;
            default: throw new System.ArgumentOutOfRangeException(nameof(activationCode));
        }
        return true;
    }

    /// <summary>
    /// The gradients of <see cref="TryAutocastFusedLinear"/>. The activation's derivative masks the output gradient
    /// (FP32); the input and weight gradients are products of the mask on the GEMM entry points; the bias gradient is
    /// the mask's FP32 column sum. <paramref name="saved"/> is what the FP32 kernels take: the pre-activation for ReLU,
    /// GELU and Swish, the output for Sigmoid and Tanh. False, with nothing launched, when no scope is open.
    /// </summary>
    private unsafe bool TryAutocastFusedLinearBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures, int activationCode)
    {
        if (!GemmAutocastActive) return false;
        int outputs = checked(batchSize * outFeatures);
        using var masked = AllocateStreamOrderedScratchBufferUninitialized(outputs);
        switch (activationCode)
        {
            case LinearReluCode: ReluBackward(gradOutput, saved, masked, outputs); break;
            case LinearSigmoidCode: SigmoidBackward(gradOutput, saved, masked, outputs); break;
            case LinearTanhCode: TanhBackward(gradOutput, saved, masked, outputs); break;
            case LinearGeluCode: GeluBackward(gradOutput, saved, masked, outputs); break;
            case LinearSwishCode: SwishBackward(gradOutput, saved, masked, outputs); break;
            default: throw new System.ArgumentOutOfRangeException(nameof(activationCode));
        }
        // dX[batch, in] = mask[batch, out] · W[in, out]ᵀ; dW[in, out] = X[batch, in]ᵀ · mask[batch, out].
        MatMulTransposed(masked, weight, gradInput, batchSize, inFeatures, outFeatures);
        MatMulTransposedA(input, masked, gradWeight, inFeatures, outFeatures, batchSize);

        if (!_kernelCache.TryGetValue("fused_linear_bias_grad", out var bgKernel))
            throw new GpuCapabilityUnavailableException("CUDA kernel not found: fused_linear_bias_grad");
        using var _ = PushContext();
        uint gridBias = (uint)((outFeatures + DefaultBlockSize - 1) / DefaultBlockSize);
        System.IntPtr goPtr = gradOutput.Handle, sPtr = saved.Handle, gbPtr = gradBias.Handle;
        int rows = batchSize, cols = outFeatures, code = activationCode;
        void** argsBias = stackalloc void*[6];
        argsBias[0] = &goPtr; argsBias[1] = &sPtr; argsBias[2] = &gbPtr;
        argsBias[3] = &rows; argsBias[4] = &cols; argsBias[5] = &code;
        LaunchKernel(bgKernel, gridBias, DefaultBlockSize, argsBias);
        return true;
    }
}
