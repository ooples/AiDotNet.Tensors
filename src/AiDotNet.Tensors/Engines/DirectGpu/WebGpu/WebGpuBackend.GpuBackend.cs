// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation for WebGpuBackend: helpers, memory, GEMM, fused GEMM, broadcast,
// element-wise, capsule, trig, hyperbolic, unary, sparse, reduction, sync, transpose, random.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Internal Helpers

    private static WebGpuBuffer AsWebGpu(IGpuBuffer buffer)
    {
        if (buffer is not WebGpuBuffer wb)
            throw new ArgumentException("Buffer is not a WebGpuBuffer.", nameof(buffer));
        return wb;
    }

    private void EnsureInitialized()
    {
        if (!_initialized || _disposed)
            throw new InvalidOperationException("WebGPU backend not initialized.");
    }

    private void UploadToBuffer(float[] data, IGpuBuffer buffer)
    {
        var wb = AsWebGpu(buffer);
        wb.CopyFrom(data);
    }

    private float[] DownloadBufferData(IGpuBuffer buffer)
    {
        var wb = AsWebGpu(buffer);
        return wb.Download();
    }

    // All compute operations dispatched via GPU kernels - no CPU fallback helpers.

    #endregion

    #region Memory Management

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        EnsureInitialized();
        return DownloadBufferData(buffer);
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        EnsureInitialized();
        var data = DownloadBufferData(buffer);
        Array.Copy(data, destination, Math.Min(data.Length, destination.Length));
    }

    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            0, 0, 0
        };
        Dispatch2BufferAsync("CopyOps", WebGpuKernels.CopyOpsSource, "copy_simple",
            source, destination, uniforms, size).GetAwaiter().GetResult();
    }

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(length),
            BitConverter.Int32BitsToSingle(sourceOffset),
            BitConverter.Int32BitsToSingle(destinationOffset),
            0
        };
        Dispatch2BufferAsync("CopyOps", WebGpuKernels.CopyOpsSource, "copy_offset",
            source, destination, uniforms, length).GetAwaiter().GetResult();
    }

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        EnsureInitialized();
        // WebGPU buffers are always float32-addressed. A true byte-addressed buffer
        // (e.g. for FP16 / mixed-precision) would require a WebGpuBuffer variant that
        // tracks bytes and element size, which is not yet implemented.
        throw new NotSupportedException(
            "Byte-addressed GPU buffers (e.g., for FP16/mixed-precision) are not supported by the WebGpuBackend. " +
            "Use float-based buffers or a backend that supports byte-addressed buffers.");
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        EnsureInitialized();
        return new WebGpuBuffer(size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        EnsureInitialized();
        var floatData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            floatData[i] = BitConverter.Int32BitsToSingle(data[i]);
        return new WebGpuBuffer(floatData);
    }

    #endregion

    #region GEMM Operations

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1f, float beta = 0f)
    {
        GemmAsync(A, B, C, M, N, K, alpha, beta).GetAwaiter().GetResult();
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        var C = AllocateBuffer(M * N);
        Gemm(A, B, C, M, N, K);
        return C;
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1f, float beta = 0f)
    {
        // BatchedGemmSource: binding(0)=A, binding(1)=B, binding(2)=C, uniform: batch_size, M, N, K, alpha, beta, pad, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchCount),
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(N),
            BitConverter.Int32BitsToSingle(K),
            alpha,
            beta,
            0, 0
        };
        int totalElements = batchCount * M * N;
        Dispatch3BufferAsync("BatchedGemm", WebGpuKernels.BatchedGemmSource, "batched_gemm",
            A, B, C, uniforms, totalElements).GetAwaiter().GetResult();
    }

    #endregion

    #region Fused GEMM Operations

    // activation: 0=none, 1=relu, 2=gelu, 3=sigmoid, 4=tanh
    private IGpuBuffer GemmBiasActivation(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, int activationType = 0)
    {
        var output = AllocateBuffer(M * N);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(N),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(activationType)
        };
        Dispatch4BufferAsync("FusedGemmBias", WebGpuKernels.FusedGemmBiasSource, "gemm_bias_act",
            A, B, bias, output, uniforms, M * N).GetAwaiter().GetResult();
        return output;
    }

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, 1);

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, 2);

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, 3);

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, 4);

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, 0);

    #endregion

    #region Broadcast Operations

    public void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        Dispatch3BufferAsync("BiasAdd", WebGpuKernels.BiasAddSource, "bias_add",
            A, bias, C, MakeUniformInts2(M, N), M * N).GetAwaiter().GetResult();
    }

    public void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        int total = batch * channels * spatialSize;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            0
        };
        Dispatch2BufferAsync("Conv2DBiasAdd", WebGpuKernels.Conv2DBiasAddSource, "conv2d_bias_add",
            output, bias, uniforms, total).GetAwaiter().GetResult();
    }

    #endregion

    #region Element-wise Operations

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => AddAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => SubAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => MulAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => DivAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => MinimumAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => MaximumAsync(A, B, C, size).GetAwaiter().GetResult();
    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size) => ScaleAsync(A, B, scalar, size).GetAwaiter().GetResult();
    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size) => PowerAsync(A, B, exponent, size).GetAwaiter().GetResult();
    public void Abs(IGpuBuffer A, IGpuBuffer B, int size) => AbsAsync(A, B, size).GetAwaiter().GetResult();
    public void Exp(IGpuBuffer A, IGpuBuffer B, int size) => ExpAsync(A, B, size).GetAwaiter().GetResult();
    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "exp2_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "exp10_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "expm1_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Log(IGpuBuffer A, IGpuBuffer B, int size) => LogAsync(A, B, size).GetAwaiter().GetResult();
    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "log2_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "log1p_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size) => SqrtAsync(A, B, size).GetAwaiter().GetResult();
    public void Sign(IGpuBuffer A, IGpuBuffer B, int size) => SignAsync(A, B, size).GetAwaiter().GetResult();
    public void Relu(IGpuBuffer A, IGpuBuffer B, int size) => ReLUAsync(A, B, size).GetAwaiter().GetResult();
    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size) => SigmoidAsync(A, B, size).GetAwaiter().GetResult();
    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size) => TanhAsync(A, B, size).GetAwaiter().GetResult();
    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size) => GeLUAsync(A, B, size).GetAwaiter().GetResult();
    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features) => SoftmaxAsync(A, B, batchSize, features).GetAwaiter().GetResult();

    #endregion

    #region Capsule Operations

    public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numCapsules),
            BitConverter.Int32BitsToSingle(capsuleDim),
            epsilon, 0
        };
        Dispatch2BufferAsync("CapsuleSquash", WebGpuKernels.CapsuleSquashSource, "squash",
            input, output, uniforms, numCapsules).GetAwaiter().GetResult();
    }

    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        // CPU fallback: compute squash Jacobian and propagate gradients
        var goData = DownloadBufferData(gradOutput);
        var inData = DownloadBufferData(input);
        var giData = new float[numCapsules * capsuleDim];
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            float sqNorm = 0;
            for (int d = 0; d < capsuleDim; d++)
                sqNorm += inData[off + d] * inData[off + d];
            float norm = MathF.Sqrt(sqNorm + epsilon);
            float scale = sqNorm / ((1 + sqNorm) * norm);
            float dscale = 1f / ((1 + sqNorm) * (1 + sqNorm) * norm);
            for (int d = 0; d < capsuleDim; d++)
            {
                float gi = 0;
                for (int k = 0; k < capsuleDim; k++)
                {
                    float jac = (d == k ? scale : 0f) - dscale * inData[off + d] * inData[off + k];
                    gi += jac * goData[off + k];
                }
                giData[off + d] = gi;
            }
        }
        UploadToBuffer(giData, gradInput);
    }

    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputCapsules),
            BitConverter.Int32BitsToSingle(outputCapsules),
            BitConverter.Int32BitsToSingle(inputDim),
            BitConverter.Int32BitsToSingle(outputDim),
            0, 0, 0
        };
        int total = batchSize * inputCapsules * outputCapsules * outputDim;
        Dispatch3BufferAsync("CapsuleOps", WebGpuKernels.CapsuleOpsSource, "capsule_predictions",
            input, weights, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
        => CapsulePredictions(input, weights, output, batchSize, inputCapsules, inputDim, numCapsules, capsuleDim);

    public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputCapsules),
            BitConverter.Int32BitsToSingle(outputCapsules),
            0, // inputDim unused
            BitConverter.Int32BitsToSingle(capsuleDim),
            0, 0, 0
        };
        int total = batchSize * outputCapsules * capsuleDim;
        Dispatch3BufferAsync("CapsuleOps", WebGpuKernels.CapsuleOpsSource, "capsule_weighted_sum",
            coupling, predictions, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputCapsules),
            BitConverter.Int32BitsToSingle(outputCapsules),
            0, // inputDim unused
            BitConverter.Int32BitsToSingle(capsuleDim),
            0, 0, 0
        };
        int total = batchSize * inputCapsules * outputCapsules;
        Dispatch3BufferAsync("CapsuleOps", WebGpuKernels.CapsuleOpsSource, "capsule_agreement",
            predictions, output, agreement, uniforms, total).GetAwaiter().GetResult();
    }

    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(innerSize),
            BitConverter.Int32BitsToSingle(repeats),
            0, 0
        };
        int total = repeats * innerSize;
        Dispatch2BufferAsync("Tile", WebGpuKernels.TileSource, "tile_batch",
            input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(innerSize),
            BitConverter.Int32BitsToSingle(repeats),
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(axisSize)
        };
        int total = outerSize * axisSize * repeats * innerSize;
        Dispatch2BufferAsync("Tile", WebGpuKernels.TileSource, "tile_axis",
            input, output, uniforms, total).GetAwaiter().GetResult();
    }

    #endregion

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "sin_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "cos_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "tan_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "asin_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "acos_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "atan_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "sinh_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("Trig", WebGpuKernels.TrigSource, "cosh_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("InverseHyperbolic", WebGpuKernels.InverseHyperbolicSource, "asinh_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("InverseHyperbolic", WebGpuKernels.InverseHyperbolicSource, "acosh_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("InverseHyperbolic", WebGpuKernels.InverseHyperbolicSource, "atanh_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchUnaryOpAsync("reciprocal_op", A, B, size).GetAwaiter().GetResult();
    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "cbrt_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "log10_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();
    public void Negate(IGpuBuffer A, IGpuBuffer B, int size) => NegAsync(A, B, size).GetAwaiter().GetResult();
    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchUnaryOpAsync("floor_op", A, B, size).GetAwaiter().GetResult();
    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchUnaryOpAsync("ceil_op", A, B, size).GetAwaiter().GetResult();
    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
        => DispatchUnaryOpAsync("round_op", A, B, size).GetAwaiter().GetResult();
    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
        => Dispatch2BufferAsync("AdditionalUnary", WebGpuKernels.AdditionalUnarySource, "trunc_op", A, B, MakeUniform1(size), size).GetAwaiter().GetResult();

    #endregion

    #region Sparse Operations

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        int sparseK = K / 2;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(sparseK),
            0
        };
        int totalBlocks = M * (K / 4);
        Dispatch3BufferAsync("Sparse2x4", WebGpuKernels.Sparse2x4Source, "enforce_2x4_sparsity",
            denseInput, sparseValues, sparseIndices, uniforms, totalBlocks).GetAwaiter().GetResult();
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        int sparseK = K / 2;
        Fill(denseOutput, 0f, M * K);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(sparseK),
            0
        };
        int totalBlocks = M * (K / 4);
        Dispatch3BufferAsync("Sparse2x4", WebGpuKernels.Sparse2x4Source, "decompress_2x4_sparse",
            sparseValues, sparseIndices, denseOutput, uniforms, totalBlocks).GetAwaiter().GetResult();
    }

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha = 1f, float beta = 0f)
    {
        EnsureInitialized();
        using var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        Gemm(denseA, B, C, M, N, K, alpha, beta);
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        EnsureInitialized();
        using var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        var result = GemmBiasRelu(denseA, B, bias, M, N, K);
        return result;
    }

    public void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(N),
            BitConverter.Int32BitsToSingle(nnz)
        };
        int total = M * N;
        Dispatch5BufferAsync("CsrSpMM", WebGpuKernels.CsrSpMMSource, "csr_spmm",
            csrValues, csrColIndices, csrRowPointers, denseB, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);
        BiasAdd(output, bias, output, M, N);
    }

    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        // Copy input to output first (base values)
        Copy(input, output, numNodes * features);
        // Create dummy edge values buffer if null
        IGpuBuffer evBuf;
        WebGpuBuffer? ownedDummy = null;
        if (edgeValues is not null)
        {
            evBuf = edgeValues;
        }
        else
        {
            ownedDummy = (WebGpuBuffer)AllocateBuffer(1);
            evBuf = ownedDummy;
        }
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numNodes),
            BitConverter.Int32BitsToSingle(numEdges),
            BitConverter.Int32BitsToSingle(features),
            BitConverter.Int32BitsToSingle(edgeValues is not null ? 1 : 0)
        };
        int total = numNodes * features;
        Dispatch5BufferAsync("ScatterEdges", WebGpuKernels.ScatterEdgesSource, "scatter_add_edges",
            input, sourceIndices, targetIndices, evBuf, output, uniforms, total).GetAwaiter().GetResult();
        ownedDummy?.Dispose();
    }

    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(N),
            0
        };
        int total = M * N;
        Dispatch4BufferAsync("CsrSegmented", WebGpuKernels.CsrSegmentedSource, "csr_segmented_max",
            csrColIndices, csrRowPointers, input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(N),
            0
        };
        int total = M * N;
        Dispatch4BufferAsync("CsrSegmented", WebGpuKernels.CsrSegmentedSource, "csr_segmented_min",
            csrColIndices, csrRowPointers, input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(K),
            BitConverter.Int32BitsToSingle(N),
            epsilon
        };
        int total = M * N;
        Dispatch4BufferAsync("CsrSegmented", WebGpuKernels.CsrSegmentedSource, "csr_segmented_stddev",
            csrColIndices, csrRowPointers, input, output, uniforms, total).GetAwaiter().GetResult();
    }

    #endregion

    #region Reduction Operations

    public float Sum(IGpuBuffer A, int size) => SumAsync(A, size).GetAwaiter().GetResult();
    public float Max(IGpuBuffer A, int size) => MaxAsync(A, size).GetAwaiter().GetResult();

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "sum_axis",
            A, B, MakeUniformInts2(outerSize, reduceSize), outerSize).GetAwaiter().GetResult();
    }

    #endregion

    #region Synchronization

    public void Synchronize() { /* WebGPU operations are submitted synchronously via GetAwaiter().GetResult() */ }

    #endregion

    #region Transpose and Reshape Operations

    public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        TransposeAsync(A, B, rows, cols).GetAwaiter().GetResult();
    }

    public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        // BatchedTransposeSource: binding(0)=input, binding(1)=output, binding(2)=uniform
        // Uniform: batch_size, rows, cols, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(rows),
            BitConverter.Int32BitsToSingle(cols),
            0
        };
        int totalElements = batch * rows * cols;
        Dispatch2BufferAsync("BatchedTranspose", WebGpuKernels.BatchedTransposeSource, "batched_transpose",
            A, B, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        int ndim = shape.Length;
        if (ndim > 4)
        {
            throw new ArgumentException(
                $"Permute supports up to 4 dimensions, but got {ndim}. " +
                "The GPU kernel uses fixed 4-element stride arrays.",
                nameof(shape));
        }
        int total = 1;
        for (int i = 0; i < ndim; i++) total *= shape[i];
        // Compute output shape and strides
        var newShape = new int[ndim];
        for (int i = 0; i < ndim; i++) newShape[i] = shape[permutation[i]];
        var outStrides = new int[4]; // max 4 dims
        var inStrides = new int[4];
        var perm = new int[4];
        var shapeArr = new int[4];
        // Initialize with identity (for unused dims)
        for (int i = 0; i < 4; i++) { outStrides[i] = 1; inStrides[i] = 1; perm[i] = i; shapeArr[i] = 1; }
        // Compute strides
        if (ndim > 0)
        {
            var srcStr = new int[ndim]; var dstStr = new int[ndim];
            srcStr[ndim - 1] = 1; dstStr[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; i--)
            {
                srcStr[i] = srcStr[i + 1] * shape[i + 1];
                dstStr[i] = dstStr[i + 1] * newShape[i + 1];
            }
            for (int i = 0; i < ndim && i < 4; i++)
            {
                outStrides[i] = dstStr[i];
                inStrides[i] = srcStr[i];
                perm[i] = permutation[i];
                shapeArr[i] = newShape[i];
            }
        }
        // Pack uniform: total, ndim, pad, pad, shape(4), out_strides(4), in_strides(4), perm(4)
        var uniforms = new float[20];
        uniforms[0] = BitConverter.Int32BitsToSingle(total);
        uniforms[1] = BitConverter.Int32BitsToSingle(ndim);
        uniforms[2] = 0; uniforms[3] = 0;
        for (int i = 0; i < 4; i++)
        {
            uniforms[4 + i] = BitConverter.Int32BitsToSingle(shapeArr[i]);
            uniforms[8 + i] = BitConverter.Int32BitsToSingle(outStrides[i]);
            uniforms[12 + i] = BitConverter.Int32BitsToSingle(inStrides[i]);
            uniforms[16 + i] = BitConverter.Int32BitsToSingle(perm[i]);
        }
        Dispatch2BufferAsync("Permute", WebGpuKernels.PermuteSource, "permute_op",
            input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        // Copy2DStridedSource copy_2d_strided_offset: handles all cases including non-zero offset
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numRows),
            BitConverter.Int32BitsToSingle(srcCols),
            BitConverter.Int32BitsToSingle(destTotalCols),
            BitConverter.Int32BitsToSingle(destColOffset)
        };
        int totalElements = numRows * srcCols;
        Dispatch2BufferAsync("Copy2DStrided", WebGpuKernels.Copy2DStridedSource, "copy_2d_strided_offset",
            source, destination, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        // UpsampleSource nearest_upsample2d: binding(0)=input, binding(1)=output, binding(2)=uniform
        // Uniform: batch_channels, in_height, in_width, scale_h, scale_w, pad, pad, pad
        int outH = height * scaleFactor;
        int outW = width * scaleFactor;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchChannels),
            BitConverter.Int32BitsToSingle(height),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(scaleFactor),
            BitConverter.Int32BitsToSingle(scaleFactor),
            0, 0, 0
        };
        int totalElements = batchChannels * outH * outW;
        Dispatch2BufferAsync("Upsample", WebGpuKernels.UpsampleSource, "nearest_upsample2d",
            input, output, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        // UpsampleSource nearest_upsample2d_backward: binding(0)=gradOutput(input), binding(1)=gradInput(output)
        // Accumulates from upsampled space back to original space
        Fill(gradInput, 0f, batchChannels * height * width);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchChannels),
            BitConverter.Int32BitsToSingle(height),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(scaleFactor),
            BitConverter.Int32BitsToSingle(scaleFactor),
            0, 0, 0
        };
        int totalElements = batchChannels * height * width;
        Dispatch2BufferAsync("Upsample", WebGpuKernels.UpsampleSource, "nearest_upsample2d_backward",
            gradOutput, gradInput, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        Dispatch1BufferAsync("Fill", WebGpuKernels.FillSource, "fill_value",
            buffer, MakeUniform2(size, value), size).GetAwaiter().GetResult();
    }

    #endregion

    #region Random Number Generation

    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        // PhiloxRngSource gpu_random: mode=0 (uniform)
        uint seedLo = (uint)(seed & 0xFFFFFFFF);
        uint seedHi = (uint)((seed >> 32) & 0xFFFFFFFF);
        if (seed == 0) { seedLo = (uint)Environment.TickCount; seedHi = (uint)(Environment.TickCount >> 16) ^ 0xDEADBEEF; }
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle((int)size),
            BitConverter.Int32BitsToSingle((int)seedLo),
            BitConverter.Int32BitsToSingle((int)seedHi),
            BitConverter.Int32BitsToSingle(0), // mode=0 uniform
            min, max, 0, 0
        };
        Dispatch1BufferAsync("PhiloxRng", WebGpuKernels.PhiloxRngSource, "gpu_random",
            output, uniforms, size).GetAwaiter().GetResult();
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        // PhiloxRngSource gpu_random: mode=1 (normal via Box-Muller)
        uint seedLo = (uint)(seed & 0xFFFFFFFF);
        uint seedHi = (uint)((seed >> 32) & 0xFFFFFFFF);
        if (seed == 0) { seedLo = (uint)Environment.TickCount; seedHi = (uint)(Environment.TickCount >> 16) ^ 0xDEADBEEF; }
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle((int)size),
            BitConverter.Int32BitsToSingle((int)seedLo),
            BitConverter.Int32BitsToSingle((int)seedHi),
            BitConverter.Int32BitsToSingle(1), // mode=1 normal
            0, 0, mean, stdDev
        };
        Dispatch1BufferAsync("PhiloxRng", WebGpuKernels.PhiloxRngSource, "gpu_random",
            output, uniforms, size).GetAwaiter().GetResult();
    }

    #endregion
}
#endif
