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

    private void CpuUnary(IGpuBuffer A, IGpuBuffer B, int size, Func<float, float> op)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A);
        var b = new float[size];
        for (int i = 0; i < size; i++) b[i] = op(a[i]);
        UploadToBuffer(b, B);
    }

    private void CpuBinary(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size, Func<float, float, float> op)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A);
        var b = DownloadBufferData(B);
        var c = new float[size];
        for (int i = 0; i < size; i++) c[i] = op(a[i], b[i]);
        UploadToBuffer(c, C);
    }

    private float CpuReduce(IGpuBuffer A, int size, float seed, Func<float, float, float> accumulate)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A);
        float result = seed;
        for (int i = 0; i < size; i++) result = accumulate(result, a[i]);
        return result;
    }

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
        EnsureInitialized();
        var data = DownloadBufferData(source);
        UploadToBuffer(data, destination);
    }

    public void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        EnsureInitialized();
        var src = DownloadBufferData(source);
        var dst = DownloadBufferData(destination);
        Array.Copy(src, sourceOffset, dst, destinationOffset, length);
        UploadToBuffer(dst, destination);
    }

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        EnsureInitialized();
        int floatCount = (size + sizeof(float) - 1) / sizeof(float);
        return new WebGpuBuffer(floatCount);
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
        EnsureInitialized();
        var a = DownloadBufferData(A); var b = DownloadBufferData(B); var c = DownloadBufferData(C);
        int aStride = M * K, bStride = K * N, cStride = M * N;
        for (int batch = 0; batch < batchCount; batch++)
        {
            int aOff = batch * aStride, bOff = batch * bStride, cOff = batch * cStride;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < K; k++) sum += a[aOff + i * K + k] * b[bOff + k * N + j];
                    c[cOff + i * N + j] = alpha * sum + beta * c[cOff + i * N + j];
                }
        }
        UploadToBuffer(c, C);
    }

    #endregion

    #region Fused GEMM Operations

    private IGpuBuffer GemmBiasActivation(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, Func<float, float>? activation = null)
    {
        EnsureInitialized();
        var a = DownloadBufferData(A); var b = DownloadBufferData(B); var bi = DownloadBufferData(bias);
        var c = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
                sum += bi[j];
                c[i * N + j] = activation is not null ? activation(sum) : sum;
            }
        return AllocateBuffer(c);
    }

    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, v => MathF.Max(0, v));

    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, v => 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))));

    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, v => 1f / (1f + MathF.Exp(-v)));

    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K, MathF.Tanh);

    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmBiasActivation(A, B, bias, M, N, K);

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
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[numCapsules * capsuleDim];
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            float sq = 0;
            for (int d = 0; d < capsuleDim; d++) sq += inp[off + d] * inp[off + d];
            float scale = sq / ((1f + sq) * MathF.Sqrt(sq + epsilon));
            for (int d = 0; d < capsuleDim; d++) o[off + d] = inp[off + d] * scale;
        }
        UploadToBuffer(o, output);
    }

    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        Fill(gradInput, 0f, numCapsules * capsuleDim);
    }

    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var w = DownloadBufferData(weights);
        var o = new float[batchSize * inputCapsules * outputCapsules * outputDim];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < inputCapsules; i++)
                for (int j = 0; j < outputCapsules; j++)
                    for (int od = 0; od < outputDim; od++)
                    {
                        float sum = 0;
                        for (int id = 0; id < inputDim; id++)
                            sum += inp[(b * inputCapsules + i) * inputDim + id] * w[((i * outputCapsules + j) * inputDim + id) * outputDim + od];
                        o[((b * inputCapsules + i) * outputCapsules + j) * outputDim + od] = sum;
                    }
        UploadToBuffer(o, output);
    }

    public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
        => CapsulePredictions(input, weights, output, batchSize, inputCapsules, inputDim, numCapsules, capsuleDim);

    public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        EnsureInitialized();
        var cc = DownloadBufferData(coupling); var pred = DownloadBufferData(predictions);
        var o = new float[batchSize * outputCapsules * capsuleDim];
        for (int b = 0; b < batchSize; b++)
            for (int j = 0; j < outputCapsules; j++)
                for (int d = 0; d < capsuleDim; d++)
                {
                    float sum = 0;
                    for (int i = 0; i < inputCapsules; i++)
                        sum += cc[b * inputCapsules * outputCapsules + i * outputCapsules + j]
                             * pred[((b * inputCapsules + i) * outputCapsules + j) * capsuleDim + d];
                    o[(b * outputCapsules + j) * capsuleDim + d] = sum;
                }
        UploadToBuffer(o, output);
    }

    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        EnsureInitialized();
        var pred = DownloadBufferData(predictions); var cur = DownloadBufferData(output); var lo = DownloadBufferData(agreement);
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < inputCapsules; i++)
                for (int j = 0; j < outputCapsules; j++)
                {
                    float dot = 0;
                    for (int d = 0; d < capsuleDim; d++)
                        dot += pred[((b * inputCapsules + i) * outputCapsules + j) * capsuleDim + d]
                             * cur[(b * outputCapsules + j) * capsuleDim + d];
                    lo[b * inputCapsules * outputCapsules + i * outputCapsules + j] += dot;
                }
        UploadToBuffer(lo, agreement);
    }

    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[repeats * innerSize];
        for (int t = 0; t < repeats; t++)
            Array.Copy(inp, 0, o, t * innerSize, innerSize);
        UploadToBuffer(o, output);
    }

    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[outerSize * axisSize * repeats * innerSize];
        for (int i = 0; i < outerSize; i++)
            for (int a = 0; a < axisSize; a++)
                for (int r = 0; r < repeats; r++)
                    Array.Copy(inp, (i * axisSize + a) * innerSize, o, ((i * axisSize * repeats + a * repeats + r)) * innerSize, innerSize);
        UploadToBuffer(o, output);
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
        EnsureInitialized();
        var dense = DownloadBufferData(denseInput);
        int sparseK = K / 2;
        var vals = new float[M * sparseK]; var idxs = new float[M * sparseK];
        for (int row = 0; row < M; row++)
            for (int blk = 0; blk < K / 4; blk++)
            {
                int baseIdx = row * K + blk * 4;
                var pairs = new (float val, int idx)[4];
                for (int i = 0; i < 4; i++) pairs[i] = (MathF.Abs(dense[baseIdx + i]), i);
                Array.Sort(pairs, (a, b) => b.val.CompareTo(a.val));
                int sOff = row * sparseK + blk * 2;
                for (int i = 0; i < 2; i++)
                {
                    vals[sOff + i] = dense[baseIdx + pairs[i].idx];
                    idxs[sOff + i] = BitConverter.Int32BitsToSingle(pairs[i].idx);
                }
            }
        UploadToBuffer(vals, sparseValues); UploadToBuffer(idxs, sparseIndices);
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        EnsureInitialized();
        var vals = DownloadBufferData(sparseValues); var idxs = DownloadBufferData(sparseIndices);
        int sparseK = K / 2;
        var dense = new float[M * K];
        for (int row = 0; row < M; row++)
            for (int blk = 0; blk < K / 4; blk++)
            {
                int sOff = row * sparseK + blk * 2;
                int baseIdx = row * K + blk * 4;
                for (int i = 0; i < 2; i++)
                {
                    int col = BitConverter.SingleToInt32Bits(idxs[sOff + i]);
                    dense[baseIdx + col] = vals[sOff + i];
                }
            }
        UploadToBuffer(dense, denseOutput);
    }

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha = 1f, float beta = 0f)
    {
        EnsureInitialized();
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        Gemm(denseA, B, C, M, N, K, alpha, beta);
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        EnsureInitialized();
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        return GemmBiasRelu(denseA, B, bias, M, N, K);
    }

    public void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        EnsureInitialized();
        var vals = DownloadBufferData(csrValues); var cols = DownloadBufferData(csrColIndices);
        var ptrs = DownloadBufferData(csrRowPointers); var b = DownloadBufferData(denseB);
        var o = new float[M * N];
        for (int row = 0; row < M; row++)
        {
            int start = BitConverter.SingleToInt32Bits(ptrs[row]);
            int end = BitConverter.SingleToInt32Bits(ptrs[row + 1]);
            for (int idx = start; idx < end; idx++)
            {
                int col = BitConverter.SingleToInt32Bits(cols[idx]);
                float val = vals[idx];
                for (int j = 0; j < N; j++) o[row * N + j] += val * b[col * N + j];
            }
        }
        UploadToBuffer(o, output);
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);
        var o = DownloadBufferData(output); var bi = DownloadBufferData(bias);
        for (int row = 0; row < M; row++)
            for (int j = 0; j < N; j++) o[row * N + j] += bi[j];
        UploadToBuffer(o, output);
    }

    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var src = DownloadBufferData(sourceIndices); var tgt = DownloadBufferData(targetIndices);
        float[]? ev = edgeValues is not null ? DownloadBufferData(edgeValues) : null;
        var o = new float[numNodes * features];
        Array.Copy(inp, o, Math.Min(inp.Length, o.Length));
        for (int e = 0; e < numEdges; e++)
        {
            int s = BitConverter.SingleToInt32Bits(src[e]);
            int t = BitConverter.SingleToInt32Bits(tgt[e]);
            float w = ev is not null ? ev[e] : 1f;
            for (int f = 0; f < features; f++) o[t * features + f] += inp[s * features + f] * w;
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        var cols = DownloadBufferData(csrColIndices); var ptrs = DownloadBufferData(csrRowPointers);
        var inp = DownloadBufferData(input);
        var o = new float[M * N];
        Array.Fill(o, float.MinValue);
        for (int row = 0; row < M; row++)
        {
            int start = BitConverter.SingleToInt32Bits(ptrs[row]);
            int end = BitConverter.SingleToInt32Bits(ptrs[row + 1]);
            for (int idx = start; idx < end; idx++)
            {
                int col = BitConverter.SingleToInt32Bits(cols[idx]);
                for (int j = 0; j < N; j++) o[row * N + j] = MathF.Max(o[row * N + j], inp[col * N + j]);
            }
            if (start == end)
                for (int j = 0; j < N; j++) o[row * N + j] = 0;
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        var cols = DownloadBufferData(csrColIndices); var ptrs = DownloadBufferData(csrRowPointers);
        var inp = DownloadBufferData(input);
        var o = new float[M * N];
        Array.Fill(o, float.MaxValue);
        for (int row = 0; row < M; row++)
        {
            int start = BitConverter.SingleToInt32Bits(ptrs[row]);
            int end = BitConverter.SingleToInt32Bits(ptrs[row + 1]);
            for (int idx = start; idx < end; idx++)
            {
                int col = BitConverter.SingleToInt32Bits(cols[idx]);
                for (int j = 0; j < N; j++) o[row * N + j] = MathF.Min(o[row * N + j], inp[col * N + j]);
            }
            if (start == end)
                for (int j = 0; j < N; j++) o[row * N + j] = 0;
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        EnsureInitialized();
        var cols = DownloadBufferData(csrColIndices); var ptrs = DownloadBufferData(csrRowPointers);
        var inp = DownloadBufferData(input);
        var o = new float[M * N];
        for (int row = 0; row < M; row++)
        {
            int start = BitConverter.SingleToInt32Bits(ptrs[row]);
            int end = BitConverter.SingleToInt32Bits(ptrs[row + 1]);
            int count = end - start;
            if (count == 0) continue;
            for (int j = 0; j < N; j++)
            {
                float mean = 0;
                for (int idx = start; idx < end; idx++) mean += inp[BitConverter.SingleToInt32Bits(cols[idx]) * N + j];
                mean /= count;
                float var_ = 0;
                for (int idx = start; idx < end; idx++) { float d = inp[BitConverter.SingleToInt32Bits(cols[idx]) * N + j] - mean; var_ += d * d; }
                o[row * N + j] = MathF.Sqrt(var_ / count + epsilon);
            }
        }
        UploadToBuffer(o, output);
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
        EnsureInitialized();
        var inp = DownloadBufferData(A);
        var o = new float[batch * rows * cols];
        int stride = rows * cols;
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) o[b * stride + c * rows + r] = inp[b * stride + r * cols + c];
        UploadToBuffer(o, B);
    }

    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        int ndim = shape.Length; int total = 1;
        for (int i = 0; i < ndim; i++) total *= shape[i];
        var newShape = new int[ndim];
        for (int i = 0; i < ndim; i++) newShape[i] = shape[permutation[i]];
        var o = new float[total];
        var srcStrides = new int[ndim]; var dstStrides = new int[ndim];
        srcStrides[ndim - 1] = 1; dstStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) { srcStrides[i] = srcStrides[i + 1] * shape[i + 1]; dstStrides[i] = dstStrides[i + 1] * newShape[i + 1]; }
        var indices = new int[ndim];
        for (int flat = 0; flat < total; flat++)
        {
            int rem = flat;
            for (int d = 0; d < ndim; d++) { indices[d] = rem / srcStrides[d]; rem %= srcStrides[d]; }
            int dstFlat = 0;
            for (int d = 0; d < ndim; d++) dstFlat += indices[permutation[d]] * dstStrides[d];
            o[dstFlat] = inp[flat];
        }
        UploadToBuffer(o, output);
    }

    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset)
    {
        EnsureInitialized();
        var src = DownloadBufferData(source);
        var dst = DownloadBufferData(destination);
        for (int r = 0; r < numRows; r++)
            for (int c = 0; c < srcCols; c++) dst[r * destTotalCols + destColOffset + c] = src[r * srcCols + c];
        UploadToBuffer(dst, destination);
    }

    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        int outH = height * scaleFactor, outW = width * scaleFactor;
        var o = new float[batchChannels * outH * outW];
        for (int bc = 0; bc < batchChannels; bc++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                    o[(bc * outH + oh) * outW + ow] = inp[(bc * height + oh / scaleFactor) * width + ow / scaleFactor];
        UploadToBuffer(o, output);
    }

    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        EnsureInitialized();
        var grad = DownloadBufferData(gradOutput);
        int outH = height * scaleFactor, outW = width * scaleFactor;
        var gi = new float[batchChannels * height * width];
        for (int bc = 0; bc < batchChannels; bc++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                    gi[(bc * height + oh / scaleFactor) * width + ow / scaleFactor] += grad[(bc * outH + oh) * outW + ow];
        UploadToBuffer(gi, gradInput);
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
        EnsureInitialized();
        var rand = seed != 0 ? new Random((int)(seed & 0x7FFFFFFF)) : new Random();
        var data = new float[size];
        float range = max - min;
        for (int i = 0; i < size; i++) data[i] = (float)rand.NextDouble() * range + min;
        UploadToBuffer(data, output);
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        EnsureInitialized();
        var rand = seed != 0 ? new Random((int)(seed & 0x7FFFFFFF)) : new Random();
        var data = new float[size];
        for (int i = 0; i < size; i += 2)
        {
            float u1 = 1f - (float)rand.NextDouble();
            float u2 = (float)rand.NextDouble();
            float mag = stdDev * MathF.Sqrt(-2f * MathF.Log(u1));
            data[i] = mag * MathF.Cos(2f * MathF.PI * u2) + mean;
            if (i + 1 < size) data[i + 1] = mag * MathF.Sin(2f * MathF.PI * u2) + mean;
        }
        UploadToBuffer(data, output);
    }

    #endregion
}
#endif
