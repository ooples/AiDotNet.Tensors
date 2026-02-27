// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation for VulkanBackend using GPU-accelerated and CPU-fallback operations.

using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    #region Internal Helpers

    // .NET Framework 4.7.1 compatibility helpers
    private static int SingleToInt32BitsCompat(float value)
    {
        return *(int*)&value;
    }

    private static float Int32BitsToSingleCompat(int value)
    {
        return *(float*)&value;
    }

    private static float AsinhCompat(float x)
    {
        return (float)Math.Log(x + Math.Sqrt(x * (double)x + 1.0));
    }

    private static float AcoshCompat(float x)
    {
        return (float)Math.Log(x + Math.Sqrt(x * (double)x - 1.0));
    }

    private static float AtanhCompat(float x)
    {
        return 0.5f * (float)Math.Log((1.0 + x) / (1.0 - x));
    }

    private static float CbrtCompat(float x)
    {
        return (float)Math.Pow(Math.Abs(x), 1.0 / 3.0) * Math.Sign(x);
    }

    private static float Log2Compat(float x)
    {
        return (float)(Math.Log(x) / Math.Log(2.0));
    }

    private static double AcoshDoubleCompat(double x)
    {
        return Math.Log(x + Math.Sqrt(x * x - 1.0));
    }

    private static void ArrayFillCompat(float[] array, float value)
    {
        for (int i = 0; i < array.Length; i++) array[i] = value;
    }

    private static VulkanGpuBuffer AsVulkan(IGpuBuffer buffer)
    {
        if (buffer is not VulkanGpuBuffer vb)
            throw new ArgumentException("Buffer is not a VulkanGpuBuffer.", nameof(buffer));
        return vb;
    }

    private void EnsureInitialized()
    {
        if (!_initialized || _disposed)
            throw new InvalidOperationException("Vulkan backend not initialized.");
    }

    private void UploadToBuffer(float[] data, IGpuBuffer buffer)
    {
        var vb = AsVulkan(buffer);
        vb.Staging.WriteData(data);
        _transfer!.CopyToDevice(vb.Staging, vb.Storage);
    }

    private void GpuBinaryOp(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size, VulkanKernelType kernelType)
    {
        EnsureInitialized();
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var vbC = AsVulkan(C);

        var pipeline = GetOrCreatePipeline(kernelType, 3, sizeof(uint));
        if (pipeline is null)
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");

        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size);
        }
    }

    private void GpuUnaryOp(IGpuBuffer A, IGpuBuffer B, int size, VulkanKernelType kernelType)
    {
        EnsureInitialized();
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);

        var pipeline = GetOrCreatePipeline(kernelType, 2, sizeof(uint));
        if (pipeline is null)
            throw new InvalidOperationException($"Failed to create pipeline for {kernelType}.");

        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage);
            RecordAndExecuteComputeUnlocked(pipeline, size);
        }
    }

    private void CpuUnary(IGpuBuffer A, IGpuBuffer B, int size, Func<float, float> op)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[size];
        for (int i = 0; i < size; i++) b[i] = op(a[i]);
        UploadToBuffer(b, B);
    }

    private void CpuBinary(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size, Func<float, float, float> op)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = DownloadBuffer(B);
        var c = new float[size];
        for (int i = 0; i < size; i++) c[i] = op(a[i], b[i]);
        UploadToBuffer(c, C);
    }

    private float CpuReduce(IGpuBuffer A, int size, float seed, Func<float, float, float> accumulate)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        float result = seed;
        for (int i = 0; i < size; i++) result = accumulate(result, a[i]);
        return result;
    }

    #endregion

    #region Memory Management

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        EnsureInitialized();
        return VulkanGpuBuffer.Create(data, _transfer!);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        EnsureInitialized();
        return VulkanGpuBuffer.Create(size);
    }

    public float[] DownloadBuffer(IGpuBuffer buffer)
    {
        EnsureInitialized();
        var vb = AsVulkan(buffer);
        _transfer!.CopyFromDevice(vb.Storage, vb.Staging);
        var result = new float[vb.Size];
        vb.Staging.ReadData(result);
        return result;
    }

    public void DownloadBuffer(IGpuBuffer buffer, float[] destination)
    {
        EnsureInitialized();
        var vb = AsVulkan(buffer);
        _transfer!.CopyFromDevice(vb.Storage, vb.Staging);
        vb.Staging.ReadData(destination);
    }

    public void Copy(IGpuBuffer source, int srcOffset, IGpuBuffer destination, int destOffset, int size)
    {
        EnsureInitialized();
        var src = DownloadBuffer(source);
        var dst = DownloadBuffer(destination);
        Array.Copy(src, srcOffset, dst, destOffset, size);
        UploadToBuffer(dst, destination);
    }

    public IGpuBuffer AllocateIntBuffer(int size)
    {
        EnsureInitialized();
        return VulkanGpuBuffer.Create(size);
    }

    public IGpuBuffer AllocateIntBuffer(int[] data)
    {
        EnsureInitialized();
        var floatData = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
            floatData[i] = Int32BitsToSingleCompat(data[i]);
        return VulkanGpuBuffer.Create(floatData, _transfer!);
    }

    public IGpuBuffer AllocateByteBuffer(int size)
    {
        EnsureInitialized();
        int floatCount = (size + sizeof(float) - 1) / sizeof(float);
        return VulkanGpuBuffer.Create(floatCount);
    }

    #endregion

    #region GEMM Operations

    public void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = DownloadBuffer(B);
        var c = DownloadBuffer(C);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a[i * K + k] * b[k * N + j];
                c[i * N + j] = alpha * sum + beta * c[i * N + j];
            }
        }
        UploadToBuffer(c, C);
    }

    public IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        var result = AllocateBuffer(M * N);
        Gemm(A, B, result, M, N, K);
        return result;
    }

    public void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = DownloadBuffer(B);
        var c = DownloadBuffer(C);
        int aStride = M * K, bStride = K * N, cStride = M * N;
        for (int batch = 0; batch < batchCount; batch++)
        {
            int aOff = batch * aStride, bOff = batch * bStride, cOff = batch * cStride;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += a[aOff + i * K + k] * b[bOff + k * N + j];
                    c[cOff + i * N + j] = alpha * sum + beta * c[cOff + i * N + j];
                }
        }
        UploadToBuffer(c, C);
    }

    #endregion

    #region Fused GEMM Operations

    private IGpuBuffer GemmBiasActivation(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, Func<float, float> activation)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = DownloadBuffer(B);
        var bi = DownloadBuffer(bias);
        var c = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = bi[j];
                for (int k = 0; k < K; k++)
                    sum += a[i * K + k] * b[k * N + j];
                c[i * N + j] = activation(sum);
            }
        var result = AllocateBuffer(c);
        return result;
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
        => GemmBiasActivation(A, B, bias, M, N, K, v => v);

    #endregion

    #region Broadcast Operations

    public void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var bi = DownloadBuffer(bias);
        var c = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                c[i * N + j] = a[i * N + j] + bi[j];
        UploadToBuffer(c, C);
    }

    public void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        EnsureInitialized();
        var o = DownloadBuffer(output);
        var bi = DownloadBuffer(bias);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                float bv = bi[c];
                int offset = (b * channels + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                    o[offset + s] += bv;
            }
        UploadToBuffer(o, output);
    }

    #endregion

    #region Element-wise Operations (GPU-accelerated)

    public void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorAdd);

    public void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorSubtract);

    public void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorMultiply);

    public void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => GpuBinaryOp(A, B, C, size, VulkanKernelType.VectorDivide);

    public void Relu(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.ReLU);

    public void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.Sigmoid);

    public void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
        => GpuUnaryOp(A, B, size, VulkanKernelType.Tanh);

    public void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        EnsureInitialized();
        var vbA = AsVulkan(A);
        var vbB = AsVulkan(B);
        var pipeline = GetOrCreatePipeline(VulkanKernelType.ScalarMultiply, 2, sizeof(uint) + sizeof(float));
        if (pipeline is null)
            throw new InvalidOperationException("Failed to create scalar multiply pipeline.");

        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage);
            RecordAndExecuteComputeWithScalarUnlocked(pipeline, size, scalar);
        }
    }

    #endregion

    #region Element-wise Operations (CPU fallback)

    public void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => CpuBinary(A, B, C, size, MathF.Min);

    public void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => CpuBinary(A, B, C, size, MathF.Max);

    public void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size)
        => CpuUnary(A, B, size, v => MathF.Pow(v, exponent));

    public void Abs(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, MathF.Abs);

    public void Exp(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, MathF.Exp);

    public void Exp2(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Pow(2f, v));

    public void Exp10(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Pow(10f, v));

    public void ExpM1(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Exp(v) - 1f);

    public void Log(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, MathF.Log);

    public void Log2(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, Log2Compat);

    public void Log1P(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Log(1f + v));

    public void Sqrt(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, MathF.Sqrt);

    public void Sign(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => MathF.Sign(v));

    public void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
        => CpuUnary(A, B, size, v => 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))));

    public void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[batchSize * features];
        for (int i = 0; i < batchSize; i++)
        {
            int off = i * features;
            float max = float.MinValue;
            for (int j = 0; j < features; j++)
                if (a[off + j] > max) max = a[off + j];
            float sum = 0;
            for (int j = 0; j < features; j++)
            {
                b[off + j] = MathF.Exp(a[off + j] - max);
                sum += b[off + j];
            }
            if (sum > 0)
                for (int j = 0; j < features; j++)
                    b[off + j] /= sum;
        }
        UploadToBuffer(b, B);
    }

    #endregion

    #region Capsule Operations

    public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var outp = new float[numCapsules * capsuleDim];
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            float sqNorm = 0;
            for (int d = 0; d < capsuleDim; d++) sqNorm += inp[off + d] * inp[off + d];
            float scale = sqNorm / ((1f + sqNorm) * MathF.Sqrt(sqNorm + epsilon));
            for (int d = 0; d < capsuleDim; d++) outp[off + d] = inp[off + d] * scale;
        }
        UploadToBuffer(outp, output);
    }

    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        EnsureInitialized();
        var grad = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var gi = new float[numCapsules * capsuleDim];
        for (int c = 0; c < numCapsules; c++)
        {
            int off = c * capsuleDim;
            float sqNorm = 0;
            for (int d = 0; d < capsuleDim; d++) sqNorm += inp[off + d] * inp[off + d];
            float norm = MathF.Sqrt(sqNorm + epsilon);
            float denom = (1f + sqNorm) * norm;
            float factor = 1f / (denom * denom) * (sqNorm + 2f * sqNorm / (1f + sqNorm));
            for (int d = 0; d < capsuleDim; d++)
                gi[off + d] = grad[off + d] * factor;
        }
        UploadToBuffer(gi, gradInput);
    }

    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        var o = new float[batchSize * inputCapsules * outputCapsules * outputDim];
        for (int b = 0; b < batchSize; b++)
            for (int ic = 0; ic < inputCapsules; ic++)
                for (int oc = 0; oc < outputCapsules; oc++)
                    for (int od = 0; od < outputDim; od++)
                    {
                        float sum = 0;
                        for (int id = 0; id < inputDim; id++)
                            sum += inp[b * inputCapsules * inputDim + ic * inputDim + id]
                                * w[(ic * outputCapsules + oc) * outputDim * inputDim + od * inputDim + id];
                        o[((b * inputCapsules + ic) * outputCapsules + oc) * outputDim + od] = sum;
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
        var c = DownloadBuffer(coupling);
        var p = DownloadBuffer(predictions);
        var o = new float[batchSize * outputCapsules * capsuleDim];
        for (int b = 0; b < batchSize; b++)
            for (int oc = 0; oc < outputCapsules; oc++)
                for (int d = 0; d < capsuleDim; d++)
                {
                    float sum = 0;
                    for (int ic = 0; ic < inputCapsules; ic++)
                        sum += c[b * inputCapsules * outputCapsules + ic * outputCapsules + oc]
                            * p[((b * inputCapsules + ic) * outputCapsules + oc) * capsuleDim + d];
                    o[(b * outputCapsules + oc) * capsuleDim + d] = sum;
                }
        UploadToBuffer(o, output);
    }

    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        EnsureInitialized();
        var p = DownloadBuffer(predictions);
        var o = DownloadBuffer(output);
        var ag = new float[batchSize * inputCapsules * outputCapsules];
        for (int b = 0; b < batchSize; b++)
            for (int ic = 0; ic < inputCapsules; ic++)
                for (int oc = 0; oc < outputCapsules; oc++)
                {
                    float dot = 0;
                    for (int d = 0; d < capsuleDim; d++)
                        dot += p[((b * inputCapsules + ic) * outputCapsules + oc) * capsuleDim + d]
                            * o[(b * outputCapsules + oc) * capsuleDim + d];
                    ag[b * inputCapsules * outputCapsules + ic * outputCapsules + oc] = dot;
                }
        UploadToBuffer(ag, agreement);
    }

    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var o = new float[repeats * innerSize];
        for (int r = 0; r < repeats; r++)
            Array.Copy(inp, 0, o, r * innerSize, innerSize);
        UploadToBuffer(o, output);
    }

    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var o = new float[outerSize * axisSize * repeats * innerSize];
        for (int outer = 0; outer < outerSize; outer++)
            for (int ax = 0; ax < axisSize; ax++)
                for (int r = 0; r < repeats; r++)
                    Array.Copy(inp, (outer * axisSize + ax) * innerSize, o,
                        ((outer * axisSize + ax) * repeats + r) * innerSize, innerSize);
        UploadToBuffer(o, output);
    }

    #endregion

    #region Trigonometric Operations

    public void Sin(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Sin);
    public void Cos(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Cos);
    public void Tan(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Tan);
    public void Asin(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Asin);
    public void Acos(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Acos);
    public void Atan(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Atan);

    #endregion

    #region Hyperbolic Operations

    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Sinh);
    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Cosh);
    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, AsinhCompat);
    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, AcoshCompat);
    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, AtanhCompat);

    #endregion

    #region Additional Unary Operations

    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => 1f / v);
    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, CbrtCompat);
    public void Log10(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Log10);
    public void Negate(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, v => -v);
    public void Floor(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Floor);
    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Ceiling);
    public void Round(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Round);
    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size) => CpuUnary(A, B, size, MathF.Truncate);

    #endregion

    #region Reduction Operations

    public float Sum(IGpuBuffer A, int size) => CpuReduce(A, size, 0f, (acc, v) => acc + v);

    public float Max(IGpuBuffer A, int size) => CpuReduce(A, size, float.MinValue, MathF.Max);

    public void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[outerSize];
        for (int i = 0; i < outerSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < reduceSize; j++)
                sum += a[i * reduceSize + j];
            b[i] = sum;
        }
        UploadToBuffer(b, B);
    }

    #endregion

    #region Synchronization

    public void Synchronize()
    {
        if (_initialized && !_disposed)
            _device.WaitIdle();
    }

    #endregion

    #region Transpose and Reshape

    public void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                b[j * rows + i] = a[i * cols + j];
        UploadToBuffer(b, B);
    }

    public void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
    {
        EnsureInitialized();
        var a = DownloadBuffer(A);
        var b = new float[batch * rows * cols];
        int stride = rows * cols;
        for (int bi = 0; bi < batch; bi++)
        {
            int off = bi * stride;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    b[off + j * rows + i] = a[off + i * cols + j];
        }
        UploadToBuffer(b, B);
    }

    public void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        int ndim = shape.Length;
        int totalSize = 1;
        for (int i = 0; i < ndim; i++) totalSize *= shape[i];
        var outp = new float[totalSize];

        var outShape = new int[ndim];
        for (int i = 0; i < ndim; i++) outShape[i] = shape[permutation[i]];

        var outStrides = new int[ndim];
        var inStrides = new int[ndim];
        outStrides[ndim - 1] = 1;
        inStrides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            outStrides[i] = outStrides[i + 1] * outShape[i + 1];
            inStrides[i] = inStrides[i + 1] * shape[i + 1];
        }

        for (int idx = 0; idx < totalSize; idx++)
        {
            int remaining = idx;
            var coords = new int[ndim];
            for (int d = 0; d < ndim; d++)
            {
                coords[d] = remaining / outStrides[d];
                remaining %= outStrides[d];
            }

            int srcIdx = 0;
            for (int d = 0; d < ndim; d++)
                srcIdx += coords[d] * inStrides[permutation[d]];
            outp[idx] = inp[srcIdx];
        }
        UploadToBuffer(outp, output);
    }

    public void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        EnsureInitialized();
        var s = DownloadBuffer(source);
        var d = new float[size];
        Array.Copy(s, 0, d, 0, size);
        UploadToBuffer(d, destination);
    }

    public void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows,
        int srcCols, int destTotalCols, int destColOffset)
    {
        EnsureInitialized();
        var s = DownloadBuffer(source);
        var d = DownloadBuffer(destination);
        for (int r = 0; r < numRows; r++)
            Array.Copy(s, r * srcCols, d, r * destTotalCols + destColOffset, srcCols);
        UploadToBuffer(d, destination);
    }

    public void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        int outH = height * scaleFactor, outW = width * scaleFactor;
        var outp = new float[batchChannels * outH * outW];
        for (int bc = 0; bc < batchChannels; bc++)
        {
            int inOff = bc * height * width;
            int outOff = bc * outH * outW;
            for (int oh = 0; oh < outH; oh++)
            {
                int ih = oh / scaleFactor;
                for (int ow = 0; ow < outW; ow++)
                    outp[outOff + oh * outW + ow] = inp[inOff + ih * width + ow / scaleFactor];
            }
        }
        UploadToBuffer(outp, output);
    }

    public void NearestNeighborUpsampleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchChannels, int height, int width, int scaleFactor)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        int outH = height * scaleFactor, outW = width * scaleFactor;
        var gi = new float[batchChannels * height * width];
        for (int bc = 0; bc < batchChannels; bc++)
        {
            int giOff = bc * height * width;
            int goOff = bc * outH * outW;
            for (int oh = 0; oh < outH; oh++)
            {
                int ih = oh / scaleFactor;
                for (int ow = 0; ow < outW; ow++)
                    gi[giOff + ih * width + ow / scaleFactor] += go[goOff + oh * outW + ow];
            }
        }
        UploadToBuffer(gi, gradInput);
    }

    public void Fill(IGpuBuffer buffer, float value, int size)
    {
        EnsureInitialized();
        var data = new float[size];
        ArrayFillCompat(data, value);
        UploadToBuffer(data, buffer);
    }

    #endregion

    #region Random Number Generation

    public void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
    {
        EnsureInitialized();
        var rng = new Random((int)(seed & 0x7FFFFFFF));
        var data = new float[size];
        float range = max - min;
        for (int i = 0; i < size; i++)
            data[i] = (float)(rng.NextDouble() * range + min);
        UploadToBuffer(data, output);
    }

    public void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
    {
        EnsureInitialized();
        var rng = new Random((int)(seed & 0x7FFFFFFF));
        var data = new float[size];
        for (int i = 0; i < size; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            data[i] = (float)(mag * Math.Cos(2.0 * Math.PI * u2) * stdDev + mean);
            if (i + 1 < size)
                data[i + 1] = (float)(mag * Math.Sin(2.0 * Math.PI * u2) * stdDev + mean);
        }
        UploadToBuffer(data, output);
    }

    #endregion
}
