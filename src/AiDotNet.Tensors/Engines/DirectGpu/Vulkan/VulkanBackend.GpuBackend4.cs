// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 4: Optimizers, Sparse, FFT, RNN, Hyperbolic, Octonion, Quantum.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    #region Optimizer Operations

    public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient, float learningRate, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        for (int i = 0; i < size; i++)
            p[i] -= learningRate * (g[i] + weightDecay * p[i]);
        UploadToBuffer(p, param);
    }

    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var v = DownloadBuffer(velocity);
        for (int i = 0; i < size; i++)
        {
            v[i] = momentum * v[i] + g[i] + weightDecay * p[i];
            p[i] -= learningRate * v[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(v, velocity);
    }

    public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var vArr = DownloadBuffer(v);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < size; i++)
        {
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            vArr[i] = beta2 * vArr[i] + (1f - beta2) * g[i] * g[i];
            float mHat = mArr[i] / bc1, vHat = vArr[i] / bc2;
            p[i] -= learningRate * (mHat / (MathF.Sqrt(vHat) + epsilon) + weightDecay * p[i]);
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(vArr, v);
    }

    public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var vArr = DownloadBuffer(v);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < size; i++)
        {
            p[i] *= 1f - learningRate * weightDecay;
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            vArr[i] = beta2 * vArr[i] + (1f - beta2) * g[i] * g[i];
            float mHat = mArr[i] / bc1, vHat = vArr[i] / bc2;
            p[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(vArr, v);
    }

    public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var sa = DownloadBuffer(squaredAvg);
        for (int i = 0; i < size; i++)
        {
            sa[i] = rho * sa[i] + (1f - rho) * g[i] * g[i];
            p[i] -= learningRate * (g[i] / (MathF.Sqrt(sa[i]) + epsilon) + weightDecay * p[i]);
        }
        UploadToBuffer(p, param); UploadToBuffer(sa, squaredAvg);
    }

    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var ag = DownloadBuffer(accumulatedGrad);
        for (int i = 0; i < size; i++)
        {
            ag[i] += g[i] * g[i];
            p[i] -= learningRate * (g[i] / (MathF.Sqrt(ag[i]) + epsilon) + weightDecay * p[i]);
        }
        UploadToBuffer(p, param); UploadToBuffer(ag, accumulatedGrad);
    }

    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var v = DownloadBuffer(velocity);
        for (int i = 0; i < size; i++)
        {
            float vPrev = v[i];
            v[i] = momentum * v[i] - learningRate * (g[i] + weightDecay * p[i]);
            p[i] += -momentum * vPrev + (1f + momentum) * v[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(v, velocity);
    }

    public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var v = DownloadBuffer(velocity);
        float pNorm = 0, gNorm = 0;
        for (int i = 0; i < size; i++) { pNorm += p[i] * p[i]; gNorm += g[i] * g[i]; }
        pNorm = MathF.Sqrt(pNorm); gNorm = MathF.Sqrt(gNorm);
        float localLr = pNorm > 0 && gNorm > 0 ? trustCoeff * pNorm / (gNorm + weightDecay * pNorm) : 1f;
        for (int i = 0; i < size; i++)
        {
            v[i] = momentum * v[i] + localLr * (g[i] + weightDecay * p[i]);
            p[i] -= learningRate * v[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(v, velocity);
    }

    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var vArr = DownloadBuffer(v);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        var update = new float[size];
        for (int i = 0; i < size; i++)
        {
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            vArr[i] = beta2 * vArr[i] + (1f - beta2) * g[i] * g[i];
            update[i] = mArr[i] / bc1 / (MathF.Sqrt(vArr[i] / bc2) + epsilon) + weightDecay * p[i];
        }
        float pNorm = 0, uNorm = 0;
        for (int i = 0; i < size; i++) { pNorm += p[i] * p[i]; uNorm += update[i] * update[i]; }
        float ratio = uNorm > 0 ? MathF.Sqrt(pNorm) / MathF.Sqrt(uNorm) : 1f;
        for (int i = 0; i < size; i++) p[i] -= learningRate * ratio * update[i];
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(vArr, v);
    }

    public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var ag = DownloadBuffer(accumGrad); var au = DownloadBuffer(accumUpdate);
        for (int i = 0; i < size; i++)
        {
            ag[i] = rho * ag[i] + (1f - rho) * g[i] * g[i];
            float update = MathF.Sqrt(au[i] + epsilon) / MathF.Sqrt(ag[i] + epsilon) * g[i];
            au[i] = rho * au[i] + (1f - rho) * update * update;
            p[i] -= update + weightDecay * p[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(ag, accumGrad); UploadToBuffer(au, accumUpdate);
    }

    public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var vArr = DownloadBuffer(v); var vmArr = DownloadBuffer(vMax);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < size; i++)
        {
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            vArr[i] = beta2 * vArr[i] + (1f - beta2) * g[i] * g[i];
            vmArr[i] = MathF.Max(vmArr[i], vArr[i]);
            float mHat = mArr[i] / bc1;
            p[i] -= learningRate * (mHat / (MathF.Sqrt(vmArr[i] / bc2) + epsilon) + weightDecay * p[i]);
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(vArr, v); UploadToBuffer(vmArr, vMax);
    }

    public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var uArr = DownloadBuffer(u);
        float bc1 = 1f - MathF.Pow(beta1, step);
        for (int i = 0; i < size; i++)
        {
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            uArr[i] = MathF.Max(beta2 * uArr[i], MathF.Abs(g[i]));
            p[i] -= learningRate / bc1 * mArr[i] / (uArr[i] + epsilon) + weightDecay * p[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(uArr, u);
    }

    public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient); var mArr = DownloadBuffer(m);
        for (int i = 0; i < size; i++)
        {
            float update = beta1 * mArr[i] + (1f - beta1) * g[i];
            p[i] -= learningRate * (MathF.Sign(update) + weightDecay * p[i]);
            mArr[i] = beta2 * mArr[i] + (1f - beta2) * g[i];
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m);
    }

    public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var mArr = DownloadBuffer(m); var vArr = DownloadBuffer(v);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < size; i++)
        {
            mArr[i] = beta1 * mArr[i] + (1f - beta1) * g[i];
            vArr[i] = beta2 * vArr[i] + (1f - beta2) * g[i] * g[i];
            float mHat = mArr[i] / bc1, vHat = vArr[i] / bc2;
            float nesterov = beta1 * mHat + (1f - beta1) / bc1 * g[i];
            p[i] -= learningRate * (nesterov / (MathF.Sqrt(vHat) + epsilon) + weightDecay * p[i]);
        }
        UploadToBuffer(p, param); UploadToBuffer(mArr, m); UploadToBuffer(vArr, v);
    }

    public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        EnsureInitialized();
        var p = DownloadBuffer(param); var g = DownloadBuffer(gradient);
        var zArr = DownloadBuffer(z); var nArr = DownloadBuffer(n);
        for (int i = 0; i < size; i++)
        {
            float sigma = (MathF.Sqrt(nArr[i] + g[i] * g[i]) - MathF.Sqrt(nArr[i])) / learningRate;
            zArr[i] += g[i] - sigma * p[i];
            nArr[i] += g[i] * g[i];
            if (MathF.Abs(zArr[i]) <= l1Reg) p[i] = 0;
            else p[i] = -(zArr[i] - l1Reg * MathF.Sign(zArr[i])) / (l2Reg + (beta + MathF.Sqrt(nArr[i])) / learningRate);
        }
        UploadToBuffer(p, param); UploadToBuffer(zArr, z); UploadToBuffer(nArr, n);
    }

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        var inputData = DownloadBuffer(input);
        var outputBytes = new byte[size * 2];

        for (int i = 0; i < size; i++)
        {
            ushort fp16 = FloatToHalfCompat(inputData[i]);
            outputBytes[i * 2] = (byte)(fp16 & 0xFF);
            outputBytes[i * 2 + 1] = (byte)((fp16 >> 8) & 0xFF);
        }

        var floatData = new float[(size + 1) / 2];
        Buffer.BlockCopy(outputBytes, 0, floatData, 0, size * 2);
        UploadToBuffer(floatData, output);
    }

    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        var inputFloatData = DownloadBuffer(input);

        int requiredFloats = (size + 1) / 2;
        if (inputFloatData.Length < requiredFloats)
        {
            throw new ArgumentException(
                $"Input buffer too small: has {inputFloatData.Length} floats but needs at least {requiredFloats} to hold {size} FP16 values.",
                nameof(input));
        }

        int bytesToCopy = size * 2;
        var inputBytes = new byte[bytesToCopy];
        Buffer.BlockCopy(inputFloatData, 0, inputBytes, 0, bytesToCopy);

        var outputData = new float[size];
        for (int i = 0; i < size; i++)
        {
            ushort fp16 = (ushort)(inputBytes[i * 2] | (inputBytes[i * 2 + 1] << 8));
            outputData[i] = HalfToFloatCompat(fp16);
        }

        UploadToBuffer(outputData, output);
    }

    private static ushort FloatToHalfCompat(float value)
    {
        int bits = SingleToInt32BitsCompat(value);
        int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 112;
        int mantissa = bits & 0x7FFFFF;

        if (exp <= 0)
        {
            if (exp < -10)
            {
                return (ushort)sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - exp;
            mantissa >>= shift;
            return (ushort)(sign | mantissa);
        }
        else if (exp >= 31)
        {
            if ((bits & 0x7FFFFFFF) > 0x7F800000)
            {
                return 0x7FFF; // NaN
            }
            return (ushort)(sign | 0x7C00); // Infinity
        }

        return (ushort)(sign | (exp << 10) | (mantissa >> 13));
    }

    private static float HalfToFloatCompat(ushort value)
    {
        int sign = (value & 0x8000) << 16;
        int exp = (value >> 10) & 0x1F;
        int mantissa = value & 0x3FF;

        if (exp == 0)
        {
            if (mantissa == 0)
            {
                return Int32BitsToSingleCompat(sign);
            }
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= 0x3FF;
        }
        else if (exp == 31)
        {
            if (mantissa == 0)
            {
                return Int32BitsToSingleCompat(sign | 0x7F800000);
            }
            return Int32BitsToSingleCompat(sign | 0x7FC00000);
        }

        exp += 112;
        int bits = sign | (exp << 23) | (mantissa << 13);
        return Int32BitsToSingleCompat(bits);
    }

    #endregion

    #region Sparse Operations

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        EnsureInitialized();
        if (K % 4 != 0)
            throw new ArgumentException($"K ({K}) must be a multiple of 4 for 2:4 sparsity.", nameof(K));
        if (M <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "M must be positive.");
        var dense = DownloadBuffer(denseInput);
        int sparseK = K / 2;
        var sv = new float[M * sparseK];
        var sparseIndicesData = new byte[M * K / 4];

        for (int row = 0; row < M; row++)
        {
            for (int blk = 0; blk < K / 4; blk++)
            {
                int baseIdx = row * K + blk * 4;
                var vals = new (float v, int idx)[4];
                for (int j = 0; j < 4; j++) vals[j] = (MathF.Abs(dense[baseIdx + j]), j);
                Array.Sort(vals, (a, b) => b.v.CompareTo(a.v));
                int sOff = row * sparseK + blk * 2;
                sv[sOff] = dense[baseIdx + vals[0].idx];
                sv[sOff + 1] = dense[baseIdx + vals[1].idx];

                // Pack indices into byte (2 bits per index), matching IDirectGpuBackend contract
                byte packedIndices = (byte)((vals[0].idx & 0x3) | ((vals[1].idx & 0x3) << 2));
                sparseIndicesData[row * K / 4 + blk] = packedIndices;
            }
        }

        UploadToBuffer(sv, sparseValues);
        var floatIndices = new float[(M * K / 4 + 3) / 4];
        Buffer.BlockCopy(sparseIndicesData, 0, floatIndices, 0, sparseIndicesData.Length);
        UploadToBuffer(floatIndices, sparseIndices);
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        EnsureInitialized();
        if (K % 4 != 0)
            throw new ArgumentException($"K ({K}) must be a multiple of 4 for 2:4 sparsity.", nameof(K));
        if (M <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "M must be positive.");
        var sv = DownloadBuffer(sparseValues);
        var floatIndices = DownloadBuffer(sparseIndices);
        var sparseIndicesData = new byte[M * K / 4];
        Buffer.BlockCopy(floatIndices, 0, sparseIndicesData, 0, sparseIndicesData.Length);

        var dense = new float[M * K];
        int sparseK = K / 2;
        for (int row = 0; row < M; row++)
        {
            for (int blk = 0; blk < K / 4; blk++)
            {
                int sparseValueOffset = row * sparseK + blk * 2;
                byte packedIndices = sparseIndicesData[row * K / 4 + blk];

                int idx0 = packedIndices & 0x3;
                int idx1 = (packedIndices >> 2) & 0x3;

                int baseIdx = row * K + blk * 4;
                dense[baseIdx + idx0] = sv[sparseValueOffset];
                dense[baseIdx + idx1] = sv[sparseValueOffset + 1];
            }
        }
        UploadToBuffer(dense, denseOutput);
    }

    public void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        EnsureInitialized();
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        Gemm(denseA, B, C, M, N, K, alpha, beta);
        denseA.Dispose();
    }

    public IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        EnsureInitialized();
        var denseA = AllocateBuffer(M * K);
        Decompress2x4Sparse(sparseAValues, sparseAIndices, denseA, M, K);
        var result = GemmBiasRelu(denseA, B, bias, M, N, K);
        denseA.Dispose();
        return result;
    }

    public void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        EnsureInitialized();
        var vals = DownloadBuffer(csrValues);
        var cols = DownloadBuffer(csrColIndices);
        var rows = DownloadBuffer(csrRowPointers);
        var b = DownloadBuffer(denseB);
        var o = new float[M * N];
        for (int i = 0; i < M; i++)
        {
            int rowStart = SingleToInt32BitsCompat(rows[i]);
            int rowEnd = SingleToInt32BitsCompat(rows[i + 1]);
            if (rowStart < 0 || rowEnd < rowStart || rowEnd > nnz)
                throw new ArgumentOutOfRangeException(nameof(csrRowPointers), $"CSR row pointer range [{rowStart}, {rowEnd}) for row {i} is invalid (nnz={nnz}).");
            for (int idx = rowStart; idx < rowEnd; idx++)
            {
                if ((uint)idx >= (uint)vals.Length)
                    throw new ArgumentOutOfRangeException(nameof(csrValues), $"CSR element index {idx} exceeds values buffer length ({vals.Length}).");
                int col = SingleToInt32BitsCompat(cols[idx]);
                if ((uint)col >= (uint)K)
                    throw new ArgumentOutOfRangeException(nameof(csrColIndices), $"Decoded CSR column index {col} at position {idx} is out of range [0, {K}).");
                float val = vals[idx];
                for (int j = 0; j < N; j++) o[i * N + j] += val * b[col * N + j];
            }
        }
        UploadToBuffer(o, output);
    }

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);
        var o = DownloadBuffer(output);
        var bi = DownloadBuffer(bias);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) o[i * N + j] += bi[j];
        UploadToBuffer(o, output);
    }

    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices, IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var src = DownloadBuffer(sourceIndices);
        var tgt = DownloadBuffer(targetIndices);
        float[]? ev = edgeValues is not null ? DownloadBuffer(edgeValues) : null;
        var o = new float[numNodes * features];
        Array.Copy(inp, 0, o, 0, Math.Min(inp.Length, o.Length));
        for (int e = 0; e < numEdges; e++)
        {
            int s = SingleToInt32BitsCompat(src[e]);
            int t = SingleToInt32BitsCompat(tgt[e]);
            if ((uint)s >= (uint)numNodes)
                throw new ArgumentOutOfRangeException(nameof(sourceIndices), $"Decoded source index {s} at edge {e} is out of range [0, {numNodes}).");
            if ((uint)t >= (uint)numNodes)
                throw new ArgumentOutOfRangeException(nameof(targetIndices), $"Decoded target index {t} at edge {e} is out of range [0, {numNodes}).");
            float w = ev is not null ? ev[e] : 1f;
            for (int f = 0; f < features; f++) o[t * features + f] += w * inp[s * features + f];
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        var cols = DownloadBuffer(csrColIndices); var rows = DownloadBuffer(csrRowPointers);
        var inp = DownloadBuffer(input); var o = new float[M * N];
        ArrayFillCompat(o, float.MinValue);
        for (int i = 0; i < M; i++)
        {
            int rowStart = SingleToInt32BitsCompat(rows[i]);
            int rowEnd = SingleToInt32BitsCompat(rows[i + 1]);
            for (int idx = rowStart; idx < rowEnd; idx++)
            {
                int col = SingleToInt32BitsCompat(cols[idx]);
                if ((uint)col >= (uint)K)
                    throw new ArgumentOutOfRangeException(nameof(csrColIndices), $"Decoded CSR column index {col} at position {idx} is out of range [0, {K}).");
                for (int j = 0; j < N; j++) o[i * N + j] = MathF.Max(o[i * N + j], inp[col * N + j]);
            }
            if (rowStart == rowEnd) for (int j = 0; j < N; j++) o[i * N + j] = 0;
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        var cols = DownloadBuffer(csrColIndices); var rows = DownloadBuffer(csrRowPointers);
        var inp = DownloadBuffer(input); var o = new float[M * N];
        ArrayFillCompat(o, float.MaxValue);
        for (int i = 0; i < M; i++)
        {
            int rowStart = SingleToInt32BitsCompat(rows[i]);
            int rowEnd = SingleToInt32BitsCompat(rows[i + 1]);
            for (int idx = rowStart; idx < rowEnd; idx++)
            {
                int col = SingleToInt32BitsCompat(cols[idx]);
                if ((uint)col >= (uint)K)
                    throw new ArgumentOutOfRangeException(nameof(csrColIndices), $"Decoded CSR column index {col} at position {idx} is out of range [0, {K}).");
                for (int j = 0; j < N; j++) o[i * N + j] = MathF.Min(o[i * N + j], inp[col * N + j]);
            }
            if (rowStart == rowEnd) for (int j = 0; j < N; j++) o[i * N + j] = 0;
        }
        UploadToBuffer(o, output);
    }

    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        EnsureInitialized();
        var cols = DownloadBuffer(csrColIndices); var rows = DownloadBuffer(csrRowPointers);
        var inp = DownloadBuffer(input); var o = new float[M * N];
        for (int i = 0; i < M; i++)
        {
            int rowStart = SingleToInt32BitsCompat(rows[i]);
            int rowEnd = SingleToInt32BitsCompat(rows[i + 1]);
            int count = rowEnd - rowStart;
            if (count == 0) continue;
            for (int j = 0; j < N; j++)
            {
                float mean = 0;
                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    int col = SingleToInt32BitsCompat(cols[idx]);
                    if ((uint)col >= (uint)K)
                        throw new ArgumentOutOfRangeException(nameof(csrColIndices), $"Decoded CSR column index {col} at position {idx} is out of range [0, {K}).");
                    mean += inp[col * N + j];
                }
                mean /= count;
                float var_ = 0;
                for (int idx = rowStart; idx < rowEnd; idx++)
                {
                    int col = SingleToInt32BitsCompat(cols[idx]);
                    if ((uint)col >= (uint)K)
                        throw new ArgumentOutOfRangeException(nameof(csrColIndices), $"Decoded CSR column index {col} at position {idx} is out of range [0, {K}).");
                    float d = inp[col * N + j] - mean;
                    var_ += d * d;
                }
                o[i * N + j] = MathF.Sqrt(var_ / count + epsilon);
            }
        }
        UploadToBuffer(o, output);
    }

    #endregion

    #region Specialized Layer Operations

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input); var c = DownloadBuffer(centers); var eps = DownloadBuffer(epsilons);
        var o = new float[batchSize * numCenters];
        for (int b = 0; b < batchSize; b++)
            for (int nc = 0; nc < numCenters; nc++)
            {
                float dist = 0;
                for (int d = 0; d < inputDim; d++) { float diff = inp[b * inputDim + d] - c[nc * inputDim + d]; dist += diff * diff; }
                o[b * numCenters + nc] = MathF.Exp(-eps[nc] * dist);
            }
        UploadToBuffer(o, output);
    }

    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        EnsureInitialized();
        var w = DownloadBuffer(weights); var preT = DownloadBuffer(preTrace); var postT = DownloadBuffer(postTrace);
        var preS = DownloadBuffer(preSpike); var postS = DownloadBuffer(postSpike);
        for (int i = 0; i < numPre; i++)
            for (int j = 0; j < numPost; j++)
            {
                int idx = i * numPost + j;
                if (postS[j] > 0.5f) w[idx] += ltpRate * preT[i];
                if (preS[i] > 0.5f) w[idx] -= ltdRate * postT[j];
                w[idx] = MathF.Max(minWeight, MathF.Min(maxWeight, w[idx]));
            }
        UploadToBuffer(w, weights);
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        EnsureInitialized();
        var t = DownloadBuffer(traces); var s = DownloadBuffer(spikes); var inp = DownloadBuffer(input);
        for (int i = 0; i < size; i++)
        {
            t[i] = decay * t[i];
            if (s[i] > 0.5f) t[i] += 1f;
        }
        UploadToBuffer(t, traces);
    }

    #endregion

    #region Hyperbolic Geometry Operations

    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        EnsureInitialized();
        if (curvature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(curvature), curvature, "Curvature must be positive for Poincare ball model.");
        var pc = new uint[] { (uint)batchSize, (uint)dim, FloatBits(curvature), FloatBits(epsilon) };
        GlslUnaryOp(VulkanGlslKernels.PoincareProject, input, output, batchSize, pc, 4 * sizeof(uint));
    }

    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        if (curvature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(curvature), curvature, "Curvature must be positive for Mobius addition.");
        var pc = new uint[] { (uint)batchSize, (uint)dim, FloatBits(curvature) };
        GlslBinaryOp(VulkanGlslKernels.MobiusAdd, x, y, output, batchSize, pc, 3 * sizeof(uint));
    }

    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        if (curvature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(curvature), curvature, "Curvature must be positive for Poincare exponential map.");
        var pc = new uint[] { (uint)batchSize, (uint)dim, FloatBits(curvature) };
        GlslBinaryOp(VulkanGlslKernels.PoincareExpMap, basePoint, tangentVec, output, batchSize, pc, 3 * sizeof(uint));
    }

    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        if (curvature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(curvature), curvature, "Curvature must be positive for Poincare distance.");
        var pc = new uint[] { (uint)batchSize, (uint)dim, FloatBits(curvature) };
        GlslBinaryOp(VulkanGlslKernels.PoincareDistance, x, y, output, batchSize, pc, 3 * sizeof(uint));
    }

    public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        // Fused GLSL kernel: matmul + bias in one pass
        int total = batchSize * outputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures, FloatBits(curvature), FloatBits(epsilon) };
        GlslQuadOp(VulkanGlslKernels.HyperbolicLinearForward, input, weights, biases, output, total, pc, 5 * sizeof(uint));
        // Apply Poincaré projection as a second pass (needs full output vector for norm)
        PoincareProject(output, output, batchSize, outputFeatures, curvature, epsilon);
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradInput = gradOutput * weights^T (Euclidean approximation)
        // Compute via two GEMMs: create transposed weight buffer then multiply
        EnsureInitialized();
        var wt = DownloadBuffer(weights);
        var wtT = new float[inputFeatures * outputFeatures];
        for (int o = 0; o < outputFeatures; o++)
            for (int i = 0; i < inputFeatures; i++)
                wtT[i * outputFeatures + o] = wt[o * inputFeatures + i];
        var wtTBuf = AllocateBuffer(inputFeatures * outputFeatures);
        UploadToBuffer(wtT, wtTBuf);
        Gemm(gradOutput, wtTBuf, gradInput, batchSize, inputFeatures, outputFeatures);
        wtTBuf.Dispose();
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradWeights = gradOutput^T * input
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var goT = new float[outputFeatures * batchSize];
        for (int b = 0; b < batchSize; b++)
            for (int o = 0; o < outputFeatures; o++)
                goT[o * batchSize + b] = go[b * outputFeatures + o];
        var goTBuf = AllocateBuffer(outputFeatures * batchSize);
        UploadToBuffer(goT, goTBuf);
        Gemm(goTBuf, input, gradWeights, outputFeatures, inputFeatures, batchSize);
        goTBuf.Dispose();
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradBiases[o] = sum_b gradOutput[b,o]
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var gb = new float[outputFeatures];
        for (int o = 0; o < outputFeatures; o++)
        {
            float sum = 0;
            for (int b = 0; b < batchSize; b++)
                sum += go[b * outputFeatures + o];
            gb[o] = sum;
        }
        UploadToBuffer(gb, gradBiases);
    }

    #endregion

    #region Octonion Algebra Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        EnsureInitialized();
        var pc = new uint[] { (uint)count };
        GlslBinaryOp(VulkanGlslKernels.OctonionMultiply, a, b, output, count, pc, sizeof(uint));
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        => Add(a, b, output, count * 8);

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        EnsureInitialized();
        int totalPairs = batchSize * outputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures };
        GlslQuadOp(VulkanGlslKernels.OctonionLinearForward, input, weights, biases, output, totalPairs, pc, 3 * sizeof(uint));
    }

    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        EnsureInitialized();
        int totalPairs = batchSize * inputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures };
        GlslBinaryOp(VulkanGlslKernels.OctonionLinearBackwardInput, gradOutput, weights, gradInput, totalPairs, pc, 3 * sizeof(uint));
    }

    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        EnsureInitialized();
        int totalPairs = outputFeatures * inputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures };
        GlslBinaryOp(VulkanGlslKernels.OctonionLinearBackwardWeights, gradOutput, input, gradWeights, totalPairs, pc, 3 * sizeof(uint));
    }

    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures)
    {
        EnsureInitialized();
        var pc = new uint[] { (uint)batchSize, (uint)outputFeatures };
        GlslUnaryOp(VulkanGlslKernels.OctonionLinearBackwardBiases, gradOutput, gradBiases, outputFeatures, pc, 2 * sizeof(uint));
    }

    #endregion

    #region Fused Kernel Operations

    public void HyperbolicLinearForwardFused(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        // Fused: matmul+bias in one GLSL pass, then projection in second GLSL pass
        // This eliminates the intermediate buffer allocation of the non-fused 3-pass version
        int total = batchSize * outputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures, FloatBits(curvature), FloatBits(epsilon) };
        GlslQuadOp(VulkanGlslKernels.HyperbolicLinearForward, input, weights, biases, output, total, pc, 5 * sizeof(uint));
        // Project in-place (second pass — reads and writes same buffer)
        var projPc = new uint[] { (uint)batchSize, (uint)outputFeatures, FloatBits(curvature), FloatBits(epsilon) };
        GlslUnaryOp(VulkanGlslKernels.PoincareProject, output, output, batchSize, projPc, 4 * sizeof(uint));
    }

    public void OctonionLinearForwardFusedReLU(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        EnsureInitialized();
        int totalPairs = batchSize * outputFeatures;
        var pc = new uint[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures };
        GlslQuadOp(VulkanGlslKernels.OctonionLinearForwardFusedReLU, input, weights, biases, output, totalPairs, pc, 3 * sizeof(uint));
    }

    #endregion

    #region Complex Tensor Operations

    public void ComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int numPairs)
    {
        var pc = new uint[] { (uint)numPairs };
        GlslBinaryOp(VulkanGlslKernels.ComplexMultiply, a, b, output, numPairs, pc, sizeof(uint));
    }

    public void ComplexConjugate(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        var pc = new uint[] { (uint)numPairs };
        GlslUnaryOp(VulkanGlslKernels.ComplexConjugate, input, output, numPairs, pc, sizeof(uint));
    }

    public void ComplexMagnitude(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        var pc = new uint[] { (uint)numPairs };
        GlslUnaryOp(VulkanGlslKernels.ComplexMagnitude, input, output, numPairs, pc, sizeof(uint));
    }

    #endregion

    #region Quantum Computing Operations

    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var re = DownloadBuffer(realPart); var im = DownloadBuffer(imagPart);
        var prob = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < stateSize; s++)
            {
                int idx = b * stateSize + s;
                prob[idx] = re[idx] * re[idx] + im[idx] * im[idx];
            }
        UploadToBuffer(prob, probabilities);
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var prob = DownloadBuffer(probabilities);
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * stateSize; float sum = 0;
            for (int s = 0; s < stateSize; s++) sum += prob[off + s];
            if (sum > 0) for (int s = 0; s < stateSize; s++) prob[off + s] /= sum;
        }
        UploadToBuffer(prob, probabilities);
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        EnsureInitialized();
        var mr = DownloadBuffer(matReal); var mi = DownloadBuffer(matImag);
        var vr = DownloadBuffer(vecReal); var vi = DownloadBuffer(vecImag);
        var or_ = new float[batchSize * dim]; var oi = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int vOff = b * dim, mOff = b * dim * dim;
            for (int i = 0; i < dim; i++)
            {
                float sumR = 0, sumI = 0;
                for (int j = 0; j < dim; j++)
                {
                    int mIdx = mOff + i * dim + j;
                    sumR += mr[mIdx] * vr[vOff + j] - mi[mIdx] * vi[vOff + j];
                    sumI += mr[mIdx] * vi[vOff + j] + mi[mIdx] * vr[vOff + j];
                }
                or_[vOff + i] = sumR; oi[vOff + i] = sumI;
            }
        }
        UploadToBuffer(or_, outReal); UploadToBuffer(oi, outImag);
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        EnsureInitialized();
        var sr = DownloadBuffer(stateReal); var si = DownloadBuffer(stateImag);
        var ang = DownloadBuffer(angles);
        int stateSize = 1 << numQubits;
        var or_ = new float[batchSize * stateSize]; var oi = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * stateSize;
            float cosA = MathF.Cos(ang[b % ang.Length]);
            float sinA = MathF.Sin(ang[b % ang.Length]);
            for (int s = 0; s < stateSize; s++)
            {
                or_[off + s] = cosA * sr[off + s] - sinA * si[off + s];
                oi[off + s] = sinA * sr[off + s] + cosA * si[off + s];
            }
        }
        UploadToBuffer(or_, outReal); UploadToBuffer(oi, outImag);
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var o = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * stateSize; float sum = 0;
            for (int s = 0; s < stateSize; s++) { o[off + s] = inp[off + s] * inp[off + s]; sum += o[off + s]; }
            if (sum > 0) for (int s = 0; s < stateSize; s++) o[off + s] /= sum;
        }
        UploadToBuffer(o, output);
    }

    #endregion

    #region FFT and Signal Processing

    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        EnsureInitialized();
        var ir = DownloadBuffer(inputReal); var ii = DownloadBuffer(inputImag);
        var or_ = new float[n]; var oi = new float[n];

        if (n > 0 && (n & (n - 1)) == 0)
        {
            // Cooley-Tukey radix-2 FFT for power-of-2 sizes: O(n log n)
            CooleyTukeyFFT(ir, ii, or_, oi, n, inverse);
        }
        else
        {
            // Fallback to naive DFT for non-power-of-2 sizes: O(n^2)
            NaiveDFT(ir, ii, or_, oi, n, inverse);
        }

        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    private static void CooleyTukeyFFT(float[] inR, float[] inI, float[] outR, float[] outI, int n, bool inverse)
    {
        // Bit-reversal permutation
        Array.Copy(inR, outR, n);
        Array.Copy(inI, outI, n);

        int logN = 0;
        for (int tmp = n; tmp > 1; tmp >>= 1) logN++;

        for (int i = 0; i < n; i++)
        {
            int rev = 0;
            for (int bit = 0; bit < logN; bit++)
                if ((i & (1 << bit)) != 0)
                    rev |= 1 << (logN - 1 - bit);
            if (rev > i)
            {
                (outR[i], outR[rev]) = (outR[rev], outR[i]);
                (outI[i], outI[rev]) = (outI[rev], outI[i]);
            }
        }

        // Butterfly stages
        float sign = inverse ? 1f : -1f;
        for (int size = 2; size <= n; size <<= 1)
        {
            int halfSize = size / 2;
            float wAngle = sign * 2f * MathF.PI / size;
            float wR = MathF.Cos(wAngle);
            float wI = MathF.Sin(wAngle);

            for (int start = 0; start < n; start += size)
            {
                float twR = 1f, twI = 0f;
                for (int j = 0; j < halfSize; j++)
                {
                    int even = start + j;
                    int odd = start + j + halfSize;
                    float tR = twR * outR[odd] - twI * outI[odd];
                    float tI = twR * outI[odd] + twI * outR[odd];
                    outR[odd] = outR[even] - tR;
                    outI[odd] = outI[even] - tI;
                    outR[even] += tR;
                    outI[even] += tI;
                    float newTwR = twR * wR - twI * wI;
                    twI = twR * wI + twI * wR;
                    twR = newTwR;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++) { outR[i] /= n; outI[i] /= n; }
        }
    }

    private static void NaiveDFT(float[] inR, float[] inI, float[] outR, float[] outI, int n, bool inverse)
    {
        float sign = inverse ? 1f : -1f;
        for (int k = 0; k < n; k++)
        {
            float sumR = 0, sumI = 0;
            for (int j = 0; j < n; j++)
            {
                float angle = sign * 2f * MathF.PI * k * j / n;
                sumR += inR[j] * MathF.Cos(angle) - inI[j] * MathF.Sin(angle);
                sumI += inR[j] * MathF.Sin(angle) + inI[j] * MathF.Cos(angle);
            }
            if (inverse) { sumR /= n; sumI /= n; }
            outR[k] = sumR; outI[k] = sumI;
        }
    }

    public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        var zeroImag = AllocateBuffer(n);
        Fill(zeroImag, 0f, n);
        FFT(input, zeroImag, outputReal, outputImag, n, false);
        zeroImag.Dispose();
    }

    public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        var outImag = AllocateBuffer(n);
        FFT(inputReal, inputImag, output, outImag, n, true);
        outImag.Dispose();
    }

    public void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        EnsureInitialized();
        var ir = DownloadBuffer(inputReal); var ii = DownloadBuffer(inputImag);
        var or_ = new float[batch * n]; var oi = new float[batch * n];
        bool isPow2 = n > 0 && (n & (n - 1)) == 0;

        for (int b = 0; b < batch; b++)
        {
            int off = b * n;
            var rowR = new float[n]; var rowI = new float[n];
            var resR = new float[n]; var resI = new float[n];
            Array.Copy(ir, off, rowR, 0, n);
            Array.Copy(ii, off, rowI, 0, n);

            if (isPow2)
                CooleyTukeyFFT(rowR, rowI, resR, resI, n, inverse);
            else
                NaiveDFT(rowR, rowI, resR, resI, n, inverse);

            Array.Copy(resR, 0, or_, off, n);
            Array.Copy(resI, 0, oi, off, n);
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        EnsureInitialized();

        // Step 1: FFT along rows (treating height rows of width elements)
        BatchedFFT(inputReal, inputImag, outputReal, outputImag, height, width, inverse);

        // Step 2: FFT along columns (transpose, FFT rows, transpose back)
        var rowR = DownloadBuffer(outputReal); var rowI = DownloadBuffer(outputImag);

        // Transpose: height x width -> width x height
        var trR = new float[width * height]; var trI = new float[width * height];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
            {
                trR[c * height + r] = rowR[r * width + c];
                trI[c * height + r] = rowI[r * width + c];
            }

        // FFT along transposed rows (= original columns)
        bool isPow2 = height > 0 && (height & (height - 1)) == 0;
        var colR = new float[width * height]; var colI = new float[width * height];
        for (int c = 0; c < width; c++)
        {
            var inR = new float[height]; var inI = new float[height];
            var outR = new float[height]; var outI = new float[height];
            Array.Copy(trR, c * height, inR, 0, height);
            Array.Copy(trI, c * height, inI, 0, height);

            if (isPow2)
                CooleyTukeyFFT(inR, inI, outR, outI, height, inverse);
            else
                NaiveDFT(inR, inI, outR, outI, height, inverse);

            Array.Copy(outR, 0, colR, c * height, height);
            Array.Copy(outI, 0, colI, c * height, height);
        }

        // Transpose back: width x height -> height x width
        var finalR = new float[height * width]; var finalI = new float[height * width];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
            {
                finalR[r * width + c] = colR[c * height + r];
                finalI[r * width + c] = colI[c * height + r];
            }

        UploadToBuffer(finalR, outputReal); UploadToBuffer(finalI, outputImag);
    }

    /// <inheritdoc/>
    public void BatchedFFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int height, int width, bool inverse)
    {
        if (batch <= 0 || height <= 0 || width <= 0) return;
        int sliceSize = height * width;

        if (batch == 1)
        {
            // Single image — direct FFT2D, no temp buffers needed
            FFT2D(inputReal, inputImag, outputReal, outputImag, height, width, inverse);
            return;
        }

        // Allocate temp slice-sized buffers ONCE (reused across all slices)
        var tempInR = AllocateBuffer(sliceSize);
        var tempInI = AllocateBuffer(sliceSize);
        var tempOutR = AllocateBuffer(sliceSize);
        var tempOutI = AllocateBuffer(sliceSize);
        try
        {
            for (int b = 0; b < batch; b++)
            {
                int off = b * sliceSize;
                // GPU-to-GPU copy: extract slice from batched buffer
                Copy(inputReal, off, tempInR, 0, sliceSize);
                Copy(inputImag, off, tempInI, 0, sliceSize);
                // FFT2D on GPU (fully GPU-resident, no CPU round-trip)
                FFT2D(tempInR, tempInI, tempOutR, tempOutI, height, width, inverse);
                // GPU-to-GPU copy: write result back to batched output
                Copy(tempOutR, 0, outputReal, off, sliceSize);
                Copy(tempOutI, 0, outputImag, off, sliceSize);
            }
        }
        finally
        {
            tempInR.Dispose(); tempInI.Dispose();
            tempOutR.Dispose(); tempOutI.Dispose();
        }
    }

    public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
        => CpuBinary(input, window, output, n, (a, w) => a * w);

    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
        => CpuBinary(real, imag, magnitude, n, (r, i) => MathF.Sqrt(r * r + i * i));

    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
        => CpuBinary(real, imag, phase, n, (r, i) => MathF.Atan2(i, r));

    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        EnsureInitialized();
        var mag = DownloadBuffer(magnitude); var ph = DownloadBuffer(phase);
        var r = new float[n]; var im = new float[n];
        for (int i = 0; i < n; i++) { r[i] = mag[i] * MathF.Cos(ph[i]); im[i] = mag[i] * MathF.Sin(ph[i]); }
        UploadToBuffer(r, real); UploadToBuffer(im, imag);
    }

    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        EnsureInitialized();
        var ps = DownloadBuffer(powerSpec); var fb = DownloadBuffer(filterbank);
        var ms = new float[numFrames * nMels];
        for (int f = 0; f < numFrames; f++)
            for (int m = 0; m < nMels; m++)
            {
                float sum = 0;
                for (int k = 0; k < numFreqs; k++) sum += ps[f * numFreqs + k] * fb[m * numFreqs + k];
                ms[f * nMels + m] = sum;
            }
        UploadToBuffer(ms, melSpec);
    }

    public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
        => CpuUnary(power, db, n, v => MathF.Max(minDb, 10f * MathF.Log10(MathF.Max(1e-10f, v / refValue))));

    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
        => CpuUnary(db, power, n, v => refValue * MathF.Pow(10f, v / 10f));

    #endregion

    #region RNN (LSTM/GRU) Operations

    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var h = DownloadBuffer(hInit); var c = DownloadBuffer(cInit);
        var wIh = DownloadBuffer(weightsIh); var wHh = DownloadBuffer(weightsHh);
        var bIh = DownloadBuffer(biasIh); var bHh = DownloadBuffer(biasHh);
        int gateSize = 4 * hiddenSize;
        var outp = new float[seqLen * batch * hiddenSize];
        var aH = new float[(seqLen + 1) * batch * hiddenSize];
        var aC = new float[(seqLen + 1) * batch * hiddenSize];
        var cg = new float[seqLen * batch * gateSize];
        Array.Copy(h, 0, aH, 0, batch * hiddenSize);
        Array.Copy(c, 0, aC, 0, batch * hiddenSize);

        for (int t = 0; t < seqLen; t++)
        {
            int hOff = t * batch * hiddenSize;
            int hNextOff = (t + 1) * batch * hiddenSize;
            for (int b = 0; b < batch; b++)
            {
                var gates = new float[gateSize];
                for (int g = 0; g < gateSize; g++)
                {
                    float sum = bIh[g] + bHh[g];
                    for (int k = 0; k < inputSize; k++) sum += inp[t * batch * inputSize + b * inputSize + k] * wIh[g * inputSize + k];
                    for (int k = 0; k < hiddenSize; k++) sum += aH[hOff + b * hiddenSize + k] * wHh[g * hiddenSize + k];
                    gates[g] = sum;
                }
                Array.Copy(gates, 0, cg, t * batch * gateSize + b * gateSize, gateSize);
                for (int i = 0; i < hiddenSize; i++)
                {
                    float ig = 1f / (1f + MathF.Exp(-gates[i]));
                    float fg = 1f / (1f + MathF.Exp(-gates[hiddenSize + i]));
                    float gg = MathF.Tanh(gates[2 * hiddenSize + i]);
                    float og = 1f / (1f + MathF.Exp(-gates[3 * hiddenSize + i]));
                    float ct = fg * aC[hOff + b * hiddenSize + i] + ig * gg;
                    float ht = og * MathF.Tanh(ct);
                    aC[hNextOff + b * hiddenSize + i] = ct;
                    aH[hNextOff + b * hiddenSize + i] = ht;
                    outp[t * batch * hiddenSize + b * hiddenSize + i] = ht;
                }
            }
        }

        UploadToBuffer(outp, output);
        var hf = new float[batch * hiddenSize]; var cf = new float[batch * hiddenSize];
        Array.Copy(aH, seqLen * batch * hiddenSize, hf, 0, batch * hiddenSize);
        Array.Copy(aC, seqLen * batch * hiddenSize, cf, 0, batch * hiddenSize);
        UploadToBuffer(hf, hFinal); UploadToBuffer(cf, cFinal);
        UploadToBuffer(aH, allH); UploadToBuffer(aC, allC); UploadToBuffer(cg, cacheGates);
    }

    public void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var aH = DownloadBuffer(allH);
        var aC = DownloadBuffer(allC);
        var cg = DownloadBuffer(cacheGates);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var inp = DownloadBuffer(input);
        int gateSize = 4 * hiddenSize;

        var gI = new float[seqLen * batch * inputSize];
        var gWih = new float[gateSize * inputSize];
        var gWhh = new float[gateSize * hiddenSize];
        var gBih = new float[gateSize];
        var gBhh = new float[gateSize];
        var dH = new float[batch * hiddenSize];
        var dC = new float[batch * hiddenSize];

        for (int t = seqLen - 1; t >= 0; t--)
        {
            int hOff = t * batch * hiddenSize;
            int hNextOff = (t + 1) * batch * hiddenSize;
            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < hiddenSize; i++)
                    dH[b * hiddenSize + i] += go[t * batch * hiddenSize + b * hiddenSize + i];

                int cgOff = t * batch * gateSize + b * gateSize;
                for (int i = 0; i < hiddenSize; i++)
                {
                    float ig = 1f / (1f + MathF.Exp(-cg[cgOff + i]));
                    float fg = 1f / (1f + MathF.Exp(-cg[cgOff + hiddenSize + i]));
                    float gg = MathF.Tanh(cg[cgOff + 2 * hiddenSize + i]);
                    float og = 1f / (1f + MathF.Exp(-cg[cgOff + 3 * hiddenSize + i]));
                    float ct = aC[hNextOff + b * hiddenSize + i];
                    float tanhCt = MathF.Tanh(ct);

                    float dOg = dH[b * hiddenSize + i] * tanhCt * og * (1f - og);
                    dC[b * hiddenSize + i] += dH[b * hiddenSize + i] * og * (1f - tanhCt * tanhCt);
                    float dIg = dC[b * hiddenSize + i] * gg * ig * (1f - ig);
                    float dFg = dC[b * hiddenSize + i] * aC[hOff + b * hiddenSize + i] * fg * (1f - fg);
                    float dGg = dC[b * hiddenSize + i] * ig * (1f - gg * gg);

                    float[] dGates = { dIg, dFg, dGg, dOg };
                    for (int gi = 0; gi < 4; gi++)
                    {
                        int gIdx = gi * hiddenSize + i;
                        gBih[gIdx] += dGates[gi];
                        gBhh[gIdx] += dGates[gi];
                        for (int k = 0; k < inputSize; k++)
                            gWih[gIdx * inputSize + k] += dGates[gi] * inp[t * batch * inputSize + b * inputSize + k];
                        for (int k = 0; k < hiddenSize; k++)
                            gWhh[gIdx * hiddenSize + k] += dGates[gi] * aH[hOff + b * hiddenSize + k];
                    }

                    for (int k = 0; k < inputSize; k++)
                        for (int gi = 0; gi < 4; gi++)
                            gI[t * batch * inputSize + b * inputSize + k] += dGates[gi] * wIh[(gi * hiddenSize + i) * inputSize + k];

                    dC[b * hiddenSize + i] *= fg;
                }

                var newDh = new float[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                {
                    float ig = 1f / (1f + MathF.Exp(-cg[cgOff + i]));
                    float fg = 1f / (1f + MathF.Exp(-cg[cgOff + hiddenSize + i]));
                    float gg = MathF.Tanh(cg[cgOff + 2 * hiddenSize + i]);
                    float og = 1f / (1f + MathF.Exp(-cg[cgOff + 3 * hiddenSize + i]));
                    float ct = aC[hNextOff + b * hiddenSize + i];
                    float tanhCt = MathF.Tanh(ct);
                    float localDh = dH[b * hiddenSize + i];
                    float localDc = dC[b * hiddenSize + i];

                    float dOgPre = localDh * tanhCt * og * (1f - og);
                    float dIgPre = localDc * gg * ig * (1f - ig);
                    float dFgPre = localDc * aC[hOff + b * hiddenSize + i] * fg * (1f - fg);
                    float dGgPre = localDc * ig * (1f - gg * gg);

                    for (int k = 0; k < hiddenSize; k++)
                    {
                        newDh[k] += dIgPre * wHh[i * hiddenSize + k];
                        newDh[k] += dFgPre * wHh[(hiddenSize + i) * hiddenSize + k];
                        newDh[k] += dGgPre * wHh[(2 * hiddenSize + i) * hiddenSize + k];
                        newDh[k] += dOgPre * wHh[(3 * hiddenSize + i) * hiddenSize + k];
                    }
                }
                Array.Copy(newDh, 0, dH, b * hiddenSize, hiddenSize);
            }
        }

        UploadToBuffer(gI, gradInput);
        UploadToBuffer(dH, gradHInit);
        UploadToBuffer(dC, gradCInit);
        UploadToBuffer(gWih, gradWeightsIh);
        UploadToBuffer(gWhh, gradWeightsHh);
        UploadToBuffer(gBih, gradBiasIh);
        UploadToBuffer(gBhh, gradBiasHh);
    }

    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var inp = DownloadBuffer(input);
        var h = DownloadBuffer(hInit);
        var wIh = DownloadBuffer(weightsIh); var wHh = DownloadBuffer(weightsHh);
        var bIh = DownloadBuffer(biasIh); var bHh = DownloadBuffer(biasHh);
        int gateSize = 3 * hiddenSize;
        var outp = new float[seqLen * batch * hiddenSize];
        var aH = new float[(seqLen + 1) * batch * hiddenSize];
        var cg = new float[seqLen * batch * gateSize];
        Array.Copy(h, 0, aH, 0, batch * hiddenSize);

        for (int t = 0; t < seqLen; t++)
        {
            int hOff = t * batch * hiddenSize;
            int hNextOff = (t + 1) * batch * hiddenSize;
            for (int b = 0; b < batch; b++)
            {
                // Step 1: Compute r (reset) and z (update) gates
                var gates = new float[gateSize];
                for (int g = 0; g < 2 * hiddenSize; g++)
                {
                    float sum = bIh[g] + bHh[g];
                    for (int k = 0; k < inputSize; k++) sum += inp[t * batch * inputSize + b * inputSize + k] * wIh[g * inputSize + k];
                    for (int k = 0; k < hiddenSize; k++) sum += aH[hOff + b * hiddenSize + k] * wHh[g * hiddenSize + k];
                    gates[g] = sum;
                }

                // Apply sigmoid to r and z gates
                for (int i = 0; i < hiddenSize; i++)
                {
                    gates[i] = 1f / (1f + MathF.Exp(-gates[i]));                 // r gate
                    gates[hiddenSize + i] = 1f / (1f + MathF.Exp(-gates[hiddenSize + i])); // z gate
                }

                // Step 2: Compute candidate n using reset-gated hidden state (r * hPrev)
                for (int i = 0; i < hiddenSize; i++)
                {
                    float sum = bIh[2 * hiddenSize + i] + bHh[2 * hiddenSize + i];
                    for (int k = 0; k < inputSize; k++) sum += inp[t * batch * inputSize + b * inputSize + k] * wIh[(2 * hiddenSize + i) * inputSize + k];
                    // Apply reset gate: use r * hPrev instead of raw hPrev
                    for (int k = 0; k < hiddenSize; k++) sum += gates[k] * aH[hOff + b * hiddenSize + k] * wHh[(2 * hiddenSize + i) * hiddenSize + k];
                    gates[2 * hiddenSize + i] = MathF.Tanh(sum);
                }

                // Cache all gates for backward pass
                Array.Copy(gates, 0, cg, t * batch * gateSize + b * gateSize, gateSize);

                // Step 3: Compute new hidden state: ht = (1-z)*n + z*hPrev
                for (int i = 0; i < hiddenSize; i++)
                {
                    float r = gates[i];
                    float z = gates[hiddenSize + i];
                    float n_ = gates[2 * hiddenSize + i];
                    float ht = (1f - z) * n_ + z * aH[hOff + b * hiddenSize + i];
                    aH[hNextOff + b * hiddenSize + i] = ht;
                    outp[t * batch * hiddenSize + b * hiddenSize + i] = ht;
                }
            }
        }

        UploadToBuffer(outp, output);
        var hf = new float[batch * hiddenSize];
        Array.Copy(aH, seqLen * batch * hiddenSize, hf, 0, batch * hiddenSize);
        UploadToBuffer(hf, hFinal); UploadToBuffer(aH, allH); UploadToBuffer(cg, cacheGates);
    }

    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var aH = DownloadBuffer(allH);
        var cg = DownloadBuffer(cacheGates);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var inp = DownloadBuffer(input);
        int gateSize = 3 * hiddenSize;

        var gI = new float[seqLen * batch * inputSize];
        var gWih = new float[gateSize * inputSize];
        var gWhh = new float[gateSize * hiddenSize];
        var gBih = new float[gateSize];
        var gBhh = new float[gateSize];
        var dH = new float[batch * hiddenSize];

        for (int t = seqLen - 1; t >= 0; t--)
        {
            int hOff = t * batch * hiddenSize;
            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < hiddenSize; i++)
                    dH[b * hiddenSize + i] += go[t * batch * hiddenSize + b * hiddenSize + i];

                int cgOff = t * batch * gateSize + b * gateSize;
                var newDh = new float[hiddenSize];

                // Cached gates are already activated (sigmoid for r/z, tanh for n)
                // so we use them directly without re-applying activations.
                // First pass: compute dN for each hidden unit (needed for dR computation)
                var dNArr = new float[hiddenSize];
                var dZArr = new float[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                {
                    float r = cg[cgOff + i];               // already sigmoid-activated
                    float z = cg[cgOff + hiddenSize + i];   // already sigmoid-activated
                    float n_ = cg[cgOff + 2 * hiddenSize + i]; // already tanh-activated
                    float hPrev = aH[hOff + b * hiddenSize + i];
                    float localDh = dH[b * hiddenSize + i];

                    // ht = (1-z)*n + z*hPrev
                    // dL/dn (pre-tanh) = dL/dht * (1-z) * (1-n^2)
                    dNArr[i] = localDh * (1f - z) * (1f - n_ * n_);
                    // dL/dz (pre-sigmoid) = dL/dht * (hPrev - n) * z * (1-z)
                    dZArr[i] = localDh * (hPrev - n_) * z * (1f - z);
                }

                // Second pass: compute dR using chain rule through n's dependence on r
                // n = tanh(W_in*x + b_in + r*(W_hn*hPrev + b_hn))
                // dn/dr_i = sum_j(dN_j * W_hn[j,i] * hPrev_i) -- but the inner sum couples through r*hPrev
                // For each hidden unit i: dR_i = r_i*(1-r_i) * sum_j(dN_j * hPrev_i * W_hn[j,i])
                for (int i = 0; i < hiddenSize; i++)
                {
                    float r = cg[cgOff + i];
                    float hPrev = aH[hOff + b * hiddenSize + i];

                    float dRPre = 0f;
                    for (int j = 0; j < hiddenSize; j++)
                        dRPre += dNArr[j] * wHh[(2 * hiddenSize + j) * hiddenSize + i] * hPrev;
                    float dR = dRPre * r * (1f - r);

                    float dZ = dZArr[i];
                    float dN = dNArr[i];
                    float localDh = dH[b * hiddenSize + i];

                    float[] dGates = { dR, dZ, dN };
                    for (int gi = 0; gi < 3; gi++)
                    {
                        int gIdx = gi * hiddenSize + i;
                        gBih[gIdx] += dGates[gi];
                        gBhh[gIdx] += dGates[gi];
                        for (int k = 0; k < inputSize; k++)
                            gWih[gIdx * inputSize + k] += dGates[gi] * inp[t * batch * inputSize + b * inputSize + k];
                        for (int k = 0; k < hiddenSize; k++)
                            gWhh[gIdx * hiddenSize + k] += dGates[gi] * aH[hOff + b * hiddenSize + k];
                    }

                    for (int k = 0; k < inputSize; k++)
                        for (int gi = 0; gi < 3; gi++)
                            gI[t * batch * inputSize + b * inputSize + k] += dGates[gi] * wIh[(gi * hiddenSize + i) * inputSize + k];

                    // gradient to previous hidden state
                    newDh[i] = localDh * cg[cgOff + hiddenSize + i]; // z
                    for (int k = 0; k < hiddenSize; k++)
                    {
                        newDh[k] += dR * wHh[i * hiddenSize + k];
                        newDh[k] += dZ * wHh[(hiddenSize + i) * hiddenSize + k];
                        newDh[k] += dN * wHh[(2 * hiddenSize + i) * hiddenSize + k];
                    }
                }
                Array.Copy(newDh, 0, dH, b * hiddenSize, hiddenSize);
            }
        }

        UploadToBuffer(gI, gradInput);
        UploadToBuffer(dH, gradHInit);
        UploadToBuffer(dH, dHBuffer);
        UploadToBuffer(gWih, gradWeightsIh);
        UploadToBuffer(gWhh, gradWeightsHh);
        UploadToBuffer(gBih, gradBiasIh);
        UploadToBuffer(gBhh, gradBiasHh);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        EnsureInitialized();
        var dH = DownloadBuffer(gradH);
        var rGate = DownloadBuffer(gateR);
        var zGate = DownloadBuffer(gateZ);
        var nGate = DownloadBuffer(gateN);
        var pH = DownloadBuffer(prevH);
        var wHh = DownloadBuffer(weightsHh);

        var gPH = new float[batch * hiddenSize];
        var gR = new float[batch * hiddenSize];
        var gZ = new float[batch * hiddenSize];
        var gN = new float[batch * hiddenSize];

        for (int b = 0; b < batch; b++)
        {
            int off = b * hiddenSize;

            // First pass: compute dN for all hidden units (needed for dR)
            var dNArr = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                float z = zGate[off + i];
                float n_ = nGate[off + i];
                float localDh = dH[off + i];

                // ht = (1-z)*n + z*hPrev
                dNArr[i] = localDh * (1f - z) * (1f - n_ * n_);
            }

            // Second pass: compute all gate gradients including proper dR
            for (int i = 0; i < hiddenSize; i++)
            {
                float r = rGate[off + i];
                float z = zGate[off + i];
                float n_ = nGate[off + i];
                float hPrev = pH[off + i];
                float localDh = dH[off + i];

                // dR_i = r_i*(1-r_i) * sum_j(dN_j * W_hn[j,i] * hPrev_i)
                float dRPre = 0f;
                for (int j = 0; j < hiddenSize; j++)
                    dRPre += dNArr[j] * wHh[(2 * hiddenSize + j) * hiddenSize + i] * hPrev;

                gZ[off + i] = localDh * (hPrev - n_) * z * (1f - z);
                gN[off + i] = dNArr[i];
                gR[off + i] = dRPre * r * (1f - r);
                gPH[off + i] = localDh * z;
            }
        }

        UploadToBuffer(gPH, gradPrevH);
        UploadToBuffer(gR, gradGateR);
        UploadToBuffer(gZ, gradGateZ);
        UploadToBuffer(gN, gradGateN);
    }

    #endregion

    // --- Split-buffer native Complex<T> operations (Vulkan) ---
    // Composes from existing GPU primitives with cached scratch buffers.

    private readonly object _complexScratchLock = new();
    private IGpuBuffer[]? _complexScratch;
    private int _complexScratchSize;

    private void EnsureComplexScratch(int n)
    {
        if (_complexScratch is not null && _complexScratchSize >= n) return;
        // Dispose old scratch
        if (_complexScratch is not null)
            foreach (var b in _complexScratch) b?.Dispose();
        _complexScratchSize = n;
        _complexScratch = new[] { AllocateBuffer(n), AllocateBuffer(n), AllocateBuffer(n), AllocateBuffer(n) };
    }

    public void SplitComplexMultiply(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
    {
        if (n <= 0) return;
        lock (_complexScratchLock)
        {
            EnsureComplexScratch(n);
            var t1 = _complexScratch![0]; var t2 = _complexScratch[1];
            var t3 = _complexScratch[2]; var t4 = _complexScratch[3];
            Multiply(aR, bR, t1, n); Multiply(aI, bI, t2, n); Subtract(t1, t2, oR, n);
            Multiply(aR, bI, t3, n); Multiply(aI, bR, t4, n); Add(t3, t4, oI, n);
        }
    }

    public void SplitComplexConjugate(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, int n)
    {
        if (n <= 0) return;
        Copy(iR, 0, oR, 0, n);
        Negate(iI, oI, n);
    }

    public void SplitComplexMagnitude(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n) => ComplexMagnitude(iR, iI, o, n);
    public void SplitComplexMagnitudeSquared(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n)
    {
        if (n <= 0) return;
        lock (_complexScratchLock)
        {
            EnsureComplexScratch(n);
            Multiply(iR, iR, _complexScratch![0], n); Multiply(iI, iI, _complexScratch[1], n);
            Add(_complexScratch[0], _complexScratch[1], o, n);
        }
    }
    public void SplitComplexPhase(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer o, int n) => ComplexPhase(iR, iI, o, n);
    public void SplitComplexFromPolar(IGpuBuffer m, IGpuBuffer p, IGpuBuffer oR, IGpuBuffer oI, int n) => PolarToComplex(m, p, oR, oI, n);

    public void SplitComplexScale(IGpuBuffer iR, IGpuBuffer iI, IGpuBuffer oR, IGpuBuffer oI, float s, int n)
    {
        if (n <= 0) return;
        Scale(iR, oR, s, n); Scale(iI, oI, s, n);
    }

    public void SplitComplexAdd(IGpuBuffer aR, IGpuBuffer aI, IGpuBuffer bR, IGpuBuffer bI, IGpuBuffer oR, IGpuBuffer oI, int n)
    {
        if (n <= 0) return;
        Add(aR, bR, oR, n); Add(aI, bI, oI, n);
    }

    public void SplitComplexCrossSpectral(IGpuBuffer xR, IGpuBuffer xI, IGpuBuffer yR, IGpuBuffer yI, IGpuBuffer oR, IGpuBuffer oI, int n)
    {
        if (n <= 0) return;
        lock (_complexScratchLock)
        {
            EnsureComplexScratch(n);
            var t1 = _complexScratch![0]; var t2 = _complexScratch[1];
            var t3 = _complexScratch[2]; var t4 = _complexScratch[3];
            Multiply(xR, yR, t1, n); Multiply(xI, yI, t2, n); Add(t1, t2, oR, n);
            Multiply(xI, yR, t3, n); Multiply(xR, yI, t4, n); Subtract(t3, t4, oI, n);
        }
    }

    public void SplitComplexTopK(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n, int k)
    {
        if (n <= 0 || k <= 0) return;
        // GPU mag computation + CPU threshold + GPU mask
        var magBuf = AllocateBuffer(n);
        try
        {
            SplitComplexMagnitudeSquared(inReal, inImag, magBuf, n);
            var magData = DownloadBuffer(magBuf);
            var rData = DownloadBuffer(inReal);
            var iData = DownloadBuffer(inImag);
            Array.Sort(magData); Array.Reverse(magData);
            float threshold = k <= n ? magData[Math.Min(k, n) - 1] : 0f;
            for (int i = 0; i < n; i++)
            {
                float ms = rData[i] * rData[i] + iData[i] * iData[i];
                if (ms < threshold) { rData[i] = 0; iData[i] = 0; }
            }
            var rBuf = AllocateBuffer(rData); var iBuf = AllocateBuffer(iData);
            try { Copy(rBuf, 0, outReal, 0, n); Copy(iBuf, 0, outImag, 0, n); }
            finally { rBuf.Dispose(); iBuf.Dispose(); }
        }
        finally { magBuf.Dispose(); }
    }

    public void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        var data = DownloadBuffer(input);
        var result = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            float mx = float.MinValue;
            for (int c = 0; c < cols; c++) mx = Math.Max(mx, data[off + c]);
            float se = 0;
            for (int c = 0; c < cols; c++) { result[off + c] = MathF.Exp(data[off + c] - mx); se += result[off + c]; }
            for (int c = 0; c < cols; c++) result[off + c] /= se;
        }
        var rb = AllocateBuffer(result);
        try { Copy(rb, 0, output, 0, rows * cols); }
        finally { rb.Dispose(); }
    }

    /// <inheritdoc/>
    public void SpectralFilter(IGpuBuffer inputReal, IGpuBuffer filterReal, IGpuBuffer filterImag,
        IGpuBuffer outputReal, int batch, int height, int width, int filterSliceCount)
    {
        if (batch <= 0 || height <= 0 || width <= 0) return;
        if (filterSliceCount <= 0 || (filterSliceCount != 1 && filterSliceCount != batch))
            throw new ArgumentException($"filterSliceCount must be 1 (shared) or batch ({batch}). Got {filterSliceCount}.");

        int sliceSize = height * width;
        int totalSize = batch * sliceSize;

        IGpuBuffer? fftR = null, fftI = null, mulR = null, mulI = null, ifftI = null, zeroI = null;
        try
        {
            fftR = AllocateBuffer(totalSize);
            fftI = AllocateBuffer(totalSize);
            mulR = AllocateBuffer(totalSize);
            mulI = AllocateBuffer(totalSize);
            ifftI = AllocateBuffer(totalSize);
            zeroI = AllocateBuffer(totalSize);
        Fill(zeroI, 0f, totalSize);

            BatchedFFT2D(inputReal, zeroI, fftR, fftI, batch, height, width, inverse: false);

            if (filterSliceCount == batch)
            {
                SplitComplexMultiply(fftR, fftI, filterReal, filterImag, mulR, mulI, totalSize);
            }
            else
            {
                var bcastFR = AllocateBuffer(totalSize);
                var bcastFI = AllocateBuffer(totalSize);
                try
                {
                    for (int b = 0; b < batch; b++)
                    {
                        Copy(filterReal, 0, bcastFR, b * sliceSize, sliceSize);
                        Copy(filterImag, 0, bcastFI, b * sliceSize, sliceSize);
                    }
                    SplitComplexMultiply(fftR, fftI, bcastFR, bcastFI, mulR, mulI, totalSize);
                }
                finally { bcastFR.Dispose(); bcastFI.Dispose(); }
            }

            BatchedFFT2D(mulR, mulI, outputReal, ifftI, batch, height, width, inverse: true);
        }
        finally
        {
            fftR?.Dispose(); fftI?.Dispose();
            mulR?.Dispose(); mulI?.Dispose();
            ifftI?.Dispose(); zeroI?.Dispose();
        }
    }

    // Issue #160 spectral perf kernels — Vulkan implementations.
    public void Atan2Elementwise(IGpuBuffer imag, IGpuBuffer real, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.Atan2Elementwise, imag, real, output, n,
            new uint[] { (uint)n }, sizeof(uint));
    }

    public void NormalizeRowsFused(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.NormalizeRowsFused, input, output, rows,
            new uint[] { (uint)rows, (uint)cols }, 2 * sizeof(uint));
    }

    public void AnalyticSignalMask(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batch, int fftSize, int binLow, int binHigh)
    {
        int total = batch * fftSize;
        if (total <= 0) return;
        // Apply mask to real and imag with single-buffer dispatches each
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.AnalyticSignalMaskScalar, specReal, outReal, total,
            new uint[] { (uint)batch, (uint)fftSize, (uint)binLow, (uint)binHigh }, 4 * sizeof(uint));
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.AnalyticSignalMaskScalar, specImag, outImag, total,
            new uint[] { (uint)batch, (uint)fftSize, (uint)binLow, (uint)binHigh }, 4 * sizeof(uint));
    }

    public void BispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2)
    {
        int total = maxF1 * maxF2;
        if (total <= 0) return;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.BispectrumReal, specReal, specImag, outReal, total,
            new uint[] { (uint)maxF1, (uint)maxF2 }, 2 * sizeof(uint));
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.BispectrumImag, specReal, specImag, outImag, total,
            new uint[] { (uint)maxF1, (uint)maxF2 }, 2 * sizeof(uint));
    }

    public void TrispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2, int maxF3)
    {
        int total = maxF1 * maxF2 * maxF3;
        if (total <= 0) return;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.TrispectrumReal, specReal, specImag, outReal, total,
            new uint[] { (uint)maxF1, (uint)maxF2, (uint)maxF3 }, 3 * sizeof(uint));
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.TrispectrumImag, specReal, specImag, outImag, total,
            new uint[] { (uint)maxF1, (uint)maxF2, (uint)maxF3 }, 3 * sizeof(uint));
    }

    public void CavityBounceInplace(IGpuBuffer workReal, IGpuBuffer workImag, int total, float invN)
    {
        if (total <= 0) return;
        uint invNbits = BitConverter.ToUInt32(BitConverter.GetBytes(invN), 0);
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.CavityBounceReal, workReal, workReal, total,
            new uint[] { (uint)total, invNbits }, 2 * sizeof(uint));
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.ZeroBuffer, workImag, workImag, total,
            new uint[] { (uint)total }, sizeof(uint));
    }

    public void WidebandLogBinPool(IGpuBuffer magBuf, IGpuBuffer output,
        int totalSegBatch, int fftSize, int numBins, int usable)
    {
        int total = totalSegBatch * numBins;
        if (total <= 0) return;
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.WidebandLogBinPool, magBuf, output, total,
            new uint[] { (uint)totalSegBatch, (uint)fftSize, (uint)numBins, (uint)usable }, 4 * sizeof(uint));
    }

    public void MelFilterbankApply(IGpuBuffer powerSpec, IGpuBuffer melFilters, IGpuBuffer melEnergy,
        int totalSegBatch, int specBins, int melBins)
    {
        int total = totalSegBatch * melBins;
        if (total <= 0) return;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.MelFilterbankApply, powerSpec, melFilters, melEnergy, total,
            new uint[] { (uint)totalSegBatch, (uint)specBins, (uint)melBins }, 3 * sizeof(uint));
    }

    public void MfccLog1p(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.MfccLog1p, input, output, n,
            new uint[] { (uint)n }, sizeof(uint));
    }

    public void PacPhaseBinMi(IGpuBuffer thetaPhase, IGpuBuffer gammaAmp, IGpuBuffer output,
        int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        if (batch <= 0) return;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.PacPhaseBinMi, thetaPhase, gammaAmp, output, batch,
            new uint[] { (uint)batch, (uint)numSamples, (uint)numGammaBands, (uint)gammaIdx }, 4 * sizeof(uint));
    }
}
