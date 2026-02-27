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
        => Copy(input, output, size);

    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
        => Copy(input, output, size);

    #endregion

    #region Sparse Operations

    public void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
    {
        EnsureInitialized();
        var dense = DownloadBuffer(denseInput);
        int sparseK = K / 2;
        var sv = new float[M * sparseK];
        var si = new float[M * sparseK];
        for (int row = 0; row < M; row++)
            for (int blk = 0; blk < K / 4; blk++)
            {
                int baseIdx = row * K + blk * 4;
                var vals = new (float v, int idx)[4];
                for (int j = 0; j < 4; j++) vals[j] = (MathF.Abs(dense[baseIdx + j]), j);
                Array.Sort(vals, (a, b) => b.v.CompareTo(a.v));
                int sOff = row * sparseK + blk * 2;
                for (int j = 0; j < 2; j++)
                {
                    sv[sOff + j] = dense[baseIdx + vals[j].idx];
                    si[sOff + j] = Int32BitsToSingleCompat(vals[j].idx);
                }
            }
        UploadToBuffer(sv, sparseValues);
        UploadToBuffer(si, sparseIndices);
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        EnsureInitialized();
        var sv = DownloadBuffer(sparseValues);
        var si = DownloadBuffer(sparseIndices);
        var dense = new float[M * K];
        int sparseK = K / 2;
        for (int row = 0; row < M; row++)
            for (int blk = 0; blk < K / 4; blk++)
            {
                int sOff = row * sparseK + blk * 2;
                int baseIdx = row * K + blk * 4;
                for (int j = 0; j < 2; j++)
                {
                    int idx = SingleToInt32BitsCompat(si[sOff + j]);
                    dense[baseIdx + idx] = sv[sOff + j];
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
            for (int idx = rowStart; idx < rowEnd; idx++)
            {
                int col = SingleToInt32BitsCompat(cols[idx]);
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
                for (int idx = rowStart; idx < rowEnd; idx++) mean += inp[SingleToInt32BitsCompat(cols[idx]) * N + j];
                mean /= count;
                float var_ = 0;
                for (int idx = rowStart; idx < rowEnd; idx++) { float d = inp[SingleToInt32BitsCompat(cols[idx]) * N + j] - mean; var_ += d * d; }
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
        var inp = DownloadBuffer(input); var o = new float[batchSize * dim];
        float maxNorm = 1f / MathF.Sqrt(curvature) - epsilon;
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim; float norm = 0;
            for (int d = 0; d < dim; d++) norm += inp[off + d] * inp[off + d];
            norm = MathF.Sqrt(norm);
            float scale = norm > maxNorm ? maxNorm / norm : 1f;
            for (int d = 0; d < dim; d++) o[off + d] = inp[off + d] * scale;
        }
        UploadToBuffer(o, output);
    }

    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        var xd = DownloadBuffer(x); var yd = DownloadBuffer(y); var o = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float xSq = 0, ySq = 0, xy = 0;
            for (int d = 0; d < dim; d++) { xSq += xd[off + d] * xd[off + d]; ySq += yd[off + d] * yd[off + d]; xy += xd[off + d] * yd[off + d]; }
            float c = curvature;
            float denom = 1f + 2f * c * xy + c * c * xSq * ySq;
            if (MathF.Abs(denom) < 1e-10f) denom = 1e-10f;
            for (int d = 0; d < dim; d++)
                o[off + d] = ((1f + 2f * c * xy + c * ySq) * xd[off + d] + (1f - c * xSq) * yd[off + d]) / denom;
        }
        UploadToBuffer(o, output);
    }

    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        var bp = DownloadBuffer(basePoint); var tv = DownloadBuffer(tangentVec); var o = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float bpSq = 0, tvNorm = 0;
            for (int d = 0; d < dim; d++) { bpSq += bp[off + d] * bp[off + d]; tvNorm += tv[off + d] * tv[off + d]; }
            tvNorm = MathF.Sqrt(tvNorm);
            float lambda = 2f / (1f - curvature * bpSq);
            if (tvNorm < 1e-10f) { Array.Copy(bp, off, o, off, dim); continue; }
            float t = MathF.Tanh(MathF.Sqrt(curvature) * lambda * tvNorm / 2f) / (MathF.Sqrt(curvature) * tvNorm);
            var scaled = new float[dim];
            for (int d = 0; d < dim; d++) scaled[d] = t * tv[off + d];
            float sSq = 0, bs = 0;
            for (int d = 0; d < dim; d++) { sSq += scaled[d] * scaled[d]; bs += bp[off + d] * scaled[d]; }
            float denom = 1f + 2f * curvature * bs + curvature * curvature * bpSq * sSq;
            if (MathF.Abs(denom) < 1e-10f) denom = 1e-10f;
            for (int d = 0; d < dim; d++)
                o[off + d] = ((1f + 2f * curvature * bs + curvature * sSq) * bp[off + d] + (1f - curvature * bpSq) * scaled[d]) / denom;
        }
        UploadToBuffer(o, output);
    }

    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        var xd = DownloadBuffer(x); var yd = DownloadBuffer(y); var o = new float[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float diffSq = 0, xSq = 0, ySq = 0;
            for (int d = 0; d < dim; d++) { float diff = xd[off + d] - yd[off + d]; diffSq += diff * diff; xSq += xd[off + d] * xd[off + d]; ySq += yd[off + d] * yd[off + d]; }
            float arg = 1f + 2f * curvature * diffSq / ((1f - curvature * xSq) * (1f - curvature * ySq));
            o[b] = (float)(AcoshDoubleCompat(Math.Max(1.0, arg)) / Math.Sqrt(curvature));
        }
        UploadToBuffer(o, output);
    }

    public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        var matResult = AllocateBuffer(batchSize * outputFeatures);
        Gemm(input, weights, matResult, batchSize, outputFeatures, inputFeatures);
        var biasResult = AllocateBuffer(batchSize * outputFeatures);
        BiasAdd(matResult, biases, biasResult, batchSize, outputFeatures);
        PoincareProject(biasResult, output, batchSize, outputFeatures, curvature, epsilon);
        matResult.Dispose(); biasResult.Dispose();
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        Fill(gradInput, 0f, gradInput.Size);
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        Fill(gradWeights, 0f, gradWeights.Size);
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        Fill(gradBiases, 0f, gradBiases.Size);
    }

    #endregion

    #region Octonion Algebra Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        EnsureInitialized();
        var ad = DownloadBuffer(a); var bd = DownloadBuffer(b); var o = new float[count * 8];
        for (int i = 0; i < count; i++)
        {
            int off = i * 8;
            float a0 = ad[off], a1 = ad[off+1], a2 = ad[off+2], a3 = ad[off+3],
                  a4 = ad[off+4], a5 = ad[off+5], a6 = ad[off+6], a7 = ad[off+7];
            float b0 = bd[off], b1 = bd[off+1], b2 = bd[off+2], b3 = bd[off+3],
                  b4 = bd[off+4], b5 = bd[off+5], b6 = bd[off+6], b7 = bd[off+7];
            o[off+0] = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
            o[off+1] = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
            o[off+2] = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
            o[off+3] = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
            o[off+4] = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
            o[off+5] = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
            o[off+6] = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
            o[off+7] = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
        }
        UploadToBuffer(o, output);
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        => Add(a, b, output, count * 8);

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        Gemm(input, weights, output, batchSize, outputFeatures * 8, inputFeatures * 8);
        BiasAdd(output, biases, output, batchSize, outputFeatures * 8);
    }

    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        Fill(gradInput, 0f, gradInput.Size);
    }

    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        Fill(gradWeights, 0f, gradWeights.Size);
    }

    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures)
    {
        EnsureInitialized();
        var go = DownloadBuffer(gradOutput);
        var gb = new float[outputFeatures * 8];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < outputFeatures * 8; i++)
                gb[i] += go[b * outputFeatures * 8 + i];
        UploadToBuffer(gb, gradBiases);
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
        float sign = inverse ? 1f : -1f;
        for (int k = 0; k < n; k++)
        {
            float sumR = 0, sumI = 0;
            for (int j = 0; j < n; j++)
            {
                float angle = sign * 2f * MathF.PI * k * j / n;
                sumR += ir[j] * MathF.Cos(angle) - ii[j] * MathF.Sin(angle);
                sumI += ir[j] * MathF.Sin(angle) + ii[j] * MathF.Cos(angle);
            }
            if (inverse) { sumR /= n; sumI /= n; }
            or_[k] = sumR; oi[k] = sumI;
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
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
        float sign = inverse ? 1f : -1f;
        for (int b = 0; b < batch; b++)
        {
            int off = b * n;
            for (int k = 0; k < n; k++)
            {
                float sumR = 0, sumI = 0;
                for (int j = 0; j < n; j++)
                {
                    float angle = sign * 2f * MathF.PI * k * j / n;
                    sumR += ir[off + j] * MathF.Cos(angle) - ii[off + j] * MathF.Sin(angle);
                    sumI += ir[off + j] * MathF.Sin(angle) + ii[off + j] * MathF.Cos(angle);
                }
                if (inverse) { sumR /= n; sumI /= n; }
                or_[off + k] = sumR; oi[off + k] = sumI;
            }
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        BatchedFFT(inputReal, inputImag, outputReal, outputImag, height, width, inverse);
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
        Fill(gradInput, 0f, gradInput.Size);
        Fill(gradHInit, 0f, gradHInit.Size);
        Fill(gradCInit, 0f, gradCInit.Size);
        Fill(gradWeightsIh, 0f, gradWeightsIh.Size);
        Fill(gradWeightsHh, 0f, gradWeightsHh.Size);
        Fill(gradBiasIh, 0f, gradBiasIh.Size);
        Fill(gradBiasHh, 0f, gradBiasHh.Size);
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
                    float r = 1f / (1f + MathF.Exp(-gates[i]));
                    float z = 1f / (1f + MathF.Exp(-gates[hiddenSize + i]));
                    float n_ = MathF.Tanh(gates[2 * hiddenSize + i]);
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
        Fill(gradInput, 0f, gradInput.Size);
        Fill(gradHInit, 0f, gradHInit.Size);
        Fill(gradWeightsIh, 0f, gradWeightsIh.Size);
        Fill(gradWeightsHh, 0f, gradWeightsHh.Size);
        Fill(gradBiasIh, 0f, gradBiasIh.Size);
        Fill(gradBiasHh, 0f, gradBiasHh.Size);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        Fill(gradPrevH, 0f, gradPrevH.Size);
        Fill(gradGateR, 0f, gradGateR.Size);
        Fill(gradGateZ, 0f, gradGateZ.Size);
        Fill(gradGateN, 0f, gradGateN.Size);
    }

    #endregion
}
