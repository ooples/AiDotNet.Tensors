// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 4: Optimizers, FP16/32, Specialized, Hyperbolic, Octonion, Quantum, FFT, RNN.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Optimizer Operations

    public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient, float learningRate, float weightDecay, int size)
    {
        using var dummyM = (WebGpuBuffer)AllocateBuffer(1);
        using var dummyV = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeOptimizerUniforms(size, learningRate, 0, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "sgd",
            param, gradient, dummyM, dummyV, uniforms, size).GetAwaiter().GetResult();
    }

    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        using var dummyV = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "sgd_momentum",
            param, gradient, velocity, dummyV, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "adam",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "adamw",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
    }

    public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        using var dummyState2 = (WebGpuBuffer)AllocateBuffer(1);
        // beta1=rho for rmsprop kernel
        var uniforms = MakeOptimizerUniforms(size, learningRate, rho, 0, epsilon, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "rmsprop",
            param, gradient, squaredAvg, dummyState2, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        using var dummyState2 = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeOptimizerUniforms(size, learningRate, 0, 0, epsilon, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "adagrad",
            param, gradient, accumulatedGrad, dummyState2, uniforms, size).GetAwaiter().GetResult();
    }

    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        using var dummyState2 = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "nag",
            param, gradient, velocity, dummyState2, uniforms, size).GetAwaiter().GetResult();
    }

    public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        EnsureInitialized();
        var p = DownloadBufferData(param); var g = DownloadBufferData(gradient); var v = DownloadBufferData(velocity);
        float pNorm = 0, gNorm = 0;
        for (int i = 0; i < size; i++) { pNorm += p[i] * p[i]; gNorm += g[i] * g[i]; }
        pNorm = MathF.Sqrt(pNorm); gNorm = MathF.Sqrt(gNorm);
        float localLr = pNorm > 0 && gNorm > 0 ? trustCoeff * pNorm / (gNorm + weightDecay * pNorm) : 1f;
        for (int i = 0; i < size; i++) { v[i] = momentum * v[i] + localLr * (g[i] + weightDecay * p[i]); p[i] -= learningRate * v[i]; }
        UploadToBuffer(p, param); UploadToBuffer(v, velocity);
    }

    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        EnsureInitialized();
        var p = DownloadBufferData(param); var g = DownloadBufferData(gradient);
        var mData = DownloadBufferData(m); var vData = DownloadBufferData(v);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        var update = new float[size];
        for (int i = 0; i < size; i++)
        {
            mData[i] = beta1 * mData[i] + (1 - beta1) * g[i];
            vData[i] = beta2 * vData[i] + (1 - beta2) * g[i] * g[i];
            update[i] = (mData[i] / bc1) / (MathF.Sqrt(vData[i] / bc2) + epsilon) + weightDecay * p[i];
        }
        float pNorm = 0, uNorm = 0;
        for (int i = 0; i < size; i++) { pNorm += p[i] * p[i]; uNorm += update[i] * update[i]; }
        pNorm = MathF.Sqrt(pNorm); uNorm = MathF.Sqrt(uNorm);
        float ratio = uNorm > 0 ? pNorm / uNorm : 1f;
        for (int i = 0; i < size; i++) p[i] -= learningRate * ratio * update[i];
        UploadToBuffer(p, param); UploadToBuffer(mData, m); UploadToBuffer(vData, v);
    }

    public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        // state1=accumGrad, state2=accumUpdate, beta1=rho
        var uniforms = MakeOptimizerUniforms(size, 0, rho, 0, epsilon, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "adadelta",
            param, gradient, accumGrad, accumUpdate, uniforms, size).GetAwaiter().GetResult();
    }

    public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // 5-buffer kernel: param, gradient, m, v, v_max each as separate storage buffers
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch5BufferAsync("Amsgrad5BufferOptimizer", WebGpuKernels.Amsgrad5BufferOptimizerSource, "amsgrad5",
            param, gradient, m, v, vMax, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // state1=m, state2=u
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "adamax",
            param, gradient, m, u, uniforms, size).GetAwaiter().GetResult();
    }

    public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        using var dummyState2 = (WebGpuBuffer)AllocateBuffer(1);
        // Lion kernel uses state1=m, does not use state2
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, 0, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "lion",
            param, gradient, m, dummyState2, uniforms, size).GetAwaiter().GetResult();
    }

    public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // state1=m, state2=v
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "nadam",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
    }

    public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        EnsureInitialized();
        var p = DownloadBufferData(param); var g = DownloadBufferData(gradient);
        var zData = DownloadBufferData(z); var nData = DownloadBufferData(n);
        for (int i = 0; i < size; i++)
        {
            float sigma = (MathF.Sqrt(nData[i] + g[i] * g[i]) - MathF.Sqrt(nData[i])) / learningRate;
            zData[i] += g[i] - sigma * p[i];
            nData[i] += g[i] * g[i];
            if (MathF.Abs(zData[i]) <= l1Reg) p[i] = 0;
            else p[i] = -(zData[i] - MathF.Sign(zData[i]) * l1Reg) / (l2Reg + (beta + MathF.Sqrt(nData[i])) / learningRate);
        }
        UploadToBuffer(p, param); UploadToBuffer(zData, z); UploadToBuffer(nData, n);
    }

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size) => Copy(input, output, size);
    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size) => Copy(input, output, size);

    #endregion

    #region Specialized Layer Operations

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var c = DownloadBufferData(centers); var eps = DownloadBufferData(epsilons);
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
        var w = DownloadBufferData(weights);
        var preT = DownloadBufferData(preTrace); var postT = DownloadBufferData(postTrace);
        var preS = DownloadBufferData(preSpike); var postS = DownloadBufferData(postSpike);
        for (int i = 0; i < numPre; i++)
            for (int j = 0; j < numPost; j++)
            {
                int idx = i * numPost + j;
                // LTP: pre-spike paired with post-trace
                if (preS[i] > 0.5f) w[idx] += ltpRate * postT[j];
                // LTD: post-spike paired with pre-trace
                if (postS[j] > 0.5f) w[idx] -= ltdRate * preT[i];
                // Homeostasis
                w[idx] += homeostasisRate * (0.5f - w[idx]);
                // Clamp
                w[idx] = MathF.Max(minWeight, MathF.Min(maxWeight, w[idx]));
            }
        UploadToBuffer(w, weights);
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        EnsureInitialized();
        var t = DownloadBufferData(traces); var s = DownloadBufferData(spikes); var inp = DownloadBufferData(input);
        for (int i = 0; i < size; i++)
        {
            t[i] = decay * t[i];
            if (inp[i] > threshold) { s[i] = 1f; t[i] += 1f; }
            else s[i] = 0f;
        }
        UploadToBuffer(t, traces); UploadToBuffer(s, spikes);
    }

    #endregion

    #region Hyperbolic Geometry

    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var o = new float[batchSize * dim];
        float maxNorm = (1f - epsilon) / MathF.Sqrt(curvature);
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
        var xd = DownloadBufferData(x); var yd = DownloadBufferData(y); var o = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float xSq = 0, ySq = 0, xy = 0;
            for (int d = 0; d < dim; d++) { xSq += xd[off + d] * xd[off + d]; ySq += yd[off + d] * yd[off + d]; xy += xd[off + d] * yd[off + d]; }
            float denom = 1f + 2f * curvature * xy + curvature * curvature * xSq * ySq;
            if (MathF.Abs(denom) < 1e-10f) denom = 1e-10f;
            for (int d = 0; d < dim; d++)
                o[off + d] = ((1f + 2f * curvature * xy + curvature * ySq) * xd[off + d] + (1f - curvature * xSq) * yd[off + d]) / denom;
        }
        UploadToBuffer(o, output);
    }

    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        var bp = DownloadBufferData(basePoint); var tv = DownloadBufferData(tangentVec); var o = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim; float bpSq = 0, tNorm = 0;
            for (int d = 0; d < dim; d++) { bpSq += bp[off + d] * bp[off + d]; tNorm += tv[off + d] * tv[off + d]; }
            tNorm = MathF.Sqrt(tNorm);
            float conformal = 2f / (1f - curvature * bpSq + 1e-10f);
            float th = MathF.Tanh(conformal * tNorm / 2f);
            float scale = tNorm > 1e-10f ? th / (tNorm * MathF.Sqrt(curvature)) : 0;
            var scaled = new float[dim];
            for (int d = 0; d < dim; d++) scaled[d] = tv[off + d] * scale;
            float sSq = 0, bs = 0;
            for (int d = 0; d < dim; d++) { sSq += scaled[d] * scaled[d]; bs += bp[off + d] * scaled[d]; }
            float den = 1f + 2f * curvature * bs + curvature * curvature * bpSq * sSq;
            if (MathF.Abs(den) < 1e-10f) den = 1e-10f;
            for (int d = 0; d < dim; d++)
                o[off + d] = ((1f + 2f * curvature * bs + curvature * sSq) * bp[off + d] + (1f - curvature * bpSq) * scaled[d]) / den;
        }
        UploadToBuffer(o, output);
    }

    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        EnsureInitialized();
        var xd = DownloadBufferData(x); var yd = DownloadBufferData(y); var o = new float[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim; float diffSq = 0, xSq = 0, ySq = 0;
            for (int d = 0; d < dim; d++) { float diff = xd[off + d] - yd[off + d]; diffSq += diff * diff; xSq += xd[off + d] * xd[off + d]; ySq += yd[off + d] * yd[off + d]; }
            float arg = 1f + 2f * curvature * diffSq / ((1f - curvature * xSq) * (1f - curvature * ySq));
            o[b] = (float)(Math.Acosh(Math.Max(1.0, arg)) / Math.Sqrt(curvature));
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
        int batchSize, int inputFeatures, int outputFeatures, float curvature) =>
        Fill(gradInput, 0f, batchSize * inputFeatures);

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature) =>
        Fill(gradWeights, 0f, inputFeatures * outputFeatures);

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature) =>
        Fill(gradBiases, 0f, outputFeatures);

    #endregion

    #region Octonion Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        EnsureInitialized();
        var ad = DownloadBufferData(a); var bd = DownloadBufferData(b); var o = new float[count * 8];
        for (int i = 0; i < count; i++)
        {
            int off = i * 8;
            float a0 = ad[off], a1 = ad[off + 1], a2 = ad[off + 2], a3 = ad[off + 3];
            float a4 = ad[off + 4], a5 = ad[off + 5], a6 = ad[off + 6], a7 = ad[off + 7];
            float b0 = bd[off], b1 = bd[off + 1], b2 = bd[off + 2], b3 = bd[off + 3];
            float b4 = bd[off + 4], b5 = bd[off + 5], b6 = bd[off + 6], b7 = bd[off + 7];
            o[off]     = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
            o[off + 1] = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
            o[off + 2] = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
            o[off + 3] = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
            o[off + 4] = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
            o[off + 5] = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
            o[off + 6] = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
            o[off + 7] = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
        }
        UploadToBuffer(o, output);
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count) => Add(a, b, output, count * 8);

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        Gemm(input, weights, output, batchSize, outputFeatures, inputFeatures);
        BiasAdd(output, biases, output, batchSize, outputFeatures);
    }

    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures) =>
        Fill(gradInput, 0f, batchSize * inputFeatures);

    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures) =>
        Fill(gradWeights, 0f, inputFeatures * outputFeatures);

    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures) =>
        Fill(gradBiases, 0f, outputFeatures);

    #endregion

    #region Quantum Operations

    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var re = DownloadBufferData(realPart); var im = DownloadBufferData(imagPart);
        var o = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < stateSize; s++) { int idx = b * stateSize + s; o[idx] = re[idx] * re[idx] + im[idx] * im[idx]; }
        UploadToBuffer(o, probabilities);
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var p = DownloadBufferData(probabilities);
        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0;
            for (int s = 0; s < stateSize; s++) sum += p[b * stateSize + s];
            if (sum > 0) for (int s = 0; s < stateSize; s++) p[b * stateSize + s] /= sum;
        }
        UploadToBuffer(p, probabilities);
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        EnsureInitialized();
        var mr = DownloadBufferData(matReal); var mi = DownloadBufferData(matImag);
        var vr = DownloadBufferData(vecReal); var vi = DownloadBufferData(vecImag);
        var or_ = new float[batchSize * dim]; var oi = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
            for (int r = 0; r < dim; r++)
            {
                float re = 0, im = 0;
                for (int c = 0; c < dim; c++) { int mIdx = r * dim + c; int vIdx = b * dim + c; re += mr[mIdx] * vr[vIdx] - mi[mIdx] * vi[vIdx]; im += mr[mIdx] * vi[vIdx] + mi[mIdx] * vr[vIdx]; }
                or_[b * dim + r] = re; oi[b * dim + r] = im;
            }
        UploadToBuffer(or_, outReal); UploadToBuffer(oi, outImag);
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        EnsureInitialized();
        var re = DownloadBufferData(stateReal); var im = DownloadBufferData(stateImag); var ang = DownloadBufferData(angles);
        var or_ = new float[batchSize * numQubits]; var oi = new float[batchSize * numQubits];
        for (int b = 0; b < batchSize; b++)
            for (int q = 0; q < numQubits; q++)
            {
                int idx = b * numQubits + q; float c = MathF.Cos(ang[idx]), s = MathF.Sin(ang[idx]);
                or_[idx] = c * re[idx] - s * im[idx]; oi[idx] = s * re[idx] + c * im[idx];
            }
        UploadToBuffer(or_, outReal); UploadToBuffer(oi, outImag);
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var o = new float[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0;
            for (int s = 0; s < stateSize; s++) sum += inp[b * stateSize + s] * inp[b * stateSize + s];
            o[b] = sum;
        }
        UploadToBuffer(o, output);
    }

    #endregion

    #region FFT Operations

    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        EnsureInitialized();
        var re = DownloadBufferData(inputReal); var im = DownloadBufferData(inputImag);
        var or_ = new float[n]; var oi = new float[n];
        float sign = inverse ? 1f : -1f;
        for (int k = 0; k < n; k++)
        {
            float sumR = 0, sumI = 0;
            for (int j = 0; j < n; j++)
            {
                float angle = sign * 2f * MathF.PI * k * j / n;
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                sumR += re[j] * c - im[j] * s; sumI += re[j] * s + im[j] * c;
            }
            if (inverse) { or_[k] = sumR / n; oi[k] = sumI / n; }
            else { or_[k] = sumR; oi[k] = sumI; }
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    public void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
    {
        var zeros = AllocateBuffer(n); Fill(zeros, 0f, n);
        FFT(input, zeros, outputReal, outputImag, n, false);
        zeros.Dispose();
    }

    public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        var tempImag = AllocateBuffer(n);
        FFT(inputReal, inputImag, output, tempImag, n, true);
        tempImag.Dispose();
    }

    public void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse)
    {
        EnsureInitialized();
        var re = DownloadBufferData(inputReal); var im = DownloadBufferData(inputImag);
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
                    float c = MathF.Cos(angle), s = MathF.Sin(angle);
                    sumR += re[off + j] * c - im[off + j] * s; sumI += re[off + j] * s + im[off + j] * c;
                }
                if (inverse) { or_[off + k] = sumR / n; oi[off + k] = sumI / n; }
                else { or_[off + k] = sumR; oi[off + k] = sumI; }
            }
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
    {
        EnsureInitialized();
        var tempR = new float[height * width]; var tempI = new float[height * width];
        var re = DownloadBufferData(inputReal); var im = DownloadBufferData(inputImag);
        float sign = inverse ? 1f : -1f;
        for (int r = 0; r < height; r++)
        {
            for (int k = 0; k < width; k++)
            {
                float sr = 0, si = 0;
                for (int j = 0; j < width; j++) { float a = sign * 2f * MathF.PI * k * j / width; float c = MathF.Cos(a), s = MathF.Sin(a); sr += re[r * width + j] * c - im[r * width + j] * s; si += re[r * width + j] * s + im[r * width + j] * c; }
                tempR[r * width + k] = sr; tempI[r * width + k] = si;
            }
        }
        var or_ = new float[height * width]; var oi = new float[height * width];
        for (int c = 0; c < width; c++)
        {
            for (int k = 0; k < height; k++)
            {
                float sr = 0, si = 0;
                for (int j = 0; j < height; j++) { float a = sign * 2f * MathF.PI * k * j / height; float co = MathF.Cos(a), s = MathF.Sin(a); sr += tempR[j * width + c] * co - tempI[j * width + c] * s; si += tempR[j * width + c] * s + tempI[j * width + c] * co; }
                if (inverse) { or_[k * width + c] = sr / (height * width); oi[k * width + c] = si / (height * width); }
                else { or_[k * width + c] = sr; oi[k * width + c] = si; }
            }
        }
        UploadToBuffer(or_, outputReal); UploadToBuffer(oi, outputImag);
    }

    public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n) => CpuBinary(input, window, output, n, (a, b) => a * b);

    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        var r = DownloadBufferData(real); var im = DownloadBufferData(imag); var o = new float[n];
        for (int i = 0; i < n; i++) o[i] = MathF.Sqrt(r[i] * r[i] + im[i] * im[i]);
        UploadToBuffer(o, magnitude);
    }

    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        var r = DownloadBufferData(real); var im = DownloadBufferData(imag); var o = new float[n];
        for (int i = 0; i < n; i++) o[i] = MathF.Atan2(im[i], r[i]);
        UploadToBuffer(o, phase);
    }

    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        var m = DownloadBufferData(magnitude); var p = DownloadBufferData(phase);
        var or_ = new float[n]; var oi = new float[n];
        for (int i = 0; i < n; i++) { or_[i] = m[i] * MathF.Cos(p[i]); oi[i] = m[i] * MathF.Sin(p[i]); }
        UploadToBuffer(or_, real); UploadToBuffer(oi, imag);
    }

    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
    {
        EnsureInitialized();
        var spec = DownloadBufferData(powerSpec); var fb = DownloadBufferData(filterbank);
        var o = new float[numFrames * nMels];
        for (int b = 0; b < numFrames; b++)
            for (int m = 0; m < nMels; m++)
            {
                float sum = 0;
                for (int f = 0; f < numFreqs; f++) sum += spec[b * numFreqs + f] * fb[m * numFreqs + f];
                o[b * nMels + m] = sum;
            }
        UploadToBuffer(o, melSpec);
    }

    public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
        => CpuUnary(power, db, n, v => 10f * MathF.Log10(MathF.Max(v, refValue)));

    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
        => CpuUnary(db, power, n, v => MathF.Pow(10f, v / 10f));

    #endregion

    #region RNN Operations

    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var wih = DownloadBufferData(weightsIh); var whh = DownloadBufferData(weightsHh);
        var bih = DownloadBufferData(biasIh); var bhh = DownloadBufferData(biasHh);
        var h = DownloadBufferData(hInit); var c = DownloadBufferData(cInit);
        var o = new float[batch * seqLen * hiddenSize];
        var allHData = new float[seqLen * batch * hiddenSize];
        var allCData = new float[seqLen * batch * hiddenSize];
        var gatesData = new float[seqLen * batch * 4 * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            var gates = new float[batch * 4 * hiddenSize];
            for (int b = 0; b < batch; b++)
                for (int g = 0; g < 4 * hiddenSize; g++)
                {
                    float sum = bih[g] + bhh[g];
                    for (int i = 0; i < inputSize; i++) sum += inp[(b * seqLen + t) * inputSize + i] * wih[g * inputSize + i];
                    for (int j = 0; j < hiddenSize; j++) sum += h[b * hiddenSize + j] * whh[g * hiddenSize + j];
                    gates[b * 4 * hiddenSize + g] = sum;
                }
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < hiddenSize; j++)
                {
                    int gOff = b * 4 * hiddenSize;
                    float it = 1f / (1f + MathF.Exp(-gates[gOff + j]));
                    float ft = 1f / (1f + MathF.Exp(-gates[gOff + hiddenSize + j]));
                    float gt = MathF.Tanh(gates[gOff + 2 * hiddenSize + j]);
                    float ot = 1f / (1f + MathF.Exp(-gates[gOff + 3 * hiddenSize + j]));
                    c[b * hiddenSize + j] = ft * c[b * hiddenSize + j] + it * gt;
                    h[b * hiddenSize + j] = ot * MathF.Tanh(c[b * hiddenSize + j]);
                    o[(b * seqLen + t) * hiddenSize + j] = h[b * hiddenSize + j];
                }
            Array.Copy(h, 0, allHData, t * batch * hiddenSize, batch * hiddenSize);
            Array.Copy(c, 0, allCData, t * batch * hiddenSize, batch * hiddenSize);
            Array.Copy(gates, 0, gatesData, t * batch * 4 * hiddenSize, batch * 4 * hiddenSize);
        }
        UploadToBuffer(o, output); UploadToBuffer(h, hFinal); UploadToBuffer(c, cFinal);
        UploadToBuffer(allHData, allH); UploadToBuffer(allCData, allC); UploadToBuffer(gatesData, cacheGates);
    }

    public void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        Fill(gradInput, 0f, batch * seqLen * inputSize);
        Fill(gradHInit, 0f, batch * hiddenSize);
        Fill(gradCInit, 0f, batch * hiddenSize);
        Fill(gradWeightsIh, 0f, 4 * hiddenSize * inputSize);
        Fill(gradWeightsHh, 0f, 4 * hiddenSize * hiddenSize);
        Fill(gradBiasIh, 0f, 4 * hiddenSize);
        Fill(gradBiasHh, 0f, 4 * hiddenSize);
    }

    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        var inp = DownloadBufferData(input); var wih = DownloadBufferData(weightsIh); var whh = DownloadBufferData(weightsHh);
        var bih = DownloadBufferData(biasIh); var bhh = DownloadBufferData(biasHh);
        var h = DownloadBufferData(hInit);
        var o = new float[batch * seqLen * hiddenSize];
        var allHData = new float[seqLen * batch * hiddenSize];
        var gatesData = new float[seqLen * batch * 3 * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            var gates = new float[batch * 3 * hiddenSize];
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < hiddenSize; j++)
                {
                    float rGate = bih[j] + bhh[j], zGate = bih[hiddenSize + j] + bhh[hiddenSize + j];
                    for (int i = 0; i < inputSize; i++) { rGate += inp[(b * seqLen + t) * inputSize + i] * wih[j * inputSize + i]; zGate += inp[(b * seqLen + t) * inputSize + i] * wih[(hiddenSize + j) * inputSize + i]; }
                    for (int k = 0; k < hiddenSize; k++) { rGate += h[b * hiddenSize + k] * whh[j * hiddenSize + k]; zGate += h[b * hiddenSize + k] * whh[(hiddenSize + j) * hiddenSize + k]; }
                    float r = 1f / (1f + MathF.Exp(-rGate)), z = 1f / (1f + MathF.Exp(-zGate));
                    float nGate = bih[2 * hiddenSize + j];
                    for (int i = 0; i < inputSize; i++) nGate += inp[(b * seqLen + t) * inputSize + i] * wih[(2 * hiddenSize + j) * inputSize + i];
                    float nh = bhh[2 * hiddenSize + j];
                    for (int k = 0; k < hiddenSize; k++) nh += h[b * hiddenSize + k] * whh[(2 * hiddenSize + j) * hiddenSize + k];
                    nGate += r * nh;
                    float n = MathF.Tanh(nGate);
                    h[b * hiddenSize + j] = (1f - z) * n + z * h[b * hiddenSize + j];
                    o[(b * seqLen + t) * hiddenSize + j] = h[b * hiddenSize + j];
                    gates[b * 3 * hiddenSize + j] = r;
                    gates[b * 3 * hiddenSize + hiddenSize + j] = z;
                    gates[b * 3 * hiddenSize + 2 * hiddenSize + j] = n;
                }
            Array.Copy(h, 0, allHData, t * batch * hiddenSize, batch * hiddenSize);
            Array.Copy(gates, 0, gatesData, t * batch * 3 * hiddenSize, batch * 3 * hiddenSize);
        }
        UploadToBuffer(o, output); UploadToBuffer(h, hFinal);
        UploadToBuffer(allHData, allH); UploadToBuffer(gatesData, cacheGates);
    }

    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        Fill(gradInput, 0f, batch * seqLen * inputSize);
        Fill(gradHInit, 0f, batch * hiddenSize);
        Fill(dHBuffer, 0f, batch * hiddenSize);
        Fill(gradWeightsIh, 0f, 3 * hiddenSize * inputSize);
        Fill(gradWeightsHh, 0f, 3 * hiddenSize * hiddenSize);
        Fill(gradBiasIh, 0f, 3 * hiddenSize);
        Fill(gradBiasHh, 0f, 3 * hiddenSize);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        Fill(gradPrevH, 0f, batch * hiddenSize);
        Fill(gradGateR, 0f, batch * hiddenSize);
        Fill(gradGateZ, 0f, batch * hiddenSize);
        Fill(gradGateN, 0f, batch * hiddenSize);
    }

    #endregion
}
#endif
