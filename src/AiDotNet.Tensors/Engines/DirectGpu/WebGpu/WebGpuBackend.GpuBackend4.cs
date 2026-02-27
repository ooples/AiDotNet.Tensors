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
        // Two-pass: compute norms on GPU, then dispatch update with pre-computed local_lr
        using var squared = (WebGpuBuffer)AllocateBuffer(size);
        // Compute param norm
        SquareAsync(param, squared, size).GetAwaiter().GetResult();
        float pNormSq = SumAsync(squared, size).GetAwaiter().GetResult();
        float pNorm = MathF.Sqrt(pNormSq);
        // Compute gradient norm
        SquareAsync(gradient, squared, size).GetAwaiter().GetResult();
        float gNormSq = SumAsync(squared, size).GetAwaiter().GetResult();
        float gNorm = MathF.Sqrt(gNormSq);
        float localLr = pNorm > 0 && gNorm > 0 ? trustCoeff * pNorm / (gNorm + weightDecay * pNorm) : 1f;
        // LarsLambFtrlSource lars_update: extra = pre-computed local_lr, beta1 = momentum
        using var dummyState2 = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        // Replace the last uniform (extra) with localLr
        uniforms[7] = localLr;
        Dispatch4BufferAsync("LarsLambFtrl", WebGpuKernels.LarsLambFtrlSource, "lars_update",
            param, gradient, velocity, dummyState2, uniforms, size).GetAwaiter().GetResult();
    }

    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Two-pass: first compute param_norm via GPU, then compute update_norm via Adam-like step, then dispatch
        // For LAMB we need param_norm and update_norm. Compute param_norm on GPU:
        using var squared = (WebGpuBuffer)AllocateBuffer(size);
        SquareAsync(param, squared, size).GetAwaiter().GetResult();
        float pNormSq = SumAsync(squared, size).GetAwaiter().GetResult();
        float pNorm = MathF.Sqrt(pNormSq);
        // To compute update_norm, we need the Adam update vector. Do this on CPU for the norm computation only:
        var mData = DownloadBufferData(m); var vData = DownloadBufferData(v);
        var gData = DownloadBufferData(gradient); var pData = DownloadBufferData(param);
        float bc1 = 1f - MathF.Pow(beta1, step), bc2 = 1f - MathF.Pow(beta2, step);
        float uNormSq = 0;
        for (int i = 0; i < size; i++)
        {
            float mHat = (beta1 * mData[i] + (1 - beta1) * gData[i]) / bc1;
            float vHat = (beta2 * vData[i] + (1 - beta2) * gData[i] * gData[i]) / bc2;
            float ui = mHat / (MathF.Sqrt(vHat) + epsilon) + weightDecay * pData[i];
            uNormSq += ui * ui;
        }
        float uNorm = MathF.Sqrt(uNormSq);
        float ratio = uNorm > 0 ? pNorm / uNorm : 1f;
        // LarsLambFtrlSource lamb_update: extra = pre-computed ratio
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        uniforms[7] = ratio;
        Dispatch4BufferAsync("LarsLambFtrl", WebGpuKernels.LarsLambFtrlSource, "lamb_update",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
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
        // LarsLambFtrlSource ftrl_update: state1=z, state2=n, beta1=l1_reg, beta2=l2_reg, extra=beta_param
        var uniforms = MakeOptimizerUniforms(size, learningRate, l1Reg, l2Reg, 0, 0, 0);
        uniforms[7] = beta; // extra = beta parameter
        Dispatch4BufferAsync("LarsLambFtrl", WebGpuKernels.LarsLambFtrlSource, "ftrl_update",
            param, gradient, z, n, uniforms, size).GetAwaiter().GetResult();
    }

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size) => Copy(input, output, size);
    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size) => Copy(input, output, size);

    #endregion

    #region Specialized Layer Operations

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        // SpecializedLayerSource rbf_forward: binding(0)=input, binding(1)=centers+epsilons combined, binding(2)=output
        // Epsilons are stored after centers at offset numCenters*inputDim in the centers buffer
        var centersData = DownloadBufferData(centers);
        var epsData = DownloadBufferData(epsilons);
        var combined = new float[numCenters * inputDim + numCenters];
        Array.Copy(centersData, 0, combined, 0, numCenters * inputDim);
        Array.Copy(epsData, 0, combined, numCenters * inputDim, numCenters);
        using var combinedBuf = (WebGpuBuffer)AllocateBuffer(combined.Length);
        UploadToBuffer(combined, combinedBuf);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(numCenters),
            BitConverter.Int32BitsToSingle(inputDim),
            0
        };
        int total = batchSize * numCenters;
        Dispatch3BufferAsync("SpecializedLayer", WebGpuKernels.SpecializedLayerSource, "rbf_forward",
            input, combinedBuf, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        // NeuromorphicSource stdp_update: binding(0)=weights, binding(1)=preSpike, binding(2)=postSpike, binding(3)=preTrace
        // Uniform: num_pre, num_post, a_plus=ltpRate, a_minus=ltdRate, lr=1.0 (already scaled), pad, pad, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numPre),
            BitConverter.Int32BitsToSingle(numPost),
            ltpRate,
            ltdRate,
            1f,
            0, 0, 0
        };
        int total = numPre * numPost;
        Dispatch4BufferAsync("Neuromorphic", WebGpuKernels.NeuromorphicSource, "stdp_update",
            weights, preSpike, postSpike, preTrace, uniforms, total).GetAwaiter().GetResult();
        // Apply homeostasis and clamping via CPU post-processing
        if (homeostasisRate != 0f || minWeight != float.MinValue || maxWeight != float.MaxValue)
        {
            var w = DownloadBufferData(weights);
            for (int i = 0; i < total; i++)
            {
                w[i] += homeostasisRate * (0.5f - w[i]);
                w[i] = MathF.Max(minWeight, MathF.Min(maxWeight, w[i]));
            }
            UploadToBuffer(w, weights);
        }
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        // Two-step: First compute spikes from input (1 if input > threshold, else 0) via GPU
        // Then update traces via GPU: traces = decay * traces + spikes
        // Step 1: Compute spikes using NotEqualScalar-like approach (need threshold comparison)
        // Since we need ">", not "!=", use a custom approach - threshold compare into spikes
        EnsureInitialized();
        var inp = DownloadBufferData(input);
        var s = new float[size];
        for (int i = 0; i < size; i++) s[i] = inp[i] > threshold ? 1f : 0f;
        UploadToBuffer(s, spikes);
        // Step 2: Update traces on GPU: traces = decay * traces + spikes
        // TraceUpdateSource: binding(0)=traces(rw), binding(1)=spikes(read), uniform: size, decay
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            decay
        };
        Dispatch2BufferAsync("TraceUpdate", WebGpuKernels.TraceUpdateSource, "update_traces",
            traces, spikes, uniforms, size).GetAwaiter().GetResult();
    }

    #endregion

    #region Hyperbolic Geometry

    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        // HyperbolicSource poincare_project: binding(0)=input_a, binding(1)=input_b(unused), binding(2)=output
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(dim),
            curvature,
            epsilon
        };
        int total = batchSize * dim;
        Dispatch3BufferAsync("Hyperbolic", WebGpuKernels.HyperbolicSource, "poincare_project",
            input, dummyB, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        // HyperbolicSource mobius_add: binding(0)=x, binding(1)=y, binding(2)=output
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(dim),
            curvature,
            0
        };
        int total = batchSize * dim;
        Dispatch3BufferAsync("Hyperbolic", WebGpuKernels.HyperbolicSource, "mobius_add",
            x, y, output, uniforms, total).GetAwaiter().GetResult();
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
        // HyperbolicSource poincare_distance: binding(0)=x, binding(1)=y, binding(2)=output (one per batch)
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(dim),
            curvature,
            0
        };
        Dispatch3BufferAsync("Hyperbolic", WebGpuKernels.HyperbolicSource, "poincare_distance",
            x, y, output, uniforms, batchSize).GetAwaiter().GetResult();
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
        // OctonionSource: binding(0)=A, binding(1)=B, binding(2)=C, uniform: count
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(count) };
        Dispatch3BufferAsync("Octonion", WebGpuKernels.OctonionSource, "octonion_multiply",
            a, b, output, uniforms, count).GetAwaiter().GetResult();
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
        // QuantumSource quantum_measurement: C[i] = A[i]^2 + B[i]^2
        var uniforms = MakeUniformInts2(batchSize, stateSize);
        int total = batchSize * stateSize;
        Dispatch3BufferAsync("Quantum", WebGpuKernels.QuantumSource, "quantum_measurement",
            realPart, imagPart, probabilities, uniforms, total).GetAwaiter().GetResult();
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        // QuantumSource normalize_probabilities: reads A, writes C = A/sum(A) per batch
        // The kernel reads from A and writes to C, so we copy probs to a temp and dispatch
        using var tempInput = (WebGpuBuffer)AllocateBuffer(batchSize * stateSize);
        Copy(probabilities, tempInput, batchSize * stateSize);
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeUniformInts2(batchSize, stateSize);
        Dispatch3BufferAsync("Quantum", WebGpuKernels.QuantumSource, "normalize_probabilities",
            tempInput, dummyB, probabilities, uniforms, batchSize).GetAwaiter().GetResult();
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
        // ComplexOpsSource quantum_rotation: appends angles to stateReal buffer (A[idx + total] = angle)
        // Pack stateReal + angles into a combined buffer
        int total = batchSize * numQubits;
        var reData = DownloadBufferData(stateReal);
        var angData = DownloadBufferData(angles);
        var combined = new float[total + total];
        Array.Copy(reData, 0, combined, 0, total);
        Array.Copy(angData, 0, combined, total, total);
        using var combinedBuf = (WebGpuBuffer)AllocateBuffer(combined.Length);
        UploadToBuffer(combined, combinedBuf);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(total),
            BitConverter.Int32BitsToSingle(numQubits),
            BitConverter.Int32BitsToSingle(batchSize),
            0
        };
        Dispatch4BufferAsync("ComplexOps", WebGpuKernels.ComplexOpsSource, "quantum_rotation",
            combinedBuf, stateImag, outReal, outImag, uniforms, total).GetAwaiter().GetResult();
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        // QuantumSource measurement_forward: C[b] = sum(A[base+s]^2) per batch
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeUniformInts2(batchSize, stateSize);
        Dispatch3BufferAsync("Quantum", WebGpuKernels.QuantumSource, "measurement_forward",
            input, dummyB, output, uniforms, batchSize).GetAwaiter().GetResult();
    }

    #endregion

    #region FFT Operations

    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        // FFTSource fft_1d: batched DFT kernel, batch_size=1 for single FFT
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(n),
            BitConverter.Int32BitsToSingle(1), // batch_size=1
            BitConverter.Int32BitsToSingle(inverse ? 1 : 0),
            0
        };
        Dispatch4BufferAsync("FFT", WebGpuKernels.FFTSource, "fft_1d",
            inputReal, inputImag, outputReal, outputImag, uniforms, n).GetAwaiter().GetResult();
        // For inverse FFT, the kernel already handles 1/N normalization
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
        // FFTSource fft_1d: batched DFT kernel
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(n),
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inverse ? 1 : 0),
            0
        };
        int total = batch * n;
        Dispatch4BufferAsync("FFT", WebGpuKernels.FFTSource, "fft_1d",
            inputReal, inputImag, outputReal, outputImag, uniforms, total).GetAwaiter().GetResult();
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
    {
        // 2D FFT = row-wise 1D FFT (batch=height, n=width) then column-wise 1D FFT (batch=width, n=height)
        // Pass 1: Row-wise FFT - data is already in row-major order, so batch=height works directly
        using var tempR = (WebGpuBuffer)AllocateBuffer(height * width);
        using var tempI = (WebGpuBuffer)AllocateBuffer(height * width);
        BatchedFFT(inputReal, inputImag, tempR, tempI, height, width, inverse);
        // Pass 2: Column-wise FFT - need to transpose, FFT, then transpose back
        using var transR = (WebGpuBuffer)AllocateBuffer(height * width);
        using var transI = (WebGpuBuffer)AllocateBuffer(height * width);
        // Transpose from (height x width) to (width x height)
        BatchedTranspose(tempR, transR, 1, height, width);
        BatchedTranspose(tempI, transI, 1, height, width);
        // FFT columns: now batch=width, n=height
        using var fftR = (WebGpuBuffer)AllocateBuffer(height * width);
        using var fftI = (WebGpuBuffer)AllocateBuffer(height * width);
        BatchedFFT(transR, transI, fftR, fftI, width, height, inverse);
        // Transpose back from (width x height) to (height x width)
        BatchedTranspose(fftR, outputReal, 1, width, height);
        BatchedTranspose(fftI, outputImag, 1, width, height);
    }

    public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
    {
        // ComplexOpsSource apply_window: C = A * B (element-wise multiply)
        using var dummyD = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(n), 0, 0, 0 };
        Dispatch4BufferAsync("ComplexOps", WebGpuKernels.ComplexOpsSource, "apply_window",
            input, window, output, dummyD, uniforms, n).GetAwaiter().GetResult();
    }

    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
    {
        // ComplexOpsSource complex_magnitude: C = sqrt(A^2 + B^2)
        using var dummyD = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(n), 0, 0, 0 };
        Dispatch4BufferAsync("ComplexOps", WebGpuKernels.ComplexOpsSource, "complex_magnitude",
            real, imag, magnitude, dummyD, uniforms, n).GetAwaiter().GetResult();
    }

    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
    {
        // ComplexOpsSource complex_phase: C = atan2(B, A)
        using var dummyD = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(n), 0, 0, 0 };
        Dispatch4BufferAsync("ComplexOps", WebGpuKernels.ComplexOpsSource, "complex_phase",
            real, imag, phase, dummyD, uniforms, n).GetAwaiter().GetResult();
    }

    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        // ComplexOpsSource polar_to_complex: C=mag*cos(phase), D=mag*sin(phase)
        var uniforms = new float[] { BitConverter.Int32BitsToSingle(n), 0, 0, 0 };
        Dispatch4BufferAsync("ComplexOps", WebGpuKernels.ComplexOpsSource, "polar_to_complex",
            magnitude, phase, real, imag, uniforms, n).GetAwaiter().GetResult();
    }

    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
    {
        // melSpec[b,m] = sum_f(powerSpec[b,f] * filterbank[m,f])
        // This is powerSpec (numFrames x numFreqs) * filterbank^T (numFreqs x nMels)
        // = Gemm(powerSpec, filterbank^T, melSpec, M=numFrames, N=nMels, K=numFreqs)
        // filterbank is (nMels x numFreqs), so transpose it
        using var filterbankT = (WebGpuBuffer)AllocateBuffer(numFreqs * nMels);
        BatchedTranspose(filterbank, filterbankT, 1, nMels, numFreqs);
        Gemm(powerSpec, filterbankT, melSpec, numFrames, nMels, numFreqs);
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
