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
        var dummy = SharedDummyBuffer;
        var uniforms = MakeOptimizerUniforms(size, learningRate, 0, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "sgd",
            param, gradient, dummy, dummy, uniforms, size).GetAwaiter().GetResult();
    }

    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "sgd_momentum",
            param, gradient, velocity, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
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
        // beta1=rho for rmsprop kernel
        var uniforms = MakeOptimizerUniforms(size, learningRate, rho, 0, epsilon, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "rmsprop",
            param, gradient, squaredAvg, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, 0, 0, epsilon, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "adagrad",
            param, gradient, accumulatedGrad, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
    }

    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "nag",
            param, gradient, velocity, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
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
        var uniforms = MakeOptimizerUniforms(size, learningRate, momentum, 0, 0, weightDecay, 0);
        // Replace the last uniform (extra) with localLr
        uniforms[7] = localLr;
        Dispatch4BufferAsync("LarsLambFtrl", WebGpuKernels.LarsLambFtrlSource, "lars_update",
            param, gradient, velocity, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
    }

    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        // Two-pass GPU: compute param_norm and update_norm entirely on GPU
        // Step 1: param_norm via GPU square + sum
        using var paramSq = (WebGpuBuffer)AllocateBuffer(size);
        SquareAsync(param, paramSq, size).GetAwaiter().GetResult();
        float pNormSq = SumAsync(paramSq, size).GetAwaiter().GetResult();
        float pNorm = MathF.Sqrt(pNormSq);
        // Step 2: update_norm via GPU kernel that computes Adam update^2 per element, then sum
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        using var updateSq = (WebGpuBuffer)AllocateBuffer(size);
        var lambNormUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            beta1, beta2, epsilon,
            bc1, bc2, weightDecay, 0
        };
        Dispatch4BufferAsync("LambNorm", WebGpuKernels.LambNormSource, "lamb_update_sq",
            m, v, gradient, updateSq, lambNormUniforms, size).GetAwaiter().GetResult();
        float uNormSq = SumAsync(updateSq, size).GetAwaiter().GetResult();
        float uNorm = MathF.Sqrt(uNormSq);
        float ratio = uNorm > 0 ? pNorm / uNorm : 1f;
        // Step 3: Apply LAMB update with pre-computed trust ratio
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
        // Lion kernel uses state1=m, does not use state2
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, 0, weightDecay, 0);
        Dispatch4BufferAsync("AdditionalOptimizer", WebGpuKernels.AdditionalOptimizerSource, "lion",
            param, gradient, m, SharedDummyBuffer, uniforms, size).GetAwaiter().GetResult();
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

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // FP32 -> FP16 emulation: truncate mantissa to 10 bits via GPU kernel.
        // WebGPU buffers are always f32, so we emulate FP16 by reducing precision
        // (zeroing the bottom 13 mantissa bits). This matches IEEE 754 half-precision
        // rounding behavior (round-toward-zero) and handles overflow/underflow.
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            BitConverter.Int32BitsToSingle(0) // direction=0: to_fp16
        };
        Dispatch2BufferAsync("Fp16Convert", WebGpuKernels.Fp16ConvertSource, "fp16_convert",
            input, output, uniforms, size).GetAwaiter().GetResult();
    }

    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        // FP16 -> FP32: data is already stored as f32 with FP16-truncated precision.
        // Simply copy the values through (they are valid f32 representations of FP16 values).
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            BitConverter.Int32BitsToSingle(1) // direction=1: from_fp16
        };
        Dispatch2BufferAsync("Fp16Convert", WebGpuKernels.Fp16ConvertSource, "fp16_convert",
            input, output, uniforms, size).GetAwaiter().GetResult();
    }

    #endregion

    #region Specialized Layer Operations

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        // SpecializedLayerSource rbf_forward: binding(0)=input, binding(1)=centers+epsilons combined, binding(2)=output
        // Pack centers and epsilons via GPU Copy with offsets
        int centersLen = numCenters * inputDim;
        using var combinedBuf = (WebGpuBuffer)AllocateBuffer(centersLen + numCenters);
        Copy(centers, 0, combinedBuf, 0, centersLen);
        Copy(epsilons, 0, combinedBuf, centersLen, numCenters);
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
        // The STDP rule uses both pre-synaptic and post-synaptic traces:
        //   dw += a_plus * pre_trace[i]   when post_spike[j] fires (LTP)
        //   dw -= a_minus * post_trace[j] when pre_spike[i] fires (LTD)
        // The kernel currently reads pre_traces at binding(3). We pack both trace buffers
        // into the preTrace binding by interleaving or using postTrace via the kernel.
        // Since the kernel indexes pre_traces[j] for LTD (using post-neuron index on pre_traces),
        // we need to replace that with post_traces[j]. Pack postTrace into the kernel's pre_traces
        // binding at offset numPre so the kernel can read post_traces[j] = pre_traces[numPre + j].
        // Actually, the simpler fix: update the kernel to use a 5-buffer dispatch.
        // For now, combine preTrace and postTrace into a single buffer.
        using var combinedTraces = (WebGpuBuffer)AllocateBuffer(numPre + numPost);
        Copy(preTrace, 0, combinedTraces, 0, numPre);
        Copy(postTrace, 0, combinedTraces, numPre, numPost);
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
            weights, preSpike, postSpike, combinedTraces, uniforms, total).GetAwaiter().GetResult();
        // Apply homeostasis and clamping via GPU dispatch
        if (homeostasisRate != 0f || minWeight != float.MinValue || maxWeight != float.MaxValue)
        {
            using var dummyR = (WebGpuBuffer)AllocateBuffer(1);
            var hcUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(total),
                homeostasisRate,
                minWeight,
                maxWeight
            };
            Dispatch2BufferAsync("HomeostasisClamp", WebGpuKernels.HomeostasisClampSource, "homeostasis_clamp",
                weights, dummyR, hcUniforms, total).GetAwaiter().GetResult();
        }
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        // Step 1: Compute spikes via GPU threshold kernel: spikes[i] = input[i] > threshold ? 1 : 0
        var threshUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(size),
            threshold, 0, 0
        };
        Dispatch2BufferAsync("ThresholdStep", WebGpuKernels.ThresholdStepSource, "threshold_step",
            input, spikes, threshUniforms, size).GetAwaiter().GetResult();
        // Step 2: Update traces on GPU: traces = decay * traces + spikes
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
        // PoincareExpMapSource poincare_exp_map: binding(0)=base_pt, binding(1)=tangent, binding(2)=output
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(dim),
            curvature, 0
        };
        Dispatch3BufferAsync("PoincareExpMap", WebGpuKernels.PoincareExpMapSource, "poincare_exp_map",
            basePoint, tangentVec, output, uniforms, batchSize).GetAwaiter().GetResult();
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
        using var matResult = (WebGpuBuffer)AllocateBuffer(batchSize * outputFeatures);
        Gemm(input, weights, matResult, batchSize, outputFeatures, inputFeatures);
        using var biasResult = (WebGpuBuffer)AllocateBuffer(batchSize * outputFeatures);
        BiasAdd(matResult, biases, biasResult, batchSize, outputFeatures);
        PoincareProject(biasResult, output, batchSize, outputFeatures, curvature, epsilon);
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // Hyperbolic linear backward for input: gradInput = gradOutput * weights^T
        // This is the Euclidean gradient; the Riemannian correction from Poincare projection
        // is left to the optimizer's Riemannian gradient step.
        using var weightsT = (WebGpuBuffer)AllocateBuffer(outputFeatures * inputFeatures);
        BatchedTranspose(weights, weightsT, 1, inputFeatures, outputFeatures);
        Gemm(gradOutput, weightsT, gradInput, batchSize, inputFeatures, outputFeatures);
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradWeights = input^T * gradOutput
        using var inputT = (WebGpuBuffer)AllocateBuffer(inputFeatures * batchSize);
        BatchedTranspose(input, inputT, 1, batchSize, inputFeatures);
        Gemm(inputT, gradOutput, gradWeights, inputFeatures, outputFeatures, batchSize);
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradBiases = sum of gradOutput along batch dimension
        Fill(gradBiases, 0f, outputFeatures);
        for (int b = 0; b < batchSize; b++)
        {
            using var slice = (WebGpuBuffer)AllocateBuffer(outputFeatures);
            Copy(gradOutput, b * outputFeatures, slice, 0, outputFeatures);
            Add(gradBiases, slice, gradBiases, outputFeatures);
        }
    }

    #endregion

    #region Octonion Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        // OctonionSource: binding(0)=A, binding(1)=B, binding(2)=C, uniform: count
        Dispatch3BufferAsync("Octonion", WebGpuKernels.OctonionSource, "octonion_multiply",
            a, b, output, MakeUniform1(count), count).GetAwaiter().GetResult();
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count) => Add(a, b, output, count * 8);

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        Gemm(input, weights, output, batchSize, outputFeatures, inputFeatures);
        BiasAdd(output, biases, output, batchSize, outputFeatures);
    }

    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        // gradInput = gradOutput * weights^T
        // gradOutput: (batchSize x outputFeatures), weights: (inputFeatures x outputFeatures)
        // gradInput: (batchSize x inputFeatures) = gradOutput * weights^T
        using var weightsT = (WebGpuBuffer)AllocateBuffer(outputFeatures * inputFeatures);
        BatchedTranspose(weights, weightsT, 1, inputFeatures, outputFeatures);
        Gemm(gradOutput, weightsT, gradInput, batchSize, inputFeatures, outputFeatures);
    }

    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        // gradWeights = input^T * gradOutput
        // input: (batchSize x inputFeatures), gradOutput: (batchSize x outputFeatures)
        // gradWeights: (inputFeatures x outputFeatures) = input^T * gradOutput
        using var inputT = (WebGpuBuffer)AllocateBuffer(inputFeatures * batchSize);
        BatchedTranspose(input, inputT, 1, batchSize, inputFeatures);
        Gemm(inputT, gradOutput, gradWeights, inputFeatures, outputFeatures, batchSize);
    }

    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures)
    {
        // gradBiases = sum of gradOutput along batch dimension
        // gradBiases[j] = sum_b(gradOutput[b, j])
        // Use MeanAxis * batchSize or just reduce along axis 0
        MeanAxis(gradOutput, gradBiases, outputFeatures, batchSize);
        // MeanAxis divides by batchSize; we want the sum, so scale by batchSize
        // Actually, for bias gradient, the mean is fine for averaged gradients.
        // Standard practice: gradBiases = sum over batch, then typically divided by batch later.
        // We compute the sum directly by treating gradOutput as (outputFeatures x batchSize) and summing.
        // Use a different approach: reduce along the batch dimension
        Fill(gradBiases, 0f, outputFeatures);
        // Sum rows: for each output feature, sum over all batch elements
        for (int b = 0; b < batchSize; b++)
        {
            // gradBiases += gradOutput[b, :]
            using var slice = (WebGpuBuffer)AllocateBuffer(outputFeatures);
            Copy(gradOutput, b * outputFeatures, slice, 0, outputFeatures);
            Add(gradBiases, slice, gradBiases, outputFeatures);
        }
    }

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
        // ComplexMatVecSource reads separate real/imag buffers directly — no interleave needed
        int total = batchSize * dim;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(dim),
            0, 0
        };
        Dispatch6BufferAsync("ComplexMatVec", WebGpuKernels.ComplexMatVecSource, "complex_mat_vec",
            matReal, matImag, vecReal, vecImag, outReal, outImag, uniforms, total).GetAwaiter().GetResult();
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        // ComplexOpsSource quantum_rotation: appends angles to stateReal buffer (A[idx + total] = angle)
        // Pack stateReal + angles into combined buffer via GPU Copy
        int total = batchSize * numQubits;
        using var combinedBuf = (WebGpuBuffer)AllocateBuffer(total * 2);
        Copy(stateReal, 0, combinedBuf, 0, total);
        Copy(angles, 0, combinedBuf, total, total);
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
        using var zeros = (WebGpuBuffer)AllocateBuffer(n);
        Fill(zeros, 0f, n);
        FFT(input, zeros, outputReal, outputImag, n, false);
    }

    public void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
    {
        using var tempImag = (WebGpuBuffer)AllocateBuffer(n);
        FFT(inputReal, inputImag, output, tempImag, n, true);
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
    {
        // PowerDbSource power_to_db: binding(0)=A(power), binding(1)=B(db), uniform: size, ref_value, min_db
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(n),
            refValue, minDb, 0
        };
        Dispatch2BufferAsync("PowerDb", WebGpuKernels.PowerDbSource, "power_to_db",
            power, db, uniforms, n).GetAwaiter().GetResult();
    }

    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
    {
        // PowerDbSource db_to_power: binding(0)=A(db), binding(1)=B(power), uniform: size, ref_value
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(n),
            refValue, 0, 0
        };
        Dispatch2BufferAsync("PowerDb", WebGpuKernels.PowerDbSource, "db_to_power",
            db, power, uniforms, n).GetAwaiter().GetResult();
    }

    #endregion

    #region RNN Operations

    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // RnnCellSource lstm_cell: per-timestep GPU dispatch
        // Pack weights into combined buffer via GPU: [W_ih (4*hs x in_s), W_hh (4*hs x hs), bias (4*hs)]
        int biasLen = 4 * hiddenSize;
        int wihLen = 4 * hiddenSize * inputSize;
        int whhLen = 4 * hiddenSize * hiddenSize;
        using var weightsBuf = (WebGpuBuffer)AllocateBuffer(wihLen + whhLen + biasLen);
        Copy(weightsIh, 0, weightsBuf, 0, wihLen);
        Copy(weightsHh, 0, weightsBuf, wihLen, whhLen);
        // bias = biasIh + biasHh: add on GPU into the weights buffer at offset
        using var biasCombined = (WebGpuBuffer)AllocateBuffer(biasLen);
        Add(biasIh, biasHh, biasCombined, biasLen);
        Copy(biasCombined, 0, weightsBuf, wihLen + whhLen, biasLen);

        // hBuf holds current hidden state, cBuf holds cell state (state buffer)
        using var hBuf = (WebGpuBuffer)AllocateBuffer(batch * hiddenSize);
        using var cBuf = (WebGpuBuffer)AllocateBuffer(batch * hiddenSize);
        Copy(hInit, hBuf, batch * hiddenSize);
        Copy(cInit, cBuf, batch * hiddenSize);

        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inputSize),
            BitConverter.Int32BitsToSingle(hiddenSize),
            0
        };
        int cellTotal = batch * hiddenSize;

        // Extract per-timestep input slices and dispatch
        for (int t = 0; t < seqLen; t++)
        {
            // Extract input for timestep t via GPU strided slice: input[b, t, :] -> inputSlice[b, :]
            using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            var sliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),  // src_stride per batch
                BitConverter.Int32BitsToSingle(inputSize),           // dst_stride per batch
                BitConverter.Int32BitsToSingle(t * inputSize),       // src_offset (timestep t)
                BitConverter.Int32BitsToSingle(0),                   // dst_offset
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                input, inputSlice, sliceUniforms, batch * inputSize).GetAwaiter().GetResult();

            // lstm_cell kernel: input=inputSlice, weights=packedWeights, state=cBuf, output=hBuf
            Dispatch4BufferAsync("RnnCell", WebGpuKernels.RnnCellSource, "lstm_cell",
                inputSlice, weightsBuf, cBuf, hBuf, uniforms, cellTotal).GetAwaiter().GetResult();

            // Copy h and c to allH and allC for this timestep
            Copy(hBuf, 0, allH, t * batch * hiddenSize, batch * hiddenSize);
            Copy(cBuf, 0, allC, t * batch * hiddenSize, batch * hiddenSize);

            // Scatter h to output via GPU strided slice: hBuf[b, :] -> output[b, t, :]
            var scatterUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(hiddenSize),
                BitConverter.Int32BitsToSingle(hiddenSize),          // src_stride per batch
                BitConverter.Int32BitsToSingle(seqLen * hiddenSize), // dst_stride per batch
                BitConverter.Int32BitsToSingle(0),                   // src_offset
                BitConverter.Int32BitsToSingle(t * hiddenSize),      // dst_offset (timestep t)
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                hBuf, output, scatterUniforms, batch * hiddenSize).GetAwaiter().GetResult();
        }

        Copy(hBuf, hFinal, batch * hiddenSize);
        Copy(cBuf, cFinal, batch * hiddenSize);
        // cacheGates not populated by this kernel path; fill with zeros for compatibility
        Fill(cacheGates, 0f, seqLen * batch * 4 * hiddenSize);
    }

    public void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // LSTM Backpropagation Through Time (BPTT)
        // Initialize gradient accumulators to zero
        int hs = hiddenSize;
        int cellTotal = batch * hs;
        Fill(gradInput, 0f, batch * seqLen * inputSize);
        Fill(gradWeightsIh, 0f, 4 * hs * inputSize);
        Fill(gradWeightsHh, 0f, 4 * hs * hs);
        Fill(gradBiasIh, 0f, 4 * hs);
        Fill(gradBiasHh, 0f, 4 * hs);

        // Running gradient of hidden state and cell state
        using var gradH = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var gradC = (WebGpuBuffer)AllocateBuffer(cellTotal);
        Fill(gradH, 0f, cellTotal);
        Fill(gradC, 0f, cellTotal);

        // Download weights and cell states once for CPU-side gradient accumulation
        // (allC is not mutated during backward, so download it once outside the loop)
        var wihData = DownloadBufferData(weightsIh);
        var whhData = DownloadBufferData(weightsHh);
        var allCData = DownloadBufferData(allC);

        // Download weight gradient accumulators once before the loop to avoid
        // O(seqLen) GPU roundtrips. We accumulate in CPU arrays and upload once after.
        var gWihData = DownloadBufferData(gradWeightsIh);
        var gWhhData = DownloadBufferData(gradWeightsHh);
        var gBiasData = DownloadBufferData(gradBiasIh);

        // Iterate backwards through time
        for (int t = seqLen - 1; t >= 0; t--)
        {
            // Add gradient from output at this timestep to gradH
            // gradOutput[b, t, :] -> gradH[b, :]
            using var gradOutSlice = (WebGpuBuffer)AllocateBuffer(cellTotal);
            var sliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(hs),
                BitConverter.Int32BitsToSingle(seqLen * hs),
                BitConverter.Int32BitsToSingle(hs),
                BitConverter.Int32BitsToSingle(t * hs),
                BitConverter.Int32BitsToSingle(0),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                gradOutput, gradOutSlice, sliceUniforms, cellTotal).GetAwaiter().GetResult();
            Add(gradH, gradOutSlice, gradH, cellTotal);

            // Get h_prev and c_prev for this timestep
            // c_prev = t > 0 ? allC[(t-1)*batch*hs : t*batch*hs] : cInit
            using var cPrev = (WebGpuBuffer)AllocateBuffer(cellTotal);
            if (t > 0)
                Copy(allC, (t - 1) * cellTotal, cPrev, 0, cellTotal);
            else
                Copy(cInit, cPrev, cellTotal);

            // Download current states for CPU computation of gate gradients
            var hData = DownloadBufferData(gradH);
            var cData = DownloadBufferData(gradC);
            var cPrevData = DownloadBufferData(cPrev);

            // Recompute gates for this timestep from allH, allC, input
            // h_prev for this timestep
            using var hPrev = (WebGpuBuffer)AllocateBuffer(cellTotal);
            if (t > 0)
                Copy(allH, (t - 1) * cellTotal, hPrev, 0, cellTotal);
            else
                Copy(hInit, hPrev, cellTotal);

            var hPrevData = DownloadBufferData(hPrev);

            // Get input slice for this timestep
            using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            var inSliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(t * inputSize),
                BitConverter.Int32BitsToSingle(0),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                input, inputSlice, inSliceUniforms, batch * inputSize).GetAwaiter().GetResult();
            var inputData = DownloadBufferData(inputSlice);

            // Recompute gates and compute gradients on CPU
            var gradGates = new float[batch * 4 * hs]; // [di, df, dg, do] per element
            var newGradC = new float[cellTotal];
            var newGradH = new float[cellTotal];
            var gradInputT = new float[batch * inputSize];

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < hs; h++)
                {
                    int idx = b * hs + h;
                    // Recompute gates
                    float[] gates = new float[4];
                    for (int gate = 0; gate < 4; gate++)
                    {
                        float val = 0;
                        int wihOff = (gate * hs + h) * inputSize;
                        for (int j = 0; j < inputSize; j++)
                            val += wihData[wihOff + j] * inputData[b * inputSize + j];
                        int whhOff = (gate * hs + h) * hs;
                        for (int j = 0; j < hs; j++)
                            val += whhData[whhOff + j] * hPrevData[b * hs + j];
                        gates[gate] = val;
                    }
                    float iGate = 1f / (1f + MathF.Exp(-gates[0]));
                    float fGate = 1f / (1f + MathF.Exp(-gates[1]));
                    float gGate = MathF.Tanh(gates[2]);
                    float oGate = 1f / (1f + MathF.Exp(-gates[3]));
                    float cNew = allCData[t * cellTotal + idx];
                    float tanhC = MathF.Tanh(cNew);

                    float dh = hData[idx];
                    float dcNext = cData[idx];

                    // dc = dh * o * (1 - tanh(c)^2) + dc_next
                    float dc = dh * oGate * (1 - tanhC * tanhC) + dcNext;

                    // Gate gradients
                    float di = dc * gGate * iGate * (1 - iGate);
                    float df = dc * cPrevData[idx] * fGate * (1 - fGate);
                    float dg = dc * iGate * (1 - gGate * gGate);
                    float dOGate = dh * tanhC * oGate * (1 - oGate);

                    gradGates[b * 4 * hs + 0 * hs + h] = di;
                    gradGates[b * 4 * hs + 1 * hs + h] = df;
                    gradGates[b * 4 * hs + 2 * hs + h] = dg;
                    gradGates[b * 4 * hs + 3 * hs + h] = dOGate;

                    // Gradient for c_prev
                    newGradC[idx] = dc * fGate;

                    // Gradient for h_prev (through W_hh)
                    // Will be accumulated below
                }
            }

            // Compute gradInput for this timestep: gradInput_t = gradGates * W_ih^T
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    float sum = 0;
                    for (int gate = 0; gate < 4; gate++)
                    {
                        for (int h = 0; h < hs; h++)
                        {
                            int wihOff = (gate * hs + h) * inputSize + j;
                            sum += gradGates[b * 4 * hs + gate * hs + h] * wihData[wihOff];
                        }
                    }
                    gradInputT[b * inputSize + j] = sum;
                }
            }

            // Upload gradInput for this timestep
            using var gradInputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            UploadToBuffer(gradInputT, gradInputSlice);
            // Scatter to gradInput[b, t, :]
            var scatterUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),
                BitConverter.Int32BitsToSingle(0),
                BitConverter.Int32BitsToSingle(t * inputSize),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                gradInputSlice, gradInput, scatterUniforms, batch * inputSize).GetAwaiter().GetResult();

            // Compute gradH_prev = gradGates * W_hh^T
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < hs; j++)
                {
                    float sum = 0;
                    for (int gate = 0; gate < 4; gate++)
                    {
                        for (int h = 0; h < hs; h++)
                        {
                            int whhOff = (gate * hs + h) * hs + j;
                            sum += gradGates[b * 4 * hs + gate * hs + h] * whhData[whhOff];
                        }
                    }
                    newGradH[b * hs + j] = sum;
                }
            }

            // Accumulate weight gradients into CPU arrays (hoisted outside the loop)
            for (int b = 0; b < batch; b++)
            {
                for (int gate = 0; gate < 4; gate++)
                {
                    for (int h = 0; h < hs; h++)
                    {
                        float dGate = gradGates[b * 4 * hs + gate * hs + h];
                        // W_ih gradient
                        for (int j = 0; j < inputSize; j++)
                            gWihData[(gate * hs + h) * inputSize + j] += dGate * inputData[b * inputSize + j];
                        // W_hh gradient
                        for (int j = 0; j < hs; j++)
                            gWhhData[(gate * hs + h) * hs + j] += dGate * hPrevData[b * hs + j];
                        // Bias gradient
                        gBiasData[gate * hs + h] += dGate;
                    }
                }
            }

            // Update running gradients
            UploadToBuffer(newGradH, gradH);
            UploadToBuffer(newGradC, gradC);
        }

        // Upload accumulated weight gradients once after the loop completes
        UploadToBuffer(gWihData, gradWeightsIh);
        UploadToBuffer(gWhhData, gradWeightsHh);
        UploadToBuffer(gBiasData, gradBiasIh);

        // Copy final gradients
        Copy(gradH, gradHInit, cellTotal);
        Copy(gradC, gradCInit, cellTotal);
        // gradBiasHh = gradBiasIh (same bias gradient accumulation)
        Copy(gradBiasIh, gradBiasHh, 4 * hs);
    }

    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // RnnCellSource gru_cell: per-timestep GPU dispatch
        // Pack weights into combined buffer via GPU: [W_ih (3*hs x in_s), W_hh (3*hs x hs), bias (3*hs)]
        int biasLen = 3 * hiddenSize;
        int wihLen = 3 * hiddenSize * inputSize;
        int whhLen = 3 * hiddenSize * hiddenSize;
        using var weightsBuf = (WebGpuBuffer)AllocateBuffer(wihLen + whhLen + biasLen);
        Copy(weightsIh, 0, weightsBuf, 0, wihLen);
        Copy(weightsHh, 0, weightsBuf, wihLen, whhLen);
        // bias = biasIh + biasHh: add on GPU
        using var biasCombined = (WebGpuBuffer)AllocateBuffer(biasLen);
        Add(biasIh, biasHh, biasCombined, biasLen);
        Copy(biasCombined, 0, weightsBuf, wihLen + whhLen, biasLen);

        // hBuf holds current hidden state
        using var hBuf = (WebGpuBuffer)AllocateBuffer(batch * hiddenSize);
        // For gru_cell kernel, state buffer is unused (dummy) since GRU has no cell state
        using var dummyState = (WebGpuBuffer)AllocateBuffer(batch * hiddenSize);
        Copy(hInit, hBuf, batch * hiddenSize);

        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inputSize),
            BitConverter.Int32BitsToSingle(hiddenSize),
            BitConverter.Int32BitsToSingle(1) // is_gru=1
        };
        int cellTotal = batch * hiddenSize;

        for (int t = 0; t < seqLen; t++)
        {
            // Extract input for timestep t via GPU strided slice: input[b, t, :] -> inputSlice[b, :]
            using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            var sliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),  // src_stride per batch
                BitConverter.Int32BitsToSingle(inputSize),           // dst_stride per batch
                BitConverter.Int32BitsToSingle(t * inputSize),       // src_offset (timestep t)
                BitConverter.Int32BitsToSingle(0),                   // dst_offset
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                input, inputSlice, sliceUniforms, batch * inputSize).GetAwaiter().GetResult();

            // gru_cell kernel: input=inputSlice, weights=packedWeights, state=dummyState, output=hBuf
            Dispatch4BufferAsync("RnnCell", WebGpuKernels.RnnCellSource, "gru_cell",
                inputSlice, weightsBuf, dummyState, hBuf, uniforms, cellTotal).GetAwaiter().GetResult();

            // Copy h to allH for this timestep
            Copy(hBuf, 0, allH, t * batch * hiddenSize, batch * hiddenSize);

            // Scatter h to output via GPU strided slice: hBuf[b, :] -> output[b, t, :]
            var scatterUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(hiddenSize),
                BitConverter.Int32BitsToSingle(hiddenSize),          // src_stride per batch
                BitConverter.Int32BitsToSingle(seqLen * hiddenSize), // dst_stride per batch
                BitConverter.Int32BitsToSingle(0),                   // src_offset
                BitConverter.Int32BitsToSingle(t * hiddenSize),      // dst_offset (timestep t)
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                hBuf, output, scatterUniforms, batch * hiddenSize).GetAwaiter().GetResult();
        }

        Copy(hBuf, hFinal, batch * hiddenSize);
        // cacheGates not populated by this kernel path; fill with zeros for compatibility
        Fill(cacheGates, 0f, seqLen * batch * 3 * hiddenSize);
    }

    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // GRU Backpropagation Through Time (BPTT)
        int hs = hiddenSize;
        int cellTotal = batch * hs;
        Fill(gradInput, 0f, batch * seqLen * inputSize);
        Fill(gradWeightsIh, 0f, 3 * hs * inputSize);
        Fill(gradWeightsHh, 0f, 3 * hs * hs);
        Fill(gradBiasIh, 0f, 3 * hs);
        Fill(gradBiasHh, 0f, 3 * hs);

        using var gradH = (WebGpuBuffer)AllocateBuffer(cellTotal);
        Fill(gradH, 0f, cellTotal);

        var wihData = DownloadBufferData(weightsIh);
        var whhData = DownloadBufferData(weightsHh);

        for (int t = seqLen - 1; t >= 0; t--)
        {
            // Add gradient from output at this timestep
            using var gradOutSlice = (WebGpuBuffer)AllocateBuffer(cellTotal);
            var sliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(hs),
                BitConverter.Int32BitsToSingle(seqLen * hs),
                BitConverter.Int32BitsToSingle(hs),
                BitConverter.Int32BitsToSingle(t * hs),
                BitConverter.Int32BitsToSingle(0),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                gradOutput, gradOutSlice, sliceUniforms, cellTotal).GetAwaiter().GetResult();
            Add(gradH, gradOutSlice, gradH, cellTotal);

            // Get h_prev
            using var hPrev = (WebGpuBuffer)AllocateBuffer(cellTotal);
            if (t > 0)
                Copy(allH, (t - 1) * cellTotal, hPrev, 0, cellTotal);
            else
                Copy(dHBuffer, hPrev, cellTotal); // dHBuffer holds hInit initially

            var hPrevData = DownloadBufferData(hPrev);
            var dhData = DownloadBufferData(gradH);

            // Get input slice
            using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            var inSliceUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(t * inputSize),
                BitConverter.Int32BitsToSingle(0),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                input, inputSlice, inSliceUniforms, batch * inputSize).GetAwaiter().GetResult();
            var inputData = DownloadBufferData(inputSlice);

            // Recompute GRU gates and compute gradients on CPU
            var gradGates = new float[batch * 3 * hs]; // [dr, dz, dn]
            var newGradH = new float[cellTotal];
            var gradInputT = new float[batch * inputSize];

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < hs; h++)
                {
                    int idx = b * hs + h;
                    // Recompute gates
                    float[] gates = new float[3];
                    for (int gate = 0; gate < 3; gate++)
                    {
                        float val = 0;
                        int wihOff = (gate * hs + h) * inputSize;
                        for (int j = 0; j < inputSize; j++)
                            val += wihData[wihOff + j] * inputData[b * inputSize + j];
                        int whhOff = (gate * hs + h) * hs;
                        for (int j = 0; j < hs; j++)
                            val += whhData[whhOff + j] * hPrevData[b * hs + j];
                        gates[gate] = val;
                    }
                    float rGate = 1f / (1f + MathF.Exp(-gates[0]));
                    float zGate = 1f / (1f + MathF.Exp(-gates[1]));

                    // Recompute n_gate with reset
                    float nVal = 0;
                    int wihNOff = (2 * hs + h) * inputSize;
                    for (int j = 0; j < inputSize; j++)
                        nVal += wihData[wihNOff + j] * inputData[b * inputSize + j];
                    float nHid = 0;
                    int whhNOff = (2 * hs + h) * hs;
                    for (int j = 0; j < hs; j++)
                        nHid += whhData[whhNOff + j] * hPrevData[b * hs + j];
                    float nGate = MathF.Tanh(nVal + rGate * nHid);

                    float dh = dhData[idx];
                    // h_new = (1-z)*n + z*h_prev
                    float dn = dh * (1 - zGate) * (1 - nGate * nGate);
                    float dz = dh * (hPrevData[idx] - nGate) * zGate * (1 - zGate);
                    float dr = dn * nHid * rGate * (1 - rGate);

                    gradGates[b * 3 * hs + 0 * hs + h] = dr;
                    gradGates[b * 3 * hs + 1 * hs + h] = dz;
                    gradGates[b * 3 * hs + 2 * hs + h] = dn;

                    // Gradient for h_prev: dh_prev += dh * z
                    newGradH[idx] = dh * zGate;
                }
            }

            // Compute gradInput for this timestep
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    float sum = 0;
                    for (int gate = 0; gate < 3; gate++)
                    {
                        for (int h = 0; h < hs; h++)
                        {
                            int wihOff = (gate * hs + h) * inputSize + j;
                            sum += gradGates[b * 3 * hs + gate * hs + h] * wihData[wihOff];
                        }
                    }
                    gradInputT[b * inputSize + j] = sum;
                }
            }

            using var gradInputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            UploadToBuffer(gradInputT, gradInputSlice);
            var scatterUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(seqLen * inputSize),
                BitConverter.Int32BitsToSingle(0),
                BitConverter.Int32BitsToSingle(t * inputSize),
                0, 0
            };
            Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
                gradInputSlice, gradInput, scatterUniforms, batch * inputSize).GetAwaiter().GetResult();

            // Accumulate weight gradients
            var gWihData = DownloadBufferData(gradWeightsIh);
            var gWhhData = DownloadBufferData(gradWeightsHh);
            var gBiasData = DownloadBufferData(gradBiasIh);

            for (int b = 0; b < batch; b++)
            {
                for (int gate = 0; gate < 3; gate++)
                {
                    for (int h = 0; h < hs; h++)
                    {
                        float dGate = gradGates[b * 3 * hs + gate * hs + h];
                        for (int j = 0; j < inputSize; j++)
                            gWihData[(gate * hs + h) * inputSize + j] += dGate * inputData[b * inputSize + j];
                        for (int j = 0; j < hs; j++)
                            gWhhData[(gate * hs + h) * hs + j] += dGate * hPrevData[b * hs + j];
                        gBiasData[gate * hs + h] += dGate;
                    }
                }
            }

            UploadToBuffer(gWihData, gradWeightsIh);
            UploadToBuffer(gWhhData, gradWeightsHh);
            UploadToBuffer(gBiasData, gradBiasIh);

            // Compute gradH_prev through W_hh
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < hs; j++)
                {
                    float sum = 0;
                    for (int gate = 0; gate < 3; gate++)
                    {
                        for (int h = 0; h < hs; h++)
                        {
                            int whhOff = (gate * hs + h) * hs + j;
                            sum += gradGates[b * 3 * hs + gate * hs + h] * whhData[whhOff];
                        }
                    }
                    newGradH[b * hs + j] += sum;
                }
            }

            UploadToBuffer(newGradH, gradH);
        }

        Copy(gradH, gradHInit, cellTotal);
        Copy(gradBiasIh, gradBiasHh, 3 * hs);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        // GRU cell backward: compute gate gradients from dH
        int cellTotal = batch * hiddenSize;
        var dhData = DownloadBufferData(gradH);
        var rData = DownloadBufferData(gateR);
        var zData = DownloadBufferData(gateZ);
        var nData = DownloadBufferData(gateN);
        var hPrevData = DownloadBufferData(prevH);
        var whhData = DownloadBufferData(weightsHh);

        var gradRData = new float[cellTotal];
        var gradZData = new float[cellTotal];
        var gradNData = new float[cellTotal];
        var gradPrevHData = new float[cellTotal];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                int idx = b * hiddenSize + h;
                float dh = dhData[idx];
                float r = rData[idx];
                float z = zData[idx];
                float n = nData[idx];

                // h_new = (1 - z) * n + z * h_prev
                float dn = dh * (1 - z) * (1 - n * n);
                float dz = dh * (hPrevData[idx] - n) * z * (1 - z);

                // Compute hidden contribution for reset gate gradient
                float nHid = 0;
                int whhNOff = (2 * hiddenSize + h) * hiddenSize;
                for (int j = 0; j < hiddenSize; j++)
                    nHid += whhData[whhNOff + j] * hPrevData[b * hiddenSize + j];
                float dr = dn * nHid * r * (1 - r);

                gradRData[idx] = dr;
                gradZData[idx] = dz;
                gradNData[idx] = dn;
                gradPrevHData[idx] = dh * z;
            }
        }

        // Accumulate gradPrevH through W_hh
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                float sum = 0;
                for (int gate = 0; gate < 3; gate++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        int whhOff = (gate * hiddenSize + h) * hiddenSize + j;
                        float dGate = gate == 0 ? gradRData[b * hiddenSize + h]
                                    : gate == 1 ? gradZData[b * hiddenSize + h]
                                    : gradNData[b * hiddenSize + h];
                        sum += dGate * whhData[whhOff];
                    }
                }
                gradPrevHData[b * hiddenSize + j] += sum;
            }
        }

        UploadToBuffer(gradRData, gradGateR);
        UploadToBuffer(gradZData, gradGateZ);
        UploadToBuffer(gradNData, gradGateN);
        UploadToBuffer(gradPrevHData, gradPrevH);
    }

    #endregion
}
#endif
