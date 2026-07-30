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

    public void ProximalL1Update(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float l1Strength, int size)
    {
        var dummy = SharedDummyBuffer;
        var uniforms = MakeOptimizerUniforms(size, learningRate, l1Strength, 0, 0, 0, 0);
        Dispatch4BufferAsync("Optimizer", WebGpuKernels.OptimizerSource, "proximal_l1",
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

    public void AdamUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("CompressedOptimizerBf16", WebGpuKernels.CompressedOptimizerBf16Source, "adam_bf16",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
    }

    public void AdamWUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        var uniforms = MakeOptimizerUniforms(size, learningRate, beta1, beta2, epsilon, weightDecay, step);
        Dispatch4BufferAsync("CompressedOptimizerBf16", WebGpuKernels.CompressedOptimizerBf16Source, "adamw_bf16",
            param, gradient, m, v, uniforms, size).GetAwaiter().GetResult();
    }

    public void Adam8BitUpdate(IGpuBuffer param, IGpuBuffer gradient,
        IGpuBuffer mQuant, IGpuBuffer vQuant, IGpuBuffer mScales, IGpuBuffer vScales,
        float learningRate, float beta1, float beta2, float epsilon,
        float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
        int blockSize, int paramLength, int numBlocks)
    {
        var uniforms = new[]
        {
            BitConverter.Int32BitsToSingle(paramLength),
            BitConverter.Int32BitsToSingle(blockSize),
            BitConverter.Int32BitsToSingle(numBlocks),
            0f,
            learningRate,
            beta1,
            beta2,
            epsilon,
            oneMinusBeta1,
            oneMinusBeta2,
            biasCorrection1,
            biasCorrection2
        };
        Dispatch6BufferAsync("CompressedOptimizerInt8", WebGpuKernels.CompressedOptimizerInt8Source, "adam8bit",
            param, gradient, mQuant, vQuant, mScales, vScales, uniforms, numBlocks * 256).GetAwaiter().GetResult();
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
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputFeatures),
            BitConverter.Int32BitsToSingle(outputFeatures)
        };
        Dispatch4BufferAsync("HyperbolicLinear", WebGpuKernels.HyperbolicLinearSource,
            "hyperbolic_linear_forward", input, weights, biases, output, uniforms,
            checked(batchSize * outputFeatures)).GetAwaiter().GetResult();
        PoincareProject(output, output, batchSize, outputFeatures, curvature, epsilon);
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputFeatures),
            BitConverter.Int32BitsToSingle(outputFeatures)
        };
        Dispatch3BufferAsync("HyperbolicLinearBackward", WebGpuKernels.HyperbolicLinearBackwardSource,
            "hyperbolic_linear_backward_input", gradOutput, weights, gradInput, uniforms,
            checked(batchSize * inputFeatures)).GetAwaiter().GetResult();
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputFeatures),
            BitConverter.Int32BitsToSingle(outputFeatures)
        };
        Dispatch3BufferAsync("HyperbolicLinearBackward", WebGpuKernels.HyperbolicLinearBackwardSource,
            "hyperbolic_linear_backward_weights", gradOutput, input, gradWeights, uniforms,
            checked(outputFeatures * inputFeatures)).GetAwaiter().GetResult();
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(outputFeatures)
        };
        Dispatch2BufferAsync("HyperbolicLinearBackwardBias", WebGpuKernels.HyperbolicLinearBackwardBiasSource,
            "hyperbolic_linear_backward_biases", gradOutput, gradBiases, uniforms,
            outputFeatures).GetAwaiter().GetResult();
    }

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
        int totalPairs = batchSize * outputFeatures;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputFeatures),
            BitConverter.Int32BitsToSingle(outputFeatures),
        };
        Dispatch4BufferAsync("OctonionLinearForward", WebGpuKernels.OctonionLinearForwardSource, "octonion_linear_forward",
            input, weights, biases, output, uniforms, totalPairs).GetAwaiter().GetResult();
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

    #region Fused Kernel Operations

    public void HyperbolicLinearForwardFused(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
        => HyperbolicLinearForward(input, weights, biases, output, batchSize, inputFeatures, outputFeatures, curvature, epsilon);

    public void OctonionLinearForwardFusedReLU(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        int totalPairs = batchSize * outputFeatures;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(inputFeatures),
            BitConverter.Int32BitsToSingle(outputFeatures),
        };
        Dispatch4BufferAsync("OctonionFusedReLU", WebGpuKernels.OctonionFusedSource, "octonion_linear_fused_relu",
            input, weights, biases, output, uniforms, totalPairs).GetAwaiter().GetResult();
    }

    #endregion

    #region Complex Tensor Operations

    public void ComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > a.Size || numPairs * 2 > b.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are a={a.Size}, b={b.Size}, out={output.Size}.");
        Dispatch3BufferAsync("ComplexInterleavedMultiply", WebGpuKernels.ComplexInterleavedMultiplySource,
            "complex_interleaved_multiply", a, b, output, MakeUniform1(numPairs), numPairs).GetAwaiter().GetResult();
    }

    public void ComplexConjugate(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size || numPairs * 2 > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but buffer sizes are in={input.Size}, out={output.Size}.");
        Dispatch2BufferAsync("ComplexInterleavedUnary", WebGpuKernels.ComplexInterleavedUnarySource,
            "complex_interleaved_conjugate", input, output, MakeUniform1(numPairs), numPairs).GetAwaiter().GetResult();
    }

    public void ComplexMagnitude(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        if (numPairs <= 0) return;
        if (numPairs * 2 > input.Size)
            throw new ArgumentException($"numPairs ({numPairs}) requires {numPairs * 2} elements but input buffer has {input.Size}.");
        if (numPairs > output.Size)
            throw new ArgumentException($"numPairs ({numPairs}) exceeds output buffer size ({output.Size}).");
        Dispatch2BufferAsync("ComplexInterleavedUnary", WebGpuKernels.ComplexInterleavedUnarySource,
            "complex_interleaved_magnitude", input, output, MakeUniform1(numPairs), numPairs).GetAwaiter().GetResult();
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

    // ── Fused recurrence / LM-head GPU kernels (#1464) ─────────────────────────────────────
    // Real WGSL compute shaders (WebGpuRecurrenceKernels). Int params come in via a uniform struct
    // bound after the storage buffers. Forward only — the differentiable backward runs through the
    // CpuEngine tape (the engine override is forward-only and defers to base under a tape).

    private async Task DispatchRecurrenceAsync(string key, string wgsl, int total, IGpuBuffer[] buffers, int[] scalars)
    {
        var pipe = await GetOrCreatePipelineAsync("Recurrence:" + key, wgsl, "main");
        using var uniforms = new WebGpuBuffer(UniformInts(scalars), WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        var wb = new WebGpuBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) wb[i] = AsWgpu(buffers[i]);
        using var bind = new WebGpuBindGroup(pipe, wb);
        var (wg, _) = _device.CalculateWorkgroups1D(total);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, uniforms.BufferId, wg, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    private void DispatchRecurrence(string key, string wgsl, int total, IGpuBuffer[] buffers, int[] scalars)
        => DispatchRecurrenceAsync(key, wgsl, total, buffers, scalars).GetAwaiter().GetResult();

    public void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("gla", WebGpuRecurrenceKernels.GlaScan, batch * numHeads * headDim,
            new[] { q, k, v, gate, output }, new[] { batch, seqLen, modelDim, numHeads, headDim });

    // GLA BPTT backward — real WGSL kernels (recompute trajectory + reverse sweep with bitcast-CAS
    // u32 float atomics for the cross-row dQ/dK/dG). dQ/dK/dG must be pre-zeroed (the accumulators).
    public void GlaScanBackward(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || modelDim <= 0 || numHeads <= 0 || headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions must be positive.");
        int hh = headDim * headDim;
        long totalLong = checked((long)batch * numHeads * headDim);
        long trajLenLong = checked((long)batch * numHeads * seqLen * hh);
        if (totalLong > int.MaxValue || trajLenLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions exceed WebGPU launch/buffer limits.");
        int total = (int)totalLong;
        // dQ/dK/dG are atomic accumulators in the backward kernel — zero them here so the
        // method is self-contained and correct even if the caller reuses dirty buffers.
        Fill(dQ, 0f, batch * seqLen * modelDim);
        Fill(dK, 0f, batch * seqLen * modelDim);
        Fill(dG, 0f, batch * seqLen * numHeads);
        using var traj = AllocateBuffer((int)trajLenLong);
        var pc = new[] { batch, seqLen, modelDim, numHeads, headDim };
        DispatchRecurrence("gla_recompute", WebGpuRecurrenceKernels.GlaRecompute, total,
            new[] { k, v, gate, traj }, pc);
        DispatchRecurrence("gla_backward", WebGpuRecurrenceKernels.GlaBackward, total,
            new[] { dOut, q, k, v, gate, traj, dQ, dK, dV, dG }, pc);
    }

    public void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("xlstm", WebGpuRecurrenceKernels.XLstmScan, batch * numHeads,
            new[] { q, k, v, iGate, fGate, oGate, output }, new[] { batch, seqLen, modelDim, numHeads, headDim });

    public void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => DispatchRecurrence("gdn", WebGpuRecurrenceKernels.GatedDeltaScan, batch * numHeads * headDim,
            new[] { q, k, v, alpha, beta, output }, new[] { batch, seqLen, modelDim, numHeads, headDim });

    public void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
        => DispatchRecurrence("rglru", WebGpuRecurrenceKernels.RgLruScan, batch * recDim,
            new[] { value, recGate, inpGate, decay, output }, new[] { batch, seqLen, recDim });

    public void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
        => DispatchRecurrence("rwkv4", WebGpuRecurrenceKernels.Rwkv4Wkv, batch * modelDim,
            new[] { r, k, v, timeDecay, timeFirst, output }, new[] { batch, seqLen, modelDim });

    public void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
        => DispatchRecurrence("mamba", WebGpuRecurrenceKernels.MambaScan, batch * innerDim,
            new[] { x, delta, aLog, bParam, cParam, dParam, output }, new[] { batch, seqLen, innerDim, stateDim });

    public void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
        => DispatchRecurrence("mamba2", WebGpuRecurrenceKernels.Mamba2Ssd, batch * innerDim,
            new[] { x, delta, aLog, bParam, cParam, dParam, output },
            new[] { batch, seqLen, innerDim, numHeads, headDim, stateDim });

    public float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeRowLoss("ce_index", WebGpuRecurrenceKernels.FusedCeIndex, hidden, weight, bias, targetIds, n, d, vocab);

    public float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeRowLoss("ce_dense", WebGpuRecurrenceKernels.FusedCeDense, hidden, weight, bias, target, n, d, vocab);

    public void FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident("ce_index", WebGpuRecurrenceKernels.FusedCeIndex,
            hidden, weight, bias, targetIds, meanLoss, n, d, vocab);

    public void FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident("ce_dense", WebGpuRecurrenceKernels.FusedCeDense,
            hidden, weight, bias, target, meanLoss, n, d, vocab);

    // CE kernels write a per-row loss vector; the host sums the N-element vector. Returns mean CE.
    private float FusedCeRowLoss(
        string key, string wgsl, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        using var meanLoss = AllocateBuffer(1);
        FusedCeRowLossResident(key, wgsl, hidden, weight, bias, tgt, meanLoss, n, d, vocab);
        return DownloadBuffer(meanLoss)[0];
    }

    private void FusedCeRowLossResident(
        string key, string wgsl, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt,
        IGpuBuffer meanLoss, int n, int d, int vocab)
    {
        if (n <= 0 || d <= 0 || vocab <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Fused CE dimensions (n, d, vocab) must be positive.");
        using var rowLoss = AllocateBuffer(n);
        DispatchRecurrence(key, wgsl, n, new[] { hidden, weight, bias, tgt, rowLoss }, new[] { n, d, vocab });
        SumAxis(rowLoss, meanLoss, 1, n);
        Scale(meanLoss, meanLoss, 1f / n, 1);
    }

    private void GatherSequenceSlice(IGpuBuffer source, IGpuBuffer destination,
        int timestep, int sequenceLength, int batch, int width)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(sequenceLength * width),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(timestep * width),
            BitConverter.Int32BitsToSingle(0),
            0, 0
        };
        Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
            source, destination, uniforms, batch * width).GetAwaiter().GetResult();
    }

    private void ScatterSequenceSlice(IGpuBuffer source, IGpuBuffer destination,
        int timestep, int sequenceLength, int batch, int width)
    {
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(width),
            BitConverter.Int32BitsToSingle(sequenceLength * width),
            BitConverter.Int32BitsToSingle(0),
            BitConverter.Int32BitsToSingle(timestep * width),
            0, 0
        };
        Dispatch2BufferAsync("StridedSlice", WebGpuKernels.StridedSliceSource, "strided_slice",
            source, destination, uniforms, batch * width).GetAwaiter().GetResult();
    }

    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        int cellTotal = batch * hiddenSize;
        int gateTotal = batch * 4 * hiddenSize;
        if (seqLen <= 0 || cellTotal <= 0)
        {
            if (cellTotal > 0)
            {
                Copy(hInit, hFinal, cellTotal);
                Copy(cInit, cFinal, cellTotal);
            }
            return;
        }
        using var hA = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var hB = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var cA = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var cB = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
        using var gates = (WebGpuBuffer)AllocateBuffer(gateTotal);
        Copy(hInit, hA, cellTotal);
        Copy(cInit, cA, cellTotal);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inputSize),
            BitConverter.Int32BitsToSingle(hiddenSize),
            0
        };
        var stateUniforms = MakeUniformInts2(batch, hiddenSize);
        for (int t = 0; t < seqLen; t++)
        {
            var prevH = (t & 1) == 0 ? hA : hB;
            var nextH = (t & 1) == 0 ? hB : hA;
            var prevC = (t & 1) == 0 ? cA : cB;
            var nextC = (t & 1) == 0 ? cB : cA;
            GatherSequenceSlice(input, inputSlice, t, seqLen, batch, inputSize);
            DispatchNBufferAsync("LstmForwardLinear", WebGpuKernels.LstmForwardLinearSource,
                "lstm_forward_linear",
                new IGpuBuffer[] { inputSlice, prevH, weightsIh, weightsHh, biasIh, biasHh, gates },
                uniforms, gateTotal).GetAwaiter().GetResult();
            Dispatch4BufferAsync("LstmForwardState", WebGpuKernels.LstmForwardStateSource,
                "lstm_forward_state", gates, prevC, nextH, nextC, stateUniforms, cellTotal).GetAwaiter().GetResult();
            Copy(nextH, 0, allH, t * cellTotal, cellTotal);
            Copy(nextC, 0, allC, t * cellTotal, cellTotal);
            Copy(gates, 0, cacheGates, t * gateTotal, gateTotal);
            ScatterSequenceSlice(nextH, output, t, seqLen, batch, hiddenSize);
        }
        var finalH = (seqLen & 1) == 0 ? hA : hB;
        var finalC = (seqLen & 1) == 0 ? cA : cB;
        Copy(finalH, hFinal, cellTotal);
        Copy(finalC, cFinal, cellTotal);
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

            // h_prev for this timestep
            using var hPrev = (WebGpuBuffer)AllocateBuffer(cellTotal);
            if (t > 0)
                Copy(allH, (t - 1) * cellTotal, hPrev, 0, cellTotal);
            else
                Copy(hInit, hPrev, cellTotal);

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
            int gateSize = 4 * hs;
            int gateTotal = batch * gateSize;
            using var gates = (WebGpuBuffer)AllocateBuffer(gateTotal);
            using var cNew = (WebGpuBuffer)AllocateBuffer(cellTotal);
            using var gradGates = (WebGpuBuffer)AllocateBuffer(gateTotal);
            using var newGradC = (WebGpuBuffer)AllocateBuffer(cellTotal);
            using var newGradH = (WebGpuBuffer)AllocateBuffer(cellTotal);
            using var gradInputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            Copy(cacheGates, t * gateTotal, gates, 0, gateTotal);
            Copy(allC, t * cellTotal, cNew, 0, cellTotal);
            DispatchNBufferAsync("LstmBackwardGates", WebGpuKernels.LstmBackwardGatesSource,
                "lstm_backward_gates",
                new IGpuBuffer[] { gradH, gradC, gates, cPrev, cNew, gradGates, newGradC },
                MakeUniformInts2(batch, hs), cellTotal).GetAwaiter().GetResult();
            var inputProjectUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(gateSize),
                BitConverter.Int32BitsToSingle(inputSize),
                0
            };
            Dispatch3BufferAsync("RnnBackwardProject", WebGpuKernels.RnnBackwardProjectSource,
                "rnn_backward_project", gradGates, weightsIh, gradInputSlice,
                inputProjectUniforms, batch * inputSize).GetAwaiter().GetResult();
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

            var hiddenProjectUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(gateSize),
                BitConverter.Int32BitsToSingle(hs),
                0
            };
            Dispatch3BufferAsync("RnnBackwardProject", WebGpuKernels.RnnBackwardProjectSource,
                "rnn_backward_project", gradGates, weightsHh, newGradH,
                hiddenProjectUniforms, cellTotal).GetAwaiter().GetResult();

            var accumulateUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(gateSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(hs)
            };
            int accumulateSize = Math.Max(gateSize * inputSize, Math.Max(gateSize * hs, gateSize));
            DispatchNBufferAsync("LstmBackwardAccumulate", WebGpuKernels.LstmBackwardAccumulateSource,
                "lstm_backward_accumulate",
                new IGpuBuffer[] { gradGates, inputSlice, hPrev, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh },
                accumulateUniforms, accumulateSize).GetAwaiter().GetResult();

            Copy(newGradH, gradH, cellTotal);
            Copy(newGradC, gradC, cellTotal);
        }

        // Copy final gradients
        Copy(gradH, gradHInit, cellTotal);
        Copy(gradC, gradCInit, cellTotal);
    }

    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        int cellTotal = batch * hiddenSize;
        int gateTotal = batch * 3 * hiddenSize;
        if (seqLen <= 0 || cellTotal <= 0)
        {
            if (cellTotal > 0) Copy(hInit, hFinal, cellTotal);
            return;
        }
        using var hA = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var hB = (WebGpuBuffer)AllocateBuffer(cellTotal);
        using var inputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
        using var parts = (WebGpuBuffer)AllocateBuffer(batch * 4 * hiddenSize);
        using var gates = (WebGpuBuffer)AllocateBuffer(gateTotal);
        Copy(hInit, hA, cellTotal);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inputSize),
            BitConverter.Int32BitsToSingle(hiddenSize),
            0
        };
        var stateUniforms = MakeUniformInts2(batch, hiddenSize);
        for (int t = 0; t < seqLen; t++)
        {
            var prevH = (t & 1) == 0 ? hA : hB;
            var nextH = (t & 1) == 0 ? hB : hA;
            GatherSequenceSlice(input, inputSlice, t, seqLen, batch, inputSize);
            DispatchNBufferAsync("GruForwardLinear", WebGpuKernels.GruForwardLinearSource,
                "gru_forward_linear",
                new IGpuBuffer[] { inputSlice, prevH, weightsIh, weightsHh, biasIh, biasHh, parts },
                uniforms, batch * 4 * hiddenSize).GetAwaiter().GetResult();
            Dispatch4BufferAsync("GruForwardState", WebGpuKernels.GruForwardStateSource,
                "gru_forward_state", parts, prevH, nextH, gates,
                stateUniforms, cellTotal).GetAwaiter().GetResult();
            Copy(nextH, 0, allH, t * cellTotal, cellTotal);
            Copy(gates, 0, cacheGates, t * gateTotal, gateTotal);
            ScatterSequenceSlice(nextH, output, t, seqLen, batch, hiddenSize);
        }
        var finalH = (seqLen & 1) == 0 ? hA : hB;
        Copy(finalH, hFinal, cellTotal);
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
            int gateSize = 3 * hs;
            int gateTotal = batch * gateSize;
            using var gates = (WebGpuBuffer)AllocateBuffer(gateTotal);
            using var gradGates = (WebGpuBuffer)AllocateBuffer(gateTotal);
            using var newGradH = (WebGpuBuffer)AllocateBuffer(cellTotal);
            using var gradInputSlice = (WebGpuBuffer)AllocateBuffer(batch * inputSize);
            Copy(cacheGates, t * gateTotal, gates, 0, gateTotal);
            Dispatch5BufferAsync("GruBackwardGates", WebGpuKernels.GruBackwardGatesSource,
                "gru_backward_gates", gradH, gates, hPrev, weightsHh, gradGates,
                MakeUniformInts2(batch, hs), cellTotal).GetAwaiter().GetResult();
            var inputProjectUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(gateSize),
                BitConverter.Int32BitsToSingle(inputSize),
                0
            };
            Dispatch3BufferAsync("RnnBackwardProject", WebGpuKernels.RnnBackwardProjectSource,
                "rnn_backward_project", gradGates, weightsIh, gradInputSlice,
                inputProjectUniforms, batch * inputSize).GetAwaiter().GetResult();
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

            Dispatch5BufferAsync("GruBackwardHidden", WebGpuKernels.GruBackwardHiddenSource,
                "gru_backward_hidden", gradH, gates, gradGates, weightsHh, newGradH,
                MakeUniformInts2(batch, hs), cellTotal).GetAwaiter().GetResult();
            var accumulateUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(gateSize),
                BitConverter.Int32BitsToSingle(inputSize),
                BitConverter.Int32BitsToSingle(hs)
            };
            int accumulateSize = Math.Max(gateSize * inputSize, Math.Max(gateSize * hs, gateSize));
            DispatchNBufferAsync("GruBackwardAccumulate", WebGpuKernels.GruBackwardAccumulateSource,
                "gru_backward_accumulate",
                new IGpuBuffer[] { gradGates, inputSlice, hPrev, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh },
                accumulateUniforms, accumulateSize).GetAwaiter().GetResult();
            Copy(newGradH, gradH, cellTotal);
        }

        Copy(gradH, gradHInit, cellTotal);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        int cellTotal = batch * hiddenSize;
        if (cellTotal <= 0) return;
        var uniforms = MakeUniformInts2(batch, hiddenSize);
        DispatchNBufferAsync("GruCellBackwardGates", WebGpuKernels.GruCellBackwardGatesSource,
            "gru_cell_backward_gates",
            new[] { gradH, gateR, gateZ, gateN, prevH, weightsHh, gradGateR, gradGateZ, gradGateN },
            uniforms, cellTotal).GetAwaiter().GetResult();
        DispatchNBufferAsync("GruCellBackwardHidden", WebGpuKernels.GruCellBackwardHiddenSource,
            "gru_cell_backward_hidden",
            new[] { gradH, gateZ, gradGateR, gradGateZ, gradGateN, weightsHh, gradPrevH },
            uniforms, cellTotal).GetAwaiter().GetResult();
    }

    #endregion

    /// <inheritdoc/>
    public void SpectralFilter(IGpuBuffer inputReal, IGpuBuffer filterReal, IGpuBuffer filterImag,
        IGpuBuffer outputReal, int batch, int height, int width, int filterSliceCount)
    {
        if (filterSliceCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(filterSliceCount), "Must be >= 1.");
        if (height <= 0 || width <= 0 || batch <= 0)
            throw new ArgumentOutOfRangeException("Dimensions must be positive.");
        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
            throw new ArgumentException("height and width must be powers of 2 for FFT.");

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

    // ============================================================================
    // Issue #160 spectral perf kernels — WebGPU implementations.
    // ============================================================================

    public void Atan2Elementwise(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        // Kernel binding order is (imag, real, output) internally; keep it and pass accordingly.
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.Atan2Source,
            "atan2_elementwise", imag, real, output,
            new float[] { BitConverter.Int32BitsToSingle(n) }, n).GetAwaiter().GetResult();
    }

    public void NormalizeRowsFused(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.NormalizeRowsSource,
            "normalize_rows_fused", input, output,
            new float[] { BitConverter.Int32BitsToSingle(rows), BitConverter.Int32BitsToSingle(cols) }, rows).GetAwaiter().GetResult();
    }

    public void AnalyticSignalMask(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batch, int fftSize, int binLow, int binHigh)
    {
        if (batch <= 0 || fftSize <= 0) return;
        if (binLow < 0 || binHigh < binLow || binHigh > fftSize)
            throw new ArgumentOutOfRangeException(nameof(binHigh), $"Require 0 <= binLow ({binLow}) <= binHigh ({binHigh}) <= fftSize ({fftSize}).");
        long totalL = (long)batch * fftSize;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pc = new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(fftSize),
            BitConverter.Int32BitsToSingle(binLow), BitConverter.Int32BitsToSingle(binHigh) };
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.AnalyticSignalMaskSource,
            "analytic_signal_mask", specReal, outReal, pc, total).GetAwaiter().GetResult();
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.AnalyticSignalMaskSource,
            "analytic_signal_mask", specImag, outImag, pc, total).GetAwaiter().GetResult();
    }

    public void BispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2)
    {
        if (maxF1 <= 0 || maxF2 <= 0) return;
        long totalL = (long)maxF1 * maxF2;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pcReal = new float[] { BitConverter.Int32BitsToSingle(maxF1), BitConverter.Int32BitsToSingle(maxF2), BitConverter.Int32BitsToSingle(0) };
        var pcImag = new float[] { BitConverter.Int32BitsToSingle(maxF1), BitConverter.Int32BitsToSingle(maxF2), BitConverter.Int32BitsToSingle(1) };
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.BispectrumSource,
            "bispectrum_gather", specReal, specImag, outReal, pcReal, total).GetAwaiter().GetResult();
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.BispectrumSource,
            "bispectrum_gather", specReal, specImag, outImag, pcImag, total).GetAwaiter().GetResult();
    }

    public void TrispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2, int maxF3)
    {
        if (maxF1 <= 0 || maxF2 <= 0 || maxF3 <= 0) return;
        long totalL = (long)maxF1 * maxF2 * maxF3;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        var pcReal = new float[] { BitConverter.Int32BitsToSingle(maxF1), BitConverter.Int32BitsToSingle(maxF2), BitConverter.Int32BitsToSingle(maxF3), BitConverter.Int32BitsToSingle(0) };
        var pcImag = new float[] { BitConverter.Int32BitsToSingle(maxF1), BitConverter.Int32BitsToSingle(maxF2), BitConverter.Int32BitsToSingle(maxF3), BitConverter.Int32BitsToSingle(1) };
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.TrispectrumSource,
            "trispectrum_gather", specReal, specImag, outReal, pcReal, total).GetAwaiter().GetResult();
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.TrispectrumSource,
            "trispectrum_gather", specReal, specImag, outImag, pcImag, total).GetAwaiter().GetResult();
    }

    public void CavityBounceInplace(IGpuBuffer workReal, IGpuBuffer workImag, int total, float invN)
    {
        if (total <= 0) return;
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.CavityBounceSource,
            "cavity_bounce_real", workReal, workReal,
            new float[] { BitConverter.Int32BitsToSingle(total), invN }, total).GetAwaiter().GetResult();
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.ZeroBufferSource,
            "zero_buffer", workImag, workImag,
            new float[] { BitConverter.Int32BitsToSingle(total) }, total).GetAwaiter().GetResult();
    }

    public void WidebandLogBinPool(IGpuBuffer magBuf, IGpuBuffer output,
        int totalSegBatch, int fftSize, int numBins, int usable)
    {
        if (totalSegBatch <= 0 || fftSize <= 0 || numBins <= 0 || usable <= 0) return;
        long totalL = (long)totalSegBatch * numBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.WidebandLogBinPoolSource,
            "wideband_log_bin_pool", magBuf, output,
            new float[] { BitConverter.Int32BitsToSingle(totalSegBatch), BitConverter.Int32BitsToSingle(fftSize),
                BitConverter.Int32BitsToSingle(numBins), BitConverter.Int32BitsToSingle(usable) }, total).GetAwaiter().GetResult();
    }

    public void MelFilterbankApply(IGpuBuffer powerSpec, IGpuBuffer melFilters, IGpuBuffer melEnergy,
        int totalSegBatch, int specBins, int melBins)
    {
        if (totalSegBatch <= 0 || specBins <= 0 || melBins <= 0) return;
        long totalL = (long)totalSegBatch * melBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.MelFilterbankSource,
            "mel_filterbank_apply", powerSpec, melFilters, melEnergy,
            new float[] { BitConverter.Int32BitsToSingle(totalSegBatch), BitConverter.Int32BitsToSingle(specBins),
                BitConverter.Int32BitsToSingle(melBins) }, total).GetAwaiter().GetResult();
    }

    public void MfccLog1p(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        Dispatch2BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.MfccLog1pSource,
            "mfcc_log1p", input, output,
            new float[] { BitConverter.Int32BitsToSingle(n) }, n).GetAwaiter().GetResult();
    }

    public void PacPhaseBinMi(IGpuBuffer thetaPhase, IGpuBuffer gammaAmp, IGpuBuffer output,
        int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        if (batch <= 0) return;
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "numSamples must be positive.");
        if (numGammaBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGammaBands), "numGammaBands must be positive.");
        if (gammaIdx < 0 || gammaIdx >= numGammaBands)
            throw new ArgumentOutOfRangeException(nameof(gammaIdx), $"gammaIdx must be in [0, {numGammaBands}).");
        Dispatch3BufferAsync("SpectralPerf", WebGpuSpectralPerfKernels.PacPhaseBinMiSource,
            "pac_phase_bin_mi", thetaPhase, gammaAmp, output,
            new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(numSamples),
                BitConverter.Int32BitsToSingle(numGammaBands), BitConverter.Int32BitsToSingle(gammaIdx) }, batch).GetAwaiter().GetResult();
    }
}
#endif
