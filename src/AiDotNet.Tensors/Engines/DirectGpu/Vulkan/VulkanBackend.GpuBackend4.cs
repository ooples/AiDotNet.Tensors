// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 4: Optimizers, Sparse, FFT, RNN, Hyperbolic, Octonion, Quantum.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    #region Optimizer Operations

    private void DispatchOptimizer(string shader, int threads, IGpuBuffer[] buffers, params uint[] pushConstants)
        => GlslNaryOp(shader, buffers, threads, pushConstants);

    public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient, float learningRate, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.TwoBuffer, size, new[] { param, gradient },
            (uint)size, 0, FloatBits(learningRate), FloatBits(weightDecay));
    }

    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ThreeBuffer, size, new[] { param, gradient, velocity },
            (uint)size, 0, FloatBits(learningRate), FloatBits(momentum), FloatBits(weightDecay), 0);
    }

    public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, m, v },
            (uint)size, 0, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, m, v },
            (uint)size, 1, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    private static void ValidateOptimizerStep(int step)
    {
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step,
                "Step must be > 0 for bias-corrected optimizers to avoid division by zero.");
    }

    private static uint FBits(float value)
#if NETFRAMEWORK
        => unchecked((uint)BitConverter.ToInt32(BitConverter.GetBytes(value), 0));
#else
        => BitConverter.SingleToUInt32Bits(value);
#endif

    public void AdamUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => AdamBf16Impl(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size, false);

    public void AdamWUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => AdamBf16Impl(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size, true);

    private void AdamBf16Impl(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size, bool decoupled)
    {
        EnsureInitialized();
        ValidateOptimizerStep(step);
        if (size <= 0) return;
        GlslDispatchN(VulkanCompressedOptimizerKernels.AdamBf16, (size + 1) / 2,
            new[] { param, gradient, m, v },
            new[] { (uint)size, (uint)step, decoupled ? 1u : 0u,
                FBits(learningRate), FBits(beta1), FBits(beta2), FBits(epsilon), FBits(weightDecay) });
    }

    public void Adam8BitUpdate(IGpuBuffer param, IGpuBuffer gradient,
        IGpuBuffer mQuant, IGpuBuffer vQuant, IGpuBuffer mScales, IGpuBuffer vScales,
        float learningRate, float beta1, float beta2, float epsilon,
        float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
        int blockSize, int paramLength, int numBlocks)
    {
        EnsureInitialized();
        if (paramLength <= 0 || numBlocks <= 0) return;
        GlslDispatchN(VulkanCompressedOptimizerKernels.Adam8Bit, checked(numBlocks * 256),
            new[] { param, gradient, mQuant, vQuant, mScales, vScales },
            new[] { (uint)paramLength, (uint)blockSize, (uint)numBlocks,
                FBits(learningRate), FBits(beta1), FBits(beta2), FBits(epsilon),
                FBits(oneMinusBeta1), FBits(oneMinusBeta2), FBits(biasCorrection1), FBits(biasCorrection2) });
    }

    public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ThreeBuffer, size, new[] { param, gradient, squaredAvg },
            (uint)size, 1, FloatBits(learningRate), FloatBits(rho), FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ThreeBuffer, size, new[] { param, gradient, accumulatedGrad },
            (uint)size, 2, FloatBits(learningRate), FloatBits(epsilon), FloatBits(weightDecay), 0);
    }

    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ThreeBuffer, size, new[] { param, gradient, velocity },
            (uint)size, 3, FloatBits(learningRate), FloatBits(momentum), FloatBits(weightDecay), 0);
    }

    public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.Lars, 1, new[] { param, gradient, velocity },
            (uint)size, FloatBits(learningRate), FloatBits(momentum), FloatBits(weightDecay), FloatBits(trustCoeff));
    }

    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.Lamb, 1, new[] { param, gradient, m, v },
            (uint)size, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, accumGrad, accumUpdate },
            (uint)size, 2, 0, 0, FloatBits(rho), FloatBits(epsilon), FloatBits(weightDecay), 0);
    }

    public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.Amsgrad, size, new[] { param, gradient, m, v, vMax },
            (uint)size, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, m, u },
            (uint)size, 3, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ThreeBuffer, size, new[] { param, gradient, m },
            (uint)size, 4, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2), FloatBits(weightDecay));
    }

    public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ValidateOptimizerStep(step);
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, m, v },
            (uint)size, 4, (uint)step, FloatBits(learningRate), FloatBits(beta1), FloatBits(beta2),
            FloatBits(epsilon), FloatBits(weightDecay));
    }

    public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.FourBuffer, size, new[] { param, gradient, z, n },
            (uint)size, 5, 0, FloatBits(learningRate), FloatBits(l1Reg), FloatBits(l2Reg), FloatBits(beta), 0);
    }

    public void ProximalL1Update(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float l1Strength, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.TwoBuffer, size, new[] { param, gradient },
            (uint)size, 1, FloatBits(learningRate), FloatBits(l1Strength));
    }

    public void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ConvertToFp16, (size + 1) / 2,
            new[] { input, output }, (uint)size);
    }

    public void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
    {
        EnsureInitialized();
        DispatchOptimizer(VulkanOptimizerKernels.ConvertToFp32, size, new[] { input, output }, (uint)size);
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
        int groups = checked(M * (K / 4));
        GlslNaryOp(VulkanResidentKernels.Enforce2x4,
            new[] { denseInput, sparseValues, sparseIndices }, (groups + 3) / 4,
            new[] { (uint)M, (uint)K, (uint)groups });
    }

    public void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
    {
        EnsureInitialized();
        if (K % 4 != 0)
            throw new ArgumentException($"K ({K}) must be a multiple of 4 for 2:4 sparsity.", nameof(K));
        if (M <= 0)
            throw new ArgumentOutOfRangeException(nameof(M), "M must be positive.");
        GlslNaryOp(VulkanResidentKernels.Decompress2x4,
            new[] { sparseValues, sparseIndices, denseOutput }, checked(M * K),
            new[] { (uint)M, (uint)K });
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

    // NOTE: CsrSpMM moved to VulkanBackend.Sparse.cs and re-implemented as a real
    // GPU compute dispatch (GLSL csr_spmm via GlslQuintOp) — the previous
    // download-compute-upload stub here would defeat the point of routing SparseOps'
    // GPU tier through Vulkan. CsrSpMMBias below still uses that CsrSpMM as its
    // base then adds bias on host; that's a follow-up to fuse into an MSL bias
    // variant, but the SpMM itself now stays on-device.

    public void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
    {
        CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);
        BiasAdd(output, bias, output, M, N);
    }

    public void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices, IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
    {
        EnsureInitialized();
        using var edgeValueDummy = edgeValues is null ? AllocateBuffer(1) : null;
        GlslNaryOp(VulkanResidentKernels.ScatterAddEdges,
            new[] { input, sourceIndices, targetIndices, edgeValues ?? edgeValueDummy!, output },
            checked(numNodes * features),
            new[] { (uint)numNodes, (uint)numEdges, (uint)features, edgeValues is null ? 0u : 1u });
    }

    public void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.CsrSegmented,
            new[] { csrColIndices, csrRowPointers, input, output }, checked(M * N),
            new[] { (uint)M, (uint)K, (uint)N, 0u, FloatBits(0f) });
    }

    public void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.CsrSegmented,
            new[] { csrColIndices, csrRowPointers, input, output }, checked(M * N),
            new[] { (uint)M, (uint)K, (uint)N, 1u, FloatBits(0f) });
    }

    public void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.CsrSegmented,
            new[] { csrColIndices, csrRowPointers, input, output }, checked(M * N),
            new[] { (uint)M, (uint)K, (uint)N, 2u, FloatBits(epsilon) });
    }

    #endregion

    #region Specialized Layer Operations

    public void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.RbfForward, new[] { input, centers, epsilons, output },
            checked(batchSize * numCenters), new[] { (uint)batchSize, (uint)numCenters, (uint)inputDim });
    }

    public void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.StdpUpdate,
            new[] { weights, preTrace, postTrace, preSpike, postSpike }, checked(numPre * numPost),
            new[] { (uint)numPre, (uint)numPost, FloatBits(ltpRate), FloatBits(ltdRate), FloatBits(minWeight), FloatBits(maxWeight) });
    }

    public void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.UpdateTraces, new[] { traces, spikes }, size,
            new[] { (uint)size, FloatBits(decay) });
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
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.HyperbolicBackwardInput,
            new[] { gradOutput, weights, gradInput }, checked(batchSize * inputFeatures),
            new[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures });
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.HyperbolicBackwardWeights,
            new[] { gradOutput, input, gradWeights }, checked(outputFeatures * inputFeatures),
            new[] { (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures });
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.HyperbolicBackwardBiases,
            new[] { gradOutput, gradBiases }, outputFeatures,
            new[] { (uint)batchSize, (uint)outputFeatures });
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
        int size = checked(batchSize * stateSize);
        GlslNaryOp(VulkanResidentKernels.QuantumMeasurement,
            new[] { realPart, imagPart, probabilities }, size, new[] { (uint)size });
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.NormalizeProbabilities, new[] { probabilities }, batchSize,
            new[] { (uint)batchSize, (uint)stateSize });
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.ComplexMatVec,
            new[] { matReal, matImag, vecReal, vecImag, outReal, outImag }, checked(batchSize * dim),
            new[] { (uint)batchSize, (uint)dim });
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        EnsureInitialized();
        int stateSize = checked(1 << numQubits);
        GlslNaryOp(VulkanResidentKernels.QuantumRotation,
            new[] { stateReal, stateImag, angles, outReal, outImag }, checked(batchSize * stateSize),
            new[] { (uint)batchSize, (uint)stateSize, (uint)angles.Size });
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.MeasurementForward, new[] { input, output }, batchSize,
            new[] { (uint)batchSize, (uint)stateSize });
    }

    #endregion

    #region FFT and Signal Processing

    private void DispatchSplitDft(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int n, bool inverse)
    {
        int size = checked(batch * n);
        IGpuBuffer realSource = inputReal;
        IGpuBuffer imagSource = inputImag;
        IGpuBuffer? realCopy = null;
        IGpuBuffer? imagCopy = null;
        try
        {
            if (ReferenceEquals(inputReal, outputReal) || ReferenceEquals(inputReal, outputImag))
            {
                realCopy = AllocateBuffer(size);
                Copy(inputReal, realCopy, size);
                realSource = realCopy;
            }
            if (ReferenceEquals(inputImag, outputReal) || ReferenceEquals(inputImag, outputImag))
            {
                imagCopy = AllocateBuffer(size);
                Copy(inputImag, imagCopy, size);
                imagSource = imagCopy;
            }
            GlslNaryOp(VulkanResidentKernels.SplitDft,
                new[] { realSource, imagSource, outputReal, outputImag }, size,
                new[] { (uint)batch, (uint)n, inverse ? 1u : 0u });
        }
        finally
        {
            realCopy?.Dispose();
            imagCopy?.Dispose();
        }
    }

    private void DispatchSplitDft2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag, int batch, int height, int width, bool inverse)
    {
        int size = checked(checked(batch * height) * width);
        IGpuBuffer realSource = inputReal;
        IGpuBuffer imagSource = inputImag;
        IGpuBuffer? realCopy = null;
        IGpuBuffer? imagCopy = null;
        try
        {
            if (ReferenceEquals(inputReal, outputReal) || ReferenceEquals(inputReal, outputImag))
            {
                realCopy = AllocateBuffer(size);
                Copy(inputReal, realCopy, size);
                realSource = realCopy;
            }
            if (ReferenceEquals(inputImag, outputReal) || ReferenceEquals(inputImag, outputImag))
            {
                imagCopy = AllocateBuffer(size);
                Copy(inputImag, imagCopy, size);
                imagSource = imagCopy;
            }
            GlslNaryOp(VulkanResidentKernels.SplitDft2D,
                new[] { realSource, imagSource, outputReal, outputImag }, size,
                new[] { (uint)batch, (uint)height, (uint)width, inverse ? 1u : 0u });
        }
        finally
        {
            realCopy?.Dispose();
            imagCopy?.Dispose();
        }
    }

    public void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
    {
        EnsureInitialized();
        DispatchSplitDft(inputReal, inputImag, outputReal, outputImag, 1, n, inverse);
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
        DispatchSplitDft(inputReal, inputImag, outputReal, outputImag, batch, n, inverse);
    }

    public void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int height, int width, bool inverse)
    {
        EnsureInitialized();
        DispatchSplitDft2D(inputReal, inputImag, outputReal, outputImag, 1, height, width, inverse);
    }

    /// <inheritdoc/>
    public void BatchedFFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag,
        IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int height, int width, bool inverse)
    {
        EnsureInitialized();
        DispatchSplitDft2D(inputReal, inputImag, outputReal, outputImag, batch, height, width, inverse);
    }

    public void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
        => ResidentBinary(ResidentBinaryOp.Multiply, input, window, output, n);

    public void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
        => ResidentBinary(ResidentBinaryOp.ComplexMagnitude, real, imag, magnitude, n);

    public void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
        => ResidentBinary(ResidentBinaryOp.ComplexPhase, real, imag, phase, n);

    public void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.PolarToComplex,
            new[] { magnitude, phase, real, imag }, n, new[] { (uint)n });
    }

    public void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec, int numFrames, int numFreqs, int nMels)
    {
        EnsureInitialized();
        GlslNaryOp(VulkanResidentKernels.MelFilterbank,
            new[] { powerSpec, filterbank, melSpec }, checked(numFrames * nMels),
            new[] { (uint)numFrames, (uint)numFreqs, (uint)nMels });
    }

    public void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
        => ResidentUnary(ResidentUnaryOp.PowerToDb, power, db, n, refValue, minDb);

    public void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
        => ResidentUnary(ResidentUnaryOp.DbToPower, db, power, n, refValue);

    #endregion

    #region RNN (LSTM/GRU) Operations

    // ── Fused recurrence / LM-head GPU kernels (#1464) ─────────────────────────────────────
    // Real GLSL compute shaders (VulkanRecurrenceKernels), compiled to SPIR-V at runtime. Int
    // params go through push constants. Forward only — the differentiable backward runs through
    // the CpuEngine tape (the engine override is forward-only and defers to base under a tape).

    public void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => GlslNaryOp(VulkanRecurrenceKernels.GlaScan, new[] { q, k, v, gate, output },
            batch * numHeads * headDim,
            new[] { (uint)batch, (uint)seqLen, (uint)modelDim, (uint)numHeads, (uint)headDim });

    // GLA BPTT backward — real GLSL kernels (recompute trajectory + reverse sweep with CAS float
    // atomics for the cross-row dQ/dK/dG). dQ/dK/dG must be pre-zeroed (the atomic accumulators).
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
            throw new ArgumentOutOfRangeException(nameof(batch), "GLA dimensions exceed Vulkan launch/buffer limits.");
        int total = (int)totalLong;
        // dQ/dK/dG are atomic accumulators in the backward kernel — zero them here so the
        // method is self-contained and correct even if the caller reuses dirty buffers.
        Fill(dQ, 0f, batch * seqLen * modelDim);
        Fill(dK, 0f, batch * seqLen * modelDim);
        Fill(dG, 0f, batch * seqLen * numHeads);
        using var traj = AllocateBuffer((int)trajLenLong);
        var pc = new[] { (uint)batch, (uint)seqLen, (uint)modelDim, (uint)numHeads, (uint)headDim };
        GlslNaryOp(VulkanRecurrenceKernels.GlaRecompute, new[] { k, v, gate, traj }, total, pc);
        GlslNaryOp(VulkanRecurrenceKernels.GlaBackward,
            new[] { dOut, q, k, v, gate, traj, dQ, dK, dV, dG }, total, pc);
    }

    public void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => GlslNaryOp(VulkanRecurrenceKernels.XLstmScan, new[] { q, k, v, iGate, fGate, oGate, output },
            batch * numHeads,
            new[] { (uint)batch, (uint)seqLen, (uint)modelDim, (uint)numHeads, (uint)headDim });

    public void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
        => GlslNaryOp(VulkanRecurrenceKernels.GatedDeltaScan, new[] { q, k, v, alpha, beta, output },
            batch * numHeads * headDim,
            new[] { (uint)batch, (uint)seqLen, (uint)modelDim, (uint)numHeads, (uint)headDim });

    public void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
        => GlslNaryOp(VulkanRecurrenceKernels.RgLruScan, new[] { value, recGate, inpGate, decay, output },
            batch * recDim, new[] { (uint)batch, (uint)seqLen, (uint)recDim });

    public void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
        => GlslNaryOp(VulkanRecurrenceKernels.Rwkv4Wkv, new[] { r, k, v, timeDecay, timeFirst, output },
            batch * modelDim, new[] { (uint)batch, (uint)seqLen, (uint)modelDim });

    public void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
        => GlslNaryOp(VulkanRecurrenceKernels.MambaScan, new[] { x, delta, aLog, bParam, cParam, dParam, output },
            batch * innerDim, new[] { (uint)batch, (uint)seqLen, (uint)innerDim, (uint)stateDim });

    public void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
        => GlslNaryOp(VulkanRecurrenceKernels.Mamba2Ssd, new[] { x, delta, aLog, bParam, cParam, dParam, output },
            batch * innerDim,
            new[] { (uint)batch, (uint)seqLen, (uint)innerDim, (uint)numHeads, (uint)headDim, (uint)stateDim });

    public float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeRowLoss(VulkanRecurrenceKernels.FusedCeIndex, hidden, weight, bias, targetIds, n, d, vocab);

    public float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeRowLoss(VulkanRecurrenceKernels.FusedCeDense, hidden, weight, bias, target, n, d, vocab);

    public void FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident(VulkanRecurrenceKernels.FusedCeIndex, hidden, weight, bias, targetIds, meanLoss, n, d, vocab);

    public void FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target,
        IGpuBuffer meanLoss, int n, int d, int vocab)
        => FusedCeRowLossResident(VulkanRecurrenceKernels.FusedCeDense, hidden, weight, bias, target, meanLoss, n, d, vocab);

    // CE kernels write a per-row loss vector; the host sums the N-element vector. Returns mean CE.
    private float FusedCeRowLoss(
        string glsl, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        using var meanLoss = AllocateBuffer(1);
        FusedCeRowLossResident(glsl, hidden, weight, bias, tgt, meanLoss, n, d, vocab);
        return DownloadBuffer(meanLoss)[0];
    }

    private void FusedCeRowLossResident(
        string glsl, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt,
        IGpuBuffer meanLoss, int n, int d, int vocab)
    {
        if (n <= 0 || d <= 0 || vocab <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "Fused CE dimensions (n, d, vocab) must be positive.");
        using var rowLoss = AllocateBuffer(n);
        GlslNaryOp(glsl, new[] { hidden, weight, bias, tgt, rowLoss }, n,
            new[] { (uint)n, (uint)d, (uint)vocab });
        SumAxis(rowLoss, meanLoss, 1, n);
        Scale(meanLoss, meanLoss, 1f / n, 1);
    }

    private bool DispatchLstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return true;
        int stateSize = checked(batch * hiddenSize);
        Copy(hInit, 0, allH, 0, stateSize);
        Copy(cInit, 0, allC, 0, stateSize);
        GlslNaryOp(VulkanResidentKernels.LstmForwardSequence,
            new[] { input, weightsIh, weightsHh, biasIh, biasHh, output, allH, allC, cacheGates }, batch,
            new[] { (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize });
        Copy(allH, checked(seqLen * stateSize), hFinal, 0, stateSize);
        Copy(allC, checked(seqLen * stateSize), cFinal, 0, stateSize);
        return true;
    }

    private bool DispatchLstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return true;
        using var nextDh = AllocateBuffer(checked(batch * hiddenSize));
        GlslNaryOp(VulkanResidentKernels.LstmBackwardSequence,
            new[] { gradOutput, allH, allC, cacheGates, weightsIh, weightsHh, input,
                gradInput, gradHInit, gradCInit, gradWeightsIh, gradWeightsHh, gradBiasIh, nextDh }, 1,
            new[] { (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize });
        Copy(gradBiasIh, gradBiasHh, checked(4 * hiddenSize));
        return true;
    }

    private bool DispatchGruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return true;
        int stateSize = checked(batch * hiddenSize);
        Copy(hInit, 0, allH, 0, stateSize);
        GlslNaryOp(VulkanResidentKernels.GruForwardSequence,
            new[] { input, weightsIh, weightsHh, biasIh, biasHh, output, allH, cacheGates }, batch,
            new[] { (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize });
        Copy(allH, checked(seqLen * stateSize), hFinal, 0, stateSize);
        return true;
    }

    private bool DispatchGruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        if (seqLen <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0) return true;
        int stateSize = checked(batch * hiddenSize);
        using var nextDh = AllocateBuffer(stateSize);
        GlslNaryOp(VulkanResidentKernels.GruBackwardSequence,
            new[] { gradOutput, allH, cacheGates, weightsIh, weightsHh, input,
                gradInput, gradHInit, gradWeightsIh, gradWeightsHh, gradBiasIh, nextDh }, 1,
            new[] { (uint)seqLen, (uint)batch, (uint)inputSize, (uint)hiddenSize });
        Copy(gradHInit, dHBuffer, stateSize);
        Copy(gradBiasIh, gradBiasHh, checked(3 * hiddenSize));
        return true;
    }

    private bool DispatchGruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh, IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ,
        IGpuBuffer gradGateN, int batch, int hiddenSize)
    {
        GlslNaryOp(VulkanResidentKernels.GruCellBackward,
            new[] { gradH, gateR, gateZ, gateN, prevH, weightsHh, gradPrevH, gradGateR, gradGateZ, gradGateN },
            checked(batch * hiddenSize), new[] { (uint)batch, (uint)hiddenSize });
        return true;
    }

    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        DispatchLstmForwardSequence(input, hInit, cInit, weightsIh, weightsHh, biasIh, biasHh,
            output, hFinal, cFinal, allH, allC, cacheGates, seqLen, batch, inputSize, hiddenSize);
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
        DispatchLstmBackwardSequence(gradOutput, allH, allC, cacheGates, weightsIh, weightsHh, input,
            gradInput, gradHInit, gradCInit, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh,
            seqLen, batch, inputSize, hiddenSize);
    }

    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        DispatchGruForwardSequence(input, hInit, weightsIh, weightsHh, biasIh, biasHh,
            output, hFinal, allH, cacheGates, seqLen, batch, inputSize, hiddenSize);
    }

    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        EnsureInitialized();
        DispatchGruBackwardSequence(gradOutput, allH, cacheGates, weightsIh, weightsHh, input,
            gradInput, gradHInit, dHBuffer, gradWeightsIh, gradWeightsHh, gradBiasIh, gradBiasHh,
            seqLen, batch, inputSize, hiddenSize);
    }

    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        EnsureInitialized();
        DispatchGruCellBackward(gradH, gateR, gateZ, gateN, prevH, weightsHh,
            gradPrevH, gradGateR, gradGateZ, gradGateN, batch, hiddenSize);
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

    public void InterleaveComplex(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer interleaved, int n)
    {
        if (n <= 0) return;
        ValidateComplexLayoutBuffers(real, imag, interleaved, n, nameof(InterleaveComplex));
        GlslDispatchN(VulkanAuditKernels.InterleaveComplex, n,
            new IGpuBuffer[] { real, imag, interleaved }, new uint[] { (uint)n });
    }

    public void DeinterleaveComplex(IGpuBuffer interleaved, IGpuBuffer real, IGpuBuffer imag, int n)
    {
        if (n <= 0) return;
        ValidateComplexLayoutBuffers(real, imag, interleaved, n, nameof(DeinterleaveComplex));
        GlslDispatchN(VulkanAuditKernels.DeinterleaveComplex, n,
            new IGpuBuffer[] { interleaved, real, imag }, new uint[] { (uint)n });
    }

    private static void ValidateComplexLayoutBuffers(
        IGpuBuffer real, IGpuBuffer imag, IGpuBuffer interleaved, int n, string opName)
    {
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imag is null) throw new ArgumentNullException(nameof(imag));
        if (interleaved is null) throw new ArgumentNullException(nameof(interleaved));
        long requiredInterleaved = checked((long)n * 2);
        if (real.Size < n || imag.Size < n || interleaved.Size < requiredInterleaved)
            throw new ArgumentException(
                $"{opName}: n ({n}) requires real/imag buffers of at least {n} elements and an interleaved buffer of at least {requiredInterleaved} elements; sizes are real={real.Size}, imag={imag.Size}, interleaved={interleaved.Size}.");
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
        k = Math.Min(k, n);
        GlslDispatchN(VulkanGlslKernels.SplitComplexTopK, n,
            new[] { inReal, inImag, outReal, outImag }, new[] { (uint)n, (uint)k });
    }

    public void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
        => Softmax(input, output, rows, cols);

    // ─── HRR binding primitives (issue #248) ────────────────────────
    //
    // Native GLSL compute shaders — work stays on the GPU end-to-end.
    // Per-cell phases use the same 32-bit Murmur3 fmix and phase
    // construction as the CPU reference and every other GPU backend.

    public void SplitComplexUnitPhaseCodebook(
        IGpuBuffer outReal, IGpuBuffer outImag, int seed, int V, int D, bool kPsk, int k)
    {
        // Reject negative dims up front — if both are negative, their
        // product is positive and would slip past "total <= 0", then
        // the shader would cast to uint and index out of bounds.
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (V == 0 || D == 0) return;
        long total = (long)V * D;
        if (total > int.MaxValue) throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        if (kPsk && k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        int n = (int)total;

        // Push constants: { seed, V, D, kPsk, k, total } — 24 bytes.
        var push = new uint[]
        {
            unchecked((uint)seed),
            (uint)V,
            (uint)D,
            (uint)(kPsk ? 1 : 0),
            (uint)k,
            (uint)n,
        };
        GlslUnaryOp(VulkanHrrKernels.UnitPhaseCodebook, outReal, outImag, n, push, 6 * sizeof(uint));
    }

    public void SplitComplexPhaseCoherenceDecode(
        IGpuBuffer codesReal, IGpuBuffer codesImag,
        IGpuBuffer queryReal, IGpuBuffer queryImag,
        IGpuBuffer outScores, int V, int D)
    {
        if (V <= 0 || D <= 0) return;
        var push = new uint[] { (uint)V, (uint)D };
        GlslQuintOp(
            VulkanHrrKernels.PhaseCoherenceDecode,
            codesReal, codesImag, queryReal, queryImag, outScores,
            V, push, 2 * sizeof(uint));
    }

    public void SplitComplexHrrBindAccumulate(
        IGpuBuffer keyCodeReal, IGpuBuffer keyCodeImag,
        IGpuBuffer valPermCodeReal, IGpuBuffer valPermCodeImag,
        IGpuBuffer keyIds, IGpuBuffer valIds,
        IGpuBuffer memoryReal, IGpuBuffer memoryImag,
        int N, int D)
    {
        if (N <= 0 || D <= 0) return;
        // Derive vocabulary sizes from codebook buffer capacities so
        // the shader can reject out-of-range ids without reading past
        // allocated memory — mirrors the CPU path's
        // (uint)kId >= (uint)nKeys check in NativeHRRBindAccumulate.
        long nKeysL = keyCodeReal.Size / D;
        long nValsL = valPermCodeReal.Size / D;
        if (nKeysL <= 0 || nValsL <= 0)
            throw new ArgumentException(
                $"Codebook buffers must hold at least one full row of D = {D} elements.");
        if (nKeysL > int.MaxValue || nValsL > int.MaxValue)
            throw new ArgumentException("Codebook vocabulary size exceeds int.MaxValue.");
        var push = new uint[] { (uint)N, (uint)D, (uint)nKeysL, (uint)nValsL };
        // Dispatch one thread per output dimension; each thread loops N.
        // keyIds/valIds stored as Int32BitsToSingle floats — shader reverses
        // with floatBitsToInt() and validates against nKeys / nVals.
        GlslOctOp(
            VulkanHrrKernels.HrrBindAccumulate,
            keyCodeReal, keyCodeImag, valPermCodeReal, valPermCodeImag,
            keyIds, valIds, memoryReal, memoryImag,
            D, push, 4 * sizeof(uint));
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
    public void Atan2Elementwise(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        // Kernel binding order is (imag, real, output); keep it stable and pass accordingly.
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
        if (batch <= 0 || fftSize <= 0) return;
        long totalL = (long)batch * fftSize;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        // Apply mask to real and imag with single-buffer dispatches each
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.AnalyticSignalMaskScalar, specReal, outReal, total,
            new uint[] { (uint)batch, (uint)fftSize, (uint)binLow, (uint)binHigh }, 4 * sizeof(uint));
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.AnalyticSignalMaskScalar, specImag, outImag, total,
            new uint[] { (uint)batch, (uint)fftSize, (uint)binLow, (uint)binHigh }, 4 * sizeof(uint));
    }

    public void BispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2)
    {
        if (maxF1 <= 0 || maxF2 <= 0) return;
        long totalL = (long)maxF1 * maxF2;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.BispectrumReal, specReal, specImag, outReal, total,
            new uint[] { (uint)maxF1, (uint)maxF2 }, 2 * sizeof(uint));
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.BispectrumImag, specReal, specImag, outImag, total,
            new uint[] { (uint)maxF1, (uint)maxF2 }, 2 * sizeof(uint));
    }

    public void TrispectrumGather(IGpuBuffer specReal, IGpuBuffer specImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int maxF1, int maxF2, int maxF3)
    {
        if (maxF1 <= 0 || maxF2 <= 0 || maxF3 <= 0) return;
        long totalL = (long)maxF1 * maxF2 * maxF3;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
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
        if (totalSegBatch <= 0 || fftSize <= 0 || numBins <= 0 || usable <= 0) return;
        long totalL = (long)totalSegBatch * numBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
        GlslUnaryOp(VulkanGlslSpectralPerfKernels.WidebandLogBinPool, magBuf, output, total,
            new uint[] { (uint)totalSegBatch, (uint)fftSize, (uint)numBins, (uint)usable }, 4 * sizeof(uint));
    }

    public void MelFilterbankApply(IGpuBuffer powerSpec, IGpuBuffer melFilters, IGpuBuffer melEnergy,
        int totalSegBatch, int specBins, int melBins)
    {
        if (totalSegBatch <= 0 || specBins <= 0 || melBins <= 0) return;
        long totalL = (long)totalSegBatch * melBins;
        if (totalL <= 0 || totalL > int.MaxValue) return;
        int total = (int)totalL;
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
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "numSamples must be positive.");
        if (numGammaBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGammaBands), "numGammaBands must be positive.");
        if (gammaIdx < 0 || gammaIdx >= numGammaBands)
            throw new ArgumentOutOfRangeException(nameof(gammaIdx), $"gammaIdx must be in [0, {numGammaBands}).");
        GlslBinaryOp(VulkanGlslSpectralPerfKernels.PacPhaseBinMi, thetaPhase, gammaAmp, output, batch,
            new uint[] { (uint)batch, (uint)numSamples, (uint)numGammaBands, (uint)gammaIdx }, 4 * sizeof(uint));
    }
}
