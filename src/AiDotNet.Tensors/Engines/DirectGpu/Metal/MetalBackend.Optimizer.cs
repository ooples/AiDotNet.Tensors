// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Optimizer operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private void DispatchOptimizerMetal(uint operation, IGpuBuffer parameter, IGpuBuffer gradient,
        IGpuBuffer? state1, IGpuBuffer? state2, IGpuBuffer? state3, int size, int step,
        float learningRate, float a = 0f, float b = 0f, float c = 0f, float d = 0f)
    {
        if (size <= 0) return;
        using var dummy1 = state1 is null ? AllocateBuffer(1) : null;
        using var dummy2 = state2 is null ? AllocateBuffer(1) : null;
        using var dummy3 = state3 is null ? AllocateBuffer(1) : null;
        DispatchResidentMetal("optimizer_update", size,
            new[] { parameter, gradient, state1 ?? dummy1!, state2 ?? dummy2!, state3 ?? dummy3! },
            (uint)size, operation, (uint)step,
            unchecked((uint)SingleToInt32BitsCompat(learningRate)),
            unchecked((uint)SingleToInt32BitsCompat(a)),
            unchecked((uint)SingleToInt32BitsCompat(b)),
            unchecked((uint)SingleToInt32BitsCompat(c)),
            unchecked((uint)SingleToInt32BitsCompat(d)));
    }

    #region Optimizer Operations

    /// <summary>
    /// SGD with momentum update.
    /// </summary>
    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(1u, param, gradient, velocity, null, null, size, 0,
            learningRate, momentum, weightDecay);
    }

    /// <summary>
    /// Vanilla SGD update (no momentum).
    /// </summary>
    public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient, float learningRate, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(0u, param, gradient, null, null, null, size, 0,
            learningRate, weightDecay);
    }

    /// <summary>
    /// Adam optimizer update.
    /// </summary>
    public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(2u, param, gradient, m, v, null, size, step,
            learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /// <summary>
    /// AdamW optimizer update (decoupled weight decay).
    /// </summary>
    public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(3u, param, gradient, m, v, null, size, step,
            learningRate, beta1, beta2, epsilon, weightDecay);
    }

    // Lazily-compiled MSL library for the native compressed-moment kernels. IntPtr.Zero if the compile
    // failed (older Metal / MSL feature gap) → the callers fall back to the host path.
    private IntPtr _compressedOptLibrary = IntPtr.Zero;
    private bool _compressedOptLibTried;
    private readonly object _compressedOptLibLock = new();

    private IntPtr GetCompressedOptLibrary()
    {
        if (_compressedOptLibTried) return _compressedOptLibrary;
        lock (_compressedOptLibLock)
        {
            if (_compressedOptLibTried) return _compressedOptLibrary;
            try { _compressedOptLibrary = _shaderLibrary.CompileLibrary("CompressedOptimizer", MetalCompressedOptimizerKernels.Source); }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Metal compressed-optimizer library compile failed: {ex.Message}");
                _compressedOptLibrary = IntPtr.Zero;
            }
            _compressedOptLibTried = true;
            return _compressedOptLibrary;
        }
    }

    public void AdamUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => AdamBf16Impl(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size, decoupled: 0u);

    public void AdamWUpdateBf16(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => AdamBf16Impl(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size, decoupled: 1u);

    private void AdamBf16Impl(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size, uint decoupled)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        IntPtr library = GetCompressedOptLibrary();
        if (library == IntPtr.Zero)
            throw new InvalidOperationException("Metal compressed optimizer kernels are unavailable.");
        if (param is not MetalGpuBuffer parameter || gradient is not MetalGpuBuffer grad ||
            m is not MetalGpuBuffer firstMoment || v is not MetalGpuBuffer secondMoment)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");

        var pipeline = GetPipeline("CompressedOptimizer", library, "adam_bf16");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch((size + 1) / 2);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(parameter, 0);
        encoder.SetBuffer(grad, 1);
        encoder.SetBuffer(firstMoment, 2);
        encoder.SetBuffer(secondMoment, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.SetBytes((uint)step, 5);
        encoder.SetBytes(decoupled, 6);
        encoder.SetBytes(learningRate, 7);
        encoder.SetBytes(beta1, 8);
        encoder.SetBytes(beta2, 9);
        encoder.SetBytes(epsilon, 10);
        encoder.SetBytes(weightDecay, 11);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Adam8BitUpdate(IGpuBuffer param, IGpuBuffer gradient,
        IGpuBuffer mQuant, IGpuBuffer vQuant, IGpuBuffer mScales, IGpuBuffer vScales,
        float learningRate, float beta1, float beta2, float epsilon,
        float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2,
        int blockSize, int paramLength, int numBlocks)
    {
        ThrowIfDisposed();
        if (paramLength <= 0 || numBlocks <= 0) return;
        IntPtr library = GetCompressedOptLibrary();
        if (library == IntPtr.Zero)
            throw new InvalidOperationException("Metal compressed optimizer kernels are unavailable.");
        if (param is not MetalGpuBuffer parameter || gradient is not MetalGpuBuffer grad ||
            mQuant is not MetalGpuBuffer firstQuantized || vQuant is not MetalGpuBuffer secondQuantized ||
            mScales is not MetalGpuBuffer firstScales || vScales is not MetalGpuBuffer secondScales)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");

        var pipeline = GetPipeline("CompressedOptimizer", library, "adam8bit");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(checked(numBlocks * 256));
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(parameter, 0);
        encoder.SetBuffer(grad, 1);
        encoder.SetBuffer(firstQuantized, 2);
        encoder.SetBuffer(secondQuantized, 3);
        encoder.SetBuffer(firstScales, 4);
        encoder.SetBuffer(secondScales, 5);
        encoder.SetBytes((uint)paramLength, 6);
        encoder.SetBytes((uint)blockSize, 7);
        encoder.SetBytes((uint)numBlocks, 8);
        encoder.SetBytes(learningRate, 9);
        encoder.SetBytes(beta1, 10);
        encoder.SetBytes(beta2, 11);
        encoder.SetBytes(epsilon, 12);
        encoder.SetBytes(oneMinusBeta1, 13);
        encoder.SetBytes(oneMinusBeta2, 14);
        encoder.SetBytes(biasCorrection1, 15);
        encoder.SetBytes(biasCorrection2, 16);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// RMSprop optimizer update.
    /// </summary>
    public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(4u, param, gradient, squaredAvg, null, null, size, 0,
            learningRate, rho, epsilon, weightDecay);
    }

    /// <summary>
    /// Adagrad optimizer update.
    /// </summary>
    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(5u, param, gradient, accumulatedGrad, null, null, size, 0,
            learningRate, epsilon, weightDecay);
    }

    /// <summary>
    /// Nesterov Accelerated Gradient (NAG) update.
    /// </summary>
    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(6u, param, gradient, velocity, null, null, size, 0,
            learningRate, momentum, weightDecay);
    }

    /// <summary>
    /// LARS (Layer-wise Adaptive Rate Scaling) update.
    /// </summary>
    public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("lars_update_serial", 1, new[] { param, gradient, velocity },
            (uint)size, unchecked((uint)SingleToInt32BitsCompat(learningRate)), unchecked((uint)SingleToInt32BitsCompat(momentum)),
            unchecked((uint)SingleToInt32BitsCompat(weightDecay)), unchecked((uint)SingleToInt32BitsCompat(trustCoeff)));
    }

    /// <summary>
    /// LAMB (Layer-wise Adaptive Moments) update.
    /// </summary>
    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("lamb_update_serial", 1, new[] { param, gradient, m, v },
            (uint)size, (uint)step, unchecked((uint)SingleToInt32BitsCompat(learningRate)), unchecked((uint)SingleToInt32BitsCompat(beta1)),
            unchecked((uint)SingleToInt32BitsCompat(beta2)), unchecked((uint)SingleToInt32BitsCompat(epsilon)), unchecked((uint)SingleToInt32BitsCompat(weightDecay)));
    }

    /// <summary>
    /// Adadelta optimizer update.
    /// </summary>
    public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(7u, param, gradient, accumGrad, accumUpdate, null, size, 0,
            0f, rho, epsilon, weightDecay);
    }

    /// <summary>
    /// AMSGrad optimizer update.
    /// </summary>
    public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(8u, param, gradient, m, v, vMax, size, step,
            learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /// <summary>
    /// AdaMax optimizer update.
    /// </summary>
    public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(9u, param, gradient, m, u, null, size, step,
            learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /// <summary>
    /// Lion optimizer update.
    /// </summary>
    public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(10u, param, gradient, m, null, null, size, 0,
            learningRate, beta1, beta2, weightDecay);
    }

    /// <summary>
    /// Nadam optimizer update.
    /// </summary>
    public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(11u, param, gradient, m, v, null, size, step,
            learningRate, beta1, beta2, epsilon, weightDecay);
    }

    /// <summary>
    /// FTRL optimizer update.
    /// </summary>
    public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(12u, param, gradient, z, n, null, size, 0,
            learningRate, l1Reg, l2Reg, beta);
    }

    /// <summary>
    /// Proximal gradient (ISTA) update with L1 soft-thresholding.
    /// </summary>
    public void ProximalL1Update(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float l1Strength, int size)
    {
        ThrowIfDisposed();
        DispatchOptimizerMetal(13u, param, gradient, null, null, null, size, 0,
            learningRate, l1Strength);
    }

    #endregion
}
