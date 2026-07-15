// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Activation functions and gradients.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Additional Activation Functions

    /// <summary>
    /// Leaky ReLU activation: B = max(alpha * A, A)
    /// </summary>
    public void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "leaky_relu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBytes(alpha, 1);
        encoder.SetBuffer(bBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// ELU activation: B = (A > 0) ? A : alpha * (exp(A) - 1)
    /// </summary>
    public void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "elu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBytes(alpha, 1);
        encoder.SetBuffer(bBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Swish activation: B = A * sigmoid(A)
    /// </summary>
    public void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("swish", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// SiLU activation (same as Swish): B = A * sigmoid(A)
    /// </summary>
    public void Silu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        Swish(A, B, size);
    }

    /// <summary>
    /// Mish activation: B = A * tanh(softplus(A))
    /// </summary>
    public void Mish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("mish", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Softplus activation: B = log(1 + exp(A))
    /// </summary>
    public void Softplus(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("softplus", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Hardswish activation: B = A * min(max(A + 3, 0), 6) / 6
    /// </summary>
    public void Hardswish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("hardswish", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// SELU activation: B = scale * (max(0, A) + min(0, alpha * (exp(A) - 1)))
    /// </summary>
    public void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size)
    {
        ThrowIfDisposed();

        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "selu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBytes(alpha, 1);
        encoder.SetBytes(scale, 2);
        encoder.SetBuffer(bBuffer, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Hardsigmoid activation: B = max(0, min(1, (A + 3) / 6))
    /// </summary>
    public void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("hardsigmoid", A, B, size, _activationLibrary);
    }

    /// <summary>
    /// Hardtanh activation: B = max(minVal, min(maxVal, A))
    /// </summary>
    public void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
    {
        ThrowIfDisposed();
        Clamp(A, B, minVal, maxVal, size);
    }

    #endregion

    #region Activation Gradients

    /// <summary>
    /// ReLU backward: gradInput = gradOutput * (input > 0 ? 1 : 0)
    /// </summary>
    public void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "relu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Sigmoid backward: gradInput = gradOutput * output * (1 - output)
    /// </summary>
    public void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            output is not MetalGpuBuffer oBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "sigmoid_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(oBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Tanh backward: gradInput = gradOutput * (1 - output * output)
    /// </summary>
    public void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            output is not MetalGpuBuffer oBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "tanh_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(oBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// GELU backward
    /// </summary>
    public void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "gelu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Softmax backward
    /// </summary>
    public void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            output is not MetalGpuBuffer oBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Normalization", _normalizationLibrary, "softmax_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(oBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)features, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Leaky ReLU backward
    /// </summary>
    public void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "leaky_relu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBytes(alpha, 2);
        encoder.SetBuffer(giBuffer, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// ELU backward
    /// </summary>
    public void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            output is not MetalGpuBuffer oBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "elu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes(alpha, 3);
        encoder.SetBuffer(giBuffer, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Swish backward
    /// </summary>
    public void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "swish_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// SiLU backward (same as Swish)
    /// </summary>
    public void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        SwishBackward(gradOutput, input, gradInput, size);
    }

    /// <summary>
    /// Mish backward
    /// </summary>
    public void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "mish_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Softplus backward
    /// </summary>
    public void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        // d/dx softplus(x) = sigmoid(x)
        Sigmoid(input, gradInput, size);
        Multiply(gradOutput, gradInput, gradInput, size);
    }

    /// <summary>
    /// Hardswish backward
    /// </summary>
    public void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "hardswish_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// SELU backward
    /// </summary>
    public void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "selu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBytes(alpha, 2);
        encoder.SetBytes(scale, 3);
        encoder.SetBuffer(giBuffer, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Hardsigmoid backward
    /// </summary>
    public void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "hardsigmoid_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    /// <summary>
    /// Hardtanh backward
    /// </summary>
    public void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
    {
        ThrowIfDisposed();

        if (gradOutput is not MetalGpuBuffer goBuffer ||
            input is not MetalGpuBuffer iBuffer ||
            gradInput is not MetalGpuBuffer giBuffer)
        {
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        }

        var pipeline = GetPipeline("Activation", _activationLibrary, "hardtanh_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBytes(minVal, 2);
        encoder.SetBytes(maxVal, 3);
        encoder.SetBuffer(giBuffer, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Relu6(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("relu6", A, B, size, _activationLibrary);
    }

    public void Relu6Backward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer2 || input is not MetalGpuBuffer iBuffer2 || gradInput is not MetalGpuBuffer giBuffer2)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline2 = GetPipeline("Activation", _activationLibrary, "relu6_backward");
        var (tg2, tpg2) = pipeline2.Calculate1DDispatch(size);
        using var enc2 = _commandQueue.CreateScopedComputeEncoder();
        enc2.SetPipelineState(pipeline2.Handle);
        enc2.SetBuffer(goBuffer2, 0); enc2.SetBuffer(iBuffer2, 1); enc2.SetBuffer(giBuffer2, 2);
        enc2.SetBytes((uint)size, 3);
        enc2.DispatchThreadgroups(tg2, tpg2);
    }

    public void PRelu(IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer output, int size, int alphaSize)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer iBuffer || alpha is not MetalGpuBuffer aBuffer || output is not MetalGpuBuffer oBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "prelu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(aBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.SetBytes((uint)alphaSize, 4);
        encoder.SetBytes((uint)1, 5); // spatial_size = 1 for flat alpha
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PReluBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer alpha, IGpuBuffer gradInput, int size, int alphaSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || input is not MetalGpuBuffer iBuffer ||
            alpha is not MetalGpuBuffer aBuffer || gradInput is not MetalGpuBuffer giBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "prelu_backward_input");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(aBuffer, 2);
        encoder.SetBuffer(giBuffer, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.SetBytes((uint)alphaSize, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PReluBackwardAlpha(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradAlpha, int size, int alphaSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || input is not MetalGpuBuffer iBuffer || gradAlpha is not MetalGpuBuffer gaBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "prelu_backward_alpha");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(alphaSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(gaBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.SetBytes((uint)alphaSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void RRelu(IGpuBuffer input, IGpuBuffer noise, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer iBuffer || noise is not MetalGpuBuffer nBuffer || output is not MetalGpuBuffer oBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "rrelu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(nBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void RReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer noise, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || input is not MetalGpuBuffer iBuffer ||
            noise is not MetalGpuBuffer nBuffer || gradInput is not MetalGpuBuffer giBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "rrelu_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(nBuffer, 2);
        encoder.SetBuffer(giBuffer, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Threshold(IGpuBuffer input, IGpuBuffer output, float threshold, float value, int size)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer iBuffer || output is not MetalGpuBuffer oBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "threshold_forward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(oBuffer, 1);
        encoder.SetBytes(threshold, 2);
        encoder.SetBytes(value, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ThresholdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float threshold, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || input is not MetalGpuBuffer iBuffer || gradInput is not MetalGpuBuffer giBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "threshold_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(iBuffer, 1);
        encoder.SetBuffer(giBuffer, 2);
        encoder.SetBytes(threshold, 3);
        encoder.SetBytes((uint)size, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ReciprocalBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer3 || input is not MetalGpuBuffer iBuffer3 || gradInput is not MetalGpuBuffer giBuffer3)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline3 = GetPipeline("Activation", _activationLibrary, "reciprocal_backward");
        var (tg3, tpg3) = pipeline3.Calculate1DDispatch(size);
        using var enc3 = _commandQueue.CreateScopedComputeEncoder();
        enc3.SetPipelineState(pipeline3.Handle);
        enc3.SetBuffer(goBuffer3, 0); enc3.SetBuffer(iBuffer3, 1); enc3.SetBuffer(giBuffer3, 2);
        enc3.SetBytes((uint)size, 3);
        enc3.DispatchThreadgroups(tg3, tpg3);
    }

    public void AvgPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inp || output is not MetalGpuBuffer outp)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "avg_pool1d");
        int total = batch * channels * outLength;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(inp, 0); enc.SetBuffer(outp, 1);
        enc.SetBytes((uint)batch, 2); enc.SetBytes((uint)channels, 3);
        enc.SetBytes((uint)inLength, 4); enc.SetBytes((uint)outLength, 5);
        enc.SetBytes((uint)kernelSize, 6); enc.SetBytes((uint)stride, 7);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void MaxPool1D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inLength, int outLength, int kernelSize, int stride)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inp || output is not MetalGpuBuffer outp)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "max_pool1d");
        int total = batch * channels * outLength;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(inp, 0); enc.SetBuffer(outp, 1);
        enc.SetBytes((uint)batch, 2); enc.SetBytes((uint)channels, 3);
        enc.SetBytes((uint)inLength, 4); enc.SetBytes((uint)outLength, 5);
        enc.SetBytes((uint)kernelSize, 6); enc.SetBytes((uint)stride, 7);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void BilinearUpsample2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inH, int inW, int outH, int outW)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inp || output is not MetalGpuBuffer outp)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "bilinear_upsample2d");
        int total = batch * channels * outH * outW;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(inp, 0); enc.SetBuffer(outp, 1);
        enc.SetBytes((uint)batch, 2); enc.SetBytes((uint)channels, 3);
        enc.SetBytes((uint)inH, 4); enc.SetBytes((uint)inW, 5);
        enc.SetBytes((uint)outH, 6); enc.SetBytes((uint)outW, 7);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void ScatterMean(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        ThrowIfDisposed();
        if (source is not MetalGpuBuffer src || indices is not MetalGpuBuffer idx || output is not MetalGpuBuffer outp || counts is not MetalGpuBuffer cnt)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        if (GpuDeterminism.IsActive)
        {
            // Issue #382: route to atomic-free variant — per (dstRow, col) output cell.
            var pD = GetPipeline("Activation", _activationLibrary, "scatter_mean_deterministic");
            // 2D dispatch: dim x = featureSize (col), dim y = outputSize (dstRow)
            uint threadsPerThreadgroupX = (uint)Math.Min(16, featureSize);
            uint threadsPerThreadgroupY = (uint)Math.Min(16, outputSize);
            uint groupsX = (uint)((featureSize + (int)threadsPerThreadgroupX - 1) / (int)threadsPerThreadgroupX);
            uint groupsY = (uint)((outputSize + (int)threadsPerThreadgroupY - 1) / (int)threadsPerThreadgroupY);
            using var encD = _commandQueue.CreateScopedComputeEncoder();
            encD.SetPipelineState(pD.Handle);
            encD.SetBuffer(src, 0); encD.SetBuffer(idx, 1); encD.SetBuffer(outp, 2); encD.SetBuffer(cnt, 3);
            encD.SetBytes((uint)sourceSize, 4); encD.SetBytes((uint)outputSize, 5); encD.SetBytes((uint)featureSize, 6);
            encD.DispatchThreadgroups(
                new MetalNativeBindings.MTLSize(groupsX, groupsY, 1),
                new MetalNativeBindings.MTLSize(threadsPerThreadgroupX, threadsPerThreadgroupY, 1));
        }
        else
        {
            // Pass 1: scatter-add (atomic)
            var p1 = GetPipeline("Activation", _activationLibrary, "scatter_mean");
            var (tg1, tpg1) = p1.Calculate1DDispatch(sourceSize);
            using var enc1 = _commandQueue.CreateScopedComputeEncoder();
            enc1.SetPipelineState(p1.Handle);
            enc1.SetBuffer(src, 0); enc1.SetBuffer(idx, 1); enc1.SetBuffer(outp, 2); enc1.SetBuffer(cnt, 3);
            enc1.SetBytes((uint)sourceSize, 4); enc1.SetBytes((uint)featureSize, 5);
            enc1.DispatchThreadgroups(tg1, tpg1);
        }

        // Pass 2: divide (already deterministic by construction)
        var p2 = GetPipeline("Activation", _activationLibrary, "scatter_mean_divide");
        var (tg2, tpg2) = p2.Calculate1DDispatch(outputSize);
        using var enc2 = _commandQueue.CreateScopedComputeEncoder();
        enc2.SetPipelineState(p2.Handle);
        enc2.SetBuffer(outp, 0); enc2.SetBuffer(cnt, 1);
        enc2.SetBytes((uint)outputSize, 2); enc2.SetBytes((uint)featureSize, 3);
        enc2.DispatchThreadgroups(tg2, tpg2);
    }

    public void VarBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || input is not MetalGpuBuffer inp || mean is not MetalGpuBuffer m || gradInput is not MetalGpuBuffer gi)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "var_backward");
        int total = outerSize * reduceSize;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(inp, 1); enc.SetBuffer(m, 2); enc.SetBuffer(gi, 3);
        enc.SetBytes((uint)outerSize, 4); enc.SetBytes((uint)reduceSize, 5);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void StdBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer std, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || input is not MetalGpuBuffer inp || mean is not MetalGpuBuffer m || std is not MetalGpuBuffer s || gradInput is not MetalGpuBuffer gi)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "std_backward");
        int total = outerSize * reduceSize;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(inp, 1); enc.SetBuffer(m, 2); enc.SetBuffer(s, 3); enc.SetBuffer(gi, 4);
        enc.SetBytes((uint)outerSize, 5); enc.SetBytes((uint)reduceSize, 6);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void MaskedFillBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || mask is not MetalGpuBuffer m || gradInput is not MetalGpuBuffer gi)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "masked_fill_backward");
        var (tg, tpg) = pipeline.Calculate1DDispatch(size);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(m, 1); enc.SetBuffer(gi, 2);
        enc.SetBytes((uint)size, 3);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void WhereBackward(IGpuBuffer gradOutput, IGpuBuffer condition, IGpuBuffer gradX, IGpuBuffer gradY, int size)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || condition is not MetalGpuBuffer c || gradX is not MetalGpuBuffer gx || gradY is not MetalGpuBuffer gy)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "where_backward");
        var (tg, tpg) = pipeline.Calculate1DDispatch(size);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(c, 1); enc.SetBuffer(gx, 2); enc.SetBuffer(gy, 3);
        enc.SetBytes((uint)size, 4);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void NormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer norm, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || input is not MetalGpuBuffer inp || norm is not MetalGpuBuffer n || gradInput is not MetalGpuBuffer gi)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "norm_backward");
        int total = outerSize * reduceSize;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(inp, 1); enc.SetBuffer(n, 2); enc.SetBuffer(gi, 3);
        enc.SetBytes((uint)outerSize, 4); enc.SetBytes((uint)reduceSize, 5);
        enc.DispatchThreadgroups(tg, tpg);
    }

    public void LogSumExpBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer lse, IGpuBuffer gradInput, int outerSize, int reduceSize)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer go || input is not MetalGpuBuffer inp || lse is not MetalGpuBuffer l || gradInput is not MetalGpuBuffer gi)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Activation", _activationLibrary, "logsumexp_backward");
        int total = outerSize * reduceSize;
        var (tg, tpg) = pipeline.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipeline.Handle);
        enc.SetBuffer(go, 0); enc.SetBuffer(inp, 1); enc.SetBuffer(l, 2); enc.SetBuffer(gi, 3);
        enc.SetBytes((uint)outerSize, 4); enc.SetBytes((uint)reduceSize, 5);
        enc.DispatchThreadgroups(tg, tpg);
    }

    #endregion

    #region Capsule Network Operations

    /// <summary>
    /// Squash activation for capsule networks.
    /// </summary>
    public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        ThrowIfDisposed();
        int count = checked(numCapsules * capsuleDim);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        if (input is not MetalGpuBuffer source || target is not MetalGpuBuffer destination)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        if (_residentLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal resident kernels are unavailable.");
        var pipeline = GetPipeline("Resident", _residentLibrary, "capsule_squash");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(count);
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(source, 0);
            encoder.SetBuffer(destination, 1);
            encoder.SetBytes((uint)numCapsules, 2);
            encoder.SetBytes((uint)capsuleDim, 3);
            encoder.SetBytes(epsilon, 4);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Squash backward for capsule networks.
    /// </summary>
    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        ThrowIfDisposed();
        int count = checked(numCapsules * capsuleDim);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(gradInput, gradOutput) || ReferenceEquals(gradInput, input);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? gradInput;
        if (gradOutput is not MetalGpuBuffer gradient || input is not MetalGpuBuffer source ||
            target is not MetalGpuBuffer destination)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        if (_residentLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal resident kernels are unavailable.");
        var pipeline = GetPipeline("Resident", _residentLibrary, "capsule_squash_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(count);
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(gradient, 0);
            encoder.SetBuffer(source, 1);
            encoder.SetBuffer(destination, 2);
            encoder.SetBytes((uint)numCapsules, 3);
            encoder.SetBytes((uint)capsuleDim, 4);
            encoder.SetBytes(epsilon, 5);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }
        if (temporary is not null) Copy(temporary, gradInput, count);
    }

    /// <summary>
    /// Capsule prediction transform.
    /// </summary>
    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * inputCapsules * outputCapsules * outputDim);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(output, input) || ReferenceEquals(output, weights);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("capsule_predictions", count, new[] { input, weights, target },
            (uint)batchSize, (uint)inputCapsules, (uint)inputDim, (uint)outputCapsules, (uint)outputDim);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Capsule transform.
    /// </summary>
    public void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
    {
        ThrowIfDisposed();
        CapsulePredictions(input, weights, output, batchSize, inputCapsules, inputDim, numCapsules, capsuleDim);
    }

    /// <summary>
    /// Dynamic routing weighted sum.
    /// </summary>
    public void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * outputCapsules * capsuleDim);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(output, coupling) || ReferenceEquals(output, predictions);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("capsule_weighted_sum", count, new[] { coupling, predictions, target },
            (uint)batchSize, (uint)inputCapsules, (uint)outputCapsules, (uint)capsuleDim);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Dynamic routing agreement computation.
    /// </summary>
    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * inputCapsules * outputCapsules);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(agreement, predictions) || ReferenceEquals(agreement, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? agreement;
        DispatchResidentMetal("capsule_agreement", count, new[] { predictions, output, target },
            (uint)batchSize, (uint)inputCapsules, (uint)outputCapsules, (uint)capsuleDim);
        if (temporary is not null) Copy(temporary, agreement, count);
    }

    /// <summary>
    /// Tile tensor along batch dimension.
    /// </summary>
    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        ThrowIfDisposed();
        int count = checked(repeats * innerSize);
        if (count <= 0) return;
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("tile_batch", count, new[] { input, target }, (uint)repeats, (uint)innerSize);
        if (temporary is not null) Copy(temporary, output, count);
    }

    /// <summary>
    /// Tile tensor along any axis.
    /// </summary>
    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inputBuffer || output is not MetalGpuBuffer outputBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        int total = checked(outerSize * axisSize * repeats * innerSize);
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, "tile_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inputBuffer, 0);
        encoder.SetBuffer(outputBuffer, 1);
        encoder.SetBytes((uint)outerSize, 2);
        encoder.SetBytes((uint)axisSize, 3);
        encoder.SetBytes((uint)innerSize, 4);
        encoder.SetBytes((uint)repeats, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PixelShuffle(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inH, int inW, int scale)
        => DispatchPixelShuffle("pixel_shuffle", input, output,
            checked(batch * channels * inH * scale * inW * scale), batch, channels, inH, inW, scale);

    public void PixelShuffleBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inH, int inW, int scale)
        => DispatchPixelShuffle("pixel_shuffle_backward", gradOutput, gradInput,
            checked(batch * channels * scale * scale * inH * inW), batch, channels, inH, inW, scale);

    private void DispatchPixelShuffle(string kernelName, IGpuBuffer input, IGpuBuffer output,
        int total, int batch, int channels, int inH, int inW, int scale)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inputBuffer || output is not MetalGpuBuffer outputBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("ElementWise", _elementWiseLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inputBuffer, 0);
        encoder.SetBuffer(outputBuffer, 1);
        encoder.SetBytes((uint)batch, 2);
        encoder.SetBytes((uint)channels, 3);
        encoder.SetBytes((uint)inH, 4);
        encoder.SetBytes((uint)inW, 5);
        encoder.SetBytes((uint)scale, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion
}
