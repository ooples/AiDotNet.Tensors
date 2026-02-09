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

    #endregion

    #region Capsule Network Operations

    /// <summary>
    /// Squash activation for capsule networks.
    /// </summary>
    public void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
    {
        ThrowIfDisposed();
        // CPU fallback for squash
        var data = DownloadBuffer(input);
        var result = new float[numCapsules * capsuleDim];

        for (int c = 0; c < numCapsules; c++)
        {
            float normSq = 0;
            for (int d = 0; d < capsuleDim; d++)
            {
                float v = data[c * capsuleDim + d];
                normSq += v * v;
            }
            float norm = MathF.Sqrt(normSq + epsilon);
            float scale = normSq / ((1 + normSq) * norm);

            for (int d = 0; d < capsuleDim; d++)
            {
                result[c * capsuleDim + d] = data[c * capsuleDim + d] * scale;
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Squash backward for capsule networks.
    /// </summary>
    public void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
    {
        ThrowIfDisposed();
        // Simplified CPU fallback
        var gradOut = DownloadBuffer(gradOutput);
        var inp = DownloadBuffer(input);
        var gradIn = new float[numCapsules * capsuleDim];

        for (int c = 0; c < numCapsules; c++)
        {
            float normSq = 0;
            for (int d = 0; d < capsuleDim; d++)
            {
                float v = inp[c * capsuleDim + d];
                normSq += v * v;
            }
            float norm = MathF.Sqrt(normSq + epsilon);
            float scale = normSq / ((1 + normSq) * norm);

            for (int d = 0; d < capsuleDim; d++)
            {
                gradIn[c * capsuleDim + d] = gradOut[c * capsuleDim + d] * scale;
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(gradIn);
        }
    }

    /// <summary>
    /// Capsule prediction transform.
    /// </summary>
    public void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
    {
        ThrowIfDisposed();
        // CPU fallback - proper implementation would use GPU kernel
        var inp = DownloadBuffer(input);
        var w = DownloadBuffer(weights);
        var result = new float[batchSize * inputCapsules * outputCapsules * outputDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputCapsules; i++)
            {
                for (int c = 0; c < outputCapsules; c++)
                {
                    for (int d = 0; d < outputDim; d++)
                    {
                        float sum = 0;
                        for (int k = 0; k < inputDim; k++)
                        {
                            int inIdx = b * inputCapsules * inputDim + i * inputDim + k;
                            int wIdx = i * outputCapsules * inputDim * outputDim +
                                      c * inputDim * outputDim + k * outputDim + d;
                            sum += inp[inIdx] * w[wIdx];
                        }
                        int outIdx = b * inputCapsules * outputCapsules * outputDim +
                                    i * outputCapsules * outputDim + c * outputDim + d;
                        result[outIdx] = sum;
                    }
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
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
        // CPU fallback
        var c = DownloadBuffer(coupling);
        var p = DownloadBuffer(predictions);
        var result = new float[batchSize * outputCapsules * capsuleDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int oc = 0; oc < outputCapsules; oc++)
            {
                for (int d = 0; d < capsuleDim; d++)
                {
                    float sum = 0;
                    for (int ic = 0; ic < inputCapsules; ic++)
                    {
                        int cIdx = b * inputCapsules * outputCapsules + ic * outputCapsules + oc;
                        int pIdx = b * inputCapsules * outputCapsules * capsuleDim +
                                  ic * outputCapsules * capsuleDim + oc * capsuleDim + d;
                        sum += c[cIdx] * p[pIdx];
                    }
                    result[b * outputCapsules * capsuleDim + oc * capsuleDim + d] = sum;
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Dynamic routing agreement computation.
    /// </summary>
    public void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
    {
        ThrowIfDisposed();
        // CPU fallback
        var p = DownloadBuffer(predictions);
        var o = DownloadBuffer(output);
        var result = new float[batchSize * inputCapsules * outputCapsules];

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputCapsules; ic++)
            {
                for (int oc = 0; oc < outputCapsules; oc++)
                {
                    float sum = 0;
                    for (int d = 0; d < capsuleDim; d++)
                    {
                        int pIdx = b * inputCapsules * outputCapsules * capsuleDim +
                                  ic * outputCapsules * capsuleDim + oc * capsuleDim + d;
                        int oIdx = b * outputCapsules * capsuleDim + oc * capsuleDim + d;
                        sum += p[pIdx] * o[oIdx];
                    }
                    result[b * inputCapsules * outputCapsules + ic * outputCapsules + oc] = sum;
                }
            }
        }

        if (agreement is MetalGpuBuffer aBuffer)
        {
            aBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Tile tensor along batch dimension.
    /// </summary>
    public void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
    {
        ThrowIfDisposed();
        var data = DownloadBuffer(input);
        var result = new float[repeats * innerSize];

        for (int r = 0; r < repeats; r++)
        {
            Array.Copy(data, 0, result, r * innerSize, innerSize);
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    /// <summary>
    /// Tile tensor along any axis.
    /// </summary>
    public void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
    {
        ThrowIfDisposed();
        var data = DownloadBuffer(input);
        var result = new float[outerSize * axisSize * repeats * innerSize];

        for (int o = 0; o < outerSize; o++)
        {
            for (int a = 0; a < axisSize; a++)
            {
                for (int r = 0; r < repeats; r++)
                {
                    int srcOffset = o * axisSize * innerSize + a * innerSize;
                    int dstOffset = o * axisSize * repeats * innerSize + (a * repeats + r) * innerSize;
                    Array.Copy(data, srcOffset, result, dstOffset, innerSize);
                }
            }
        }

        if (output is MetalGpuBuffer outBuffer)
        {
            outBuffer.CopyFrom(result);
        }
    }

    #endregion
}
