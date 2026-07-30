// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Loss functions and gradients.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Loss Functions

    private float ResidentLossScalar(IGpuBuffer predictions, IGpuBuffer targets, int count, int divisor,
        uint operation, float parameter0 = 0f, float parameter1 = 0f)
    {
        if (count <= 0 || divisor <= 0) return 0f;
        using var result = AllocateBuffer(1);
        DispatchResidentMetal("loss_scalar_serial", 1, [predictions, targets, result],
            (uint)count, (uint)divisor, operation,
            unchecked((uint)SingleToInt32BitsCompat(parameter0)), unchecked((uint)SingleToInt32BitsCompat(parameter1)));
        return DownloadBuffer(result)[0];
    }

    private void ResidentLossBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput,
        int count, uint operation, float parameter0 = 0f, float parameter1 = 0f)
    {
        if (count <= 0) return;
        DispatchResidentMetal("loss_elementwise_backward", count, [predictions, targets, gradInput],
            (uint)count, operation, unchecked((uint)SingleToInt32BitsCompat(parameter0)), unchecked((uint)SingleToInt32BitsCompat(parameter1)));
    }

    /// <summary>
    /// Cross-entropy loss for multi-class classification.
    /// </summary>
    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, checked(batchSize * numClasses), batchSize, 0);
    }

    /// <summary>
    /// Cross-entropy backward.
    /// </summary>
    public void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
    {
        ThrowIfDisposed();
        // gradInput = (predictions - targets) / batchSize
        Subtract(predictions, targets, gradInput, batchSize * numClasses);
        Scale(gradInput, gradInput, 1.0f / batchSize, batchSize * numClasses);
    }

    /// <summary>
    /// Binary cross-entropy loss.
    /// </summary>
    public float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 1);
    }

    /// <summary>
    /// Binary cross-entropy backward.
    /// </summary>
    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 1);
    }

    /// <summary>
    /// Mean squared error loss.
    /// </summary>
    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 2);
    }

    /// <summary>
    /// MSE backward.
    /// </summary>
    public void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        Subtract(predictions, targets, gradInput, size);
        Scale(gradInput, gradInput, 2.0f / size, size);
    }

    /// <summary>
    /// Smooth L1 (Huber) loss.
    /// </summary>
    public float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 3, beta);
    }

    /// <summary>
    /// Smooth L1 backward.
    /// </summary>
    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 3, beta);
    }

    /// <summary>
    /// Huber loss.
    /// </summary>
    public float HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float delta)
    {
        ThrowIfDisposed();
        return SmoothL1Loss(predictions, targets, size, delta);
    }

    /// <summary>
    /// Huber backward.
    /// </summary>
    public void HuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float delta)
    {
        ThrowIfDisposed();
        SmoothL1Backward(predictions, targets, gradInput, size, delta);
    }

    /// <summary>
    /// Focal loss for class imbalance.
    /// </summary>
    public float FocalLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float alpha, float gamma)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 4, alpha, gamma);
    }

    /// <summary>
    /// Focal backward.
    /// </summary>
    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 4, alpha, gamma);
    }

    /// <summary>
    /// Mean absolute error loss.
    /// </summary>
    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 5);
    }

    /// <summary>
    /// MAE backward.
    /// </summary>
    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 5);
    }

    /// <summary>
    /// Log-Cosh loss.
    /// </summary>
    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 6);
    }

    /// <summary>
    /// Log-Cosh backward.
    /// </summary>
    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 6);
    }

    /// <summary>
    /// Quantile loss.
    /// </summary>
    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 7, quantile);
    }

    /// <summary>
    /// Quantile backward.
    /// </summary>
    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 7, quantile);
    }

    /// <summary>
    /// Hinge loss.
    /// </summary>
    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 8);
    }

    /// <summary>
    /// Hinge backward.
    /// </summary>
    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 8);
    }

    /// <summary>
    /// Squared hinge loss.
    /// </summary>
    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 9);
    }

    /// <summary>
    /// Squared hinge backward.
    /// </summary>
    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 9);
    }

    /// <summary>
    /// Poisson loss.
    /// </summary>
    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 10);
    }

    /// <summary>
    /// Poisson backward.
    /// </summary>
    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 10);
    }

    /// <summary>
    /// Exponential loss.
    /// </summary>
    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 11);
    }

    /// <summary>
    /// Exponential backward.
    /// </summary>
    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 11);
    }

    /// <summary>
    /// Modified Huber loss.
    /// </summary>
    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 12);
    }

    /// <summary>
    /// Modified Huber backward.
    /// </summary>
    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 12);
    }

    /// <summary>
    /// Categorical cross-entropy loss.
    /// </summary>
    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 13);
    }

    /// <summary>
    /// Categorical cross-entropy backward.
    /// </summary>
    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 13);
    }

    /// <summary>
    /// Charbonnier loss.
    /// </summary>
    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 14, epsilon);
    }

    /// <summary>
    /// Charbonnier backward.
    /// </summary>
    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 14, epsilon);
    }

    /// <summary>
    /// Elastic net loss.
    /// </summary>
    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
    {
        ThrowIfDisposed();
        return ResidentLossScalar(predictions, targets, size, size, 15, l1Weight, l2Weight);
    }

    /// <summary>
    /// Elastic net backward.
    /// </summary>
    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        ThrowIfDisposed();
        ResidentLossBackward(predictions, targets, gradInput, size, 15, l1Weight, l2Weight);
    }

    /// <summary>
    /// Triplet loss.
    /// </summary>
    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        if (batchSize <= 0) return 0f;
        using var result = AllocateBuffer(1);
        DispatchResidentMetal("triplet_loss_serial", 1, [anchor, positive, negative, result],
            (uint)batchSize, (uint)embeddingDim, unchecked((uint)SingleToInt32BitsCompat(margin)));
        return DownloadBuffer(result)[0];
    }

    /// <summary>
    /// Triplet loss backward.
    /// </summary>
    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || embeddingDim <= 0) return;
        int count = checked(batchSize * embeddingDim);
        using var anchorGradient = AllocateBuffer(count);
        using var positiveGradient = AllocateBuffer(count);
        using var negativeGradient = AllocateBuffer(count);
        DispatchResidentMetal("triplet_loss_backward", batchSize,
            [anchor, positive, negative, anchorGradient, positiveGradient, negativeGradient],
            (uint)batchSize, (uint)embeddingDim, unchecked((uint)SingleToInt32BitsCompat(margin)));
        Copy(anchorGradient, gradAnchor, count);
        Copy(positiveGradient, gradPositive, count);
        Copy(negativeGradient, gradNegative, count);
    }

    /// <summary>
    /// Contrastive loss.
    /// </summary>
    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        if (batchSize <= 0) return 0f;
        using var result = AllocateBuffer(1);
        DispatchResidentMetal("contrastive_loss_serial", 1, [output1, output2, labels, result],
            (uint)batchSize, (uint)embeddingDim, unchecked((uint)SingleToInt32BitsCompat(margin)));
        return DownloadBuffer(result)[0];
    }

    /// <summary>
    /// Contrastive loss backward.
    /// </summary>
    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        if (batchSize <= 0 || embeddingDim <= 0) return;
        int count = checked(batchSize * embeddingDim);
        using var output1Gradient = AllocateBuffer(count);
        using var output2Gradient = AllocateBuffer(count);
        DispatchResidentMetal("contrastive_loss_backward", batchSize,
            [output1, output2, labels, output1Gradient, output2Gradient],
            (uint)batchSize, (uint)embeddingDim, unchecked((uint)SingleToInt32BitsCompat(margin)));
        Copy(output1Gradient, gradOutput1, count);
        Copy(output2Gradient, gradOutput2, count);
    }

    public void L1Loss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures)
    {
        ThrowIfDisposed();
        if (predictions is not MetalGpuBuffer pBuffer || targets is not MetalGpuBuffer tBuffer || loss is not MetalGpuBuffer lBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "l1_loss_batch");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(pBuffer, 0);
        encoder.SetBuffer(tBuffer, 1);
        encoder.SetBuffer(lBuffer, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)numFeatures, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void HuberLoss(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numFeatures, float delta)
    {
        ThrowIfDisposed();
        if (predictions is not MetalGpuBuffer pBuffer || targets is not MetalGpuBuffer tBuffer || loss is not MetalGpuBuffer lBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "huber_loss_batch");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(pBuffer, 0);
        encoder.SetBuffer(tBuffer, 1);
        encoder.SetBuffer(lBuffer, 2);
        encoder.SetBytes(delta, 3);
        encoder.SetBytes((uint)batchSize, 4);
        encoder.SetBytes((uint)numFeatures, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void BceWithLogitsLoss(IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer loss, int size)
    {
        ThrowIfDisposed();
        if (logits is not MetalGpuBuffer lBuffer || targets is not MetalGpuBuffer tBuffer || loss is not MetalGpuBuffer oBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "bce_with_logits_elementwise");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(lBuffer, 0);
        encoder.SetBuffer(tBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void NllLoss(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer loss, int batchSize, int numClasses)
    {
        ThrowIfDisposed();
        if (logProbs is not MetalGpuBuffer lpBuffer || targets is not MetalGpuBuffer tBuffer || loss is not MetalGpuBuffer lBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "nll_loss_batch");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(lpBuffer, 0);
        encoder.SetBuffer(tBuffer, 1);
        encoder.SetBuffer(lBuffer, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)numClasses, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void KlDivLoss(IGpuBuffer input, IGpuBuffer target, IGpuBuffer loss, int size)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer iBuffer || target is not MetalGpuBuffer tBuffer || loss is not MetalGpuBuffer oBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "kl_div_elementwise");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(iBuffer, 0);
        encoder.SetBuffer(tBuffer, 1);
        encoder.SetBuffer(oBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void MseLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("loss_backward_with_scalar_gradient", size,
            [gradOutput, predictions, targets, gradInput], (uint)size,
            unchecked((uint)SingleToInt32BitsCompat(invN)), 0u, 0u);
    }

    public void L1LossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || predictions is not MetalGpuBuffer pBuffer ||
            targets is not MetalGpuBuffer tBuffer || gradInput is not MetalGpuBuffer giBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "l1_loss_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(pBuffer, 1);
        encoder.SetBuffer(tBuffer, 2);
        encoder.SetBuffer(giBuffer, 3);
        encoder.SetBytes(invN, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void HuberLossBackward(IGpuBuffer gradOutput, IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN, float delta)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        DispatchResidentMetal("loss_backward_with_scalar_gradient", size,
            [gradOutput, predictions, targets, gradInput], (uint)size,
            unchecked((uint)SingleToInt32BitsCompat(invN)), unchecked((uint)SingleToInt32BitsCompat(delta)), 1u);
    }

    public void BceWithLogitsBackward(IGpuBuffer gradOutput, IGpuBuffer logits, IGpuBuffer targets, IGpuBuffer gradInput, int size, float invN)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuffer || logits is not MetalGpuBuffer lBuffer ||
            targets is not MetalGpuBuffer tBuffer || gradInput is not MetalGpuBuffer giBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Loss", _lossLibrary, "bce_with_logits_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(goBuffer, 0);
        encoder.SetBuffer(lBuffer, 1);
        encoder.SetBuffer(tBuffer, 2);
        encoder.SetBuffer(giBuffer, 3);
        encoder.SetBytes(invN, 4);
        encoder.SetBytes((uint)size, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region StopGradient, Fused Linear, and IoU Operations

    public void CopyBuffer(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        ThrowIfDisposed();
        Copy(source, destination, size);
    }

    public void FusedLinearReLU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearMetal("fused_linear_relu", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearSigmoid(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearMetal("fused_linear_sigmoid", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearTanh(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearMetal("fused_linear_tanh", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearGELU(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearMetal("fused_linear_gelu", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearSwish(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearMetal("fused_linear_swish", input, weight, bias, output, batchSize, inFeatures, outFeatures); }
    public void FusedLinearReLUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardMetal("fused_linear_relu_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 0); }
    public void FusedLinearSigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardMetal("fused_linear_sigmoid_backward_grad_input", gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 1); }
    public void FusedLinearTanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer output, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardMetal("fused_linear_tanh_backward_grad_input", gradOutput, input, weight, output, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 2); }
    public void FusedLinearGELUBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardMetal("fused_linear_gelu_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 3); }
    public void FusedLinearSwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer preActivation, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias, int batchSize, int inFeatures, int outFeatures) { LaunchFusedLinearBackwardMetal("fused_linear_swish_backward_grad_input", gradOutput, input, weight, preActivation, gradInput, gradWeight, gradBias, batchSize, inFeatures, outFeatures, 4); }
    public void IoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUMetal("iou_loss", predicted, target, loss, numBoxes); }
    public void GIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUMetal("giou_loss", predicted, target, loss, numBoxes); }
    public void DIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUMetal("diou_loss", predicted, target, loss, numBoxes); }
    public void CIoULoss(IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes) { LaunchIoUMetal("ciou_loss", predicted, target, loss, numBoxes); }
    public void IoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackwardMetal("iou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void GIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackwardMetal("giou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void DIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackwardMetal("diou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }
    public void CIoULossBackward(IGpuBuffer gradOutput, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes) { LaunchIoUBackwardMetal("ciou_loss_backward", gradOutput, predicted, target, gradPredicted, numBoxes); }

    private void LaunchFusedLinearMetal(string kernelName, IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output, int batchSize, int inFeatures, int outFeatures)
    {
        ThrowIfDisposed();
        var pipeline = GetPipeline("FusedLinear", _fusedLinearLibrary, kernelName);
        int total = batchSize * outFeatures;
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)weight, 1);
        encoder.SetBuffer((MetalGpuBuffer)bias, 2);
        encoder.SetBuffer((MetalGpuBuffer)output, 3);
        encoder.SetBytes(batchSize, 4);
        encoder.SetBytes(inFeatures, 5);
        encoder.SetBytes(outFeatures, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void LaunchIoUMetal(string kernelName, IGpuBuffer predicted, IGpuBuffer target, IGpuBuffer loss, int numBoxes)
    {
        ThrowIfDisposed();
        var pipeline = GetPipeline("IoULoss", _iouLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numBoxes);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)predicted, 0);
        encoder.SetBuffer((MetalGpuBuffer)target, 1);
        encoder.SetBuffer((MetalGpuBuffer)loss, 2);
        encoder.SetBytes(numBoxes, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void LaunchFusedLinearBackwardMetal(string gradInputKernelName, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer weight, IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batchSize, int inFeatures, int outFeatures, int activationType)
    {
        ThrowIfDisposed();
        var goBuffer = (MetalGpuBuffer)gradOutput;
        var inBuffer = (MetalGpuBuffer)input;
        var wBuffer = (MetalGpuBuffer)weight;
        var sBuffer = (MetalGpuBuffer)saved;
        var giBuffer = (MetalGpuBuffer)gradInput;
        var gwBuffer = (MetalGpuBuffer)gradWeight;
        var gbBuffer = (MetalGpuBuffer)gradBias;

        // Kernel 1: grad_input[b,i] = sum_j(masked_grad[b,j] * weight[i,j])
        {
            var pipeline = GetPipeline("FusedLinear", _fusedLinearLibrary, gradInputKernelName);
            int total = batchSize * inFeatures;
            var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(goBuffer, 0);
            encoder.SetBuffer(wBuffer, 1);
            encoder.SetBuffer(sBuffer, 2);
            encoder.SetBuffer(giBuffer, 3);
            encoder.SetBytes(batchSize, 4);
            encoder.SetBytes(inFeatures, 5);
            encoder.SetBytes(outFeatures, 6);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }

        // Kernel 2: weight gradient — gradWeight[i,j] = sum_b(input[b,i] * masked_grad[b,j])
        {
            var pipeline = GetPipeline("FusedLinear", _fusedLinearLibrary, "fused_linear_weight_grad");
            int total = inFeatures * outFeatures;
            var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(goBuffer, 0);
            encoder.SetBuffer(inBuffer, 1);
            encoder.SetBuffer(sBuffer, 2);
            encoder.SetBuffer(gwBuffer, 3);
            encoder.SetBytes(batchSize, 4);
            encoder.SetBytes(inFeatures, 5);
            encoder.SetBytes(outFeatures, 6);
            encoder.SetBytes(activationType, 7);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }

        // Kernel 3: bias gradient — gradBias[j] = sum_b(masked_grad[b,j])
        {
            var pipeline = GetPipeline("FusedLinear", _fusedLinearLibrary, "fused_linear_bias_grad");
            var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outFeatures);
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            encoder.SetPipelineState(pipeline.Handle);
            encoder.SetBuffer(goBuffer, 0);
            encoder.SetBuffer(sBuffer, 1);
            encoder.SetBuffer(gbBuffer, 2);
            encoder.SetBytes(batchSize, 3);
            encoder.SetBytes(outFeatures, 4);
            encoder.SetBytes(activationType, 5);
            encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        }
    }

    private void LaunchIoUBackwardMetal(string kernelName, IGpuBuffer gradOutput, IGpuBuffer predicted,
        IGpuBuffer target, IGpuBuffer gradPredicted, int numBoxes)
    {
        ThrowIfDisposed();
        var pipeline = GetPipeline("IoULoss", _iouLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numBoxes);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)gradOutput, 0);
        encoder.SetBuffer((MetalGpuBuffer)predicted, 1);
        encoder.SetBuffer((MetalGpuBuffer)target, 2);
        encoder.SetBuffer((MetalGpuBuffer)gradPredicted, 3);
        encoder.SetBytes(numBoxes, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion
}
