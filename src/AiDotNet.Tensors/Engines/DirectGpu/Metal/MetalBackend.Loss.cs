// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Loss functions and gradients.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Loss Functions

    /// <summary>
    /// Cross-entropy loss for multi-class classification.
    /// </summary>
    public float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
    {
        ThrowIfDisposed();
        // CPU fallback
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                int idx = b * numClasses + c;
                float p = MathF.Max(preds[idx], 1e-7f);
                totalLoss -= targs[idx] * MathF.Log(p);
            }
        }
        return totalLoss / batchSize;
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
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(MathF.Min(preds[i], 1 - 1e-7f), 1e-7f);
            float t = targs[i];
            totalLoss -= t * MathF.Log(p) + (1 - t) * MathF.Log(1 - p);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Binary cross-entropy backward.
    /// </summary>
    public void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(MathF.Min(preds[i], 1 - 1e-7f), 1e-7f);
            grad[i] = ((p - targs[i]) / (p * (1 - p))) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Mean squared error loss.
    /// </summary>
    public float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            totalLoss += diff * diff;
        }
        return totalLoss / size;
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
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = MathF.Abs(preds[i] - targs[i]);
            if (diff < beta)
            {
                totalLoss += 0.5f * diff * diff / beta;
            }
            else
            {
                totalLoss += diff - 0.5f * beta;
            }
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Smooth L1 backward.
    /// </summary>
    public void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            if (MathF.Abs(diff) < beta)
            {
                grad[i] = diff / beta / size;
            }
            else
            {
                grad[i] = MathF.Sign(diff) / size;
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
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
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(MathF.Min(preds[i], 1 - 1e-7f), 1e-7f);
            float t = targs[i];
            float pt = t * p + (1 - t) * (1 - p);
            float alphaT = t * alpha + (1 - t) * (1 - alpha);
            totalLoss -= alphaT * MathF.Pow(1 - pt, gamma) * MathF.Log(pt);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Focal backward.
    /// </summary>
    public void FocalBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float alpha, float gamma)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(MathF.Min(preds[i], 1 - 1e-7f), 1e-7f);
            float t = targs[i];
            float pt = t * p + (1 - t) * (1 - p);
            float alphaT = t * alpha + (1 - t) * (1 - alpha);
            float factor = MathF.Pow(1 - pt, gamma);

            grad[i] = alphaT * factor * (gamma * pt * MathF.Log(pt) + pt - 1) * (2 * t - 1) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Mean absolute error loss.
    /// </summary>
    public float MaeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            totalLoss += MathF.Abs(preds[i] - targs[i]);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// MAE backward.
    /// </summary>
    public void MaeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            grad[i] = MathF.Sign(preds[i] - targs[i]) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Log-Cosh loss.
    /// </summary>
    public float LogCoshLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            totalLoss += MathF.Log(MathF.Cosh(diff));
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Log-Cosh backward.
    /// </summary>
    public void LogCoshBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            grad[i] = MathF.Tanh(diff) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Quantile loss.
    /// </summary>
    public float QuantileLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float quantile)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = targs[i] - preds[i];
            totalLoss += diff >= 0 ? quantile * diff : (quantile - 1) * diff;
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Quantile backward.
    /// </summary>
    public void QuantileBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float quantile)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float diff = targs[i] - preds[i];
            grad[i] = (diff >= 0 ? -quantile : (1 - quantile)) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Hinge loss.
    /// </summary>
    public float HingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            totalLoss += MathF.Max(0, 1 - targs[i] * preds[i]);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Hinge backward.
    /// </summary>
    public void HingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            grad[i] = (1 - targs[i] * preds[i] > 0) ? -targs[i] / size : 0;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Squared hinge loss.
    /// </summary>
    public float SquaredHingeLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float margin = MathF.Max(0, 1 - targs[i] * preds[i]);
            totalLoss += margin * margin;
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Squared hinge backward.
    /// </summary>
    public void SquaredHingeBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float margin = 1 - targs[i] * preds[i];
            grad[i] = margin > 0 ? -2 * margin * targs[i] / size : 0;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Poisson loss.
    /// </summary>
    public float PoissonLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(preds[i], 1e-7f);
            totalLoss += p - targs[i] * MathF.Log(p);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Poisson backward.
    /// </summary>
    public void PoissonBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(preds[i], 1e-7f);
            grad[i] = (1 - targs[i] / p) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Exponential loss.
    /// </summary>
    public float ExponentialLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            totalLoss += MathF.Exp(-targs[i] * preds[i]);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Exponential backward.
    /// </summary>
    public void ExponentialBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            grad[i] = -targs[i] * MathF.Exp(-targs[i] * preds[i]) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Modified Huber loss.
    /// </summary>
    public float ModifiedHuberLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float yt = targs[i] * preds[i];
            if (yt >= -1)
            {
                float margin = MathF.Max(0, 1 - yt);
                totalLoss += margin * margin;
            }
            else
            {
                totalLoss += -4 * yt;
            }
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Modified Huber backward.
    /// </summary>
    public void ModifiedHuberBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float yt = targs[i] * preds[i];
            if (yt >= 1)
            {
                grad[i] = 0;
            }
            else if (yt >= -1)
            {
                grad[i] = -2 * (1 - yt) * targs[i] / size;
            }
            else
            {
                grad[i] = -4 * targs[i] / size;
            }
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Categorical cross-entropy loss.
    /// </summary>
    public float CategoricalCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(preds[i], 1e-7f);
            totalLoss -= targs[i] * MathF.Log(p);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Categorical cross-entropy backward.
    /// </summary>
    public void CategoricalCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float p = MathF.Max(preds[i], 1e-7f);
            grad[i] = -targs[i] / p / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Charbonnier loss.
    /// </summary>
    public float CharbonnierLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float epsilon)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            totalLoss += MathF.Sqrt(diff * diff + epsilon * epsilon);
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Charbonnier backward.
    /// </summary>
    public void CharbonnierBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float epsilon)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            grad[i] = diff / MathF.Sqrt(diff * diff + epsilon * epsilon) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Elastic net loss.
    /// </summary>
    public float ElasticNetLoss(IGpuBuffer predictions, IGpuBuffer targets, int size, float l1Weight, float l2Weight)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);

        float totalLoss = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            totalLoss += l1Weight * MathF.Abs(diff) + l2Weight * diff * diff;
        }
        return totalLoss / size;
    }

    /// <summary>
    /// Elastic net backward.
    /// </summary>
    public void ElasticNetBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float l1Weight, float l2Weight)
    {
        ThrowIfDisposed();
        var preds = DownloadBuffer(predictions);
        var targs = DownloadBuffer(targets);
        var grad = new float[size];

        for (int i = 0; i < size; i++)
        {
            float diff = preds[i] - targs[i];
            grad[i] = (l1Weight * MathF.Sign(diff) + 2 * l2Weight * diff) / size;
        }

        if (gradInput is MetalGpuBuffer giBuffer)
        {
            giBuffer.CopyFrom(grad);
        }
    }

    /// <summary>
    /// Triplet loss.
    /// </summary>
    public float TripletLoss(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative, int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        var a = DownloadBuffer(anchor);
        var p = DownloadBuffer(positive);
        var n = DownloadBuffer(negative);

        float totalLoss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            float posDist = 0, negDist = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = b * embeddingDim + d;
                float apDiff = a[idx] - p[idx];
                float anDiff = a[idx] - n[idx];
                posDist += apDiff * apDiff;
                negDist += anDiff * anDiff;
            }
            totalLoss += MathF.Max(0, posDist - negDist + margin);
        }
        return totalLoss / batchSize;
    }

    /// <summary>
    /// Triplet loss backward.
    /// </summary>
    public void TripletLossBackward(IGpuBuffer anchor, IGpuBuffer positive, IGpuBuffer negative,
        IGpuBuffer gradAnchor, IGpuBuffer gradPositive, IGpuBuffer gradNegative,
        int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        var a = DownloadBuffer(anchor);
        var p = DownloadBuffer(positive);
        var n = DownloadBuffer(negative);
        var gA = new float[batchSize * embeddingDim];
        var gP = new float[batchSize * embeddingDim];
        var gN = new float[batchSize * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            float posDist = 0, negDist = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = b * embeddingDim + d;
                float apDiff = a[idx] - p[idx];
                float anDiff = a[idx] - n[idx];
                posDist += apDiff * apDiff;
                negDist += anDiff * anDiff;
            }

            if (posDist - negDist + margin > 0)
            {
                for (int d = 0; d < embeddingDim; d++)
                {
                    int idx = b * embeddingDim + d;
                    gA[idx] = 2 * ((a[idx] - p[idx]) - (a[idx] - n[idx])) / batchSize;
                    gP[idx] = -2 * (a[idx] - p[idx]) / batchSize;
                    gN[idx] = 2 * (a[idx] - n[idx]) / batchSize;
                }
            }
        }

        if (gradAnchor is MetalGpuBuffer gaBuffer)
        {
            gaBuffer.CopyFrom(gA);
        }
        if (gradPositive is MetalGpuBuffer gpBuffer)
        {
            gpBuffer.CopyFrom(gP);
        }
        if (gradNegative is MetalGpuBuffer gnBuffer)
        {
            gnBuffer.CopyFrom(gN);
        }
    }

    /// <summary>
    /// Contrastive loss.
    /// </summary>
    public float ContrastiveLoss(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels, int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        var o1 = DownloadBuffer(output1);
        var o2 = DownloadBuffer(output2);
        var l = DownloadBuffer(labels);

        float totalLoss = 0;
        for (int b = 0; b < batchSize; b++)
        {
            float dist = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = b * embeddingDim + d;
                float diff = o1[idx] - o2[idx];
                dist += diff * diff;
            }
            dist = MathF.Sqrt(dist);

            float label = l[b];
            if (label == 0)
            {
                // Similar pairs: minimize distance
                totalLoss += dist * dist;
            }
            else
            {
                // Dissimilar pairs: squared hinge loss
                float marginDist = MathF.Max(0, margin - dist);
                totalLoss += marginDist * marginDist;
            }
        }
        return totalLoss / batchSize;
    }

    /// <summary>
    /// Contrastive loss backward.
    /// </summary>
    public void ContrastiveBackward(IGpuBuffer output1, IGpuBuffer output2, IGpuBuffer labels,
        IGpuBuffer gradOutput1, IGpuBuffer gradOutput2,
        int batchSize, int embeddingDim, float margin)
    {
        ThrowIfDisposed();
        var o1 = DownloadBuffer(output1);
        var o2 = DownloadBuffer(output2);
        var l = DownloadBuffer(labels);
        var g1 = new float[batchSize * embeddingDim];
        var g2 = new float[batchSize * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            float dist = 0;
            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = b * embeddingDim + d;
                float diff = o1[idx] - o2[idx];
                dist += diff * diff;
            }
            dist = MathF.Sqrt(dist + 1e-7f);

            float label = l[b];
            for (int d = 0; d < embeddingDim; d++)
            {
                int idx = b * embeddingDim + d;
                float diff = o1[idx] - o2[idx];

                if (label == 0)
                {
                    g1[idx] = 2 * diff / batchSize;
                    g2[idx] = -2 * diff / batchSize;
                }
                else if (margin - dist > 0)
                {
                    // Gradient of 0.5 * (margin - dist)^2: d/dx = -(margin - dist) * (diff / dist)
                    float marginScale = 2 * (margin - dist);
                    g1[idx] = -marginScale * diff / dist / batchSize;
                    g2[idx] = marginScale * diff / dist / batchSize;
                }
            }
        }

        if (gradOutput1 is MetalGpuBuffer go1Buffer)
        {
            go1Buffer.CopyFrom(g1);
        }
        if (gradOutput2 is MetalGpuBuffer go2Buffer)
        {
            go2Buffer.CopyFrom(g2);
        }
    }

    #endregion
}
