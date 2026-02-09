// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Optimizer operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Optimizer Operations

    /// <summary>
    /// SGD with momentum update.
    /// </summary>
    public void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var velData = DownloadBuffer(velocity);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            velData[i] = momentum * velData[i] + grad;
            paramData[i] -= learningRate * velData[i];
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(velocity, velData);
    }

    /// <summary>
    /// Vanilla SGD update (no momentum).
    /// </summary>
    public void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient, float learningRate, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            paramData[i] -= learningRate * grad;
        }

        UploadToBuffer(param, paramData);
    }

    /// <summary>
    /// Adam optimizer update.
    /// </summary>
    public void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var vData = DownloadBuffer(v);

        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc2 = 1.0f - MathF.Pow(beta2, step);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            mData[i] = beta1 * mData[i] + (1 - beta1) * grad;
            vData[i] = beta2 * vData[i] + (1 - beta2) * grad * grad;

            float mHat = mData[i] / bc1;
            float vHat = vData[i] / bc2;

            paramData[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(v, vData);
    }

    /// <summary>
    /// AdamW optimizer update (decoupled weight decay).
    /// </summary>
    public void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var vData = DownloadBuffer(v);

        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc2 = 1.0f - MathF.Pow(beta2, step);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i];
            mData[i] = beta1 * mData[i] + (1 - beta1) * grad;
            vData[i] = beta2 * vData[i] + (1 - beta2) * grad * grad;

            float mHat = mData[i] / bc1;
            float vHat = vData[i] / bc2;

            // Decoupled weight decay
            paramData[i] = paramData[i] * (1 - learningRate * weightDecay) - learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(v, vData);
    }

    /// <summary>
    /// RMSprop optimizer update.
    /// </summary>
    public void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var sqAvgData = DownloadBuffer(squaredAvg);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            sqAvgData[i] = rho * sqAvgData[i] + (1 - rho) * grad * grad;
            paramData[i] -= learningRate * grad / (MathF.Sqrt(sqAvgData[i]) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(squaredAvg, sqAvgData);
    }

    /// <summary>
    /// Adagrad optimizer update.
    /// </summary>
    public void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var accumData = DownloadBuffer(accumulatedGrad);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            accumData[i] += grad * grad;
            paramData[i] -= learningRate * grad / (MathF.Sqrt(accumData[i]) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(accumulatedGrad, accumData);
    }

    /// <summary>
    /// Nesterov Accelerated Gradient (NAG) update.
    /// </summary>
    public void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var velData = DownloadBuffer(velocity);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            float vPrev = velData[i];
            velData[i] = momentum * velData[i] - learningRate * grad;
            paramData[i] += -momentum * vPrev + (1 + momentum) * velData[i];
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(velocity, velData);
    }

    /// <summary>
    /// LARS (Layer-wise Adaptive Rate Scaling) update.
    /// </summary>
    public void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var velData = DownloadBuffer(velocity);

        // Compute norms
        float paramNorm = 0, gradNorm = 0;
        for (int i = 0; i < size; i++)
        {
            paramNorm += paramData[i] * paramData[i];
            gradNorm += gradData[i] * gradData[i];
        }
        paramNorm = MathF.Sqrt(paramNorm);
        gradNorm = MathF.Sqrt(gradNorm);

        // Compute local learning rate
        float localLr = learningRate;
        if (paramNorm > 0 && gradNorm > 0)
        {
            localLr = learningRate * trustCoeff * paramNorm / (gradNorm + weightDecay * paramNorm);
        }

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            velData[i] = momentum * velData[i] + localLr * grad;
            paramData[i] -= velData[i];
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(velocity, velData);
    }

    /// <summary>
    /// LAMB (Layer-wise Adaptive Moments) update.
    /// </summary>
    public void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var vData = DownloadBuffer(v);

        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc2 = 1.0f - MathF.Pow(beta2, step);

        // First compute Adam update direction
        var updateDir = new float[size];
        float updateNorm = 0, paramNorm = 0;

        for (int i = 0; i < size; i++)
        {
            mData[i] = beta1 * mData[i] + (1 - beta1) * gradData[i];
            vData[i] = beta2 * vData[i] + (1 - beta2) * gradData[i] * gradData[i];

            float mHat = mData[i] / bc1;
            float vHat = vData[i] / bc2;

            updateDir[i] = mHat / (MathF.Sqrt(vHat) + epsilon) + weightDecay * paramData[i];
            updateNorm += updateDir[i] * updateDir[i];
            paramNorm += paramData[i] * paramData[i];
        }
        updateNorm = MathF.Sqrt(updateNorm);
        paramNorm = MathF.Sqrt(paramNorm);

        // Compute trust ratio
        float trustRatio = 1.0f;
        if (paramNorm > 0 && updateNorm > 0)
        {
            trustRatio = paramNorm / updateNorm;
        }

        // Apply update
        for (int i = 0; i < size; i++)
        {
            paramData[i] -= learningRate * trustRatio * updateDir[i];
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(v, vData);
    }

    /// <summary>
    /// Adadelta optimizer update.
    /// </summary>
    public void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var accumGradData = DownloadBuffer(accumGrad);
        var accumUpdateData = DownloadBuffer(accumUpdate);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            accumGradData[i] = rho * accumGradData[i] + (1 - rho) * grad * grad;

            float rmsUpdate = MathF.Sqrt(accumUpdateData[i] + epsilon);
            float rmsGrad = MathF.Sqrt(accumGradData[i] + epsilon);
            float update = rmsUpdate / rmsGrad * grad;

            accumUpdateData[i] = rho * accumUpdateData[i] + (1 - rho) * update * update;
            paramData[i] -= update;
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(accumGrad, accumGradData);
        UploadToBuffer(accumUpdate, accumUpdateData);
    }

    /// <summary>
    /// AMSGrad optimizer update.
    /// </summary>
    public void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var vData = DownloadBuffer(v);
        var vMaxData = DownloadBuffer(vMax);

        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc2 = 1.0f - MathF.Pow(beta2, step);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            mData[i] = beta1 * mData[i] + (1 - beta1) * grad;
            vData[i] = beta2 * vData[i] + (1 - beta2) * grad * grad;
            vMaxData[i] = MathF.Max(vMaxData[i], vData[i]);

            float mHat = mData[i] / bc1;
            float vMaxHat = vMaxData[i] / bc2;

            paramData[i] -= learningRate * mHat / (MathF.Sqrt(vMaxHat) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(v, vData);
        UploadToBuffer(vMax, vMaxData);
    }

    /// <summary>
    /// AdaMax optimizer update.
    /// </summary>
    public void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var uData = DownloadBuffer(u);

        float bc1 = 1.0f - MathF.Pow(beta1, step);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            mData[i] = beta1 * mData[i] + (1 - beta1) * grad;
            uData[i] = MathF.Max(beta2 * uData[i], MathF.Abs(grad));

            float mHat = mData[i] / bc1;
            paramData[i] -= learningRate * mHat / (uData[i] + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(u, uData);
    }

    /// <summary>
    /// Lion optimizer update.
    /// </summary>
    public void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);

        for (int i = 0; i < size; i++)
        {
            // Compute update direction using interpolated momentum
            float update = beta1 * mData[i] + (1 - beta1) * gradData[i];
            float signUpdate = MathF.Sign(update);

            // Update momentum
            mData[i] = beta2 * mData[i] + (1 - beta2) * gradData[i];

            // Decoupled weight decay + sign update
            paramData[i] = paramData[i] * (1 - learningRate * weightDecay) - learningRate * signUpdate;
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
    }

    /// <summary>
    /// Nadam optimizer update.
    /// </summary>
    public void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var mData = DownloadBuffer(m);
        var vData = DownloadBuffer(v);

        float bc1 = 1.0f - MathF.Pow(beta1, step);
        float bc1Next = 1.0f - MathF.Pow(beta1, step + 1);
        float bc2 = 1.0f - MathF.Pow(beta2, step);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i] + weightDecay * paramData[i];
            mData[i] = beta1 * mData[i] + (1 - beta1) * grad;
            vData[i] = beta2 * vData[i] + (1 - beta2) * grad * grad;

            float mHat = mData[i] / bc1;
            float vHat = vData[i] / bc2;

            // Nesterov momentum
            float mNesterov = beta1 * mHat + (1 - beta1) * grad / bc1Next;

            paramData[i] -= learningRate * mNesterov / (MathF.Sqrt(vHat) + epsilon);
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(m, mData);
        UploadToBuffer(v, vData);
    }

    /// <summary>
    /// FTRL optimizer update.
    /// </summary>
    public void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
    {
        ThrowIfDisposed();

        var paramData = DownloadBuffer(param);
        var gradData = DownloadBuffer(gradient);
        var zData = DownloadBuffer(z);
        var nData = DownloadBuffer(n);

        for (int i = 0; i < size; i++)
        {
            float grad = gradData[i];
            float nPrev = nData[i];
            nData[i] += grad * grad;

            float sigma = (MathF.Sqrt(nData[i]) - MathF.Sqrt(nPrev)) / learningRate;
            zData[i] += grad - sigma * paramData[i];

            // Soft thresholding
            float zSign = MathF.Sign(zData[i]);
            if (MathF.Abs(zData[i]) <= l1Reg)
            {
                paramData[i] = 0;
            }
            else
            {
                float denom = (beta + MathF.Sqrt(nData[i])) / learningRate + l2Reg;
                paramData[i] = -(zData[i] - zSign * l1Reg) / denom;
            }
        }

        UploadToBuffer(param, paramData);
        UploadToBuffer(z, zData);
        UploadToBuffer(n, nData);
    }

    #endregion
}
