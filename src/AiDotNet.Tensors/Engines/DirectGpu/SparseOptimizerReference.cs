// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// CPU reference implementation for sparse optimizer scatter updates.
/// Used by staging-based GPU backends so their semantics match the native sparse kernels.
/// </summary>
internal static class SparseOptimizerReference
{
    public static void ValidateSparseArgs(float[] param, float[] indices, float[] values, int nnz)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
        if (nnz > indices.Length)
            throw new ArgumentOutOfRangeException(nameof(nnz), "nnz exceeds sparse index buffer length.");
        if (nnz > values.Length)
            throw new ArgumentOutOfRangeException(nameof(nnz), "nnz exceeds sparse value buffer length.");
    }

    public static int DecodeIndex(float raw, int maxExclusive)
    {
        int bitcast = SingleToInt32BitsCompat(raw);
        if (bitcast >= 0 && bitcast < maxExclusive)
            return bitcast;

        int numeric = (int)raw;
        if (numeric >= 0 && numeric < maxExclusive && raw == numeric)
            return numeric;

        throw new ArgumentOutOfRangeException(nameof(raw), $"Sparse index is outside [0, {maxExclusive}).");
    }

    public static void SparseSgdUpdate(
        float[] param, float[] indices, float[] values, int nnz,
        float learningRate, float weightDecay)
    {
        ValidateSparseArgs(param, indices, values, nnz);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            param[i] -= learningRate * grad;
        }
    }

    public static void SparseSgdMomentumUpdate(
        float[] param, float[] velocity, float[] indices, float[] values, int nnz,
        float learningRate, float momentum, float weightDecay)
    {
        ValidateSparseState(param, velocity, indices, values, nnz);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float v = momentum * velocity[i] + grad;
            velocity[i] = v;
            param[i] -= learningRate * v;
        }
    }

    public static void SparseAdamUpdate(
        float[] param, float[] m, float[] v, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateBiasCorrected(param, m, v, indices, values, nnz, epsilon, step);
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            float mVal = beta1 * m[i] + (1f - beta1) * grad;
            float vVal = beta2 * v[i] + (1f - beta2) * grad * grad;
            m[i] = mVal;
            v[i] = vVal;
            float update = learningRate * (mVal / bc1) / (MathF.Sqrt(vVal / bc2) + epsilon);
            if (weightDecay > 0f) update += learningRate * weightDecay * param[i];
            param[i] -= update;
        }
    }

    public static void SparseAdamWUpdate(
        float[] param, float[] m, float[] v, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateBiasCorrected(param, m, v, indices, values, nnz, epsilon, step);
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            float p = param[i];
            if (weightDecay > 0f) p -= learningRate * weightDecay * p;
            float mVal = beta1 * m[i] + (1f - beta1) * grad;
            float vVal = beta2 * v[i] + (1f - beta2) * grad * grad;
            m[i] = mVal;
            v[i] = vVal;
            param[i] = p - learningRate * (mVal / bc1) / (MathF.Sqrt(vVal / bc2) + epsilon);
        }
    }

    public static void SparseRmspropUpdate(
        float[] param, float[] squaredAvg, float[] indices, float[] values, int nnz,
        float learningRate, float rho, float epsilon, float weightDecay)
    {
        ValidateSparseState(param, squaredAvg, indices, values, nnz);
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float sq = rho * squaredAvg[i] + (1f - rho) * grad * grad;
            squaredAvg[i] = sq;
            param[i] -= learningRate * grad / (MathF.Sqrt(sq) + epsilon);
        }
    }

    public static void SparseAdagradUpdate(
        float[] param, float[] accum, float[] indices, float[] values, int nnz,
        float learningRate, float epsilon, float weightDecay)
    {
        ValidateSparseState(param, accum, indices, values, nnz);
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float a = accum[i] + grad * grad;
            accum[i] = a;
            param[i] -= learningRate * grad / (MathF.Sqrt(a) + epsilon);
        }
    }

    public static void SparseNagUpdate(
        float[] param, float[] velocity, float[] indices, float[] values, int nnz,
        float learningRate, float momentum, float weightDecay)
    {
        ValidateSparseState(param, velocity, indices, values, nnz);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float vOld = velocity[i];
            float vNew = momentum * vOld + grad;
            velocity[i] = vNew;
            param[i] -= learningRate * ((1f + momentum) * vNew - momentum * vOld);
        }
    }

    public static void SparseAdadeltaUpdate(
        float[] param, float[] accumGrad, float[] accumUpdate, float[] indices, float[] values, int nnz,
        float rho, float epsilon, float weightDecay)
    {
        ValidateSparseState(param, accumGrad, indices, values, nnz);
        EnsureStateLength(param, accumUpdate, nameof(accumUpdate));
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
        float oneMinusRho = 1f - rho;
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float ag = rho * accumGrad[i] + oneMinusRho * grad * grad;
            accumGrad[i] = ag;
            float dx = MathF.Sqrt(accumUpdate[i] + epsilon) / MathF.Sqrt(ag + epsilon) * grad;
            accumUpdate[i] = rho * accumUpdate[i] + oneMinusRho * dx * dx;
            param[i] -= dx;
        }
    }

    public static void SparseAmsgradUpdate(
        float[] param, float[] m, float[] v, float[] vMax, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateBiasCorrected(param, m, v, indices, values, nnz, epsilon, step);
        EnsureStateLength(param, vMax, nameof(vMax));
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float mVal = beta1 * m[i] + (1f - beta1) * grad;
            float vVal = beta2 * v[i] + (1f - beta2) * grad * grad;
            m[i] = mVal;
            v[i] = vVal;
            float vMaxNew = MathF.Max(vMax[i], vVal);
            vMax[i] = vMaxNew;
            param[i] -= learningRate * (mVal / bc1) / (MathF.Sqrt(vMaxNew / bc2) + epsilon);
        }
    }

    public static void SparseAdamaxUpdate(
        float[] param, float[] m, float[] u, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateBiasCorrected(param, m, u, indices, values, nnz, epsilon, step);
        float lrAdj = learningRate / (1f - MathF.Pow(beta1, step));
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float mVal = beta1 * m[i] + (1f - beta1) * grad;
            float uVal = MathF.Max(beta2 * u[i], MathF.Abs(grad));
            m[i] = mVal;
            u[i] = uVal;
            param[i] -= lrAdj * mVal / (uVal + epsilon);
        }
    }

    public static void SparseLionUpdate(
        float[] param, float[] m, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float weightDecay)
    {
        ValidateSparseState(param, m, indices, values, nnz);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            float c = beta1 * m[i] + (1f - beta1) * grad;
            float update = MathF.Sign(c);
            if (weightDecay > 0f) update += weightDecay * param[i];
            param[i] -= learningRate * update;
            m[i] = beta2 * m[i] + (1f - beta2) * grad;
        }
    }

    public static void SparseNadamUpdate(
        float[] param, float[] m, float[] v, float[] indices, float[] values, int nnz,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        ValidateBiasCorrected(param, m, v, indices, values, nnz, epsilon, step);
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            if (weightDecay > 0f) grad += weightDecay * param[i];
            float mVal = beta1 * m[i] + (1f - beta1) * grad;
            float vVal = beta2 * v[i] + (1f - beta2) * grad * grad;
            m[i] = mVal;
            v[i] = vVal;
            float mHat = (beta1 * mVal + (1f - beta1) * grad) / bc1;
            param[i] -= learningRate * mHat / (MathF.Sqrt(vVal / bc2) + epsilon);
        }
    }

    public static void SparseFtrlUpdate(
        float[] param, float[] z, float[] n, float[] indices, float[] values, int nnz,
        float learningRate, float l1Reg, float l2Reg, float beta)
    {
        ValidateSparseState(param, z, indices, values, nnz);
        EnsureStateLength(param, n, nameof(n));
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float grad = values[k];
            float nOld = n[i];
            float nNew = nOld + grad * grad;
            n[i] = nNew;
            float sigma = (MathF.Sqrt(nNew) - MathF.Sqrt(nOld)) / learningRate;
            z[i] += grad - sigma * param[i];
            float zVal = z[i];
            if (MathF.Abs(zVal) <= l1Reg)
            {
                param[i] = 0f;
            }
            else
            {
                float sign = zVal > 0f ? 1f : -1f;
                param[i] = (sign * l1Reg - zVal) / ((beta + MathF.Sqrt(nNew)) / learningRate + l2Reg);
            }
        }
    }

    public static void SparseProximalL1Update(
        float[] param, float[] indices, float[] values, int nnz,
        float learningRate, float l1Strength)
    {
        ValidateSparseArgs(param, indices, values, nnz);
        for (int k = 0; k < nnz; k++)
        {
            int i = DecodeIndex(indices[k], param.Length);
            float p = param[i] - learningRate * values[k];
            float threshold = learningRate * l1Strength;
            if (p > threshold) param[i] = p - threshold;
            else if (p < -threshold) param[i] = p + threshold;
            else param[i] = 0f;
        }
    }

    private static void ValidateSparseState(float[] param, float[] state, float[] indices, float[] values, int nnz)
    {
        ValidateSparseArgs(param, indices, values, nnz);
        EnsureStateLength(param, state, nameof(state));
    }

    private static void ValidateBiasCorrected(
        float[] param, float[] state1, float[] state2, float[] indices, float[] values, int nnz,
        float epsilon, int step)
    {
        ValidateSparseState(param, state1, indices, values, nnz);
        EnsureStateLength(param, state2, nameof(state2));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));
    }

    private static void EnsureStateLength(float[] param, float[] state, string name)
    {
        if (state is null) throw new ArgumentNullException(name);
        if (state.Length < param.Length)
            throw new ArgumentException("Sparse optimizer state buffer is shorter than the parameter buffer.", name);
    }

#if NETFRAMEWORK
    // BitConverter.SingleToInt32Bits is unavailable on net471; reinterpret the bits via a pointer there.
    private static unsafe int SingleToInt32BitsCompat(float value) => *(int*)&value;
#else
    private static int SingleToInt32BitsCompat(float value) => BitConverter.SingleToInt32Bits(value);
#endif
}
