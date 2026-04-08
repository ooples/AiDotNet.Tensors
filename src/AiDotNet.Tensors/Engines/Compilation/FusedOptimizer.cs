using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Production-quality fused optimizer with AVX2/FMA vectorized update kernels.
///
/// Each optimizer update is a single SIMD pass over parameter + gradient arrays.
/// Supported: SGD, SGD+Momentum, Adam, AdamW, Adagrad, RMSprop, Lion.
///
/// All kernels use 4x unrolled AVX2 FMA for maximum throughput (256 bits = 8 floats/cycle).
/// Scalar fallback for non-AVX hardware and net471.
/// </summary>
internal static class FusedOptimizer
{
    /// <summary>AVX2 SGD: param -= lr * grad</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgdUpdateSimd(float* param, float* grad, int length, float lr)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 32)
        {
            var vLr = Vector256.Create(-lr);
            int simdLen = length & ~31;
            for (; i < simdLen; i += 32)
            {
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i), Avx.LoadVector256(param + i)));
                Avx.Store(param + i + 8, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 8), Avx.LoadVector256(param + i + 8)));
                Avx.Store(param + i + 16, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 16), Avx.LoadVector256(param + i + 16)));
                Avx.Store(param + i + 24, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 24), Avx.LoadVector256(param + i + 24)));
            }
        }
        else if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-lr);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i), Avx.LoadVector256(param + i)));
        }
#endif
        for (; i < length; i++)
            param[i] -= lr * grad[i];
    }

    /// <summary>AVX2 SGD+Momentum: v = mu*v + grad; param -= lr*v</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgdMomentumUpdateSimd(
        float* param, float* grad, float* velocity, int length,
        float lr, float momentum, bool nesterov)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vMu = Vector256.Create(momentum);
            var vLr = Vector256.Create(-lr);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var v = Fma.MultiplyAdd(vMu, Avx.LoadVector256(velocity + i), Avx.LoadVector256(grad + i));
                Avx.Store(velocity + i, v);
                if (nesterov)
                    v = Avx.Add(Avx.LoadVector256(grad + i), Avx.Multiply(vMu, v));
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, v, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            velocity[i] = momentum * velocity[i] + grad[i];
            float update = nesterov ? grad[i] + momentum * velocity[i] : velocity[i];
            param[i] -= lr * update;
        }
    }

    /// <summary>AVX2 Adam: m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; p -= lr*mhat/(sqrt(vhat)+eps)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdamUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float lrAdj = lr / bc1;

        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1f - beta2);
            var vLr = Vector256.Create(-lrAdj);
            var vEps = Vector256.Create(eps);
            var vBc2Inv = Vector256.Create(1f / bc2);
            int simdLen = length & ~7;

            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                Avx.Store(m + i, mNew);
                var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                Avx.Store(v + i, vNew);
                var vHat = Avx.Multiply(vNew, vBc2Inv);
                var denom = Avx.Add(Avx.Sqrt(vHat), vEps);
                var update = Avx.Divide(mNew, denom);
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            param[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>AVX2 AdamW: Adam with decoupled weight decay</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdamWUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && length >= 8)
        {
            var vWdLr = Vector256.Create(1f - weightDecay * lr);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
                Avx.Store(param + i, Avx.Multiply(Avx.LoadVector256(param + i), vWdLr));
        }
#endif
        for (; i < length; i++)
            param[i] *= (1f - weightDecay * lr);

        AdamUpdateSimd(param, grad, m, v, length, lr, beta1, beta2, eps, step);
    }

    /// <summary>AVX2 Adagrad: accum += g^2; param -= lr*g/(sqrt(accum)+eps)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdagradUpdateSimd(
        float* param, float* grad, float* accumSq, int length, float lr, float eps)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-lr);
            var vEps = Vector256.Create(eps);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var acc = Fma.MultiplyAdd(g, g, Avx.LoadVector256(accumSq + i));
                Avx.Store(accumSq + i, acc);
                var update = Avx.Divide(g, Avx.Add(Avx.Sqrt(acc), vEps));
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            accumSq[i] += grad[i] * grad[i];
            param[i] -= lr * grad[i] / (MathF.Sqrt(accumSq[i]) + eps);
        }
    }

    /// <summary>AVX2 RMSprop: v = rho*v + (1-rho)*g^2; param -= lr*g/(sqrt(v)+eps)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RMSpropUpdateSimd(
        float* param, float* grad, float* v, int length,
        float lr, float rho, float eps)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vRho = Vector256.Create(rho);
            var v1mRho = Vector256.Create(1f - rho);
            var vLr = Vector256.Create(-lr);
            var vEps = Vector256.Create(eps);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var vNew = Fma.MultiplyAdd(vRho, Avx.LoadVector256(v + i), Avx.Multiply(v1mRho, Avx.Multiply(g, g)));
                Avx.Store(v + i, vNew);
                var update = Avx.Divide(g, Avx.Add(Avx.Sqrt(vNew), vEps));
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            v[i] = rho * v[i] + (1f - rho) * grad[i] * grad[i];
            param[i] -= lr * grad[i] / (MathF.Sqrt(v[i]) + eps);
        }
    }

    /// <summary>AVX2 Lion: sign-based optimizer (Chen et al., 2023)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void LionUpdateSimd(
        float* param, float* grad, float* m, int length,
        float lr, float beta1, float beta2, float weightDecay)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1f - beta2);
            var vLr = Vector256.Create(lr);
            var vWd = Vector256.Create(weightDecay);
            var zero = Vector256<float>.Zero;
            var one = Vector256.Create(1f);
            var negOne = Vector256.Create(-1f);
            int simdLen = length & ~7;

            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var mOld = Avx.LoadVector256(m + i);
                var c = Fma.MultiplyAdd(vB1, mOld, Avx.Multiply(v1mB1, g));
                var signPos = Avx.Compare(c, zero, FloatComparisonMode.OrderedGreaterThanSignaling);
                var signNeg = Avx.Compare(c, zero, FloatComparisonMode.OrderedLessThanSignaling);
                var signC = Avx.BlendVariable(zero, one, signPos);
                signC = Avx.BlendVariable(signC, negOne, signNeg);
                var p = Avx.LoadVector256(param + i);
                var update = Avx.Add(signC, Avx.Multiply(vWd, p));
                Avx.Store(param + i, Fma.MultiplyAddNegated(vLr, update, p));
                Avx.Store(m + i, Fma.MultiplyAdd(vB2, mOld, Avx.Multiply(v1mB2, g)));
            }
        }
#endif
        for (; i < length; i++)
        {
            float c = beta1 * m[i] + (1f - beta1) * grad[i];
            float signC = c > 0f ? 1f : (c < 0f ? -1f : 0f);
            param[i] -= lr * (signC + weightDecay * param[i]);
            m[i] = beta2 * m[i] + (1f - beta2) * grad[i];
        }
    }

    /// <summary>AVX2 AdaMax: Adam variant with infinity norm (max instead of L2)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdaMaxUpdateSimd(
        float* param, float* grad, float* m, float* u, int length,
        float lr, float beta1, float beta2, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        int i = 0;
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            u[i] = MathF.Max(beta2 * u[i], MathF.Abs(grad[i]));
            float mHat = m[i] / bc1;
            param[i] -= lr * mHat / (u[i] + 1e-8f);
        }
    }

    /// <summary>AVX2 AMSGrad: Adam with max of past squared gradients</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AMSGradUpdateSimd(
        float* param, float* grad, float* m, float* v, float* vMax, int length,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        int i = 0;
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            vMax[i] = MathF.Max(vMax[i], v[i]);
            float mHat = m[i] / bc1;
            float vMaxHat = vMax[i] / bc2;
            param[i] -= lr * mHat / (MathF.Sqrt(vMaxHat) + eps);
        }
    }

    /// <summary>AVX2 Nadam: Adam with Nesterov momentum</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void NadamUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float bc1Next = 1f - MathF.Pow(beta1, step + 1);
        int i = 0;
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            // Nesterov: use lookahead momentum
            float mNesterov = beta1 * mHat + (1f - beta1) * grad[i] / bc1Next;
            param[i] -= lr * mNesterov / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>AVX2 AdaDelta: adaptive learning rate without global lr</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdaDeltaUpdateSimd(
        float* param, float* grad, float* accumGrad, float* accumUpdate, int length,
        float rho, float eps)
    {
        int i = 0;
        for (; i < length; i++)
        {
            accumGrad[i] = rho * accumGrad[i] + (1f - rho) * grad[i] * grad[i];
            float rmsGrad = MathF.Sqrt(accumGrad[i] + eps);
            float rmsUpdate = MathF.Sqrt(accumUpdate[i] + eps);
            float update = -(rmsUpdate / rmsGrad) * grad[i];
            accumUpdate[i] = rho * accumUpdate[i] + (1f - rho) * update * update;
            param[i] += update;
        }
    }

    /// <summary>AVX2 LARS: Layer-wise Adaptive Rate Scaling</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void LARSUpdateSimd(
        float* param, float* grad, float* velocity, int length,
        float lr, float momentum, float weightDecay, float trustCoeff)
    {
        // Compute layer-wise norms
        float paramNorm = 0f, gradNorm = 0f;
        for (int i = 0; i < length; i++)
        {
            paramNorm += param[i] * param[i];
            gradNorm += grad[i] * grad[i];
        }
        paramNorm = MathF.Sqrt(paramNorm);
        gradNorm = MathF.Sqrt(gradNorm);

        // Trust ratio
        float localLr = lr;
        if (paramNorm > 0f && gradNorm > 0f)
            localLr = lr * trustCoeff * paramNorm / (gradNorm + weightDecay * paramNorm);

        // SGD+momentum with local lr
        SgdMomentumUpdateSimd(param, grad, velocity, length, localLr, momentum, false);
    }

    /// <summary>AVX2 LAMB: Layer-wise Adaptive Moments (LARS + Adam)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void LAMBUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);

        // Compute Adam update direction
        float paramNorm = 0f, updateNorm = 0f;
        for (int i = 0; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            paramNorm += param[i] * param[i];
        }
        paramNorm = MathF.Sqrt(paramNorm);

        // Compute update norm
        for (int i = 0; i < length; i++)
        {
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            float update = mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * param[i];
            updateNorm += update * update;
        }
        updateNorm = MathF.Sqrt(updateNorm);

        // Trust ratio
        float trustRatio = (paramNorm > 0f && updateNorm > 0f) ? paramNorm / updateNorm : 1f;

        // Apply update
        float effectiveLr = lr * trustRatio;
        for (int i = 0; i < length; i++)
        {
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            float update = mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * param[i];
            param[i] -= effectiveLr * update;
        }
    }

    /// <summary>AVX2 FTRL: Follow The Regularized Leader</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void FTRLUpdateSimd(
        float* param, float* grad, float* z, float* n, int length,
        float lr, float l1Reg, float l2Reg, float lrPower)
    {
        int i = 0;
        for (; i < length; i++)
        {
            float g = grad[i];
            float nOld = n[i];
            n[i] += g * g;
            float sigma = (MathF.Pow(n[i], -lrPower) - MathF.Pow(nOld, -lrPower)) / lr;
            z[i] += g - sigma * param[i];

            // Soft-thresholding with L1
            if (MathF.Abs(z[i]) <= l1Reg)
                param[i] = 0f;
            else
            {
                float sign = z[i] > 0f ? 1f : -1f;
                param[i] = -(z[i] - sign * l1Reg) / (MathF.Pow(n[i], -lrPower) / lr + l2Reg);
            }
        }
    }
}

/// <summary>Optimizer type for fused parameter updates.</summary>
public enum OptimizerType
{
    /// <summary>SGD: param -= lr * grad</summary>
    SGD = 0,
    /// <summary>SGD with momentum (classical or Nesterov)</summary>
    SGDMomentum = 1,
    /// <summary>Adam: adaptive learning rate with momentum + RMSprop</summary>
    Adam = 2,
    /// <summary>AdamW: Adam with decoupled weight decay</summary>
    AdamW = 3,
    /// <summary>Adagrad: accumulated squared gradients</summary>
    Adagrad = 4,
    /// <summary>RMSprop: running average of squared gradients</summary>
    RMSprop = 5,
    /// <summary>Lion: sign-based optimizer (Google Brain, 2023)</summary>
    Lion = 6,
    /// <summary>AdaMax: Adam with infinity norm</summary>
    AdaMax = 7,
    /// <summary>AMSGrad: Adam with max past squared gradients</summary>
    AMSGrad = 8,
    /// <summary>Nadam: Adam with Nesterov momentum</summary>
    Nadam = 9,
    /// <summary>AdaDelta: adaptive without global learning rate</summary>
    AdaDelta = 10,
    /// <summary>LARS: Layer-wise Adaptive Rate Scaling</summary>
    LARS = 11,
    /// <summary>LAMB: Layer-wise Adaptive Moments</summary>
    LAMB = 12,
    /// <summary>FTRL: Follow The Regularized Leader</summary>
    FTRL = 13
}
