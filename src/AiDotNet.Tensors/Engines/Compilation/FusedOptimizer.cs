using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;
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
        float lrAdj = lr / bc1;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var vLr = Vector256.Create(-lrAdj);
            var vEps = Vector256.Create(1e-8f);
            var signMask = Vector256.Create(0x7FFFFFFFu).AsSingle(); // abs mask
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                Avx.Store(m + i, mNew);
                var absG = Avx.And(g, signMask);
                var uScaled = Avx.Multiply(vB2, Avx.LoadVector256(u + i));
                var uNew = Avx.Max(uScaled, absG);
                Avx.Store(u + i, uNew);
                var update = Avx.Divide(mNew, Avx.Add(uNew, vEps));
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, Avx.LoadVector256(param + i)));
            }
        }
#endif
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
                var vmNew = Avx.Max(Avx.LoadVector256(vMax + i), vNew);
                Avx.Store(vMax + i, vmNew);
                var vMaxHat = Avx.Multiply(vmNew, vBc2Inv);
                var denom = Avx.Add(Avx.Sqrt(vMaxHat), vEps);
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.Divide(mNew, denom), Avx.LoadVector256(param + i)));
            }
        }
#endif
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
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1f - beta2);
            var vLr = Vector256.Create(-lr);
            var vEps = Vector256.Create(eps);
            var vBc1Inv = Vector256.Create(1f / bc1);
            var vBc2Inv = Vector256.Create(1f / bc2);
            var vBc1NextInv = Vector256.Create(1f / bc1Next);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                Avx.Store(m + i, mNew);
                var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                Avx.Store(v + i, vNew);
                var mHat = Avx.Multiply(mNew, vBc1Inv);
                var vHat = Avx.Multiply(vNew, vBc2Inv);
                var mNesterov = Avx.Add(Avx.Multiply(vB1, mHat), Avx.Multiply(Avx.Multiply(v1mB1, g), vBc1NextInv));
                var denom = Avx.Add(Avx.Sqrt(vHat), vEps);
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.Divide(mNesterov, denom), Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
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
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vRho = Vector256.Create(rho);
            var v1mRho = Vector256.Create(1f - rho);
            var vEps = Vector256.Create(eps);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var ag = Fma.MultiplyAdd(vRho, Avx.LoadVector256(accumGrad + i), Avx.Multiply(v1mRho, Avx.Multiply(g, g)));
                Avx.Store(accumGrad + i, ag);
                var rmsGrad = Avx.Sqrt(Avx.Add(ag, vEps));
                var rmsUpd = Avx.Sqrt(Avx.Add(Avx.LoadVector256(accumUpdate + i), vEps));
                var update = Avx.Subtract(Vector256<float>.Zero, Avx.Multiply(Avx.Divide(rmsUpd, rmsGrad), g));
                var au = Fma.MultiplyAdd(vRho, Avx.LoadVector256(accumUpdate + i), Avx.Multiply(v1mRho, Avx.Multiply(update, update)));
                Avx.Store(accumUpdate + i, au);
                Avx.Store(param + i, Avx.Add(Avx.LoadVector256(param + i), update));
            }
        }
#endif
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
        // Compute layer-wise norms with AVX2
        float paramNorm = 0f, gradNorm = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var pAcc = Vector256<float>.Zero;
            var gAcc = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var p = Avx.LoadVector256(param + i);
                var g = Avx.LoadVector256(grad + i);
                pAcc = Fma.MultiplyAdd(p, p, pAcc);
                gAcc = Fma.MultiplyAdd(g, g, gAcc);
            }
            paramNorm = SimdKernels.HorizontalSum(pAcc);
            gradNorm = SimdKernels.HorizontalSum(gAcc);
        }
#endif
        for (; i < length; i++)
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

        // Pass 1: update moments + compute param norm (AVX2)
        float paramNormSq = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1f - beta2);
            var pNormAcc = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var p = Avx.LoadVector256(param + i);
                Avx.Store(m + i, Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g)));
                Avx.Store(v + i, Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g))));
                pNormAcc = Fma.MultiplyAdd(p, p, pNormAcc);
            }
            paramNormSq = SimdKernels.HorizontalSum(pNormAcc);
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            paramNormSq += param[i] * param[i];
        }
        float paramNorm = MathF.Sqrt(paramNormSq);

        // Pass 2: compute update norm
        float updateNormSq = 0f;
        for (i = 0; i < length; i++)
        {
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            float update = mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * param[i];
            updateNormSq += update * update;
        }
        float updateNorm = MathF.Sqrt(updateNormSq);

        // Trust ratio + apply update
        float trustRatio = (paramNorm > 0f && updateNorm > 0f) ? paramNorm / updateNorm : 1f;
        float effectiveLr = lr * trustRatio;
        i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-effectiveLr);
            var vEps = Vector256.Create(eps);
            var vWd = Vector256.Create(weightDecay);
            var vBc1Inv = Vector256.Create(1f / bc1);
            var vBc2Inv = Vector256.Create(1f / bc2);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var mHat = Avx.Multiply(Avx.LoadVector256(m + i), vBc1Inv);
                var vHat = Avx.Multiply(Avx.LoadVector256(v + i), vBc2Inv);
                var adamUpdate = Avx.Divide(mHat, Avx.Add(Avx.Sqrt(vHat), vEps));
                var p = Avx.LoadVector256(param + i);
                var fullUpdate = Avx.Add(adamUpdate, Avx.Multiply(vWd, p));
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, fullUpdate, p));
            }
        }
#endif
        for (; i < length; i++)
        {
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            float update = mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * param[i];
            param[i] -= effectiveLr * update;
        }
    }

    /// <summary>AVX2 FTRL: Follow The Regularized Leader.
    /// Partial vectorization: n accumulation and z update use FMA,
    /// soft-thresholding uses SIMD compare+blend for branchless L1 proximal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void FTRLUpdateSimd(
        float* param, float* grad, float* z, float* n, int length,
        float lr, float l1Reg, float l2Reg, float lrPower)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vL1 = Vector256.Create(l1Reg);
            var vNegOne = Vector256.Create(-1f);
            var vOne = Vector256.Create(1f);
            var vZero = Vector256<float>.Zero;
            var absMask = Vector256.Create(0x7FFFFFFFu).AsSingle();
            int simdLen = length & ~7;

            // Scratch buffers outside the loop to avoid stackalloc-in-loop warning
            var sigmaArr = stackalloc float[8];
            var nNewArr = stackalloc float[8];
            var nOldArr = stackalloc float[8];
            var denomArr = stackalloc float[8];

            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var nOld = Avx.LoadVector256(n + i);
                var nNew = Fma.MultiplyAdd(g, g, nOld);
                Avx.Store(n + i, nNew);

                // sigma = (pow(nNew, -lrPower) - pow(nOld, -lrPower)) / lr
                // General case: compute per-element (pow has no SIMD intrinsic)
                Avx.Store(nNewArr, nNew);
                Avx.Store(nOldArr, nOld);
                for (int j = 0; j < 8; j++)
                    sigmaArr[j] = (MathF.Pow(nNewArr[j], -lrPower) - MathF.Pow(nOldArr[j], -lrPower)) / lr;
                var sigma = Avx.LoadVector256(sigmaArr);

                // z += g - sigma * param
                var p = Avx.LoadVector256(param + i);
                var zOld = Avx.LoadVector256(z + i);
                var zNew = Avx.Add(zOld, Avx.Subtract(g, Avx.Multiply(sigma, p)));
                Avx.Store(z + i, zNew);

                // Soft-thresholding: branchless via SIMD compare+blend
                var absZ = Avx.And(zNew, absMask);
                var belowThreshold = Avx.Compare(absZ, vL1, FloatComparisonMode.OrderedLessThanOrEqualSignaling);

                // sign(z): +1 where z>0, -1 where z<0
                var posM = Avx.Compare(zNew, vZero, FloatComparisonMode.OrderedGreaterThanSignaling);
                var negM = Avx.Compare(zNew, vZero, FloatComparisonMode.OrderedLessThanSignaling);
                var signZ = Avx.BlendVariable(vZero, vOne, posM);
                signZ = Avx.BlendVariable(signZ, vNegOne, negM);

                // denom = pow(nNew, -lrPower)/lr + l2Reg
                for (int j = 0; j < 8; j++)
                    denomArr[j] = MathF.Pow(nNewArr[j], -lrPower) / lr + l2Reg;
                var denom = Avx.LoadVector256(denomArr);

                // result = -(z - sign*l1Reg) / denom
                var numerator = Avx.Subtract(zNew, Avx.Multiply(signZ, vL1));
                var result = Avx.Subtract(vZero, Avx.Divide(numerator, denom));

                // Blend: zero where |z| <= l1Reg, result otherwise
                result = Avx.BlendVariable(result, vZero, belowThreshold);
                Avx.Store(param + i, result);
            }
        }
#endif
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
