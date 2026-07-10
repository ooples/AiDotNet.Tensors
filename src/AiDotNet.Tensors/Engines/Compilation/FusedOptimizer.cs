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
        // Thin wrapper — hoist the step-global bias-correction MathF.Pow (see the
        // double AdamUpdateSimd wrapper) so a per-parameter loop pays it once/step.
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        AdamUpdateSimd(param, grad, m, v, length, lr, beta1, beta2, eps, bc1, bc2);
    }

    /// <summary>Adam (float) with precomputed step-global bias corrections. Bit-identical
    /// to the step overload; lets a per-parameter loop pay the two MathF.Pow once/step.</summary>
    internal static unsafe void AdamUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, float bc1, float bc2)
    {
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

    // ────────────────────────────────────────────────────────────────────
    // BF16 moment-storage Adam/AdamW (#1745 follow-up). The Adam moment
    // buffers m, v are the dominant training-step memory cost — 2x the model
    // weights at fp32. Storing them as bfloat16 halves that to 1x while keeping
    // the FULL float32 exponent (only the mantissa shortens 23→7 bits), so
    // Adam's trajectory changes negligibly and — unlike int8 block-quant — no
    // per-block scales are needed. These kernels widen each stored bf16 moment
    // to fp32, run the EXACT same update math as the fp32 kernels, and narrow
    // the result back to bf16 with round-to-nearest-even. Compute stays fp32;
    // only the resident moment storage is halved. This lets large models keep
    // the fused fast path AND the halved optimizer-state footprint, instead of
    // falling off the fused path onto the eager tape when memory matters.
    //
    // Parameters and gradients remain fp32 (their precision is load-bearing for
    // the weight update); bf16 applies only to the m/v accumulators, exactly as
    // bitsandbytes / DeepSpeed store the optimizer state at reduced precision.
    // Scalar-only: bf16 widen/narrow has no broad AVX2 intrinsic (AVX512-BF16
    // is not assumed), and the win here is resident bytes, not peak throughput —
    // this path is still far faster than the eager autograd tape it replaces.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Widen a bfloat16 (the top 16 bits of a float32) to float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float Bf16ToFloat(ushort b)
    {
        uint u = (uint)b << 16;
        return *(float*)&u;
    }

    /// <summary>
    /// Narrow a float32 to bfloat16 with round-to-nearest-even. NaN is mapped to a
    /// quiet bf16 NaN (never silently truncated into an infinity), matching the
    /// IEEE/​PyTorch bf16 cast.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe ushort Bf16FromFloat(float f)
    {
        uint bits = *(uint*)&f;
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u); // preserve sign, force quiet NaN
        uint rounding = 0x7FFFu + ((bits >> 16) & 1u); // round-to-nearest-even bias
        return (ushort)((bits + rounding) >> 16);
    }

    /// <summary>Adam update with bfloat16 moment storage (fp32 params/grads, fp32 math).</summary>
    internal static unsafe void AdamUpdateBf16Simd(
        float* param, float* grad, ushort* m, ushort* v, int length,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < length; i++)
        {
            float mi = beta1 * Bf16ToFloat(m[i]) + (1f - beta1) * grad[i];
            float vi = beta2 * Bf16ToFloat(v[i]) + (1f - beta2) * grad[i] * grad[i];
            m[i] = Bf16FromFloat(mi);
            v[i] = Bf16FromFloat(vi);
            float mHat = mi / bc1;
            float vHat = vi / bc2;
            param[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>AdamW (decoupled weight decay) with bfloat16 moment storage.</summary>
    internal static unsafe void AdamWUpdateBf16Simd(
        float* param, float* grad, ushort* m, ushort* v, int length,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        for (int i = 0; i < length; i++)
            param[i] *= (1f - weightDecay * lr);
        AdamUpdateBf16Simd(param, grad, m, v, length, lr, beta1, beta2, eps, step);
    }

    /// <summary>
    /// Adam update with true block-quantized int8 moment storage. The resident
    /// state is one byte per m/v element plus one fp64 scale per block; update
    /// math widens to fp32 and then requantizes the new moments.
    /// </summary>
    internal static unsafe void AdamUpdateInt8BlockQuantized(
        float* param,
        float* grad,
        byte* mQuant,
        byte* vQuant,
        double* mScales,
        double* vScales,
        int length,
        int blockSize,
        float lr,
        float beta1,
        float beta2,
        float eps,
        int step)
    {
        // blockSize == 0 makes `start += blockSize` spin forever; a negative value walks `start`
        // backwards and grows `block` unbounded → out-of-bounds scale access. Reject up front.
        if (blockSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize,
                "int8 block-quantized moment update requires blockSize > 0.");

        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float oneMinusBeta1 = 1f - beta1;
        float oneMinusBeta2 = 1f - beta2;

        for (int block = 0, start = 0; start < length; block++, start += blockSize)
        {
            int end = Math.Min(start + blockSize, length);
            // A stored scale of 0 is the natural ALL-ZERO moment state (initial state / an untouched
            // block). Decode those as exactly 0 rather than flooring the scale to 1e-10 — the floor
            // would turn the default/zero quantized bytes into spurious non-zero moments (e.g. an M
            // byte of 0 decodes to (0-128)*1e-10) and nudge parameters on the first step.
            double mScaleRaw = mScales[block];
            double vScaleRaw = vScales[block];
            bool mHasHistory = mScaleRaw > 0.0;
            bool vHasHistory = vScaleRaw > 0.0;
            float oldMScale = (float)mScaleRaw;
            float oldVScale = (float)vScaleRaw;
            float maxAbsM = 0f;
            float maxAbsV = 0f;

            for (int i = start; i < end; i++)
            {
                float oldM = mHasHistory ? ((int)mQuant[i] - 128) * oldMScale : 0f;
                float oldV = vHasHistory ? vQuant[i] * oldVScale : 0f;
                float g = grad[i];
                float newM = beta1 * oldM + oneMinusBeta1 * g;
                float newV = beta2 * oldV + oneMinusBeta2 * g * g;
                maxAbsM = MathF.Max(maxAbsM, MathF.Abs(newM));
                maxAbsV = MathF.Max(maxAbsV, MathF.Abs(newV));
            }

            float newMScale = MathF.Max(maxAbsM / 127f, 1e-10f);
            float newVScale = MathF.Max(maxAbsV / 255f, 1e-10f);
            mScales[block] = newMScale;
            vScales[block] = newVScale;

            for (int i = start; i < end; i++)
            {
                float oldM = mHasHistory ? ((int)mQuant[i] - 128) * oldMScale : 0f;
                float oldV = vHasHistory ? vQuant[i] * oldVScale : 0f;
                float g = grad[i];
                float newM = beta1 * oldM + oneMinusBeta1 * g;
                float newV = beta2 * oldV + oneMinusBeta2 * g * g;
                float mHat = newM / bc1;
                float vHat = newV / bc2;
                param[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);

                int qM = (int)Math.Round(newM / newMScale, MidpointRounding.ToEven);
                if (qM < -127) qM = -127;
                else if (qM > 127) qM = 127;
                int qV = (int)Math.Round(newV / newVScale, MidpointRounding.ToEven);
                if (qV < 0) qV = 0;
                else if (qV > 255) qV = 255;
                mQuant[i] = (byte)(qM + 128);
                vQuant[i] = (byte)qV;
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Double-precision overloads — engaged for `Tensor<double>` models on
    // the fused-compiled training path. PR #319 follow-up: the
    // CompiledTrainingPlan / CompiledTapeTrainingStep fast paths used to
    // gate `typeof(T) != typeof(float)` returning false, so models with
    // `Tensor<double>` parameters fell back to the eager autograd tape —
    // a 7-10× wall-clock penalty on ViT-Base scale (most of the
    // 3024 ms/iter "tape framework overhead" the consumer harness
    // measured). Adding double mirrors lets those models hit the same
    // compile-once-replay-many path.
    //
    // Vector256<double> is 4-wide vs Vector256<float>'s 8-wide, so the
    // unroll factor halves and the SIMD-bound length thresholds halve
    // accordingly. Hardware support: same Avx + Fma + Sse intrinsics.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>AVX2 SGD (double): param -= lr * grad</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgdUpdateSimd(double* param, double* grad, int length, double lr)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 16)
        {
            var vLr = Vector256.Create(-lr);
            int simdLen = length & ~15;
            for (; i < simdLen; i += 16)
            {
                Avx.Store(param + i,      Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i),      Avx.LoadVector256(param + i)));
                Avx.Store(param + i + 4,  Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 4),  Avx.LoadVector256(param + i + 4)));
                Avx.Store(param + i + 8,  Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 8),  Avx.LoadVector256(param + i + 8)));
                Avx.Store(param + i + 12, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i + 12), Avx.LoadVector256(param + i + 12)));
            }
        }
        else if (Fma.IsSupported && length >= 4)
        {
            var vLr = Vector256.Create(-lr);
            int simdLen = length & ~3;
            for (; i < simdLen; i += 4)
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.LoadVector256(grad + i), Avx.LoadVector256(param + i)));
        }
#endif
        for (; i < length; i++)
            param[i] -= lr * grad[i];
    }

    /// <summary>AVX2 Adam (double): m = β1·m + (1-β1)·g; v = β2·v + (1-β2)·g²; param -= lr·m̂/(√v̂ + ε)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdamUpdateSimd(
        double* param, double* grad, double* m, double* v, int length,
        double lr, double beta1, double beta2, double eps, int step)
    {
        // Thin wrapper: bc1 = 1-β1^step, bc2 = 1-β2^step are STEP-GLOBAL (identical
        // for every parameter in a step), but this kernel runs once PER PARAMETER —
        // so the two Math.Pow here cost ~2×(param count) transcendental calls per
        // step, concentrated on a deep model's many small tensors. A per-parameter
        // loop should compute bc1/bc2 ONCE and call the bc-taking overload below.
        double bc1 = 1.0 - System.Math.Pow(beta1, step);
        double bc2 = 1.0 - System.Math.Pow(beta2, step);
        AdamUpdateSimd(param, grad, m, v, length, lr, beta1, beta2, eps, bc1, bc2);
    }

    /// <summary>Adam update with the STEP-GLOBAL bias corrections precomputed by the
    /// caller (bc1 = 1-β1^step, bc2 = 1-β2^step). Bit-identical to the step overload;
    /// exists so a per-parameter loop pays the two Math.Pow once per step, not per
    /// parameter.</summary>
    internal static unsafe void AdamUpdateSimd(
        double* param, double* grad, double* m, double* v, int length,
        double lr, double beta1, double beta2, double eps, double bc1, double bc2)
    {
        double lrAdj = lr / bc1;

        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 4)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1.0 - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1.0 - beta2);
            var vLr = Vector256.Create(-lrAdj);
            var vEps = Vector256.Create(eps);
            var vBc2Inv = Vector256.Create(1.0 / bc2);
            int simdLen = length & ~3;

            for (; i < simdLen; i += 4)
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
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            double mHat = m[i] / bc1;
            double vHat = v[i] / bc2;
            param[i] -= lr * mHat / (System.Math.Sqrt(vHat) + eps);
        }
    }

    /// <summary>AVX2 AdamW (double): Adam with decoupled weight decay</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdamWUpdateSimd(
        double* param, double* grad, double* m, double* v, int length,
        double lr, double beta1, double beta2, double eps, double weightDecay, int step)
    {
        // Thin wrapper — hoist the step-global bias-correction Math.Pow (see the
        // AdamUpdateSimd wrapper) so a per-parameter loop pays it once per step.
        double bc1 = 1.0 - System.Math.Pow(beta1, step);
        double bc2 = 1.0 - System.Math.Pow(beta2, step);
        AdamWUpdateSimd(param, grad, m, v, length, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    /// <summary>Single-pass fused AdamW with precomputed step-global bias corrections
    /// (bc1 = 1-β1^step, bc2 = 1-β2^step). Bit-identical to the step overload.</summary>
    internal static unsafe void AdamWUpdateSimd(
        double* param, double* grad, double* m, double* v, int length,
        double lr, double beta1, double beta2, double eps, double weightDecay, double bc1, double bc2)
    {
        // SINGLE-PASS fused AdamW. The prior form did a separate full pass over
        // `param` (read+write every element) to apply the decoupled weight decay,
        // then called AdamUpdateSimd which read+wrote `param` AGAIN — two
        // memory-bound passes over the parameter array where Adam does one. Fold
        // the decay (param *= 1 - wd*lr) into the Adam update loop so `param` is
        // touched once. Bit-identical to the two-pass form: the scaled parameter
        // is a double, so keeping it in a register is exactly the value the old
        // pre-scale pass stored to memory and the Adam pass reloaded. Mirrors the
        // single-pass structure Adam already had (the optimization AdamW missed).
        double lrAdj = lr / bc1;
        double wdScale = 1.0 - weightDecay * lr;   // hoisted out of the loop

        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 4)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1.0 - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1.0 - beta2);
            var vLr = Vector256.Create(-lrAdj);
            var vEps = Vector256.Create(eps);
            var vBc2Inv = Vector256.Create(1.0 / bc2);
            var vWd = Vector256.Create(wdScale);
            int simdLen = length & ~3;

            for (; i < simdLen; i += 4)
            {
                var g = Avx.LoadVector256(grad + i);
                var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                Avx.Store(m + i, mNew);
                var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                Avx.Store(v + i, vNew);
                var vHat = Avx.Multiply(vNew, vBc2Inv);
                var denom = Avx.Add(Avx.Sqrt(vHat), vEps);
                var update = Avx.Divide(mNew, denom);
                var pScaled = Avx.Multiply(Avx.LoadVector256(param + i), vWd);   // decoupled wd, in-register
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, pScaled));
            }
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            double mHat = m[i] / bc1;
            double vHat = v[i] / bc2;
            double pScaled = param[i] * wdScale;
            param[i] = pScaled - lr * mHat / (System.Math.Sqrt(vHat) + eps);
        }
    }

    /// <summary>
    /// Multi-tensor (PyTorch <c>foreach</c>-style) fused AdamW over a LIST of parameter
    /// tensors. Builds the step-global SIMD constant vectors ONCE and reuses them for
    /// every parameter, instead of rebuilding all eight broadcasts (and paying a method
    /// call + the per-call bias-correction) once per tensor — the redundancy that a deep
    /// model with hundreds of parameter tensors pays every step. Bit-identical to calling
    /// the single-tensor bc1/bc2 overload per parameter: identical arithmetic, identical
    /// SIMD / scalar-tail split per tensor. Tensors with <c>lengths[t] == 0</c> (unexercised
    /// params with no live gradient) are skipped, matching the per-param loop's guard.
    /// </summary>
    internal static unsafe void AdamWUpdateSimdMulti(
        double[][] paramArrays, double[][] gradArrays, double[][] mArrays, double[][] vArrays,
        int[] lengths, int count,
        double lr, double beta1, double beta2, double eps, double weightDecay, double bc1, double bc2)
    {
        double lrAdj = lr / bc1;
        double wdScale = 1.0 - weightDecay * lr;
#if NET5_0_OR_GREATER
        bool useSimd = Fma.IsSupported;
        var vB1 = Vector256.Create(beta1);
        var v1mB1 = Vector256.Create(1.0 - beta1);
        var vB2 = Vector256.Create(beta2);
        var v1mB2 = Vector256.Create(1.0 - beta2);
        var vLr = Vector256.Create(-lrAdj);
        var vEps = Vector256.Create(eps);
        var vBc2Inv = Vector256.Create(1.0 / bc2);
        var vWd = Vector256.Create(wdScale);
#endif
        for (int t = 0; t < count; t++)
        {
            int length = lengths[t];
            if (length == 0) continue;
            var pa = paramArrays[t]; var ga = gradArrays[t]; var ma = mArrays[t]; var va = vArrays[t];
            if (pa.Length == 0 || ga.Length == 0) continue;
            fixed (double* param = pa, grad = ga, m = ma, v = va)
            {
                int i = 0;
#if NET5_0_OR_GREATER
                if (useSimd && length >= 4)
                {
                    int simdLen = length & ~3;
                    for (; i < simdLen; i += 4)
                    {
                        var g = Avx.LoadVector256(grad + i);
                        var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                        Avx.Store(m + i, mNew);
                        var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                        Avx.Store(v + i, vNew);
                        var vHat = Avx.Multiply(vNew, vBc2Inv);
                        var denom = Avx.Add(Avx.Sqrt(vHat), vEps);
                        var update = Avx.Divide(mNew, denom);
                        var pScaled = Avx.Multiply(Avx.LoadVector256(param + i), vWd);
                        Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, pScaled));
                    }
                }
#endif
                for (; i < length; i++)
                {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                    double mHat = m[i] / bc1;
                    double vHat = v[i] / bc2;
                    double pScaled = param[i] * wdScale;
                    param[i] = pScaled - lr * mHat / (System.Math.Sqrt(vHat) + eps);
                }
            }
        }
    }

    /// <summary>Multi-tensor (foreach-style) Adam over a LIST of double parameter tensors.
    /// Builds the SIMD constants once, reuses across every tensor; bit-identical to the
    /// per-parameter bc1/bc2 Adam overload. (Adam == AdamW without the decoupled decay.)</summary>
    internal static unsafe void AdamUpdateSimdMulti(
        double[][] paramArrays, double[][] gradArrays, double[][] mArrays, double[][] vArrays,
        int[] lengths, int count,
        double lr, double beta1, double beta2, double eps, double bc1, double bc2)
    {
        double lrAdj = lr / bc1;
#if NET5_0_OR_GREATER
        bool useSimd = Fma.IsSupported;
        var vB1 = Vector256.Create(beta1);
        var v1mB1 = Vector256.Create(1.0 - beta1);
        var vB2 = Vector256.Create(beta2);
        var v1mB2 = Vector256.Create(1.0 - beta2);
        var vLr = Vector256.Create(-lrAdj);
        var vEps = Vector256.Create(eps);
        var vBc2Inv = Vector256.Create(1.0 / bc2);
#endif
        for (int t = 0; t < count; t++)
        {
            int length = lengths[t];
            if (length == 0) continue;
            var pa = paramArrays[t]; var ga = gradArrays[t]; var ma = mArrays[t]; var va = vArrays[t];
            if (pa.Length == 0 || ga.Length == 0) continue;
            fixed (double* param = pa, grad = ga, m = ma, v = va)
            {
                int i = 0;
#if NET5_0_OR_GREATER
                if (useSimd && length >= 4)
                {
                    int simdLen = length & ~3;
                    for (; i < simdLen; i += 4)
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
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                    double mHat = m[i] / bc1;
                    double vHat = v[i] / bc2;
                    param[i] -= lr * mHat / (System.Math.Sqrt(vHat) + eps);
                }
            }
        }
    }

    /// <summary>AVX2 AdamW: Adam with decoupled weight decay</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdamWUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, float weightDecay, int step)
    {
        // Thin wrapper — hoist the step-global bias-correction MathF.Pow so a
        // per-parameter loop pays it once/step (see the double overloads).
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        AdamWUpdateSimd(param, grad, m, v, length, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    /// <summary>Single-pass fused AdamW (float) with precomputed step-global bias
    /// corrections. Bit-identical to the step overload.</summary>
    internal static unsafe void AdamWUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2)
    {
        // SINGLE-PASS fused AdamW — see the double overload above. Folds the
        // decoupled weight decay into the Adam update loop so `param` is read+
        // written once instead of twice. Bit-identical to the prior two-pass form.
        float lrAdj = lr / bc1;
        float wdScale = 1f - weightDecay * lr;

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
            var vWd = Vector256.Create(wdScale);
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
                var pScaled = Avx.Multiply(Avx.LoadVector256(param + i), vWd);
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, pScaled));
            }
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            float pScaled = param[i] * wdScale;
            param[i] = pScaled - lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>Multi-tensor (foreach-style) fused AdamW over a LIST of float parameter
    /// tensors — see the double overload. Builds the SIMD constants once, reuses across
    /// every tensor; bit-identical to the per-parameter bc1/bc2 overload.</summary>
    internal static unsafe void AdamWUpdateSimdMulti(
        float[][] paramArrays, float[][] gradArrays, float[][] mArrays, float[][] vArrays,
        int[] lengths, int count,
        float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2)
    {
        float lrAdj = lr / bc1;
        float wdScale = 1f - weightDecay * lr;
#if NET5_0_OR_GREATER
        bool useSimd = Fma.IsSupported;
        var vB1 = Vector256.Create(beta1);
        var v1mB1 = Vector256.Create(1f - beta1);
        var vB2 = Vector256.Create(beta2);
        var v1mB2 = Vector256.Create(1f - beta2);
        var vLr = Vector256.Create(-lrAdj);
        var vEps = Vector256.Create(eps);
        var vBc2Inv = Vector256.Create(1f / bc2);
        var vWd = Vector256.Create(wdScale);
#endif
        for (int t = 0; t < count; t++)
        {
            int length = lengths[t];
            if (length == 0) continue;
            var pa = paramArrays[t]; var ga = gradArrays[t]; var ma = mArrays[t]; var va = vArrays[t];
            if (pa.Length == 0 || ga.Length == 0) continue;
            fixed (float* param = pa, grad = ga, m = ma, v = va)
            {
                int i = 0;
#if NET5_0_OR_GREATER
                if (useSimd && length >= 8)
                {
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
                        var pScaled = Avx.Multiply(Avx.LoadVector256(param + i), vWd);
                        Avx.Store(param + i, Fma.MultiplyAdd(vLr, update, pScaled));
                    }
                }
#endif
                for (; i < length; i++)
                {
                    m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
                    v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
                    float mHat = m[i] / bc1;
                    float vHat = v[i] / bc2;
                    float pScaled = param[i] * wdScale;
                    param[i] = pScaled - lr * mHat / (MathF.Sqrt(vHat) + eps);
                }
            }
        }
    }

    /// <summary>Multi-tensor (foreach-style) Adam over a LIST of float parameter tensors —
    /// constants built once, bit-identical to the per-parameter bc1/bc2 Adam overload.</summary>
    internal static unsafe void AdamUpdateSimdMulti(
        float[][] paramArrays, float[][] gradArrays, float[][] mArrays, float[][] vArrays,
        int[] lengths, int count,
        float lr, float beta1, float beta2, float eps, float bc1, float bc2)
    {
        float lrAdj = lr / bc1;
#if NET5_0_OR_GREATER
        bool useSimd = Fma.IsSupported;
        var vB1 = Vector256.Create(beta1);
        var v1mB1 = Vector256.Create(1f - beta1);
        var vB2 = Vector256.Create(beta2);
        var v1mB2 = Vector256.Create(1f - beta2);
        var vLr = Vector256.Create(-lrAdj);
        var vEps = Vector256.Create(eps);
        var vBc2Inv = Vector256.Create(1f / bc2);
#endif
        for (int t = 0; t < count; t++)
        {
            int length = lengths[t];
            if (length == 0) continue;
            var pa = paramArrays[t]; var ga = gradArrays[t]; var ma = mArrays[t]; var va = vArrays[t];
            if (pa.Length == 0 || ga.Length == 0) continue;
            fixed (float* param = pa, grad = ga, m = ma, v = va)
            {
                int i = 0;
#if NET5_0_OR_GREATER
                if (useSimd && length >= 8)
                {
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
        }
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

    /// <summary>RMSprop with centered variant. Tracks a running mean of gradients g_avg
    /// and uses the variance estimate v − g_avg² in the denominator (Graves, 2013):
    ///   g_avg = ρ·g_avg + (1−ρ)·g
    ///   v     = ρ·v + (1−ρ)·g²
    ///   denom = sqrt(v − g_avg²) + eps
    /// Falls back to plain RMSprop when <paramref name="gradAvg"/> is <c>null</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RMSpropCenteredUpdate(
        float* param, float* grad, float* v, float* gradAvg, int length,
        float lr, float rho, float eps)
    {
        for (int i = 0; i < length; i++)
        {
            v[i] = rho * v[i] + (1f - rho) * grad[i] * grad[i];
            gradAvg[i] = rho * gradAvg[i] + (1f - rho) * grad[i];
            float variance = v[i] - gradAvg[i] * gradAvg[i];
            // Numerical safety: variance can be slightly negative due to FP error.
            if (variance < 0f) variance = 0f;
            param[i] -= lr * grad[i] / (MathF.Sqrt(variance) + eps);
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

    /// <summary>AVX2 AdaMax: Adam variant with infinity norm (max instead of L2).
    /// Caller-supplied <paramref name="eps"/> matches PyTorch <c>torch.optim.Adamax(eps=)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdaMaxUpdateSimd(
        float* param, float* grad, float* m, float* u, int length,
        float lr, float beta1, float beta2, float eps, int step)
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
            var vEps = Vector256.Create(eps);
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
            param[i] -= lr * mHat / (u[i] + eps);
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

    /// <summary>AVX2 AMSGrad (double): Adam with max of past squared gradients.
    /// Mirrors the float <see cref="AMSGradUpdateSimd(float*,float*,float*,float*,float*,int,float,float,float,float,int)"/>
    /// exactly so a double-precision CompiledTrainingPlan keeps the same
    /// non-increasing-denominator guarantee (Reddi, Kale, Kumar 2018) that the
    /// FeedForwardNeuralNetwork default relies on (drift fix, AiDotNet #1332).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AMSGradUpdateSimd(
        double* param, double* grad, double* m, double* v, double* vMax, int length,
        double lr, double beta1, double beta2, double eps, int step)
    {
        double bc1 = 1.0 - System.Math.Pow(beta1, step);
        double bc2 = 1.0 - System.Math.Pow(beta2, step);
        double lrAdj = lr / bc1;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 4)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1.0 - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1.0 - beta2);
            var vLr = Vector256.Create(-lrAdj);
            var vEps = Vector256.Create(eps);
            var vBc2Inv = Vector256.Create(1.0 / bc2);
            int simdLen = length & ~3;
            for (; i < simdLen; i += 4)
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
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            vMax[i] = System.Math.Max(vMax[i], v[i]);
            double mHat = m[i] / bc1;
            double vMaxHat = vMax[i] / bc2;
            param[i] -= lr * mHat / (System.Math.Sqrt(vMaxHat) + eps);
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

    /// <summary>AVX2 AdaDelta. Matches torch.optim.Adadelta:
    ///   v_t = ρ·v_{t-1} + (1-ρ)·g²
    ///   Δx  = √(u_{t-1}+eps) / √(v_t+eps) · g          (scale-free magnitude)
    ///   u_t = ρ·u_{t-1} + (1-ρ)·Δx²                     (accumulator tracks unscaled Δx)
    ///   p   ← p - lr · Δx
    /// The lr scaling is applied only on the final parameter write so that the
    /// running-RMS-of-updates accumulator remains scale-invariant. This makes
    /// param-group LR overrides and LR schedulers actually take effect.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AdaDeltaUpdateSimd(
        float* param, float* grad, float* accumGrad, float* accumUpdate, int length,
        float lr, float rho, float eps)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vRho = Vector256.Create(rho);
            var v1mRho = Vector256.Create(1f - rho);
            var vEps = Vector256.Create(eps);
            var vLr = Vector256.Create(lr);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var ag = Fma.MultiplyAdd(vRho, Avx.LoadVector256(accumGrad + i), Avx.Multiply(v1mRho, Avx.Multiply(g, g)));
                Avx.Store(accumGrad + i, ag);
                var rmsGrad = Avx.Sqrt(Avx.Add(ag, vEps));
                var rmsUpd = Avx.Sqrt(Avx.Add(Avx.LoadVector256(accumUpdate + i), vEps));
                // Δx magnitude (unscaled by lr); negative because we descend.
                var deltaX = Avx.Subtract(Vector256<float>.Zero, Avx.Multiply(Avx.Divide(rmsUpd, rmsGrad), g));
                var au = Fma.MultiplyAdd(vRho, Avx.LoadVector256(accumUpdate + i), Avx.Multiply(v1mRho, Avx.Multiply(deltaX, deltaX)));
                Avx.Store(accumUpdate + i, au);
                // Apply lr scaling only on the parameter write.
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, deltaX, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
        {
            accumGrad[i] = rho * accumGrad[i] + (1f - rho) * grad[i] * grad[i];
            float rmsGrad = MathF.Sqrt(accumGrad[i] + eps);
            float rmsUpdate = MathF.Sqrt(accumUpdate[i] + eps);
            float deltaX = -(rmsUpdate / rmsGrad) * grad[i];
            accumUpdate[i] = rho * accumUpdate[i] + (1f - rho) * deltaX * deltaX;
            param[i] += lr * deltaX;
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

    /// <summary>AVX2 RAdam: Rectified Adam (Liu et al., 2020).
    /// Variance-rectification: when ρ_t > 4, use the corrected adaptive term;
    /// otherwise fall back to plain SGD-like momentum step.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void RAdamUpdateSimd(
        float* param, float* grad, float* m, float* v, int length,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float rhoInf = 2f / (1f - beta2) - 1f;
        float rhoT = rhoInf - 2f * step * MathF.Pow(beta2, step) / bc2;
        bool rectified = rhoT > 4f;
        float rt = rectified
            ? MathF.Sqrt(((rhoT - 4f) * (rhoT - 2f) * rhoInf) /
                        ((rhoInf - 4f) * (rhoInf - 2f) * rhoT))
            : 0f;

        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vB1 = Vector256.Create(beta1);
            var v1mB1 = Vector256.Create(1f - beta1);
            var vB2 = Vector256.Create(beta2);
            var v1mB2 = Vector256.Create(1f - beta2);
            var vBc1Inv = Vector256.Create(1f / bc1);
            int simdLen = length & ~7;

            if (rectified)
            {
                var vBc2Inv = Vector256.Create(1f / bc2);
                var vEps = Vector256.Create(eps);
                var vLr = Vector256.Create(-lr * rt);
                for (; i < simdLen; i += 8)
                {
                    var g = Avx.LoadVector256(grad + i);
                    var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                    Avx.Store(m + i, mNew);
                    var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                    Avx.Store(v + i, vNew);
                    var mHat = Avx.Multiply(mNew, vBc1Inv);
                    var vHat = Avx.Multiply(vNew, vBc2Inv);
                    var denom = Avx.Add(Avx.Sqrt(vHat), vEps);
                    Avx.Store(param + i, Fma.MultiplyAdd(vLr, Avx.Divide(mHat, denom), Avx.LoadVector256(param + i)));
                }
            }
            else
            {
                var vLr = Vector256.Create(-lr);
                for (; i < simdLen; i += 8)
                {
                    var g = Avx.LoadVector256(grad + i);
                    var mNew = Fma.MultiplyAdd(vB1, Avx.LoadVector256(m + i), Avx.Multiply(v1mB1, g));
                    Avx.Store(m + i, mNew);
                    var vNew = Fma.MultiplyAdd(vB2, Avx.LoadVector256(v + i), Avx.Multiply(v1mB2, Avx.Multiply(g, g)));
                    Avx.Store(v + i, vNew);
                    var mHat = Avx.Multiply(mNew, vBc1Inv);
                    Avx.Store(param + i, Fma.MultiplyAdd(vLr, mHat, Avx.LoadVector256(param + i)));
                }
            }
        }
#endif
        for (; i < length; i++)
        {
            m[i] = beta1 * m[i] + (1f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1;
            if (rectified)
            {
                float vHat = v[i] / bc2;
                param[i] -= lr * rt * mHat / (MathF.Sqrt(vHat) + eps);
            }
            else
            {
                param[i] -= lr * mHat;
            }
        }
    }

    /// <summary>SparseAdam: Adam restricted to indices with non-zero gradients.
    /// Caller supplies <paramref name="indices"/> (compact view) of length <paramref name="nnz"/>;
    /// <paramref name="values"/> are the corresponding gradient values. Only those
    /// (param, m, v) entries are updated; bias correction uses <paramref name="step"/>.</summary>
    internal static unsafe void SparseAdamUpdate(
        float* param, int* indices, float* values, float* m, float* v, int nnz,
        float lr, float beta1, float beta2, float eps, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float lrAdj = lr / bc1;
        float bc2Inv = 1f / bc2;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float vHat = vNew * bc2Inv;
            param[idx] -= lrAdj * mNew / (MathF.Sqrt(vHat) + eps);
        }
    }

    // ----------------------------------------------------------------------
    // SPARSE SCATTER-UPDATE KERNELS  (paired with OptimizerBase.SetSparseGradient)
    //
    // Each kernel below is the sparse counterpart of an existing dense optimizer
    // kernel: instead of iterating over the full parameter buffer, only the
    // (index, value) pairs supplied by the autodiff side are visited. The dense
    // state buffers (m, v, accum, ...) keep their full size — only the touched
    // positions are read & written, so the rest of the buffers remain valid
    // across the call. Math matches the corresponding dense fused kernel
    // verbatim (no SIMD — the workload is small by construction, so the
    // managed scalar loop is faster than packing-then-AVX dance).
    // ----------------------------------------------------------------------

    /// <summary>SparseAdamW: AdamW restricted to non-zero indices.
    /// Decoupled weight decay (PyTorch parity) applies at touched positions only.</summary>
    internal static unsafe void SparseAdamWUpdate(
        float* param, int* indices, float* values, float* m, float* v, int nnz,
        float lr, float beta1, float beta2, float eps, float wd, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float lrAdj = lr / bc1;
        float bc2Inv = 1f / bc2;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float p = param[idx];
            if (wd != 0f) p -= lr * wd * p;                       // decoupled weight decay
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float vHat = vNew * bc2Inv;
            param[idx] = p - lrAdj * mNew / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>SparseSGD: SGD + optional momentum + Nesterov + weight decay, restricted to non-zero indices.</summary>
    internal static unsafe void SparseSgdUpdate(
        float* param, int* indices, float* values, float* momentum, int nnz,
        float lr, float mom, float dampening, float wd, bool nesterov, bool hasMomentum)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            if (hasMomentum && mom != 0f)
            {
                float buf = momentum[idx] * mom + g * (1f - dampening);
                momentum[idx] = buf;
                g = nesterov ? g + mom * buf : buf;
            }
            param[idx] -= lr * g;
        }
    }

    /// <summary>SparseNadam: Nesterov-momentum Adam, restricted to non-zero indices.</summary>
    internal static unsafe void SparseNadamUpdate(
        float* param, int* indices, float* values, float* m, float* v, int nnz,
        float lr, float beta1, float beta2, float eps, float wd, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float bc2Inv = 1f / bc2;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float mHat = (beta1 * mNew + (1f - beta1) * g) / bc1;  // Nesterov-corrected
            float vHat = vNew * bc2Inv;
            param[idx] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    /// <summary>SparseAdamax: Adam with L∞ norm (u-state), restricted to non-zero indices.</summary>
    internal static unsafe void SparseAdamaxUpdate(
        float* param, int* indices, float* values, float* m, float* u, int nnz,
        float lr, float beta1, float beta2, float eps, float wd, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float lrAdj = lr / bc1;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float uOld = u[idx];
            float uNew = MathF.Max(beta2 * uOld, MathF.Abs(g));
            m[idx] = mNew;
            u[idx] = uNew;
            param[idx] -= lrAdj * mNew / (uNew + eps);
        }
    }

    /// <summary>SparseAdagrad: per-parameter accumulator update, restricted to non-zero indices.</summary>
    internal static unsafe void SparseAdagradUpdate(
        float* param, int* indices, float* values, float* accum, int nnz,
        float lr, float eps, float wd, float lrDecay, int step)
    {
        float effLr = lr / (1f + step * lrDecay);
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float a = accum[idx] + g * g;
            accum[idx] = a;
            param[idx] -= effLr * g / (MathF.Sqrt(a) + eps);
        }
    }

    /// <summary>SparseRmsProp: RMSProp restricted to non-zero indices.</summary>
    internal static unsafe void SparseRmsPropUpdate(
        float* param, int* indices, float* values, float* sqAvg, float* momentumBuf, int nnz,
        float lr, float alpha, float eps, float wd, float momentum, bool centered, float* gradAvg,
        bool hasMomentum)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float sq = alpha * sqAvg[idx] + (1f - alpha) * g * g;
            sqAvg[idx] = sq;
            float denom;
            if (centered)
            {
                float ga = alpha * gradAvg[idx] + (1f - alpha) * g;
                gradAvg[idx] = ga;
                denom = MathF.Sqrt(sq - ga * ga) + eps;
            }
            else
            {
                denom = MathF.Sqrt(sq) + eps;
            }
            if (hasMomentum && momentum != 0f)
            {
                float buf = momentumBuf[idx] * momentum + g / denom;
                momentumBuf[idx] = buf;
                param[idx] -= lr * buf;
            }
            else
            {
                param[idx] -= lr * g / denom;
            }
        }
    }

    /// <summary>SparseAdaDelta: AdaDelta restricted to non-zero indices. Maintains EMA of
    /// squared grad and squared update.</summary>
    internal static unsafe void SparseAdaDeltaUpdate(
        float* param, int* indices, float* values, float* sqAvg, float* accDelta, int nnz,
        float lr, float rho, float eps, float wd)
    {
        float oneMinusRho = 1f - rho;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float sq = rho * sqAvg[idx] + oneMinusRho * g * g;
            sqAvg[idx] = sq;
            float dx = MathF.Sqrt(accDelta[idx] + eps) / MathF.Sqrt(sq + eps) * g;
            accDelta[idx] = rho * accDelta[idx] + oneMinusRho * dx * dx;
            param[idx] -= lr * dx;
        }
    }

    /// <summary>SparseLion: sign-of-EMA update restricted to non-zero indices.</summary>
    internal static unsafe void SparseLionUpdate(
        float* param, int* indices, float* values, float* m, int nnz,
        float lr, float beta1, float beta2, float wd)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float c = beta1 * m[idx] + (1f - beta1) * g;
            float update = MathF.Sign(c);
            if (wd != 0f)
                param[idx] -= lr * (update + wd * param[idx]);
            else
                param[idx] -= lr * update;
            m[idx] = beta2 * m[idx] + (1f - beta2) * g;
        }
    }

    /// <summary>SparseFtrl: FTRL-proximal restricted to non-zero indices.</summary>
    internal static unsafe void SparseFtrlUpdate(
        float* param, int* indices, float* values, float* accumulator, float* linear, int nnz,
        float lr, float lrPower, float lambda1, float lambda2)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float prevAccum = accumulator[idx];
            float newAccum = prevAccum + g * g;
            float sigma = (MathF.Pow(newAccum, -lrPower) - MathF.Pow(prevAccum, -lrPower)) / lr;
            if (float.IsNaN(sigma) || float.IsInfinity(sigma)) sigma = 0f;
            linear[idx] += g - sigma * param[idx];
            accumulator[idx] = newAccum;
            float z = linear[idx];
            if (MathF.Abs(z) <= lambda1) { param[idx] = 0f; continue; }
            float pre = MathF.Pow(newAccum, -lrPower) / lr + lambda2;
            param[idx] = (MathF.Sign(z) * lambda1 - z) / pre;
        }
    }

    /// <summary>SparseRAdam: Rectified Adam restricted to non-zero indices.
    /// When variance rectification is inactive (early-step), falls back to plain momentum.</summary>
    internal static unsafe void SparseRAdamUpdate(
        float* param, int* indices, float* values, float* m, float* v, int nnz,
        float lr, float beta1, float beta2, float eps, float wd, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        float rhoInf = 2f / (1f - beta2) - 1f;
        float rhoT = rhoInf - 2f * step * MathF.Pow(beta2, step) / bc2;
        bool rect = rhoT > 4f;
        float r = 0f;
        if (rect)
        {
            float num = (rhoT - 4f) * (rhoT - 2f) * rhoInf;
            float den = (rhoInf - 4f) * (rhoInf - 2f) * rhoT;
            r = MathF.Sqrt(num / den);
        }
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float mHat = mNew / bc1;
            if (rect)
            {
                float vHat = MathF.Sqrt(vNew / bc2);
                param[idx] -= lr * r * mHat / (vHat + eps);
            }
            else
            {
                param[idx] -= lr * mHat;
            }
        }
    }

    // ==========================================================================
    // Sparse trust-ratio & exotic-state kernels (LAMB / LARS / BF16Adam / Rprop /
    // FP8Lion / ASGD-lazy / D-Adapt / Prodigy / Shampoo). The trust-ratio family
    // needs a full-param ‖p‖ reduction once per step but does only sparse
    // scatter-updates on (m, v, velocity) and the param at touched indices.
    // Sparse-history semantics on Rprop preserve per-index step-size adaptation
    // across the gaps when an index is rarely updated.
    // ==========================================================================

    /// <summary>SparseLAMB: trust-ratio scaled Adam restricted to non-zero indices.
    /// ‖p‖₂ is computed over the FULL param vector (single O(N) reduction); the
    /// Adam moments and parameter step are scatter-updated at touched indices only.
    /// Since untouched indices have zero update, ‖update‖₂ across all i equals
    /// ‖update_sparse‖₂ across touched i — both yield the same trust ratio.</summary>
    internal static unsafe void SparseLAMBUpdate(
        float* param, int* indices, float* values, float* m, float* v, int paramLen, int nnz,
        float lr, float beta1, float beta2, float eps, float wd, int step)
    {
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);

        // Full ‖p‖₂ reduction — one O(paramLen) read.
        float pNormSq = 0f;
        for (int i = 0; i < paramLen; i++) pNormSq += param[i] * param[i];
        float pNorm = MathF.Sqrt(pNormSq);

        // Sparse Adam-like update + collect ‖update‖₂² over touched i.
        // Stage updates in a temp buffer so we can apply the trust-ratio scale uniformly.
        var updates = new float[nnz];
        float uNormSq = 0f;
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float mNew = beta1 * m[idx] + (1f - beta1) * g;
            float vNew = beta2 * v[idx] + (1f - beta2) * g * g;
            m[idx] = mNew;
            v[idx] = vNew;
            float mHat = mNew / bc1;
            float vHat = vNew / bc2;
            float u = mHat / (MathF.Sqrt(vHat) + eps);
            if (wd != 0f) u += wd * param[idx];
            updates[k] = u;
            uNormSq += u * u;
        }
        float uNorm = MathF.Sqrt(uNormSq);

        // LAMB trust ratio: 1.0 if either norm is zero (no scaling), else ‖p‖ / ‖u‖.
        float trustRatio = (pNorm > 0f && uNorm > 0f) ? (pNorm / uNorm) : 1f;
        float effLr = lr * trustRatio;
        for (int k = 0; k < nnz; k++)
        {
            param[indices[k]] -= effLr * updates[k];
        }
    }

    /// <summary>SparseLARS: trust-ratio scaled SGD-momentum restricted to non-zero
    /// indices. ‖p‖₂ via full reduction; ‖g_sparse‖₂² = sum over touched values²
    /// (equals dense ‖g‖₂² since untouched indices have zero gradient).</summary>
    internal static unsafe void SparseLARSUpdate(
        float* param, int* indices, float* values, float* velocity, int paramLen, int nnz,
        float lr, float momentum, float wd, float trustCoeff)
    {
        // Full ‖p‖₂ reduction.
        float pNormSq = 0f;
        for (int i = 0; i < paramLen; i++) pNormSq += param[i] * param[i];
        float pNorm = MathF.Sqrt(pNormSq);

        // ‖g‖₂² over touched (= dense, since untouched g = 0).
        float gNormSq = 0f;
        for (int k = 0; k < nnz; k++) gNormSq += values[k] * values[k];
        float gNorm = MathF.Sqrt(gNormSq);

        float denom = gNorm + wd * pNorm;
        float localLr = (pNorm > 0f && denom > 0f) ? (lr * trustCoeff * pNorm / denom) : lr;

        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float v = momentum * velocity[idx] + localLr * g;
            velocity[idx] = v;
            param[idx] -= v;
        }
    }

    /// <summary>SparseRprop: per-element step-size adaptation at touched indices
    /// only. Untouched indices keep their (prev_grad, step_size) state — this is
    /// "sparse-history" semantics, preserving the per-index adaptation across
    /// gaps when an embedding row is rarely touched. Differs from dense Rprop
    /// (which writes prev_grad[i] = grad[i] = 0 at every untouched i), but is
    /// the right behavior for sparse use cases.</summary>
    internal static unsafe void SparseRpropUpdate(
        float* param, int* indices, float* values, float* prevGrad, float* stepSize, int nnz,
        float etaPlus, float etaMinus, float stepMin, float stepMax)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            float prev = prevGrad[idx];
            float ss = stepSize[idx];
            float signProduct = prev * g;
            if (signProduct > 0f)
            {
                ss = MathF.Min(ss * etaPlus, stepMax);
            }
            else if (signProduct < 0f)
            {
                ss = MathF.Max(ss * etaMinus, stepMin);
                g = 0f;  // reset gradient memory on sign change (paper convention)
            }
            stepSize[idx] = ss;
            prevGrad[idx] = g;
            if (g > 0f)        param[idx] -= ss;
            else if (g < 0f)   param[idx] += ss;
        }
    }

    /// <summary>SparseASGD: lazy averaging — only touched indices update both p and ax.
    /// Untouched ax[i] drift via lazy catch-up: caller maintains last-touch step per
    /// index and applies missed (1 - mu) decays when next touched. This kernel does
    /// the per-touched-index update assuming caller has already applied lazy catch-up
    /// to ax[idx] for the missed steps.</summary>
    internal static unsafe void SparseASGDUpdate(
        float* param, int* indices, float* values, float* ax, int nnz,
        float eta, float lambd, float wd, float mu)
    {
        for (int k = 0; k < nnz; k++)
        {
            int idx = indices[k];
            float g = values[k];
            if (wd != 0f) g += wd * param[idx];
            float pNew = param[idx] * (1f - eta * lambd) - eta * g;
            param[idx] = pNew;
            ax[idx] += mu * (pNew - ax[idx]);
        }
    }

    /// <summary>SparseBF16Adam helper: dequantize one BF16-packed slot.</summary>
    internal static float SparseBF16Unpack(float[] packed, int i)
    {
        int cell = i >> 1;
        uint cellBits = (uint)BitConverter.ToInt32(BitConverter.GetBytes(packed[cell]), 0);
        ushort bf = (i & 1) == 0 ? (ushort)(cellBits & 0xFFFFu) : (ushort)(cellBits >> 16);
        var bytes = BitConverter.GetBytes((uint)bf << 16);
        return BitConverter.ToSingle(bytes, 0);
    }

    /// <summary>SparseBF16Adam helper: quantize one FP32 value back to BF16-packed slot.</summary>
    internal static void SparseBF16Pack(float[] packed, int i, float v)
    {
        uint bits = (uint)BitConverter.ToInt32(BitConverter.GetBytes(v), 0);
        uint rounding = ((bits >> 16) & 1u) + 0x7FFFu;
        ushort bf = (ushort)((bits + rounding) >> 16);
        int cell = i >> 1;
        uint cellBits = (uint)BitConverter.ToInt32(BitConverter.GetBytes(packed[cell]), 0);
        if ((i & 1) == 0) cellBits = (cellBits & 0xFFFF0000u) | bf;
        else              cellBits = (cellBits & 0x0000FFFFu) | ((uint)bf << 16);
        packed[cell] = BitConverter.ToSingle(BitConverter.GetBytes(cellBits), 0);
    }

    /// <summary>AVX2 ASGD: Averaged SGD (Polyak/Ruppert).
    /// Step decay <c>η_t = lr / (1 + λ·lr·t)^α</c>, weight decay applied to gradient,
    /// and exponential moving average of params written to <paramref name="ax"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ASGDUpdateSimd(
        float* param, float* grad, float* ax, int length,
        float lr, float lambd, float alpha, float weightDecay, float mu)
    {
        // Effective learning rate uses the per-call lr (decayed externally).
        // mu controls how fast the running average ax is pulled toward param.
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-lr);
            var vWd = Vector256.Create(weightDecay);
            var vLambd = Vector256.Create(lr * lambd);
            var vMu = Vector256.Create(mu);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var p = Avx.LoadVector256(param + i);
                var g = Avx.LoadVector256(grad + i);
                if (weightDecay != 0f)
                    g = Fma.MultiplyAdd(vWd, p, g);
                // p ← p · (1 − lr·λ) − lr · g
                var decayed = Fma.MultiplyAddNegated(vLambd, p, p);
                var pNew = Fma.MultiplyAdd(vLr, g, decayed);
                Avx.Store(param + i, pNew);
                var axOld = Avx.LoadVector256(ax + i);
                // ax += mu * (p − ax)
                var axNew = Fma.MultiplyAdd(vMu, Avx.Subtract(pNew, axOld), axOld);
                Avx.Store(ax + i, axNew);
            }
        }
#endif
        for (; i < length; i++)
        {
            float g = grad[i] + weightDecay * param[i];
            param[i] = param[i] * (1f - lr * lambd) - lr * g;
            ax[i] += mu * (param[i] - ax[i]);
        }
        _ = alpha; // alpha is consumed by external schedule when lr is computed
    }

    /// <summary>Rprop: Resilient backpropagation.
    /// Per-element step sizes adapt based on sign-changes of consecutive gradients.
    /// Reference: Riedmiller &amp; Braun, 1993.</summary>
    internal static unsafe void RpropUpdate(
        float* param, float* grad, float* prevGrad, float* stepSize, int length,
        float etaPlus, float etaMinus, float stepMin, float stepMax)
    {
        for (int i = 0; i < length; i++)
        {
            float gPrev = prevGrad[i];
            float g = grad[i];
            float sgnChange = gPrev * g;
            float step = stepSize[i];
            if (sgnChange > 0f)
            {
                step = MathF.Min(step * etaPlus, stepMax);
                stepSize[i] = step;
                param[i] -= MathF.Sign(g) * step;
                prevGrad[i] = g;
            }
            else if (sgnChange < 0f)
            {
                step = MathF.Max(step * etaMinus, stepMin);
                stepSize[i] = step;
                // Skip this update; reset gradient memory so next step is treated as a fresh sign.
                prevGrad[i] = 0f;
            }
            else
            {
                param[i] -= MathF.Sign(g) * step;
                prevGrad[i] = g;
            }
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

    // ────────────────────────────────────────────────────────────────────
    // Adaptive learning-rate kernels (Issue #348)
    //
    // These three kernels self-tune the learning rate inside the fused
    // pass — no scheduler call needed, no managed-side hyperparameter
    // search. PyTorch has no fused equivalent for any of them:
    //   - Hypergradient: not built in; community implementations are
    //     all eager Python (per-element loops in Python).
    //   - Schedule-Free (Defazio 2024): the `schedulefree` package
    //     ships an eager-Python implementation; CUDA fused is not
    //     released as of 2024.
    //   - D-Adaptation: the `dadaptation` package is eager Python +
    //     external reductions.
    //
    // Because these write learning-rate / distance state through
    // `ref` (or via a single-element scratch buffer), the caller owns
    // the state lifetime — same pattern as Adam's `m`/`v` buffers but
    // a single scalar.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Hypergradient SGD (Baydin et al., 2018).
    /// Tunes <paramref name="lr"/> online via the inner product of the
    /// current and previous gradients:
    ///   lr_t = lr_{t-1} + hyperLr · ⟨g_t, g_{t-1}⟩
    ///   p   -= lr_t · g_t
    /// The +sign on the inner product is correct: ∂L/∂lr = -⟨g_t, g_{t-1}⟩,
    /// so descent on lr adds the inner product to lr. Returns the new lr
    /// so the caller can persist it across steps.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float HypergradientSgdUpdateSimd(
        float* param, float* grad, float* prevGrad, int length,
        float lr, float hyperLr)
    {
        // Pass 1: inner product g·g_prev. AVX2 FMA reduction.
        float innerProduct = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var acc = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var gp = Avx.LoadVector256(prevGrad + i);
                acc = Fma.MultiplyAdd(g, gp, acc);
            }
            innerProduct = SimdKernels.HorizontalSum(acc);
        }
#endif
        for (; i < length; i++)
            innerProduct += grad[i] * prevGrad[i];

        float newLr = lr + hyperLr * innerProduct;

        // Pass 2: param -= newLr * grad ; prevGrad = grad
        i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-newLr);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                Avx.Store(param + i, Fma.MultiplyAdd(vLr, g, Avx.LoadVector256(param + i)));
                Avx.Store(prevGrad + i, g);
            }
        }
#endif
        for (; i < length; i++)
        {
            param[i] -= newLr * grad[i];
            prevGrad[i] = grad[i];
        }
        return newLr;
    }

    /// <summary>Schedule-Free SGD (Defazio et al., 2024 — "The Road Less Scheduled").
    /// Maintains two parameter copies: <paramref name="z"/> is the "primary"
    /// (SGD-on-z trajectory) and <paramref name="x"/> is the
    /// weighted-average evaluation copy returned for inference.
    ///   y      = (1-β) z + β x         — done by ScheduleFreeYUpdateSimd, before the forward
    ///   z      -= lr · grad             — standard SGD on z
    ///   w_t    = lr²                    — weight for this step (p=2, q=0 default)
    ///   c_t    = w_t / (weightSum + w_t)
    ///   x      = (1-c_t) x + c_t z      — running weighted average
    /// Caller persists <paramref name="weightSum"/> across steps. Returns
    /// the new weightSum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float ScheduleFreeSgdUpdateSimd(
        float* z, float* x, float* grad, int length,
        float lr, float weightSum)
    {
        // weight_t = lr^2 — Defazio default (p=2, q=0). Empirically the
        // most stable choice from the paper; we expose it as the default
        // and let users tune via overloads if they need other (p, q).
        float w_t = lr * lr;
        float c_t = w_t / (weightSum + w_t);
        float oneMinusC = 1f - c_t;

        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vLr = Vector256.Create(-lr);
            var vC = Vector256.Create(c_t);
            var vOneMinusC = Vector256.Create(oneMinusC);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var zNew = Fma.MultiplyAdd(vLr, g, Avx.LoadVector256(z + i));
                Avx.Store(z + i, zNew);
                var xNew = Fma.MultiplyAdd(vOneMinusC, Avx.LoadVector256(x + i), Avx.Multiply(vC, zNew));
                Avx.Store(x + i, xNew);
            }
        }
#endif
        for (; i < length; i++)
        {
            z[i] -= lr * grad[i];
            x[i] = oneMinusC * x[i] + c_t * z[i];
        }
        return weightSum + w_t;
    }

    /// <summary>Schedule-Free y-update — caller must run this BEFORE the
    /// forward pass each step. Writes <c>y[i] = (1-β) z[i] + β x[i]</c>
    /// into the supplied <paramref name="y"/> buffer; the forward then
    /// reads from <paramref name="y"/>. PyTorch's eager implementation
    /// does this per-element in Python, which dominates wall time on
    /// small models — fusing it removes that overhead entirely.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void ScheduleFreeYUpdateSimd(
        float* y, float* z, float* x, int length, float beta)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vBeta = Vector256.Create(beta);
            var vOneMinusBeta = Vector256.Create(1f - beta);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var zV = Avx.LoadVector256(z + i);
                var xV = Avx.LoadVector256(x + i);
                Avx.Store(y + i, Fma.MultiplyAdd(vBeta, xV, Avx.Multiply(vOneMinusBeta, zV)));
            }
        }
#endif
        for (; i < length; i++)
            y[i] = (1f - beta) * z[i] + beta * x[i];
    }

    /// <summary>D-Adaptation SGD (Defazio &amp; Mishchenko, 2023 — "Learning-Rate-Free
    /// Learning by D-Adaptation"; growth-bounded variant from Prodigy,
    /// Mishchenko &amp; Defazio 2024). Provably converges to the optimal SGD
    /// lr without any tuning. State per parameter group:
    ///   d_k        — current distance estimate (scalar; caller persists)
    ///   s_buf      — running weighted-grad accumulator (same shape as params)
    ///   r_scalar   — accumulator for d update (scalar; caller persists)
    /// Update rule (per step):
    ///   γ_k    = d_k · lr                   — effective step (lr ≈ 1.0 typical)
    ///   s_buf += γ_k · g
    ///   r     += γ_k² · ⟨g, g⟩
    ///   d_hat  = ||s_buf||² / (√r + 1e-30)
    ///   d_{k+1}= max(d_k, min(d_hat, d_k · growthRate))
    ///   p     -= γ_{k+1} · g
    /// <paramref name="growthRate"/> caps per-step growth so a too-small
    /// initial d_0 can't blow up the trajectory on steep problems —
    /// matches the Prodigy "growth_rate" hyperparameter.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float DAdaptationSgdUpdateSimd(
        float* param, float* grad, float* sBuf, int length,
        float d, ref float rAccum, float lr, float growthRate)
    {
        // γ_k = d * lr. Paper's "step at iteration k" — separate from the
        // updated γ_{k+1} we'll use to write parameters at the END of the
        // step, which uses the new d.
        float gammaCur = d * lr;

        // Pass 1: update s_buf with current γ, accumulate ||g||² and ||s||².
        float gNorm2 = 0f;
        float sNorm2 = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vGamma = Vector256.Create(gammaCur);
            var gAcc = Vector256<float>.Zero;
            var sAcc = Vector256<float>.Zero;
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                var s = Fma.MultiplyAdd(vGamma, g, Avx.LoadVector256(sBuf + i));
                Avx.Store(sBuf + i, s);
                gAcc = Fma.MultiplyAdd(g, g, gAcc);
                sAcc = Fma.MultiplyAdd(s, s, sAcc);
            }
            gNorm2 = SimdKernels.HorizontalSum(gAcc);
            sNorm2 = SimdKernels.HorizontalSum(sAcc);
        }
#endif
        for (; i < length; i++)
        {
            sBuf[i] += gammaCur * grad[i];
            gNorm2 += grad[i] * grad[i];
            sNorm2 += sBuf[i] * sBuf[i];
        }

        rAccum += gammaCur * gammaCur * gNorm2;
        // 1e-30 (not eps) — numerical floor only; we don't want to dampen
        // the distance signal.
        float dHat = sNorm2 / (MathF.Sqrt(rAccum) + 1e-30f);
        // Cap per-step growth (Prodigy growth_rate). Without this bound,
        // an initial d_0 ≪ d* causes d to leap orders of magnitude per
        // step and blow up before s/r catch up.
        float dCapped = MathF.Min(dHat, d * growthRate);
        float dNew = MathF.Max(d, dCapped);

        // Pass 2: param -= d_new · lr · g.
        float gammaNew = dNew * lr;
        i = 0;
#if NET5_0_OR_GREATER
        if (Fma.IsSupported && length >= 8)
        {
            var vStep = Vector256.Create(-gammaNew);
            int simdLen = length & ~7;
            for (; i < simdLen; i += 8)
            {
                var g = Avx.LoadVector256(grad + i);
                Avx.Store(param + i, Fma.MultiplyAdd(vStep, g, Avx.LoadVector256(param + i)));
            }
        }
#endif
        for (; i < length; i++)
            param[i] -= gammaNew * grad[i];

        return dNew;
    }
}

/// <summary>
/// Extra per-optimizer hyperparameters that don't fit the generic
/// (learningRate, beta1, beta2, eps, weightDecay) slots of
/// <c>CompiledTrainingPlan.ConfigureOptimizer</c>. Pass an instance to configure
/// AdaDelta / LARS / FTRL / ASGD / Rprop; omit it to use the per-field defaults.
/// All other optimizers ignore it.
/// </summary>
/// <remarks>
/// A class with property initializers (not a record struct) so <c>new
/// FusedOptimizerExtras()</c> reliably yields the documented defaults — a
/// <c>default</c> record-struct would zero every field, which is wrong for e.g.
/// LARS <c>TrustCoefficient</c> or Rprop step bounds.
/// </remarks>
public sealed class FusedOptimizerExtras
{
    /// <summary>LARS momentum coefficient. Default 0.9.</summary>
    public float Momentum { get; init; } = 0.9f;
    /// <summary>LARS trust coefficient (η in the layer-wise trust ratio). Default 0.001.</summary>
    public float TrustCoefficient { get; init; } = 0.001f;
    /// <summary>FTRL L1 regularization strength. Default 0.</summary>
    public float L1 { get; init; } = 0f;
    /// <summary>FTRL L2 regularization strength. Default 0.</summary>
    public float L2 { get; init; } = 0f;
    /// <summary>FTRL learning-rate power (β in n^β). Default -0.5.</summary>
    public float LrPower { get; init; } = -0.5f;
    /// <summary>ASGD decay term λ in η_t = lr/(1+λ·lr·t)^α. Default 1e-4.</summary>
    public float Lambd { get; init; } = 1e-4f;
    /// <summary>ASGD decay exponent α. Default 0.75.</summary>
    public float Alpha { get; init; } = 0.75f;
    /// <summary>ASGD averaging start step t0 (μ_t = 1/max(1, t−t0)). Default 1e6.</summary>
    public float T0 { get; init; } = 1e6f;
    /// <summary>Rprop step-increase factor η⁺. Default 1.2.</summary>
    public float RpropEtaPlus { get; init; } = 1.2f;
    /// <summary>Rprop step-decrease factor η⁻. Default 0.5.</summary>
    public float RpropEtaMinus { get; init; } = 0.5f;
    /// <summary>Rprop minimum step size. Default 1e-6.</summary>
    public float RpropStepMin { get; init; } = 1e-6f;
    /// <summary>Rprop maximum step size. Default 50.</summary>
    public float RpropStepMax { get; init; } = 50f;
    /// <summary>Rprop initial per-element step size. Default 0.01.</summary>
    public float RpropInitialStep { get; init; } = 0.01f;
    /// <summary>Hypergradient-SGD learning rate of the learning rate (β). Default 1e-7.</summary>
    public float HyperLr { get; init; } = 1e-7f;
    /// <summary>D-Adaptation initial distance estimate d0. Default 1e-6.</summary>
    public float D0 { get; init; } = 1e-6f;
    /// <summary>D-Adaptation per-step growth cap (Prodigy growth_rate). Default +inf (uncapped).</summary>
    public float DGrowthRate { get; init; } = float.PositiveInfinity;
    /// <summary>Schedule-Free SGD interpolation factor β for the y-update
    /// <c>y = (1-β)z + βx</c> (Defazio et al., 2024). Default 0.9 — the
    /// paper's momentum-equivalent default.</summary>
    public float SfBeta { get; init; } = 0.9f;

    /// <summary>
    /// Validates the hyperparameters that would otherwise produce undefined or
    /// divergent fused-optimizer math, throwing <see cref="ArgumentOutOfRangeException"/>
    /// at configuration time rather than silently corrupting a training run.
    /// Called by <c>CompiledTrainingPlan.ConfigureOptimizer*</c> before the
    /// per-step closures capture these values.
    /// </summary>
    public void Validate()
    {
        if (!(HyperLr >= 0f) || float.IsNaN(HyperLr))
            throw new ArgumentOutOfRangeException(nameof(HyperLr), HyperLr,
                "Hypergradient-SGD HyperLr (β) must be >= 0.");
        if (!(D0 > 0f) || float.IsNaN(D0))
            throw new ArgumentOutOfRangeException(nameof(D0), D0,
                "D-Adaptation initial distance estimate D0 must be > 0.");
        if (!(DGrowthRate >= 1f))   // +inf is allowed (uncapped); only < 1 (incl. NaN) is rejected
            throw new ArgumentOutOfRangeException(nameof(DGrowthRate), DGrowthRate,
                "D-Adaptation DGrowthRate must be >= 1 (per-step growth cap; +inf = uncapped).");
        if (!(SfBeta >= 0f && SfBeta <= 1f))
            throw new ArgumentOutOfRangeException(nameof(SfBeta), SfBeta,
                "Schedule-Free SfBeta must be in [0, 1].");
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
    FTRL = 13,
    /// <summary>RAdam: Rectified Adam (Liu et al., 2020)</summary>
    RAdam = 14,
    /// <summary>SparseAdam: Adam variant for sparse gradients</summary>
    SparseAdam = 15,
    /// <summary>ASGD: Averaged Stochastic Gradient Descent (Polyak/Ruppert)</summary>
    ASGD = 16,
    /// <summary>Rprop: Resilient back-propagation (Riedmiller, 1993)</summary>
    Rprop = 17,
    /// <summary>LBFGS: Limited-memory BFGS (closure-based, see <c>LBFGSOptimizer</c>)</summary>
    LBFGS = 18,
    /// <summary>Hypergradient SGD: online lr tuning via gradient inner product (Baydin et al., 2018)</summary>
    HypergradientSGD = 19,
    /// <summary>Schedule-Free SGD: scheduler-free, weighted-average evaluation (Defazio et al., 2024)</summary>
    ScheduleFreeSGD = 20,
    /// <summary>D-Adaptation SGD: learning-rate-free SGD (Defazio &amp; Mishchenko, 2023)</summary>
    DAdaptationSGD = 21
}
