using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Fused backward for multi-layer GEMM: computes all gradients for
/// output = W2 @ activation(W1 @ input + b1) + b2 in a single pass
/// using transposed BLAS GEMM (no transpose allocation).
///
/// Eliminates 4 intermediate tensor allocations that the unfused backward requires:
/// - No W1^T allocation (TryGemmEx with transA=true)
/// - No W2^T allocation (TryGemmEx with transB=true)
/// - No separate ReLU mask allocation (in-place on grad_pre)
/// - Activated intermediate reused from forward savedState
/// </summary>
internal static class FusedMultiLayerBackward
{
    /// <summary>
    /// Computes all gradients for a 2-layer fused forward:
    ///   h = activation(input @ W1 + b1)
    ///   output = h @ W2 + b2
    ///
    /// Given gradOutput (dL/doutput), computes:
    ///   gradW2 = activated^T @ gradOutput         (transposed BLAS)
    ///   gradB2 = sum(gradOutput, axis=0)           (inline reduction)
    ///   grad_h = gradOutput @ W2^T                 (transposed BLAS)
    ///   grad_pre = grad_h * activation'(activated) (SIMD element-wise)
    ///   gradW1 = input^T @ grad_pre               (transposed BLAS)
    ///   gradB1 = sum(grad_pre, axis=0)             (inline reduction)
    ///   gradInput = grad_pre @ W1^T               (transposed BLAS, only if needed)
    /// </summary>
    [MethodImpl(Hot)]
    internal static unsafe void ComputeGradients(
        float[] gradOutput,    // [M, N]
        float[] input,         // [M, K]
        float[] w1,            // [K, H]
        float[] w2,            // [H, N]
        float[] activated,     // [M, H] — saved from forward
        float[] gradW1,        // [K, H] output
        float[] gradW2,        // [H, N] output
        float[] gradB1,        // [H] output (nullable via length check)
        float[] gradB2,        // [N] output (nullable via length check)
        float[] gradInput,     // [M, K] output (nullable via length check)
        int m, int k, int h, int n,
        Func<float, float> activationDerivative,
        float[]? workspace = null) // [M, H] pre-allocated buffer for grad_h to avoid allocation
    {
        // Step 1: gradW2 = activated^T @ gradOutput  [H,M] @ [M,N] = [H,N]
        // Uses TryGemmEx(transA=true) — no transpose allocation
        if (!BlasProvider.TryGemmEx(h, n, m,
                activated, 0, h, true,
                gradOutput, 0, n, false,
                gradW2, 0, n))
        {
            // Fallback: manual transposed matmul
            SimdGemm.Sgemm(
                activated.AsSpan(), h, true,
                gradOutput.AsSpan(), n, false,
                gradW2.AsSpan(), h, m, n);
        }

        // Step 2: gradB2 = sum(gradOutput, axis=0)  → [N]
        if (gradB2.Length >= n)
        {
            Array.Clear(gradB2, 0, n);
            fixed (float* pG = gradOutput, pB = gradB2)
            {
                for (int row = 0; row < m; row++)
                {
                    float* gRow = pG + row * n;
                    for (int j = 0; j < n; j++)
                        pB[j] += gRow[j];
                }
            }
        }

        // Step 3: grad_h = gradOutput @ W2^T  [M,N] @ [N,H] = [M,H]
        // Uses TryGemmEx(transB=true) — no transpose allocation
        int requiredSize = m * h;
        var grad_h = workspace != null && workspace.Length >= requiredSize
            ? workspace
            : new float[requiredSize];
        if (!BlasProvider.TryGemmEx(m, h, n,
                gradOutput, 0, n, false,
                w2, 0, n, true,
                grad_h, 0, h))
        {
            SimdGemm.Sgemm(
                gradOutput.AsSpan(), n, false,
                w2.AsSpan(), n, true,
                grad_h.AsSpan(), m, n, h);
        }

        // Step 4: Apply activation derivative in-place: grad_pre = grad_h * act'(activated)
        // This is a fused SIMD operation — no separate allocation
        fixed (float* pGH = grad_h, pAct = activated)
        {
            for (int i = 0; i < m * h; i++)
                pGH[i] *= activationDerivative(pAct[i]);
        }

        // Step 5: gradW1 = input^T @ grad_pre  [K,M] @ [M,H] = [K,H]
        if (!BlasProvider.TryGemmEx(k, h, m,
                input, 0, k, true,
                grad_h, 0, h, false,
                gradW1, 0, h))
        {
            SimdGemm.Sgemm(
                input.AsSpan(), k, true,
                grad_h.AsSpan(), h, false,
                gradW1.AsSpan(), k, m, h);
        }

        // Step 6: gradB1 = sum(grad_pre, axis=0) → [H]
        if (gradB1.Length >= h)
        {
            Array.Clear(gradB1, 0, h);
            fixed (float* pGP = grad_h, pB = gradB1)
            {
                for (int row = 0; row < m; row++)
                {
                    float* gpRow = pGP + row * h;
                    for (int j = 0; j < h; j++)
                        pB[j] += gpRow[j];
                }
            }
        }

        // Step 7: gradInput = grad_pre @ W1^T  [M,H] @ [H,K] = [M,K]
        if (gradInput.Length >= m * k)
        {
            if (!BlasProvider.TryGemmEx(m, k, h,
                    grad_h, 0, h, false,
                    w1, 0, h, true,
                    gradInput, 0, k))
            {
                SimdGemm.Sgemm(
                    grad_h.AsSpan(), h, false,
                    w1.AsSpan(), h, true,
                    gradInput.AsSpan(), m, h, k);
            }
        }
    }

    /// <summary>ReLU derivative: 1 if x > 0, else 0.</summary>
    internal static float ReLUDerivative(float x) => x > 0f ? 1f : 0f;

    /// <summary>Sigmoid derivative from post-activation value: sig * (1 - sig).</summary>
    internal static float SigmoidDerivative(float sigOutput) => sigOutput * (1f - sigOutput);

    /// <summary>Tanh derivative from post-activation value: 1 - tanh^2.</summary>
    internal static float TanhDerivative(float tanhOutput) => 1f - tanhOutput * tanhOutput;

    /// <summary>
    /// GELU derivative (tanh-approximation, matching SimdKernels.GELUUnsafe).
    /// Input is PRE-activation x (the GEMM1 output before applying GELU).
    /// <para>
    /// GELU(x) = 0.5 * x * (1 + tanh(u)) where u = √(2/π) * (x + 0.044715 * x³).
    /// dGELU/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx
    ///         where du/dx = √(2/π) * (1 + 3 * 0.044715 * x²) and sech²(u) = 1 - tanh²(u).
    /// </para>
    /// </summary>
    internal static float GELUDerivative(float x)
    {
        const float SQRT_2_OVER_PI = 0.7978845608028654f;
        const float COEFF = 0.044715f;
        float xx = x * x;
        float u = SQRT_2_OVER_PI * (x + COEFF * xx * x);
        float tanhU = MathF.Tanh(u);
        float sech2U = 1f - tanhU * tanhU;
        float duDx = SQRT_2_OVER_PI * (1f + 3f * COEFF * xx);
        return 0.5f * (1f + tanhU) + 0.5f * x * sech2U * duDx;
    }

    /// <summary>
    /// Issue #338 Phase G.2: backward overload for activations where the
    /// derivative depends on PRE-activation (e.g. GELU). Forward must have
    /// captured the GEMM1 output into <paramref name="preActivation"/>
    /// before applying the activation. Both buffers are needed:
    /// <c>activated</c> (post) goes into the gradW2 GEMM,
    /// <c>preActivation</c> feeds the per-element derivative.
    /// </summary>
    [MethodImpl(Hot)]
    internal static unsafe void ComputeGradients(
        float[] gradOutput,
        float[] input,
        float[] w1,
        float[] w2,
        float[] activated,
        float[] preActivation,
        float[] gradW1,
        float[] gradW2,
        float[] gradB1,
        float[] gradB2,
        float[] gradInput,
        int m, int k, int h, int n,
        Func<float, float> activationDerivativeFromPreActivation,
        float[]? workspace = null)
    {
        // Step 1: gradW2 = activated^T @ gradOutput  [H,M] @ [M,N] = [H,N]
        if (!BlasProvider.TryGemmEx(h, n, m,
                activated, 0, h, true,
                gradOutput, 0, n, false,
                gradW2, 0, n))
        {
            SimdGemm.Sgemm(
                activated.AsSpan(), h, true,
                gradOutput.AsSpan(), n, false,
                gradW2.AsSpan(), h, m, n);
        }

        // Step 2: gradB2 = sum(gradOutput, axis=0) → [N]
        if (gradB2.Length >= n)
        {
            Array.Clear(gradB2, 0, n);
            fixed (float* pG = gradOutput, pB = gradB2)
            {
                for (int row = 0; row < m; row++)
                {
                    float* gRow = pG + row * n;
                    for (int j = 0; j < n; j++)
                        pB[j] += gRow[j];
                }
            }
        }

        // Step 3: grad_h = gradOutput @ W2^T  [M,N] @ [N,H] = [M,H]
        int requiredSize = m * h;
        var grad_h = workspace != null && workspace.Length >= requiredSize
            ? workspace
            : new float[requiredSize];
        if (!BlasProvider.TryGemmEx(m, h, n,
                gradOutput, 0, n, false,
                w2, 0, n, true,
                grad_h, 0, h))
        {
            SimdGemm.Sgemm(
                gradOutput.AsSpan(), n, false,
                w2.AsSpan(), n, true,
                grad_h.AsSpan(), m, n, h);
        }

        // Step 4: Apply activation derivative against PRE-activation
        // (this is the only line that differs from the ReLU/post-activation
        // overload above).
        fixed (float* pGH = grad_h, pPre = preActivation)
        {
            for (int i = 0; i < m * h; i++)
                pGH[i] *= activationDerivativeFromPreActivation(pPre[i]);
        }

        // Step 5: gradW1 = input^T @ grad_pre  [K,M] @ [M,H] = [K,H]
        if (!BlasProvider.TryGemmEx(k, h, m,
                input, 0, k, true,
                grad_h, 0, h, false,
                gradW1, 0, h))
        {
            SimdGemm.Sgemm(
                input.AsSpan(), k, true,
                grad_h.AsSpan(), h, false,
                gradW1.AsSpan(), k, m, h);
        }

        // Step 6: gradB1 = sum(grad_pre, axis=0) → [H]
        if (gradB1.Length >= h)
        {
            Array.Clear(gradB1, 0, h);
            fixed (float* pGP = grad_h, pB = gradB1)
            {
                for (int row = 0; row < m; row++)
                {
                    float* gpRow = pGP + row * h;
                    for (int j = 0; j < h; j++)
                        pB[j] += gpRow[j];
                }
            }
        }

        // Step 7: gradInput = grad_pre @ W1^T  [M,H] @ [H,K] = [M,K]
        if (gradInput.Length >= m * k)
        {
            if (!BlasProvider.TryGemmEx(m, k, h,
                    grad_h, 0, h, false,
                    w1, 0, h, true,
                    gradInput, 0, k))
            {
                SimdGemm.Sgemm(
                    grad_h.AsSpan(), h, false,
                    w1.AsSpan(), h, true,
                    gradInput.AsSpan(), m, h, k);
            }
        }
    }
}
