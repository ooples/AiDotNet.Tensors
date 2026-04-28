using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Shampoo (Gupta et al., 2018) — full-matrix preconditioned gradient method.
///
/// For a parameter of shape <c>[d1, d2]</c>, maintains two preconditioner accumulators
/// <c>L (d1×d1)</c> and <c>R (d2×d2)</c>:
///   L_t = L_{t-1} + g · gᵀ
///   R_t = R_{t-1} + gᵀ · g
/// and applies the Kronecker-factored preconditioned update
///   p ← p − lr · L^(-1/4) · g · R^(-1/4)
///
/// Inverse-fourth-root is computed via symmetric eigendecomposition (Jacobi sweep,
/// suitable for the small/medium per-layer matrices typical in transformers).
/// To amortise the cost, the inverse roots are recomputed only every
/// <c>precondition_freq</c> steps and cached. For 1-D parameters or matrices wider
/// than <c>max_precondition_dim</c>, the optimizer falls back to a diagonal AdaGrad-
/// style preconditioner (equivalent to Shampoo on a diagonal matrix), which is the
/// standard Block-Diagonal Shampoo recommendation.
/// </summary>
public sealed class ShampooOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2,
        ["momentum"] = 0.9,
        ["weight_decay"] = 0.0,
        ["eps"] = 1e-12,
        ["precondition_freq"] = 10.0,    // recompute L^-1/4, R^-1/4 every N steps
        ["max_precondition_dim"] = 1024.0, // axes wider than this fall back to diagonal
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <summary>
    /// User must declare each parameter's matrix shape <c>[d1, d2]</c> (or 1-D length)
    /// because the wrapper sees flat <see cref="float"/>[] buffers. <see cref="Add2DParameter"/>
    /// records the shape so Shampoo can build correctly-sized preconditioners.
    /// </summary>
    public ParamGroup Add2DParameter(int d1, int d2, float[] parameter, float[] gradient,
                                     IDictionary<string, double>? overrides = null)
    {
        if (d1 * d2 != parameter.Length)
            throw new ArgumentException($"d1 * d2 = {d1 * d2} != parameter.Length = {parameter.Length}.");
        var group = AddParamGroup(overrides);
        group.AddParameter(parameter, gradient);
        _shapes[(ParamGroups.Count - 1, group.Parameters.Count - 1)] = (d1, d2);
        return group;
    }

    private readonly Dictionary<(int gi, int pi), (int d1, int d2)> _shapes = new();

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float momentum = (float)g.GetOption("momentum", 0.9);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            float eps = (float)g.GetOption("eps", 1e-12);
            int preFreq = (int)g.GetOption("precondition_freq", 10.0);
            int maxDim  = (int)g.GetOption("max_precondition_dim", 1024.0);

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi];
                float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];

                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;

                bool has2D = _shapes.TryGetValue((gi, pi), out var shape);
                bool useFull = has2D
                    && shape.d1 <= maxDim && shape.d2 <= maxDim
                    && shape.d1 > 1 && shape.d2 > 1;

                if (useFull)
                {
                    EnsureFullState(slot, shape.d1, shape.d2);
                    UpdateFull(slot, grad, p, shape.d1, shape.d2, step, lr, momentum, preFreq, eps);
                }
                else
                {
                    EnsureDiagonalState(slot, p.Length);
                    UpdateDiagonal(slot, grad, p, lr, momentum, eps);
                }
            }
        }
    }

    private static void EnsureFullState(Dictionary<string, OptimizerStateValue> slot, int d1, int d2)
    {
        if (!slot.ContainsKey("L"))         slot["L"]         = OptimizerStateValue.FromTensor(new float[d1 * d1]);
        if (!slot.ContainsKey("R"))         slot["R"]         = OptimizerStateValue.FromTensor(new float[d2 * d2]);
        if (!slot.ContainsKey("L_inv_root")) slot["L_inv_root"] = OptimizerStateValue.FromTensor(IdentityMatrix(d1));
        if (!slot.ContainsKey("R_inv_root")) slot["R_inv_root"] = OptimizerStateValue.FromTensor(IdentityMatrix(d2));
        if (!slot.ContainsKey("momentum_buffer"))
            slot["momentum_buffer"] = OptimizerStateValue.FromTensor(new float[d1 * d2]);
    }

    private static void EnsureDiagonalState(Dictionary<string, OptimizerStateValue> slot, int length)
    {
        if (!slot.ContainsKey("diag_acc"))
            slot["diag_acc"] = OptimizerStateValue.FromTensor(new float[length]);
        if (!slot.ContainsKey("momentum_buffer"))
            slot["momentum_buffer"] = OptimizerStateValue.FromTensor(new float[length]);
    }

    private static void UpdateDiagonal(
        Dictionary<string, OptimizerStateValue> slot, float[] grad, float[] p,
        float lr, float momentum, float eps)
    {
        var acc = slot["diag_acc"].Tensor!;
        var mb  = slot["momentum_buffer"].Tensor!;
        for (int i = 0; i < p.Length; i++)
        {
            acc[i] += grad[i] * grad[i];
            // Inverse 4th root of a scalar: 1/sqrt(sqrt(acc)) — stable for any acc≥0.
            float invRoot = 1f / MathF.Sqrt(MathF.Sqrt(acc[i]) + eps);
            float pre = grad[i] * invRoot;
            mb[i] = momentum * mb[i] + pre;
            p[i] -= lr * mb[i];
        }
    }

    private static void UpdateFull(
        Dictionary<string, OptimizerStateValue> slot, float[] grad, float[] p,
        int d1, int d2, int step, float lr, float momentum, int preFreq, float eps)
    {
        var L = slot["L"].Tensor!;
        var R = slot["R"].Tensor!;
        var lInv = slot["L_inv_root"].Tensor!;
        var rInv = slot["R_inv_root"].Tensor!;
        var mb   = slot["momentum_buffer"].Tensor!;

        // L += g · gᵀ   (d1×d1)
        // R += gᵀ · g  (d2×d2)
        // Treat grad as a row-major [d1, d2] matrix g[r*d2+c].
        for (int r = 0; r < d1; r++)
        {
            for (int s = 0; s < d1; s++)
            {
                float acc = 0f;
                for (int c = 0; c < d2; c++) acc += grad[r * d2 + c] * grad[s * d2 + c];
                L[r * d1 + s] += acc;
            }
        }
        for (int a = 0; a < d2; a++)
        {
            for (int b = 0; b < d2; b++)
            {
                float acc = 0f;
                for (int r = 0; r < d1; r++) acc += grad[r * d2 + a] * grad[r * d2 + b];
                R[a * d2 + b] += acc;
            }
        }

        // Refresh the inverse-fourth-roots periodically.
        if (step == 1 || step % preFreq == 0)
        {
            ComputeInverseFourthRoot(L, d1, eps, lInv);
            ComputeInverseFourthRoot(R, d2, eps, rInv);
        }

        // pre = lInv · g · rInv  (d1×d2)
        var tmp = new float[d1 * d2];
        for (int r = 0; r < d1; r++)
            for (int c = 0; c < d2; c++)
            {
                float acc = 0f;
                for (int k = 0; k < d1; k++) acc += lInv[r * d1 + k] * grad[k * d2 + c];
                tmp[r * d2 + c] = acc;
            }
        var pre = new float[d1 * d2];
        for (int r = 0; r < d1; r++)
            for (int c = 0; c < d2; c++)
            {
                float acc = 0f;
                for (int k = 0; k < d2; k++) acc += tmp[r * d2 + k] * rInv[k * d2 + c];
                pre[r * d2 + c] = acc;
            }

        for (int i = 0; i < p.Length; i++)
        {
            mb[i] = momentum * mb[i] + pre[i];
            p[i] -= lr * mb[i];
        }
    }

    /// <summary>Compute <c>(A + ε·I)^(-1/4)</c> for a symmetric PSD matrix via Jacobi eigendecomposition.</summary>
    private static void ComputeInverseFourthRoot(float[] A, int n, float eps, float[] outRoot)
    {
        // Copy A → S (Jacobi modifies S in place); regularise the diagonal.
        var S = new double[n * n];
        for (int i = 0; i < n * n; i++) S[i] = A[i];
        for (int i = 0; i < n; i++) S[i * n + i] += eps;

        // V holds eigenvectors; initialise as identity.
        var V = new double[n * n];
        for (int i = 0; i < n; i++) V[i * n + i] = 1.0;

        // Cyclic Jacobi rotations until off-diagonal Frobenius norm is small.
        const int maxSweeps = 64;
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++) off += S[p * n + q] * S[p * n + q];
            if (off < 1e-20) break;

            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double app = S[p * n + p], aqq = S[q * n + q], apq = S[p * n + q];
                    if (Math.Abs(apq) < 1e-15) continue;
                    double tau = (aqq - app) / (2 * apq);
                    double t = tau >= 0 ? 1 / (tau + Math.Sqrt(1 + tau * tau))
                                        : 1 / (tau - Math.Sqrt(1 + tau * tau));
                    double c = 1 / Math.Sqrt(1 + t * t);
                    double s = t * c;
                    // Rotate columns/rows p,q of S
                    for (int i = 0; i < n; i++)
                    {
                        double Sip = S[i * n + p], Siq = S[i * n + q];
                        S[i * n + p] = c * Sip - s * Siq;
                        S[i * n + q] = s * Sip + c * Siq;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        double Spj = S[p * n + j], Sqj = S[q * n + j];
                        S[p * n + j] = c * Spj - s * Sqj;
                        S[q * n + j] = s * Spj + c * Sqj;
                    }
                    // Force exact zero on the off-diagonal that was rotated.
                    S[p * n + q] = 0; S[q * n + p] = 0;
                    // Accumulate eigenvectors
                    for (int i = 0; i < n; i++)
                    {
                        double Vip = V[i * n + p], Viq = V[i * n + q];
                        V[i * n + p] = c * Vip - s * Viq;
                        V[i * n + q] = s * Vip + c * Viq;
                    }
                }
            }
        }

        // Eigenvalues are now on the diagonal of S; raise each to -1/4 and reconstruct.
        var lambda = new double[n];
        for (int i = 0; i < n; i++)
            lambda[i] = Math.Pow(Math.Max(S[i * n + i], 1e-30), -0.25);

        // outRoot = V · diag(lambda) · Vᵀ
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int k = 0; k < n; k++) acc += V[i * n + k] * lambda[k] * V[j * n + k];
                outRoot[i * n + j] = (float)acc;
            }
    }

    private static float[] IdentityMatrix(int n)
    {
        var m = new float[n * n];
        for (int i = 0; i < n; i++) m[i * n + i] = 1f;
        return m;
    }
}
