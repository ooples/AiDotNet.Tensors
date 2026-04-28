using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Limited-memory BFGS optimizer (Nocedal &amp; Wright, Numerical Optimization §7.2).
///
/// Closure-based: each call to <see cref="Step"/> repeatedly evaluates the user-supplied
/// loss + gradient closure while taking quasi-Newton steps along the L-BFGS direction
/// from the two-loop recursion. Optional strong-Wolfe line search. Maintains a history
/// of size <c>HistorySize</c> of (s, y) pairs flattened over all parameter buffers.
///
/// Layout matches PyTorch <c>torch.optim.LBFGS</c>:
///   * single parameter group only
///   * lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size
///   * line_search_fn ∈ { null, "strong_wolfe" }
///   * step(closure) returns the final loss
/// </summary>
public sealed class LBFGSOptimizer
{
    /// <summary>
    /// Shape of every parameter buffer the optimizer owns. The closure must write gradients
    /// of identical length back into the matching index of <see cref="GradientBuffers"/>.
    /// </summary>
    public IReadOnlyList<float[]> ParameterBuffers { get; }

    /// <summary>Gradient buffers in 1-1 correspondence with <see cref="ParameterBuffers"/>.</summary>
    public IReadOnlyList<float[]> GradientBuffers { get; }

    /// <summary>Step length. Defaults to 1.0 (PyTorch default).</summary>
    public float LearningRate { get; set; }

    /// <summary>Max inner iterations per <see cref="Step"/> call. Default 20.</summary>
    public int MaxIter { get; set; }

    /// <summary>Max function/gradient evaluations per <see cref="Step"/> call. Default 1.25 × <see cref="MaxIter"/>.</summary>
    public int MaxEval { get; set; }

    /// <summary>Termination if <c>max|g|</c> ≤ this value. Default 1e-7.</summary>
    public float ToleranceGrad { get; set; }

    /// <summary>Termination if <c>|step·d|</c> ≤ this value. Default 1e-9.</summary>
    public float ToleranceChange { get; set; }

    /// <summary>Number of (s, y) pairs to keep. Default 100.</summary>
    public int HistorySize { get; set; }

    /// <summary>Line-search strategy. <c>null</c> = constant step <see cref="LearningRate"/>; <c>"strong_wolfe"</c> = strong-Wolfe.</summary>
    public string? LineSearchFn { get; set; }

    private readonly int _flatLen;

    // History buffers; each holds <see cref="HistorySize"/> flattened parameter-sized vectors.
    private readonly Queue<float[]> _oldS = new Queue<float[]>();
    private readonly Queue<float[]> _oldY = new Queue<float[]>();
    private readonly Queue<float> _oldRho = new Queue<float>();
    private float _h0 = 1f; // diagonal Hessian estimate

    private float[]? _prevFlatGrad;

    /// <summary>Total number of <see cref="Step"/> calls made.</summary>
    public int StepCount { get; private set; }

    /// <summary>Total number of closure evaluations across all <see cref="Step"/> calls.</summary>
    public int FuncEvals { get; private set; }

    /// <summary>
    /// Build an LBFGS optimizer. Parameters and gradients are referenced (not copied);
    /// the caller is responsible for keeping them alive and for writing gradients into
    /// <see cref="GradientBuffers"/> from the closure.
    /// </summary>
    public LBFGSOptimizer(
        IReadOnlyList<float[]> parameterBuffers,
        IReadOnlyList<float[]> gradientBuffers,
        float lr = 1f,
        int maxIter = 20,
        int? maxEval = null,
        float toleranceGrad = 1e-7f,
        float toleranceChange = 1e-9f,
        int historySize = 100,
        string? lineSearchFn = null)
    {
        if (parameterBuffers == null) throw new ArgumentNullException(nameof(parameterBuffers));
        if (gradientBuffers == null) throw new ArgumentNullException(nameof(gradientBuffers));
        if (parameterBuffers.Count != gradientBuffers.Count)
            throw new ArgumentException("parameter and gradient buffers must have the same count.");
        for (int i = 0; i < parameterBuffers.Count; i++)
        {
            if (parameterBuffers[i].Length != gradientBuffers[i].Length)
                throw new ArgumentException($"parameter[{i}] length != gradient[{i}] length.");
        }
        if (lr <= 0f) throw new ArgumentOutOfRangeException(nameof(lr));
        if (maxIter <= 0) throw new ArgumentOutOfRangeException(nameof(maxIter));
        if (toleranceGrad < 0f) throw new ArgumentOutOfRangeException(nameof(toleranceGrad));
        if (toleranceChange < 0f) throw new ArgumentOutOfRangeException(nameof(toleranceChange));
        if (historySize <= 0) throw new ArgumentOutOfRangeException(nameof(historySize));
        if (lineSearchFn != null && lineSearchFn != "strong_wolfe")
            throw new ArgumentException($"unsupported line search '{lineSearchFn}'.", nameof(lineSearchFn));

        ParameterBuffers = parameterBuffers;
        GradientBuffers = gradientBuffers;
        LearningRate = lr;
        MaxIter = maxIter;
        MaxEval = maxEval ?? (int)(maxIter * 1.25);
        ToleranceGrad = toleranceGrad;
        ToleranceChange = toleranceChange;
        HistorySize = historySize;
        LineSearchFn = lineSearchFn;

        int flat = 0;
        for (int i = 0; i < parameterBuffers.Count; i++) flat += parameterBuffers[i].Length;
        _flatLen = flat;
    }

    /// <summary>
    /// Run up to <see cref="MaxIter"/> L-BFGS iterations. The closure must:
    /// 1) zero the gradient buffers (or allow this loop to do so),
    /// 2) compute the current loss, and
    /// 3) write ∇loss into <see cref="GradientBuffers"/>.
    /// Returns the final loss value.
    /// </summary>
    public float Step(Func<float> closure)
    {
        if (closure == null) throw new ArgumentNullException(nameof(closure));

        StepCount++;
        // Per-PyTorch semantics, MaxEval is a budget for THIS Step() call, not cumulative.
        int funcEvalsAtStepStart = FuncEvals;
        float loss = closure();
        FuncEvals++;
        float[] flatGrad = FlattenGradient();
        float gMax = MaxAbs(flatGrad);
        if (gMax <= ToleranceGrad) return loss;

        float[] d = new float[_flatLen]; // search direction (negative two-loop result)
        int nIter = 0;
        float t = LearningRate;
        bool firstIter = _prevFlatGrad == null;

        while (nIter < MaxIter)
        {
            nIter++;

            // ---------------- direction = − H · grad (two-loop recursion) ----------------
            if (firstIter)
            {
                Negate(flatGrad, d);
                _h0 = 1f;
            }
            else
            {
                // y = grad - prev_grad
                float[] y = new float[_flatLen];
                for (int j = 0; j < _flatLen; j++) y[j] = flatGrad[j] - _prevFlatGrad![j];
                // s = t · d_prev (we stored d_prev as the previous direction; it's overwritten below)
                // BUT: in PyTorch's implementation, s = t * d (the step actually taken).
                // We rebuild s here from the buffer we save below as `prevD`.
                float[] s = ScaleCopy(_prevD!, _prevT);
                float ys = Dot(y, s);
                if (ys > 1e-10f)
                {
                    if (_oldS.Count == HistorySize)
                    {
                        _oldS.Dequeue(); _oldY.Dequeue(); _oldRho.Dequeue();
                    }
                    _oldS.Enqueue(s);
                    _oldY.Enqueue(y);
                    _oldRho.Enqueue(1f / ys);
                    _h0 = ys / Dot(y, y);
                }

                TwoLoopRecursion(flatGrad, d);
            }
            firstIter = false;

            // copy current grad into prev_grad for next step's y = g − g_prev
            if (_prevFlatGrad == null) _prevFlatGrad = new float[_flatLen];
            Array.Copy(flatGrad, _prevFlatGrad, _flatLen);
            float prevLoss = loss;

            // ---------------- step size selection ----------------
            float gtd = Dot(flatGrad, d);
            if (gtd > -ToleranceChange) break; // direction is not a descent direction

            // Reset t on the first inner step to lr; afterwards use 1.0 (matches PyTorch).
            t = nIter == 1 ? Math.Min(1f, 1f / Math.Max(1f, AbsSum(flatGrad))) * LearningRate : 1f;

            float[] xInit = FlattenParams();

            float lossNew;
            float[] gradNew;
            float tApplied;
            if (LineSearchFn == "strong_wolfe")
            {
                (lossNew, gradNew, tApplied) = StrongWolfe(xInit, t, d, loss, flatGrad, gtd, closure);
            }
            else
            {
                SetParamsFrom(xInit, d, t);
                lossNew = closure();
                FuncEvals++;
                gradNew = FlattenGradient();
                tApplied = t;
            }

            // Save what step we actually took, for the next iteration's s reconstruction.
            _prevD = (float[])d.Clone();
            _prevT = tApplied;
            t = tApplied;

            float[] dNorm = Abs(d);
            float dAbsMax = MaxAbs(dNorm);
            if (Math.Abs(t) * dAbsMax <= ToleranceChange) { loss = lossNew; flatGrad = gradNew; break; }
            if (Math.Abs(lossNew - prevLoss) < ToleranceChange) { loss = lossNew; flatGrad = gradNew; break; }

            loss = lossNew;
            flatGrad = gradNew;
            gMax = MaxAbs(flatGrad);
            if (gMax <= ToleranceGrad) break;
            if (FuncEvals - funcEvalsAtStepStart >= MaxEval) break;
        }

        return loss;
    }

    /// <summary>Reset L-BFGS history (forget the curvature pairs).</summary>
    public void Reset()
    {
        _oldS.Clear(); _oldY.Clear(); _oldRho.Clear();
        _prevFlatGrad = null;
        _prevD = null;
        _prevT = 0f;
        _h0 = 1f;
    }

    private float[]? _prevD;
    private float _prevT;

    private void TwoLoopRecursion(float[] grad, float[] d)
    {
        // Two-loop recursion (Nocedal & Wright Algorithm 7.4):
        //   q ← g
        //   for i = m-1..0:  α_i = ρ_i s_iᵀ q;   q ← q − α_i y_i
        //   r ← H₀ q
        //   for i = 0..m-1:  β = ρ_i y_iᵀ r;     r ← r + s_i (α_i − β)
        //   direction = −r
        float[] q = (float[])grad.Clone();
        int m = _oldS.Count;
        var sArr = _oldS.ToArray();
        var yArr = _oldY.ToArray();
        var rhoArr = _oldRho.ToArray();
        var alpha = new float[m];
        for (int i = m - 1; i >= 0; i--)
        {
            alpha[i] = rhoArr[i] * Dot(sArr[i], q);
            for (int j = 0; j < _flatLen; j++) q[j] -= alpha[i] * yArr[i][j];
        }
        var r = new float[_flatLen];
        for (int j = 0; j < _flatLen; j++) r[j] = _h0 * q[j];
        for (int i = 0; i < m; i++)
        {
            float beta = rhoArr[i] * Dot(yArr[i], r);
            for (int j = 0; j < _flatLen; j++) r[j] += sArr[i][j] * (alpha[i] - beta);
        }
        for (int j = 0; j < _flatLen; j++) d[j] = -r[j];
    }

    private (float lossNew, float[] gradNew, float t) StrongWolfe(
        float[] xInit, float t, float[] d, float fInit, float[] gInit, float gtd, Func<float> closure)
    {
        // Compact strong-Wolfe (Nocedal & Wright Algorithm 3.5/3.6).
        const float c1 = 1e-4f;
        const float c2 = 0.9f;
        const float xtol = 1e-9f;
        const int maxLs = 25;

        // Probe φ(t)
        SetParamsFrom(xInit, d, t);
        float fNew = closure(); FuncEvals++;
        float[] gNew = FlattenGradient();
        float gtdNew = Dot(gNew, d);

        float tPrev = 0f, fPrev = fInit, gtdPrev = gtd;
        float[]? gPrev = null;
        bool done = false;
        int ls = 0;
        while (!done && ls < maxLs)
        {
            ls++;
            if (fNew > fInit + c1 * t * gtd || (ls > 1 && fNew >= fPrev))
            {
                (fNew, gNew, t) = Zoom(xInit, d, fInit, gtd, tPrev, t, fPrev, fNew, gtdPrev, gtdNew, c1, c2, xtol, closure, gPrev, gNew);
                done = true; break;
            }
            if (Math.Abs(gtdNew) <= -c2 * gtd) { done = true; break; }
            if (gtdNew >= 0f)
            {
                (fNew, gNew, t) = Zoom(xInit, d, fInit, gtd, t, tPrev, fNew, fPrev, gtdNew, gtdPrev, c1, c2, xtol, closure, gNew, gPrev);
                done = true; break;
            }

            // expand t
            float tNew = Math.Min(t * 2.5f, t + 100f);
            tPrev = t; fPrev = fNew; gtdPrev = gtdNew; gPrev = gNew;
            t = tNew;
            SetParamsFrom(xInit, d, t);
            fNew = closure(); FuncEvals++;
            gNew = FlattenGradient();
            gtdNew = Dot(gNew, d);
        }
        return (fNew, gNew, t);
    }

    private (float, float[], float) Zoom(
        float[] xInit, float[] d, float fInit, float gtd,
        float lo, float hi, float fLo, float fHi, float gtdLo, float gtdHi,
        float c1, float c2, float xtol, Func<float> closure,
        float[]? gLoOpt, float[]? gHiOpt)
    {
        float fNew = fLo; float[] gNew = gLoOpt ?? new float[_flatLen]; float t = lo;
        for (int it = 0; it < 25; it++)
        {
            t = 0.5f * (lo + hi);
            if (Math.Abs(hi - lo) < xtol) break;
            SetParamsFrom(xInit, d, t);
            fNew = closure(); FuncEvals++;
            gNew = FlattenGradient();
            float gtdNew = Dot(gNew, d);
            if (fNew > fInit + c1 * t * gtd || fNew >= fLo)
            {
                hi = t; fHi = fNew; gtdHi = gtdNew;
            }
            else
            {
                if (Math.Abs(gtdNew) <= -c2 * gtd) break;
                if (gtdNew * (hi - lo) >= 0f) { hi = lo; fHi = fLo; gtdHi = gtdLo; }
                lo = t; fLo = fNew; gtdLo = gtdNew;
            }
        }
        return (fNew, gNew, t);
    }

    // ----------------- helpers (flat <-> per-buffer) -----------------

    private float[] FlattenGradient()
    {
        var flat = new float[_flatLen];
        int off = 0;
        for (int i = 0; i < GradientBuffers.Count; i++)
        {
            var src = GradientBuffers[i];
            Array.Copy(src, 0, flat, off, src.Length);
            off += src.Length;
        }
        return flat;
    }

    private float[] FlattenParams()
    {
        var flat = new float[_flatLen];
        int off = 0;
        for (int i = 0; i < ParameterBuffers.Count; i++)
        {
            var src = ParameterBuffers[i];
            Array.Copy(src, 0, flat, off, src.Length);
            off += src.Length;
        }
        return flat;
    }

    private void CopyToParams(float[] flat)
    {
        int off = 0;
        for (int i = 0; i < ParameterBuffers.Count; i++)
        {
            var dst = ParameterBuffers[i];
            Array.Copy(flat, off, dst, 0, dst.Length);
            off += dst.Length;
        }
    }

    private static void AddScaled(float[] dst, float[] dir, float t)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] += t * dir[i];
    }

    /// <summary>Write <c>params ← xInit + t · d</c> directly into <see cref="ParameterBuffers"/>
    /// without mutating <paramref name="xInit"/>. Used by the line-search probes so the
    /// starting point remains stable across multiple t evaluations.</summary>
    private void SetParamsFrom(float[] xInit, float[] d, float t)
    {
        int off = 0;
        for (int b = 0; b < ParameterBuffers.Count; b++)
        {
            var dst = ParameterBuffers[b];
            for (int i = 0; i < dst.Length; i++) dst[i] = xInit[off + i] + t * d[off + i];
            off += dst.Length;
        }
    }
    private static void Negate(float[] src, float[] dst)
    {
        for (int i = 0; i < src.Length; i++) dst[i] = -src[i];
    }
    private static float Dot(float[] a, float[] b)
    {
        double acc = 0;
        for (int i = 0; i < a.Length; i++) acc += (double)a[i] * b[i];
        return (float)acc;
    }
    private static float MaxAbs(float[] a)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) { var x = Math.Abs(a[i]); if (x > m) m = x; }
        return m;
    }
    private static float AbsSum(float[] a)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += Math.Abs(a[i]);
        return (float)s;
    }
    private static float[] Abs(float[] a)
    {
        var r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = Math.Abs(a[i]);
        return r;
    }
    private static float[] ScaleCopy(float[] a, float t)
    {
        var r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] * t;
        return r;
    }
}
