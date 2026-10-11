using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Shape-aware state for optimizers whose state is not one buffer per parameter element: the parameter's logical
/// shape comes from its bound tensor, or from <see cref="AddShapedParameter"/> for raw arrays.
/// </summary>
public abstract class ShapedOptimizerBase : OptimizerBase
{
    private readonly Dictionary<(int gi, int pi), int[]> _shapes = new();

    /// <summary>
    /// Adds a raw-array parameter with its logical <paramref name="shape"/> (the optimizer sees flat buffers otherwise).
    /// </summary>
    public ParamGroup AddShapedParameter(int[] shape, float[] parameter, float[] gradient, IDictionary<string, double>? overrides = null)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (parameter is null) throw new ArgumentNullException(nameof(parameter));
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (gradient.Length != parameter.Length)
            throw new ArgumentException($"The gradient has {gradient.Length} elements but the parameter has {parameter.Length}.", nameof(gradient));
        long count = 1;
        // Positive only: a negative pair can multiply to the length, and a zero one leaves the factored row/column
        // statistics an empty (divide-by-zero) average.
        foreach (int d in shape)
        {
            if (d <= 0) throw new ArgumentException($"Every dimension must be positive; got [{string.Join(", ", shape)}].", nameof(shape));
            count *= d;
        }
        if (count != parameter.Length)
            throw new ArgumentException($"The shape holds {count} elements but the parameter has {parameter.Length}.", nameof(shape));
        var group = AddParamGroup(overrides);
        group.AddParameter(parameter, gradient);
        _shapes[(ParamGroups.Count - 1, group.Parameters.Count - 1)] = (int[])shape.Clone();
        return group;
    }

    /// <summary>The logical shape of <c>group[gi].param[pi]</c>: registered, from its tensor, or flat.</summary>
    private protected int[] ParameterShape(int gi, int pi, int length)
    {
        if (_shapes.TryGetValue((gi, pi), out var s)) return s;
        var tensor = ParamGroups[gi].ParameterTensor(pi);
        return tensor is not null ? tensor.Shape.ToArray() : new[] { length };
    }

    /// <summary>A state buffer of <paramref name="length"/> floats, created zeroed on first use.</summary>
    private protected static float[] StateBuffer(Dictionary<string, OptimizerStateValue> slot, string name, int length)
    {
        if (!slot.TryGetValue(name, out var value) || value.Tensor is null || value.Tensor.Length != length)
        {
            value = OptimizerStateValue.FromTensor(new float[length]);
            slot[name] = value;
        }
        return value.Tensor!;
    }

    /// <summary>The dense, maximize-signed gradient as a flat array (sparse gradients are scattered into scratch).</summary>
    private protected float[] GradientArray(int gi, int pi, float[] parameter)
    {
        if (TryGetSparseGradient(gi, pi, out var idx, out var val, out var nnz))
        {
            var dense = new float[parameter.Length];
            for (int k = 0; k < nnz; k++) dense[idx[k]] += val[k];
            return dense;
        }
        var g = DenseGradient(gi, pi, parameter);
        if (g.Offset == 0 && g.Array.Length == parameter.Length) return g.Array;
        var copy = new float[parameter.Length];
        Array.Copy(g.Array, g.Offset, copy, 0, parameter.Length);
        return copy;
    }

    private protected static double Norm(float[] v)
    {
        double s = 0;
        for (int i = 0; i < v.Length; i++) s += (double)v[i] * v[i];
        return Math.Sqrt(s);
    }
}

/// <summary>
/// Adafactor (Shazeer &amp; Stern, 2018), as torch.optim.Adafactor: second moments of a matrix parameter kept as a
/// row vector and a column vector instead of a full matrix, relative step sizes, and update clipping.
/// </summary>
/// <remarks>
/// <para>Per step t, with the defaults of torch 2.11: decay 1 - beta2_t = t^beta2_decay; step size
/// rho_t = min(lr, 1/sqrt(t)) scaled by alpha = max(eps2, RMS(theta)) * rho_t; decoupled weight decay
/// theta *= 1 - lr * weight_decay. For a parameter of rank &gt;= 2 (the last two dims are the matrix, leading dims a
/// batch) the row and column means of g^2 are tracked, and the second-moment estimate is
/// row_var * col_var / max(mean(row_var), eps1); for a vector, an elementwise EMA of g^2. The update
/// g / sqrt(max(estimate, eps1^2)) is divided by max(1, RMS(update) / d) before it is applied.</para>
/// <para>Hyper-parameters: <c>lr</c> (1e-2), <c>beta2_decay</c> (-0.8), <c>eps1</c> (negative = the float machine
/// epsilon, torch's <c>None</c>), <c>eps2</c> (1e-3), <c>d</c> (1.0), <c>weight_decay</c> (0), <c>maximize</c>.</para>
/// </remarks>
public sealed class AdafactorOptimizer : ShapedOptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["beta2_decay"] = -0.8, ["eps1"] = -1.0, ["eps2"] = 1e-3, ["d"] = 1.0,
        ["weight_decay"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try
        {
            for (int gi = 0; gi < ParamGroups.Count; gi++)
            {
                var g = ParamGroups[gi];
                double lr = g.LearningRate;
                double beta2Decay = g.GetOption("beta2_decay", -0.8);
                double eps1Option = g.GetOption("eps1", -1.0);
                double eps1 = eps1Option < 0 ? 1.1920928955078125e-07 : eps1Option;   // torch.finfo(float32).eps
                double eps2 = g.GetOption("eps2", 1e-3);
                double d = g.GetOption("d", 1.0);
                double wd = g.GetOption("weight_decay", 0.0);
                if (beta2Decay > 0) throw new ArgumentOutOfRangeException(nameof(beta2Decay), "beta2_decay must be <= 0.");
                if (d < 1) throw new ArgumentOutOfRangeException(nameof(d), "The clipping threshold d must be >= 1.");

                for (int pi = 0; pi < g.Parameters.Count; pi++)
                {
                    if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                    float[] p = HostParameter(gi, pi);
                    var slot = GetOrCreateState(gi, pi, p.Length);
                    int step = (slot["step"].IntValue ?? 0) + 1;
                    slot["step"].IntValue = step;
                    var grad = GradientArray(gi, pi, p);
                    var shape = ParameterShape(gi, pi, p.Length);
                    int n = p.Length;

                    double w = Math.Pow(step, beta2Decay);                  // 1 - beta2_t
                    double rho = Math.Min(lr, 1.0 / Math.Sqrt(step));
                    double alpha = Math.Max(eps2, Norm(p) / Math.Sqrt(n)) * rho;   // from the parameter before decay
                    if (wd != 0)
                    {
                        float keep = (float)(1 - lr * wd);
                        for (int i = 0; i < n; i++) p[i] *= keep;
                    }

                    var update = new float[n];
                    if (shape.Length > 1)
                    {
                        int cols = shape[shape.Length - 1], rows = shape[shape.Length - 2];
                        int batch = n / (rows * cols);
                        var rowVar = StateBuffer(slot, "row_var", batch * rows);
                        var colVar = StateBuffer(slot, "col_var", batch * cols);
                        var colSum = new double[cols];
                        for (int b = 0; b < batch; b++)
                        {
                            int baseIdx = b * rows * cols;
                            Array.Clear(colSum, 0, cols);
                            for (int r = 0; r < rows; r++)
                            {
                                double rowSum = 0;
                                int rowBase = baseIdx + r * cols;
                                for (int c = 0; c < cols; c++)
                                {
                                    double sq = (double)grad[rowBase + c] * grad[rowBase + c];
                                    rowSum += sq;
                                    colSum[c] += sq;
                                }
                                ref float rv = ref rowVar[b * rows + r];
                                rv += (float)(w * (rowSum / cols - rv));
                            }
                            for (int c = 0; c < cols; c++)
                            {
                                ref float cv = ref colVar[b * cols + c];
                                cv += (float)(w * (colSum[c] / rows - cv));
                            }
                            double rowMean = 0;
                            for (int r = 0; r < rows; r++) rowMean += rowVar[b * rows + r];
                            rowMean = Math.Max(rowMean / rows, eps1);
                            for (int r = 0; r < rows; r++)
                            {
                                double rv = rowVar[b * rows + r];
                                int rowBase = baseIdx + r * cols;
                                for (int c = 0; c < cols; c++)
                                {
                                    double estimate = rv * colVar[b * cols + c] / rowMean;
                                    update[rowBase + c] = (float)(grad[rowBase + c] / Math.Sqrt(Math.Max(estimate, eps1 * eps1)));
                                }
                            }
                        }
                    }
                    else
                    {
                        var variance = StateBuffer(slot, "variance", n);
                        for (int i = 0; i < n; i++)
                        {
                            double sq = (double)grad[i] * grad[i];
                            variance[i] += (float)(w * (sq - variance[i]));
                            update[i] = (float)(grad[i] / Math.Sqrt(Math.Max(variance[i], eps1 * eps1)));
                        }
                    }
                    double denom = Math.Max(1.0, Norm(update) / (Math.Sqrt(n) * d));
                    float scale = (float)(alpha / denom);
                    for (int i = 0; i < n; i++) p[i] -= scale * update[i];
                }
            }
        }
        finally { EndStep(); }
    }
}

/// <summary>
/// Muon (Jordan et al., 2024), as torch.optim.Muon: momentum on a 2-D parameter, orthogonalized by a quintic
/// Newton-Schulz iteration before it is applied.
/// </summary>
/// <remarks>
/// <para>Per step: buf = lerp(buf, g, 1 - momentum); the update is lerp(g, buf, momentum) with Nesterov, else buf;
/// O = NS_k(update) with coefficients (a, b, c); theta *= 1 - lr * weight_decay; theta -= lr' * O, where lr' is
/// lr * sqrt(max(1, A/B)) ("original", Keller Jordan) or lr * 0.2 * sqrt(max(A, B)) ("match_rms_adamw", Moonshot)
/// for a parameter of shape [A, B]. The Newton-Schulz iteration runs in float32 (torch runs it in bfloat16), so the
/// orthogonalized update is at least as accurate as torch's. Muon is for the 2-D weights of hidden layers; biases and
/// embeddings belong with an elementwise optimizer such as AdamW, and a parameter of another rank is rejected.</para>
/// <para>Hyper-parameters: <c>lr</c> (1e-3), <c>weight_decay</c> (0.1), <c>momentum</c> (0.95), <c>nesterov</c>
/// (1), <c>ns_a</c>/<c>ns_b</c>/<c>ns_c</c> (3.4445, -4.7750, 2.0315), <c>eps</c> (1e-7), <c>ns_steps</c> (5),
/// <c>adjust_lr_fn</c> (0 = original, 1 = match_rms_adamw).</para>
/// </remarks>
public sealed class MuonOptimizer : ShapedOptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["weight_decay"] = 0.1, ["momentum"] = 0.95, ["nesterov"] = 1.0,
        ["ns_a"] = 3.4445, ["ns_b"] = -4.7750, ["ns_c"] = 2.0315, ["eps"] = 1e-7, ["ns_steps"] = 5.0,
        ["adjust_lr_fn"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    private readonly IEngine _engine = new CpuEngine();

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try
        {
            for (int gi = 0; gi < ParamGroups.Count; gi++)
            {
                var g = ParamGroups[gi];
                double lr = g.LearningRate;
                double wd = g.GetOption("weight_decay", 0.1);
                float momentum = (float)g.GetOption("momentum", 0.95);
                bool nesterov = g.GetOption("nesterov", 1.0) != 0;
                double a = g.GetOption("ns_a", 3.4445), b = g.GetOption("ns_b", -4.7750), c = g.GetOption("ns_c", 2.0315);
                double eps = g.GetOption("eps", 1e-7);
                int nsSteps = (int)g.GetOption("ns_steps", 5.0);
                int adjust = (int)g.GetOption("adjust_lr_fn", 0.0);
                if (nsSteps < 0 || nsSteps >= 100) throw new ArgumentOutOfRangeException(nameof(nsSteps), "ns_steps must be in [0, 100).");
                if (adjust != 0 && adjust != 1) throw new ArgumentOutOfRangeException(nameof(adjust), "adjust_lr_fn is 0 (original) or 1 (match_rms_adamw).");

                for (int pi = 0; pi < g.Parameters.Count; pi++)
                {
                    if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                    float[] p = HostParameter(gi, pi);
                    var shape = ParameterShape(gi, pi, p.Length);
                    if (shape.Length != 2)
                        throw new InvalidOperationException(
                            $"Muon optimizes 2-D parameters only; got shape [{string.Join(", ", shape)}]. Optimize biases and " +
                            "embeddings with an elementwise optimizer such as AdamW.");
                    var slot = GetOrCreateState(gi, pi, p.Length);
                    slot["step"].IntValue = (slot["step"].IntValue ?? 0) + 1;
                    var grad = GradientArray(gi, pi, p);
                    var buf = StateBuffer(slot, "momentum_buffer", p.Length);
                    var update = new float[p.Length];
                    for (int i = 0; i < p.Length; i++)
                    {
                        buf[i] += (1 - momentum) * (grad[i] - buf[i]);
                        update[i] = nesterov ? grad[i] + momentum * (buf[i] - grad[i]) : buf[i];
                    }
                    var ortho = NewtonSchulz(update, shape[0], shape[1], a, b, c, nsSteps, eps);
                    double adjusted = adjust == 0
                        ? lr * Math.Sqrt(Math.Max(1.0, (double)shape[0] / shape[1]))
                        : lr * 0.2 * Math.Sqrt(Math.Max(shape[0], shape[1]));
                    float keep = (float)(1 - lr * wd);
                    float step = (float)adjusted;
                    for (int i = 0; i < p.Length; i++) p[i] = p[i] * keep - step * ortho[i];
                }
            }
        }
        finally { EndStep(); }
    }

    /// <summary>
    /// The quintic Newton-Schulz orthogonalization of torch.optim.Muon: X = G / max(||G||_F, eps) (transposed so it is
    /// wide), then k times X = a X + (b A + c A^2) X with A = X X^T.
    /// </summary>
    private float[] NewtonSchulz(float[] g, int rows, int cols, double a, double b, double c, int steps, double eps)
    {
        bool tall = rows > cols;
        int r = tall ? cols : rows, k = tall ? rows : cols;
        var x = new Tensor<float>(new[] { r, k });
        var xs = x.AsWritableSpan();
        if (tall) for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) xs[j * rows + i] = g[i * cols + j];
        else g.AsSpan().CopyTo(xs);
        double norm = 0;
        for (int i = 0; i < xs.Length; i++) norm += (double)xs[i] * xs[i];
        float inv = (float)(1.0 / Math.Max(Math.Sqrt(norm), eps));
        for (int i = 0; i < xs.Length; i++) xs[i] *= inv;

        float fa = (float)a, fb = (float)b, fc = (float)c;
        for (int s = 0; s < steps; s++)
        {
            var xt = _engine.TensorTranspose(x);
            var gram = _engine.TensorMatMul(x, xt);                       // [r, r]
            var gram2 = _engine.TensorMatMul(gram, gram);
            var poly = _engine.TensorAdd(_engine.TensorMultiplyScalar(gram, fb), _engine.TensorMultiplyScalar(gram2, fc));
            x = _engine.TensorAdd(_engine.TensorMultiplyScalar(x, fa), _engine.TensorMatMul(poly, x));
        }

        var result = new float[rows * cols];
        var xr = x.IsContiguous ? x : x.Contiguous();
        var o = xr.AsSpan();
        if (tall) for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) result[i * cols + j] = o[j * rows + i];
        else o.CopyTo(result);
        return result;
    }
}
