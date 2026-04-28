using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>SGD with optional momentum, dampening, weight-decay, Nesterov, maximize.</summary>
[CudaGraphSafe]
public sealed class SgdOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3,
        ["momentum"] = 0.0,
        ["dampening"] = 0.0,
        ["weight_decay"] = 0.0,
        ["nesterov"] = 0.0,
        ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "momentum_buffer" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float momentum = (float)g.GetOption("momentum", 0.0);
            float dampening = (float)g.GetOption("dampening", 0.0);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            bool nesterov = g.GetOption("nesterov", 0.0) != 0.0;

            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi];
                float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];

                if (momentum != 0f)
                {
                    var slot = GetOrCreateState(gi, pi, p.Length);
                    var v = slot["momentum_buffer"].Tensor!;
                    if (dampening == 0f)
                    {
                        unsafe
                        {
                            fixed (float* pp = p)
                            fixed (float* pg = grad)
                            fixed (float* pv = v)
                                FusedOptimizer.SgdMomentumUpdateSimd(pp, pg, pv, p.Length, lr, momentum, nesterov);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < p.Length; i++)
                        {
                            v[i] = momentum * v[i] + (1f - dampening) * grad[i];
                            float upd = nesterov ? grad[i] + momentum * v[i] : v[i];
                            p[i] -= lr * upd;
                        }
                    }
                }
                else
                {
                    unsafe
                    {
                        fixed (float* pp = p)
                        fixed (float* pg = grad)
                            FusedOptimizer.SgdUpdateSimd(pp, pg, p.Length, lr);
                    }
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>Adam (Kingma &amp; Ba, 2015), with optional AMSGrad and maximize.</summary>
[CudaGraphSafe]
public sealed class AdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["amsgrad"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            bool amsgrad = g.GetOption("amsgrad", 0.0) != 0.0;
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];

                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pv = v)
                    {
                        if (amsgrad)
                        {
                            var vmax = slot["max_exp_avg_sq"].Tensor!;
                            fixed (float* pvm = vmax)
                                FusedOptimizer.AMSGradUpdateSimd(pp, pg, pm, pv, pvm, p.Length, lr, b1, b2, eps, step);
                        }
                        else
                        {
                            FusedOptimizer.AdamUpdateSimd(pp, pg, pm, pv, p.Length, lr, b1, b2, eps, step);
                        }
                    }
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>AdamW (Loshchilov &amp; Hutter, 2019): Adam with decoupled weight decay.</summary>
[CudaGraphSafe]
public sealed class AdamWOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 1e-2, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 1e-2);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pv = v)
                        FusedOptimizer.AdamWUpdateSimd(pp, pg, pm, pv, p.Length, lr, b1, b2, eps, wd, step);
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>RAdam (Liu et al., 2020): rectified Adam.</summary>
[CudaGraphSafe(Note = "rectification branch depends only on the step counter, not tensor values")]
public sealed class RAdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pv = v)
                        FusedOptimizer.RAdamUpdateSimd(pp, pg, pm, pv, p.Length, lr, b1, b2, eps, step);
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>NAdam: Adam with Nesterov-look-ahead.</summary>
[CudaGraphSafe]
public sealed class NAdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 2e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pv = v)
                        FusedOptimizer.NadamUpdateSimd(pp, pg, pm, pv, p.Length, lr, b1, b2, eps, step);
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>Adamax: Adam with infinity norm.</summary>
[CudaGraphSafe]
public sealed class AdamaxOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 2e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
        ["weight_decay"] = 0.0, ["maximize"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_inf" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        bool maximized = ApplyMaximize();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var u = slot["exp_inf"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pu = u)
                        FusedOptimizer.AdaMaxUpdateSimd(pp, pg, pm, pu, p.Length, lr, b1, b2, eps, step);
                }
            }
        }
        } finally { if (maximized) UnflipMaximize(); }
    }
}

/// <summary>Adagrad: accumulated squared gradients.</summary>
[CudaGraphSafe]
public sealed class AdagradOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["eps"] = 1e-10, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "sum" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float eps = (float)g.GetOption("eps", 1e-10);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var s = slot["sum"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* ps = s)
                        FusedOptimizer.AdagradUpdateSimd(pp, pg, ps, p.Length, lr, eps);
                }
            }
        }
    }
}

/// <summary>RMSprop (Hinton lecture): running average of squared gradients.
/// Supports the centered variant (Graves, 2013) and Polyak momentum, matching
/// <c>torch.optim.RMSprop(centered=, momentum=)</c>.</summary>
[CudaGraphSafe(Note = "centered/momentum branches selected at construction; the variance<0 clamp is a value clamp, not control flow")]
public sealed class RmsPropOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["alpha"] = 0.99, ["eps"] = 1e-8, ["weight_decay"] = 0.0,
        ["momentum"] = 0.0, ["centered"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "square_avg", "grad_avg", "momentum_buffer" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float rho = (float)g.GetOption("alpha", 0.99);
            float eps = (float)g.GetOption("eps", 1e-8);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            float mom = (float)g.GetOption("momentum", 0.0);
            bool centered = g.GetOption("centered", 0.0) != 0.0;
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var v = slot["square_avg"].Tensor!;

                if (centered)
                {
                    var gAvg = slot["grad_avg"].Tensor!;
                    if (mom == 0f)
                    {
                        unsafe
                        {
                            fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pv = v) fixed (float* pga = gAvg)
                                FusedOptimizer.RMSpropCenteredUpdate(pp, pg, pv, pga, p.Length, lr, rho, eps);
                        }
                    }
                    else
                    {
                        // Centered + Polyak momentum: maintain mom_buf = mom·mom_buf + g/(sqrt(var)+eps); p -= lr·mom_buf
                        var mb = slot["momentum_buffer"].Tensor!;
                        for (int i = 0; i < p.Length; i++)
                        {
                            v[i] = rho * v[i] + (1f - rho) * grad[i] * grad[i];
                            gAvg[i] = rho * gAvg[i] + (1f - rho) * grad[i];
                            float variance = v[i] - gAvg[i] * gAvg[i];
                            if (variance < 0f) variance = 0f;
                            float scaled = grad[i] / (MathF.Sqrt(variance) + eps);
                            mb[i] = mom * mb[i] + scaled;
                            p[i] -= lr * mb[i];
                        }
                    }
                }
                else if (mom != 0f)
                {
                    var mb = slot["momentum_buffer"].Tensor!;
                    for (int i = 0; i < p.Length; i++)
                    {
                        v[i] = rho * v[i] + (1f - rho) * grad[i] * grad[i];
                        float scaled = grad[i] / (MathF.Sqrt(v[i]) + eps);
                        mb[i] = mom * mb[i] + scaled;
                        p[i] -= lr * mb[i];
                    }
                }
                else
                {
                    unsafe
                    {
                        fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pv = v)
                            FusedOptimizer.RMSpropUpdateSimd(pp, pg, pv, p.Length, lr, rho, eps);
                    }
                }
            }
        }
    }
}

/// <summary>AdaDelta: adaptive learning rate without a global lr.</summary>
[CudaGraphSafe]
public sealed class AdaDeltaOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1.0, ["rho"] = 0.9, ["eps"] = 1e-6, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "square_avg", "acc_delta" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float rho = (float)g.GetOption("rho", 0.9);
            float eps = (float)g.GetOption("eps", 1e-6);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                if (wd != 0f)
                    for (int i = 0; i < p.Length; i++) grad[i] += wd * p[i];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var sq = slot["square_avg"].Tensor!;
                var ad = slot["acc_delta"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* psq = sq) fixed (float* pad = ad)
                        FusedOptimizer.AdaDeltaUpdateSimd(pp, pg, psq, pad, p.Length, lr, rho, eps);
                }
            }
        }
    }
}

/// <summary>Lion (Chen et al., 2023): sign-based optimizer with EMA.</summary>
[CudaGraphSafe]
public sealed class LionOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-4, ["beta1"] = 0.9, ["beta2"] = 0.99, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "exp_avg" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.99);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var m = slot["exp_avg"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m)
                        FusedOptimizer.LionUpdateSimd(pp, pg, pm, p.Length, lr, b1, b2, wd);
                }
            }
        }
    }
}

/// <summary>ASGD: Averaged SGD (Polyak/Ruppert).</summary>
[CudaGraphSafe]
public sealed class AsgdOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["lambd"] = 1e-4, ["alpha"] = 0.75, ["t0"] = 1e6, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "eta", "mu", "ax" };
    private static readonly string[] _scalarNames = new[] { "step", "eta", "mu" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;
    /// <inheritdoc />
    protected override IReadOnlyList<string> ScalarStateNames => _scalarNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float baseLr = (float)g.LearningRate;
            float lambd = (float)g.GetOption("lambd", 1e-4);
            float alpha = (float)g.GetOption("alpha", 0.75);
            float t0 = (float)g.GetOption("t0", 1e6);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;

                // Schedule (PyTorch parity):
                //   eta = lr / (1 + λ·lr·step)^α
                //   mu  = 1 / max(1, step − t0)
                float eta = baseLr / (float)Math.Pow(1.0 + lambd * baseLr * step, alpha);
                float mu = 1f / MathF.Max(1f, step - t0);
                slot["eta"].FloatValue = eta;
                slot["mu"].FloatValue = mu;
                var ax = slot["ax"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pax = ax)
                        FusedOptimizer.ASGDUpdateSimd(pp, pg, pax, p.Length, eta, lambd, alpha, wd, mu);
                }
            }
        }
    }
}

/// <summary>Rprop: Resilient back-propagation.</summary>
public sealed class RpropOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["eta_plus"] = 1.2, ["eta_minus"] = 0.5, ["step_min"] = 1e-6, ["step_max"] = 50.0,
    };
    private static readonly string[] _stateNames = new[] { "prev_grad", "step_size" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float etaP = (float)g.GetOption("eta_plus", 1.2);
            float etaM = (float)g.GetOption("eta_minus", 0.5);
            float sMin = (float)g.GetOption("step_min", 1e-6);
            float sMax = (float)g.GetOption("step_max", 50.0);
            float lr0 = (float)g.LearningRate;
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var prev = slot["prev_grad"].Tensor!;
                var ss = slot["step_size"].Tensor!;
                bool firstCall = false;
                if (slot["step_size"].IntValue == null) slot["step_size"].IntValue = 0;
                if ((slot["step_size"].IntValue ?? 0) == 0)
                {
                    for (int k = 0; k < ss.Length; k++) ss[k] = lr0;
                    slot["step_size"].IntValue = 1;
                    firstCall = true;
                }
                _ = firstCall;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* ppr = prev) fixed (float* pss = ss)
                        FusedOptimizer.RpropUpdate(pp, pg, ppr, pss, p.Length, etaP, etaM, sMin, sMax);
                }
            }
        }
    }
}

/// <summary>LAMB (You et al., 2020): layer-wise Adaptive Moments for Batch training.
/// Adam-style moments + per-layer trust-ratio scaling for very large batch sizes.</summary>
[CudaGraphSafe(Note = "trust-ratio branch selects between two value paths; no host-side control flow change")]
public sealed class LambOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-6, ["weight_decay"] = 0.0,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-6);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pm = m) fixed (float* pv = v)
                        FusedOptimizer.LAMBUpdateSimd(pp, pg, pm, pv, p.Length, lr, b1, b2, eps, wd, step);
                }
            }
        }
    }
}

/// <summary>LARS (You et al., 2017): Layer-wise Adaptive Rate Scaling.
/// SGD+momentum with a layer-local trust ratio that scales LR by ‖p‖/‖g‖.</summary>
[CudaGraphSafe]
public sealed class LarsOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["momentum"] = 0.9, ["weight_decay"] = 0.0, ["trust_coeff"] = 1e-3,
    };
    private static readonly string[] _stateNames = new[] { "momentum_buffer" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float mom = (float)g.GetOption("momentum", 0.9);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            float tc = (float)g.GetOption("trust_coeff", 1e-3);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var v = slot["momentum_buffer"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pv = v)
                        FusedOptimizer.LARSUpdateSimd(pp, pg, pv, p.Length, lr, mom, wd, tc);
                }
            }
        }
    }
}

/// <summary>FTRL (McMahan, 2013): Follow-The-Regularized-Leader with L1+L2 regularisation.
/// Per-feature accumulators z and n; exact L1 soft-thresholding produces sparse weights.</summary>
public sealed class FtrlOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-2, ["l1_reg"] = 0.0, ["l2_reg"] = 0.0, ["lr_power"] = -0.5,
    };
    private static readonly string[] _stateNames = new[] { "z", "n" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float l1 = (float)g.GetOption("l1_reg", 0.0);
            float l2 = (float)g.GetOption("l2_reg", 0.0);
            // FTRL paper convention: lr_power = −0.5 yields the sqrt(n) schedule.
            // The fused kernel computes n^(-lrPower), so we pass lr_power directly
            // (not negated) — feeding −0.5 produces n^(0.5) = sqrt(n) inside.
            float lrPow = (float)g.GetOption("lr_power", -0.5);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                var z = slot["z"].Tensor!;
                var n = slot["n"].Tensor!;
                unsafe
                {
                    fixed (float* pp = p) fixed (float* pg = grad) fixed (float* pz = z) fixed (float* pn = n)
                        FusedOptimizer.FTRLUpdateSimd(pp, pg, pz, pn, p.Length, lr, l1, l2, lrPow);
                }
            }
        }
    }
}

/// <summary>SparseAdam: Adam restricted to non-zero gradient indices.
/// Caller writes a dense gradient buffer; the optimizer detects the non-zero
/// indices and applies the Adam update only to those entries.</summary>
public sealed class SparseAdamOptimizer : OptimizerBase
{
    private static readonly Dictionary<string, double> _defaults = new Dictionary<string, double>
    {
        ["lr"] = 1e-3, ["beta1"] = 0.9, ["beta2"] = 0.999, ["eps"] = 1e-8,
    };
    private static readonly string[] _stateNames = new[] { "step", "exp_avg", "exp_avg_sq" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                float[] p = g.Parameters[pi]; float[] grad = g.Gradients[pi];
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                int nnz = 0;
                for (int i = 0; i < grad.Length; i++) if (grad[i] != 0f) nnz++;
                if (nnz == 0) continue;
                var idx = new int[nnz];
                var val = new float[nnz];
                int k = 0;
                for (int i = 0; i < grad.Length; i++)
                {
                    if (grad[i] != 0f) { idx[k] = i; val[k] = grad[i]; k++; }
                }
                unsafe
                {
                    fixed (float* pp = p) fixed (int* pi2 = idx) fixed (float* pv2 = val)
                    fixed (float* pm = m) fixed (float* pvv = v)
                        FusedOptimizer.SparseAdamUpdate(pp, pi2, pv2, pm, pvv, nnz, lr, b1, b2, eps, step);
                }
            }
        }
    }
}
