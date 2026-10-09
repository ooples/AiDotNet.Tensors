using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        float lr = (float)g.LearningRate;
        float momentum = (float)g.GetOption("momentum", 0.0);
        float wd = (float)g.GetOption("weight_decay", 0.0);
        // The device kernels have neither PyTorch's Nesterov form nor dampening; those run on the host.
        if (g.GetOption("nesterov", 0.0) != 0.0 || g.GetOption("dampening", 0.0) != 0.0) return false;
        if (momentum == 0f) return GpuOptimizer.TrySgdStep(parameter, gradient, lr, wd);
        var slot = DeviceState(gi, pi, parameter.Length);
        return GpuOptimizer.TrySgdMomentumStep(parameter, gradient, DeviceSlot(slot, "momentum_buffer"), lr, momentum, wd);
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    var slotSp = momentum != 0f ? GetOrCreateState(gi, pi, p.Length) : null;
                    var vSp = slotSp != null ? slotSp["momentum_buffer"].Tensor! : null;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* pv = vSp)
                            FusedOptimizer.SparseSgdUpdate(pp, pIdx, pVal, pv, sNnz, lr, momentum, dampening, wd, nesterov, momentum != 0f);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);
                if (momentum != 0f)
                {
                    // PyTorch starts the buffer at the first gradient itself (buf = grad.clone()), undamped.
                    bool firstStep = !_state.ContainsKey((gi, pi));
                    var slot = GetOrCreateState(gi, pi, p.Length);
                    var v = slot["momentum_buffer"].Tensor!;
                    float damping = firstStep ? 0f : dampening;
                    if (damping == 0f)
                    {
                        // The kernel computes v = momentum·v + g, which with v still zero is the first-step clone.
                        ForEachChunk(p.Length, (start, count) =>
                        {
                            unsafe
                            {
                                fixed (float* pp = p, pg = grad.Array, pv = v)
                                    FusedOptimizer.SgdMomentumUpdateSimd(pp + start, pg + grad.Offset + start, pv + start,
                                        count, lr, momentum, nesterov);
                            }
                        });
                    }
                    else
                    {
                        ForEachChunk(p.Length, (start, count) =>
                        {
                            float[] ga = grad.Array;
                            int go = grad.Offset;
                            for (int i = start; i < start + count; i++)
                            {
                                v[i] = momentum * v[i] + (1f - damping) * ga[go + i];
                                float upd = nesterov ? ga[go + i] + momentum * v[i] : v[i];
                                p[i] -= lr * upd;
                            }
                        });
                    }
                }
                else
                {
                    ForEachChunk(p.Length, (start, count) =>
                    {
                        unsafe
                        {
                            fixed (float* pp = p, pg = grad.Array)
                                FusedOptimizer.SgdUpdateSimd(pp + start, pg + grad.Offset + start, count, lr);
                        }
                    });
                }
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        int step = (slot["step"].IntValue ?? 0) + 1;
        float lr = (float)g.LearningRate, b1 = (float)g.GetOption("beta1", 0.9), b2 = (float)g.GetOption("beta2", 0.999);
        float eps = (float)g.GetOption("eps", 1e-8), wd = (float)g.GetOption("weight_decay", 0.0);
        var m = DeviceSlot(slot, "exp_avg");
        var v = DeviceSlot(slot, "exp_avg_sq");
        bool ran = g.GetOption("amsgrad", 0.0) != 0.0
            ? GpuOptimizer.TryAmsgradStep(parameter, gradient, m, v, DeviceSlot(slot, "max_exp_avg_sq"), lr, b1, b2, eps, wd, step)
            : GpuOptimizer.TryAdamStep(parameter, gradient, m, v, lr, b1, b2, eps, wd, step);
        if (ran) slot["step"].IntValue = step;
        return ran;
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                // Sparse fast path — caller (e.g. PyTorch-style training loop on top of
                // SparseEmbeddingGradient) published (indices, values) for this param.
                // Only the touched entries are updated; the rest of (param, m, v) stay valid.
                // AMSGrad isn't supported via sparse: max_exp_avg_sq must be coupled with EVERY
                // update site, so the optimizer falls back to dense when amsgrad is enabled.
                if (!amsgrad && TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    if (wd != 0f)
                    {
                        // Apply L2 weight-decay only at touched indices: g_eff = g + wd*p.
                        // We allocate-free-ish: mutate a copy of sVal so caller's snapshot is untouched.
                        var sValEff = new float[sNnz];
                        for (int k = 0; k < sNnz; k++) sValEff[k] = sVal[k] + wd * p[sIdx[k]];
                        unsafe
                        {
                            fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sValEff)
                            fixed (float* pm = m) fixed (float* pvv = v)
                                FusedOptimizer.SparseAdamUpdate(pp, pIdx, pVal, pm, pvv, sNnz, lr, b1, b2, eps, step);
                        }
                    }
                    else
                    {
                        unsafe
                        {
                            fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                            fixed (float* pm = m) fixed (float* pvv = v)
                                FusedOptimizer.SparseAdamUpdate(pp, pIdx, pVal, pm, pvv, sNnz, lr, b1, b2, eps, step);
                        }
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);

                var vmax = amsgrad ? slot["max_exp_avg_sq"].Tensor! : null;
                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m, pv = v, pvm = vmax)
                        {
                            if (pvm != null)
                                FusedOptimizer.AMSGradUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pv + start,
                                    pvm + start, count, lr, b1, b2, eps, step);
                            else
                                FusedOptimizer.AdamUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pv + start,
                                    count, lr, b1, b2, eps, step);
                        }
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        int step = (slot["step"].IntValue ?? 0) + 1;
        bool ran = GpuOptimizer.TryAdamWStep(parameter, gradient, DeviceSlot(slot, "exp_avg"), DeviceSlot(slot, "exp_avg_sq"),
            (float)g.LearningRate, (float)g.GetOption("beta1", 0.9), (float)g.GetOption("beta2", 0.999),
            (float)g.GetOption("eps", 1e-8), (float)g.GetOption("weight_decay", 1e-2), step);
        if (ran) slot["step"].IntValue = step;
        return ran;
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pm = m) fixed (float* pvv = v)
                            FusedOptimizer.SparseAdamWUpdate(pp, pIdx, pVal, pm, pvv, sNnz, lr, b1, b2, eps, wd, step);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p);
                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m, pv = v)
                            FusedOptimizer.AdamWUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pv + start, count, lr, b1, b2, eps, wd, step);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pm = m) fixed (float* pvv = v)
                            FusedOptimizer.SparseRAdamUpdate(pp, pIdx, pVal, pm, pvv, sNnz, lr, b1, b2, eps, wd, step);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);

                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m, pv = v)
                            FusedOptimizer.RAdamUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pv + start, count, lr, b1, b2, eps, step);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        int step = (slot["step"].IntValue ?? 0) + 1;
        bool ran = GpuOptimizer.TryNadamStep(parameter, gradient, DeviceSlot(slot, "exp_avg"), DeviceSlot(slot, "exp_avg_sq"),
            (float)g.LearningRate, (float)g.GetOption("beta1", 0.9), (float)g.GetOption("beta2", 0.999),
            (float)g.GetOption("eps", 1e-8), (float)g.GetOption("weight_decay", 0.0), step);
        if (ran) slot["step"].IntValue = step;
        return ran;
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pm = m) fixed (float* pvv = v)
                            FusedOptimizer.SparseNadamUpdate(pp, pIdx, pVal, pm, pvv, sNnz, lr, b1, b2, eps, wd, step);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);

                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m, pv = v)
                            FusedOptimizer.NadamUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pv + start, count, lr, b1, b2, eps, step);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        int step = (slot["step"].IntValue ?? 0) + 1;
        bool ran = GpuOptimizer.TryAdamaxStep(parameter, gradient, DeviceSlot(slot, "exp_avg"), DeviceSlot(slot, "exp_inf"),
            (float)g.LearningRate, (float)g.GetOption("beta1", 0.9), (float)g.GetOption("beta2", 0.999),
            (float)g.GetOption("eps", 1e-8), (float)g.GetOption("weight_decay", 0.0), step);
        if (ran) slot["step"].IntValue = step;
        return ran;
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var u = slot["exp_inf"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pm = m) fixed (float* pu = u)
                            FusedOptimizer.SparseAdamaxUpdate(pp, pIdx, pVal, pm, pu, sNnz, lr, b1, b2, eps, wd, step);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);

                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m, pu = u)
                            FusedOptimizer.AdaMaxUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, pu + start, count, lr, b1, b2, eps, step);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private static readonly string[] _stateNames = new[] { "step", "sum" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        int priorSteps = slot.TryGetValue("step", out var sv) ? sv.IntValue ?? 0 : 0;
        float clr = (float)g.LearningRate / (1f + priorSteps * (float)g.GetOption("lr_decay", 0.0));
        bool ran = GpuOptimizer.TryAdagradStep(parameter, gradient, DeviceSlot(slot, "sum"), clr,
            (float)g.GetOption("eps", 1e-10), (float)g.GetOption("weight_decay", 0.0));
        if (ran)
        {
            if (sv is null) slot["step"] = OptimizerStateValue.FromInt(priorSteps + 1);
            else sv.IntValue = priorSteps + 1;
        }
        return ran;
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float eps = (float)g.GetOption("eps", 1e-10);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            float lrDecay = (float)g.GetOption("lr_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var s = slot["sum"].Tensor!;
                // PyTorch: clr = lr / (1 + (step - 1)·lr_decay), step counted from 1. A state loaded from before the
                // step slot existed starts counting here.
                int priorSteps = (slot.TryGetValue("step", out var sv) ? sv.IntValue : 0) ?? 0;
                if (sv is null) slot["step"] = OptimizerStateValue.FromInt(priorSteps + 1);
                else sv.IntValue = priorSteps + 1;
                float clr = lr / (1f + priorSteps * lrDecay);

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* ps = s)
                            FusedOptimizer.SparseAdagradUpdate(pp, pIdx, pVal, ps, sNnz, lr, eps, wd, lrDecay, priorSteps);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);
                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, ps = s)
                            FusedOptimizer.AdagradUpdateSimd(pp + start, pg + grad.Offset + start, ps + start, count, clr, eps);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        // The device kernel is the plain form; centered and momentum variants run on the host.
        if (g.GetOption("centered", 0.0) != 0.0 || g.GetOption("momentum", 0.0) != 0.0) return false;
        var slot = DeviceState(gi, pi, parameter.Length);
        return GpuOptimizer.TryRmspropStep(parameter, gradient, DeviceSlot(slot, "square_avg"), (float)g.LearningRate,
            (float)g.GetOption("alpha", 0.99), (float)g.GetOption("eps", 1e-8), (float)g.GetOption("weight_decay", 0.0));
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var v = slot["square_avg"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    var gAvgSp = centered ? slot["grad_avg"].Tensor! : null;
                    var mbSp = (mom != 0f) ? slot["momentum_buffer"].Tensor! : null;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* pv = v)
                        fixed (float* pga = gAvgSp) fixed (float* pmb = mbSp)
                        {
                            FusedOptimizer.SparseRmsPropUpdate(pp, pIdx, pVal, pv, pmb, sNnz, lr, rho, eps, wd, mom, centered, pga, mom != 0f);
                        }
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);

                if (centered)
                {
                    var gAvg = slot["grad_avg"].Tensor!;
                    if (mom == 0f)
                    {
                        ForEachChunk(p.Length, (start, count) =>
                        {
                            unsafe
                            {
                                fixed (float* pp = p, pg = grad.Array, pv = v, pga = gAvg)
                                    FusedOptimizer.RMSpropCenteredUpdate(pp + start, pg + grad.Offset + start, pv + start, pga + start, count, lr, rho, eps);
                            }
                        });
                    }
                    else
                    {
                        // Centered + Polyak momentum: maintain mom_buf = mom·mom_buf + g/(sqrt(var)+eps); p -= lr·mom_buf
                        var mb = slot["momentum_buffer"].Tensor!;
                        ForEachChunk(p.Length, (start, count) =>
                        {
                            float[] ga = grad.Array;
                            int go = grad.Offset;
                            for (int i = start; i < start + count; i++)
                            {
                                float gi_ = ga[go + i];
                                v[i] = rho * v[i] + (1f - rho) * gi_ * gi_;
                                gAvg[i] = rho * gAvg[i] + (1f - rho) * gi_;
                                float variance = v[i] - gAvg[i] * gAvg[i];
                                if (variance < 0f) variance = 0f;
                                float scaled = gi_ / (MathF.Sqrt(variance) + eps);
                                mb[i] = mom * mb[i] + scaled;
                                p[i] -= lr * mb[i];
                            }
                        });
                    }
                }
                else if (mom != 0f)
                {
                    var mb = slot["momentum_buffer"].Tensor!;
                    ForEachChunk(p.Length, (start, count) =>
                    {
                        float[] ga = grad.Array;
                        int go = grad.Offset;
                        for (int i = start; i < start + count; i++)
                        {
                            float gi_ = ga[go + i];
                            v[i] = rho * v[i] + (1f - rho) * gi_ * gi_;
                            float scaled = gi_ / (MathF.Sqrt(v[i]) + eps);
                            mb[i] = mom * mb[i] + scaled;
                            p[i] -= lr * mb[i];
                        }
                    });
                }
                else
                {
                    ForEachChunk(p.Length, (start, count) =>
                    {
                        unsafe
                        {
                            fixed (float* pp = p, pg = grad.Array, pv = v)
                                FusedOptimizer.RMSpropUpdateSimd(pp + start, pg + grad.Offset + start, pv + start, count, lr, rho, eps);
                        }
                    });
                }
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        // The device kernel applies the update unscaled, i.e. lr = 1 (the paper's and PyTorch's default).
        if (g.LearningRate != 1.0) return false;
        var slot = DeviceState(gi, pi, parameter.Length);
        return GpuOptimizer.TryAdadeltaStep(parameter, gradient, DeviceSlot(slot, "square_avg"), DeviceSlot(slot, "acc_delta"),
            (float)g.GetOption("rho", 0.9), (float)g.GetOption("eps", 1e-6), (float)g.GetOption("weight_decay", 0.0));
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float rho = (float)g.GetOption("rho", 0.9);
            float eps = (float)g.GetOption("eps", 1e-6);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var sq = slot["square_avg"].Tensor!;
                var ad = slot["acc_delta"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* psq = sq) fixed (float* pad = ad)
                            FusedOptimizer.SparseAdaDeltaUpdate(pp, pIdx, pVal, psq, pad, sNnz, lr, rho, eps, wd);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p, wd);
                ForEachChunk(p.Length, (start, count) =>
                {
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, psq = sq, pad = ad)
                            FusedOptimizer.AdaDeltaUpdateSimd(pp + start, pg + grad.Offset + start, psq + start, pad + start, count, lr, rho, eps);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        var slot = DeviceState(gi, pi, parameter.Length);
        return GpuOptimizer.TryLionStep(parameter, gradient, DeviceSlot(slot, "exp_avg"), (float)g.LearningRate,
            (float)g.GetOption("beta1", 0.9), (float)g.GetOption("beta2", 0.99), (float)g.GetOption("weight_decay", 0.0));
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.99);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var m = slot["exp_avg"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* pm = m)
                            FusedOptimizer.SparseLionUpdate(pp, pIdx, pVal, pm, sNnz, lr, b1, b2, wd);
                    }
                    continue;
                }

                ForEachChunk(p.Length, (start, count) =>
                {
                    var grad = DenseGradient(gi, pi, p);
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pm = m)
                            FusedOptimizer.LionUpdateSimd(pp + start, pg + grad.Offset + start, pm + start, count, lr, b1, b2, wd);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
        BeginStep();
        try {
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
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

                // True sparse path with bounded-drift approximation: ax[i] updates only at
                // touched indices. The averaged-weight buffer for untouched i diverges from
                // the exact Polyak-Ruppert average by at most |p[i] - ax[i]_at_last_touch|,
                // which is small for embedding rows that change incrementally. Exact lazy
                // averaging would require maintaining cumulative-product tables for the
                // time-varying mu_k schedule and per-index last-touch step — accepted as a
                // future optimization. For the embedding-table use case this is the right
                // trade.
                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* pax = ax)
                            FusedOptimizer.SparseASGDUpdate(pp, pIdx, pVal, pax, sNnz, eta, lambd, wd, mu);
                    }
                    continue;
                }

                ForEachChunk(p.Length, (start, count) =>
                {
                    var grad = DenseGradient(gi, pi, p);
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pax = ax)
                            FusedOptimizer.ASGDUpdateSimd(pp + start, pg + grad.Offset + start, pax + start, count, eta, lambd, alpha, wd, mu);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
        BeginStep();
        try {
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
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

                // Sparse-history semantics: at touched indices do sign comparison + step
                // adaptation + param step; at untouched indices leave (prev_grad, step_size)
                // unchanged. This preserves per-index step-size adaptation across gaps when
                // an embedding row is rarely touched — a deliberate departure from dense
                // Rprop (which would zero prev_grad at every untouched index) because that
                // is the correct behavior for sparse use cases.
                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* ppr = prev) fixed (float* pss = ss)
                            FusedOptimizer.SparseRpropUpdate(pp, pIdx, pVal, ppr, pss, sNnz, etaP, etaM, sMin, sMax);
                    }
                    continue;
                }

                ForEachChunk(p.Length, (start, count) =>
                {
                    var grad = DenseGradient(gi, pi, p);
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, ppr = prev, pss = ss)
                            FusedOptimizer.RpropUpdate(pp + start, pg + grad.Offset + start, ppr + start, pss + start, count, etaP, etaM, sMin, sMax);
                    }
                });
            }
        }
        } finally { EndStep(); }
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
        BeginStep();
        try {
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                // True sparse path: ‖p‖₂ via one full-param reduction; Adam math +
                // ‖update‖₂² collected only at touched indices; scatter scaled-update.
                // Since untouched indices have zero update, ‖update_sparse‖₂ = ‖update_dense‖₂
                // so the trust ratio matches dense exactly.
                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pm = m) fixed (float* pvv = v)
                            FusedOptimizer.SparseLAMBUpdate(pp, pIdx, pVal, pm, pvv, p.Length, sNnz, lr, b1, b2, eps, wd, step);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p);
                unsafe
                {
                    fixed (float* pp = p, pg = grad.Array, pm = m, pv = v)
                        FusedOptimizer.LAMBUpdateSimd(pp, pg + grad.Offset, pm, pv, p.Length, lr, b1, b2, eps, wd, step);
                }
            }
        }
        } finally { EndStep(); }
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
        ["eps"] = 1e-8,
    };
    private static readonly string[] _stateNames = new[] { "momentum_buffer" };
    /// <inheritdoc />
    protected override IReadOnlyDictionary<string, double> Defaults => _defaults;
    /// <inheritdoc />
    protected override IReadOnlyList<string> StateNames => _stateNames;

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float mom = (float)g.GetOption("momentum", 0.9);
            float wd = (float)g.GetOption("weight_decay", 0.0);
            float tc = (float)g.GetOption("trust_coeff", 1e-3);
            float eps = (float)g.GetOption("eps", 1e-8);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var v = slot["momentum_buffer"].Tensor!;

                // True sparse path: ‖p‖₂ via full-param reduction; ‖g‖₂² over touched
                // values (= dense ‖g‖₂² since untouched g = 0); scatter velocity + param
                // updates at touched indices only.
                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal) fixed (float* pv = v)
                            FusedOptimizer.SparseLARSUpdate(pp, pIdx, pVal, pv, p.Length, sNnz, lr, mom, wd, tc, eps);
                    }
                    continue;
                }

                var grad = DenseGradient(gi, pi, p);
                unsafe
                {
                    fixed (float* pp = p, pg = grad.Array, pv = v)
                        FusedOptimizer.LARSUpdateSimd(pp, pg + grad.Offset, pv, p.Length, lr, mom, wd, tc, eps);
                }
            }
        }
        } finally { EndStep(); }
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
    private protected override bool TryStepOnDevice(int gi, int pi, Tensor<float> parameter, Tensor<float> gradient)
    {
        var g = ParamGroups[gi];
        // The device kernel is the sqrt(n) schedule, lr_power = -0.5; other powers run on the host.
        if (g.GetOption("lr_power", -0.5) != -0.5) return false;
        var slot = DeviceState(gi, pi, parameter.Length);
        return GpuOptimizer.TryFtrlStep(parameter, gradient, DeviceSlot(slot, "z"), DeviceSlot(slot, "n"), (float)g.LearningRate,
            (float)g.GetOption("l1_reg", 0.0), (float)g.GetOption("l2_reg", 0.0), 0f);
    }

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
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
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                var z = slot["z"].Tensor!;
                var n = slot["n"].Tensor!;

                if (TryGetSparseGradient(gi, pi, out var sIdx, out var sVal, out var sNnz))
                {
                    if (sNnz == 0) continue;
                    unsafe
                    {
                        fixed (float* pp = p) fixed (int* pIdx = sIdx) fixed (float* pVal = sVal)
                        fixed (float* pn = n) fixed (float* pz = z)
                            FusedOptimizer.SparseFtrlUpdate(pp, pIdx, pVal, pn, pz, sNnz, lr, lrPow, l1, l2);
                    }
                    continue;
                }

                ForEachChunk(p.Length, (start, count) =>
                {
                    var grad = DenseGradient(gi, pi, p);
                    unsafe
                    {
                        fixed (float* pp = p, pg = grad.Array, pz = z, pn = n)
                            FusedOptimizer.FTRLUpdateSimd(pp + start, pg + grad.Offset + start, pz + start, pn + start, count, lr, l1, l2, lrPow);
                    }
                });
            }
        }
        } finally { EndStep(); }
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

    // Per-(gi, pi) scratch buffers reused across steps to avoid the per-step int[] / float[]
    // allocation that previously dominated GC pressure on large sparse tensors. Buffers grow
    // monotonically to fit the largest nnz seen so far.
    private readonly Dictionary<(int gi, int pi), (int[] idx, float[] val)> _scratch = new();

    /// <inheritdoc />
    public override void Step()
    {
        BeginStep();
        try {
        for (int gi = 0; gi < ParamGroups.Count; gi++)
        {
            var g = ParamGroups[gi];
            float lr = (float)g.LearningRate;
            float b1 = (float)g.GetOption("beta1", 0.9);
            float b2 = (float)g.GetOption("beta2", 0.999);
            float eps = (float)g.GetOption("eps", 1e-8);
            for (int pi = 0; pi < g.Parameters.Count; pi++)
            {
                if (!ShouldStep(gi, pi) || StepOnDevice(gi, pi)) continue;
                float[] p = HostParameter(gi, pi);
                var slot = GetOrCreateState(gi, pi, p.Length);
                int step = (slot["step"].IntValue ?? 0) + 1;
                slot["step"].IntValue = step;
                var m = slot["exp_avg"].Tensor!;
                var v = slot["exp_avg_sq"].Tensor!;

                int nnz;
                int[] idx;
                float[] val;
                if (TryGetSparseGradient(gi, pi, out idx!, out val!, out nnz))
                {
                    // Caller supplied sparse indices+values directly via the base
                    // OptimizerBase.SetSparseGradient API — zero scans, zero allocations.
                }
                else
                {
                    var grad = DenseGradient(gi, pi, p).Span;
                    // Build the sparse view in a single pass over the dense gradient, growing
                    // the per-parameter scratch buffers in place rather than allocating each step.
                    if (!_scratch.TryGetValue((gi, pi), out var pair))
                    {
                        pair = (new int[Math.Min(p.Length, 16)], new float[Math.Min(p.Length, 16)]);
                        _scratch[(gi, pi)] = pair;
                    }
                    int cap = pair.idx.Length;
                    int k = 0;
                    for (int i = 0; i < grad.Length; i++)
                    {
                        if (grad[i] == 0f) continue;
                        if (k == cap)
                        {
                            // Geometric growth — amortised O(1) per insertion.
                            int newCap = Math.Min(cap * 2, p.Length);
                            Array.Resize(ref pair.idx, newCap);
                            Array.Resize(ref pair.val, newCap);
                            cap = newCap;
                        }
                        pair.idx[k] = i;
                        pair.val[k] = grad[i];
                        k++;
                    }
                    _scratch[(gi, pi)] = pair;
                    if (k == 0) continue;
                    idx = pair.idx;
                    val = pair.val;
                    nnz = k;
                }

                unsafe
                {
                    fixed (float* pp = p) fixed (int* pi2 = idx) fixed (float* pv2 = val)
                    fixed (float* pm = m) fixed (float* pvv = v)
                        FusedOptimizer.SparseAdamUpdate(pp, pi2, pv2, pm, pvv, nnz, lr, b1, b2, eps, step);
                }
            }
        }
        } finally { EndStep(); }
    }
}
