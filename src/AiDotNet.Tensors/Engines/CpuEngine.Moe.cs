// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using MoeTemplate = AiDotNet.Tensors.Engines.DirectGpu.MoeKernelTemplate;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The fused SwiGLU mixture-of-experts layer: routing, the expert projections, and the gated combine as one op with one
/// tape node.
/// </summary>
public static class MixtureOfExperts
{
    /// <summary>
    /// <c>y[t] = sum over the top-k experts e of t: gate[t, e] * (silu(x[t] w1[e]) * (x[t] w3[e])) w2[e]</c>.
    /// </summary>
    /// <param name="engine">The engine; a DirectGpu engine runs every step on the device without a host round trip.</param>
    /// <param name="x">Token rows [N, D].</param>
    /// <param name="scores">Routing scores [N, G] (for example sigmoid(x w_router)); differentiable through the gates.</param>
    /// <param name="selectionBias">Optional [G] bias added to the scores for expert selection only (auxiliary-loss-free
    /// load balancing); it never enters the gates and gets no gradient.</param>
    /// <param name="w1">Gate projections [G, D, F].</param>
    /// <param name="w3">Up projections [G, D, F].</param>
    /// <param name="w2">Down projections [G, F, D].</param>
    /// <param name="topK">Experts per token (1 to 16, at most G).</param>
    /// <param name="sequences">The token rows split evenly into this many sequences, for <paramref name="loads"/>.</param>
    /// <param name="loads">[sequences, G]: the tokens each sequence sent to each expert (not differentiable).</param>
    /// <returns>[N, D]. Gates are the selected scores divided by their sum; ties in selection go to the lower expert.
    /// Gradients flow to x, scores, w1, w3 and w2 (first order).</returns>
    public static Tensor<T> SwiGluExperts<T>(IEngine engine, Tensor<T> x, Tensor<T> scores, Tensor<T>? selectionBias,
        Tensor<T> w1, Tensor<T> w3, Tensor<T> w2, int topK, int sequences, out Tensor<T> loads)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        return (engine as CpuEngine ?? new CpuEngine()).MoeSwiGlu(x, scores, selectionBias, w1, w3, w2, topK, sequences, out loads);
    }
}

/// <summary>The tensors a fused MoE forward keeps for its backward.</summary>
internal sealed class MoeSaved
{
    internal MoeSaved(int n, int d, int g, int k, int f, int b) => Shape = new DirectGpu.MoeShape(n, d, g, k, f, b);

    internal DirectGpu.MoeShape Shape { get; }

    // Host forward: every intermediate in double.
    internal double[]? Gates, Ssum, H13, O;
    internal int[]? Ids, Perm, Pos, Ends;

    // Device forward: the same intermediates as resident float tensors (indices as floats).
    internal DirectGpu.IDirectGpuBackend? DeviceBackend;
    internal Tensor<float>? DIds, DGates, DSsum, DPerm, DPos, DEnds, DH13, DO;
}

public partial class CpuEngine
{
    internal static void ValidateMoe<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, out int n, out int d, out int g, out int f)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        if (w1 is null) throw new ArgumentNullException(nameof(w1));
        if (w3 is null) throw new ArgumentNullException(nameof(w3));
        if (w2 is null) throw new ArgumentNullException(nameof(w2));
        if (x.Rank != 2 || scores.Rank != 2 || w1.Rank != 3 || w3.Rank != 3 || w2.Rank != 3)
            throw new ArgumentException("Expected x [N, D], scores [N, G], w1/w3 [G, D, F] and w2 [G, F, D].");
        n = x._shape[0]; d = x._shape[1]; g = w1._shape[0]; f = w1._shape[2];
        if (scores._shape[0] != n || scores._shape[1] != g) throw new ArgumentException($"scores must be [{n}, {g}].", nameof(scores));
        if (w1._shape[1] != d || w3._shape[0] != g || w3._shape[1] != d || w3._shape[2] != f)
            throw new ArgumentException($"w1 and w3 must be [{g}, {d}, {f}].", nameof(w3));
        if (w2._shape[0] != g || w2._shape[1] != f || w2._shape[2] != d)
            throw new ArgumentException($"w2 must be [{g}, {f}, {d}].", nameof(w2));
        if (bias is not null && (bias.Length != g)) throw new ArgumentException($"selectionBias must hold {g} values.", nameof(bias));
        if (topK < 1 || topK > Math.Min(g, MoeTemplate.MaxTopK))
            throw new ArgumentOutOfRangeException(nameof(topK), $"topK must be in [1, {Math.Min(g, MoeTemplate.MaxTopK)}].");
        if (sequences < 1 || n % sequences != 0)
            throw new ArgumentOutOfRangeException(nameof(sequences), "sequences must divide the token rows.");
    }

    /// <summary>The fused MoE forward (<see cref="MixtureOfExperts.SwiGluExperts"/>); the host computes in double.</summary>
    internal Tensor<T> MoeSwiGlu<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, out Tensor<T> loads)
    {
        // In a compiled training trace the op is one node that re-runs every step: routing, loads and output follow
        // each step's inputs and the current selection bias. (Run eagerly at trace time instead, it froze into the
        // plan as a constant - the capture batch's output and loads every step, and no gradient to its inputs.) The
        // node's backward reads the intermediates its own forward kept that step, so it costs no second forward.
        var slot = new MoeSavedSlot<T>();
        if (TryRecordCompiledOp<T, Tensor<T>>("MoeSwiGlu", new[] { x, scores, bias, w1, w3, w2 },
                eng =>
                {
                    var y = (eng as CpuEngine ?? this).MoeSwiGluForward(x, scores, bias, w1, w3, w2, topK, sequences,
                        keepSaved: true, out slot.Loads, out slot.Saved);
                    return (y, slot.Loads);
                },
                static (eng, fresh, held) => { },
                out var recorded, out var tracedLoads,
                backward: (gradOutput, inputs, output, savedState, engine, grads) =>
                    AccumulateMoeGradients(gradOutput, x, scores, w1, w3, w2, slot.Saved!, engine, grads))
            && GraphMode.Current is { } scope)
        {
            // The loads are a node of their own, downstream of the op: a consumer (the selection-bias update) then
            // runs after each step's routing and reads that step's loads. As a bare tensor they had no producer in the
            // graph, so nothing ordered their readers after the op.
            loads = scope.RecordMaterializedUnary(LazyNodeType.Custom, "MoeSwiGluLoads", recorded,
                (Tensor<T>)tracedLoads.CloneDeepCopy(),
                (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, slot.Loads!, output));
            return recorded;
        }

        var result = MoeSwiGluForward(x, scores, bias, w1, w3, w2, topK, sequences,
            keepSaved: DifferentiableOps.IsRecording<T>(), out loads, out var saved);
        if (saved is not null) RecordMoe(result, x, scores, w1, w3, w2, saved);
        return result;
    }

    /// <summary>The intermediates of the latest replay of a recorded MoE node, for that node's backward.</summary>
    private sealed class MoeSavedSlot<T>
    {
        internal MoeSaved? Saved;
        internal Tensor<T>? Loads;
    }

    /// <summary>
    /// The fused MoE forward without recording: returns the output and the loads, and (when <paramref name="keepSaved"/>)
    /// the intermediates its backward needs. The host computes in double.
    /// </summary>
    internal virtual Tensor<T> MoeSwiGluForward<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, bool keepSaved, out Tensor<T> loads, out MoeSaved? saved)
    {
        var output = MoeSwiGluHostForward(x, scores, bias, w1, w3, w2, topK, sequences, out loads, out var hostSaved);
        saved = keepSaved ? hostSaved : null;
        return output;
    }

    /// <summary>The host forward, also returning what it keeps for the backward.</summary>
    internal Tensor<T> MoeSwiGluHost<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, out Tensor<T> loads, out MoeSaved saved)
    {
        var output = MoeSwiGluHostForward(x, scores, bias, w1, w3, w2, topK, sequences, out loads, out saved);
        RecordMoe(output, x, scores, w1, w3, w2, saved);
        return output;
    }

    private Tensor<T> MoeSwiGluHostForward<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, out Tensor<T> loads, out MoeSaved saved)
    {
        ValidateMoe(x, scores, bias, w1, w3, w2, topK, sequences, out int n, out int d, out int g, out int f);
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        var ops = MathHelper.GetNumericOperations<T>();
        double[] X = MoeHost(x), S = MoeHost(scores), W1 = MoeHost(w1), W3 = MoeHost(w3), W2 = MoeHost(w2);
        double[]? Bv = bias is null ? null : MoeHost(bias);

        RouteTokensHost(S, Bv, n, g, topK, out var ids, out var gates, out var ssum);
        SortSlotsByExpertHost(ids, n * topK, g, sequences, out var ends, out var perm, out var pos, out var load);
        ComputeSwiGluExpertProjectionsHost(X, W1, W3, W2, perm, ids, pos, gates, n, d, topK, f, out var h13, out var o, out var y);

        saved = new MoeSaved(n, d, g, topK, f, sequences)
        {
            Ids = ids, Gates = gates, Ssum = ssum, Perm = perm, Pos = pos, Ends = ends, H13 = h13, O = o,
        };
        loads = MoeTensor(load, new[] { sequences, g }, ops);
        return MoeTensor(y, new[] { n, d }, ops);
    }

    private static void RouteTokensHost(double[] S, double[]? Bv, int n, int g, int topK,
        out int[] ids, out double[] gates, out double[] ssum)
    {
        ids = new int[n * topK];
        gates = new double[n * topK];
        ssum = new double[n];
        var bv = new double[topK];
        var bi = new int[topK];
        for (int t = 0; t < n; t++)
            RouteSingleTokenHost(S, Bv, t, g, topK, bv, bi, ids, gates, ssum);
    }

    private static void RouteSingleTokenHost(double[] S, double[]? Bv, int t, int g, int topK,
        double[] bv, int[] bi, int[] ids, double[] gates, double[] ssum)
    {
        for (int j = 0; j < topK; j++) { bv[j] = double.NegativeInfinity; bi[j] = 0; }
        for (int e = 0; e < g; e++)
        {
            double v = S[t * g + e] + (Bv is null ? 0 : Bv[e]);
            if (v > bv[topK - 1])
            {
                int p = topK - 1;
                while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--; }
                bv[p] = v;
                bi[p] = e;
            }
        }
        FinalizeTokenGatesHost(S, t, g, topK, bi, ids, gates, ssum);
    }

    private static void FinalizeTokenGatesHost(double[] S, int t, int g, int topK,
        int[] bi, int[] ids, double[] gates, double[] ssum)
    {
        double s = 0;
        for (int j = 0; j < topK; j++) s += S[t * g + bi[j]];
        for (int j = 0; j < topK; j++)
        {
            ids[t * topK + j] = bi[j];
            gates[t * topK + j] = S[t * g + bi[j]] / s;
        }
        ssum[t] = s;
    }

    private static void SortSlotsByExpertHost(int[] ids, int nk, int g, int sequences,
        out int[] ends, out int[] perm, out int[] pos, out double[] load)
    {
        ends = new int[g];
        perm = new int[nk];
        pos = new int[nk];
        load = new double[sequences * g];
        foreach (int e in ids) ends[e]++;
        int sps = nk / sequences;
        for (int sl = 0; sl < nk; sl++) load[(sl / sps) * g + ids[sl]]++;
        var next = new int[g];
        for (int e = 1; e < g; e++)
        {
            ends[e] += ends[e - 1];
            next[e] = ends[e - 1];
        }
        for (int sl = 0; sl < nk; sl++) { int p = next[ids[sl]]++; perm[p] = sl; pos[sl] = p; }
    }

    private static void ComputeSwiGluExpertProjectionsHost(double[] X, double[] W1, double[] W3, double[] W2,
        int[] perm, int[] ids, int[] pos, double[] gates, int n, int d, int topK, int f,
        out double[] h13, out double[] o, out double[] y)
    {
        int nk = n * topK, f2 = 2 * f;
        h13 = new double[nk * f2];
        o = new double[nk * d];
        var act = new double[f];
        for (int r = 0; r < nk; r++)
            EvaluateSortedSlotHost(X, W1, W3, W2, perm[r] / topK, ids[perm[r]], r, d, f, act, h13, o);

        y = new double[n * d];
        for (int t = 0; t < n; t++)
            for (int j = 0; j < topK; j++)
            {
                int sl = t * topK + j, p = pos[sl];
                for (int c = 0; c < d; c++) y[t * d + c] += gates[sl] * o[p * d + c];
            }
    }

    private static void EvaluateSortedSlotHost(double[] X, double[] W1, double[] W3, double[] W2,
        int tok, int e, int r, int d, int f, double[] act, double[] h13, double[] o)
    {
        int f2 = 2 * f;
        for (int c = 0; c < f; c++)
        {
            double a1 = 0, a3 = 0;
            for (int q = 0; q < d; q++)
            {
                double xv = X[tok * d + q];
                a1 += xv * W1[(e * d + q) * f + c];
                a3 += xv * W3[(e * d + q) * f + c];
            }
            h13[r * f2 + c] = a1;
            h13[r * f2 + f + c] = a3;
            act[c] = Silu(a1) * a3;
        }
        for (int c = 0; c < d; c++)
        {
            double a = 0;
            for (int q = 0; q < f; q++) a += act[q] * W2[(e * f + q) * d + c];
            o[r * d + c] = a;
        }
    }

    private protected static void RecordMoe<T>(Tensor<T> output, Tensor<T> x, Tensor<T> scores, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, MoeSaved saved) =>
        DifferentiableOps.RecordIfActive("MoeSwiGlu", output, new[] { x, scores, w1, w3, w2 }, MoeSwiGluBackward<T>,
            new object[] { saved });

    internal static void MoeSwiGluBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var saved = (MoeSaved)savedState[0];
        var gradEngine = ResolveMoeGradEngine(engine, saved);
        var (dx, ds, dw1, dw3, dw2) = gradEngine.MoeSwiGluGradients(gradOutput, inputs[0], inputs[2], inputs[3], inputs[4], saved);
        var outs = new[] { dx, ds, dw1, dw3, dw2 };
        for (int i = 0; i < 5; i++)
            if (DifferentiableOps.IsGradientRequired(inputs[i])) DifferentiableOps.AccumulateGrad(grads, inputs[i], outs[i], engine);
    }

    // A compiled MoE node's backward: the plan keeps the gradients it needs (the selection bias gets none).
    private static void AccumulateMoeGradients<T>(Tensor<T> gradOutput, Tensor<T> x, Tensor<T> scores, Tensor<T> w1,
        Tensor<T> w3, Tensor<T> w2, MoeSaved saved, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var (dx, ds, dw1, dw3, dw2) = ResolveMoeGradEngine(engine, saved).MoeSwiGluGradients(gradOutput, x, w1, w3, w2, saved);
        DifferentiableOps.AccumulateGrad(grads, x, dx, engine);
        DifferentiableOps.AccumulateGrad(grads, scores, ds, engine);
        DifferentiableOps.AccumulateGrad(grads, w1, dw1, engine);
        DifferentiableOps.AccumulateGrad(grads, w3, dw3, engine);
        DifferentiableOps.AccumulateGrad(grads, w2, dw2, engine);
    }

    private static CpuEngine ResolveMoeGradEngine(IEngine tapeEngine, MoeSaved saved)
    {
        if (saved.DH13 is not null)
        {
            if (tapeEngine is DirectGpuTensorEngine tapeGpu && !tapeGpu.IsDisposed
                && (saved.DeviceBackend is null
                    || ReferenceEquals(tapeGpu.PlacementBackend, saved.DeviceBackend)
                    || ReferenceEquals(tapeGpu.QueryBackend, saved.DeviceBackend)))
                return tapeGpu;

            var owner = DirectGpuTensorEngine.EngineOwning(saved.DeviceBackend ?? saved.DH13._gpuBackend);
            if (owner is not null && !owner.IsDisposed)
                return owner;

            if (AiDotNetEngine.Current is DirectGpuTensorEngine currentGpu && !currentGpu.IsDisposed
                && (saved.DeviceBackend is null || ReferenceEquals(currentGpu.PlacementBackend, saved.DeviceBackend)))
                return currentGpu;

            if (tapeEngine is DirectGpuTensorEngine fallbackGpu)
                return fallbackGpu;
        }
        return tapeEngine as CpuEngine ?? new CpuEngine();
    }

    /// <summary>Gradients of the fused MoE (dx, dscores, dw1, dw3, dw2) from the saved forward; the host computes in double.</summary>
    internal virtual (Tensor<T> Dx, Tensor<T> Ds, Tensor<T> Dw1, Tensor<T> Dw3, Tensor<T> Dw2) MoeSwiGluGradients<T>(
        Tensor<T> dy, Tensor<T> x, Tensor<T> w1, Tensor<T> w3, Tensor<T> w2, MoeSaved saved)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var s = saved.Shape;
        int n = s.N, d = s.D, g = s.G, k = s.K, f = s.F, nk = s.NK, f2 = 2 * f;
        if (saved.H13 is null) throw new InvalidOperationException("The fused MoE forward ran on a device; its gradients need that engine.");
        double[] DY = MoeHost(dy), X = MoeHost(x), W1 = MoeHost(w1), W3 = MoeHost(w3), W2 = MoeHost(w2);
        var ids = saved.Ids!; var gates = saved.Gates!; var ssum = saved.Ssum!; var perm = saved.Perm!; var pos = saved.Pos!;
        var h13 = saved.H13; var o = saved.O!;

        var ds = new double[n * g];
        for (int t = 0; t < n; t++)
        {
            var dg = new double[k]; double dot = 0;
            for (int j = 0; j < k; j++)
            {
                int sl = t * k + j, p = pos[sl]; double a = 0;
                for (int c = 0; c < d; c++) a += o[p * d + c] * DY[t * d + c];
                dg[j] = a; dot += a * gates[sl];
            }
            for (int j = 0; j < k; j++) ds[t * g + ids[t * k + j]] = (dg[j] - dot) / ssum[t];
        }

        var dx = new double[n * d]; var dw1 = new double[g * d * f]; var dw3 = new double[g * d * f]; var dw2 = new double[g * f * d];
        var dact = new double[f]; var dh1 = new double[f]; var dh3 = new double[f];
        for (int r = 0; r < nk; r++)
        {
            int sl = perm[r], tok = sl / k, e = ids[sl];
            double gate = gates[sl];
            for (int c = 0; c < f; c++)
            {
                double a = 0;
                for (int q = 0; q < d; q++) a += gate * DY[tok * d + q] * W2[(e * f + c) * d + q];
                dact[c] = a;
                double h = h13[r * f2 + c], u = h13[r * f2 + f + c], sg = 1 / (1 + Math.Exp(-h));
                dh1[c] = a * u * sg * (1 + h * (1 - sg));
                dh3[c] = a * h * sg;
                double act = h * sg * u;
                for (int q = 0; q < d; q++) dw2[(e * f + c) * d + q] += act * gate * DY[tok * d + q];
            }
            for (int q = 0; q < d; q++)
            {
                double a = 0, xv = X[tok * d + q];
                for (int c = 0; c < f; c++)
                {
                    a += dh1[c] * W1[(e * d + q) * f + c] + dh3[c] * W3[(e * d + q) * f + c];
                    dw1[(e * d + q) * f + c] += xv * dh1[c];
                    dw3[(e * d + q) * f + c] += xv * dh3[c];
                }
                dx[tok * d + q] += a;
            }
        }
        return (MoeTensor(dx, new[] { n, d }, ops), MoeTensor(ds, new[] { n, g }, ops), MoeTensor(dw1, new[] { g, d, f }, ops),
            MoeTensor(dw3, new[] { g, d, f }, ops), MoeTensor(dw2, new[] { g, f, d }, ops));
    }

    private static double Silu(double h) => h / (1 + Math.Exp(-h));

    [ScalarKernel("The host reference computes in double; each element converts once on entry and once on exit.")]
    private static double[] MoeHost<T>(Tensor<T> t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var a = t.ToArray();
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = ops.ToDouble(a[i]);
        return r;
    }

    [ScalarKernel("The host reference computes in double; each element converts once on entry and once on exit.")]
    private static Tensor<T> MoeTensor<T>(double[] v, int[] shape, INumericOperations<T> ops)
    {
        var r = new Tensor<T>(shape);
        var data = r.GetDataArray();
        for (int i = 0; i < v.Length; i++) data[i] = ops.FromDouble(v[i]);
        return r;
    }
}
