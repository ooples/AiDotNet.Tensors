// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> GruSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, bool returnSequences = false)
        => GruSequenceForward(input, h0, wIh, wHh, bIh, bHh, out _, returnSequences);

    /// <inheritdoc/>
    public virtual Tensor<T> GruSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, out Tensor<T> finalHidden, bool returnSequences = false)
    {
        var (batch, seqLen, _, hidden) = ValidateGruSequence(input, h0, wIh, wHh, bIh, bHh);
        if (seqLen == 0)
        {
            finalHidden = h0 ?? new Tensor<T>(new[] { batch, hidden });
            return returnSequences ? new Tensor<T>(new[] { batch, 0, hidden }) : finalHidden;
        }

        Tensor<T> sequence;
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            // One lazy node over the whole sequence; compiled training differentiates it by replaying the eager op
            // under a tape, which records the fused node below.
            scope.BindEngineIfUnset(this);
            var inputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var present);
            sequence = scope.RecordVariadic(LazyNodeType.Custom, "GruSequenceForward", inputs,
                new[] { batch, seqLen, hidden },
                (eng, output) =>
                {
                    var (x, h, wi, wh, bi, bh) = UnpackGruSequenceInputs(inputs, present);
                    var eager = eng.GruSequenceForward(x, h, wi, wh, bi, bh, returnSequences: true);
                    DirectGpuTensorEngine.CopyResultInto(eng, eager, output);
                },
                backwardFn: BackwardFunctions<T>.ReplayUnderTape(eng =>
                {
                    var (x, h, wi, wh, bi, bh) = UnpackGruSequenceInputs(inputs, present);
                    return eng.GruSequenceForward(x, h, wi, wh, bi, bh, returnSequences: true);
                }));
        }
        else
        {
            sequence = GruSequenceRecorded(input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden);
        }

        return GruSequenceOutputs(sequence, batch, seqLen, hidden, returnSequences, out finalHidden);
    }

    /// <summary>
    /// The op's outputs from the whole sequence: the last step and the final state are recorded views of it, so their
    /// gradients reach the sequence's node.
    /// </summary>
    private protected Tensor<T> GruSequenceOutputs<T>(Tensor<T> sequence, int batch, int seqLen, int hidden,
        bool returnSequences, out Tensor<T> finalHidden)
    {
        var last = Reshape(TensorNarrow(sequence, 1, seqLen - 1, 1), new[] { batch, hidden });
        finalHidden = last;
        return returnSequences ? sequence : last;
    }

    /// <summary>The whole sequence [B, T, H] on the host, recorded as one tape node whose backward is the fused BPTT.</summary>
    private protected Tensor<T> GruSequenceRecorded<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh,
        Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh, int batch, int seqLen, int hidden)
    {
        GruSequenceCache cache;
        Tensor<T> sequence;
        using (new NoGradScope<T>())
            sequence = GruForwardOnHost(input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden, out cache);
        if (DifferentiableOps.IsRecording<T>())
        {
            var inputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var present);
            DifferentiableOps.RecordIfActive("GruSequenceForward", sequence, inputs, GruSequenceBackwardOnHost<T>,
                new object[] { cache, present });
        }
        return sequence;
    }

    private protected static (int Batch, int SeqLen, int Input, int Hidden) ValidateGruSequence<T>(Tensor<T> input, Tensor<T>? h0,
        Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (wIh is null) throw new ArgumentNullException(nameof(wIh));
        if (wHh is null) throw new ArgumentNullException(nameof(wHh));
        if (input.Rank != 3)
            throw new ArgumentException($"GruSequenceForward expects rank-3 input [B, seq, in]; got rank {input.Rank}.", nameof(input));
        if (wIh.Rank != 2 || wHh.Rank != 2) throw new ArgumentException("wIh and wHh must be rank-2.");
        int batch = input.Shape[0], seqLen = input.Shape[1], inFeatures = input.Shape[2];
        int gateRows = wIh.Shape[0];
        if (gateRows % 3 != 0 || gateRows == 0)
            throw new ArgumentException($"wIh first dim must be 3*hidden (gates r, z, n); got {gateRows}.", nameof(wIh));
        int hidden = gateRows / 3;
        if (wIh.Shape[1] != inFeatures)
            throw new ArgumentException($"wIh.Shape[1] ({wIh.Shape[1]}) must equal input feature count ({inFeatures}).", nameof(wIh));
        if (wHh.Shape[0] != gateRows || wHh.Shape[1] != hidden)
            throw new ArgumentException($"wHh must be [{gateRows}, {hidden}]; got [{wHh.Shape[0]}, {wHh.Shape[1]}].", nameof(wHh));
        if (h0 is not null && (h0.Rank != 2 || h0.Shape[0] != batch || h0.Shape[1] != hidden))
            throw new ArgumentException($"h0 must be [{batch}, {hidden}].", nameof(h0));
        if (bIh is not null && (bIh.Rank != 1 || bIh.Shape[0] != gateRows))
            throw new ArgumentException($"bIh must be [{gateRows}].", nameof(bIh));
        if (bHh is not null && (bHh.Rank != 1 || bHh.Shape[0] != gateRows))
            throw new ArgumentException($"bHh must be [{gateRows}].", nameof(bHh));
        return (batch, seqLen, inFeatures, hidden);
    }

    /// <summary>Which optional GRU inputs a recorded node carries, in the order input, h0, wIh, wHh, bIh, bHh.</summary>
    [Flags]
    private protected enum GruOptionalInputs { None = 0, InitialHidden = 1, InputBias = 2, RecurrentBias = 4 }

    private protected static Tensor<T>[] GruSequenceInputs<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, out GruOptionalInputs present)
    {
        present = GruOptionalInputs.None;
        var list = new List<Tensor<T>> { input };
        if (h0 is not null) { present |= GruOptionalInputs.InitialHidden; list.Add(h0); }
        list.Add(wIh); list.Add(wHh);
        if (bIh is not null) { present |= GruOptionalInputs.InputBias; list.Add(bIh); }
        if (bHh is not null) { present |= GruOptionalInputs.RecurrentBias; list.Add(bHh); }
        return list.ToArray();
    }

    private protected static (Tensor<T> Input, Tensor<T>? H0, Tensor<T> WIh, Tensor<T> WHh, Tensor<T>? BIh, Tensor<T>? BHh)
        UnpackGruSequenceInputs<T>(Tensor<T>[] inputs, GruOptionalInputs present)
    {
        int k = 0;
        var input = inputs[k++];
        var h0 = (present & GruOptionalInputs.InitialHidden) != 0 ? inputs[k++] : null;
        var wIh = inputs[k++];
        var wHh = inputs[k++];
        var bIh = (present & GruOptionalInputs.InputBias) != 0 ? inputs[k++] : null;
        var bHh = (present & GruOptionalInputs.RecurrentBias) != 0 ? inputs[k++] : null;
        return (input, h0, wIh, wHh, bIh, bHh);
    }

    /// <summary>The forward's saved activations, batch-major: allH [B, T + 1, H], gates [B, T, 3H] (r, z, n), ghn [B, T, H].</summary>
    private protected sealed class GruSequenceCache
    {
        internal double[] AllH = Array.Empty<double>();
        internal double[] Gates = Array.Empty<double>();
        internal double[] Ghn = Array.Empty<double>();
        internal int Batch, SeqLen, Input, Hidden;
    }

    private static double[] ToDoubles<T>(Tensor<T> t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var a = t.ToArray();
        var d = new double[a.Length];
        for (int i = 0; i < a.Length; i++) d[i] = ops.ToDouble(a[i]);
        return d;
    }

    private static Tensor<T> FromDoubles<T>(double[] d, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var a = new T[d.Length];
        for (int i = 0; i < d.Length; i++) a[i] = ops.FromDouble(d[i]);
        return new Tensor<T>(a, shape);
    }

    // x [B, T, In] -> input-side sums [B * T, 3H] in one GEMM; the recurrent sums take one [B, H] x [H, 3H] GEMM per step.
    private Tensor<T> GruForwardOnHost<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh,
        Tensor<T>? bHh, int batch, int seqLen, int hidden, out GruSequenceCache cache)
    {
        int inFeatures = input.Shape[2], g3 = 3 * hidden;
        var gi = ToDoubles(TensorMatMul(Reshape(input, new[] { batch * seqLen, inFeatures }), TensorTranspose(wIh)));
        var whhT = TensorTranspose(wHh);
        var bi = bIh is null ? new double[g3] : ToDoubles(bIh);
        var bh = bHh is null ? new double[g3] : ToDoubles(bHh);
        var allH = new double[batch * (seqLen + 1) * hidden];
        var gates = new double[batch * seqLen * g3];
        var ghn = new double[batch * seqLen * hidden];
        var y = new double[batch * seqLen * hidden];
        var h = h0 is null ? new double[batch * hidden] : ToDoubles(h0);
        for (int b = 0; b < batch; b++) Array.Copy(h, b * hidden, allH, b * (seqLen + 1) * hidden, hidden);

        for (int t = 0; t < seqLen; t++)
        {
            var gh = ToDoubles(TensorMatMul(FromDoubles<T>(h, new[] { batch, hidden }), whhT));
            for (int b = 0; b < batch; b++)
            {
                int iRow = (b * seqLen + t) * g3, hRow = b * g3, gRow = (b * seqLen + t) * g3;
                for (int j = 0; j < hidden; j++)
                {
                    double r = Sigmoid(gi[iRow + j] + bi[j] + gh[hRow + j] + bh[j]);
                    double z = Sigmoid(gi[iRow + hidden + j] + bi[hidden + j] + gh[hRow + hidden + j] + bh[hidden + j]);
                    double hn = gh[hRow + 2 * hidden + j] + bh[2 * hidden + j];
                    double n = Math.Tanh(gi[iRow + 2 * hidden + j] + bi[2 * hidden + j] + r * hn);
                    double hv = (1 - z) * n + z * h[b * hidden + j];
                    gates[gRow + j] = r; gates[gRow + hidden + j] = z; gates[gRow + 2 * hidden + j] = n;
                    ghn[(b * seqLen + t) * hidden + j] = hn;
                    y[(b * seqLen + t) * hidden + j] = hv;
                    allH[(b * (seqLen + 1) + t + 1) * hidden + j] = hv;
                }
            }
            for (int b = 0; b < batch; b++) Array.Copy(allH, (b * (seqLen + 1) + t + 1) * hidden, h, b * hidden, hidden);
        }

        cache = new GruSequenceCache
        {
            AllH = allH, Gates = gates, Ghn = ghn, Batch = batch, SeqLen = seqLen, Input = inFeatures, Hidden = hidden,
        };
        return FromDoubles<T>(y, new[] { batch, seqLen, hidden });
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // Backpropagation through time (see BatchExecutionCudaKernels.GruSource for the same equations on the device). The
    // input-side and recurrent-side pre-activation gradients dgi / dgh [B, T, 3H] feed four GEMMs at the end.
    private static void GruSequenceBackwardOnHost<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var cache = (GruSequenceCache)savedState[0];
        var present = (GruOptionalInputs)savedState[1];
        var (input, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);
        int batch = cache.Batch, seqLen = cache.SeqLen, hidden = cache.Hidden, inFeatures = cache.Input, g3 = 3 * hidden;
        var gy = ToDoubles(gradOutput);
        var dgi = new double[batch * seqLen * g3];
        var dgh = new double[batch * seqLen * g3];
        var dh = new double[batch * hidden];

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var stepDgh = new double[batch * g3];
            for (int b = 0; b < batch; b++)
            {
                int gRow = (b * seqLen + t) * g3;
                for (int j = 0; j < hidden; j++)
                {
                    double r = cache.Gates[gRow + j], z = cache.Gates[gRow + hidden + j], n = cache.Gates[gRow + 2 * hidden + j];
                    double hn = cache.Ghn[(b * seqLen + t) * hidden + j];
                    double hp = cache.AllH[(b * (seqLen + 1) + t) * hidden + j];
                    double dhv = gy[(b * seqLen + t) * hidden + j] + dh[b * hidden + j];
                    double dnp = dhv * (1 - z) * (1 - n * n);
                    double drp = dnp * hn * r * (1 - r);
                    double dzp = dhv * (hp - n) * z * (1 - z);
                    dgi[gRow + j] = drp; dgi[gRow + hidden + j] = dzp; dgi[gRow + 2 * hidden + j] = dnp;
                    dgh[gRow + j] = drp; dgh[gRow + hidden + j] = dzp; dgh[gRow + 2 * hidden + j] = dnp * r;
                    stepDgh[b * g3 + j] = drp; stepDgh[b * g3 + hidden + j] = dzp; stepDgh[b * g3 + 2 * hidden + j] = dnp * r;
                    dh[b * hidden + j] = dhv * z;   // the direct term; the recurrent one is added below
                }
            }
            Tensor<T> recurrent;
            using (new NoGradScope<T>())
                recurrent = engine.TensorMatMul(FromDoubles<T>(stepDgh, new[] { batch, g3 }), wHh);
            var rec = ToDoubles(recurrent);
            for (int k = 0; k < dh.Length; k++) dh[k] += rec[k];
        }

        using (new NoGradScope<T>())
        {
            var dgiT = FromDoubles<T>(dgi, new[] { batch * seqLen, g3 });
            var dghT = FromDoubles<T>(dgh, new[] { batch * seqLen, g3 });
            DifferentiableOps.AccumulateGrad(grads, input,
                engine.Reshape(engine.TensorMatMul(dgiT, wIh), new[] { batch, seqLen, inFeatures }), engine);
            if (h0 is not null) DifferentiableOps.AccumulateGrad(grads, h0, FromDoubles<T>(dh, new[] { batch, hidden }), engine);
            DifferentiableOps.AccumulateGrad(grads, wIh, engine.TensorMatMul(engine.TensorTranspose(dgiT),
                engine.Reshape(input, new[] { batch * seqLen, inFeatures })), engine);
            // h_{t-1} for every (b, t), batch-major like dgh.
            var prev = new double[batch * seqLen * hidden];
            for (int b = 0; b < batch; b++)
                Array.Copy(cache.AllH, b * (seqLen + 1) * hidden, prev, b * seqLen * hidden, seqLen * hidden);
            DifferentiableOps.AccumulateGrad(grads, wHh, engine.TensorMatMul(engine.TensorTranspose(dghT),
                FromDoubles<T>(prev, new[] { batch * seqLen, hidden })), engine);
            if (bIh is not null) DifferentiableOps.AccumulateGrad(grads, bIh, ColumnSums<T>(dgi, batch * seqLen, g3), engine);
            if (bHh is not null) DifferentiableOps.AccumulateGrad(grads, bHh, ColumnSums<T>(dgh, batch * seqLen, g3), engine);
        }
    }

    private static Tensor<T> ColumnSums<T>(double[] rows, int count, int width)
    {
        var sums = new double[width];
        for (int r = 0; r < count; r++)
            for (int c = 0; c < width; c++) sums[c] += rows[r * width + c];
        return FromDoubles<T>(sums, new[] { width });
    }
}
