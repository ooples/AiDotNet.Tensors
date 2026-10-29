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
        Tensor<T>? bIh, Tensor<T>? bHh, bool returnSequences = false, bool resetAfter = true)
        => GruSequenceForward(input, h0, wIh, wHh, bIh, bHh, out _, returnSequences, resetAfter);

    /// <inheritdoc/>
    public virtual Tensor<T> GruSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, out Tensor<T> finalHidden, bool returnSequences = false, bool resetAfter = true)
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
                    var eager = eng.GruSequenceForward(x, h, wi, wh, bi, bh, returnSequences: true, resetAfter: resetAfter);
                    DirectGpuTensorEngine.CopyResultInto(eng, eager, output);
                },
                backwardFn: BackwardFunctions<T>.ReplayUnderTape(eng =>
                {
                    var (x, h, wi, wh, bi, bh) = UnpackGruSequenceInputs(inputs, present);
                    return eng.GruSequenceForward(x, h, wi, wh, bi, bh, returnSequences: true, resetAfter: resetAfter);
                }));
        }
        else
        {
            sequence = GruSequenceRecorded(input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden, resetAfter);
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
        Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh, int batch, int seqLen, int hidden, bool resetAfter)
    {
        bool recording = DifferentiableOps.IsRecording<T>();
        if (typeof(T) == typeof(float))
        {
            var sequenceF = GruForwardFloat((Tensor<float>)(object)input, (Tensor<float>?)(object?)h0,
                (Tensor<float>)(object)wIh, (Tensor<float>)(object)wHh, (Tensor<float>?)(object?)bIh,
                (Tensor<float>?)(object?)bHh, batch, seqLen, hidden, resetAfter, recording, out var cacheF);
            var sequenceT = (Tensor<T>)(object)sequenceF;
            if (recording)
            {
                var floatInputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var floatPresent);
                DifferentiableOps.RecordIfActive("GruSequenceForward", sequenceT, floatInputs, GruSequenceBackwardFloat<T>,
                    new object[] { cacheF!, floatPresent });
            }
            return sequenceT;
        }

        var sequence = GruForwardOnHost(input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden, resetAfter, out var cache);
        if (recording)
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

    // ── The generic path, in double precision for any element type ─────────────────────────────────────────────────
    // Written as plain loops: it exists for exactness (double tensors, gradient checks), not speed; float takes the
    // SIMD path below.

    /// <summary>
    /// The generic path's saved activations, batch-major: allH [B, T + 1, H], gates [B, T, 3H] (r, z, n), ghn [B, T, H]
    /// (the candidate's recurrent sum with its bias), and with the reset gate before the recurrent product, rh [B, T, H]
    /// (r * h, the input of that product).
    /// </summary>
    private protected sealed class GruSequenceCache
    {
        internal double[] AllH = Array.Empty<double>();
        internal double[] Gates = Array.Empty<double>();
        internal double[] Ghn = Array.Empty<double>();
        internal double[] Rh = Array.Empty<double>();
        internal bool ResetAfter;
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

    private static Tensor<T> GruForwardOnHost<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, int batch, int seqLen, int hidden, bool resetAfter, out GruSequenceCache cache)
    {
        int inFeatures = input.Shape[2], g3 = 3 * hidden;
        var x = ToDoubles(input);
        var wi = ToDoubles(wIh);
        var wh = ToDoubles(wHh);
        var bi = bIh is null ? new double[g3] : ToDoubles(bIh);
        var bh = bHh is null ? new double[g3] : ToDoubles(bHh);
        var h0v = h0 is null ? new double[batch * hidden] : ToDoubles(h0);
        var allH = new double[batch * (seqLen + 1) * hidden];
        var gates = new double[batch * seqLen * g3];
        var ghn = new double[batch * seqLen * hidden];
        var rh = resetAfter ? Array.Empty<double>() : new double[batch * seqLen * hidden];
        var y = new double[batch * seqLen * hidden];
        var sums = new double[g3];
        var rhStep = new double[hidden];
        for (int b = 0; b < batch; b++)
        {
            Array.Copy(h0v, b * hidden, allH, b * (seqLen + 1) * hidden, hidden);
            for (int t = 0; t < seqLen; t++)
            {
                int prev = (b * (seqLen + 1) + t) * hidden, step = b * seqLen + t;
                // Input-side sums with their bias, and the r and z recurrent sums with theirs.
                for (int row = 0; row < g3; row++)
                {
                    double s = bi[row];
                    for (int i = 0; i < inFeatures; i++) s += wi[row * inFeatures + i] * x[step * inFeatures + i];
                    sums[row] = s;
                }
                for (int j = 0; j < hidden; j++)
                {
                    double rs = bh[j], zs = bh[hidden + j];
                    for (int k = 0; k < hidden; k++)
                    {
                        rs += wh[j * hidden + k] * allH[prev + k];
                        zs += wh[(hidden + j) * hidden + k] * allH[prev + k];
                    }
                    gates[step * g3 + j] = Sigmoid(sums[j] + rs);
                    gates[step * g3 + hidden + j] = Sigmoid(sums[hidden + j] + zs);
                }
                if (!resetAfter)
                    for (int k = 0; k < hidden; k++)
                    {
                        rhStep[k] = gates[step * g3 + k] * allH[prev + k];
                        rh[step * hidden + k] = rhStep[k];
                    }
                for (int j = 0; j < hidden; j++)
                {
                    // The candidate's recurrent sum: Whn h + bhn (r applied after), or Whn (r * h) + bhn (r before).
                    double hn = bh[2 * hidden + j];
                    for (int k = 0; k < hidden; k++)
                        hn += wh[(2 * hidden + j) * hidden + k] * (resetAfter ? allH[prev + k] : rhStep[k]);
                    double r = gates[step * g3 + j], z = gates[step * g3 + hidden + j];
                    double n = Math.Tanh(sums[2 * hidden + j] + (resetAfter ? r * hn : hn));
                    double hv = (1 - z) * n + z * allH[prev + j];
                    gates[step * g3 + 2 * hidden + j] = n;
                    ghn[step * hidden + j] = hn;
                    y[step * hidden + j] = hv;
                    allH[prev + hidden + j] = hv;
                }
            }
        }

        cache = new GruSequenceCache
        {
            AllH = allH, Gates = gates, Ghn = ghn, Rh = rh, ResetAfter = resetAfter,
            Batch = batch, SeqLen = seqLen, Input = inFeatures, Hidden = hidden,
        };
        return FromDoubles<T>(y, new[] { batch, seqLen, hidden });
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // Backpropagation through time (BatchExecutionCudaKernels.GruSource has the same equations on the device).
    // dgi / dgh [B, T, 3H] are the gradients of the input-side and recurrent-side pre-activation sums.
    private static void GruSequenceBackwardOnHost<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var c = (GruSequenceCache)savedState[0];
        var present = (GruOptionalInputs)savedState[1];
        var (input, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);
        int batch = c.Batch, seqLen = c.SeqLen, hidden = c.Hidden, inFeatures = c.Input, g3 = 3 * hidden;
        var gy = ToDoubles(gradOutput);
        var x = ToDoubles(input);
        var wi = ToDoubles(wIh);
        var wh = ToDoubles(wHh);
        var dgi = new double[batch * seqLen * g3];
        var dgh = new double[batch * seqLen * g3];
        var dh0 = new double[batch * hidden];
        var dh = new double[hidden];
        var drh = new double[hidden];
        var dhNext = new double[hidden];
        for (int b = 0; b < batch; b++)
        {
            Array.Clear(dh, 0, hidden);
            for (int t = seqLen - 1; t >= 0; t--)
            {
                int step = b * seqLen + t, prev = (b * (seqLen + 1) + t) * hidden, g = step * g3;
                for (int j = 0; j < hidden; j++)
                {
                    double r = c.Gates[g + j], z = c.Gates[g + hidden + j], n = c.Gates[g + 2 * hidden + j];
                    double hp = c.AllH[prev + j];
                    double dhv = gy[step * hidden + j] + dh[j];
                    double dnp = dhv * (1 - z) * (1 - n * n);
                    double dzp = dhv * (hp - n) * z * (1 - z);
                    dgi[g + hidden + j] = dzp; dgh[g + hidden + j] = dzp;
                    dgi[g + 2 * hidden + j] = dnp;
                    if (c.ResetAfter)
                    {
                        double drp = dnp * c.Ghn[step * hidden + j] * r * (1 - r);
                        dgi[g + j] = drp; dgh[g + j] = drp;
                        dgh[g + 2 * hidden + j] = dnp * r;
                    }
                    else
                    {
                        dgh[g + 2 * hidden + j] = dnp;
                    }
                    dhNext[j] = dhv * z;
                }
                if (!c.ResetAfter)
                {
                    // d(r * h) = Whn^T dn, then r's gradient and the path through r * h into h.
                    for (int k = 0; k < hidden; k++)
                    {
                        double s = 0;
                        for (int j = 0; j < hidden; j++) s += dgh[g + 2 * hidden + j] * wh[(2 * hidden + j) * hidden + k];
                        drh[k] = s;
                    }
                    for (int k = 0; k < hidden; k++)
                    {
                        double r = c.Gates[g + k];
                        double drp = drh[k] * c.AllH[prev + k] * r * (1 - r);
                        dgi[g + k] = drp; dgh[g + k] = drp;
                        dhNext[k] += drh[k] * r;
                    }
                }
                // The rows whose recurrent input is h itself: all three with r after, r and z with r before.
                int hRows = c.ResetAfter ? g3 : 2 * hidden;
                for (int k = 0; k < hidden; k++)
                {
                    double s = 0;
                    for (int row = 0; row < hRows; row++) s += dgh[g + row] * wh[row * hidden + k];
                    dh[k] = dhNext[k] + s;
                }
            }
            Array.Copy(dh, 0, dh0, b * hidden, hidden);
        }

        var gx = new double[batch * seqLen * inFeatures];
        var gwi = new double[g3 * inFeatures];
        var gwh = new double[g3 * hidden];
        var gbi = new double[g3];
        var gbh = new double[g3];
        for (int step = 0; step < batch * seqLen; step++)
        {
            int b = step / seqLen, t = step % seqLen, prev = (b * (seqLen + 1) + t) * hidden;
            for (int row = 0; row < g3; row++)
            {
                double gI = dgi[step * g3 + row], gH = dgh[step * g3 + row];
                gbi[row] += gI; gbh[row] += gH;
                for (int i = 0; i < inFeatures; i++)
                {
                    gx[step * inFeatures + i] += gI * wi[row * inFeatures + i];
                    gwi[row * inFeatures + i] += gI * x[step * inFeatures + i];
                }
                bool fromRh = !c.ResetAfter && row >= 2 * hidden;
                for (int k = 0; k < hidden; k++)
                    gwh[row * hidden + k] += gH * (fromRh ? c.Rh[step * hidden + k] : c.AllH[prev + k]);
            }
        }

        DifferentiableOps.AccumulateGrad(grads, input, FromDoubles<T>(gx, new[] { batch, seqLen, inFeatures }), engine);
        if (h0 is not null) DifferentiableOps.AccumulateGrad(grads, h0, FromDoubles<T>(dh0, new[] { batch, hidden }), engine);
        DifferentiableOps.AccumulateGrad(grads, wIh, FromDoubles<T>(gwi, new[] { g3, inFeatures }), engine);
        DifferentiableOps.AccumulateGrad(grads, wHh, FromDoubles<T>(gwh, new[] { g3, hidden }), engine);
        if (bIh is not null) DifferentiableOps.AccumulateGrad(grads, bIh, FromDoubles<T>(gbi, new[] { g3 }), engine);
        if (bHh is not null) DifferentiableOps.AccumulateGrad(grads, bHh, FromDoubles<T>(gbh, new[] { g3 }), engine);
    }

    // ── The float path: SIMD GEMMs over time-major caches, batch chunks in parallel ────────────────────────────────

    /// <summary>The float path's saved activations, time-major so each step's rows are contiguous for the GEMMs:
    /// allH [(T + 1), B, H], gates [T, B, 3H] (r, z, n), ghn [T, B, H], with the reset gate before the recurrent
    /// product rh [T, B, H], and the input [T, B, In].</summary>
    private protected sealed class GruSequenceCacheFloat
    {
        internal float[] AllH = Array.Empty<float>();
        internal float[] Gates = Array.Empty<float>();
        internal float[] Ghn = Array.Empty<float>();
        internal float[] Rh = Array.Empty<float>();
        internal float[] InputTimeMajor = Array.Empty<float>();
        internal bool ResetAfter;
        internal int Batch, SeqLen, Input, Hidden;
    }

    // One GEMM for every step's input sums [T * B, 3H]; per step, one recurrent GEMM on the contiguous previous state
    // (the three gates with r after, r and z with r before, then a second one for the candidate on r * h).
    private static Tensor<float> GruForwardFloat(Tensor<float> input, Tensor<float>? h0, Tensor<float> wIh,
        Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh, int batch, int seqLen, int hidden, bool resetAfter,
        bool keepCache, out GruSequenceCacheFloat? cache)
    {
        int inFeatures = input.Shape[2], g3 = 3 * hidden, rows = seqLen * batch;
        // Without a tape nothing outlives the call but the output: the scratch (about 11 MB at B=128, T=32, H=64) comes
        // from the pool instead of fresh, zeroed large-object allocations, which cost more than the recurrence itself.
        // A taped call keeps its arrays as the backward's cache.
        var rented = keepCache ? null : new List<float[]>();
        float[] Scratch(int length)
        {
            if (rented is null) return new float[length];
            var array = System.Buffers.ArrayPool<float>.Shared.Rent(length);
            rented.Add(array);
            return array;
        }
        try
        {
        var x = input.ToArray();
        var xt = Scratch(rows * inFeatures);
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(x, (b * seqLen + t) * inFeatures, xt, (t * batch + b) * inFeatures, inFeatures);
        var wihT = new float[inFeatures * g3];
        Simd.SimdGemm.Transpose(wIh.ToArray(), 0, inFeatures, wihT, 0, g3, g3, inFeatures);
        var whh = wHh.ToArray();
        // h's recurrent weights transposed to [H, hRows], and with r before, the candidate's to [H, H].
        int hRows = resetAfter ? g3 : 2 * hidden;
        var whT = new float[hidden * hRows];
        Simd.SimdGemm.Transpose(whh, 0, hidden, whT, 0, hRows, hRows, hidden);
        float[] wnT = Array.Empty<float>();
        if (!resetAfter)
        {
            wnT = new float[hidden * hidden];
            Simd.SimdGemm.Transpose(whh, 2 * hidden * hidden, hidden, wnT, 0, hidden, hidden, hidden);
        }
        var gi = Scratch(rows * g3);
        Simd.SimdGemm.Sgemm(xt, inFeatures, false, wihT, g3, false, gi, rows, inFeatures, g3);
        var bi = bIh?.ToArray() ?? new float[g3];
        var bh = bHh?.ToArray() ?? new float[g3];

        var allH = Scratch((seqLen + 1) * batch * hidden);
        if (h0 is not null) Array.Copy(h0.ToArray(), allH, batch * hidden);
        else Array.Clear(allH, 0, batch * hidden);   // a rented array is not zeroed
        var gates = Scratch(rows * g3);
        var ghn = Scratch(rows * hidden);
        var rh = resetAfter ? Array.Empty<float>() : Scratch(rows * hidden);
        var y = new float[batch * seqLen * hidden];
        // Batch rows never interact, so each chunk of them runs the whole recurrence on its own thread.
        ForEachBatchChunk(batch, hidden, (b0, b1) =>
        {
            int chunk = b1 - b0;
            var gh = new float[chunk * hRows];
            var gn = resetAfter ? Array.Empty<float>() : new float[chunk * hidden];
            for (int t = 0; t < seqLen; t++)
            {
                Simd.SimdGemm.SgemmSequential(allH.AsSpan((t * batch + b0) * hidden, chunk * hidden), whT, gh, chunk, hidden, hRows);
                if (!resetAfter)
                {
                    for (int b = b0; b < b1; b++)
                    {
                        int iRow = (t * batch + b) * g3, hRow = (b - b0) * hRows, prev = (t * batch + b) * hidden;
                        GruResetRow(gi, iRow, bi, gh, hRow, bh, allH, prev, gates, rh, hidden);
                    }
                    Simd.SimdGemm.SgemmSequential(rh.AsSpan((t * batch + b0) * hidden, chunk * hidden), wnT, gn, chunk, hidden, hidden);
                }
                for (int b = b0; b < b1; b++)
                {
                    int iRow = (t * batch + b) * g3, hRow = (b - b0) * hRows, prev = (t * batch + b) * hidden;
                    int next = ((t + 1) * batch + b) * hidden, outRow = (b * seqLen + t) * hidden;
                    GruCellRow(gi, iRow, bi, gh, hRow, bh, resetAfter ? gh : gn, resetAfter ? hRow + 2 * hidden : (b - b0) * hidden,
                        resetAfter, allH, prev, next, gates, ghn, y, outRow, hidden);
                }
            }
        });

        cache = keepCache
            ? new GruSequenceCacheFloat
            {
                AllH = allH, Gates = gates, Ghn = ghn, Rh = rh, InputTimeMajor = xt, ResetAfter = resetAfter,
                Batch = batch, SeqLen = seqLen, Input = inFeatures, Hidden = hidden,
            }
            : null;
        return new Tensor<float>(y, new[] { batch, seqLen, hidden });
        }
        finally
        {
            if (rented is not null)
                foreach (var array in rented) System.Buffers.ArrayPool<float>.Shared.Return(array);
        }
    }

    // r = sigmoid(input sum + recurrent sum) for one row, and r * h for the candidate's recurrent product (the reset-before
    // form); eight units at a time in AVX registers where available, as the LSTM path computes its gates.
    private static unsafe void GruResetRow(float[] gi, int iRow, float[] bi, float[] gh, int hRow, float[] bh,
        float[] allH, int prev, float[] gates, float[] rh, int hidden)
    {
        int j = 0;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            fixed (float* giP = gi) fixed (float* biP = bi) fixed (float* ghP = gh) fixed (float* bhP = bh)
            fixed (float* hP = allH) fixed (float* gP = gates) fixed (float* rhP = rh)
            {
                for (; j + 8 <= hidden; j += 8)
                {
                    var s = System.Runtime.Intrinsics.X86.Avx.Add(
                        System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.LoadVector256(giP + iRow + j), System.Runtime.Intrinsics.X86.Avx.LoadVector256(biP + j)),
                        System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ghP + hRow + j), System.Runtime.Intrinsics.X86.Avx.LoadVector256(bhP + j)));
                    var r = Simd.SimdKernels.FastSigmoid256(s);
                    System.Runtime.Intrinsics.X86.Avx.Store(gP + iRow + j, r);
                    System.Runtime.Intrinsics.X86.Avx.Store(rhP + prev + j,
                        System.Runtime.Intrinsics.X86.Avx.Multiply(r, System.Runtime.Intrinsics.X86.Avx.LoadVector256(hP + prev + j)));
                }
            }
        }
#endif
        for (; j < hidden; j++)
        {
            float r = 1f / (1f + MathF.Exp(-(gi[iRow + j] + bi[j] + gh[hRow + j] + bh[j])));
            gates[iRow + j] = r;
            rh[prev + j] = r * allH[prev + j];
        }
    }

    // One row of the cell: r (reset-after form; the reset-before form already stored it), z, the candidate n from its
    // recurrent sum hnSrc[hnRow + j] + bhn, and h' = (1 - z) n + z h; eight units at a time in AVX registers where available.
    private static unsafe void GruCellRow(float[] gi, int iRow, float[] bi, float[] gh, int hRow, float[] bh, float[] hnSrc,
        int hnRow, bool resetAfter, float[] allH, int prev, int next, float[] gates, float[] ghn, float[] y, int outRow, int hidden)
    {
        int j = 0;
        int h2 = 2 * hidden;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            fixed (float* giP = gi) fixed (float* biP = bi) fixed (float* ghP = gh) fixed (float* bhP = bh)
            fixed (float* hnP = hnSrc) fixed (float* hP = allH) fixed (float* gP = gates) fixed (float* ghnP = ghn)
            fixed (float* yP = y)
            {
                for (; j + 8 <= hidden; j += 8)
                {
                    System.Runtime.Intrinsics.Vector256<float> Sum(float* a, int ai, float* b, int bi2)
                        => System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.LoadVector256(a + ai), System.Runtime.Intrinsics.X86.Avx.LoadVector256(b + bi2));
                    var r = resetAfter
                        ? Simd.SimdKernels.FastSigmoid256(System.Runtime.Intrinsics.X86.Avx.Add(Sum(giP, iRow + j, biP, j), Sum(ghP, hRow + j, bhP, j)))
                        : System.Runtime.Intrinsics.X86.Avx.LoadVector256(gP + iRow + j);
                    var z = Simd.SimdKernels.FastSigmoid256(System.Runtime.Intrinsics.X86.Avx.Add(
                        Sum(giP, iRow + hidden + j, biP, hidden + j), Sum(ghP, hRow + hidden + j, bhP, hidden + j)));
                    var hn = Sum(hnP, hnRow + j, bhP, h2 + j);
                    var nIn = Sum(giP, iRow + h2 + j, biP, h2 + j);
                    var n = Simd.SimdKernels.FastTanh256(resetAfter
                        ? System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(r, hn, nIn)
                        : System.Runtime.Intrinsics.X86.Avx.Add(nIn, hn));
                    var hp = System.Runtime.Intrinsics.X86.Avx.LoadVector256(hP + prev + j);
                    // (1 - z) n + z h = n + z (h - n)
                    var hv = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(z, System.Runtime.Intrinsics.X86.Avx.Subtract(hp, n), n);
                    System.Runtime.Intrinsics.X86.Avx.Store(gP + iRow + j, r);
                    System.Runtime.Intrinsics.X86.Avx.Store(gP + iRow + hidden + j, z);
                    System.Runtime.Intrinsics.X86.Avx.Store(gP + iRow + h2 + j, n);
                    System.Runtime.Intrinsics.X86.Avx.Store(ghnP + prev + j, hn);
                    System.Runtime.Intrinsics.X86.Avx.Store(hP + next + j, hv);
                    System.Runtime.Intrinsics.X86.Avx.Store(yP + outRow + j, hv);
                }
            }
        }
#endif
        for (; j < hidden; j++)
        {
            float r = resetAfter
                ? 1f / (1f + MathF.Exp(-(gi[iRow + j] + bi[j] + gh[hRow + j] + bh[j])))
                : gates[iRow + j];
            float z = 1f / (1f + MathF.Exp(-(gi[iRow + hidden + j] + bi[hidden + j] + gh[hRow + hidden + j] + bh[hidden + j])));
            float hn = hnSrc[hnRow + j] + bh[h2 + j];
            float n = MathF.Tanh(gi[iRow + h2 + j] + bi[h2 + j] + (resetAfter ? r * hn : hn));
            float hv = (1f - z) * n + z * allH[prev + j];
            gates[iRow + j] = r; gates[iRow + hidden + j] = z; gates[iRow + h2 + j] = n;
            ghn[prev + j] = hn;
            allH[next + j] = hv;
            y[outRow + j] = hv;
        }
    }

    // The float BPTT: per step the gate gradients and the recurrent GEMMs into dh, then three GEMMs over all steps for
    // the input and weight gradients (four with r before: the candidate's recurrent weights take r * h).
    private static void GruSequenceBackwardFloat<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var cache = (GruSequenceCacheFloat)savedState[0];
        var present = (GruOptionalInputs)savedState[1];
        var (input, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);
        int batch = cache.Batch, seqLen = cache.SeqLen, hidden = cache.Hidden, inFeatures = cache.Input;
        int g3 = 3 * hidden, rows = seqLen * batch;
        bool resetAfter = cache.ResetAfter;
        int hRows = resetAfter ? g3 : 2 * hidden;
        var gy = ((Tensor<float>)(object)gradOutput).ToArray();
        var whh = ((Tensor<float>)(object)wHh).ToArray();
        var wih = ((Tensor<float>)(object)wIh).ToArray();
        // dgi / dgh and the time-major input gradient are this call's scratch: pooled, as every element is written before
        // it is read. The gradients handed to the tape are fresh arrays.
        var pool = System.Buffers.ArrayPool<float>.Shared;
        var dgi = pool.Rent(rows * g3);
        var dgh = pool.Rent(rows * g3);
        var gxTimeMajor = pool.Rent(rows * inFeatures);
        try
        {
        var dh = new float[batch * hidden];
        ForEachBatchChunk(batch, hidden, (b0, b1) =>
        {
            int chunk = b1 - b0;
            var rec = new float[chunk * hidden];
            // The gradients of the sums whose recurrent input is h, contiguous per row for the GEMM into dh.
            var hSums = new float[chunk * hRows];
            var dn = resetAfter ? Array.Empty<float>() : new float[chunk * hidden];
            var drh = resetAfter ? Array.Empty<float>() : new float[chunk * hidden];
            for (int t = seqLen - 1; t >= 0; t--)
            {
                for (int b = b0; b < b1; b++)
                {
                    int gRow = (t * batch + b) * g3, sRow = (t * batch + b) * hidden, c = (b - b0);
                    for (int j = 0; j < hidden; j++)
                    {
                        float r = cache.Gates[gRow + j], z = cache.Gates[gRow + hidden + j], n = cache.Gates[gRow + 2 * hidden + j];
                        float hp = cache.AllH[sRow + j];
                        float dhv = gy[(b * seqLen + t) * hidden + j] + dh[b * hidden + j];
                        float dnp = dhv * (1f - z) * (1f - n * n);
                        float dzp = dhv * (hp - n) * z * (1f - z);
                        dgi[gRow + hidden + j] = dzp; dgh[gRow + hidden + j] = dzp; hSums[c * hRows + hidden + j] = dzp;
                        dgi[gRow + 2 * hidden + j] = dnp;
                        if (resetAfter)
                        {
                            float drp = dnp * cache.Ghn[sRow + j] * r * (1f - r);
                            dgi[gRow + j] = drp; dgh[gRow + j] = drp; hSums[c * hRows + j] = drp;
                            dgh[gRow + 2 * hidden + j] = dnp * r; hSums[c * hRows + 2 * hidden + j] = dnp * r;
                        }
                        else
                        {
                            dgh[gRow + 2 * hidden + j] = dnp; dn[c * hidden + j] = dnp;
                        }
                        dh[b * hidden + j] = dhv * z;   // the direct term; the recurrent ones are added below
                    }
                }
                if (!resetAfter)
                {
                    // d(r * h) = dn Whn ([chunk, H] x [H, H]: Whn is rows 2H..3H of wHh), then r's gradient.
                    Simd.SimdGemm.SgemmSequential(dn, whh.AsSpan(2 * hidden * hidden, hidden * hidden), drh, chunk, hidden, hidden);
                    for (int b = b0; b < b1; b++)
                    {
                        int gRow = (t * batch + b) * g3, sRow = (t * batch + b) * hidden, c = (b - b0);
                        for (int k = 0; k < hidden; k++)
                        {
                            float r = cache.Gates[gRow + k];
                            float d = drh[c * hidden + k];
                            float drp = d * cache.AllH[sRow + k] * r * (1f - r);
                            dgi[gRow + k] = drp; dgh[gRow + k] = drp; hSums[c * hRows + k] = drp;
                            dh[b * hidden + k] += d * r;
                        }
                    }
                }
                Simd.SimdGemm.SgemmSequential(hSums, whh.AsSpan(0, hRows * hidden), rec, chunk, hRows, hidden);
                for (int k = 0; k < rec.Length; k++) dh[b0 * hidden + k] += rec[k];
            }
        });

        Simd.SimdGemm.Sgemm(dgi, g3, false, wih, inFeatures, false, gxTimeMajor, rows, g3, inFeatures);
        var gx = new float[batch * seqLen * inFeatures];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(gxTimeMajor, (t * batch + b) * inFeatures, gx, (b * seqLen + t) * inFeatures, inFeatures);
        var gwih = new float[g3 * inFeatures];
        Simd.SimdGemm.Sgemm(dgi, g3, true, cache.InputTimeMajor, inFeatures, false, gwih, g3, rows, inFeatures);
        var gwhh = new float[g3 * hidden];
        // Rows whose recurrent input is h take h_{t-1}; with r before, the candidate's rows take r * h_{t-1}.
        Simd.SimdGemm.Sgemm(dgh, g3, true, cache.AllH.AsSpan(0, rows * hidden), hidden, false,
            gwhh.AsSpan(0, hRows * hidden), hRows, rows, hidden);
        if (!resetAfter)
            Simd.SimdGemm.Sgemm(dgh.AsSpan(2 * hidden), g3, true, cache.Rh, hidden, false,
                gwhh.AsSpan(2 * hidden * hidden, hidden * hidden), hidden, rows, hidden);

        Tensor<T> Wrap(float[] data, params int[] shape) => (Tensor<T>)(object)new Tensor<float>(data, shape);
        DifferentiableOps.AccumulateGrad(grads, input, Wrap(gx, batch, seqLen, inFeatures), engine);
        if (h0 is not null) DifferentiableOps.AccumulateGrad(grads, h0, Wrap(dh, batch, hidden), engine);
        DifferentiableOps.AccumulateGrad(grads, wIh, Wrap(gwih, g3, inFeatures), engine);
        DifferentiableOps.AccumulateGrad(grads, wHh, Wrap(gwhh, g3, hidden), engine);
        if (bIh is not null || bHh is not null)
        {
            var sbi = new float[g3];
            var sbh = new float[g3];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < g3; c++) { sbi[c] += dgi[r * g3 + c]; sbh[c] += dgh[r * g3 + c]; }
            if (bIh is not null) DifferentiableOps.AccumulateGrad(grads, bIh, Wrap(sbi, g3), engine);
            if (bHh is not null) DifferentiableOps.AccumulateGrad(grads, bHh, Wrap(sbh, g3), engine);
        }
        }
        finally
        {
            pool.Return(dgi);
            pool.Return(dgh);
            pool.Return(gxTimeMajor);
        }
    }

    // Splits the batch into contiguous chunks, one per worker, once a step's work is worth a thread (the LSTM path's
    // rule of thumb); a small batch runs inline.
    private static void ForEachBatchChunk(int batch, int hidden, Action<int, int> body)
    {
        int workers = Math.Min(Environment.ProcessorCount, batch / 8);
        if (workers <= 1 || (long)batch * hidden < 4096) { body(0, batch); return; }
        int size = (batch + workers - 1) / workers;
        System.Threading.Tasks.Parallel.For(0, workers, w =>
        {
            int b0 = w * size, b1 = Math.Min(batch, b0 + size);
            if (b0 < b1) body(b0, b1);
        });
    }
}
