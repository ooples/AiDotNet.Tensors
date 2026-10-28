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
        if (typeof(T) == typeof(float))
        {
            bool recording = DifferentiableOps.IsRecording<T>();
            var sequenceF = GruForwardFloat((Tensor<float>)(object)input, (Tensor<float>?)(object?)h0,
                (Tensor<float>)(object)wIh, (Tensor<float>)(object)wHh, (Tensor<float>?)(object?)bIh,
                (Tensor<float>?)(object?)bHh, batch, seqLen, hidden, recording, out var cacheF);
            var sequenceT = (Tensor<T>)(object)sequenceF;
            if (recording)
            {
                var floatInputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var floatPresent);
                DifferentiableOps.RecordIfActive("GruSequenceForward", sequenceT, floatInputs, GruSequenceBackwardFloat<T>,
                    new object[] { cacheF!, floatPresent });
            }
            return sequenceT;
        }

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

    /// <summary>The float path's saved activations, time-major so each step's rows are contiguous for the GEMMs:
    /// allH [(T + 1), B, H], gates [T, B, 3H] (r, z, n), ghn [T, B, H], and the input [T, B, In].</summary>
    private protected sealed class GruSequenceCacheFloat
    {
        internal float[] AllH = Array.Empty<float>();
        internal float[] Gates = Array.Empty<float>();
        internal float[] Ghn = Array.Empty<float>();
        internal float[] InputTimeMajor = Array.Empty<float>();
        internal int Batch, SeqLen, Input, Hidden;
    }

    // One GEMM for every step's input sums [T * B, 3H], then one [B, H] x [H, 3H] recurrent GEMM per step on the
    // contiguous previous state, with the gate math fused into a single pass.
    private static Tensor<float> GruForwardFloat(Tensor<float> input, Tensor<float>? h0, Tensor<float> wIh,
        Tensor<float> wHh, Tensor<float>? bIh, Tensor<float>? bHh, int batch, int seqLen, int hidden, bool keepCache,
        out GruSequenceCacheFloat? cache)
    {
        int inFeatures = input.Shape[2], g3 = 3 * hidden, rows = seqLen * batch;
        var x = input.ToArray();
        var xt = new float[rows * inFeatures];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(x, (b * seqLen + t) * inFeatures, xt, (t * batch + b) * inFeatures, inFeatures);
        var wihT = new float[inFeatures * g3];
        Simd.SimdGemm.Transpose(wIh.ToArray(), 0, inFeatures, wihT, 0, g3, g3, inFeatures);
        var whhT = new float[hidden * g3];
        Simd.SimdGemm.Transpose(wHh.ToArray(), 0, hidden, whhT, 0, g3, g3, hidden);
        var gi = new float[rows * g3];
        Simd.SimdGemm.Sgemm(xt, inFeatures, false, wihT, g3, false, gi, rows, inFeatures, g3);
        var bi = bIh?.ToArray() ?? new float[g3];
        var bh = bHh?.ToArray() ?? new float[g3];

        var allH = new float[(seqLen + 1) * batch * hidden];
        if (h0 is not null) Array.Copy(h0.ToArray(), allH, batch * hidden);
        var gates = new float[rows * g3];
        var ghn = new float[rows * hidden];
        var y = new float[batch * seqLen * hidden];
        // Batch rows never interact, so each chunk of them runs the whole recurrence on its own thread.
        ForEachBatchChunk(batch, hidden, (b0, b1) =>
        {
            int chunk = b1 - b0;
            var gh = new float[chunk * g3];
            for (int t = 0; t < seqLen; t++)
            {
                Simd.SimdGemm.SgemmSequential(allH.AsSpan((t * batch + b0) * hidden, chunk * hidden), whhT, gh, chunk, hidden, g3);
                for (int b = b0; b < b1; b++)
                {
                    int iRow = (t * batch + b) * g3, hRow = (b - b0) * g3, prev = (t * batch + b) * hidden;
                    int next = ((t + 1) * batch + b) * hidden, outRow = (b * seqLen + t) * hidden;
                    for (int j = 0; j < hidden; j++)
                    {
                        float r = 1f / (1f + MathF.Exp(-(gi[iRow + j] + bi[j] + gh[hRow + j] + bh[j])));
                        float z = 1f / (1f + MathF.Exp(-(gi[iRow + hidden + j] + bi[hidden + j] + gh[hRow + hidden + j] + bh[hidden + j])));
                        float hn = gh[hRow + 2 * hidden + j] + bh[2 * hidden + j];
                        float n = MathF.Tanh(gi[iRow + 2 * hidden + j] + bi[2 * hidden + j] + r * hn);
                        float hv = (1f - z) * n + z * allH[prev + j];
                        gates[iRow + j] = r; gates[iRow + hidden + j] = z; gates[iRow + 2 * hidden + j] = n;
                        ghn[prev + j] = hn;
                        allH[next + j] = hv;
                        y[outRow + j] = hv;
                    }
                }
            }
        });

        cache = keepCache
            ? new GruSequenceCacheFloat
            {
                AllH = allH, Gates = gates, Ghn = ghn, InputTimeMajor = xt,
                Batch = batch, SeqLen = seqLen, Input = inFeatures, Hidden = hidden,
            }
            : null;
        return new Tensor<float>(y, new[] { batch, seqLen, hidden });
    }

    // The float BPTT: per step the gate gradients and one [B, 3H] x [3H, H] GEMM for the recurrent term, then three
    // GEMMs over all steps for the input and weight gradients.
    private static void GruSequenceBackwardFloat<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var cache = (GruSequenceCacheFloat)savedState[0];
        var present = (GruOptionalInputs)savedState[1];
        var (input, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);
        int batch = cache.Batch, seqLen = cache.SeqLen, hidden = cache.Hidden, inFeatures = cache.Input;
        int g3 = 3 * hidden, rows = seqLen * batch;
        var gy = ((Tensor<float>)(object)gradOutput).ToArray();
        var whh = ((Tensor<float>)(object)wHh).ToArray();
        var wih = ((Tensor<float>)(object)wIh).ToArray();
        var dgi = new float[rows * g3];
        var dgh = new float[rows * g3];
        var dh = new float[batch * hidden];
        ForEachBatchChunk(batch, hidden, (b0, b1) =>
        {
            int chunk = b1 - b0;
            var rec = new float[chunk * hidden];
            for (int t = seqLen - 1; t >= 0; t--)
            {
                for (int b = b0; b < b1; b++)
                {
                    int gRow = (t * batch + b) * g3, sRow = (t * batch + b) * hidden;
                    for (int j = 0; j < hidden; j++)
                    {
                        float r = cache.Gates[gRow + j], z = cache.Gates[gRow + hidden + j], n = cache.Gates[gRow + 2 * hidden + j];
                        float hn = cache.Ghn[sRow + j], hp = cache.AllH[sRow + j];
                        float dhv = gy[(b * seqLen + t) * hidden + j] + dh[b * hidden + j];
                        float dnp = dhv * (1f - z) * (1f - n * n);
                        float drp = dnp * hn * r * (1f - r);
                        float dzp = dhv * (hp - n) * z * (1f - z);
                        dgi[gRow + j] = drp; dgi[gRow + hidden + j] = dzp; dgi[gRow + 2 * hidden + j] = dnp;
                        dgh[gRow + j] = drp; dgh[gRow + hidden + j] = dzp; dgh[gRow + 2 * hidden + j] = dnp * r;
                        dh[b * hidden + j] = dhv * z;
                    }
                }
                Simd.SimdGemm.SgemmSequential(dgh.AsSpan((t * batch + b0) * g3, chunk * g3), whh, rec, chunk, g3, hidden);
                for (int k = 0; k < rec.Length; k++) dh[b0 * hidden + k] += rec[k];
            }
        });

        var gxTimeMajor = new float[rows * inFeatures];
        Simd.SimdGemm.Sgemm(dgi, g3, false, wih, inFeatures, false, gxTimeMajor, rows, g3, inFeatures);
        var gx = new float[batch * seqLen * inFeatures];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
                Array.Copy(gxTimeMajor, (t * batch + b) * inFeatures, gx, (b * seqLen + t) * inFeatures, inFeatures);
        var gwih = new float[g3 * inFeatures];
        Simd.SimdGemm.Sgemm(dgi, g3, true, cache.InputTimeMajor, inFeatures, false, gwih, g3, rows, inFeatures);
        var gwhh = new float[g3 * hidden];
        Simd.SimdGemm.Sgemm(dgh, g3, true, cache.AllH.AsSpan(0, rows * hidden), hidden, false, gwhh, g3, rows, hidden);

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