using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    /// <remarks>On the host, float32 runs as ONE recorded op: a forward pass over the logits that keeps each row's
    /// softmax, and a backward that writes (rowMass * softmax - target) * g / count straight into the logits' gradient.
    /// The composed primitives (log-softmax, multiply, four reductions, abs, sign, clamp, divide, negate) cost a
    /// classifier ~270 us of every training step on the parity MLP, more than PyTorch's whole loss. Other element types,
    /// engines and device-resident inputs run the composed primitives.</remarks>
    public virtual Tensor<T> TensorSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        ValidateSoftmaxCrossEntropy(logits, target);
        if (typeof(T) != typeof(float) || this is DirectGpuTensorEngine || logits.Length == 0
            || IsDeviceResident(logits) || IsDeviceResident(target))
            return SoftmaxCrossEntropyDecomposed(logits, target);

        var z = (Tensor<float>)(object)logits;
        var t = (Tensor<float>)(object)target;
        int cols = z._shape[z.Rank - 1];
        int rows = z.Length / cols;

        if (GraphMode.IsActive)
        {
            if (GraphMode.Current is not { } scope || !scope.BindEngine(this))
                return SoftmaxCrossEntropyDecomposed(logits, target);
            var state = new SoftmaxCrossEntropyState(rows, cols);
            var inputs = new[] { z.IsContiguous ? z : z.Contiguous(), t.IsContiguous ? t : t.Contiguous() };
            return (Tensor<T>)(object)scope.RecordVariadic(LazyNodeType.Custom, "SoftmaxCrossEntropy", inputs,
                Array.Empty<int>(),
                (eng, output) =>
                {
                    float loss = SoftmaxCrossEntropyForwardFloat(inputs[0], inputs[1], rows, cols, state);
                    if (output.IsContiguous && output._gpuBuffer is null && !output.HasPendingGpuData
                        && output.GetCpuBackingForContiguousWrite(out int oOff) is { } oArr)
                    {
                        oArr[oOff] = loss;
                        output.IncrementVersion();
                    }
                    else
                    {
                        DirectGpuTensorEngine.CopyResultInto(eng, new Tensor<float>(new[] { loss }, Array.Empty<int>()), output);
                    }
                },
                SoftmaxCrossEntropyBackwardFloat, new object[] { state });
        }

        var eagerState = new SoftmaxCrossEntropyState(rows, cols);
        var zc = z.IsContiguous ? z : z.Contiguous();
        var tc = t.IsContiguous ? t : t.Contiguous();
        var result = new Tensor<float>(new[] { SoftmaxCrossEntropyForwardFloat(zc, tc, rows, cols, eagerState) },
            Array.Empty<int>());
        var resultT = (Tensor<T>)(object)result;
        DifferentiableOps.RecordIfActive("SoftmaxCrossEntropy", resultT, new[] { logits, target },
            (BackwardFunction<T>)(object)(BackwardFunction<float>)SoftmaxCrossEntropyBackwardFloat,
            new object[] { eagerState });
        return resultT;
    }

    private static bool IsDeviceResident<T>(Tensor<T> tensor) => tensor.HasPendingGpuData || tensor.TryGetGpuBuffer() is not null;

    private protected static void ValidateSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (logits.Rank < 2)
            throw new ArgumentException($"Softmax cross-entropy expects logits of rank 2 or more; got rank {logits.Rank}.", nameof(logits));
        if (!logits._shape.SequenceEqual(target._shape))
            throw new ArgumentException(
                $"target shape [{string.Join(", ", target._shape)}] must equal logits shape [{string.Join(", ", logits._shape)}].",
                nameof(target));
    }

    /// <summary>
    /// The objective as recorded primitives -- log-softmax, the target-weighted sum, and a detached supervised-row
    /// count -- so every engine and element type differentiates it with the ops it already has.
    /// </summary>
    private Tensor<T> SoftmaxCrossEntropyDecomposed<T>(Tensor<T> logits, Tensor<T> target)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int classAxis = logits.Rank - 1;
        var batchAxes = Enumerable.Range(0, classAxis).ToArray();
        var logSoftmax = TensorLogSoftmax(logits, classAxis);
        var perSample = ReduceSum(TensorMultiply(target, logSoftmax), new[] { classAxis }, keepDims: false);
        var total = ReduceSum(perSample, batchAxes, keepDims: false);
        // A row with target mass is supervised; the count depends only on the target, so it is detached.
        var supervised = ReduceSum(TensorSign(ReduceSum(TensorAbs(target), new[] { classAxis }, keepDims: false)),
            batchAxes, keepDims: false);
        var safeCount = StopGradient(TensorClampMin(supervised, numOps.One));
        return TensorNegate(TensorDivide(total, safeCount));
    }

    /// <summary>What the fused forward leaves for its backward: each row's softmax, log-sum-exp and target mass, and
    /// the supervised-row count the loss was divided by. A compiled step reuses one instance on every replay.</summary>
    internal sealed class SoftmaxCrossEntropyState
    {
        public SoftmaxCrossEntropyState(int rows, int cols)
        {
            Probabilities = new float[rows * cols];
            LogSumExp = new float[rows];
            RowMass = new float[rows];
            RowLoss = new float[rows];
            RowAbsMass = new float[rows];
        }

        public float[] Probabilities { get; }
        public float[] LogSumExp { get; }
        public float[] RowMass { get; }
        public float[] RowLoss { get; }
        public float[] RowAbsMass { get; }
        public float Count { get; set; }
    }

    /// <summary>
    /// loss = -(sum over rows of sum_j t_j * (z_j - lse)) / max(1, supervised rows), where a row is supervised when its
    /// target has any absolute mass -- the composed objective's value. A NaN target mass makes the count NaN, as the
    /// composed sign(sum|t|) does. Rows are independent and the total is summed in row order, so the result does not
    /// depend on the thread count.
    /// </summary>
    internal static float SoftmaxCrossEntropyForwardFloat(
        Tensor<float> logits, Tensor<float> target, int rows, int cols, SoftmaxCrossEntropyState state)
    {
        var z = ReadableBacking(logits, out int zOff);
        var t = ReadableBacking(target, out int tOff);
        CpuParallelSettings.ParallelForOrSerial(0, rows, 4L * rows * cols,
            r => SoftmaxCrossEntropyForwardRow(z, zOff + r * cols, t, tOff + r * cols, cols, state, r),
            deterministicSafe: true);
        double total = 0, count = 0;
        for (int r = 0; r < rows; r++)
        {
            total += state.RowLoss[r];
            float abs = state.RowAbsMass[r];
            count += float.IsNaN(abs) ? double.NaN : abs > 0f ? 1 : 0;
        }
        state.Count = double.IsNaN(count) ? float.NaN : (float)Math.Max(1, count);
        return (float)(-total / state.Count);
    }

    private static void SoftmaxCrossEntropyForwardRow(
        float[] z, int zo, float[] t, int to, int cols, SoftmaxCrossEntropyState state, int r)
    {
        int w = System.Numerics.Vector<float>.Count, c = 0;
        float max = float.NegativeInfinity;
        if (cols >= w)
        {
            var vmax = new System.Numerics.Vector<float>(z, zo);
            for (c = w; c + w <= cols; c += w) vmax = System.Numerics.Vector.Max(vmax, new System.Numerics.Vector<float>(z, zo + c));
            for (int l = 0; l < w; l++) max = Math.Max(max, vmax[l]);
        }
        for (; c < cols; c++) max = Math.Max(max, z[zo + c]);

        // exp(z - max) goes straight into this row's softmax slot; normalized below, the backward reads it back, so
        // each logit is exponentiated once per step.
        var p = state.Probabilities;
        int po = r * cols;
        for (c = 0; c < cols; c++) p[po + c] = z[zo + c] - max;
        var pSpan = new Span<float>(p, po, cols);
        Simd.SimdKernels.Exp(pSpan, pSpan);
        float sum = 0f;
        c = 0;
        if (cols >= w)
        {
            var acc = System.Numerics.Vector<float>.Zero;
            for (; c + w <= cols; c += w) acc += new System.Numerics.Vector<float>(p, po + c);
            sum = System.Numerics.Vector.Dot(acc, System.Numerics.Vector<float>.One);
        }
        for (; c < cols; c++) sum += p[po + c];
        float lse = max + (float)Math.Log(sum);
        float inv = 1f / sum;
        for (c = 0; c < cols; c++) p[po + c] *= inv;

        float rowLoss = 0f, mass = 0f, absMass = 0f;
        for (c = 0; c < cols; c++)
        {
            float tv = t[to + c];
            rowLoss += tv * (z[zo + c] - lse);
            mass += tv;
            absMass += Math.Abs(tv);
        }
        state.LogSumExp[r] = lse;
        state.RowMass[r] = mass;
        state.RowLoss[r] = rowLoss;
        state.RowAbsMass[r] = absMass;
    }

    /// <summary>
    /// dL/dz_j = (g / count) * (rowMass * softmax_j - t_j); dL/dt_j = -(g / count) * (z_j - lse), the count being detached.
    /// The logits' gradient is written straight into its accumulator when there is one.
    /// </summary>
    private static void SoftmaxCrossEntropyBackwardFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var state = (SoftmaxCrossEntropyState)savedState[0];
        Tensor<float> logits = inputs[0], target = inputs[1];
        bool needZ = DifferentiableOps.IsGradientRequired(logits);
        bool needT = DifferentiableOps.IsGradientRequired(target);
        if (!needZ && !needT) return;

        var g = ReadableBacking(gradOutput, out int gOff);
        float scale = g[gOff] / state.Count;
        var z = ReadableBacking(logits, out int zOff);
        var t = ReadableBacking(target, out int tOff);
        var p = state.Probabilities;
        int cols = logits._shape[logits.Rank - 1];
        int rows = logits.Length / cols;
        bool distinct = !ReferenceEquals(logits, target);
        var dZ = GradTarget(needZ, logits, grads, engine, distinct);
        var dT = GradTarget(needT, target, grads, engine, distinct);
        if (distinct && dZ.Array is not null && Overlap(dZ, dT))
            dZ = GradTarget(needZ, logits, grads, engine, distinct: false);

        CpuParallelSettings.ParallelForOrSerial(0, rows, 2L * rows * cols, r =>
        {
            int zo = zOff + r * cols, to = tOff + r * cols, po = r * cols;
            if (dZ.Array is { } dz)
            {
                float rowScale = scale * state.RowMass[r];
                int o = dZ.Offset + r * cols;
                for (int c = 0; c < cols; c++)
                {
                    float v = rowScale * p[po + c] - scale * t[to + c];
                    dz[o + c] = dZ.Overwrite ? v : dz[o + c] + v;
                }
            }
            if (dT.Array is { } dt)
            {
                float lse = state.LogSumExp[r];
                int o = dT.Offset + r * cols;
                for (int c = 0; c < cols; c++)
                {
                    float v = -scale * (z[zo + c] - lse);
                    dt[o + c] = dT.Overwrite ? v : dt[o + c] + v;
                }
            }
        }, deterministicSafe: true);

        dZ.Commit(grads, logits, engine);
        dT.Commit(grads, target, engine);
    }
}
