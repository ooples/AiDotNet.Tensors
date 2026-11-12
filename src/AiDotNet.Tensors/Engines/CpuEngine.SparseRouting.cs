using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    private const int SparseRoutingChannelBlock = 256;

    /// <summary>
    /// Row-wise sparse gather along the last axis:
    /// <code>
    ///   output[..., j] = values[..., indices[..., j]]   (0 when indices[..., j] is outside [0, G))
    /// </code>
    /// for <paramref name="values"/> of shape <c>[..., G]</c> and <paramref name="indices"/> of shape <c>[..., K]</c>
    /// sharing all leading dimensions. Replaces materializing an <c>[..., K, G]</c> one-hot mask, elementwise product,
    /// and last-axis reduction in sparse MoE / codebook routers with a single <c>O(rows * K)</c> pass and one tape node.
    /// </summary>
    public virtual Tensor<T> TensorGatherRowValues<T>(Tensor<T> values, Tensor<int> indices)
    {
        var (rows, cols, k, outShape) = GatherRowValuesDims(values, indices);
        if (typeof(T) != typeof(int))
        {
            GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
            if (indices.LazySource is not null)
                GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.MixedElementTypes);
        }
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedValues = values;
            var capturedIndices = indices;
            var compiledSavedState = new object[] { null!, cols, indices };
            var graphInputs = typeof(T) == typeof(int)
                ? new[] { values, (Tensor<T>)(object)indices }
                : new[] { values };
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "TensorGatherRowValues",
                graphInputs,
                outShape,
                (eng, result) =>
                {
                    compiledSavedState[0] = CloneIndicesHost(capturedIndices);
                    var sr = eng.As<CpuEngine>() ?? new CpuEngine();
                    DirectGpuTensorEngine.CopyResultInto(eng, sr.TensorGatherRowValues(capturedValues, capturedIndices), result);
                },
                TensorGatherRowValuesAutoBackward<T>,
                compiledSavedState);
        }

        var output = new Tensor<T>(outShape);
        if (output.Length > 0)
        {
            var v = HostArray(values, out int vOff);
            var idx = HostArray(indices, out int idxOff);
            GatherRowValuesForwardHost(v, vOff, idx, idxOff, output.GetDataArray(), rows, cols, k);
        }

        if (DifferentiableOps.IsRecording<T>())
        {
            DifferentiableOps.RecordIfActive(
                "TensorGatherRowValues",
                output,
                new[] { values },
                TensorGatherRowValuesAutoBackward<T>,
                savedState: new object[] { CloneIndicesHost(indices), cols });
        }

        return output;
    }

    /// <summary>
    /// Gradient of <see cref="TensorGatherRowValues{T}"/> with respect to <c>values</c>:
    /// <c>gradValues[..., c] = sum_{j : indices[..., j] == c} gradOutput[..., j]</c>.
    /// </summary>
    public virtual Tensor<T> TensorGatherRowValuesBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int numCols)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (numCols < 1) throw new ArgumentOutOfRangeException(nameof(numCols), "numCols must be positive.");
        if (gradOutput.Rank < 1 || !ShapesEqual(gradOutput._shape, indices._shape))
            throw new ArgumentException("gradOutput and indices must have the same rank >= 1 shape.", nameof(gradOutput));

        int k = gradOutput._shape[gradOutput.Rank - 1];
        int rows = k == 0 ? LeadingProduct(gradOutput._shape, gradOutput.Rank - 1) : gradOutput.Length / k;
        var outShape = (int[])gradOutput._shape.Clone();
        outShape[outShape.Length - 1] = numCols;
        var gradValues = new Tensor<T>(outShape);
        if (gradValues.Length > 0 && k > 0)
        {
            var g = HostArray(gradOutput, out int gOff);
            var idx = HostArray(indices, out int idxOff);
            GatherRowValuesBackwardHost(g, gOff, idx, idxOff, gradValues.GetDataArray(), rows, numCols, k);
        }
        return gradValues;
    }

    /// <summary>
    /// Sparse top-K weighted row mix (fused sparse gather-accumulate without materializing an <c>[..., K, G]</c> or
    /// <c>[..., G]</c> routing matrix):
    /// <code>
    ///   output[..., c] = sum_{j = 0..K-1} weights[..., j] * table[indices[..., j], c]
    /// </code>
    /// for <paramref name="table"/> of shape <c>[G, C]</c>, <paramref name="indices"/> of shape <c>[..., K]</c>, and
    /// <paramref name="weights"/> of shape <c>[..., K]</c>. Indices outside <c>[0, G)</c> contribute zero.
    /// </summary>
    public virtual Tensor<T> TopKWeightedRowMix<T>(Tensor<T> table, Tensor<int> indices, Tensor<T> weights)
    {
        var (rows, numEntries, k, channels, outShape) = TopKMixDims(table, indices, weights);
        if (typeof(T) != typeof(int))
        {
            GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
            if (indices.LazySource is not null)
                GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.MixedElementTypes);
        }
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedTable = table;
            var capturedIndices = indices;
            var capturedWeights = weights;
            var compiledSavedState = new object[] { null!, indices };
            var graphInputs = typeof(T) == typeof(int)
                ? new[] { table, weights, (Tensor<T>)(object)indices }
                : new[] { table, weights };
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "TopKWeightedRowMix",
                graphInputs,
                outShape,
                (eng, result) =>
                {
                    compiledSavedState[0] = CloneIndicesHost(capturedIndices);
                    var sr = eng.As<CpuEngine>() ?? new CpuEngine();
                    DirectGpuTensorEngine.CopyResultInto(
                        eng, sr.TopKWeightedRowMix(capturedTable, capturedIndices, capturedWeights), result);
                },
                TopKWeightedRowMixAutoBackward<T>,
                compiledSavedState);
        }

        var output = new Tensor<T>(outShape);
        if (output.Length > 0 && k > 0)
        {
            var tbl = HostArray(table, out int tblOff);
            var idx = HostArray(indices, out int idxOff);
            var w = HostArray(weights, out int wOff);
            TopKMixForwardHost(tbl, tblOff, idx, idxOff, w, wOff, output.GetDataArray(), rows, numEntries, k, channels);
        }

        if (DifferentiableOps.IsRecording<T>())
        {
            DifferentiableOps.RecordIfActive(
                "TopKWeightedRowMix",
                output,
                new[] { table, weights },
                TopKWeightedRowMixAutoBackward<T>,
                savedState: new object[] { CloneIndicesHost(indices) });
        }

        return output;
    }

    /// <summary>
    /// Gradient of <see cref="TopKWeightedRowMix{T}"/> with respect to <c>weights</c>:
    /// <c>gradWeights[..., j] = sum_c gradOutput[..., c] * table[indices[..., j], c]</c>.
    /// </summary>
    public virtual Tensor<T> TopKWeightedRowMixBackwardWeights<T>(
        Tensor<T> gradOutput, Tensor<T> table, Tensor<int> indices)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (table is null) throw new ArgumentNullException(nameof(table));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (table.Rank != 2) throw new ArgumentException("table must be rank-2 [numEntries, channels].", nameof(table));
        if (gradOutput.Rank < 1 || indices.Rank != gradOutput.Rank)
            throw new ArgumentException("gradOutput and indices must share rank >= 1.", nameof(gradOutput));
        for (int d = 0; d < gradOutput.Rank - 1; d++)
            if (gradOutput._shape[d] != indices._shape[d])
                throw new ArgumentException("gradOutput and indices leading dimensions differ.", nameof(indices));

        int numEntries = table._shape[0], channels = table._shape[1];
        if (gradOutput._shape[gradOutput.Rank - 1] != channels)
            throw new ArgumentException($"gradOutput trailing dim ({gradOutput._shape[gradOutput.Rank - 1]}) != table channels ({channels}).", nameof(gradOutput));

        int k = indices._shape[indices.Rank - 1];
        int rows = channels == 0 ? LeadingProduct(gradOutput._shape, gradOutput.Rank - 1) : gradOutput.Length / channels;
        var gradWeights = new Tensor<T>((int[])indices._shape.Clone());
        if (gradWeights.Length > 0 && channels > 0)
        {
            var go = HostArray(gradOutput, out int goOff);
            var tbl = HostArray(table, out int tblOff);
            var idx = HostArray(indices, out int idxOff);
            TopKMixBackwardWeightsHost(go, goOff, tbl, tblOff, idx, idxOff, gradWeights.GetDataArray(),
                rows, numEntries, k, channels);
        }
        return gradWeights;
    }

    /// <summary>
    /// Gradient of <see cref="TopKWeightedRowMix{T}"/> with respect to <c>table</c>:
    /// <c>gradTable[g, c] = sum_{..., j : indices[..., j] == g} weights[..., j] * gradOutput[..., c]</c>,
    /// accumulated in fixed ascending row and slot order (deterministic).
    /// </summary>
    public virtual Tensor<T> TopKWeightedRowMixBackwardTable<T>(
        Tensor<T> gradOutput, Tensor<int> indices, Tensor<T> weights, int numEntries)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (numEntries < 1) throw new ArgumentOutOfRangeException(nameof(numEntries), "numEntries must be positive.");
        if (!ShapesEqual(indices._shape, weights._shape))
            throw new ArgumentException("indices and weights shapes must match.", nameof(weights));
        if (gradOutput.Rank < 1 || gradOutput.Rank != indices.Rank)
            throw new ArgumentException("gradOutput and indices must share rank >= 1.", nameof(gradOutput));
        for (int d = 0; d < gradOutput.Rank - 1; d++)
            if (gradOutput._shape[d] != indices._shape[d])
                throw new ArgumentException("gradOutput and indices leading dimensions differ.", nameof(indices));

        int channels = gradOutput._shape[gradOutput.Rank - 1];
        int k = indices._shape[indices.Rank - 1];
        int rows = channels == 0 ? LeadingProduct(gradOutput._shape, gradOutput.Rank - 1) : gradOutput.Length / channels;
        var gradTable = new Tensor<T>(new[] { numEntries, channels });
        if (gradTable.Length > 0 && rows > 0 && k > 0)
        {
            var go = HostArray(gradOutput, out int goOff);
            var idx = HostArray(indices, out int idxOff);
            var w = HostArray(weights, out int wOff);
            TopKMixBackwardTableHost(go, goOff, idx, idxOff, w, wOff, gradTable.GetDataArray(),
                rows, numEntries, k, channels);
        }
        return gradTable;
    }

    /// <summary>
    /// Counts occurrences of each integer bin <c>0 .. numBins - 1</c> across all elements of <paramref name="indices"/>
    /// into a fixed-shape <c>[numBins]</c> tensor of <typeparamref name="T"/> without host synchronization on GPU.
    /// Values outside <c>[0, numBins)</c> are ignored. Because <paramref name="numBins"/> fixes the output shape,
    /// MoE expert load-balancing statistics stay device-resident with zero <c>ToArray()</c> pipeline stalls.
    /// </summary>
    public virtual Tensor<T> TensorCountIndices<T>(Tensor<int> indices, int numBins)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (numBins < 0) throw new ArgumentOutOfRangeException(nameof(numBins), "numBins must be non-negative.");

        if (typeof(T) != typeof(int))
        {
            GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.HeterogeneousInput);
            if (indices.LazySource is not null)
                GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.MixedElementTypes);
        }
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedIndices = indices;
            // Piecewise-constant integer histogram: non-differentiable. When T == int, register indices directly;
            // otherwise leaf indices are retained via savedState while graph-produced int indices were rejected above.
            var graphInput = typeof(T) == typeof(int)
                ? (Tensor<T>)(object)indices
                : new Tensor<T>(new[] { 1 });
            return scope.RecordUnary(
                LazyNodeType.Custom,
                "TensorCountIndices",
                graphInput,
                new[] { numBins },
                (eng, result) =>
                {
                    var sr = eng.As<CpuEngine>() ?? new CpuEngine();
                    DirectGpuTensorEngine.CopyResultInto(eng, sr.TensorCountIndices<T>(capturedIndices, numBins), result);
                },
                savedState: new object[] { indices });
        }

        var counts = new Tensor<T>(new[] { numBins });
        if (numBins == 0 || indices.Length == 0)
            return counts;

        var idx = HostArray(indices, out int idxOff);
        int len = indices.Length;
        var intCounts = new int[numBins];
        for (int i = 0; i < len; i++)
        {
            int b = idx[idxOff + i];
            if ((uint)b < (uint)numBins)
                intCounts[b]++;
        }

        var dst = counts.GetDataArray();
        if (dst is float[] dstF)
        {
            for (int b = 0; b < numBins; b++) dstF[b] = intCounts[b];
        }
        else if (dst is double[] dstD)
        {
            for (int b = 0; b < numBins; b++) dstD[b] = intCounts[b];
        }
        else if (dst is int[] dstI)
        {
            Array.Copy(intCounts, dstI, numBins);
        }
        else
        {
            CopyCountsHostGeneric(intCounts, dst, numBins);
        }

        return counts;
    }

    [ScalarKernel("Element types other than float, double, and int have no typed conversion; those three take the typed loops above.")]
    private static void CopyCountsHostGeneric<T>(int[] intCounts, T[] dst, int numBins)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int b = 0; b < numBins; b++) dst[b] = ops.FromDouble(intCounts[b]);
    }

    internal static void TensorGatherRowValuesAutoBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (!DifferentiableOps.IsGradientRequired(inputs[0]))
            return;
        var savedIndices = (Tensor<int>)savedState[0];
        int numCols = (int)savedState[1];
        var sr = engine.As<CpuEngine>() ?? new CpuEngine();
        DifferentiableOps.AccumulateGrad(
            grads, inputs[0], sr.TensorGatherRowValuesBackward(gradOutput, savedIndices, numCols), engine);
    }

    internal static void TopKWeightedRowMixAutoBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var table = inputs[0];
        var weights = inputs[1];
        var savedIndices = (Tensor<int>)savedState[0];
        var sr = engine.As<CpuEngine>() ?? new CpuEngine();
        if (DifferentiableOps.IsGradientRequired(table))
        {
            DifferentiableOps.AccumulateGrad(
                grads, table, sr.TopKWeightedRowMixBackwardTable(gradOutput, savedIndices, weights, table._shape[0]), engine);
        }
        if (DifferentiableOps.IsGradientRequired(weights))
        {
            DifferentiableOps.AccumulateGrad(
                grads, weights, sr.TopKWeightedRowMixBackwardWeights(gradOutput, table, savedIndices), engine);
        }
    }

    private static Tensor<int> CloneIndicesHost(Tensor<int> indices)
    {
        var shape = (int[])indices._shape.Clone();
        if (indices.Length == 0)
            return new Tensor<int>(shape);
        var arr = HostArray(indices, out int off);
        var copy = new int[indices.Length];
        Array.Copy(arr, off, copy, 0, indices.Length);
        return new Tensor<int>(copy, shape);
    }

    private static int LeadingProduct(int[] shape, int count)
    {
        int p = 1;
        for (int i = 0; i < count; i++) p = checked(p * shape[i]);
        return p;
    }

    private static (int Rows, int Cols, int K, int[] OutShape) GatherRowValuesDims<T>(Tensor<T> values, Tensor<int> indices)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (values.Rank < 1)
            throw new ArgumentException("values must have rank >= 1 [..., cols].", nameof(values));
        if (indices.Rank != values.Rank)
            throw new ArgumentException($"indices rank ({indices.Rank}) must match values rank ({values.Rank}).", nameof(indices));
        for (int d = 0; d < values.Rank - 1; d++)
            if (values._shape[d] != indices._shape[d])
                throw new ArgumentException($"Leading dimension {d} mismatch: values={values._shape[d]}, indices={indices._shape[d]}.", nameof(indices));
        int cols = values._shape[values.Rank - 1];
        if (cols < 1)
            throw new ArgumentException("values trailing dimension must be at least 1.", nameof(values));
        int k = indices._shape[indices.Rank - 1];
        if (k < 0)
            throw new ArgumentException("indices trailing dimension must be non-negative.", nameof(indices));
        int rows = values.Length / cols;
        return (rows, cols, k, (int[])indices._shape.Clone());
    }

    private static (int Rows, int NumEntries, int K, int Channels, int[] OutShape) TopKMixDims<T>(
        Tensor<T> table, Tensor<int> indices, Tensor<T> weights)
    {
        if (table is null) throw new ArgumentNullException(nameof(table));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (table.Rank != 2)
            throw new ArgumentException($"table must be rank-2 [numEntries, channels]; got rank {table.Rank}.", nameof(table));
        if (indices.Rank < 1)
            throw new ArgumentException("indices must have rank >= 1 [..., K].", nameof(indices));
        if (!ShapesEqual(indices._shape, weights._shape))
            throw new ArgumentException("indices and weights shapes must match.", nameof(weights));

        int numEntries = table._shape[0], channels = table._shape[1];
        if (numEntries < 1)
            throw new ArgumentException("table must have at least one row.", nameof(table));
        int k = indices._shape[indices.Rank - 1];
        int rows = k == 0 ? LeadingProduct(indices._shape, indices.Rank - 1) : indices.Length / k;
        var outShape = (int[])indices._shape.Clone();
        outShape[outShape.Length - 1] = channels;
        return (rows, numEntries, k, channels, outShape);
    }

    private static void GatherRowValuesForwardHost<T>(T[] v, int vOff, int[] idx, int idxOff, T[] y,
        int rows, int cols, int k)
    {
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * k * 4, r =>
        {
            int vBase = vOff + r * cols;
            int kBase = r * k;
            for (int j = 0; j < k; j++)
            {
                int col = idx[idxOff + kBase + j];
                y[kBase + j] = (uint)col < (uint)cols ? v[vBase + col] : zero;
            }
        }, deterministicSafe: true);
    }

    private static void GatherRowValuesBackwardHost<T>(T[] g, int gOff, int[] idx, int idxOff, T[] dv,
        int rows, int cols, int k)
    {
        if (g is float[] gF && dv is float[] dvF)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * (cols + k), r =>
            {
                int dvBase = r * cols;
                Array.Clear(dvF, dvBase, cols);
                int kBase = r * k;
                for (int j = 0; j < k; j++)
                {
                    int col = idx[idxOff + kBase + j];
                    if ((uint)col < (uint)cols)
                        dvF[dvBase + col] += gF[gOff + kBase + j];
                }
            }, deterministicSafe: true);
        }
        else if (g is double[] gD && dv is double[] dvD)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * (cols + k), r =>
            {
                int dvBase = r * cols;
                Array.Clear(dvD, dvBase, cols);
                int kBase = r * k;
                for (int j = 0; j < k; j++)
                {
                    int col = idx[idxOff + kBase + j];
                    if ((uint)col < (uint)cols)
                        dvD[dvBase + col] += gD[gOff + kBase + j];
                }
            }, deterministicSafe: true);
        }
        else
        {
            GatherRowValuesBackwardHostGeneric(g, gOff, idx, idxOff, dv, rows, cols, k);
        }
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the typed loops above.")]
    private static void GatherRowValuesBackwardHostGeneric<T>(T[] g, int gOff, int[] idx, int idxOff, T[] dv,
        int rows, int cols, int k)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * (cols + k), r =>
        {
            int dvBase = r * cols;
            for (int c = 0; c < cols; c++) dv[dvBase + c] = ops.Zero;
            int kBase = r * k;
            for (int j = 0; j < k; j++)
            {
                int col = idx[idxOff + kBase + j];
                if ((uint)col < (uint)cols)
                    dv[dvBase + col] = ops.Add(dv[dvBase + col], g[gOff + kBase + j]);
            }
        }, deterministicSafe: true);
    }

    private static void TopKMixForwardHost<T>(T[] tbl, int tblOff, int[] idx, int idxOff, T[] w, int wOff, T[] y,
        int rows, int numEntries, int k, int channels)
    {
        if (tbl is float[] tblF && w is float[] wF && y is float[] yF)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * k * channels, r =>
            {
                int outBase = r * channels;
                Array.Clear(yF, outBase, channels);
                int kBase = r * k;
                for (int j = 0; j < k; j++)
                {
                    int entry = idx[idxOff + kBase + j];
                    if ((uint)entry < (uint)numEntries)
                        ScaledAccumulate(yF, outBase, wF[wOff + kBase + j], tblF, tblOff + entry * channels, channels);
                }
            }, deterministicSafe: true);
        }
        else if (tbl is double[] tblD && w is double[] wD && y is double[] yD)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * k * channels, r =>
            {
                int outBase = r * channels;
                Array.Clear(yD, outBase, channels);
                int kBase = r * k;
                for (int j = 0; j < k; j++)
                {
                    int entry = idx[idxOff + kBase + j];
                    if ((uint)entry < (uint)numEntries)
                        ScaledAccumulate(yD, outBase, wD[wOff + kBase + j], tblD, tblOff + entry * channels, channels);
                }
            }, deterministicSafe: true);
        }
        else
        {
            TopKMixForwardHostGeneric(tbl, tblOff, idx, idxOff, w, wOff, y, rows, numEntries, k, channels);
        }
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the SIMD loops above.")]
    private static void TopKMixForwardHostGeneric<T>(T[] tbl, int tblOff, int[] idx, int idxOff, T[] w, int wOff, T[] y,
        int rows, int numEntries, int k, int channels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * k * channels, r =>
        {
            int outBase = r * channels;
            for (int c = 0; c < channels; c++) y[outBase + c] = ops.Zero;
            int kBase = r * k;
            for (int j = 0; j < k; j++)
            {
                int entry = idx[idxOff + kBase + j];
                if ((uint)entry >= (uint)numEntries) continue;
                T wj = w[wOff + kBase + j];
                int tBase = tblOff + entry * channels;
                for (int c = 0; c < channels; c++)
                    y[outBase + c] = ops.Add(y[outBase + c], ops.Multiply(wj, tbl[tBase + c]));
            }
        }, deterministicSafe: true);
    }

    private static void TopKMixBackwardWeightsHost<T>(T[] go, int goOff, T[] tbl, int tblOff, int[] idx, int idxOff, T[] dw,
        int rows, int numEntries, int k, int channels)
    {
        if (go is float[] goF && tbl is float[] tblF && dw is float[] dwF)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows * k, (long)rows * k * channels, rk =>
            {
                int r = rk / k;
                int entry = idx[idxOff + rk];
                dwF[rk] = (uint)entry < (uint)numEntries
                    ? DotRow(goF, goOff + r * channels, tblF, tblOff + entry * channels, channels)
                    : 0f;
            }, deterministicSafe: true);
        }
        else if (go is double[] goD && tbl is double[] tblD && dw is double[] dwD)
        {
            CpuParallelSettings.ParallelForOrSerial(0, rows * k, (long)rows * k * channels, rk =>
            {
                int r = rk / k;
                int entry = idx[idxOff + rk];
                dwD[rk] = (uint)entry < (uint)numEntries
                    ? DotRow(goD, goOff + r * channels, tblD, tblOff + entry * channels, channels)
                    : 0.0;
            }, deterministicSafe: true);
        }
        else
        {
            TopKMixBackwardWeightsHostGeneric(go, goOff, tbl, tblOff, idx, idxOff, dw, rows, numEntries, k, channels);
        }
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the SIMD loops above.")]
    private static void TopKMixBackwardWeightsHostGeneric<T>(T[] go, int goOff, T[] tbl, int tblOff, int[] idx, int idxOff, T[] dw,
        int rows, int numEntries, int k, int channels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, rows * k, (long)rows * k * channels, rk =>
        {
            int r = rk / k;
            int entry = idx[idxOff + rk];
            if ((uint)entry >= (uint)numEntries)
            {
                dw[rk] = ops.Zero;
                return;
            }
            T acc = ops.Zero;
            int goBase = goOff + r * channels, tBase = tblOff + entry * channels;
            for (int c = 0; c < channels; c++)
                acc = ops.Add(acc, ops.Multiply(go[goBase + c], tbl[tBase + c]));
            dw[rk] = acc;
        }, deterministicSafe: true);
    }

    private static (int[] Offsets, int[] SortedSlots) BuildCpuEntryCsr(
        int[] idx, int idxOff, int rows, int k, int numEntries)
    {
        int totalSlots = checked(rows * k);
        var offsets = new int[checked(numEntries + 1)];
        for (int rk = 0; rk < totalSlots; rk++)
        {
            int entry = idx[idxOff + rk];
            if ((uint)entry < (uint)numEntries)
                offsets[entry + 1]++;
        }
        for (int g = 1; g <= numEntries; g++)
            offsets[g] += offsets[g - 1];

        var cursor = (int[])offsets.Clone();
        var sortedSlots = new int[offsets[numEntries]];
        for (int rk = 0; rk < totalSlots; rk++)
        {
            int entry = idx[idxOff + rk];
            if ((uint)entry < (uint)numEntries)
                sortedSlots[cursor[entry]++] = rk;
        }
        return (offsets, sortedSlots);
    }

    private static void TopKMixBackwardTableHost<T>(T[] go, int goOff, int[] idx, int idxOff, T[] w, int wOff, T[] dt,
        int rows, int numEntries, int k, int channels)
    {
        var (offsets, sortedSlots) = BuildCpuEntryCsr(idx, idxOff, rows, k, numEntries);
        long workEstimate = ((long)sortedSlots.Length + numEntries) * channels;
        int blocks = (channels + SparseRoutingChannelBlock - 1) / SparseRoutingChannelBlock;
        if (go is float[] goF && w is float[] wF && dt is float[] dtF)
        {
            // One (entry g, channel block) per work item, iterating g's CSR slots in ascending (row, slot) order.
            CpuParallelSettings.ParallelForOrSerial(0, numEntries * blocks, workEstimate, item =>
            {
                int g = item / blocks, c0 = (item % blocks) * SparseRoutingChannelBlock;
                int len = Math.Min(SparseRoutingChannelBlock, channels - c0), dst = g * channels + c0;
                Array.Clear(dtF, dst, len);
                int start = offsets[g], end = offsets[g + 1];
                for (int p = start; p < end; p++)
                {
                    int rk = sortedSlots[p];
                    int r = rk / k;
                    ScaledAccumulate(dtF, dst, wF[wOff + rk], goF, goOff + r * channels + c0, len);
                }
            }, deterministicSafe: true);
        }
        else if (go is double[] goD && w is double[] wD && dt is double[] dtD)
        {
            CpuParallelSettings.ParallelForOrSerial(0, numEntries * blocks, workEstimate, item =>
            {
                int g = item / blocks, c0 = (item % blocks) * SparseRoutingChannelBlock;
                int len = Math.Min(SparseRoutingChannelBlock, channels - c0), dst = g * channels + c0;
                Array.Clear(dtD, dst, len);
                int start = offsets[g], end = offsets[g + 1];
                for (int p = start; p < end; p++)
                {
                    int rk = sortedSlots[p];
                    int r = rk / k;
                    ScaledAccumulate(dtD, dst, wD[wOff + rk], goD, goOff + r * channels + c0, len);
                }
            }, deterministicSafe: true);
        }
        else
        {
            TopKMixBackwardTableHostGeneric(go, goOff, w, wOff, dt, offsets, sortedSlots, numEntries, k, channels, workEstimate);
        }
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the SIMD loops above.")]
    private static void TopKMixBackwardTableHostGeneric<T>(T[] go, int goOff, T[] w, int wOff, T[] dt,
        int[] offsets, int[] sortedSlots, int numEntries, int k, int channels, long workEstimate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, numEntries, workEstimate, g =>
        {
            int dst = g * channels;
            for (int c = 0; c < channels; c++) dt[dst + c] = ops.Zero;
            int start = offsets[g], end = offsets[g + 1];
            for (int p = start; p < end; p++)
            {
                int rk = sortedSlots[p];
                int r = rk / k;
                T wj = w[wOff + rk];
                int goBase = goOff + r * channels;
                for (int c = 0; c < channels; c++)
                    dt[dst + c] = ops.Add(dt[dst + c], ops.Multiply(wj, go[goBase + c]));
            }
        }, deterministicSafe: true);
    }

    private static void ScaledAccumulate(float[] acc, int accOff, float scale, float[] src, int srcOff, int n)
    {
        int width = System.Numerics.Vector<float>.Count, i = 0;
        var vs = new System.Numerics.Vector<float>(scale);
        for (; i <= n - width; i += width)
            (new System.Numerics.Vector<float>(acc, accOff + i)
                + vs * new System.Numerics.Vector<float>(src, srcOff + i)).CopyTo(acc, accOff + i);
        for (; i < n; i++) acc[accOff + i] += scale * src[srcOff + i];
    }

    private static void ScaledAccumulate(double[] acc, int accOff, double scale, double[] src, int srcOff, int n)
    {
        int width = System.Numerics.Vector<double>.Count, i = 0;
        var vs = new System.Numerics.Vector<double>(scale);
        for (; i <= n - width; i += width)
            (new System.Numerics.Vector<double>(acc, accOff + i)
                + vs * new System.Numerics.Vector<double>(src, srcOff + i)).CopyTo(acc, accOff + i);
        for (; i < n; i++) acc[accOff + i] += scale * src[srcOff + i];
    }

    private static float DotRow(float[] a, int aOff, float[] b, int bOff, int n)
    {
        int width = System.Numerics.Vector<float>.Count, i = 0;
        var vAcc = System.Numerics.Vector<float>.Zero;
        for (; i <= n - width; i += width)
            vAcc += new System.Numerics.Vector<float>(a, aOff + i) * new System.Numerics.Vector<float>(b, bOff + i);
        float acc = 0f;
        for (int lane = 0; lane < width; lane++) acc += vAcc[lane];
        for (; i < n; i++) acc += a[aOff + i] * b[bOff + i];
        return acc;
    }

    private static double DotRow(double[] a, int aOff, double[] b, int bOff, int n)
    {
        int width = System.Numerics.Vector<double>.Count, i = 0;
        var vAcc = System.Numerics.Vector<double>.Zero;
        for (; i <= n - width; i += width)
            vAcc += new System.Numerics.Vector<double>(a, aOff + i) * new System.Numerics.Vector<double>(b, bOff + i);
        double acc = 0.0;
        for (int lane = 0; lane < width; lane++) acc += vAcc[lane];
        for (; i < n; i++) acc += a[aOff + i] * b[bOff + i];
        return acc;
    }
}

/// <summary>
/// <see cref="IEngine"/> extension methods for sparse top-K MoE / codebook routing primitives, preserving binary and
/// source compatibility for external <see cref="IEngine"/> implementations.
/// </summary>
public static class SparseRoutingEngineExtensions
{
    /// <inheritdoc cref="CpuEngine.TensorGatherRowValues{T}"/>
    public static Tensor<T> TensorGatherRowValues<T>(this IEngine engine, Tensor<T> values, Tensor<int> indices)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TensorGatherRowValues(values, indices);
        return new CpuEngine().TensorGatherRowValues(values, indices);
    }

    /// <inheritdoc cref="CpuEngine.TensorGatherRowValuesBackward{T}"/>
    public static Tensor<T> TensorGatherRowValuesBackward<T>(
        this IEngine engine, Tensor<T> gradOutput, Tensor<int> indices, int numCols)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TensorGatherRowValuesBackward(gradOutput, indices, numCols);
        return new CpuEngine().TensorGatherRowValuesBackward(gradOutput, indices, numCols);
    }

    /// <inheritdoc cref="CpuEngine.TopKWeightedRowMix{T}"/>
    public static Tensor<T> TopKWeightedRowMix<T>(
        this IEngine engine, Tensor<T> table, Tensor<int> indices, Tensor<T> weights)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TopKWeightedRowMix(table, indices, weights);
        return new CpuEngine().TopKWeightedRowMix(table, indices, weights);
    }

    /// <inheritdoc cref="CpuEngine.TopKWeightedRowMixBackwardWeights{T}"/>
    public static Tensor<T> TopKWeightedRowMixBackwardWeights<T>(
        this IEngine engine, Tensor<T> gradOutput, Tensor<T> table, Tensor<int> indices)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TopKWeightedRowMixBackwardWeights(gradOutput, table, indices);
        return new CpuEngine().TopKWeightedRowMixBackwardWeights(gradOutput, table, indices);
    }

    /// <inheritdoc cref="CpuEngine.TopKWeightedRowMixBackwardTable{T}"/>
    public static Tensor<T> TopKWeightedRowMixBackwardTable<T>(
        this IEngine engine, Tensor<T> gradOutput, Tensor<int> indices, Tensor<T> weights, int numEntries)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TopKWeightedRowMixBackwardTable(gradOutput, indices, weights, numEntries);
        return new CpuEngine().TopKWeightedRowMixBackwardTable(gradOutput, indices, weights, numEntries);
    }

    /// <inheritdoc cref="CpuEngine.TensorCountIndices{T}"/>
    public static Tensor<T> TensorCountIndices<T>(this IEngine engine, Tensor<int> indices, int numBins)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (engine.TryGet<CpuEngine>(out var cpuEngine))
            return cpuEngine.TensorCountIndices<T>(indices, numBins);
        return new CpuEngine().TensorCountIndices<T>(indices, numBins);
    }
}
