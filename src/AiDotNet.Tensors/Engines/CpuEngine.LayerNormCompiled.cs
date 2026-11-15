using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Compiled-graph LayerNorm for a float32 host plan: one node whose forward normalizes straight into the node's
    /// output buffer and refreshes node-owned mean/variance rows, and whose backward writes dX, dGamma and dBeta straight
    /// into their gradient accumulators. The generic node it replaces ran an eager LayerNorm every step -- three fresh
    /// output tensors, the result copied into the plan buffer -- and an eager backward that allocated three gradients
    /// for AccumulateGrad to copy again. Returns null (the caller records the generic node) for a layout this does not
    /// handle: views with a storage offset, or gamma/beta that do not match the trailing input dimensions.
    /// </summary>
    private static Tensor<float>? TryRecordLayerNormFloat(
        LazyTensorScope scope, Tensor<float> input, Tensor<float> gamma, Tensor<float> beta, double epsilon,
        out Tensor<float>? mean, out Tensor<float>? variance)
    {
        mean = variance = null;
        int features = gamma.Length;
        if (features == 0 || beta.Length != features || input.Length % features != 0) return null;
        int normDims = gamma.Rank;
        if (normDims > input.Rank) return null;
        for (int i = 0; i < normDims; i++)
            if (gamma._shape[i] != input._shape[input.Rank - normDims + i]) return null;
        if (!IsZeroOffsetContiguous(input) || !IsZeroOffsetContiguous(gamma) || !IsZeroOffsetContiguous(beta))
            return null;

        int rows = input.Length / features;
        var batchShape = new int[Math.Max(1, input.Rank - normDims)];
        if (input.Rank - normDims == 0) batchShape[0] = 1;
        else for (int i = 0; i < batchShape.Length; i++) batchShape[i] = input._shape[i];
        // A probe trace never computes: the mean and variance are shape-only probe values, and the node's statistics
        // buffers have metadata only.
        bool probe = GraphMode.IsProbe;
        var meanT = probe ? Tensor<float>.CreateProbeStandIn(batchShape) : new Tensor<float>(batchShape);
        var varT = probe ? Tensor<float>.CreateProbeStandIn(batchShape) : new Tensor<float>(batchShape);
        var stats = probe
            ? new LayerNormStats(rows)
            : new LayerNormStats(meanT.GetCpuBackingForContiguousWrite(out _)!, varT.GetCpuBackingForContiguousWrite(out _)!);
        float eps = (float)epsilon;
        var shape = new LayerNormRowShape(rows, features, eps);

        void Forward(Tensor<float> output)
        {
            var o = output.GetCpuBackingForContiguousWrite(out int oOff);
            var x = input.GetCpuBackingForStridedRead(out _)!;
            var g = gamma.GetCpuBackingForStridedRead(out _)!;
            var b = beta.GetCpuBackingForStridedRead(out _)!;
            if (o is not null && oOff == 0)
            {
                ProcessBatchesSimd(x, g, b, o, stats.Mean, stats.Variance, rows, features, eps);
            }
            else
            {
                var staged = new float[input.Length];
                ProcessBatchesSimd(x, g, b, staged, stats.Mean, stats.Variance, rows, features, eps);
                staged.AsSpan().CopyTo(output.AsWritableSpan());
            }
            output.IncrementVersion();
            meanT.IncrementVersion();
            varT.IncrementVersion();
        }

        var result = scope.RecordVariadic(LazyNodeType.Custom, "LayerNorm", new[] { input, gamma, beta },
            (int[])input._shape.Clone(),
            (eng, output) =>
            {
                if (output._gpuBuffer is null && !output.HasPendingGpuData && output.IsContiguous) Forward(output);
                else
                {
                    var staged = new Tensor<float>(input._shape);
                    Forward(staged);
                    DirectGpuTensorEngine.CopyResultInto(eng, staged, output);
                }
            },
            LayerNormBackwardCompiledFloat, new object[] { stats, shape });
        // Values at trace time too, as the generic node had: code between ops may read them. Computed outside the
        // trace, so reading the traced input here is the op's own work, not a host read by the objective.
        if (!probe)
            using (GraphMode.SuspendRecording())
                Forward(result);
        mean = meanT;
        variance = varT;
        return result;
    }

    // An unrealized traced value is checked from its layout alone: fetching its backing would compute it mid-trace.
    private static bool IsZeroOffsetContiguous(Tensor<float> t) =>
        t.IsContiguous && (t.LazySource is { IsRealized: false }
            ? t.StorageOffset == 0
            : t.GetCpuBackingForStridedRead(out int off) is not null && off == 0);

    /// <summary>
    /// The per-row mean and variance a compiled LayerNorm node writes in its forward and reads in its backward: this
    /// step's statistics, so a closure-objective signature compares only their size.
    /// </summary>
    internal sealed class LayerNormStats : ISignatureScratch
    {
        public LayerNormStats(float[] mean, float[] variance) { Mean = mean; Variance = variance; SignatureLength = mean.Length; }
        internal LayerNormStats(int rows) { Mean = Variance = Array.Empty<float>(); SignatureLength = rows; }

        public float[] Mean { get; }
        public float[] Variance { get; }
        public int SignatureLength { get; }
    }

    internal sealed class LayerNormRowShape
    {
        public LayerNormRowShape(int rows, int features, float epsilon)
        {
            Rows = rows; Features = features; Epsilon = epsilon;
        }

        public int Rows { get; }
        public int Features { get; }
        public float Epsilon { get; }
    }

    /// <summary>
    /// dX = invStd * (g - mean(g) - xhat * mean(g * xhat)) with g = gamma * dY and xhat = (x - mean) * invStd;
    /// dGamma = sum over rows of dY * xhat; dBeta = sum over rows of dY. Runs the eager backward's kernel
    /// (<see cref="LayerNormBackwardFloatFused"/>): one parallel pass over fixed row blocks, vectorized, partials added in
    /// block order, so the compiled and eager gradients agree bit for bit at any thread count. Its own scalar loop,
    /// which updated the dGamma/dBeta partials element by element, took ~170 us per [2048,64] norm on the parity Transformer (Ryzen 7 4800H).
    /// </summary>
    private static void LayerNormBackwardCompiledFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output, object[] savedState, IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var stats = (LayerNormStats)savedState[0];
        var meanArr = stats.Mean;
        var varArr = stats.Variance;
        var s = (LayerNormRowShape)savedState[1];
        Tensor<float> input = inputs[0], gamma = inputs[1], beta = inputs[2];
        bool needX = DifferentiableOps.IsGradientRequired(input);
        bool needG = DifferentiableOps.IsGradientRequired(gamma);
        bool needB = DifferentiableOps.IsGradientRequired(beta);
        if (!needX && !needG && !needB) return;

        var x = ReadableBacking(input, out int xOff);
        var gm = ReadableBacking(gamma, out int gmOff);
        var dy = ReadableBacking(gradOutput, out int dyOff);
        bool distinct = !ReferenceEquals(gamma, beta) && !ReferenceEquals(input, gamma) && !ReferenceEquals(input, beta);
        var dX = GradTarget(needX, input, grads, engine, distinct);
        var dG = GradTarget(needG, gamma, grads, engine, distinct);
        var dB = GradTarget(needB, beta, grads, engine, distinct);

        // The eager backward's SIMD kernel (one pass over row blocks), writing each gradient where GradTarget points.
        int fs = s.Features;
        var partial = System.Buffers.ArrayPool<float>.Shared.Rent(LayerNormBackwardFloatScratchLength(s.Rows, fs));
        try
        {
            LayerNormBackwardFloatFused(dy, dyOff, x, xOff, gm, gmOff, meanArr, varArr, s.Rows, fs, s.Epsilon,
                dX.Array, dX.Offset, !dX.Overwrite, dG.Array, dG.Offset, !dG.Overwrite, dB.Array, dB.Offset, !dB.Overwrite,
                partial);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(partial);
        }

        dX.Commit(grads, input, engine);
        dG.Commit(grads, gamma, engine);
        dB.Commit(grads, beta, engine);
    }
}
