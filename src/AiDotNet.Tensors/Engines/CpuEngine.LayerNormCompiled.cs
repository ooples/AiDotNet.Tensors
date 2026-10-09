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
        var meanT = new Tensor<float>(batchShape);
        var varT = new Tensor<float>(batchShape);
        var meanArr = meanT.GetCpuBackingForContiguousWrite(out _)!;
        var varArr = varT.GetCpuBackingForContiguousWrite(out _)!;
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
                ProcessBatchesSimd(x, g, b, o, meanArr, varArr, rows, features, eps);
            }
            else
            {
                var staged = new float[input.Length];
                ProcessBatchesSimd(x, g, b, staged, meanArr, varArr, rows, features, eps);
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
            LayerNormBackwardCompiledFloat, new object[] { meanArr, varArr, shape });
        // Values at trace time too, as the generic node had: code between ops may read them.
        Forward(result);
        mean = meanT;
        variance = varT;
        return result;
    }

    private static bool IsZeroOffsetContiguous(Tensor<float> t) =>
        t.IsContiguous && t.GetCpuBackingForStridedRead(out int off) is not null && off == 0;

    internal sealed class LayerNormRowShape
    {
        public LayerNormRowShape(int rows, int features, float epsilon)
        {
            Rows = rows; Features = features; Epsilon = epsilon;
            // Fixed row chunking (a function of the shape only), so the dGamma/dBeta partial sums are combined in the
            // same order whatever the thread count: results are bit-reproducible.
            Chunks = Math.Max(1, Math.Min(64, rows / 32));
        }

        public int Rows { get; }
        public int Features { get; }
        public float Epsilon { get; }
        public int Chunks { get; }
    }

    /// <summary>
    /// dX = invStd * (g - mean(g) - xhat * mean(g * xhat)) with g = gamma * dY and xhat = (x - mean) * invStd;
    /// dGamma = sum over rows of dY * xhat; dBeta = sum over rows of dY. One parallel pass over fixed row chunks: each
    /// writes its dX rows and its own dGamma/dBeta partial, and the partials are added in chunk order.
    /// </summary>
    private static void LayerNormBackwardCompiledFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output, object[] savedState, IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var meanArr = (float[])savedState[0];
        var varArr = (float[])savedState[1];
        var s = (LayerNormRowShape)savedState[2];
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

        int fs = s.Features, chunks = s.Chunks;
        float[]? partial = needG || needB ? System.Buffers.ArrayPool<float>.Shared.Rent(chunks * 2 * fs) : null;
        try
        {
            CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)s.Rows * fs * 4, chunk =>
            {
                int r0 = (int)((long)chunk * s.Rows / chunks), r1 = (int)((long)(chunk + 1) * s.Rows / chunks);
                int pg = chunk * 2 * fs, pb = pg + fs;
                if (partial is not null) Array.Clear(partial, pg, 2 * fs);
                float invFs = 1f / fs;
                for (int r = r0; r < r1; r++)
                {
                    int xr = xOff + r * fs, dyr = dyOff + r * fs;
                    float m = meanArr[r];
                    float invStd = 1f / (float)Math.Sqrt(varArr[r] + s.Epsilon);
                    float sumG = 0f, sumGX = 0f;
                    for (int f = 0; f < fs; f++)
                    {
                        float go = dy[dyr + f];
                        float xc = x[xr + f] - m;
                        float g = gm[gmOff + f] * go;
                        sumG += g;
                        sumGX += g * xc;
                        if (partial is not null)
                        {
                            partial[pg + f] += go * xc * invStd;
                            partial[pb + f] += go;
                        }
                    }
                    if (dX.Array is not null)
                    {
                        float meanG = sumG * invFs, meanGX = sumGX * invStd * invFs;
                        int dr = dX.Offset + r * fs;
                        for (int f = 0; f < fs; f++)
                        {
                            float xhat = (x[xr + f] - m) * invStd;
                            float v = invStd * (gm[gmOff + f] * dy[dyr + f] - meanG - xhat * meanGX);
                            if (dX.Overwrite) dX.Array[dr + f] = v; else dX.Array[dr + f] += v;
                        }
                    }
                }
            }, deterministicSafe: true);

            if (partial is not null)
            {
                if (dG.Array is not null) ReducePartials(partial, 0, chunks, fs, dG);
                if (dB.Array is not null) ReducePartials(partial, fs, chunks, fs, dB);
            }
        }
        finally
        {
            if (partial is not null) System.Buffers.ArrayPool<float>.Shared.Return(partial);
        }

        dX.Commit(grads, input, engine);
        dG.Commit(grads, gamma, engine);
        dB.Commit(grads, beta, engine);
    }

    /// <summary>target (=|+=) sum over chunks c of partial[c * 2 * fs + slot .. + fs], in chunk order.</summary>
    private static void ReducePartials(float[] partial, int slot, int chunks, int fs, FusedGradTarget target)
    {
        var dst = target.Array!;
        int off = target.Offset;
        if (target.Overwrite) Array.Clear(dst, off, fs);
        for (int c = 0; c < chunks; c++)
            Axpy(1f, partial, c * 2 * fs + slot, dst, off, fs);
    }
}
