// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;

/// <summary>
/// Reference CPU implementation of <see cref="IDeviceRnn"/>. Mirrors
/// the layout PyTorch's <c>torch.nn.LSTM</c> / <c>torch.nn.GRU</c> /
/// <c>torch.nn.RNN</c> use so weights produced by either runtime are
/// portable between CPU and GPU code paths.
///
/// <para><b>Per-(layer, direction) weight packing.</b> The packed
/// <c>weights</c> tensor concatenates one block per (layer, direction)
/// in the order (l=0, fwd), (l=0, rev), (l=1, fwd), (l=1, rev), ...
/// Each block has layout
/// <c>[W_ih (G·H·I_l) | W_hh (G·H·F_out) | b_ih (G·H) | b_hh (G·H) | W_hr (P·H if proj)]</c>
/// where <c>G</c> is the gate count (4 for LSTM, 3 for GRU, 1 for plain
/// RNN), <c>I_l</c> is <c>inputSize</c> for layer 0 and <c>F_out · D</c>
/// for later layers, <c>F_out</c> is <c>P</c> if projecting else
/// <c>H</c>, and <c>D</c> is the number of directions (1 unidirectional
/// / 2 bidirectional). Gate order matches PyTorch: <c>i, f, g, o</c> for
/// LSTM and <c>r, z, n</c> for GRU.</para>
///
/// <para><b>Dropout</b> is applied to every layer's per-step output
/// except the last layer (PyTorch parity). The mask uses
/// <see cref="RandomHelper.CreateSecureRandom"/> by default; pass a
/// seed via <see cref="RnnOptions.DropoutSeed"/> for reproducibility.</para>
///
/// <para><b>Backward.</b> <see cref="BackwardRnn"/> implements managed
/// BPTT for plain RNN (tanh / relu), LSTM, and GRU — multi-layer,
/// bidirectional, projection, and dropout aware. Caches are
/// iteration-order indexed so the backward loop walks them in lock-step
/// with the forward replay.</para>
/// </summary>
public sealed class CpuRnn : IDeviceRnn
{
    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T>? CN) ForwardRnn<T>(
        RnnCellType cell,
        Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
        Tensor<T> weights, RnnOptions options)
    {
        ValidateOptions(options, cell);
        var ctx = ForwardContext<T>.Build(cell, input, h0, c0, weights, options);
        ForwardStack(ref ctx, saveCache: false);
        return (ctx.LayerInput!, ctx.HN, cell == RnnCellType.Lstm ? ctx.CN : null);
    }

    /// <inheritdoc/>
    public (Tensor<T> Output, Tensor<T> HN, Tensor<T> CN) ForwardLstm<T>(
        Tensor<T> input, Tensor<T> h0, Tensor<T> c0,
        Tensor<T> weights, RnnOptions options)
    {
        ValidateOptions(options, RnnCellType.Lstm);
        var ctx = ForwardContext<T>.Build(RnnCellType.Lstm, input, h0, c0, weights, options);
        ForwardStack(ref ctx, saveCache: false);
        return (ctx.LayerInput!, ctx.HN, ctx.CN!);
    }

    /// <inheritdoc/>
    public (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardRnn<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> output, Tensor<T> hN, Tensor<T>? cN,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN,
            Tensor<T> weights, RnnOptions options)
    {
        // The interface contract takes hN/cN as the *final* state but the closed-form
        // BPTT also needs h0/c0 so it can re-run forward and rebuild the per-step cache.
        // Standard PyTorch trains by saving h0/c0 in the autograd context anyway, so we
        // expose a sibling overload below that takes them explicitly. The interface
        // form falls back to zeros, which matches torch's default initial state.
        _ = output; _ = hN; _ = cN;
        ValidateOptions(options, cell);
        var (_, batch, _) = ParseInputShape(input);
        int direction = options.Bidirectional ? 2 : 1;
        int featOut = options.ProjSize > 0 ? options.ProjSize : options.HiddenSize;
        var h0 = new Tensor<T>(new[] { options.NumLayers * direction, batch, featOut });
        Tensor<T>? c0 = cell == RnnCellType.Lstm
            ? new Tensor<T>(new[] { options.NumLayers * direction, batch, options.HiddenSize })
            : null;
        return BackwardCore(cell, input, h0, c0, weights, options, gradOutput, gradHN, gradCN);
    }

    /// <summary>
    /// Backward overload that takes the original h0/c0 — the path callers should use
    /// during training when the forward's initial state is in scope.
    /// </summary>
    public (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardRnn<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN,
            Tensor<T> weights, RnnOptions options)
    {
        ValidateOptions(options, cell);
        return BackwardCore(cell, input, h0, c0, weights, options, gradOutput, gradHN, gradCN);
    }

    private static void ValidateOptions(RnnOptions options, RnnCellType cell)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (options.HiddenSize <= 0)
            throw new ArgumentException("HiddenSize must be positive.", nameof(options));
        if (options.NumLayers <= 0)
            throw new ArgumentException("NumLayers must be positive.", nameof(options));
        if (options.Dropout < 0 || options.Dropout >= 1)
            throw new ArgumentException("Dropout must be in [0, 1).", nameof(options));
        if (options.ProjSize < 0)
            throw new ArgumentException("ProjSize must be non-negative.", nameof(options));
        if (options.ProjSize > 0 && options.ProjSize >= options.HiddenSize)
            throw new ArgumentException("ProjSize must be strictly smaller than HiddenSize when set.", nameof(options));
        if (options.ProjSize > 0 && cell != RnnCellType.Lstm)
            throw new ArgumentException("LSTM projection (ProjSize > 0) only applies to LSTM cells.", nameof(options));
    }

    private static (int SeqLen, int Batch, int InputSize) ParseInputShape<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException($"Input must be [seqLen, batch, inputSize]; got rank {input.Rank}.", nameof(input));
        return (input._shape[0], input._shape[1], input._shape[2]);
    }

    private static int GateCount(RnnCellType cell) => cell switch
    {
        RnnCellType.Lstm => 4,
        RnnCellType.Gru => 3,
        _ => 1,
    };

    private static long PerLayerDirWeightLen(RnnCellType cell, int inputFeat, int hidden, int proj)
    {
        long G = GateCount(cell);
        long featOut = proj > 0 ? proj : hidden;
        long wIh = G * hidden * inputFeat;
        long wHh = G * hidden * featOut;
        long bIh = G * hidden;
        long bHh = G * hidden;
        long wHr = proj > 0 ? (long)proj * hidden : 0;
        return wIh + wHh + bIh + bHh + wHr;
    }

    private static long ExpectedTotalWeightLength(
        RnnCellType cell, int inputSize, int hidden, int proj, int numLayers, int numDirections)
    {
        long featOut = proj > 0 ? proj : hidden;
        long total = 0;
        for (int l = 0; l < numLayers; l++)
        {
            int layerInput = l == 0 ? inputSize : (int)featOut * numDirections;
            for (int d = 0; d < numDirections; d++)
                total += PerLayerDirWeightLen(cell, layerInput, hidden, proj);
        }
        return total;
    }

    private static void EnsureHiddenStateShape<T>(Tensor<T> h, int leading, int batch, int feature, string name)
    {
        if (h.Rank == 3 && h._shape[0] == leading && h._shape[1] == batch && h._shape[2] == feature) return;
        if (leading == 1 && h.Rank == 2 && h._shape[0] == batch && h._shape[1] == feature) return;
        throw new ArgumentException(
            $"{name} shape mismatch: expected [{leading}, {batch}, {feature}].", name);
    }

    /// <summary>
    /// Holds layout shared between forward and backward; per-(layer,direction) weight slices
    /// are computed via <see cref="LayerDirSlice"/>.
    /// </summary>
    private sealed class Layout
    {
        public RnnCellType Cell;
        public int Gates;
        public int SeqLen, Batch, InputSize;
        public int Hidden, Proj, NumLayers, NumDirections;
        public int FeatOut; // proj > 0 ? proj : hidden
        public int OutputFeature; // FeatOut * NumDirections
    }

    private struct ForwardContext<T>
    {
        public Layout Layout;
        public Tensor<T>? LayerInput; // last layer's output
        public Tensor<T> HN;           // [L*D, B, FeatOut]
        public Tensor<T>? CN;          // [L*D, B, H], LSTM only
        public T[] Weights;
        public Tensor<T> H0;
        public Tensor<T>? C0;
        public RnnOptions Options;
        public Random? DropoutRng;
        // Caches (only filled when saveCache=true)
        public T[][]? CacheLayerInputs;       // per layer ℓ: input fed to that layer (size SeqLen·B·featDim_ℓ)
        public bool[][]? CacheDropoutMasks;   // per layer ℓ: mask used at output of layer ℓ (empty if no dropout)
        public T[][]? CacheH;                 // per (ℓ*D + d): all hidden states across iter, size (SeqLen+1)·B·FeatOut
        public T[][]? CacheC;                 // per (ℓ*D + d) for LSTM: all cell states, size (SeqLen+1)·B·H
        public T[][]? CacheGates;             // per (ℓ*D + d): saved per-step gate post-activations (LSTM/GRU/RNN)
        public T[][]? CacheRawCellOutput;     // per (ℓ*D + d) for LSTM-with-projection: pre-projection h_full

        public static ForwardContext<T> Build(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
            Tensor<T> weights, RnnOptions options)
        {
            if (h0 is null) throw new ArgumentNullException(nameof(h0));
            if (cell == RnnCellType.Lstm && c0 is null)
                throw new ArgumentException("LSTM requires a non-null cell-state c0.", nameof(c0));
            if (weights is null) throw new ArgumentNullException(nameof(weights));

            var (seqLen, batch, inputSize) = ParseInputShape(input);
            int gates = GateCount(cell);
            int hidden = options.HiddenSize;
            int proj = options.ProjSize;
            int numLayers = options.NumLayers;
            int numDirections = options.Bidirectional ? 2 : 1;
            int featOut = proj > 0 ? proj : hidden;

            EnsureHiddenStateShape(h0, numLayers * numDirections, batch, featOut, name: "h0");
            if (c0 is not null)
                EnsureHiddenStateShape(c0, numLayers * numDirections, batch, hidden, name: "c0");

            long expected = ExpectedTotalWeightLength(cell, inputSize, hidden, proj, numLayers, numDirections);
            if (weights.Length != expected)
                throw new ArgumentException(
                    $"Packed weights length {weights.Length} doesn't match expected {expected} for " +
                    $"cell={cell}, layers={numLayers}, bidirectional={options.Bidirectional}, " +
                    $"hidden={hidden}, proj={proj}, inputSize={inputSize}.", nameof(weights));

            var w = new T[weights.Length];
            weights.AsSpan().CopyTo(w);

            var layerInput = new Tensor<T>(new[] { seqLen, batch, inputSize });
            input.AsSpan().CopyTo(layerInput.AsWritableSpan());

            var hN = new Tensor<T>(new[] { numLayers * numDirections, batch, featOut });
            Tensor<T>? cN = cell == RnnCellType.Lstm
                ? new Tensor<T>(new[] { numLayers * numDirections, batch, hidden })
                : null;

            Random? rng = null;
            if (options.Dropout > 0 && numLayers > 1 && options.Training)
            {
                rng = options.DropoutSeed.HasValue
                    ? RandomHelper.CreateSeededRandom(options.DropoutSeed.Value)
                    : RandomHelper.CreateSecureRandom();
            }

            return new ForwardContext<T>
            {
                Layout = new Layout
                {
                    Cell = cell,
                    Gates = gates,
                    SeqLen = seqLen,
                    Batch = batch,
                    InputSize = inputSize,
                    Hidden = hidden,
                    Proj = proj,
                    NumLayers = numLayers,
                    NumDirections = numDirections,
                    FeatOut = featOut,
                    OutputFeature = featOut * numDirections,
                },
                LayerInput = layerInput,
                HN = hN,
                CN = cN,
                Weights = w,
                H0 = h0,
                C0 = c0,
                Options = options,
                DropoutRng = rng,
            };
        }
    }

    /// <summary>
    /// Slice of the packed weights for a given (layer, direction). Returns absolute offset
    /// into the flat weight array, plus the layer's input feature size and lengths of the
    /// W_ih / W_hh / b / W_hr (optional) blocks.
    /// </summary>
    private struct WeightSlice
    {
        public int Off;
        public int LayerInput;
        public int WIhLen;
        public int WHhLen;
        public int BLen;
        public int WHrLen;
    }

    private static WeightSlice LayerDirSlice(Layout layout, int targetLayer, int targetDir)
    {
        int off = 0;
        for (int l = 0; l < layout.NumLayers; l++)
        {
            int li = l == 0 ? layout.InputSize : layout.FeatOut * layout.NumDirections;
            long perDir = PerLayerDirWeightLen(layout.Cell, li, layout.Hidden, layout.Proj);
            for (int d = 0; d < layout.NumDirections; d++)
            {
                if (l == targetLayer && d == targetDir)
                {
                    return new WeightSlice
                    {
                        Off = off,
                        LayerInput = li,
                        WIhLen = layout.Gates * layout.Hidden * li,
                        WHhLen = layout.Gates * layout.Hidden * layout.FeatOut,
                        BLen = layout.Gates * layout.Hidden,
                        WHrLen = layout.Proj > 0 ? layout.Proj * layout.Hidden : 0,
                    };
                }
                off += (int)perDir;
            }
        }
        throw new ArgumentOutOfRangeException(nameof(targetLayer));
    }

    private static void ForwardStack<T>(ref ForwardContext<T> ctx, bool saveCache)
    {
        var layout = ctx.Layout;
        if (saveCache)
        {
            int LD = layout.NumLayers * layout.NumDirections;
            ctx.CacheLayerInputs = new T[layout.NumLayers][];
            ctx.CacheDropoutMasks = new bool[layout.NumLayers][];
            ctx.CacheH = new T[LD][];
            ctx.CacheC = layout.Cell == RnnCellType.Lstm ? new T[LD][] : null;
            ctx.CacheGates = new T[LD][];
            ctx.CacheRawCellOutput = layout.Proj > 0 ? new T[LD][] : null;
        }

        var hNSpan = ctx.HN.AsWritableSpan();
        Span<T> cNSpan = ctx.CN is null ? default : ctx.CN.AsWritableSpan();

        for (int l = 0; l < layout.NumLayers; l++)
        {
            if (saveCache)
            {
                int featIn = l == 0 ? layout.InputSize : layout.OutputFeature;
                var layerInBuf = new T[layout.SeqLen * layout.Batch * featIn];
                ctx.LayerInput!.AsSpan().CopyTo(layerInBuf);
                ctx.CacheLayerInputs![l] = layerInBuf;
            }

            var dirOutputs = new T[layout.NumDirections][];
            for (int d = 0; d < layout.NumDirections; d++)
            {
                var slice = LayerDirSlice(layout, l, d);
                int ld = l * layout.NumDirections + d;

                var h0 = SliceLayerDir(ctx.H0, ld, layout.Batch, layout.FeatOut);
                var c0 = ctx.C0 is not null ? SliceLayerDir(ctx.C0, ld, layout.Batch, layout.Hidden) : null;

                var (output, hLast, cLast, gates, rawCell, allH, allC) =
                    ForwardOneLayerOneDirection(layout, slice, ctx.LayerInput!, ctx.Weights,
                        h0, c0, reverseTime: d == 1, saveCache);

                dirOutputs[d] = output;
                hLast.AsSpan().CopyTo(hNSpan.Slice(ld * layout.Batch * layout.FeatOut, layout.Batch * layout.FeatOut));
                if (layout.Cell == RnnCellType.Lstm)
                    cLast!.AsSpan().CopyTo(cNSpan.Slice(ld * layout.Batch * layout.Hidden, layout.Batch * layout.Hidden));

                if (saveCache)
                {
                    ctx.CacheH![ld] = allH!;
                    if (layout.Cell == RnnCellType.Lstm) ctx.CacheC![ld] = allC!;
                    ctx.CacheGates![ld] = gates!;
                    if (layout.Proj > 0) ctx.CacheRawCellOutput![ld] = rawCell!;
                }
            }

            var layerOutput = new Tensor<T>(new[] { layout.SeqLen, layout.Batch, layout.OutputFeature });
            ConcatDirections(dirOutputs, layerOutput.AsWritableSpan(), layout.SeqLen, layout.Batch, layout.FeatOut, layout.NumDirections);

            if (l < layout.NumLayers - 1 && ctx.DropoutRng is not null)
            {
                var mask = ApplyDropout(layerOutput.AsWritableSpan(), ctx.Options.Dropout, ctx.DropoutRng);
                if (saveCache) ctx.CacheDropoutMasks![l] = mask;
            }
            else if (saveCache)
            {
                ctx.CacheDropoutMasks![l] = Array.Empty<bool>();
            }

            ctx.LayerInput = layerOutput;
        }
    }

    private static T[] SliceLayerDir<T>(Tensor<T> source, int ld, int batch, int feature)
    {
        var dest = new T[batch * feature];
        if (source.Rank == 3)
            source.AsSpan().Slice(ld * batch * feature, batch * feature).CopyTo(dest);
        else
            source.AsSpan().CopyTo(dest);
        return dest;
    }

    private static void ConcatDirections<T>(T[][] perDir, Span<T> dest, int seqLen, int batch, int featOut, int numDirections)
    {
        if (numDirections == 1)
        {
            perDir[0].AsSpan().CopyTo(dest);
            return;
        }
        for (int t = 0; t < seqLen; t++)
            for (int b = 0; b < batch; b++)
            {
                int destBase = (t * batch + b) * featOut * numDirections;
                int srcBase = (t * batch + b) * featOut;
                for (int d = 0; d < numDirections; d++)
                    perDir[d].AsSpan().Slice(srcBase, featOut)
                        .CopyTo(dest.Slice(destBase + d * featOut, featOut));
            }
    }

    private static void SplitDirections<T>(T[] flat, T[][] perDir, int seqLen, int batch, int featOut, int numDirections)
    {
        if (numDirections == 1) { flat.AsSpan().CopyTo(perDir[0]); return; }
        for (int t = 0; t < seqLen; t++)
            for (int b = 0; b < batch; b++)
            {
                int srcBase = (t * batch + b) * featOut * numDirections;
                int dstBase = (t * batch + b) * featOut;
                for (int d = 0; d < numDirections; d++)
                    flat.AsSpan(srcBase + d * featOut, featOut)
                        .CopyTo(perDir[d].AsSpan(dstBase, featOut));
            }
    }

    private static bool[] ApplyDropout<T>(Span<T> values, double dropoutP, Random rng)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double scale = 1.0 / (1.0 - dropoutP);
        var scaleT = ops.FromDouble(scale);
        var mask = new bool[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            bool keep = rng.NextDouble() >= dropoutP;
            mask[i] = keep;
            values[i] = keep ? ops.Multiply(values[i], scaleT) : ops.Zero;
        }
        return mask;
    }

    /// <summary>
    /// Forward pass over a single (layer, direction). All caches are iteration-order indexed:
    /// step = 0 ⇒ first iteration, step = T-1 ⇒ last iteration. For the reverse direction the
    /// time index t maps as t = T-1-step.
    /// </summary>
    private static (T[] Output, T[] HLast, T[]? CLast, T[]? Gates, T[]? RawCellOutputs, T[]? AllH, T[]? AllC)
        ForwardOneLayerOneDirection<T>(
            Layout layout, WeightSlice slice,
            Tensor<T> layerInput, T[] weights,
            T[] h0, T[]? c0, bool reverseTime, bool saveCache)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int T_ = layout.SeqLen, B = layout.Batch, H = layout.Hidden;
        int P = layout.Proj, F = layout.FeatOut;
        int G = layout.Gates;
        int inputFeat = slice.LayerInput;
        var cell = layout.Cell;

        var wIh = new ReadOnlySpan<T>(weights, slice.Off, slice.WIhLen);
        var wHh = new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen, slice.WHhLen);
        var bIh = new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen + slice.WHhLen, slice.BLen);
        var bHh = new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen + slice.WHhLen + slice.BLen, slice.BLen);
        var wHr = slice.WHrLen > 0
            ? new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen + slice.WHhLen + 2 * slice.BLen, slice.WHrLen)
            : default;

        var output = new T[T_ * B * F];
        var hPrev = new T[B * F];
        h0.AsSpan().CopyTo(hPrev);
        var cPrev = c0 is null ? null : new T[B * H];
        c0?.AsSpan().CopyTo(cPrev);

        T[]? allH = saveCache ? new T[(T_ + 1) * B * F] : null;
        T[]? allC = saveCache && cell == RnnCellType.Lstm ? new T[(T_ + 1) * B * H] : null;
        T[]? gatesCache = saveCache ? new T[T_ * B * G * H] : null;
        T[]? rawCellCache = saveCache && P > 0 ? new T[T_ * B * H] : null;

        if (saveCache)
        {
            hPrev.AsSpan().CopyTo(allH.AsSpan(0, B * F));
            if (cell == RnnCellType.Lstm) cPrev!.AsSpan().CopyTo(allC.AsSpan(0, B * H));
        }

        var inSpan = layerInput.AsSpan();

        for (int step = 0; step < T_; step++)
        {
            int t = reverseTime ? T_ - 1 - step : step;
            int inOff = t * B * inputFeat;
            int gateOff = step * B * G * H;
            int rawOff = step * B * H;

            switch (cell)
            {
                case RnnCellType.Lstm:
                    StepLstm(inSpan, inOff, hPrev, cPrev!, wIh, wHh, bIh, bHh, wHr,
                        B, H, P, inputFeat, output, t, gatesCache, gateOff, rawCellCache, rawOff);
                    break;
                case RnnCellType.Gru:
                    StepGru(inSpan, inOff, hPrev, wIh, wHh, bIh, bHh,
                        B, H, inputFeat, output, t, gatesCache, gateOff);
                    break;
                case RnnCellType.RnnTanh:
                    StepPlain(inSpan, inOff, hPrev, wIh, wHh, bIh, bHh,
                        B, H, inputFeat, output, t, useTanh: true, gatesCache, gateOff);
                    break;
                case RnnCellType.RnnRelu:
                    StepPlain(inSpan, inOff, hPrev, wIh, wHh, bIh, bHh,
                        B, H, inputFeat, output, t, useTanh: false, gatesCache, gateOff);
                    break;
            }

            if (saveCache)
            {
                hPrev.AsSpan().CopyTo(allH.AsSpan((step + 1) * B * F, B * F));
                if (cell == RnnCellType.Lstm) cPrev!.AsSpan().CopyTo(allC.AsSpan((step + 1) * B * H, B * H));
            }
        }

        var hLast = new T[B * F];
        hPrev.AsSpan().CopyTo(hLast);
        T[]? cLast = cell == RnnCellType.Lstm ? new T[B * H] : null;
        cPrev?.AsSpan().CopyTo(cLast);

        return (output, hLast, cLast, gatesCache, rawCellCache, allH, allC);
    }

    private static void StepLstm<T>(
        ReadOnlySpan<T> input, int inOff, T[] hPrev, T[] cPrev,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh, ReadOnlySpan<T> bIh, ReadOnlySpan<T> bHh, ReadOnlySpan<T> wHr,
        int B, int H, int P, int inputFeat,
        T[] output, int t,
        T[]? gatesCache, int gateOff,
        T[]? rawCellCache, int rawOff)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int G = 4;
        int F = P > 0 ? P : H;

        // Local buffers per (b, g, j) so compute is independent of the cache.
        var preAct = new double[B * G * H];
        var hFull = new double[B * H];

        for (int b = 0; b < B; b++)
        {
            for (int g = 0; g < G; g++)
            {
                for (int j = 0; j < H; j++)
                {
                    double acc = ops.ToDouble(bIh[g * H + j]) + ops.ToDouble(bHh[g * H + j]);
                    int ihRow = (g * H + j) * inputFeat;
                    for (int i = 0; i < inputFeat; i++)
                        acc += ops.ToDouble(wIh[ihRow + i]) * ops.ToDouble(input[inOff + b * inputFeat + i]);
                    int hhRow = (g * H + j) * F;
                    for (int i = 0; i < F; i++)
                        acc += ops.ToDouble(wHh[hhRow + i]) * ops.ToDouble(hPrev[b * F + i]);
                    preAct[(b * G + g) * H + j] = acc;
                }
            }
            for (int j = 0; j < H; j++)
            {
                double iG = Sigmoid(preAct[(b * G + 0) * H + j]);
                double fG = Sigmoid(preAct[(b * G + 1) * H + j]);
                double gG = Math.Tanh(preAct[(b * G + 2) * H + j]);
                double oG = Sigmoid(preAct[(b * G + 3) * H + j]);
                double c = fG * ops.ToDouble(cPrev[b * H + j]) + iG * gG;
                double h = oG * Math.Tanh(c);
                cPrev[b * H + j] = ops.FromDouble(c);
                hFull[b * H + j] = h;
            }
            if (P > 0)
            {
                for (int p = 0; p < P; p++)
                {
                    double acc = 0;
                    int row = p * H;
                    for (int i = 0; i < H; i++) acc += ops.ToDouble(wHr[row + i]) * hFull[b * H + i];
                    hPrev[b * P + p] = ops.FromDouble(acc);
                    output[(t * B + b) * P + p] = ops.FromDouble(acc);
                }
            }
            else
            {
                for (int j = 0; j < H; j++)
                {
                    hPrev[b * H + j] = ops.FromDouble(hFull[b * H + j]);
                    output[(t * B + b) * H + j] = ops.FromDouble(hFull[b * H + j]);
                }
            }
        }

        if (gatesCache is not null)
        {
            for (int i = 0; i < B * G * H; i++)
                gatesCache[gateOff + i] = ops.FromDouble(preAct[i]);
        }
        if (rawCellCache is not null)
        {
            for (int i = 0; i < B * H; i++)
                rawCellCache[rawOff + i] = ops.FromDouble(hFull[i]);
        }
    }

    private static void StepGru<T>(
        ReadOnlySpan<T> input, int inOff, T[] hPrev,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh, ReadOnlySpan<T> bIh, ReadOnlySpan<T> bHh,
        int B, int H, int inputFeat,
        T[] output, int t,
        T[]? gatesCache, int gateOff)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int G = 3;
        var hNew = new T[B * H];

        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                double accR_ih = ops.ToDouble(bIh[0 * H + j]);
                double accZ_ih = ops.ToDouble(bIh[1 * H + j]);
                double accN_ih = ops.ToDouble(bIh[2 * H + j]);
                int ihRowR = (0 * H + j) * inputFeat;
                int ihRowZ = (1 * H + j) * inputFeat;
                int ihRowN = (2 * H + j) * inputFeat;
                for (int i = 0; i < inputFeat; i++)
                {
                    double xv = ops.ToDouble(input[inOff + b * inputFeat + i]);
                    accR_ih += ops.ToDouble(wIh[ihRowR + i]) * xv;
                    accZ_ih += ops.ToDouble(wIh[ihRowZ + i]) * xv;
                    accN_ih += ops.ToDouble(wIh[ihRowN + i]) * xv;
                }
                double accR_hh = ops.ToDouble(bHh[0 * H + j]);
                double accZ_hh = ops.ToDouble(bHh[1 * H + j]);
                double accN_hh = ops.ToDouble(bHh[2 * H + j]);
                int hhRowR = (0 * H + j) * H;
                int hhRowZ = (1 * H + j) * H;
                int hhRowN = (2 * H + j) * H;
                for (int i = 0; i < H; i++)
                {
                    double hv = ops.ToDouble(hPrev[b * H + i]);
                    accR_hh += ops.ToDouble(wHh[hhRowR + i]) * hv;
                    accZ_hh += ops.ToDouble(wHh[hhRowZ + i]) * hv;
                    accN_hh += ops.ToDouble(wHh[hhRowN + i]) * hv;
                }
                double r = Sigmoid(accR_ih + accR_hh);
                double z = Sigmoid(accZ_ih + accZ_hh);
                double nPre = accN_ih + r * accN_hh;
                double n = Math.Tanh(nPre);
                double h = (1.0 - z) * n + z * ops.ToDouble(hPrev[b * H + j]);
                hNew[b * H + j] = ops.FromDouble(h);
                output[(t * B + b) * H + j] = hNew[b * H + j];

                if (gatesCache is not null)
                {
                    // Pack (r, z, n_post_tanh) per (b, j). Layout: (b * G + g) * H + j.
                    gatesCache[gateOff + (b * G + 0) * H + j] = ops.FromDouble(r);
                    gatesCache[gateOff + (b * G + 1) * H + j] = ops.FromDouble(z);
                    gatesCache[gateOff + (b * G + 2) * H + j] = ops.FromDouble(n);
                }
            }
        }
        hNew.AsSpan().CopyTo(hPrev);
    }

    private static void StepPlain<T>(
        ReadOnlySpan<T> input, int inOff, T[] hPrev,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh, ReadOnlySpan<T> bIh, ReadOnlySpan<T> bHh,
        int B, int H, int inputFeat,
        T[] output, int t, bool useTanh,
        T[]? gatesCache, int gateOff)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var hNew = new T[B * H];
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                T acc = ops.Add(bIh[j], bHh[j]);
                int ihRow = j * inputFeat;
                for (int i = 0; i < inputFeat; i++)
                    acc = ops.Add(acc, ops.Multiply(wIh[ihRow + i], input[inOff + b * inputFeat + i]));
                int hhRow = j * H;
                for (int i = 0; i < H; i++)
                    acc = ops.Add(acc, ops.Multiply(wHh[hhRow + i], hPrev[b * H + i]));
                if (gatesCache is not null) gatesCache[gateOff + b * H + j] = acc; // pre-activation
                double a = ops.ToDouble(acc);
                double activated = useTanh ? Math.Tanh(a) : Math.Max(0, a);
                hNew[b * H + j] = ops.FromDouble(activated);
                output[(t * B + b) * H + j] = hNew[b * H + j];
            }
        }
        hNew.AsSpan().CopyTo(hPrev);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // ===== BACKWARD =====

    private static (Tensor<T> GradInput, Tensor<T> GradH0, Tensor<T>? GradC0, Tensor<T> GradWeights)
        BackwardCore<T>(
            RnnCellType cell,
            Tensor<T> input, Tensor<T> h0, Tensor<T>? c0,
            Tensor<T> weights, RnnOptions options,
            Tensor<T> gradOutput, Tensor<T>? gradHN, Tensor<T>? gradCN)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        var ctx = ForwardContext<T>.Build(cell, input, h0, c0, weights, options);
        ForwardStack(ref ctx, saveCache: true);
        var layout = ctx.Layout;
        var ops = MathHelper.GetNumericOperations<T>();

        var gradWeightsArr = new T[ctx.Weights.Length];
        var gradH0 = new Tensor<T>(new[] { layout.NumLayers * layout.NumDirections, layout.Batch, layout.FeatOut });
        Tensor<T>? gradC0 = cell == RnnCellType.Lstm
            ? new Tensor<T>(new[] { layout.NumLayers * layout.NumDirections, layout.Batch, layout.Hidden })
            : null;
        var gradH0Span = gradH0.AsWritableSpan();
        Span<T> gradC0Span = gradC0 is null ? default : gradC0.AsWritableSpan();

        var dY = new T[layout.SeqLen * layout.Batch * layout.OutputFeature];
        gradOutput.AsSpan().CopyTo(dY);

        T[]? dHN = null, dCN = null;
        if (gradHN is not null)
        {
            dHN = new T[layout.NumLayers * layout.NumDirections * layout.Batch * layout.FeatOut];
            gradHN.AsSpan().CopyTo(dHN);
        }
        if (gradCN is not null && cell == RnnCellType.Lstm)
        {
            dCN = new T[layout.NumLayers * layout.NumDirections * layout.Batch * layout.Hidden];
            gradCN.AsSpan().CopyTo(dCN);
        }

        for (int l = layout.NumLayers - 1; l >= 0; l--)
        {
            int featIn = l == 0 ? layout.InputSize : layout.OutputFeature;

            if (l < layout.NumLayers - 1 && ctx.CacheDropoutMasks![l].Length == dY.Length)
            {
                var mask = ctx.CacheDropoutMasks![l];
                double scale = 1.0 / (1.0 - ctx.Options.Dropout);
                var scaleT = ops.FromDouble(scale);
                for (int i = 0; i < dY.Length; i++)
                    dY[i] = mask[i] ? ops.Multiply(dY[i], scaleT) : ops.Zero;
            }

            var dYPerDir = new T[layout.NumDirections][];
            for (int d = 0; d < layout.NumDirections; d++)
                dYPerDir[d] = new T[layout.SeqLen * layout.Batch * layout.FeatOut];
            SplitDirections(dY, dYPerDir, layout.SeqLen, layout.Batch, layout.FeatOut, layout.NumDirections);

            var dInput = new T[layout.SeqLen * layout.Batch * featIn];

            for (int d = 0; d < layout.NumDirections; d++)
            {
                int ld = l * layout.NumDirections + d;
                var slice = LayerDirSlice(layout, l, d);
                bool reverseTime = d == 1;

                T[]? dHNSlice = dHN is null ? null : SliceFlat(dHN, ld, layout.Batch * layout.FeatOut);
                T[]? dCNSlice = dCN is null ? null : SliceFlat(dCN, ld, layout.Batch * layout.Hidden);

                BackwardOneLayerOneDirection(
                    layout, slice,
                    ctx.CacheLayerInputs![l],
                    ctx.CacheH![ld],
                    layout.Cell == RnnCellType.Lstm ? ctx.CacheC![ld] : null,
                    ctx.CacheGates![ld],
                    layout.Proj > 0 ? ctx.CacheRawCellOutput![ld] : null,
                    ctx.Weights, reverseTime,
                    dYPerDir[d], dHNSlice, dCNSlice,
                    dInput, gradWeightsArr,
                    out var dH0Out, out var dC0Out);

                dH0Out.AsSpan().CopyTo(gradH0Span.Slice(ld * layout.Batch * layout.FeatOut, layout.Batch * layout.FeatOut));
                if (cell == RnnCellType.Lstm)
                    dC0Out!.AsSpan().CopyTo(gradC0Span.Slice(ld * layout.Batch * layout.Hidden, layout.Batch * layout.Hidden));
            }

            dY = dInput;
        }

        var gradInput = new Tensor<T>(new[] { layout.SeqLen, layout.Batch, layout.InputSize });
        dY.AsSpan().CopyTo(gradInput.AsWritableSpan());
        var gradWeights = new Tensor<T>(new[] { ctx.Weights.Length });
        gradWeightsArr.AsSpan().CopyTo(gradWeights.AsWritableSpan());

        return (gradInput, gradH0, gradC0, gradWeights);
    }

    private static T[] SliceFlat<T>(T[] source, int ld, int sliceLen)
    {
        var dest = new T[sliceLen];
        source.AsSpan(ld * sliceLen, sliceLen).CopyTo(dest);
        return dest;
    }

    private static void BackwardOneLayerOneDirection<T>(
        Layout layout, WeightSlice slice,
        T[] layerInput,
        T[] allH, T[]? allC, T[] gates, T[]? rawCellOutputs,
        T[] weights, bool reverseTime,
        T[] dY, T[]? dHNSlice, T[]? dCNSlice,
        T[] dInputAccum, T[] gradWeights,
        out T[] dH0, out T[]? dC0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int T_ = layout.SeqLen, B = layout.Batch, H = layout.Hidden;
        int P = layout.Proj, F = layout.FeatOut;
        int G = layout.Gates;
        int inputFeat = slice.LayerInput;
        var cell = layout.Cell;

        var wIh = new ReadOnlySpan<T>(weights, slice.Off, slice.WIhLen);
        var wHh = new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen, slice.WHhLen);
        var bHhSpan = new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen + slice.WHhLen + slice.BLen, slice.BLen);
        var wHr = slice.WHrLen > 0 ? new ReadOnlySpan<T>(weights, slice.Off + slice.WIhLen + slice.WHhLen + 2 * slice.BLen, slice.WHrLen) : default;

        var dWIh = new Span<T>(gradWeights, slice.Off, slice.WIhLen);
        var dWHh = new Span<T>(gradWeights, slice.Off + slice.WIhLen, slice.WHhLen);
        var dBIh = new Span<T>(gradWeights, slice.Off + slice.WIhLen + slice.WHhLen, slice.BLen);
        var dBHh = new Span<T>(gradWeights, slice.Off + slice.WIhLen + slice.WHhLen + slice.BLen, slice.BLen);
        var dWHr = slice.WHrLen > 0 ? new Span<T>(gradWeights, slice.Off + slice.WIhLen + slice.WHhLen + 2 * slice.BLen, slice.WHrLen) : default;

        var dH = new T[B * F];
        if (dHNSlice is not null) dHNSlice.AsSpan().CopyTo(dH);
        var dC = cell == RnnCellType.Lstm ? new T[B * H] : null;
        if (dCNSlice is not null) dCNSlice.AsSpan().CopyTo(dC);

        for (int step = T_ - 1; step >= 0; step--)
        {
            int t = reverseTime ? T_ - 1 - step : step;
            // Add this step's external dY into the running dH.
            for (int i = 0; i < dH.Length; i++)
                dH[i] = ops.Add(dH[i], dY[t * B * F + i]);

            int gateOff = step * B * G * H;
            int hPrevOff = step * B * F;       // allH[step] = state before this iteration
            int rawOff = step * B * H;
            int cPrevOff = step * B * H;
            int cCurOff = (step + 1) * B * H;

            switch (cell)
            {
                case RnnCellType.Lstm:
                    StepLstmBackward(
                        layerInput, t, inputFeat,
                        allH, hPrevOff, allC!, cPrevOff, cCurOff,
                        gates, gateOff, rawCellOutputs, rawOff,
                        wIh, wHh, wHr, B, H, P, F,
                        dH, dC!,
                        dWIh, dWHh, dBIh, dBHh, dWHr,
                        dInputAccum);
                    break;
                case RnnCellType.Gru:
                    StepGruBackward(
                        layerInput, t, inputFeat,
                        allH, hPrevOff, gates, gateOff,
                        wIh, wHh, bHhSpan, B, H, F,
                        dH,
                        dWIh, dWHh, dBIh, dBHh,
                        dInputAccum);
                    break;
                case RnnCellType.RnnTanh:
                case RnnCellType.RnnRelu:
                    StepPlainBackward(
                        layerInput, t, inputFeat,
                        allH, hPrevOff, gates, gateOff,
                        wIh, wHh, B, H, F,
                        useTanh: cell == RnnCellType.RnnTanh,
                        dH,
                        dWIh, dWHh, dBIh, dBHh,
                        dInputAccum);
                    break;
            }
        }

        dH0 = dH;
        dC0 = dC;
    }

    private static void StepLstmBackward<T>(
        T[] layerInput, int t, int inputFeat,
        T[] allH, int hPrevOff, T[] allC, int cPrevOff, int cCurOff,
        T[] gates, int gateOff, T[]? rawCellOutputs, int rawOff,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh, ReadOnlySpan<T> wHr,
        int B, int H, int P, int F,
        T[] dH, T[] dC,
        Span<T> dWIh, Span<T> dWHh, Span<T> dBIh, Span<T> dBHh, Span<T> dWHr,
        T[] dInputAccum)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int G = 4;

        // Step 1: undo projection if active. dH (size F=P) → dHFull (size H), and accumulate dW_hr.
        var dHFull = new double[B * H];
        if (P > 0)
        {
            for (int b = 0; b < B; b++)
            {
                for (int p = 0; p < P; p++)
                {
                    double dhp = ops.ToDouble(dH[b * P + p]);
                    int row = p * H;
                    for (int i = 0; i < H; i++)
                    {
                        double inc = dhp * ops.ToDouble(rawCellOutputs![rawOff + b * H + i]);
                        dWHr[row + i] = ops.Add(dWHr[row + i], ops.FromDouble(inc));
                        dHFull[b * H + i] += ops.ToDouble(wHr[row + i]) * dhp;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < dH.Length; i++) dHFull[i] = ops.ToDouble(dH[i]);
        }

        // Step 2: backprop through cell to get dPre (gradient w.r.t. each gate's pre-activation).
        var dPre = new double[B * G * H];
        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                double iRaw = ops.ToDouble(gates[gateOff + (b * G + 0) * H + j]);
                double fRaw = ops.ToDouble(gates[gateOff + (b * G + 1) * H + j]);
                double gRaw = ops.ToDouble(gates[gateOff + (b * G + 2) * H + j]);
                double oRaw = ops.ToDouble(gates[gateOff + (b * G + 3) * H + j]);
                double iG = Sigmoid(iRaw);
                double fG = Sigmoid(fRaw);
                double gG = Math.Tanh(gRaw);
                double oG = Sigmoid(oRaw);

                double cCur = ops.ToDouble(allC[cCurOff + b * H + j]);
                double cPrev = ops.ToDouble(allC[cPrevOff + b * H + j]);
                double tanhC = Math.Tanh(cCur);

                double dh = dHFull[b * H + j];
                double dc = ops.ToDouble(dC[b * H + j]) + dh * oG * (1 - tanhC * tanhC);

                double dO = dh * tanhC * oG * (1 - oG);
                double dI = dc * gG * iG * (1 - iG);
                double dF = dc * cPrev * fG * (1 - fG);
                double dG = dc * iG * (1 - gG * gG);
                double dCPrev = dc * fG;

                dPre[(b * G + 0) * H + j] = dI;
                dPre[(b * G + 1) * H + j] = dF;
                dPre[(b * G + 2) * H + j] = dG;
                dPre[(b * G + 3) * H + j] = dO;
                dC[b * H + j] = ops.FromDouble(dCPrev);
            }
        }

        // Step 3: distribute dPre into dW_ih, dW_hh, db, dx, dHPrev.
        var dHPrev = new double[B * F];
        for (int b = 0; b < B; b++)
        {
            for (int g = 0; g < G; g++)
            {
                for (int j = 0; j < H; j++)
                {
                    double dp = dPre[(b * G + g) * H + j];
                    int gj = g * H + j;
                    dBIh[gj] = ops.Add(dBIh[gj], ops.FromDouble(dp));
                    dBHh[gj] = ops.Add(dBHh[gj], ops.FromDouble(dp));
                    int ihRow = gj * inputFeat;
                    int hhRow = gj * F;
                    for (int i = 0; i < inputFeat; i++)
                    {
                        double xv = ops.ToDouble(layerInput[(t * B + b) * inputFeat + i]);
                        dWIh[ihRow + i] = ops.Add(dWIh[ihRow + i], ops.FromDouble(dp * xv));
                        dInputAccum[(t * B + b) * inputFeat + i] = ops.Add(
                            dInputAccum[(t * B + b) * inputFeat + i],
                            ops.FromDouble(dp * ops.ToDouble(wIh[ihRow + i])));
                    }
                    for (int k = 0; k < F; k++)
                    {
                        double hv = ops.ToDouble(allH[hPrevOff + b * F + k]);
                        dWHh[hhRow + k] = ops.Add(dWHh[hhRow + k], ops.FromDouble(dp * hv));
                        dHPrev[b * F + k] += dp * ops.ToDouble(wHh[hhRow + k]);
                    }
                }
            }
        }
        for (int i = 0; i < dH.Length; i++) dH[i] = ops.FromDouble(dHPrev[i]);
    }

    private static void StepGruBackward<T>(
        T[] layerInput, int t, int inputFeat,
        T[] allH, int hPrevOff, T[] gates, int gateOff,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh, ReadOnlySpan<T> bHh,
        int B, int H, int F,
        T[] dH,
        Span<T> dWIh, Span<T> dWHh, Span<T> dBIh, Span<T> dBHh,
        T[] dInputAccum)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        const int G = 3;
        var dHPrev = new double[B * H];

        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                double r = ops.ToDouble(gates[gateOff + (b * G + 0) * H + j]);
                double z = ops.ToDouble(gates[gateOff + (b * G + 1) * H + j]);
                double n = ops.ToDouble(gates[gateOff + (b * G + 2) * H + j]);
                double hPrevJ = ops.ToDouble(allH[hPrevOff + b * H + j]);
                double dh = ops.ToDouble(dH[b * H + j]);

                // h_t = (1 - z) n + z * hPrev_j
                double dn = dh * (1.0 - z);
                double dz = dh * (hPrevJ - n);
                dHPrev[b * H + j] += dh * z;

                // n = tanh(accN_ih + r * accN_hh)
                double dnPre = dn * (1.0 - n * n);

                // Recompute accN_hh from current hPrev row (we have allH and bHh / wHh available).
                double accN_hh = ops.ToDouble(bHh[2 * H + j]);
                int hhRowN = (2 * H + j) * H;
                for (int i = 0; i < H; i++)
                    accN_hh += ops.ToDouble(wHh[hhRowN + i]) * ops.ToDouble(allH[hPrevOff + b * H + i]);

                double dr_from_n = dnPre * accN_hh;
                double dAccN_hh = dnPre * r;

                // r and z are sigmoids.
                double dr = dr_from_n * r * (1 - r);
                double dz_pre = dz * z * (1 - z);

                double dpreIh_r = dr;
                double dpreIh_z = dz_pre;
                double dpreIh_n = dnPre;
                double dpreHh_r = dr;
                double dpreHh_z = dz_pre;
                double dpreHh_n = dAccN_hh;

                dBIh[0 * H + j] = ops.Add(dBIh[0 * H + j], ops.FromDouble(dpreIh_r));
                dBIh[1 * H + j] = ops.Add(dBIh[1 * H + j], ops.FromDouble(dpreIh_z));
                dBIh[2 * H + j] = ops.Add(dBIh[2 * H + j], ops.FromDouble(dpreIh_n));
                dBHh[0 * H + j] = ops.Add(dBHh[0 * H + j], ops.FromDouble(dpreHh_r));
                dBHh[1 * H + j] = ops.Add(dBHh[1 * H + j], ops.FromDouble(dpreHh_z));
                dBHh[2 * H + j] = ops.Add(dBHh[2 * H + j], ops.FromDouble(dpreHh_n));

                int ihRowR = (0 * H + j) * inputFeat;
                int ihRowZ = (1 * H + j) * inputFeat;
                int ihRowN = (2 * H + j) * inputFeat;
                int hhRowR = (0 * H + j) * H;
                int hhRowZ = (1 * H + j) * H;

                for (int i = 0; i < inputFeat; i++)
                {
                    double xv = ops.ToDouble(layerInput[(t * B + b) * inputFeat + i]);
                    dWIh[ihRowR + i] = ops.Add(dWIh[ihRowR + i], ops.FromDouble(dpreIh_r * xv));
                    dWIh[ihRowZ + i] = ops.Add(dWIh[ihRowZ + i], ops.FromDouble(dpreIh_z * xv));
                    dWIh[ihRowN + i] = ops.Add(dWIh[ihRowN + i], ops.FromDouble(dpreIh_n * xv));
                    double dxAcc = dpreIh_r * ops.ToDouble(wIh[ihRowR + i])
                                 + dpreIh_z * ops.ToDouble(wIh[ihRowZ + i])
                                 + dpreIh_n * ops.ToDouble(wIh[ihRowN + i]);
                    dInputAccum[(t * B + b) * inputFeat + i] = ops.Add(
                        dInputAccum[(t * B + b) * inputFeat + i], ops.FromDouble(dxAcc));
                }
                for (int k = 0; k < H; k++)
                {
                    double hv = ops.ToDouble(allH[hPrevOff + b * H + k]);
                    dWHh[hhRowR + k] = ops.Add(dWHh[hhRowR + k], ops.FromDouble(dpreHh_r * hv));
                    dWHh[hhRowZ + k] = ops.Add(dWHh[hhRowZ + k], ops.FromDouble(dpreHh_z * hv));
                    dWHh[hhRowN + k] = ops.Add(dWHh[hhRowN + k], ops.FromDouble(dpreHh_n * hv));
                    dHPrev[b * H + k] +=
                        dpreHh_r * ops.ToDouble(wHh[hhRowR + k])
                      + dpreHh_z * ops.ToDouble(wHh[hhRowZ + k])
                      + dpreHh_n * ops.ToDouble(wHh[hhRowN + k]);
                }
            }
        }

        for (int i = 0; i < dH.Length; i++) dH[i] = ops.FromDouble(dHPrev[i]);
    }

    private static void StepPlainBackward<T>(
        T[] layerInput, int t, int inputFeat,
        T[] allH, int hPrevOff, T[] gates, int gateOff,
        ReadOnlySpan<T> wIh, ReadOnlySpan<T> wHh,
        int B, int H, int F, bool useTanh,
        T[] dH,
        Span<T> dWIh, Span<T> dWHh, Span<T> dBIh, Span<T> dBHh,
        T[] dInputAccum)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var dHPrev = new double[B * H];

        for (int b = 0; b < B; b++)
        {
            for (int j = 0; j < H; j++)
            {
                double pre = ops.ToDouble(gates[gateOff + b * H + j]);
                double activated = useTanh ? Math.Tanh(pre) : Math.Max(0, pre);
                double dActivated = ops.ToDouble(dH[b * H + j]);
                double dPre = useTanh
                    ? dActivated * (1.0 - activated * activated)
                    : (pre > 0 ? dActivated : 0.0);

                dBIh[j] = ops.Add(dBIh[j], ops.FromDouble(dPre));
                dBHh[j] = ops.Add(dBHh[j], ops.FromDouble(dPre));
                int ihRow = j * inputFeat;
                int hhRow = j * H;
                for (int i = 0; i < inputFeat; i++)
                {
                    double xv = ops.ToDouble(layerInput[(t * B + b) * inputFeat + i]);
                    dWIh[ihRow + i] = ops.Add(dWIh[ihRow + i], ops.FromDouble(dPre * xv));
                    dInputAccum[(t * B + b) * inputFeat + i] = ops.Add(
                        dInputAccum[(t * B + b) * inputFeat + i],
                        ops.FromDouble(dPre * ops.ToDouble(wIh[ihRow + i])));
                }
                for (int k = 0; k < H; k++)
                {
                    double hv = ops.ToDouble(allH[hPrevOff + b * H + k]);
                    dWHh[hhRow + k] = ops.Add(dWHh[hhRow + k], ops.FromDouble(dPre * hv));
                    dHPrev[b * H + k] += dPre * ops.ToDouble(wHh[hhRow + k]);
                }
            }
        }
        for (int i = 0; i < dH.Length; i++) dH[i] = ops.FromDouble(dHPrev[i]);
    }
}
