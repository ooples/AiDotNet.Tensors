using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // x·wᵀ + b with w in PyTorch's [out, in] layout; recorded through transpose and the fused linear.
    private Tensor<T> TorchLinear<T>(Tensor<T> x, Tensor<T> w, Tensor<T>? b)
        => FusedLinear(x, TensorTranspose(w), b, FusedActivationType.None);

    // Runs a cell on a batched view of an unbatched input and restores the caller's rank.
    private static (Tensor<T> X, Tensor<T> H, bool Unbatched) BatchCell<T>(Tensor<T> input, Tensor<T> hidden, IEngine engine)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (hidden == null) throw new ArgumentNullException(nameof(hidden));
        if (input.Rank == 1 && hidden.Rank == 1)
            return (engine.Reshape(input, new[] { 1, input.Length }), engine.Reshape(hidden, new[] { 1, hidden.Length }), true);
        if (input.Rank != 2 || hidden.Rank != 2 || input._shape[0] != hidden._shape[0])
            throw new ArgumentException("cell input and hidden must be [batch, features] with matching batch, or both unbatched.");
        return (input, hidden, false);
    }

    private Tensor<T> Unbatch<T>(Tensor<T> t, bool unbatched) => unbatched ? Reshape(t, new[] { t._shape[1] }) : t;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRnnCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh = null, Tensor<T>? bHh = null, RnnCellType cell = RnnCellType.RnnTanh)
    {
        if (cell != RnnCellType.RnnTanh && cell != RnnCellType.RnnRelu)
            throw new ArgumentException("TensorRnnCell takes RnnTanh or RnnRelu; use TensorLstmCell or TensorGruCell.", nameof(cell));
        var (x, h, unbatched) = BatchCell(input, hidden, this);
        var pre = TensorAdd(TorchLinear(x, wIh, bIh), TorchLinear(h, wHh, bHh));
        return Unbatch(cell == RnnCellType.RnnTanh ? TensorTanh(pre) : TensorReLU(pre), unbatched);
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Hidden, Tensor<T> Cell) TensorLstmCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> cell,
        Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh = null, Tensor<T>? bHh = null)
    {
        if (cell == null) throw new ArgumentNullException(nameof(cell));
        var (x, h, unbatched) = BatchCell(input, hidden, this);
        var c = unbatched ? Reshape(cell, new[] { 1, cell.Length }) : cell;
        int n = h._shape[1];
        var gates = TensorAdd(TorchLinear(x, wIh, bIh), TorchLinear(h, wHh, bHh));
        var i = TensorSigmoid(TensorNarrow(gates, 1, 0, n));
        var f = TensorSigmoid(TensorNarrow(gates, 1, n, n));
        var g = TensorTanh(TensorNarrow(gates, 1, 2 * n, n));
        var o = TensorSigmoid(TensorNarrow(gates, 1, 3 * n, n));
        var cNext = TensorAdd(TensorMultiply(f, c), TensorMultiply(i, g));
        var hNext = TensorMultiply(o, TensorTanh(cNext));
        return (Unbatch(hNext, unbatched), Unbatch(cNext, unbatched));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGruCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh = null, Tensor<T>? bHh = null)
    {
        var (x, h, unbatched) = BatchCell(input, hidden, this);
        int n = h._shape[1];
        var gi = TorchLinear(x, wIh, bIh);
        var gh = TorchLinear(h, wHh, bHh);
        var r = TensorSigmoid(TensorAdd(TensorNarrow(gi, 1, 0, n), TensorNarrow(gh, 1, 0, n)));
        var z = TensorSigmoid(TensorAdd(TensorNarrow(gi, 1, n, n), TensorNarrow(gh, 1, n, n)));
        // The reset gate scales the hidden projection after its bias: n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)).
        var candidate = TensorTanh(TensorAdd(TensorNarrow(gi, 1, 2 * n, n), TensorMultiply(r, TensorNarrow(gh, 1, 2 * n, n))));
        // h' = (1 - z) ⊙ n + z ⊙ h = n + z ⊙ (h - n)
        return Unbatch(TensorAdd(candidate, TensorMultiply(z, TensorSubtract(h, candidate))), unbatched);
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Output, Tensor<T> Hidden) TensorRecurrent<T>(RnnCellType cell, Tensor<T> input, Tensor<T>? h0,
        IReadOnlyList<Tensor<T>> weights, bool hasBiases, int numLayers, double dropout = 0, bool training = false,
        bool bidirectional = false, bool batchFirst = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (cell == RnnCellType.Lstm) throw new ArgumentException("LSTM sequences carry a cell state; use LstmSequenceForward.", nameof(cell));
        if (numLayers < 1) throw new ArgumentOutOfRangeException(nameof(numLayers), "numLayers must be at least 1.");
        if (dropout < 0 || dropout > 1) throw new ArgumentOutOfRangeException(nameof(dropout), "dropout must be in [0, 1].");
        int directions = bidirectional ? 2 : 1, perCell = hasBiases ? 4 : 2;
        if (weights.Count != numLayers * directions * perCell)
            throw new ArgumentException($"expected {numLayers * directions * perCell} weight tensors, got {weights.Count}.", nameof(weights));
        bool unbatched = input.Rank == 2;
        if (!unbatched && input.Rank != 3) throw new ArgumentException("input must be [time, batch, features] or unbatched [time, features].", nameof(input));
        // Work time-major and batched: [time, batch, features].
        var x = unbatched ? Reshape(input, new[] { input._shape[0], 1, input._shape[1] })
            : batchFirst ? TensorPermute(input, new[] { 1, 0, 2 }) : input;
        int steps = x._shape[0], batch = x._shape[1], hiddenSize = weights[1]._shape[1];
        if (h0 != null && unbatched) h0 = Reshape(h0, new[] { h0._shape[0], 1, h0._shape[1] });
        if (h0 != null && (h0.Rank != 3 || h0._shape[0] != numLayers * directions || h0._shape[1] != batch || h0._shape[2] != hiddenSize))
            throw new ArgumentException($"h0 must be [{numLayers * directions}, {batch}, {hiddenSize}].", nameof(h0));
        var finals = new Tensor<T>[numLayers * directions];
        for (int layer = 0; layer < numLayers; layer++)
        {
            var directionOutputs = new Tensor<T>[directions];
            for (int dir = 0; dir < directions; dir++)
            {
                int slot = layer * directions + dir, w = slot * perCell;
                Tensor<T> wIh = weights[w], wHh = weights[w + 1];
                Tensor<T>? bIh = hasBiases ? weights[w + 2] : null, bHh = hasBiases ? weights[w + 3] : null;
                var h = h0 != null ? TensorSelect(h0, 0, slot) : new Tensor<T>(new[] { batch, hiddenSize });
                var outputs = new Tensor<T>[steps];
                for (int k = 0; k < steps; k++)
                {
                    int t = dir == 0 ? k : steps - 1 - k;
                    var xt = TensorSelect(x, 0, t);
                    h = cell == RnnCellType.Gru ? TensorGruCell(xt, h, wIh, wHh, bIh, bHh) : TensorRnnCell(xt, h, wIh, wHh, bIh, bHh, cell);
                    outputs[t] = h;
                }
                finals[slot] = h;
                directionOutputs[dir] = TensorStack(outputs, 0);
            }
            x = directions == 1 ? directionOutputs[0] : TensorConcatenate(directionOutputs, 2);
            if (training && dropout > 0 && layer < numLayers - 1) x = Dropout(x, dropout, true, out _);
        }
        var hidden = TensorStack(finals, 0);
        if (unbatched) return (Reshape(x, new[] { steps, x._shape[2] }), Reshape(hidden, new[] { hidden._shape[0], hiddenSize }));
        return (batchFirst ? TensorPermute(x, new[] { 1, 0, 2 }) : x, hidden);
    }
}
