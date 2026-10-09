using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Recurrent cells and multi-layer sequences matching PyTorch's functional RNN API. Weights use PyTorch's layouts
/// (<c>w_ih [gates·hidden, input]</c>, <c>w_hh [gates·hidden, hidden]</c>) and gate orders (LSTM i,f,g,o; GRU r,z,n).
/// A cell input may be unbatched <c>[input]</c> or batched <c>[batch, input]</c>.
/// </summary>
public partial interface IEngine
{
    /// <summary>
    /// One Elman RNN step h' = act(x·w_ihᵀ + b_ih + h·w_hhᵀ + b_hh), act = tanh or ReLU per
    /// <paramref name="cell"/> (<c>torch.rnn_tanh_cell</c> / <c>torch.rnn_relu_cell</c>).
    /// </summary>
    Tensor<T> TensorRnnCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh = null, Tensor<T>? bHh = null, RnnCellType cell = RnnCellType.RnnTanh);

    /// <summary>One LSTM step returning (h', c') (<c>torch.lstm_cell</c>).</summary>
    (Tensor<T> Hidden, Tensor<T> Cell) TensorLstmCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> cell,
        Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh = null, Tensor<T>? bHh = null);

    /// <summary>One GRU step (<c>torch.gru_cell</c>).</summary>
    Tensor<T> TensorGruCell<T>(Tensor<T> input, Tensor<T> hidden, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh = null, Tensor<T>? bHh = null);

    /// <summary>
    /// A multi-layer, optionally bidirectional Elman RNN or GRU over a sequence (<c>torch.rnn_tanh</c>,
    /// <c>torch.rnn_relu</c>, <c>torch.gru</c>). <paramref name="input"/> is <c>[time, batch, input]</c>
    /// (<c>[batch, time, input]</c> when <paramref name="batchFirst"/>); <paramref name="h0"/> is
    /// <c>[layers·directions, batch, hidden]</c> or null for zeros; <paramref name="weights"/> holds, per layer and
    /// direction, w_ih, w_hh and (when <paramref name="hasBiases"/>) b_ih, b_hh. Dropout applies to every layer's
    /// output but the last while <paramref name="training"/>. Returns the last layer's outputs and the final hidden
    /// states.
    /// </summary>
    (Tensor<T> Output, Tensor<T> Hidden) TensorRecurrent<T>(RnnCellType cell, Tensor<T> input, Tensor<T>? h0,
        IReadOnlyList<Tensor<T>> weights, bool hasBiases, int numLayers, double dropout = 0, bool training = false,
        bool bidirectional = false, bool batchFirst = false);
}
