using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // The tape and compiled IR each own one output per node. A packet keeps sequence, h_n and c_n
    // on that edge; slices accumulate their independent adjoints before one fused BPTT traversal.
    private static int[] LstmTrainingOutputShape(int batch, int steps, int hidden, bool sequence, bool states)
        => states ? new[] { checked(((sequence ? batch * steps : batch) + 2 * batch) * hidden) }
            : sequence ? new[] { batch, steps, hidden } : new[] { batch, hidden };

    private static void WriteLstmFinalStates<T>(Span<T> packet, T[] hiddens, T[] cells,
        int batch, int steps, int hidden, bool sequence)
    {
        int offset = (sequence ? batch * steps : batch) * hidden;
        for (int b = 0; b < batch; b++)
        {
            int savedOffset = (b * (steps + 1) + steps) * hidden;
            hiddens.AsSpan(savedOffset, hidden).CopyTo(packet.Slice(offset + b * hidden, hidden));
            cells.AsSpan(savedOffset, hidden).CopyTo(packet.Slice(offset + (batch + b) * hidden, hidden));
        }
    }

    protected static Tensor<T> UnpackLstmTrainingOutput<T>(IEngine engine, Tensor<T> packet,
        int batch, int steps, int hidden, bool sequence, bool states,
        out Tensor<T> finalHidden, out Tensor<T> finalCell)
    {
        finalHidden = Tensor<T>.Empty();
        finalCell = Tensor<T>.Empty();
        if (!states) return packet;
        int stateLength = batch * hidden;
        int outputLength = sequence ? batch * steps * hidden : stateLength;
        finalHidden = engine.Reshape(engine.TensorNarrow(packet, 0, outputLength, stateLength), new[] { batch, hidden });
        finalCell = engine.Reshape(engine.TensorNarrow(packet, 0, outputLength + stateLength, stateLength), new[] { batch, hidden });
        return engine.Reshape(engine.TensorNarrow(packet, 0, 0, outputLength),
            sequence ? new[] { batch, steps, hidden } : new[] { batch, hidden });
    }

    private static Tensor<double> RecordLstmSequenceTrainDouble(LazyTensorScope scope,
        Tensor<double> input, Tensor<double>? h0, Tensor<double>? c0,
        Tensor<double> wIh, Tensor<double> wHh, Tensor<double>? bIh, Tensor<double>? bHh,
        int batch, int steps, int inFeatures, int hidden, bool sequence, bool states)
    {
        var inputs = new List<Tensor<double>> { input, wIh, wHh };
        int AddOptional(Tensor<double>? value)
        {
            if (value is null) return -1;
            int index = inputs.Count; inputs.Add(value); return index;
        }
        var meta = new[] { batch, steps, inFeatures, hidden, sequence ? 1 : 0,
            AddOptional(bIh), AddOptional(bHh), AddOptional(h0), AddOptional(c0), states ? 1 : 0 };
        var gates = new double[batch * steps * 4 * hidden];
        var cells = new double[batch * (steps + 1) * hidden];
        var hiddens = new double[cells.Length];
        var saved = new object[] { gates, cells, hiddens, meta };
        var shape = LstmTrainingOutputShape(batch, steps, hidden, sequence, states);
        return scope.RecordVariadic(LazyNodeType.Custom, "LstmSequenceTrain", inputs.ToArray(), shape,
            (engine, output) =>
            {
                // The host recorder is never selected for a device engine. Keep the plan's output
                // alive while borrowing its storage, and rewrite saved state on every replay.
                using var keep = new KeepAliveScope(output);
                var span = output.AsWritableSpanUnmarked();
                LstmTrainForwardCoreDouble(input.GetFlattenedData(), wIh.GetFlattenedData(), wHh.GetFlattenedData(),
                    bIh?.GetFlattenedData(), bHh?.GetFlattenedData(), h0?.GetFlattenedData(), c0?.GetFlattenedData(),
                    batch, steps, inFeatures, hidden, sequence, gates, cells, hiddens, span);
                if (states) WriteLstmFinalStates(span, hiddens, cells, batch, steps, hidden, sequence);
                output.IncrementVersion();
            }, LstmSequenceBackwardDouble, saved);
    }
}
