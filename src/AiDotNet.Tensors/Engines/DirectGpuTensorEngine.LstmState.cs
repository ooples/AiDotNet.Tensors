using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The shared device recurrence has independent dh_T/dc_T carries. All three public outputs
    // share one packet edge, so one backward receives their accumulated adjoints.
    private Tensor<T>? TryStatefulLstmTraining<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T>? c0,
        Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh, bool sequence,
        out Tensor<T>? finalHidden, out Tensor<T>? finalCell)
    {
        finalHidden = finalCell = null;
        if (typeof(T) != typeof(float) || PlacementBackend is not IFusedLstmSequenceTraining
            || !(DifferentiableOps.IsRecording<T>() || (GraphMode.IsActive && !GraphMode.IsInferenceTrace))
            || input.Rank != 3 || wIh.Rank != 2 || wHh.Rank != 2) return null;
        int b = input.Shape[0], t = input.Shape[1], n = input.Shape[2], g = wIh.Shape[0], h = g / 4;
        if (b <= 0 || t <= 0 || n <= 0 || h <= 0 || g % 4 != 0 || wIh.Shape[1] != n
            || wHh.Shape[0] != g || wHh.Shape[1] != h
            || (bIh is not null && (bIh.Rank != 1 || bIh.Length != g)) || (bHh is not null && (bHh.Rank != 1 || bHh.Length != g))
            || (h0 is not null && (h0.Rank != 2 || h0.Shape[0] != b || h0.Shape[1] != h))
            || (c0 is not null && (c0.Rank != 2 || c0.Shape[0] != b || c0.Shape[1] != h))) return null;
        var inputs = new List<Tensor<float>> { (Tensor<float>)(object)input,
            (Tensor<float>)(object)wIh, (Tensor<float>)(object)wHh };
        int Add(Tensor<T>? x)
        {
            if (x is null) return -1;
            int i = inputs.Count; inputs.Add((Tensor<float>)(object)x); return i;
        }
        var meta = new[] { b, t, n, h, sequence ? 1 : 0, Add(bIh), Add(bHh), Add(h0), Add(c0), 1 };
        var saved = new object[] { null!, null!, null!, meta };
        var parameters = inputs.ToArray();
        Tensor<float> packet;
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope is null || !scope.BindEngine(this)) return null;
            packet = scope.RecordVariadic(LazyNodeType.Custom, "LstmSequenceTrainStates", parameters,
                new[] { ((sequence ? b * t : b) + 2 * b) * h },
                (eng, output) => CopyResultInto(eng,
                    ((DirectGpuTensorEngine)eng).RunStatefulLstm(parameters, saved), output),
                LstmSequenceBackwardGpuFloat, saved);
        }
        else
        {
            GradientTape<float>.Current?.BindEngineIfUnset(this);
            packet = RunStatefulLstm(parameters, saved);
            DifferentiableOps.RecordIfActive("LstmSequenceTrainStates", packet, parameters,
                LstmSequenceBackwardGpuFloat, saved);
        }
        var result = UnpackLstmTrainingOutput(this, packet, b, t, h, sequence, true, out var hidden, out var cell);
        finalHidden = (Tensor<T>)(object)hidden;
        finalCell = (Tensor<T>)(object)cell;
        return (Tensor<T>)(object)result;
    }

    private Tensor<float> RunStatefulLstm(Tensor<float>[] inputs, object[] saved)
    {
        var backend = GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var meta = (int[])saved[3];
        int b = meta[0], t = meta[1], n = meta[2], h = meta[3], state = b * h;
        int outputLength = meta[4] != 0 ? b * t * h : state;
        using var x = GetOrAllocateBuffer(backend, inputs[0]);
        using var wi = GetOrAllocateBuffer(backend, inputs[1]);
        using var wh = GetOrAllocateBuffer(backend, inputs[2]);
        using var bi = meta[5] < 0 ? DeviceZeros(backend, 4 * h) : GetOrAllocateBuffer(backend, inputs[meta[5]]);
        using var bh = meta[6] < 0 ? DeviceZeros(backend, 4 * h) : GetOrAllocateBuffer(backend, inputs[meta[6]]);
        using var hi = meta[7] < 0 ? DeviceZeros(backend, state) : GetOrAllocateBuffer(backend, inputs[meta[7]]);
        using var ci = meta[8] < 0 ? DeviceZeros(backend, state) : GetOrAllocateBuffer(backend, inputs[meta[8]]);
        var owned = new List<IGpuBuffer>();
        IGpuBuffer Allocate(int length) { var buffer = backend.AllocateBuffer(length); owned.Add(buffer); return buffer; }
        try
        {
            var y = Allocate(b * t * h); var hf = Allocate(state); var cf = Allocate(state);
            var allH = Allocate((t + 1) * state); var allC = Allocate((t + 1) * state);
            var gates = Allocate(t * state * 4); var packet = Allocate(outputLength + 2 * state);
            PortableLstmSequence.Forward(backend, ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher,
                x.Buffer, hi.Buffer, ci.Buffer, wi.Buffer, wh.Buffer, bi.Buffer, bh.Buffer,
                y, hf, cf, allH, allC, gates, t, b, n, h);
            backend.Copy(meta[4] != 0 ? y : hf, 0, packet, 0, outputLength);
            backend.Copy(hf, 0, packet, outputLength, state);
            backend.Copy(cf, 0, packet, outputLength + state, state);
            backend.Synchronize();
            Tensor<float> HandOff(IGpuBuffer buffer, int length)
            {
                var result = DeferTensorResult<float>(backend, buffer, length, new[] { length });
                owned.Remove(buffer); return result;
            }
            saved[0] = HandOff(gates, t * state * 4);
            saved[1] = HandOff(allH, (t + 1) * state);
            saved[2] = HandOff(allC, (t + 1) * state);
            return HandOff(packet, outputLength + 2 * state);
        }
        finally
        {
            backend.Synchronize();
            foreach (var buffer in owned) buffer.Dispose();
        }
    }
}
