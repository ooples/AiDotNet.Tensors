using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    public virtual Tensor<T> LstmCellFromGates<T>(Tensor<T> gates, Tensor<T> previousCell, out Tensor<T> cell)
    {
        if (gates is null) throw new ArgumentNullException(nameof(gates));
        if (previousCell is null) throw new ArgumentNullException(nameof(previousCell));
        if (previousCell.Rank < 1 || gates.Rank != previousCell.Rank)
            throw new ArgumentException("Gates and cell must have matching nonzero ranks.");
        int hidden = previousCell.Shape[previousCell.Rank - 1];
        for (int d = 0; d < gates.Rank; d++)
            if (gates.Shape[d] != (d == gates.Rank - 1 ? checked(4 * hidden) : previousCell.Shape[d]))
                throw new ArgumentException("Gates must match the cell shape with four times the last dimension.", nameof(gates));
        if (hidden == 0) throw new ArgumentException("Cell width must be positive.", nameof(previousCell));
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            Tensor<T> Gate(int i) => TensorNarrow(gates, gates.Rank - 1, i * hidden, hidden);
            cell = TensorAdd(TensorMultiply(Sigmoid(Gate(1)), previousCell), TensorMultiply(Sigmoid(Gate(0)), Tanh(Gate(2))));
            return TensorMultiply(Sigmoid(Gate(3)), Tanh(cell));
        }
        int count = previousCell.Length;
        var inputs = new[] { gates, previousCell };
        Tensor<T> packet;
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
            packet = scope.RecordVariadic(LazyNodeType.Custom, "LstmCellFromGates", inputs,
                new[] { checked(2 * count) },
                (e, output) => DirectGpuTensorEngine.CopyResultInto(e, ((CpuEngine)e).RunLstmGatePacket(gates, previousCell), output),
                LstmGateBackward<T>, null);
        else
        {
            GradientTape<T>.Current?.BindEngineIfUnset(this);
            packet = RunLstmGatePacket(gates, previousCell);
            DifferentiableOps.RecordIfActive("LstmCellFromGates", packet, inputs, LstmGateBackward<T>, null);
        }
        cell = Reshape(TensorNarrow(packet, 0, count, count), previousCell.Shape.ToArray());
        return Reshape(TensorNarrow(packet, 0, 0, count), previousCell.Shape.ToArray());
    }

    private static double LstmGateSigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    protected virtual Tensor<T> RunLstmGatePacket<T>(Tensor<T> gates, Tensor<T> previousCell)
    {
        if (typeof(T) == typeof(float))
            return (Tensor<T>)(object)RunLstmGatePacketFloat((Tensor<float>)(object)gates, (Tensor<float>)(object)previousCell);
        var ops = MathHelper.GetNumericOperations<T>();
        int count = previousCell.Length, hidden = previousCell.Shape[previousCell.Rank - 1];
        var pre = gates.GetFlattenedData(); var old = previousCell.GetFlattenedData();
        var result = new Tensor<T>(new[] { checked(2 * count) });
        using var write = result.LeaseWritable();
        for (int k = 0; k < count; k++)
        {
            int g = (k / hidden) * 4 * hidden + k % hidden;
            double i = LstmGateSigmoid(ops.ToDouble(pre[g]));
            double f = LstmGateSigmoid(ops.ToDouble(pre[g + hidden]));
            double z = Math.Tanh(ops.ToDouble(pre[g + 2 * hidden]));
            double o = LstmGateSigmoid(ops.ToDouble(pre[g + 3 * hidden]));
            double c = f * ops.ToDouble(old[k]) + i * z;
            write.Span[k] = ops.FromDouble(o * Math.Tanh(c));
            write.Span[count + k] = ops.FromDouble(c);
        }
        return result;
    }

    private static void LstmGateBackward<T>(Tensor<T> gradient, Tensor<T>[] inputs, Tensor<T> output,
        object[]? saved, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (engine is DirectGpuTensorEngine gpu && gpu.TryLstmGateBackward(gradient, inputs, output, grads)) return;
        if (typeof(T) == typeof(float))
        {
            var (dgFloat, dcFloat) = LstmGateBackwardFloat((Tensor<float>)(object)gradient,
                (Tensor<float>)(object)inputs[0], (Tensor<float>)(object)inputs[1], (Tensor<float>)(object)output);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], (Tensor<T>)(object)dgFloat, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], (Tensor<T>)(object)dcFloat, engine);
            return;
        }
        var ops = MathHelper.GetNumericOperations<T>();
        int count = inputs[1].Length, hidden = inputs[1].Shape[inputs[1].Rank - 1];
        var pre = inputs[0].GetFlattenedData(); var old = inputs[1].GetFlattenedData();
        var dy = gradient.GetFlattenedData(); var y = output.GetFlattenedData();
        var dg = new Tensor<T>(inputs[0].Shape.ToArray()); var dc = new Tensor<T>(inputs[1].Shape.ToArray());
        using (var gw = dg.LeaseWritable())
        using (var cw = dc.LeaseWritable())
        for (int k = 0; k < count; k++)
        {
            int g = (k / hidden) * 4 * hidden + k % hidden;
            double i = LstmGateSigmoid(ops.ToDouble(pre[g]));
            double f = LstmGateSigmoid(ops.ToDouble(pre[g + hidden]));
            double z = Math.Tanh(ops.ToDouble(pre[g + 2 * hidden]));
            double o = LstmGateSigmoid(ops.ToDouble(pre[g + 3 * hidden]));
            double tc = Math.Tanh(ops.ToDouble(y[count + k])), dh = ops.ToDouble(dy[k]);
            double d = ops.ToDouble(dy[count + k]) + dh * o * (1 - tc * tc);
            gw.Span[g] = ops.FromDouble(d * z * i * (1 - i));
            gw.Span[g + hidden] = ops.FromDouble(d * ops.ToDouble(old[k]) * f * (1 - f));
            gw.Span[g + 2 * hidden] = ops.FromDouble(d * i * (1 - z * z));
            gw.Span[g + 3 * hidden] = ops.FromDouble(dh * tc * o * (1 - o));
            cw.Span[k] = ops.FromDouble(d * f);
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], dg, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], dc, engine);
    }
    private static float[] ActivateLstmGatesFloat(Tensor<float> gates, int hidden)
    {
        var pre = gates.GetFlattenedData();
        var activated = new float[gates.Length];
        for (int offset = 0; offset < activated.Length; offset += 4 * hidden)
        {
            Simd.SimdKernels.Sigmoid(pre.AsSpan(offset, hidden), activated.AsSpan(offset, hidden));
            Simd.SimdKernels.Sigmoid(pre.AsSpan(offset + hidden, hidden), activated.AsSpan(offset + hidden, hidden));
            Simd.SimdKernels.Tanh(pre.AsSpan(offset + 2 * hidden, hidden), activated.AsSpan(offset + 2 * hidden, hidden));
            Simd.SimdKernels.Sigmoid(pre.AsSpan(offset + 3 * hidden, hidden), activated.AsSpan(offset + 3 * hidden, hidden));
        }
        return activated;
    }

    private static Tensor<float> RunLstmGatePacketFloat(Tensor<float> gates, Tensor<float> previousCell)
    {
        int count = previousCell.Length, hidden = previousCell.Shape[previousCell.Rank - 1];
        var act = ActivateLstmGatesFloat(gates, hidden); var old = previousCell.GetFlattenedData();
        var result = new Tensor<float>(new[] { checked(2 * count) });
        using var write = result.LeaseWritable();
        var cell = write.Span.Slice(count, count); var hiddenOut = write.Span.Slice(0, count);
        for (int k = 0; k < count; k++)
        {
            int g = (k / hidden) * 4 * hidden + k % hidden;
            cell[k] = act[g + hidden] * old[k] + act[g] * act[g + 2 * hidden];
        }
        Simd.SimdKernels.Tanh(cell, hiddenOut);
        for (int k = 0; k < count; k++)
            hiddenOut[k] *= act[(k / hidden) * 4 * hidden + k % hidden + 3 * hidden];
        return result;
    }

    private static (Tensor<float> gates, Tensor<float> cell) LstmGateBackwardFloat(
        Tensor<float> gradient, Tensor<float> gates, Tensor<float> previousCell, Tensor<float> output)
    {
        int count = previousCell.Length, hidden = previousCell.Shape[previousCell.Rank - 1];
        var act = ActivateLstmGatesFloat(gates, hidden); var old = previousCell.GetFlattenedData();
        var dy = gradient.GetFlattenedData(); var y = output.GetFlattenedData();
        var tc = new float[count]; Simd.SimdKernels.Tanh(y.AsSpan(count, count), tc.AsSpan());
        var dg = new Tensor<float>(gates.Shape.ToArray()); var dc = new Tensor<float>(previousCell.Shape.ToArray());
        using var gw = dg.LeaseWritable(); using var cw = dc.LeaseWritable();
        for (int k = 0; k < count; k++)
        {
            int g = (k / hidden) * 4 * hidden + k % hidden;
            float i = act[g], f = act[g + hidden], z = act[g + 2 * hidden], o = act[g + 3 * hidden];
            float d = dy[count + k] + dy[k] * o * (1 - tc[k] * tc[k]);
            gw.Span[g] = d * z * i * (1 - i);
            gw.Span[g + hidden] = d * old[k] * f * (1 - f);
            gw.Span[g + 2 * hidden] = d * i * (1 - z * z);
            gw.Span[g + 3 * hidden] = dy[k] * tc[k] * o * (1 - o);
            cw.Span[k] = d * f;
        }
        return (dg, dc);
    }

}
