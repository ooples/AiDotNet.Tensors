using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

// Host fast paths for elementwise backwards whose generic form chains allocating engine ops. Clamp's gradient took six
// activation-sized temporaries (clamp, subtract, abs, multiply-by-zero, add-one, sign) before its multiply, Sqrt's and
// Abs's two, Log's and Exp's one, and each one was left to the collector every step. Here the gradient is computed in
// one pass into one rented array, which goes back to the pool once it has been added into the input's gradient.
//
// Each kernel performs the same floating-point operations per element as the engine ops it replaces, so results are
// bit-identical: Exp g*y, Log g/x, Sqrt g/(y*2), Abs g*sign(x) with the engine's sign (+1 / -1 / 0, NaN to 0), Clamp
// g*mask with mask 1 where min <= x <= max and 0 elsewhere (what clamp(x) == x gives for a finite x). A non-finite
// Clamp input falls back to the engine path, whose mask is NaN there.
//
// Reads go through AsSpanUnmarked under KeepAliveScope: the kernel keeps none of the storage, so the operands are not
// marked escaped (a marked tensor can never be handed back to the pool).
//
// Taken only on a host engine with no tape recording the backward (higher-order AD needs the engine ops on the tape)
// and no graph trace, for float and double contiguous host tensors of the input's length. Anything else, or an
// unsupported type, keeps the engine path.
internal static partial class BackwardFunctions<T>
{
    private enum HostUnaryGradient
    {
        Exp,
        Log,
        Sqrt,
        Abs,
        Clamp,
    }

    private static bool TryHostUnaryBackward(HostUnaryGradient kind, Tensor<T> gradOutput, Tensor<T> input,
        Tensor<T> output, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads, double min = 0, double max = 0)
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) return false;
        if (engine.SupportsGpu || DifferentiableOps.IsRecording<T>() || Compilation.GraphMode.IsActive) return false;
        int length = input.Length;
        if (gradOutput.Length != length || output.Length != length) return false;
        if (!IsPlainHost(gradOutput) || !IsPlainHost(input) || !IsPlainHost(output)) return false;

        var grad = TensorAllocator.RentUninitialized<T>((int[])input._shape.Clone());
        bool computed;
        using (var gradOutputKeep = new KeepAliveScope(gradOutput))
        using (var inputKeep = new KeepAliveScope(input))
        using (var outputKeep = new KeepAliveScope(output))
        using (var gradKeep = new KeepAliveScope(grad))
        {
            computed = typeof(T) == typeof(float)
                ? ComputeFloat(kind, (Tensor<float>)(object)gradOutput, (Tensor<float>)(object)input,
                    (Tensor<float>)(object)output, (Tensor<float>)(object)grad, (float)min, (float)max)
                : ComputeDouble(kind, (Tensor<double>)(object)gradOutput, (Tensor<double>)(object)input,
                    (Tensor<double>)(object)output, (Tensor<double>)(object)grad, min, max);
        }

        if (!computed)
        {
            ((IScopedHostTensor)grad).ReleaseHostFromScope();
            return false;
        }

        if (DifferentiableOps.AccumulateGradPoolable(grads, input, grad, engine))
            ((IScopedHostTensor)grad).ReleaseHostFromScope();
        return true;
    }

    private static bool IsPlainHost(Tensor<T> tensor)
        => tensor.IsContiguous && !tensor.IsSparse && tensor._gpuBuffer is null && tensor._device == TensorDevice.CPU
            && !tensor.HasPendingGpuData;

    private static bool ComputeFloat(HostUnaryGradient kind, Tensor<float> gradOutput, Tensor<float> input,
        Tensor<float> output, Tensor<float> grad, float min, float max)
    {
        ReadOnlySpan<float> g = gradOutput.AsSpanUnmarked();
        ReadOnlySpan<float> x = input.AsSpanUnmarked();
        ReadOnlySpan<float> y = output.AsSpanUnmarked();
        Span<float> d = grad.AsWritableSpanUnmarked();
        switch (kind)
        {
            case HostUnaryGradient.Exp:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] * y[i];
                return true;
            case HostUnaryGradient.Log:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] / x[i];
                return true;
            case HostUnaryGradient.Sqrt:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] / (y[i] * 2f);
                return true;
            case HostUnaryGradient.Abs:
                for (int i = 0; i < d.Length; i++)
                {
                    float v = x[i];
                    d[i] = g[i] * (v > 0f ? 1f : (v < 0f ? -1f : 0f));
                }
                return true;
            default:
                if (!(min <= max)) return false;
                for (int i = 0; i < d.Length; i++)
                {
                    float v = x[i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) return false;
                    d[i] = g[i] * (v >= min && v <= max ? 1f : 0f);
                }
                return true;
        }
    }

    private static bool ComputeDouble(HostUnaryGradient kind, Tensor<double> gradOutput, Tensor<double> input,
        Tensor<double> output, Tensor<double> grad, double min, double max)
    {
        ReadOnlySpan<double> g = gradOutput.AsSpanUnmarked();
        ReadOnlySpan<double> x = input.AsSpanUnmarked();
        ReadOnlySpan<double> y = output.AsSpanUnmarked();
        Span<double> d = grad.AsWritableSpanUnmarked();
        switch (kind)
        {
            case HostUnaryGradient.Exp:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] * y[i];
                return true;
            case HostUnaryGradient.Log:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] / x[i];
                return true;
            case HostUnaryGradient.Sqrt:
                for (int i = 0; i < d.Length; i++) d[i] = g[i] / (y[i] * 2d);
                return true;
            case HostUnaryGradient.Abs:
                for (int i = 0; i < d.Length; i++)
                {
                    double v = x[i];
                    d[i] = g[i] * (v > 0d ? 1d : (v < 0d ? -1d : 0d));
                }
                return true;
            default:
                if (!(min <= max)) return false;
                for (int i = 0; i < d.Length; i++)
                {
                    double v = x[i];
                    if (double.IsNaN(v) || double.IsInfinity(v)) return false;
                    d[i] = g[i] * (v >= min && v <= max ? 1d : 0d);
                }
                return true;
        }
    }
}
