using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    // The create-graph path in its own method: its lambda captures the operand, and a closure declared in the backward
    // itself would be allocated on every first-order call too.
    private static bool RfftMagnitudeBackwardCreateGraph(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, double epsilon)
    {
        var input = inputs[0];
        return TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstInput,
            e => RfftMagnitudeHigherOrder(e, input, epsilon));
    }

    internal static void RfftMagnitudeBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0];
        double epsilon = (double)savedState[0];
        if (CreatesGraph && RfftMagnitudeBackwardCreateGraph(gradOutput, inputs, engine, grads, epsilon)) return;
        if (!DifferentiableOps.IsGradientRequired(input)) return;
        var target = DifferentiableOps.PeekDirectDeviceGradTarget(grads, input, engine, out bool add);
        if (target is null && grads.TryGetValue(input, out target))
            add = DifferentiableOps.GradWriteGeneration == 0 || DifferentiableOps.IsWrittenThisStep(target);
        if (target is null)
        {
            add = false;
            target = engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                ? gpu.CreateRfftMagnitudeGradient(input) : TensorAllocator.RentUninitialized<T>(input._shape);
        }
        engine.TensorRfftMagnitudeBackwardInto(gradOutput, input, target, epsilon, add);
        DifferentiableOps.CommitDirectGradWrite(grads, input, target);
    }

    // The first-order operator uses a native FFT and an analytic adjoint. Explicit
    // higher-order requests retain a graph of resident primitive operations.
    private static Tensor<T> RfftMagnitudeHigherOrder(IEngine engine, Tensor<T> input, double epsilon)
    {
        var (rows, length, bins) = CpuEngine.ValidateRfftMagnitude(input, null, epsilon);
        var (cosine, sine) = RfftMagnitudeCoefficientConstants(length, bins);
        var framed = engine.Reshape(input, new[] { rows, length });
        var real = engine.TensorMatMul(framed, cosine);
        var imaginary = engine.TensorMatMul(framed, sine);
        var magnitude = engine.TensorSqrt(engine.TensorAddScalar(
            engine.TensorAdd(engine.TensorMultiply(real, real), engine.TensorMultiply(imaginary, imaginary)), Num(epsilon)));
        int[] shape = (int[])input._shape.Clone();
        shape[shape.Length - 1] = bins;
        return engine.Reshape(magnitude, shape);
    }

    [ScalarKernel("Builds immutable DFT coefficient constants only for explicit higher-order differentiation; it never reads tensor data or runs on the ordinary FFT path.")]
    private static (Tensor<T> Cosine, Tensor<T> Sine) RfftMagnitudeCoefficientConstants(int length, int bins)
    {
        var cosine = new double[checked(length * bins)];
        var sine = new double[cosine.Length];
        for (int n = 0; n < length; n++)
            for (int k = 0; k < bins; k++)
            {
                double angle = 2 * System.Math.PI * n * k / length;
                cosine[n * bins + k] = System.Math.Cos(angle);
                sine[n * bins + k] = -System.Math.Sin(angle);
            }
        return (Constant(cosine, new[] { length, bins }), Constant(sine, new[] { length, bins }));
    }
}
