using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// CIoU's aspect-ratio term v = (4/π²)·(atan(w_t/h_t) - atan(w_p/h_p))² of the predicted boxes as recorded ops, for
    /// the create-graph backward of the CIoUAspect node (its first-order backward is a host loop, which the second
    /// backward saw as a constant). Widths and heights are relu(x2 - x1) + 1e-7 as in the forward; the target is
    /// differentiated too when it is the node's input (a node recorded by an older forward saved it as a constant).
    /// </summary>
    internal static Tensor<T> CIoUAspectRecorded(IEngine e, Tensor<T> predicted, Tensor<T> target, bool differentiateTarget)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = predicted._shape[0];
        var eps = ops.FromDouble(1e-7);
        Tensor<T> Column(Tensor<T> boxes, int c) => e.TensorSlice(boxes, new[] { 0, c }, new[] { n, 1 });
        Tensor<T> Extent(Tensor<T> boxes, int lo, int hi)
            => e.TensorAddScalar(e.ReLU(e.TensorSubtract(Column(boxes, hi), Column(boxes, lo))), eps);

        var predAngle = e.TensorAtan(e.TensorDivide(Extent(predicted, 0, 2), Extent(predicted, 1, 3)));
        Tensor<T> targAngle;
        if (differentiateTarget)
        {
            targAngle = e.TensorAtan(e.TensorDivide(Extent(target, 0, 2), Extent(target, 1, 3)));
        }
        else
        {
            using (new NoGradScope<T>())
                targAngle = e.TensorAtan(e.TensorDivide(Extent(target, 0, 2), Extent(target, 1, 3)));
        }
        var diff = e.TensorSubtract(targAngle, predAngle);
        return e.TensorMultiplyScalar(e.TensorMultiply(diff, diff), ops.FromDouble(4.0 / (System.Math.PI * System.Math.PI)));
    }
}
