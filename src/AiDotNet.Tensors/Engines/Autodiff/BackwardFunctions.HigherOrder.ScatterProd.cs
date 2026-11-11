using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// The create-graph gradient of <c>scatter_reduce(..., "prod")</c> when a factor is zero, as recorded ops with
    /// torch's semantics (<c>scatter_reduce_backward</c>):
    /// <list type="bullet">
    /// <item>destination: g · R_self / masked_self, where masked_self is the destination with its zeros replaced by 1
    /// and R_self the reduce of masked_self with the source — for a non-zero element that is g·out/x, for a zero one the
    /// product of the slot's other factors;</item>
    /// <item>source: where the element is the slot's only zero source factor, g · R_src gathered (R_src the reduce of the
    /// destination with the source's zeros replaced by 1, i.e. the product of the others); otherwise g·out/x gathered
    /// with x's zeros replaced by 1; and 0 where the slot holds two or more zero source factors.</item>
    /// </list>
    /// The masks and the routing are constants fixed by the forward; everything else is recorded, so the result is
    /// differentiable again. (Where two source factors are zero torch refuses double backward; here the gradient is
    /// the constant 0 there, its exact value, whose own derivative in the other zero factor is dropped as in the
    /// first-order formula.)
    /// </summary>
    private static void ScatterReduceProdWithZerosRecorded(Tensor<T> gradOutput, Tensor<T> tensor, Tensor<T> source,
        Tensor<T> output, int dim, int[] idxData, bool includeSelf, bool[] selfParticipates,
        int outerSize, int innerSize, int dstAxis, int srcAxis, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var selfValues = RoIAlignPlan.ToDoubles(tensor);
        var srcValues = RoIAlignPlan.ToDoubles(source);
        int dstLen = selfValues.Length, srcLen = srcValues.Length;

        // Routing source element -> destination slot, and the number of zero source factors per slot.
        var srcTo = new List<int>();
        var srcFrom = new List<int>();
        var slotOfSource = new int[srcLen];
        for (int i = 0; i < srcLen; i++) slotOfSource[i] = -1;
        var zeroSources = new int[dstLen];
        for (int outer = 0; outer < outerSize; outer++)
        {
            int outerBase = outer * dstAxis * innerSize;
            for (int i = 0; i < srcAxis; i++)
            {
                int srcBase = outer * srcAxis * innerSize + i * innerSize;
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int flatSrc = srcBase + inner;
                    int target = idxData[flatSrc];
                    if (target < 0 || target >= dstAxis) continue;
                    int dstPos = outerBase + target * innerSize + inner;
                    slotOfSource[flatSrc] = dstPos;
                    srcTo.Add(flatSrc);
                    srcFrom.Add(dstPos);
                    if (srcValues[flatSrc] == 0.0) zeroSources[dstPos]++;
                }
            }
        }

        var selfZero = new T[dstLen];
        var selfKeep = new T[dstLen];
        for (int i = 0; i < dstLen; i++)
        {
            selfZero[i] = selfValues[i] == 0.0 ? ops.One : ops.Zero;
            selfKeep[i] = selfParticipates[i] ? ops.One : ops.Zero;
        }
        var srcZero = new T[srcLen];
        var srcSingleZero = new T[srcLen];
        var srcKeep = new T[srcLen];
        for (int i = 0; i < srcLen; i++)
        {
            int slot = slotOfSource[i];
            bool zero = srcValues[i] == 0.0;
            srcZero[i] = zero ? ops.One : ops.Zero;
            srcSingleZero[i] = zero && slot >= 0 && zeroSources[slot] == 1 ? ops.One : ops.Zero;
            srcKeep[i] = slot >= 0 && !(zero && zeroSources[slot] > 1) ? ops.One : ops.Zero;
        }

        var dstShape = (int[])tensor._shape.Clone();
        var srcShape = (int[])source._shape.Clone();
        var index = new Tensor<int>((int[])idxData.Clone(), (int[])srcShape.Clone());
        var onesDst = new Tensor<T>((int[])dstShape.Clone());
        onesDst.Fill(ops.One);
        var onesSrc = new Tensor<T>((int[])srcShape.Clone());
        onesSrc.Fill(ops.One);

        // Destination.
        var maskedSelf = engine.TensorWhere(new Tensor<T>(selfZero, (int[])dstShape.Clone()), onesDst, tensor);
        var maskedSelfResult = engine.TensorScatterReduce(maskedSelf, dim, index, source, ScatterReduceMode.Prod, includeSelf);
        var gradSelf = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorMultiply(gradOutput, maskedSelfResult), maskedSelf),
            new Tensor<T>(selfKeep, (int[])dstShape.Clone()));
        DifferentiableOps.AccumulateGrad(grads, tensor, gradSelf, engine);

        // Source.
        var to = srcTo.ToArray();
        var from = srcFrom.ToArray();
        var maskedSrc = engine.TensorWhere(new Tensor<T>(srcZero, (int[])srcShape.Clone()), onesSrc, source);
        var maskedSrcResult = engine.TensorScatterReduce(tensor, dim, index, maskedSrc, ScatterReduceMode.Prod, includeSelf);
        var single = SparseMap(engine.TensorMultiply(gradOutput, maskedSrcResult), (int[])srcShape.Clone(), to, from, null);
        var general = engine.TensorDivide(
            SparseMap(engine.TensorMultiply(gradOutput, output), (int[])srcShape.Clone(), to, from, null), maskedSrc);
        var gradSrc = engine.TensorMultiply(
            engine.TensorWhere(new Tensor<T>(srcSingleZero, (int[])srcShape.Clone()), single, general),
            new Tensor<T>(srcKeep, (int[])srcShape.Clone()));
        DifferentiableOps.AccumulateGrad(grads, source, gradSrc, engine);
    }
}
