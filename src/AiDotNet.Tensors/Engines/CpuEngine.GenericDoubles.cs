using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// The values of <paramref name="vector"/> as doubles, for a generic-T fallback that evaluates its function in
    /// double. Goes through <see cref="Interfaces.INumericOperations{T}.ToDouble"/>, which every element type
    /// supports; <c>Convert.ToDouble</c> threw for Half and BFloat16, which are not <c>IConvertible</c>.
    /// </summary>
    [ScalarKernel("Generic element types have no span conversion to double; this is the single conversion pass of a double-evaluated fallback.")]
    private static double[] GenericToDoubles<T>(Vector<T> vector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new double[vector.Length];
        for (int i = 0; i < values.Length; i++) values[i] = ops.ToDouble(vector[i]);
        return values;
    }

    /// <summary>Writes <paramref name="values"/> into <paramref name="destination"/>, each rounded to T once.</summary>
    [ScalarKernel("Generic element types have no span conversion from double; this is the single conversion pass of a double-evaluated fallback.")]
    private static void GenericFromDoubles<T>(double[] values, Vector<T> destination)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < values.Length; i++) destination[i] = ops.FromDouble(values[i]);
    }

    /// <summary>The attention-logit soft cap <c>cap * tanh(x / cap)</c> of one generic-T score, evaluated in double.</summary>
    [ScalarKernel("The generic-T attention path scores one logit at a time; the soft cap is evaluated in double for that logit.")]
    private static T GenericSoftCap<T>(T score, double cap)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.FromDouble(cap * System.Math.Tanh(ops.ToDouble(score) / cap));
    }
}
