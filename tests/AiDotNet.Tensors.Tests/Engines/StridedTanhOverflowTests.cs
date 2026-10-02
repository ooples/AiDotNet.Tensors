using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The engine kept three private copies of tanh as (e^2x - 1) / (e^2x + 1) after #1085 fixed
/// MathHelper.Tanh: the strided (non-contiguous, up to 4096 elements) paths of Tanh and Mish, and the
/// LSTM sequence kernel's scalar cell. Once e^2x overflows (x > ~44 in float) that is Inf / Inf = NaN
/// where tanh is exactly 1. A permuted activation fed to Tanh came back NaN (ooples/AiDotNet#2153:
/// ECAPA-TDNN attentive pooling, whose BatchNorm1d returns a [B, C, T] view).
/// </summary>
public class StridedTanhOverflowTests
{
    private static readonly float[] Values = { -1000f, -100f, -50f, -45f, -1f, 0f, 0.5f, 44f, 45f, 50f, 115f, 1000f };

    /// <summary>A [2, n] tensor's transpose: a non-contiguous [n, 2] view carrying Values twice.</summary>
    private static Tensor<float> StridedView(CpuEngine engine)
    {
        var data = new float[2 * Values.Length];
        for (int i = 0; i < Values.Length; i++)
        {
            data[i] = Values[i];
            data[Values.Length + i] = Values[i];
        }

        var view = engine.TensorPermute(new Tensor<float>(data, new[] { 2, Values.Length }), new[] { 1, 0 });
        Assert.False(view.IsContiguous);
        return view;
    }

    [Fact]
    public void Tanh_OnAStridedFloatView_SaturatesInsteadOfNaN()
    {
        var engine = new CpuEngine();
        var output = engine.Tanh(StridedView(engine));

        for (int i = 0; i < Values.Length; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float actual = output[i * 2 + j];
                Assert.False(float.IsNaN(actual), $"tanh({Values[i]}) was NaN");
                Assert.Equal((float)Math.Tanh(Values[i]), actual, 6);
            }
        }
    }

    [Fact]
    public void Mish_OnAStridedFloatView_IsFiniteForLargeInputs()
    {
        var engine = new CpuEngine();
        var output = engine.Mish(StridedView(engine));

        for (int i = 0; i < Values.Length; i++)
        {
            double x = Values[i];
            double softplus = x > 30 ? x : Math.Log(1 + Math.Exp(x));
            float expected = (float)(x * Math.Tanh(softplus));
            for (int j = 0; j < 2; j++)
            {
                float actual = output[i * 2 + j];
                Assert.False(float.IsNaN(actual), $"mish({Values[i]}) was NaN");
                Assert.Equal(expected, actual, Math.Max(1e-4f, Math.Abs(expected) * 1e-5f));
            }
        }
    }

    [Fact]
    public void LstmCellTanh_SaturatesInsteadOfNaN()
    {
        var method = typeof(CpuEngine).GetMethod("TanhScalar", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        var tanh = method!.MakeGenericMethod(typeof(float));

        foreach (float x in Values)
        {
            float actual = (float)tanh.Invoke(null, new object[] { x })!;
            Assert.False(float.IsNaN(actual), $"tanh({x}) was NaN");
            Assert.Equal((float)Math.Tanh(x), actual, 6);
        }
    }
}