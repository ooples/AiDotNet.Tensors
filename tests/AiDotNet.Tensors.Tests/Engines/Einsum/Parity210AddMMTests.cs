using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210AddMMTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b) => System.MathF.Abs(a - b) < 1e-4f;

    [Fact]
    public void AddMM_DefaultAlphaBeta_MatMulPlusInput()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f }, 2, 2);
        var input = T(new[] { 10f, 20f, 30f, 40f }, 2, 2);
        var r = E.TensorAddMM(input, a, b);  // default alpha=beta=1
        // A·B = [[1·5+2·7, 1·6+2·8],[3·5+4·7,3·6+4·8]] = [[19,22],[43,50]]
        // + input = [[29, 42], [73, 90]]
        Assert.Equal(new[] { 29f, 42f, 73f, 90f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void AddMM_CustomAlpha()
    {
        var a = T(new[] { 1f, 0f, 0f, 1f }, 2, 2);  // identity
        var b = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var input = T(new[] { 0f, 0f, 0f, 0f }, 2, 2);
        var r = E.TensorAddMM(input, a, b, 2f, 1f);
        // 2 · (I · B) + 0 = 2 · B = [[2,4],[6,8]]
        Assert.Equal(new[] { 2f, 4f, 6f, 8f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void AddMM_BetaZero_PureMatMul()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 5f, 6f, 7f, 8f }, 2, 2);
        var input = T(new[] { 999f, 999f, 999f, 999f }, 2, 2);
        var r = E.TensorAddMM(input, a, b, 1f, 0f);
        Assert.Equal(new[] { 19f, 22f, 43f, 50f }, r.AsSpan().ToArray());
    }

    [Fact]
    public void AddMM_ShapeMismatch_Throws()
    {
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 1f, 2f, 3f }, 3, 1);  // inner dim mismatch
        var input = T(new[] { 0f, 0f }, 2, 1);
        Assert.Throws<System.ArgumentException>(() => E.TensorAddMM(input, a, b));
    }
}
