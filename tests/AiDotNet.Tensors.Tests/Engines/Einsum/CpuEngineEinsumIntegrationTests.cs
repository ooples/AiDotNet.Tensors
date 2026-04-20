using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class CpuEngineEinsumIntegrationTests
{
    private static Tensor<float> Seq(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        int total = 1; foreach (var d in shape) total *= d;
        var idx = new int[shape.Length];
        for (int i = 0; i < total; i++)
        {
            t[idx] = i + 1;
            int k = shape.Length - 1;
            while (k >= 0)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
                k--;
            }
        }
        return t;
    }

    private static bool Close(float a, float b, float eps = 1e-3f)
        => System.MathF.Abs(a - b) <= eps * (1f + System.MathF.Abs(a) + System.MathF.Abs(b));

    [Fact]
    public void HardcodedFastPath_MatmulStillWorks()
    {
        // CpuEngine has a hardcoded fast-path for "ij,jk->ik" that routes to
        // TensorMatMul. Confirm numerical correctness even though this path
        // skips the generic executor.
        var engine = new CpuEngine();
        var A = Seq(3, 4);
        var B = Seq(4, 5);
        var R = engine.TensorEinsum("ij,jk->ik", A, B);
        Assert.Equal(new[] { 3, 5 }, R.Shape.ToArray());
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 5; k++)
            {
                float expected = 0;
                for (int j = 0; j < 4; j++) expected += A[i, j] * B[j, k];
                Assert.True(Close(expected, R[i, k]));
            }
    }

    [Fact]
    public void GenericPath_TraceWorks()
    {
        // Previously threw NotImplementedException; now routes through the
        // generic executor.
        var engine = new CpuEngine();
        var A = Seq(4, 4);
        var R = engine.TensorEinsum("ii->", A);
        Assert.Empty(R.Shape.ToArray());
        // trace of a 4x4 seq tensor = 1 + 6 + 11 + 16 = 34
        Assert.Equal(34f, R[System.Array.Empty<int>()]);
    }

    [Fact]
    public void GenericPath_ThreeWayChain()
    {
        var engine = new CpuEngine();
        var A = Seq(2, 3);
        var B = Seq(3, 4);
        var C = Seq(4, 5);
        var R = engine.TensorEinsum("ab,bc,cd->ad", A, B, C);
        Assert.Equal(new[] { 2, 5 }, R.Shape.ToArray());
        for (int i = 0; i < 2; i++)
            for (int l = 0; l < 5; l++)
            {
                float expected = 0;
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        expected += A[i, j] * B[j, k] * C[k, l];
                Assert.True(Close(expected, R[i, l]));
            }
    }

    [Fact]
    public void GenericPath_AttentionScores()
    {
        var engine = new CpuEngine();
        var A = Seq(1, 1, 3, 4);
        var B = Seq(1, 1, 5, 4);
        var R = engine.TensorEinsum("bhqd,bhkd->bhqk", A, B);
        Assert.Equal(new[] { 1, 1, 3, 5 }, R.Shape.ToArray());
    }

    [Fact]
    public void GenericPath_EllipsisBatchedMatmul()
    {
        var engine = new CpuEngine();
        var A = Seq(2, 3, 4);
        var B = Seq(2, 4, 5);
        var R = engine.TensorEinsum("...ij,...jk->...ik", A, B);
        Assert.Equal(new[] { 2, 3, 5 }, R.Shape.ToArray());
    }

    [Fact]
    public void GenericPath_ImplicitOutput()
    {
        var engine = new CpuEngine();
        var A = Seq(3, 4);
        var B = Seq(4, 5);
        // "ij,jk" without explicit output should behave like "ij,jk->ik"
        var R = engine.TensorEinsum("ij,jk", A, B);
        Assert.Equal(new[] { 3, 5 }, R.Shape.ToArray());
    }
}
