using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Compiling an inference plan must not change what eager ops compute. The compiler used to register every constant
/// matmul weight in the process-wide FrozenWeightRegistry, which moved every later eager TensorMatMul on that weight
/// onto the pre-packed BlasManaged kernel. Its summation order differs from the eager kernel's, so eager results
/// depended on whether some plan had ever been compiled over the weight: here the small-M classifier-head shapes
/// (M = 1, 2 against a [576, 1000] weight) changed, and in AiDotNet ViT, MobileNetV3 and four multimodal families
/// drifted by a few ULP after their first compiled predict. It also kept a packed copy of every compiled weight for
/// the weight's lifetime.
/// </summary>
public class CompiledPlanEagerIsolationTests
{
    private static Tensor<T> Filled<T>(int[] shape, int seed, Func<double, T> conv)
    {
        var t = new Tensor<T>(shape);
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = conv(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Theory]
    [InlineData(1, 1, 576, 1000)]
    [InlineData(1, 2, 576, 1000)]
    [InlineData(1, 8, 64, 64)]
    [InlineData(2, 17, 64, 192)]
    [InlineData(2, 17, 192, 64)]
    [InlineData(1, 64, 256, 256)]
    [InlineData(1, 300, 512, 512)]
    public void CompilingAPlan_LeavesEagerMatMulOnItsWeightBitIdentical_Float(int batch, int m, int k, int n) =>
        Check<float>(batch, m, k, n, x => (float)x);

    [Theory]
    [InlineData(1, 1, 576, 1000)]
    [InlineData(1, 2, 576, 1000)]
    [InlineData(1, 8, 64, 64)]
    [InlineData(2, 17, 64, 192)]
    [InlineData(1, 64, 256, 256)]
    [InlineData(1, 300, 512, 512)]
    public void CompilingAPlan_LeavesEagerMatMulOnItsWeightBitIdentical_Double(int batch, int m, int k, int n) =>
        Check<double>(batch, m, k, n, x => x);

    private static void Check<T>(int batch, int m, int k, int n, Func<double, T> conv)
    {
        var engine = new CpuEngine();
        var w = Filled(new[] { k, n }, 7, conv);
        var x = batch == 1 ? Filled(new[] { m, k }, 1, conv) : Filled(new[] { batch, m, k }, 1, conv);
        T[] Eager() => engine.TensorMatMul(x, w).ToArray();
        var before = Eager();
        Assert.Equal(before, Eager());

        using var cache = new CompiledModelCache<T>();
        var plan = cache.GetOrCompileInference(x, input => engine.TensorMatMul(input, w));
        plan.Execute();

        Assert.Equal(before, Eager());
    }
}
