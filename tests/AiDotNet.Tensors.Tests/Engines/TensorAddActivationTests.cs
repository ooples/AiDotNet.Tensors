using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TensorAddActivation (the fused residual join) must equal TensorAdd followed by the activation: bit for bit for ReLU on
/// the CPU and on the GPU kernel, including NaN, signed zero and infinities and lengths off every vector width, and its
/// gradient must equal the composed ops' gradient for both inputs.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class TensorAddActivationTests
{
    private const int Seed = 1638;

    private static Tensor<float> Values(int length, int salt)
    {
        var rng = RandomHelper.CreateSeededRandom(Seed + salt);
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 8 - 4);
        float[] specials = { float.NaN, -0f, 0f, float.PositiveInfinity, float.NegativeInfinity, -1e-30f };
        for (int i = 0; i < specials.Length && i < length; i++) data[(i * 7 + salt) % length] = specials[i];
        return new Tensor<float>(data, new[] { length });
    }

    private static void AssertBitEqual(Tensor<float> expected, Tensor<float> actual)
    {
        var e = expected.Contiguous().AsSpan();
        var a = actual.Contiguous().AsSpan();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Bits(e[i]) == Bits(a[i]),
                $"[{i}] expected {e[i]} got {a[i]}");
    }

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(33)]
    [InlineData(100_003)]   // above the parallel grain, odd tail
    public void CpuRelu_EqualsAddThenRelu(int length)
    {
        var engine = new CpuEngine();
        var a = Values(length, 1);
        var b = Values(length, 2);

        var fused = engine.TensorAddActivation(a, b, FusedActivationType.ReLU);

        AssertBitEqual(engine.ReLU(engine.TensorAdd(a, b)), fused);
    }

    [Theory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.Sigmoid)]
    [InlineData(FusedActivationType.GELU)]
    public void CpuOtherActivations_EqualTheComposition(FusedActivationType activation)
    {
        var engine = new CpuEngine();
        var a = Values(257, 3);
        var b = Values(257, 4);
        var sum = engine.TensorAdd(a, b);
        var expected = activation switch
        {
            FusedActivationType.None => sum,
            FusedActivationType.Sigmoid => engine.Sigmoid(sum),
            _ => engine.GELU(sum),
        };

        AssertBitEqual(expected, engine.TensorAddActivation(a, b, activation));
    }

    [Fact]
    public void CpuRelu_GradientMatchesTheComposition()
    {
        var engine = new CpuEngine();
        var a = Values(1031, 5);
        var b = Values(1031, 6);
        // Finite inputs: the reference and the fused op must agree on every mask bit.
        var aSpan = a.AsWritableSpan();
        var bSpan = b.AsWritableSpan();
        for (int i = 0; i < aSpan.Length; i++)
        {
            if (!IsFinite(aSpan[i])) aSpan[i] = 1.5f;
            if (!IsFinite(bSpan[i])) bSpan[i] = -0.5f;
        }
        var weight = Values(1031, 7);
        var weightSpan = weight.AsWritableSpan();
        for (int i = 0; i < weightSpan.Length; i++)
            if (!IsFinite(weightSpan[i])) weightSpan[i] = 2f;

        Tensor<float> fusedGradA, fusedGradB, refGradA, refGradB;
        using (var tape = new GradientTape<float>())
        {
            var loss = engine.ReduceSum(engine.TensorMultiply(engine.TensorAddActivation(a, b, FusedActivationType.ReLU), weight), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            fusedGradA = grads[a];
            fusedGradB = grads[b];
        }
        using (var tape = new GradientTape<float>())
        {
            var loss = engine.ReduceSum(engine.TensorMultiply(engine.ReLU(engine.TensorAdd(a, b)), weight), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            refGradA = grads[a];
            refGradB = grads[b];
        }

        AssertBitEqual(refGradA, fusedGradA);
        AssertBitEqual(refGradB, fusedGradB);
    }

    [Fact]
    public void CpuRelu_OnDLPackImportedOperands_EqualsAddThenRelu()
    {
        // A CPU DLPack import lives in native memory, not a managed array: the fused kernel must pin the logical memory.
        var engine = new CpuEngine();
        var a = Values(1031, 30);
        var b = Values(1031, 31);
        var importedA = Tensor<float>.FromDLPack(a.ToDLPack());
        var importedB = Tensor<float>.FromDLPack(b.ToDLPack());

        var fused = engine.TensorAddActivation(importedA, importedB, FusedActivationType.ReLU);

        AssertBitEqual(engine.ReLU(engine.TensorAdd(a, b)), fused);
    }

    [SkippableFact]
    public void GpuRelu_UnderFp16Autocast_StillClampsAtZero()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException or TypeInitializationException)
        {
            Skip.If(true, "No GPU backend: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU available");
            var a = Values(4096, 32);
            var b = Values(4096, 33);
            Tensor<float> fused;
            using (new AiDotNet.Tensors.Engines.Gpu.AutocastScope(AiDotNet.Tensors.Engines.Gpu.PrecisionMode.Float16))
                fused = gpu.TensorAddActivation(a, b, FusedActivationType.ReLU);

            var reference = new CpuEngine().ReLU(new CpuEngine().TensorAdd(a, b)).AsSpan();
            var actual = fused.Contiguous().AsSpan();
            for (int i = 0; i < actual.Length; i++)
            {
                if (!IsFinite(reference[i])) continue;
                Assert.True(actual[i] >= 0f, $"[{i}] {actual[i]} < 0: the FP16 route added without the ReLU");
                Assert.True(Math.Abs(reference[i] - actual[i]) <= 1e-2 * (1 + Math.Abs(reference[i])), $"[{i}] expected {reference[i]} got {actual[i]}");
            }
        }
    }

    // ReLU keeps NaN and turns -0 into +0, as PyTorch, CpuFusedOperations and every GPU relu kernel (relu_nan_safe) do.
    // The CPU ReLU and fused add + ReLU used max(x, +0), which turns NaN into +0: the CUDA lane's
    // GpuRelu_EqualsTheCpuResult read NaN on the GPU against 0 on the CPU. The lengths reach the scalar tail and the
    // AVX and AVX-512 loops.
    [Theory]
    [InlineData(5)]
    [InlineData(37)]
    [InlineData(4099)]
    [InlineData(100003)]
    public void CpuRelu_KeepsNaN_AndZeroesNegativeZero(int length)
    {
        var a = Values(length, 8);
        var b = Values(length, 9);
        var cpu = new CpuEngine();
        var fused = cpu.TensorAddActivation(a, b, FusedActivationType.ReLU).ToArray();
        var composed = cpu.ReLU(cpu.TensorAdd(a, b)).ToArray();
        var single = cpu.ReLU(a).ToArray();
        var av = a.ToArray();
        var bv = b.ToArray();
        int nan = 0, negativeZero = 0;
        for (int i = 0; i < length; i++)
        {
            float sum = av[i] + bv[i];
            float expected = float.IsNaN(sum) ? sum : (sum > 0 ? sum : 0f);
            Assert.True(Bits(expected) == Bits(fused[i]), $"fused [{i}] relu({sum}): expected {expected} got {fused[i]}");
            Assert.True(Bits(expected) == Bits(composed[i]), $"composed [{i}] relu({sum}): expected {expected} got {composed[i]}");
            float expectedA = float.IsNaN(av[i]) ? av[i] : (av[i] > 0 ? av[i] : 0f);
            Assert.True(Bits(expectedA) == Bits(single[i]), $"relu [{i}] relu({av[i]}): expected {expectedA} got {single[i]}");
            if (float.IsNaN(sum)) nan++;
            if (Bits(av[i]) == Bits(-0f)) negativeZero++;
        }
        Assert.True(nan > 0, "no NaN sum in the inputs: NaN propagation went unchecked");
        Assert.True(negativeZero > 0, "no -0 in the inputs: -0 -> +0 went unchecked");
    }

    [SkippableTheory]
    [InlineData(4096)]      // the float4 route where the backend has one
    [InlineData(4099)]      // the scalar route
    public void GpuRelu_EqualsTheCpuResult(int length)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException or TypeInitializationException)
        {
            Skip.If(true, "No GPU backend: " + ex.GetType().Name);
            return;
        }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU available");
            var a = Values(length, 8);
            var b = Values(length, 9);
            GpuLaunchProbe.Reset();

            var fused = gpu.TensorAddActivation(a, b, FusedActivationType.ReLU);
            var fallbacks = GpuLaunchProbe.Fallbacks;
            long launches = GpuLaunchProbe.Count;

            Assert.Empty(fallbacks);
            Assert.Equal(1, launches);   // one fused kernel, not an add launch and a relu launch
            AssertBitEqual(new CpuEngine().TensorAddActivation(a, b, FusedActivationType.ReLU), fused);
        }
    }
}
