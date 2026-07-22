using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Tests for the issue #839 pointwise/activation direct-PTX kernels.</summary>
public class DirectPtxPointwiseTests
{
    [Fact]
    public void PointwiseCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(19, DirectPtxPointwiseCoverageManifest.All.Count);
        string[] names = DirectPtxPointwiseCoverageManifest.All.Select(c => c.Api).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxPointwiseCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        foreach (string api in new[]
        {
            "CudaBackend.Relu", "CudaBackend.LeakyRelu", "CudaBackend.Sigmoid",
            "CudaBackend.Tanh", "CudaBackend.Gelu", "CudaBackend.Silu", "CudaBackend.Swish"
        })
            Assert.Equal(DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx,
                DirectPtxPointwiseCoverageManifest.Get(api).Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxPointwiseCoverageManifest.Get("UnassignedPointwiseApi"));
    }

    [Theory]
    [InlineData(0)]  // Relu
    [InlineData(1)]  // LeakyRelu
    [InlineData(2)]  // Sigmoid
    [InlineData(3)]  // Tanh
    [InlineData(4)]  // GeluTanh
    [InlineData(5)]  // Silu
    public void ActivationForwardEmitter_IsElementwiseNoScratchMemory(int activationValue)
    {
        var activation = (DirectPtxActivation)activationValue;
        string ptx = PtxActivationForwardKernel.EmitPtx(8, 6, 16384, activation, "aidotnet_activation_x");
        Assert.Equal(1, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        // Only the smooth activations invoke the transcendental approximation.
        bool smooth = activation is DirectPtxActivation.Sigmoid or DirectPtxActivation.Tanh
            or DirectPtxActivation.GeluTanh or DirectPtxActivation.Silu;
        Assert.Equal(smooth, ptx.Contains("tanh.approx.f32"));
        bool relumax = activation is DirectPtxActivation.Relu or DirectPtxActivation.LeakyRelu;
        Assert.Equal(relumax, ptx.Contains("max.f32 %f0"));
        Assert.True(PtxActivationForwardKernel.IsSupportedCount(16384));
        Assert.False(PtxActivationForwardKernel.IsSupportedCount(100));
        Assert.False(PtxActivationForwardKernel.IsPromotedCount(16384));
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void DriverOnlyActivationForward_MatchesOracle(int activationValue)
    {
        var activation = (DirectPtxActivation)activationValue;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedPointwise(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in activation specialization is measured on GA10x/SM86.");
        const int count = 16384;
        using var kernel = new PtxActivationForwardKernel(runtime, count, activation);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20265800 + (int)activation);
        float[] xHost = Values(random, count, 3.0f);
        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = (float)Activation(xHost[i], activation);

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(count * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[count];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, $"activation {activation}");
    }

    private static double Activation(double x, DirectPtxActivation activation)
    {
        double Sigmoid(double v) => 1.0 / (1.0 + Math.Exp(-v));
        return activation switch
        {
            DirectPtxActivation.Relu => Math.Max(x, 0.0),
            DirectPtxActivation.LeakyRelu => x > 0 ? x : 0.01 * x,
            DirectPtxActivation.Sigmoid => Sigmoid(x),
            DirectPtxActivation.Tanh => Math.Tanh(x),
            DirectPtxActivation.Silu => x * Sigmoid(x),
            DirectPtxActivation.GeluTanh =>
                0.5 * x * (1.0 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
            _ => throw new ArgumentOutOfRangeException(nameof(activation))
        };
    }

    private static float[] Values(Random random, int count, float magnitude)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)((random.NextDouble() * 2.0 - 1.0) * magnitude);
        return data;
    }

    private static void AssertVectorClose(float[] actual, float[] expected, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tolerance,
                $"{what}: index {i} expected {expected[i]} actual {actual[i]} (tol {tolerance}).");
    }

    private static int Count(string text, string value)
    {
        int count = 0, index = 0;
        while ((index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }
        return count;
    }
}
