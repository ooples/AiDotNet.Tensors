using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Tests for the issue #854 specialized-scientific direct-PTX kernels.</summary>
public class DirectPtxScientificTests
{
    [Fact]
    public void ScientificCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(10, DirectPtxScientificCoverageManifest.All.Count);
        string[] names = DirectPtxScientificCoverageManifest.All.Select(c => c.Api).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxScientificCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(DirectPtxScientificCoverageStatus.ExperimentalDirectPtx,
            DirectPtxScientificCoverageManifest.Get("CudaBackend.ComplexMultiply").Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxScientificCoverageManifest.Get("UnassignedScientificApi"));
    }

    [Fact]
    public void ComplexMultiplyEmitter_IsElementwisePairKernel()
    {
        string ptx = PtxComplexMultiplyKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxComplexMultiplyKernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));   // ar, ai, br, bi
        Assert.Equal(2, Count(ptx, "st.global.f32"));       // real, imag
        Assert.Contains("sub.rn.f32 %f6, %f4, %f5", ptx);   // ar*br - ai*bi
        Assert.Contains("fma.rn.f32 %f8, %f1, %f2, %f7", ptx); // ai*br + ar*bi
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxComplexMultiplyKernel.IsSupportedPairs(16384));
        Assert.False(PtxComplexMultiplyKernel.IsSupportedPairs(100));
        Assert.False(PtxComplexMultiplyKernel.IsPromotedPairs(16384));
    }

    [SkippableFact]
    public void DriverOnlyComplexMultiply_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in complex-multiply specialization is measured on GA10x/SM86.");
        const int pairs = 16384;
        using var kernel = new PtxComplexMultiplyKernel(runtime, pairs);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20265900);
        float[] aHost = Values(random, pairs * 2, 2.0f);
        float[] bHost = Values(random, pairs * 2, 2.0f);
        var expected = new float[pairs * 2];
        for (int i = 0; i < pairs; i++)
        {
            float ar = aHost[2 * i], ai = aHost[2 * i + 1], br = bHost[2 * i], bi = bHost[2 * i + 1];
            expected[2 * i] = ar * br - ai * bi;
            expected[2 * i + 1] = ar * bi + ai * br;
        }

        using var a = runtime.AllocateBytes((nuint)(aHost.Length * sizeof(float)));
        using var b = runtime.AllocateBytes((nuint)(bHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(pairs * 2 * sizeof(float)));
        a.Upload<float>(aHost);
        b.Upload<float>(bHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(a, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(b, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[pairs * 2];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, "complex multiply");
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
