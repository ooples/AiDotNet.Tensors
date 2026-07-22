using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
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
        Assert.Equal(15, DirectPtxScientificCoverageManifest.All.Count);
        string[] names = DirectPtxScientificCoverageManifest.All.Select(c => c.Api).ToArray();
        Assert.Contains("CudaBackend.RbfForward", names);
        Assert.Contains("CudaBackend.PairwiseDistance", names);
        Assert.Contains("CudaBackend.PairwiseDistanceSquared", names);
        Assert.Contains("CudaBackend.QuantumMeasurement", names);
        Assert.Contains("CudaBackend.ComplexMatVec", names);
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxScientificCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        // Every scoped op now has a direct-PTX owner.
        Assert.DoesNotContain(DirectPtxScientificCoverageManifest.All,
            c => c.Status == DirectPtxScientificCoverageStatus.PlannedDirectPtx);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxScientificCoverageManifest.Get("UnassignedScientificApi"));
    }

    [Fact]
    public void ComplexPhaseEmitter_IsMinimaxAtan2()
    {
        string ptx = PtxComplexPhaseKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxComplexPhaseKernel.EntryPoint, ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // re + im
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(2, Count(ptx, "abs.f32"));             // |re|, |im|
        Assert.Contains("max.f32 %f4", ptx);
        Assert.Contains("min.f32 %f5", ptx);
        Assert.Equal(4, Count(ptx, "selp.f32"));            // quadrant + degenerate folding
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxComplexPhaseKernel.IsSupportedCount(16384));
        Assert.False(PtxComplexPhaseKernel.IsPromotedCount(16384));
    }

    [SkippableFact]
    public void DriverOnlyComplexPhase_MatchesAtan2Oracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in complex-phase specialization is measured on GA10x/SM86.");
        const int count = 16384;
        using var kernel = new PtxComplexPhaseKernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266700);
        float[] re = Values(random, count, 2.0f);   // all four quadrants
        float[] im = Values(random, count, 2.0f);
        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = (float)Math.Atan2(im[i], re[i]);

        using var reBuf = runtime.AllocateBytes((nuint)(re.Length * sizeof(float)));
        using var imBuf = runtime.AllocateBytes((nuint)(im.Length * sizeof(float)));
        using var phaseBuf = runtime.AllocateBytes((nuint)(count * sizeof(float)));
        reBuf.Upload<float>(re);
        imBuf.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reBuf, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imBuf, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(phaseBuf, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count];
        phaseBuf.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, "complex phase (atan2)");
    }

    [Fact]
    public void ComplexConjugateEmitter_NegatesImaginaryPart()
    {
        string ptx = PtxComplexConjugateKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxComplexConjugateKernel.EntryPoint, ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.Contains("neg.f32 %f2, %f1", ptx);
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxComplexConjugateKernel.IsSupportedPairs(16384));
        Assert.False(PtxComplexConjugateKernel.IsPromotedPairs(16384));
    }

    [Fact]
    public void ComplexMagnitudeEmitter_UsesRoundToNearestSqrt()
    {
        string ptx = PtxComplexMagnitudeKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxComplexMagnitudeKernel.EntryPoint, ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Contains("fma.rn.f32 %f2, %f1, %f1, %f2", ptx);   // re^2 + im^2
        Assert.Contains("sqrt.rn.f32 %f3, %f2", ptx);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxComplexMagnitudeKernel.IsSupportedCount(16384));
        Assert.False(PtxComplexMagnitudeKernel.IsPromotedCount(16384));
    }

    [SkippableFact]
    public void DriverOnlyComplexConjugateAndMagnitude_MatchOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in complex specializations are measured on GA10x/SM86.");
        const int pairs = 16384;
        var random = RandomHelper.CreateSeededRandom(20266000);

        // Conjugate (interleaved).
        float[] cIn = Values(random, pairs * 2, 2.0f);
        var conjExpected = new float[pairs * 2];
        for (int i = 0; i < pairs; i++) { conjExpected[2 * i] = cIn[2 * i]; conjExpected[2 * i + 1] = -cIn[2 * i + 1]; }
        using (var conj = new PtxComplexConjugateKernel(runtime, pairs))
        using (var inBuf = runtime.AllocateBytes((nuint)(cIn.Length * sizeof(float))))
        using (var outBuf = runtime.AllocateBytes((nuint)(pairs * 2 * sizeof(float))))
        {
            Assert.Equal(0, conj.Audit.Function.LocalBytesPerThread);
            inBuf.Upload<float>(cIn);
            conj.Launch(
                DirectPtxTensorView.CreateOwned(inBuf, conj.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(outBuf, conj.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var got = new float[pairs * 2];
            outBuf.Download<float>(got);
            AssertVectorClose(got, conjExpected, 0f, "complex conjugate");
        }

        // Magnitude (split).
        float[] re = Values(random, pairs, 2.0f);
        float[] im = Values(random, pairs, 2.0f);
        var magExpected = new float[pairs];
        for (int i = 0; i < pairs; i++) magExpected[i] = (float)Math.Sqrt((double)re[i] * re[i] + (double)im[i] * im[i]);
        using (var mag = new PtxComplexMagnitudeKernel(runtime, pairs))
        using (var reBuf = runtime.AllocateBytes((nuint)(re.Length * sizeof(float))))
        using (var imBuf = runtime.AllocateBytes((nuint)(im.Length * sizeof(float))))
        using (var magBuf = runtime.AllocateBytes((nuint)(pairs * sizeof(float))))
        {
            reBuf.Upload<float>(re);
            imBuf.Upload<float>(im);
            mag.Launch(
                DirectPtxTensorView.CreateOwned(reBuf, mag.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imBuf, mag.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(magBuf, mag.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var got = new float[pairs];
            magBuf.Download<float>(got);
            AssertVectorClose(got, magExpected, 2e-3f, "complex magnitude");
        }
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

    [Fact]
    public void OctonionAddEmitter_AddsEightLanesPerThread()
    {
        string ptx = PtxOctonionAddKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxOctonionAddKernel.EntryPoint, ptx);
        Assert.Equal(16, Count(ptx, "ld.global.nc.f32"));   // 8 a + 8 b
        Assert.Equal(8, Count(ptx, "add.rn.f32"));
        Assert.Equal(8, Count(ptx, "st.global.f32"));
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxOctonionAddKernel.IsSupportedCount(16384));
        Assert.False(PtxOctonionAddKernel.IsPromotedCount(16384));
    }

    [SkippableFact]
    public void DriverOnlyOctonionAdd_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in octonion-add specialization is measured on GA10x/SM86.");
        const int count = 16384;
        using var kernel = new PtxOctonionAddKernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266100);
        float[] aHost = Values(random, count * 8, 2.0f);
        float[] bHost = Values(random, count * 8, 2.0f);
        var expected = new float[count * 8];
        for (int i = 0; i < expected.Length; i++) expected[i] = aHost[i] + bHost[i];

        using var a = runtime.AllocateBytes((nuint)(aHost.Length * sizeof(float)));
        using var b = runtime.AllocateBytes((nuint)(bHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(count * 8 * sizeof(float)));
        a.Upload<float>(aHost);
        b.Upload<float>(bHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(a, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(b, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count * 8];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 0f, "octonion add");
    }

    [Fact]
    public void RbfForwardEmitter_IsThreadPerPairSerialDim()
    {
        string ptx = PtxRbfForwardKernel.EmitPtx(8, 6, 256, 4, 8);
        Assert.Contains(PtxRbfForwardKernel.EntryPoint, ptx);
        Assert.Equal(3, Count(ptx, "ld.global.nc.f32"));    // input + centers in the loop, epsilon after
        Assert.Equal(1, Count(ptx, "st.global.f32"));        // one output per (batch,center) pair
        Assert.Equal(1, Count(ptx, "ex2.approx.f32"));       // expf via exp2
        Assert.Contains("$RBF_DIM_LOOP:", ptx);
        Assert.Contains("div.u32 %r3, %r2, 4", ptx);         // b = idx / numCenters
        Assert.Contains("rem.u32 %r4, %r2, 4", ptx);         // c = idx % numCenters
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxRbfForwardKernel.IsSupportedShape(256, 4, 8));
        Assert.False(PtxRbfForwardKernel.IsSupportedShape(255, 4, 8));   // pairs not a multiple of 256
        Assert.False(PtxRbfForwardKernel.IsPromotedShape(256, 4, 8));
    }

    [SkippableFact]
    public void DriverOnlyRbfForward_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in RBF-forward specialization is measured on GA10x/SM86.");
        const int batchSize = 128, numCenters = 8, inputDim = 12;   // pairs = 1024 (multiple of 256)
        using var kernel = new PtxRbfForwardKernel(runtime, batchSize, numCenters, inputDim);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20267100);
        float[] inputHost = Values(random, batchSize * inputDim, 1.0f);
        float[] centersHost = Values(random, numCenters * inputDim, 1.0f);
        float[] epsHost = new float[numCenters];
        for (int i = 0; i < numCenters; i++) epsHost[i] = (float)(0.2 + random.NextDouble());
        var expected = new float[batchSize * numCenters];
        for (int b = 0; b < batchSize; b++)
            for (int cc = 0; cc < numCenters; cc++)
            {
                double distSq = 0;
                for (int d = 0; d < inputDim; d++)
                {
                    double diff = inputHost[b * inputDim + d] - centersHost[cc * inputDim + d];
                    distSq += diff * diff;
                }
                expected[b * numCenters + cc] = (float)Math.Exp(-epsHost[cc] * distSq);
            }

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var centers = runtime.AllocateBytes((nuint)(centersHost.Length * sizeof(float)));
        using var epsilons = runtime.AllocateBytes((nuint)(epsHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<float>(inputHost);
        centers.Upload<float>(centersHost);
        epsilons.Upload<float>(epsHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(centers, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(epsilons, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, "rbf forward");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PairwiseDistanceEmitter_IsThreadPerPairSerialDim(bool squared)
    {
        string ptx = PtxPairwiseDistanceKernel.EmitPtx(8, 6, 32, 8, 4, squared);
        Assert.Contains(PtxPairwiseDistanceKernel.EntryPointFor(squared), ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));    // a + b in the loop
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(squared ? 0 : 1, Count(ptx, "sqrt.rn.f32"));   // L2 takes sqrt, squared does not
        Assert.Contains("$PD_DIM_LOOP:", ptx);
        Assert.Contains("div.u32 %r3, %r2, 8", ptx);         // i = idx / N
        Assert.Contains("rem.u32 %r4, %r2, 8", ptx);         // j = idx % N
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPairwiseDistanceKernel.IsSupportedShape(32, 8, 4));
        Assert.False(PtxPairwiseDistanceKernel.IsSupportedShape(33, 8, 4));  // pairs not multiple of 256
        Assert.False(PtxPairwiseDistanceKernel.IsPromotedShape(32, 8, 4));
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DriverOnlyPairwiseDistance_MatchesOracle(bool squared)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in pairwise-distance specialization is measured on GA10x/SM86.");
        const int m = 64, n = 16, dim = 10;   // pairs = 1024 (multiple of 256)
        using var kernel = new PtxPairwiseDistanceKernel(runtime, m, n, dim, squared);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20267200);
        float[] aHost = Values(random, m * dim, 1.0f);
        float[] bHost = Values(random, n * dim, 1.0f);
        var expected = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double distSq = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = aHost[i * dim + d] - bHost[j * dim + d];
                    distSq += diff * diff;
                }
                expected[i * n + j] = (float)(squared ? distSq : Math.Sqrt(distSq));
            }

        using var a = runtime.AllocateBytes((nuint)(aHost.Length * sizeof(float)));
        using var b = runtime.AllocateBytes((nuint)(bHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        a.Upload<float>(aHost);
        b.Upload<float>(bHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(a, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(b, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, squared ? "pairwise distance squared" : "pairwise distance");
    }

    [Fact]
    public void QuantumMeasurementEmitter_IsMagnitudeSquared()
    {
        string ptx = PtxQuantumMeasurementKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxQuantumMeasurementKernel.EntryPoint, ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));    // re + im
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(1, Count(ptx, "mul.rn.f32"));           // re*re
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));           // + im*im
        Assert.Equal(0, Count(ptx, "sqrt.rn.f32"));          // squared magnitude, no sqrt
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxQuantumMeasurementKernel.IsSupportedCount(16384));
        Assert.False(PtxQuantumMeasurementKernel.IsPromotedCount(16384));
    }

    [SkippableFact]
    public void DriverOnlyQuantumMeasurement_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in quantum-measurement specialization is measured on GA10x/SM86.");
        const int count = 16384;   // batchSize*stateSize
        using var kernel = new PtxQuantumMeasurementKernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20267300);
        float[] reHost = Values(random, count, 2.0f);
        float[] imHost = Values(random, count, 2.0f);
        var expected = new float[count];
        for (int i = 0; i < count; i++) expected[i] = reHost[i] * reHost[i] + imHost[i] * imHost[i];

        using var re = runtime.AllocateBytes((nuint)(count * sizeof(float)));
        using var im = runtime.AllocateBytes((nuint)(count * sizeof(float)));
        using var prob = runtime.AllocateBytes((nuint)(count * sizeof(float)));
        re.Upload<float>(reHost);
        im.Upload<float>(imHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(re, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(im, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(prob, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count];
        prob.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, "quantum measurement");
    }

    [Fact]
    public void ComplexMatVecEmitter_IsThreadPerRowSerialColumn()
    {
        string ptx = PtxComplexMatVecKernel.EmitPtx(8, 6, 4, 64);
        Assert.Contains(PtxComplexMatVecKernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));    // mr, mi, xr, xi in the loop
        Assert.Equal(2, Count(ptx, "st.global.f32"));        // outReal + outImag
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));           // complex MAC: 2 for real, 2 for imag
        Assert.Contains("$CMV_COL_LOOP:", ptx);
        Assert.Contains("div.u32 %r3, %r2, 64", ptx);        // b = idx / dim
        Assert.Contains("rem.u32 %r4, %r2, 64", ptx);        // row = idx % dim
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxComplexMatVecKernel.IsSupportedShape(4, 64));
        Assert.False(PtxComplexMatVecKernel.IsSupportedShape(3, 5));   // 15 not a multiple of 256
        Assert.False(PtxComplexMatVecKernel.IsPromotedShape(4, 64));
    }

    [SkippableFact]
    public void DriverOnlyComplexMatVec_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in complex-matvec specialization is measured on GA10x/SM86.");
        const int batchSize = 16, dim = 64;   // rows = 1024 (multiple of 256)
        using var kernel = new PtxComplexMatVecKernel(runtime, batchSize, dim);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20267400);
        float[] mr = Values(random, dim * dim, 1.0f), mi = Values(random, dim * dim, 1.0f);
        float[] xr = Values(random, batchSize * dim, 1.0f), xi = Values(random, batchSize * dim, 1.0f);
        var expReal = new float[batchSize * dim];
        var expImag = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
            for (int row = 0; row < dim; row++)
            {
                double sr = 0, si = 0;
                for (int col = 0; col < dim; col++)
                {
                    double a = mr[row * dim + col], bb = mi[row * dim + col];
                    double cr = xr[b * dim + col], ci = xi[b * dim + col];
                    sr += a * cr - bb * ci;
                    si += a * ci + bb * cr;
                }
                expReal[b * dim + row] = (float)sr;
                expImag[b * dim + row] = (float)si;
            }

        using var matReal = runtime.AllocateBytes((nuint)(mr.Length * sizeof(float)));
        using var matImag = runtime.AllocateBytes((nuint)(mi.Length * sizeof(float)));
        using var vecReal = runtime.AllocateBytes((nuint)(xr.Length * sizeof(float)));
        using var vecImag = runtime.AllocateBytes((nuint)(xi.Length * sizeof(float)));
        using var outReal = runtime.AllocateBytes((nuint)(expReal.Length * sizeof(float)));
        using var outImag = runtime.AllocateBytes((nuint)(expImag.Length * sizeof(float)));
        matReal.Upload<float>(mr); matImag.Upload<float>(mi);
        vecReal.Upload<float>(xr); vecImag.Upload<float>(xi);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(matReal, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(matImag, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(vecReal, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(vecImag, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(outReal, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(outImag, kernel.Blueprint.Tensors[5]));
        runtime.Synchronize();
        var actualReal = new float[expReal.Length];
        var actualImag = new float[expImag.Length];
        outReal.Download<float>(actualReal);
        outImag.Download<float>(actualImag);
        AssertVectorClose(actualReal, expReal, 3e-3f, "complex matvec real");
        AssertVectorClose(actualImag, expImag, 3e-3f, "complex matvec imag");
    }

    [Fact]
    public void OctonionMultiplyEmitter_IsRegisterResidentCayleyDickson()
    {
        string ptx = PtxOctonionMultiplyKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxOctonionMultiplyKernel.EntryPoint, ptx);
        Assert.Equal(16, Count(ptx, "ld.global.nc.f32"));   // 8 a + 8 b into registers
        Assert.Equal(8, Count(ptx, "st.global.f32"));        // 8 output components
        Assert.Equal(8, Count(ptx, "mul.rn.f32 %f16"));      // one leading product per component
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxOctonionMultiplyKernel.IsSupportedCount(16384));
        Assert.False(PtxOctonionMultiplyKernel.IsPromotedCount(16384));
    }

    [SkippableFact]
    public void DriverOnlyOctonionMultiply_MatchesCayleyDicksonOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in octonion-multiply specialization is measured on GA10x/SM86.");
        const int count = 16384;
        using var kernel = new PtxOctonionMultiplyKernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266200);
        float[] aHost = Values(random, count * 8, 1.5f);
        float[] bHost = Values(random, count * 8, 1.5f);
        var expected = new float[count * 8];
        for (int o = 0; o < count; o++)
        {
            int p = o * 8;
            double a0 = aHost[p], a1 = aHost[p + 1], a2 = aHost[p + 2], a3 = aHost[p + 3];
            double a4 = aHost[p + 4], a5 = aHost[p + 5], a6 = aHost[p + 6], a7 = aHost[p + 7];
            double b0 = bHost[p], b1 = bHost[p + 1], b2 = bHost[p + 2], b3 = bHost[p + 3];
            double b4 = bHost[p + 4], b5 = bHost[p + 5], b6 = bHost[p + 6], b7 = bHost[p + 7];
            expected[p]     = (float)(a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7);
            expected[p + 1] = (float)(a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6);
            expected[p + 2] = (float)(a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5);
            expected[p + 3] = (float)(a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4);
            expected[p + 4] = (float)(a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3);
            expected[p + 5] = (float)(a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2);
            expected[p + 6] = (float)(a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1);
            expected[p + 7] = (float)(a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0);
        }

        using var a = runtime.AllocateBytes((nuint)(aHost.Length * sizeof(float)));
        using var b = runtime.AllocateBytes((nuint)(bHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(count * 8 * sizeof(float)));
        a.Upload<float>(aHost);
        b.Upload<float>(bHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(a, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(b, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count * 8];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, "octonion multiply");
    }

    [Fact]
    public void MobiusAddEmitter_IsPerVectorTreeReducedFormula()
    {
        string ptx = PtxMobiusAddKernel.EmitPtx(8, 6, 64);
        Assert.Contains(PtxMobiusAddKernel.EntryPoint, ptx);
        Assert.Contains("ld.param.f32 %f5, [curvature];", ptx);
        Assert.Contains(".shared .align 16 .b8 x_sh[256]", ptx);   // dim=64
        Assert.Contains(".shared .align 16 .b8 red[512]", ptx);    // 128 lanes
        Assert.Contains("abs.f32 %f6, %f6", ptx);                  // |denom|
        Assert.Contains("rcp.approx.f32 %f9, %f6", ptx);           // 1/denom
        // Three reductions, each: store-barrier + 7 tree-halvings + post-load barrier.
        Assert.Equal(27, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxMobiusAddKernel.IsSupportedShape(64, 64));
        Assert.False(PtxMobiusAddKernel.IsSupportedShape(64, 100));
        Assert.False(PtxMobiusAddKernel.IsPromotedShape(64, 64));
    }

    [SkippableTheory]
    [InlineData(64, 32)]
    [InlineData(128, 64)]
    [InlineData(64, 128)]
    public void DriverOnlyMobiusAdd_MatchesOracle(int batch, int dim)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in mobius-add specialization is measured on GA10x/SM86.");
        const float c = 0.5f;
        using var kernel = new PtxMobiusAddKernel(runtime, batch, dim, c);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266300 + batch + dim);
        // Small magnitudes keep the vectors inside the Poincare ball.
        float[] xHost = Values(random, batch * dim, 0.2f);
        float[] yHost = Values(random, batch * dim, 0.2f);
        var expected = new float[batch * dim];
        for (int row = 0; row < batch; row++)
        {
            double xn = 0, yn = 0, dot = 0;
            for (int i = 0; i < dim; i++)
            {
                double xi = xHost[row * dim + i], yi = yHost[row * dim + i];
                xn += xi * xi; yn += yi * yi; dot += xi * yi;
            }
            double denom = 1.0 + 2.0 * c * dot + (double)c * c * xn * yn;
            denom = Math.Max(Math.Abs(denom), 1e-15);
            double coeff1 = 1.0 + 2.0 * c * dot + c * yn;
            double coeff2 = 1.0 - c * xn;
            for (int i = 0; i < dim; i++)
                expected[row * dim + i] = (float)((coeff1 * xHost[row * dim + i] + coeff2 * yHost[row * dim + i]) / denom);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var y = runtime.AllocateBytes((nuint)(yHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(batch * dim * sizeof(float)));
        x.Upload<float>(xHost);
        y.Upload<float>(yHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(y, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[batch * dim];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, $"mobius add {batch}x{dim}");
    }

    [Fact]
    public void PoincareDistanceEmitter_ReusesMobiusThenArctanh()
    {
        string ptx = PtxPoincareDistanceKernel.EmitPtx(8, 6, 64);
        Assert.Contains(PtxPoincareDistanceKernel.EntryPoint, ptx);
        Assert.Contains("ld.param.f32 %f5, [curvature];", ptx);
        Assert.Equal(2, Count(ptx, "sqrt.rn.f32"));   // diffNorm + sqrtC
        Assert.Equal(2, Count(ptx, "lg2.approx.f32")); // arctanh via log
        // Four reductions, each: store-barrier + 7 tree-halvings + post-load barrier.
        Assert.Equal(36, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPoincareDistanceKernel.IsSupportedShape(64, 128));
        Assert.False(PtxPoincareDistanceKernel.IsPromotedShape(64, 128));
    }

    [SkippableTheory]
    [InlineData(64, 32)]
    [InlineData(128, 64)]
    public void DriverOnlyPoincareDistance_MatchesOracle(int batch, int dim)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in poincare-distance specialization is measured on GA10x/SM86.");
        const float c = 0.5f;
        using var kernel = new PtxPoincareDistanceKernel(runtime, batch, dim, c);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266400 + batch + dim);
        float[] xHost = Values(random, batch * dim, 0.15f);
        float[] yHost = Values(random, batch * dim, 0.15f);
        var expected = new float[batch];
        for (int row = 0; row < batch; row++)
        {
            double xn = 0, yn = 0, dot = 0;
            for (int i = 0; i < dim; i++)
            {
                double xi = xHost[row * dim + i], yi = yHost[row * dim + i];
                xn += xi * xi; yn += yi * yi; dot += xi * yi;
            }
            double denom = 1.0 - 2.0 * c * dot + (double)c * c * xn * yn;
            denom = Math.Max(Math.Abs(denom), 1e-15);
            double coeff1 = 1.0 - 2.0 * c * dot + c * yn;
            double coeff2 = 1.0 - c * xn;
            double diffNormSq = 0;
            for (int i = 0; i < dim; i++)
            {
                double d = (coeff1 * -xHost[row * dim + i] + coeff2 * yHost[row * dim + i]) / denom;
                diffNormSq += d * d;
            }
            double sqrtC = Math.Sqrt(c);
            double arg = Math.Min(sqrtC * Math.Sqrt(diffNormSq), 1.0);
            expected[row] = (float)((1.0 / sqrtC) * Math.Log((1.0 + arg) / (1.0 - arg)));
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var y = runtime.AllocateBytes((nuint)(yHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(batch * sizeof(float)));
        x.Upload<float>(xHost);
        y.Upload<float>(yHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(y, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[batch];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 5e-3f, $"poincare distance {batch}x{dim}");
    }

    [Fact]
    public void PoincareProjectEmitter_IsThreadPerVectorConditionalRescale()
    {
        string ptx = PtxPoincareProjectKernel.EmitPtx(8, 6, 64);
        Assert.Contains(PtxPoincareProjectKernel.EntryPoint, ptx);
        Assert.Contains("ld.param.f32 %f1, [epsilon];", ptx);
        Assert.Contains("NORM_LOOP:", ptx);
        Assert.Contains("WRITE_LOOP:", ptx);
        Assert.Contains("setp.ge.f32 %p1, %f2, %f5", ptx);   // ||x||^2 >= maxNorm^2
        Assert.Contains("selp.f32 %f8, %f7, 0f3F800000, %p1", ptx); // scale or 1
        Assert.Equal(2, Count(ptx, "sqrt.rn.f32"));           // sqrt(c) + sqrt(sqNorm)
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPoincareProjectKernel.IsSupportedShape(256, 64));
        Assert.False(PtxPoincareProjectKernel.IsPromotedShape(256, 64));
    }

    [SkippableTheory]
    [InlineData(256, 32)]
    [InlineData(256, 128)]
    public void DriverOnlyPoincareProject_MatchesOracle(int batch, int dim)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in poincare-project specialization is measured on GA10x/SM86.");
        const float c = 0.5f, eps = 1e-5f;
        using var kernel = new PtxPoincareProjectKernel(runtime, batch, dim, c, eps);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266500 + batch + dim);
        // Alternate small (inside ball, copied) and large (outside, projected) rows.
        float[] xHost = new float[batch * dim];
        for (int row = 0; row < batch; row++)
        {
            float mag = (row % 2 == 0) ? 0.05f : 0.4f;
            for (int i = 0; i < dim; i++) xHost[row * dim + i] = (float)((random.NextDouble() * 2 - 1) * mag);
        }
        double maxNorm = 1.0 / Math.Sqrt(c) - eps;
        double maxNormSq = maxNorm * maxNorm;
        var expected = new float[batch * dim];
        for (int row = 0; row < batch; row++)
        {
            double sq = 0;
            for (int i = 0; i < dim; i++) { double v = xHost[row * dim + i]; sq += v * v; }
            double scale = sq >= maxNormSq ? maxNorm / Math.Sqrt(sq) : 1.0;
            for (int i = 0; i < dim; i++) expected[row * dim + i] = (float)(xHost[row * dim + i] * scale);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(batch * dim * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[batch * dim];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, $"poincare project {batch}x{dim}");
    }

    [Fact]
    public void PoincareExpMapEmitter_IsThreadPerVectorTanhScaledMobius()
    {
        string ptx = PtxPoincareExpMapKernel.EmitPtx(8, 6, 64);
        Assert.Contains(PtxPoincareExpMapKernel.EntryPoint, ptx);
        Assert.Contains("ld.param.f32 %f0, [curvature];", ptx);
        Assert.Contains("NORM_LOOP:", ptx);
        Assert.Contains("WRITE_LOOP:", ptx);
        Assert.Contains("COPY_LOOP:", ptx);            // zero-tangent branch
        Assert.Contains("tanh.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "sqrt.rn.f32"));    // vNorm + sqrtC
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPoincareExpMapKernel.IsSupportedShape(256, 64));
        Assert.False(PtxPoincareExpMapKernel.IsPromotedShape(256, 64));
    }

    [SkippableTheory]
    [InlineData(256, 32)]
    [InlineData(256, 128)]
    public void DriverOnlyPoincareExpMap_MatchesOracle(int batch, int dim)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in poincare-exp-map specialization is measured on GA10x/SM86.");
        const float c = 0.5f;
        using var kernel = new PtxPoincareExpMapKernel(runtime, batch, dim, c);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20266600 + batch + dim);
        float[] xHost = Values(random, batch * dim, 0.1f);   // base points inside the ball
        float[] vHost = Values(random, batch * dim, 0.15f);  // tangent vectors
        var expected = new float[batch * dim];
        for (int row = 0; row < batch; row++)
        {
            double xn = 0, vn = 0, xv = 0;
            for (int i = 0; i < dim; i++)
            {
                double xi = xHost[row * dim + i], vi = vHost[row * dim + i];
                xn += xi * xi; vn += vi * vi; xv += xi * vi;
            }
            double vNorm = Math.Sqrt(vn);
            double sqrtC = Math.Sqrt(c);
            double cf = 1.0 - c * xn;
            double arg = sqrtC * vNorm * (cf / 2.0);
            double scale = Math.Tanh(arg) / (sqrtC * vNorm);
            double svNormSq = scale * scale * vn;
            double xsvDot = scale * xv;
            double numX = 1.0 + 2.0 * c * xsvDot + c * svNormSq;
            double numSv = 1.0 - c * xn;
            double denom = 1.0 + 2.0 * c * xsvDot + (double)c * c * xn * svNormSq;
            if (Math.Abs(denom) < 1e-10) denom = 1e-10;
            for (int i = 0; i < dim; i++)
                expected[row * dim + i] = (float)((numX * xHost[row * dim + i] + numSv * (scale * vHost[row * dim + i])) / denom);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var v = runtime.AllocateBytes((nuint)(vHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(batch * dim * sizeof(float)));
        x.Upload<float>(xHost);
        v.Upload<float>(vHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[batch * dim];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 4e-3f, $"poincare exp-map {batch}x{dim}");
    }

    [SkippableFact]
    public void Backend_DirectPtxScientific_RoutesDispatchThroughPublicMethods()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.ScientificExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.ScientificExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxScientificEnabled, "Requires a validated Ampere CUDA backend.");
            var random = RandomHelper.CreateSeededRandom(20266800);

            // ComplexMultiply through the public backend method.
            const int pairs = 16384;
            float[] aHost = Values(random, pairs * 2, 1.0f);
            float[] bHost = Values(random, pairs * 2, 1.0f);
            var cmExpected = new float[pairs * 2];
            for (int i = 0; i < pairs; i++)
            {
                float ar = aHost[2 * i], ai = aHost[2 * i + 1], br = bHost[2 * i], bi = bHost[2 * i + 1];
                cmExpected[2 * i] = ar * br - ai * bi;
                cmExpected[2 * i + 1] = ar * bi + ai * br;
            }
            using (var a = backend.AllocateBuffer(aHost))
            using (var b = backend.AllocateBuffer(bHost))
            using (var outc = backend.AllocateBuffer(pairs * 2))
            {
                long before = backend.DirectPtxScientificDispatchCount;
                backend.ComplexMultiply(a, b, outc, pairs);
                backend.Synchronize();
                Assert.True(backend.DirectPtxScientificDispatchCount > before, backend.DirectPtxLastError);
                AssertVectorClose(backend.DownloadBuffer(outc), cmExpected, 2e-3f, "backend complex-multiply route");
            }

            // MobiusAdd through the public backend method.
            const int batch = 128, dim = 64;
            const float c = 0.5f;
            float[] xHost = Values(random, batch * dim, 0.2f);
            float[] yHost = Values(random, batch * dim, 0.2f);
            var maExpected = new float[batch * dim];
            for (int row = 0; row < batch; row++)
            {
                double xn = 0, yn = 0, dot = 0;
                for (int i = 0; i < dim; i++) { double xi = xHost[row * dim + i], yi = yHost[row * dim + i]; xn += xi * xi; yn += yi * yi; dot += xi * yi; }
                double denom = Math.Max(Math.Abs(1.0 + 2.0 * c * dot + (double)c * c * xn * yn), 1e-15);
                double coeff1 = 1.0 + 2.0 * c * dot + c * yn, coeff2 = 1.0 - c * xn;
                for (int i = 0; i < dim; i++) maExpected[row * dim + i] = (float)((coeff1 * xHost[row * dim + i] + coeff2 * yHost[row * dim + i]) / denom);
            }
            using (var x = backend.AllocateBuffer(xHost))
            using (var y = backend.AllocateBuffer(yHost))
            using (var outm = backend.AllocateBuffer(batch * dim))
            {
                long before = backend.DirectPtxScientificDispatchCount;
                backend.MobiusAdd(x, y, outm, batch, dim, c);
                backend.Synchronize();
                Assert.True(backend.DirectPtxScientificDispatchCount > before, backend.DirectPtxLastError);
                AssertVectorClose(backend.DownloadBuffer(outm), maExpected, 3e-3f, "backend mobius-add route");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.ScientificExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void Backend_DirectPtxScientific_AllRemainingRoutesDispatch()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.ScientificExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.ScientificExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxScientificEnabled, "Requires a validated Ampere CUDA backend.");
            var random = RandomHelper.CreateSeededRandom(20266900);
            const int count = 16384, pairs = 16384, batch = 256, dim = 64;
            const float c = 0.5f, eps = 1e-5f;

            long AssertDispatched(long before, string what)
            {
                backend.Synchronize();
                long now = backend.DirectPtxScientificDispatchCount;
                Assert.True(now > before, $"{what}: {backend.DirectPtxLastError}");
                return now;
            }
            long n = backend.DirectPtxScientificDispatchCount;

            // Complex conjugate (exact) + magnitude (exact-ish) + phase.
            float[] ci = Values(random, pairs * 2, 2.0f);
            var conjExp = new float[pairs * 2];
            for (int i = 0; i < pairs; i++) { conjExp[2 * i] = ci[2 * i]; conjExp[2 * i + 1] = -ci[2 * i + 1]; }
            using (var inb = backend.AllocateBuffer(ci)) using (var ob = backend.AllocateBuffer(pairs * 2))
            { backend.ComplexConjugate(inb, ob, pairs); n = AssertDispatched(n, "conjugate"); AssertVectorClose(backend.DownloadBuffer(ob), conjExp, 0f, "conjugate route"); }

            float[] re = Values(random, count, 2.0f), im = Values(random, count, 2.0f);
            var magExp = new float[count];
            for (int i = 0; i < count; i++) magExp[i] = (float)Math.Sqrt((double)re[i] * re[i] + (double)im[i] * im[i]);
            using (var rb = backend.AllocateBuffer(re)) using (var ib = backend.AllocateBuffer(im)) using (var mb = backend.AllocateBuffer(count))
            { backend.ComplexMagnitude(rb, ib, mb, count); n = AssertDispatched(n, "magnitude"); AssertVectorClose(backend.DownloadBuffer(mb), magExp, 2e-3f, "magnitude route"); }
            using (var rb = backend.AllocateBuffer(re)) using (var ib = backend.AllocateBuffer(im)) using (var pb = backend.AllocateBuffer(count))
            { backend.ComplexPhase(rb, ib, pb, count); n = AssertDispatched(n, "phase"); }

            // Octonion add (exact) + multiply.
            float[] oa = Values(random, count * 8, 1.5f), ob2 = Values(random, count * 8, 1.5f);
            var addExp = new float[count * 8];
            for (int i = 0; i < addExp.Length; i++) addExp[i] = oa[i] + ob2[i];
            using (var a = backend.AllocateBuffer(oa)) using (var b = backend.AllocateBuffer(ob2)) using (var o = backend.AllocateBuffer(count * 8))
            { backend.OctonionAdd(a, b, o, count); n = AssertDispatched(n, "octonion-add"); AssertVectorClose(backend.DownloadBuffer(o), addExp, 0f, "octonion-add route"); }
            using (var a = backend.AllocateBuffer(oa)) using (var b = backend.AllocateBuffer(ob2)) using (var o = backend.AllocateBuffer(count * 8))
            { backend.OctonionMultiply(a, b, o, count); n = AssertDispatched(n, "octonion-multiply"); }

            // Poincare distance / project / exp-map.
            float[] px = Values(random, batch * dim, 0.15f), py = Values(random, batch * dim, 0.15f);
            using (var x = backend.AllocateBuffer(px)) using (var y = backend.AllocateBuffer(py)) using (var o = backend.AllocateBuffer(batch))
            { backend.PoincareDistance(x, y, o, batch, dim, c); n = AssertDispatched(n, "poincare-distance"); }
            using (var x = backend.AllocateBuffer(px)) using (var o = backend.AllocateBuffer(batch * dim))
            { backend.PoincareProject(x, o, batch, dim, c, eps); n = AssertDispatched(n, "poincare-project"); }
            using (var x = backend.AllocateBuffer(px)) using (var v = backend.AllocateBuffer(py)) using (var o = backend.AllocateBuffer(batch * dim))
            { backend.PoincareExpMap(x, v, o, batch, dim, c); n = AssertDispatched(n, "poincare-exp-map"); }

            // RBF forward: pairs = batch*numCenters must be a multiple of 256.
            const int rbfBatch = 256, rbfCenters = 8, rbfDim = 12;
            float[] rin = Values(random, rbfBatch * rbfDim, 1.0f), rcen = Values(random, rbfCenters * rbfDim, 1.0f);
            float[] reps = new float[rbfCenters];
            for (int i = 0; i < rbfCenters; i++) reps[i] = 0.3f + 0.5f * i / rbfCenters;
            var rbfExp = new float[rbfBatch * rbfCenters];
            for (int b = 0; b < rbfBatch; b++)
                for (int cc = 0; cc < rbfCenters; cc++)
                {
                    double distSq = 0;
                    for (int d = 0; d < rbfDim; d++) { double diff = rin[b * rbfDim + d] - rcen[cc * rbfDim + d]; distSq += diff * diff; }
                    rbfExp[b * rbfCenters + cc] = (float)Math.Exp(-reps[cc] * distSq);
                }
            using (var inb = backend.AllocateBuffer(rin)) using (var cenb = backend.AllocateBuffer(rcen))
            using (var epsb = backend.AllocateBuffer(reps)) using (var o = backend.AllocateBuffer(rbfBatch * rbfCenters))
            { backend.RbfForward(inb, cenb, epsb, o, rbfBatch, rbfCenters, rbfDim); n = AssertDispatched(n, "rbf-forward"); AssertVectorClose(backend.DownloadBuffer(o), rbfExp, 2e-3f, "rbf-forward route"); }

            // Pairwise distance (L2) and squared: pairs = M*N must be a multiple of 256.
            const int pdM = 64, pdN = 16, pdDim = 10;
            float[] pdA = Values(random, pdM * pdDim, 1.0f), pdB = Values(random, pdN * pdDim, 1.0f);
            var pdSqExp = new float[pdM * pdN];
            var pdL2Exp = new float[pdM * pdN];
            for (int i = 0; i < pdM; i++)
                for (int j = 0; j < pdN; j++)
                {
                    double distSq = 0;
                    for (int d = 0; d < pdDim; d++) { double diff = pdA[i * pdDim + d] - pdB[j * pdDim + d]; distSq += diff * diff; }
                    pdSqExp[i * pdN + j] = (float)distSq;
                    pdL2Exp[i * pdN + j] = (float)Math.Sqrt(distSq);
                }
            using (var ab = backend.AllocateBuffer(pdA)) using (var bb = backend.AllocateBuffer(pdB)) using (var o = backend.AllocateBuffer(pdM * pdN))
            { backend.PairwiseDistance(ab, bb, o, pdM, pdN, pdDim); n = AssertDispatched(n, "pairwise-distance"); AssertVectorClose(backend.DownloadBuffer(o), pdL2Exp, 2e-3f, "pairwise-distance route"); }
            using (var ab = backend.AllocateBuffer(pdA)) using (var bb = backend.AllocateBuffer(pdB)) using (var o = backend.AllocateBuffer(pdM * pdN))
            { backend.PairwiseDistanceSquared(ab, bb, o, pdM, pdN, pdDim); n = AssertDispatched(n, "pairwise-distance-squared"); AssertVectorClose(backend.DownloadBuffer(o), pdSqExp, 2e-3f, "pairwise-distance-squared route"); }

            // Quantum measurement: |amplitude|^2 (unnormalized). count = batch*state = 16384.
            const int qmBatch = 128, qmState = 128;
            float[] qmRe = Values(random, qmBatch * qmState, 2.0f), qmIm = Values(random, qmBatch * qmState, 2.0f);
            var qmExp = new float[qmBatch * qmState];
            for (int i = 0; i < qmExp.Length; i++) qmExp[i] = qmRe[i] * qmRe[i] + qmIm[i] * qmIm[i];
            using (var rb = backend.AllocateBuffer(qmRe)) using (var ib = backend.AllocateBuffer(qmIm)) using (var o = backend.AllocateBuffer(qmBatch * qmState))
            { backend.QuantumMeasurement(rb, ib, o, qmBatch, qmState); n = AssertDispatched(n, "quantum-measurement"); AssertVectorClose(backend.DownloadBuffer(o), qmExp, 2e-3f, "quantum-measurement route"); }

            // Complex mat-vec: shared [dim,dim] matrix, per-batch vectors. rows = batch*dim = 1024.
            const int cmvBatch = 16, cmvDim = 64;
            float[] cmvMr = Values(random, cmvDim * cmvDim, 1.0f), cmvMi = Values(random, cmvDim * cmvDim, 1.0f);
            float[] cmvXr = Values(random, cmvBatch * cmvDim, 1.0f), cmvXi = Values(random, cmvBatch * cmvDim, 1.0f);
            var cmvExpRe = new float[cmvBatch * cmvDim];
            var cmvExpIm = new float[cmvBatch * cmvDim];
            for (int b = 0; b < cmvBatch; b++)
                for (int row = 0; row < cmvDim; row++)
                {
                    double sr = 0, si = 0;
                    for (int col = 0; col < cmvDim; col++)
                    {
                        double mRe = cmvMr[row * cmvDim + col], mIm = cmvMi[row * cmvDim + col];
                        double xRe = cmvXr[b * cmvDim + col], xIm = cmvXi[b * cmvDim + col];
                        sr += mRe * xRe - mIm * xIm; si += mRe * xIm + mIm * xRe;
                    }
                    cmvExpRe[b * cmvDim + row] = (float)sr; cmvExpIm[b * cmvDim + row] = (float)si;
                }
            using (var mrb = backend.AllocateBuffer(cmvMr)) using (var mib = backend.AllocateBuffer(cmvMi))
            using (var xrb = backend.AllocateBuffer(cmvXr)) using (var xib = backend.AllocateBuffer(cmvXi))
            using (var orb = backend.AllocateBuffer(cmvBatch * cmvDim)) using (var oib = backend.AllocateBuffer(cmvBatch * cmvDim))
            {
                backend.ComplexMatVec(mrb, mib, xrb, xib, orb, oib, cmvBatch, cmvDim);
                n = AssertDispatched(n, "complex-matvec");
                AssertVectorClose(backend.DownloadBuffer(orb), cmvExpRe, 3e-3f, "complex-matvec real route");
                AssertVectorClose(backend.DownloadBuffer(oib), cmvExpIm, 3e-3f, "complex-matvec imag route");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.ScientificExperimentOverride = previousExperiment;
        }
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
