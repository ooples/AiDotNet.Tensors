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
