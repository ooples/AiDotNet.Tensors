using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Static and opt-in driver contracts for issue #850.</summary>
public class DirectPtxComplexMultiplyTests
{
    [Fact]
    public void ComplexMultiplyEmitter_IsPointerOnlyAndRegisterResident()
    {
        string ptx = PtxFusedComplexMultiplyF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Contains("exact-shape pairs=262144 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.v2.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v2.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "neg.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void ComplexMultiplyDomain_IsClosedExactSmAndUnpromoted()
    {
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(65536));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(262144));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(1048576));
        Assert.True(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(4194304));
        Assert.False(PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxFusedComplexMultiplyF32Kernel.IsPromotedShape(262144));
        Assert.True(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexMultiply(8, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedComplexMultiplyF32Kernel.EmitPtx(8, 6, 1024));
    }

    [Fact]
    public void SpectralCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.True(DirectPtxSpectralCoverageManifest.All.Count >= 36);
        string[] names = DirectPtxSpectralCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxSpectralCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        // Name the live cells rather than counting them: this pins WHICH cells
        // are experimental, so a new cell cannot quietly take a live slot.
        Assert.Equal(
            new[]
            {
                "CudaBackend.ApplyMelFilterbank",
                "CudaBackend.ApplyWindow",
                "CudaBackend.BatchedFFT",
                "CudaBackend.BatchedFFT2D",
                "CudaBackend.BuildSpectrum",
                "CudaBackend.ComplexConjugate",
                "CudaBackend.ComplexMagnitude",
                "CudaBackend.ComplexMultiply",
                "CudaBackend.DbToPower",
                "CudaBackend.DeinterleaveComplex",
                "CudaBackend.FFT",
                "CudaBackend.FFT2D",
                "CudaBackend.IRFFT",
                "CudaBackend.InterleaveComplex",
                "CudaBackend.IstftFromSpectrum",
                "CudaBackend.MelFilterbankApply",
                "CudaBackend.PhaseVocoder",
                "CudaBackend.PowerToDb",
                "CudaBackend.RFFT",
                "CudaBackend.SplitComplexAdd",
                "CudaBackend.SplitComplexConjugate",
                "CudaBackend.SplitComplexCrossSpectral",
                "CudaBackend.SplitComplexFromPolar",
                "CudaBackend.SplitComplexMagnitude",
                "CudaBackend.SplitComplexMagnitudeSquared",
                "CudaBackend.SplitComplexMultiply",
                "CudaBackend.SplitComplexPhase",
                "CudaBackend.SplitComplexScale",
                "CudaBackend.StftMagPhase",
            },
            DirectPtxSpectralCoverageManifest.All
                .Where(cell => cell.Status == DirectPtxSpectralCoverageStatus.ExperimentalDirectPtx)
                .Select(cell => cell.Api)
                .OrderBy(api => api, StringComparer.Ordinal)
                .ToArray());
        Assert.DoesNotContain(DirectPtxSpectralCoverageManifest.All,
            cell => cell.Status == DirectPtxSpectralCoverageStatus.PromotedDirectPtx);
        Assert.Equal(
            DirectPtxSpectralCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSpectralCoverageManifest.Get("CudaBackend.ComplexMultiply").Status);
    }

    [Fact]
    public void ComplexMultiplyExperimentOverride_IsThreadLocal()
    {
        bool original = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = true;
            bool workerValue = true;
            var worker = new System.Threading.Thread(() =>
                workerValue = DirectPtxFeatureGate.ComplexMultiplyExperimentOverride);
            worker.Start();
            worker.Join();
            Assert.True(DirectPtxFeatureGate.ComplexMultiplyExperimentOverride);
            Assert.False(workerValue);
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyExperimentOverride = original;
        }
    }

    [Fact]
    public void ComplexMultiplyGateOverride_IsThreadLocalAndNullable()
    {
        bool? original = DirectPtxFeatureGate.ComplexMultiplyGateOverride;
        try
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = false;
            bool? workerValue = true;
            var worker = new System.Threading.Thread(() =>
                workerValue = DirectPtxFeatureGate.ComplexMultiplyGateOverride);
            worker.Start();
            worker.Join();
            Assert.False(DirectPtxFeatureGate.ComplexMultiplyGateOverride);
            Assert.Null(workerValue);
        }
        finally
        {
            DirectPtxFeatureGate.ComplexMultiplyGateOverride = original;
        }
    }

    [SkippableTheory]
    [InlineData(65536)]
    [InlineData(262144)]
    [InlineData(1048576)]
    [InlineData(4194304)]
    public void DriverOnlyComplexMultiply_MatchesDoubleOracleAndRejectsLocalBytes(int numPairs)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxFusedComplexMultiplyF32Kernel(runtime, numPairs);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);

        var left = new float[numPairs * 2];
        var right = new float[numPairs * 2];
        // Seeded so an oracle mismatch on the admitted SM86 machine reproduces exactly.
        // NextDouble (not NextSingle) keeps the harness compiling on net471.
        var random = RandomHelper.CreateSeededRandom(20260722 + numPairs);
        for (int i = 0; i < left.Length; i++)
        {
            left[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            right[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var leftBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var rightBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        leftBuffer.Upload<float>(left);
        rightBuffer.Upload<float>(right);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(leftBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(rightBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[left.Length];
        outputBuffer.Download<float>(actual);
        for (int pair = 0; pair < numPairs; pair++)
        {
            int offset = pair * 2;
            double ar = left[offset], ai = left[offset + 1];
            double br = right[offset], bi = right[offset + 1];
            float expectedReal = (float)(ar * br - ai * bi);
            float expectedImaginary = (float)(ar * bi + ai * br);
            Assert.True(MathF.Abs(actual[offset] - expectedReal) <= 3e-6f);
            Assert.True(MathF.Abs(actual[offset + 1] - expectedImaginary) <= 3e-6f);
        }
    }

    [Fact]
    public void SplitComplexMagnitudeEmitter_IsPointerOnlyAndRegisterResident()
    {
        string ptx = PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.Magnitude, 262144);
        Assert.Contains("exact-shape count=262144 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // re, im from the two split buffers
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));          // re*re, im*im
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "sqrt.rn.f32"));
        Assert.Equal(0, Count(ptx, "fma.rn.f32"));          // deliberately unfused to match sqrtf rounding
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexMagnitudeSquaredEmitter_OmitsSqrt()
    {
        string ptx = PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.MagnitudeSquared, 65536);
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(0, Count(ptx, "sqrt.rn.f32"));          // power sum, no root
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexUnaryDomain_IsClosedExactSmAndUnpromoted()
    {
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(65536));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(262144));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(1048576));
        Assert.True(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(4194304));
        Assert.False(PtxSplitComplexUnaryF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxSplitComplexUnaryF32Kernel.IsPromotedShape(262144));
        Assert.True(DirectPtxArchitecture.HasValidatedComplexUnary(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(8, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexUnaryOp.Magnitude, 1024));
    }

    [SkippableTheory]
    [InlineData(true, 65536)]
    [InlineData(false, 65536)]
    [InlineData(true, 262144)]
    [InlineData(false, 1048576)]
    public void DriverOnlySplitComplexUnary_MatchesDoubleOracle(bool magnitude, int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        var op = magnitude ? DirectPtxSplitComplexUnaryOp.Magnitude : DirectPtxSplitComplexUnaryOp.MagnitudeSquared;
        using var kernel = new PtxSplitComplexUnaryF32Kernel(runtime, op, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);

        var re = new float[count];
        var im = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260850 + count + (magnitude ? 1 : 0));
        for (int i = 0; i < count; i++)
        {
            re[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            im[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var reBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var imBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        reBuffer.Upload<float>(re);
        imBuffer.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count];
        outputBuffer.Download<float>(actual);
        for (int i = 0; i < count; i++)
        {
            double r = re[i], m = im[i];
            float expected = magnitude ? (float)Math.Sqrt(r * r + m * m) : (float)(r * r + m * m);
            Assert.True(MathF.Abs(actual[i] - expected) <= 3e-6f);
        }
    }

    [Fact]
    public void SplitComplexMultiplyEmitter_MatchesInterleavedContraction()
    {
        string ptx = PtxSplitComplexBinaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexBinaryOp.Multiply, 262144);
        Assert.Contains("exact-shape count=262144 block=256", ptx);
        Assert.Equal(6, Count(ptx, "ld.param.u64"));
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));   // ar, ai, br, bi
        Assert.Equal(2, Count(ptx, "st.global.f32"));       // outReal, outImag
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "neg.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexCrossSpectralEmitter_IsConjugateProduct()
    {
        string ptx = PtxSplitComplexBinaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexBinaryOp.CrossSpectral, 262144);
        Assert.Equal(6, Count(ptx, "ld.param.u64"));
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "neg.f32"));           // sign on the imaginary lane (a*conj(b))
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [SkippableTheory]
    [InlineData(65536)]
    [InlineData(1048576)]
    public void DriverOnlySplitComplexCrossSpectral_MatchesDoubleOracle(int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxSplitComplexBinaryF32Kernel(runtime, DirectPtxSplitComplexBinaryOp.CrossSpectral, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        var xr = new float[count]; var xi = new float[count];
        var yr = new float[count]; var yi = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260890 + count);
        for (int i = 0; i < count; i++)
        {
            xr[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            xi[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            yr[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            yi[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var xrB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var xiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var yrB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var yiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var orB = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var oiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        xrB.Upload<float>(xr); xiB.Upload<float>(xi); yrB.Upload<float>(yr); yiB.Upload<float>(yi);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(xrB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(xiB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(yrB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(yiB, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(orB, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(oiB, kernel.Blueprint.Tensors[5]));
        runtime.Synchronize();
        var outR = new float[count]; var outI = new float[count];
        orB.Download<float>(outR); oiB.Download<float>(outI);
        for (int i = 0; i < count; i++)
        {
            double a = xr[i], b = xi[i], c = yr[i], d = yi[i];
            float er = (float)(a * c + b * d);
            float ei = (float)(b * c - a * d);
            Assert.True(MathF.Abs(outR[i] - er) <= 3e-6f);
            Assert.True(MathF.Abs(outI[i] - ei) <= 3e-6f);
        }
    }

    [Fact]
    public void SplitComplexAddEmitter_IsTwoAddLanes()
    {
        string ptx = PtxSplitComplexBinaryF32Kernel.EmitPtx(8, 6, DirectPtxSplitComplexBinaryOp.Add, 65536);
        Assert.Equal(2, Count(ptx, "add.rn.f32"));
        Assert.Equal(0, Count(ptx, "mul.rn.f32"));
        Assert.Equal(0, Count(ptx, "fma.rn.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SplitComplexConjugateEmitter_CopiesRealAndFlipsImag()
    {
        string ptx = PtxSplitComplexConjugateF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // re, im
        Assert.Equal(1, Count(ptx, "neg.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));       // outReal=re, outImag=-im
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.True(PtxSplitComplexConjugateF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxSplitComplexConjugateF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxSplitComplexConjugateF32Kernel.IsPromotedShape(262144));
    }

    [SkippableTheory]
    [InlineData(true, 65536)]
    [InlineData(false, 65536)]
    [InlineData(true, 1048576)]
    public void DriverOnlySplitComplexBinary_MatchesDoubleOracle(bool multiply, int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        var op = multiply ? DirectPtxSplitComplexBinaryOp.Multiply : DirectPtxSplitComplexBinaryOp.Add;
        using var kernel = new PtxSplitComplexBinaryF32Kernel(runtime, op, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var ar = new float[count]; var ai = new float[count];
        var br = new float[count]; var bi = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260860 + count + (multiply ? 1 : 0));
        for (int i = 0; i < count; i++)
        {
            ar[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            ai[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            br[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            bi[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var arB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var aiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var brB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var biB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var orB = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var oiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        arB.Upload<float>(ar); aiB.Upload<float>(ai); brB.Upload<float>(br); biB.Upload<float>(bi);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(arB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(aiB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(brB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(biB, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(orB, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(oiB, kernel.Blueprint.Tensors[5]));
        runtime.Synchronize();
        var outR = new float[count]; var outI = new float[count];
        orB.Download<float>(outR); oiB.Download<float>(outI);
        for (int i = 0; i < count; i++)
        {
            double a = ar[i], b = ai[i], c = br[i], d = bi[i];
            float er = multiply ? (float)(a * c - b * d) : (float)(a + c);
            float ei = multiply ? (float)(a * d + b * c) : (float)(b + d);
            Assert.True(MathF.Abs(outR[i] - er) <= 3e-6f);
            Assert.True(MathF.Abs(outI[i] - ei) <= 3e-6f);
        }
    }

    [SkippableTheory]
    [InlineData(65536)]
    [InlineData(1048576)]
    public void DriverOnlySplitComplexConjugate_MatchesOracle(int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxSplitComplexConjugateF32Kernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var re = new float[count]; var im = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260870 + count);
        for (int i = 0; i < count; i++)
        {
            re[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            im[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        using var reB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var orB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var oiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        reB.Upload<float>(re); imB.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(orB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(oiB, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var outR = new float[count]; var outI = new float[count];
        orB.Download<float>(outR); oiB.Download<float>(outI);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(re[i], outR[i]);       // real lane copied bit-exact
            Assert.Equal(-im[i], outI[i]);      // imag sign-flipped bit-exact
        }
    }

    [Fact]
    public void InterleaveComplexEmitter_IsScalarLoadsThenV2Store()
    {
        string ptx = PtxComplexInterleaveF32Kernel.EmitPtx(8, 6, DirectPtxComplexInterleaveDirection.Interleave, 262144);
        Assert.Contains("exact-shape count=262144 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // real[i], imag[i]
        Assert.Equal(1, Count(ptx, "st.global.v2.f32"));    // interleaved pair
        Assert.Equal(0, Count(ptx, "ld.global.nc.v2.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void DeinterleaveComplexEmitter_IsV2LoadThenScalarStores()
    {
        string ptx = PtxComplexInterleaveF32Kernel.EmitPtx(8, 6, DirectPtxComplexInterleaveDirection.Deinterleave, 65536);
        Assert.Equal(1, Count(ptx, "ld.global.nc.v2.f32"));  // interleaved pair
        Assert.Equal(2, Count(ptx, "st.global.f32"));         // real[i], imag[i]
        Assert.Equal(0, Count(ptx, "st.global.v2.f32"));
        Assert.True(PtxComplexInterleaveF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxComplexInterleaveF32Kernel.IsSupportedShape(1024));
        Assert.False(PtxComplexInterleaveF32Kernel.IsPromotedShape(262144));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
    }

    [SkippableTheory]
    [InlineData(true, 65536)]
    [InlineData(false, 65536)]
    [InlineData(true, 1048576)]
    public void DriverOnlyComplexInterleave_MatchesOracle(bool interleave, int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        var direction = interleave
            ? DirectPtxComplexInterleaveDirection.Interleave
            : DirectPtxComplexInterleaveDirection.Deinterleave;
        using var kernel = new PtxComplexInterleaveF32Kernel(runtime, direction, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20260880 + count + (interleave ? 1 : 0));
        using var b0 = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var b1 = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var b2 = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        if (interleave)
        {
            var real = new float[count]; var imag = new float[count];
            for (int i = 0; i < count; i++) { real[i] = (float)(random.NextDouble() * 2.0 - 1.0); imag[i] = (float)(random.NextDouble() * 2.0 - 1.0); }
            b0.Upload<float>(real); b1.Upload<float>(imag);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(b0, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(b1, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(b2, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[count * 2];
            b2.Download<float>(actual);
            for (int i = 0; i < count; i++) { Assert.Equal(real[i], actual[2 * i]); Assert.Equal(imag[i], actual[2 * i + 1]); }
        }
        else
        {
            var inter = new float[count * 2];
            for (int i = 0; i < inter.Length; i++) inter[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            b0.Upload<float>(inter);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(b0, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(b1, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(b2, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var real = new float[count]; var imag = new float[count];
            b1.Download<float>(real); b2.Download<float>(imag);
            for (int i = 0; i < count; i++) { Assert.Equal(inter[2 * i], real[i]); Assert.Equal(inter[2 * i + 1], imag[i]); }
        }
    }

    [Fact]
    public void SplitComplexScaleEmitter_HasScalarParamAndTwoMuls()
    {
        string ptx = PtxSplitComplexScaleF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.param.f32"));    // the scalar operand
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));       // re*scalar, im*scalar
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.True(PtxSplitComplexScaleF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxSplitComplexScaleF32Kernel.IsPromotedShape(262144));
    }

    [SkippableTheory]
    [InlineData(65536)]
    [InlineData(1048576)]
    public void DriverOnlySplitComplexScale_MatchesOracle(int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxSplitComplexScaleF32Kernel(runtime, count);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        const float scalar = 1.7501f;
        var re = new float[count]; var im = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260900 + count);
        for (int i = 0; i < count; i++) { re[i] = (float)(random.NextDouble() * 2.0 - 1.0); im[i] = (float)(random.NextDouble() * 2.0 - 1.0); }
        using var reB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var orB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var oiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        reB.Upload<float>(re); imB.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(orB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(oiB, kernel.Blueprint.Tensors[3]), scalar);
        runtime.Synchronize();
        var outR = new float[count]; var outI = new float[count];
        orB.Download<float>(outR); oiB.Download<float>(outI);
        for (int i = 0; i < count; i++) { Assert.Equal(re[i] * scalar, outR[i]); Assert.Equal(im[i] * scalar, outI[i]); }
    }

    [Fact]
    public void SplitComplexPhaseEmitter_IsMinimaxAtan2()
    {
        string ptx = PtxSplitComplexPhaseF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // re, im
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(2, Count(ptx, "abs.f32"));
        Assert.Equal(4, Count(ptx, "selp.f32"));            // quadrant + degenerate folding
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
    }

    [SkippableTheory]
    [InlineData(65536)]
    public void DriverOnlySplitComplexPhase_MatchesAtan2WithinTolerance(int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxSplitComplexPhaseF32Kernel(runtime, count);
        var re = new float[count]; var im = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260910 + count);
        for (int i = 0; i < count; i++) { re[i] = (float)(random.NextDouble() * 2.0 - 1.0); im[i] = (float)(random.NextDouble() * 2.0 - 1.0); }
        using var reB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var oB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        reB.Upload<float>(re); imB.Upload<float>(im);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(reB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(oB, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[count]; oB.Download<float>(actual);
        for (int i = 0; i < count; i++) Assert.True(MathF.Abs(actual[i] - (float)Math.Atan2(im[i], re[i])) <= 2e-4f);
    }

    [Fact]
    public void SplitComplexFromPolarEmitter_UsesCosSinApprox()
    {
        string ptx = PtxSplitComplexFromPolarF32Kernel.EmitPtx(8, 6, 262144);
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "cos.approx.f32"));
        Assert.Equal(1, Count(ptx, "sin.approx.f32"));
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));           // m*cos, m*sin
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.True(PtxSplitComplexFromPolarF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxSplitComplexFromPolarF32Kernel.IsPromotedShape(262144));
    }

    [SkippableTheory]
    [InlineData(65536)]
    public void DriverOnlySplitComplexFromPolar_MatchesWithinTolerance(int count)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        using var kernel = new PtxSplitComplexFromPolarF32Kernel(runtime, count);
        var mag = new float[count]; var phase = new float[count];
        var random = RandomHelper.CreateSeededRandom(20260920 + count);
        for (int i = 0; i < count; i++) { mag[i] = (float)(random.NextDouble() * 2.0); phase[i] = (float)((random.NextDouble() * 2.0 - 1.0) * Math.PI); }
        using var mB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var pB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var orB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var oiB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        mB.Upload<float>(mag); pB.Upload<float>(phase);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(mB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(pB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(orB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(oiB, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var outR = new float[count]; var outI = new float[count];
        orB.Download<float>(outR); oiB.Download<float>(outI);
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(outR[i] - (float)(mag[i] * Math.Cos(phase[i]))) <= 2e-4f);
            Assert.True(MathF.Abs(outI[i] - (float)(mag[i] * Math.Sin(phase[i]))) <= 2e-4f);
        }
    }

    [Fact]
    public void ApplyMelFilterbankEmitter_IsThreadPerCellFmaReduction()
    {
        string ptx = PtxApplyMelFilterbankF32Kernel.EmitPtx(8, 6, 32, 64, 8);   // frames*mels = 256
        Assert.Contains("exact-shape frames=32 freqs=64 mels=8 block=256", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));   // power, filter in the loop
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Contains("$MEL_F_LOOP:", ptx);
        Assert.Contains("div.u32 %r3, %r2, 8", ptx);        // frame = gid / nMels
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxApplyMelFilterbankF32Kernel.IsSupportedShape(32, 64, 8));
        Assert.False(PtxApplyMelFilterbankF32Kernel.IsSupportedShape(32, 64, 7));   // 224 not mult of 256
        Assert.False(PtxApplyMelFilterbankF32Kernel.IsPromotedShape(32, 64, 8));
    }

    [SkippableFact]
    public void DriverOnlyApplyMelFilterbank_MatchesDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int frames = 32, freqs = 64, mels = 8;
        using var kernel = new PtxApplyMelFilterbankF32Kernel(runtime, frames, freqs, mels);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        var power = new float[frames * freqs];
        var filter = new float[mels * freqs];
        var random = RandomHelper.CreateSeededRandom(20260930);
        for (int i = 0; i < power.Length; i++) power[i] = (float)random.NextDouble();
        for (int i = 0; i < filter.Length; i++) filter[i] = (float)random.NextDouble();
        var expected = new float[frames * mels];
        for (int fr = 0; fr < frames; fr++)
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                for (int f = 0; f < freqs; f++) sum += (double)power[fr * freqs + f] * filter[m * freqs + f];
                expected[fr * mels + m] = (float)sum;
            }
        using var pB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var fB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var mB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        pB.Upload<float>(power); fB.Upload<float>(filter);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(pB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(fB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(mB, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[frames * mels];
        mB.Download<float>(actual);
        for (int i = 0; i < actual.Length; i++) Assert.True(MathF.Abs(actual[i] - expected[i]) <= 2e-4f);
    }

    [Fact]
    public void FftBitReverseEmitter_IsGuardedBrevSwap()
    {
        string ptx = PtxBitReversePermutationF32Kernel.EmitPtx(8, 6, 512);   // log2n=9, shift=23
        Assert.Contains("exact-shape n=512 log2n=9 block=256 op=fft-bit-reverse", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Contains("brev.b32 %r3, %r2", ptx);
        Assert.Contains("shr.u32 %r3, %r3, 23", ptx);          // 32 - log2(512)
        Assert.Contains("setp.le.u32 %p0, %r3, %r2", ptx);     // skip when reversed <= idx
        Assert.Contains("@%p0 bra $BR_END", ptx);
        Assert.Equal(4, Count(ptx, "ld.global.f32"));          // real/imag at idx and rev
        Assert.Equal(4, Count(ptx, "st.global.f32"));          // swapped back
        Assert.DoesNotContain("fma", ptx, StringComparison.Ordinal);   // pure data movement
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBitReversePermutationF32Kernel.IsSupportedShape(512));
        Assert.False(PtxBitReversePermutationF32Kernel.IsSupportedShape(384));   // not power of two
        Assert.False(PtxBitReversePermutationF32Kernel.IsPromotedShape(512));
    }

    [Fact]
    public void FftButterflyEmitter_IsThreadPerWingTwiddle()
    {
        string ptx = PtxFftButterflyF32Kernel.EmitPtx(8, 6, 512);
        Assert.Contains("exact-shape n=512 block=256 op=fft-butterfly", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.param.u32"));           // stride and inverse controls
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));             // complex twiddle contraction
        Assert.Equal(4, Count(ptx, "ld.global.f32"));          // top/bot real/imag
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);   // exactly n/2 wings, no guard
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxFftButterflyF32Kernel.IsSupportedShape(512));
        Assert.False(PtxFftButterflyF32Kernel.IsSupportedShape(256));   // (n/2) not a multiple of 256
        Assert.False(PtxFftButterflyF32Kernel.IsPromotedShape(512));
    }

    [SkippableFact]
    public void DriverOnlyFft_MatchesDoubleDftOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 512;
        using var reverse = new PtxBitReversePermutationF32Kernel(runtime, n);
        using var butterfly = new PtxFftButterflyF32Kernel(runtime, n);
        Assert.Equal(0, reverse.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, butterfly.Audit.Function.LocalBytesPerThread);

        var real = new float[n];
        var imag = new float[n];
        var random = RandomHelper.CreateSeededRandom(20261115);
        for (int i = 0; i < n; i++) { real[i] = (float)(random.NextDouble() * 2 - 1); imag[i] = 0f; }

        // Forward DFT oracle in double precision.
        var expectedRe = new double[n];
        var expectedIm = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < n; t++)
            {
                double angle = -2.0 * Math.PI * k * t / n;
                sr += real[t] * Math.Cos(angle) - imag[t] * Math.Sin(angle);
                si += real[t] * Math.Sin(angle) + imag[t] * Math.Cos(angle);
            }
            expectedRe[k] = sr; expectedIm[k] = si;
        }

        using var reB = runtime.AllocateBytes(reverse.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(reverse.Blueprint.Tensors[1].RequiredBytes);
        reB.Upload<float>(real); imB.Upload<float>(imag);
        var reView = DirectPtxTensorView.CreateOwned(reB, reverse.Blueprint.Tensors[0]);
        var imView = DirectPtxTensorView.CreateOwned(imB, reverse.Blueprint.Tensors[1]);
        var reButterflyView = DirectPtxTensorView.CreateOwned(reB, butterfly.Blueprint.Tensors[0]);
        var imButterflyView = DirectPtxTensorView.CreateOwned(imB, butterfly.Blueprint.Tensors[1]);

        reverse.Launch(reView, imView);
        for (int stride = 2; stride <= n; stride <<= 1)
            butterfly.Launch(reButterflyView, imButterflyView, stride, 0);
        runtime.Synchronize();

        var actualRe = new float[n];
        var actualIm = new float[n];
        reB.Download<float>(actualRe); imB.Download<float>(actualIm);
        for (int k = 0; k < n; k++)
        {
            Assert.True(Math.Abs(actualRe[k] - expectedRe[k]) <= 1e-2 * (1 + Math.Abs(expectedRe[k])));
            Assert.True(Math.Abs(actualIm[k] - expectedIm[k]) <= 1e-2 * (1 + Math.Abs(expectedIm[k])));
        }
    }

    [Fact]
    public void RfftPostprocessEmitter_IsGuardedCopy()
    {
        string ptx = PtxRfftPostprocessF32Kernel.EmitPtx(8, 6, 512);   // outLen = 257
        Assert.Contains("exact-shape n=512 outlen=257 block=256 op=rfft-postprocess", ptx);
        Assert.Equal(4, Count(ptx, "ld.param.u64"));
        Assert.Contains("setp.ge.u32 %p0, %r2, 257", ptx);
        Assert.Contains("@%p0 bra $RFFT_END", ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("fma", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxRfftPostprocessF32Kernel.IsSupportedShape(512));
        Assert.False(PtxRfftPostprocessF32Kernel.IsSupportedShape(768));   // not power of two
        Assert.False(PtxRfftPostprocessF32Kernel.IsPromotedShape(512));
    }

    [Fact]
    public void IrfftPreprocessEmitter_IsHermitianExpand()
    {
        string ptx = PtxIrfftPreprocessF32Kernel.EmitPtx(8, 6, 512);   // inLen = 257
        Assert.Contains("exact-shape n=512 inlen=257 block=256 op=irfft-preprocess", ptx);
        Assert.Contains("setp.ge.u32 %p0, %r2, 257", ptx);
        Assert.Contains("@%p0 bra $IRFFT_MIRROR", ptx);
        Assert.Contains("sub.u32 %r3, 512, %r2", ptx);      // mirrorIdx = n - idx
        Assert.Equal(1, Count(ptx, "neg.f32"));             // conjugate the mirror lane only
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));    // copy path + mirror path
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxIrfftPreprocessF32Kernel.IsSupportedShape(512));
        Assert.False(PtxIrfftPreprocessF32Kernel.IsSupportedShape(384));
        Assert.False(PtxIrfftPreprocessF32Kernel.IsPromotedShape(512));
    }

    [Fact]
    public void ScaleInverseEmitter_IsTwoLaneMul()
    {
        string ptx = PtxScaleInverseF32Kernel.EmitPtx(8, 6, 512);
        Assert.Contains("exact-shape count=512 block=256 op=scale-inverse", ptx);
        Assert.Contains("ld.param.f32 %f0, [scale_val]", ptx);
        Assert.Equal(2, Count(ptx, "mul.rn.f32"));
        Assert.Equal(2, Count(ptx, "ld.global.f32"));
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);   // exact launch, no guard
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxScaleInverseF32Kernel.IsSupportedShape(512));
        Assert.True(PtxScaleInverseF32Kernel.IsSupportedShape(1536));   // batch*n need not be a power of two
        Assert.False(PtxScaleInverseF32Kernel.IsSupportedShape(500));   // not a multiple of 256
        Assert.False(PtxScaleInverseF32Kernel.IsPromotedShape(512));
    }

    [SkippableFact]
    public void DriverOnlyRfft_MatchesRealDftOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 512, outLen = n / 2 + 1;
        using var reverse = new PtxBitReversePermutationF32Kernel(runtime, n);
        using var butterfly = new PtxFftButterflyF32Kernel(runtime, n);
        using var postprocess = new PtxRfftPostprocessF32Kernel(runtime, n);

        var real = new float[n];
        var imag = new float[n];
        var random = RandomHelper.CreateSeededRandom(20261201);
        for (int i = 0; i < n; i++) { real[i] = (float)(random.NextDouble() * 2 - 1); imag[i] = 0f; }

        var expectedRe = new double[outLen];
        var expectedIm = new double[outLen];
        for (int k = 0; k < outLen; k++)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < n; t++)
            {
                double angle = -2.0 * Math.PI * k * t / n;
                sr += real[t] * Math.Cos(angle);
                si += real[t] * Math.Sin(angle);
            }
            expectedRe[k] = sr; expectedIm[k] = si;
        }

        using var reB = runtime.AllocateBytes(reverse.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(reverse.Blueprint.Tensors[1].RequiredBytes);
        using var outReB = runtime.AllocateBytes(postprocess.Blueprint.Tensors[2].RequiredBytes);
        using var outImB = runtime.AllocateBytes(postprocess.Blueprint.Tensors[3].RequiredBytes);
        reB.Upload<float>(real); imB.Upload<float>(imag);

        reverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, reverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, reverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= n; stride <<= 1)
            butterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, butterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, butterfly.Blueprint.Tensors[1]), stride, 0);
        postprocess.Launch(
            DirectPtxTensorView.CreateOwned(reB, postprocess.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, postprocess.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outReB, postprocess.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(outImB, postprocess.Blueprint.Tensors[3]));
        runtime.Synchronize();

        var actualRe = new float[outLen];
        var actualIm = new float[outLen];
        outReB.Download<float>(actualRe); outImB.Download<float>(actualIm);
        for (int k = 0; k < outLen; k++)
        {
            Assert.True(Math.Abs(actualRe[k] - expectedRe[k]) <= 1e-2 * (1 + Math.Abs(expectedRe[k])));
            Assert.True(Math.Abs(actualIm[k] - expectedIm[k]) <= 1e-2 * (1 + Math.Abs(expectedIm[k])));
        }
    }

    [SkippableFact]
    public void DriverOnlyIrfft_RecoversRealSignal()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 512, inLen = n / 2 + 1;
        using var preprocess = new PtxIrfftPreprocessF32Kernel(runtime, n);
        using var reverse = new PtxBitReversePermutationF32Kernel(runtime, n);
        using var butterfly = new PtxFftButterflyF32Kernel(runtime, n);
        using var scale = new PtxScaleInverseF32Kernel(runtime, n);

        // Original real signal and its packed positive-frequency spectrum (double DFT).
        var signal = new float[n];
        var random = RandomHelper.CreateSeededRandom(20261202);
        for (int t = 0; t < n; t++) signal[t] = (float)(random.NextDouble() * 2 - 1);
        var packedRe = new float[inLen];
        var packedIm = new float[inLen];
        for (int k = 0; k < inLen; k++)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < n; t++)
            {
                double angle = -2.0 * Math.PI * k * t / n;
                sr += signal[t] * Math.Cos(angle);
                si += signal[t] * Math.Sin(angle);
            }
            packedRe[k] = (float)sr; packedIm[k] = (float)si;
        }

        using var inReB = runtime.AllocateBytes(preprocess.Blueprint.Tensors[0].RequiredBytes);
        using var inImB = runtime.AllocateBytes(preprocess.Blueprint.Tensors[1].RequiredBytes);
        using var fullReB = runtime.AllocateBytes(preprocess.Blueprint.Tensors[2].RequiredBytes);
        using var fullImB = runtime.AllocateBytes(preprocess.Blueprint.Tensors[3].RequiredBytes);
        inReB.Upload<float>(packedRe); inImB.Upload<float>(packedIm);

        preprocess.Launch(
            DirectPtxTensorView.CreateOwned(inReB, preprocess.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inImB, preprocess.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(fullReB, preprocess.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(fullImB, preprocess.Blueprint.Tensors[3]));
        reverse.Launch(
            DirectPtxTensorView.CreateOwned(fullReB, reverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(fullImB, reverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= n; stride <<= 1)
            butterfly.Launch(
                DirectPtxTensorView.CreateOwned(fullReB, butterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(fullImB, butterfly.Blueprint.Tensors[1]), stride, 1);
        scale.Launch(
            DirectPtxTensorView.CreateOwned(fullReB, scale.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(fullImB, scale.Blueprint.Tensors[1]), 1f / n);
        runtime.Synchronize();

        var recoveredRe = new float[n];
        var recoveredIm = new float[n];
        fullReB.Download<float>(recoveredRe); fullImB.Download<float>(recoveredIm);
        for (int t = 0; t < n; t++)
        {
            Assert.True(Math.Abs(recoveredRe[t] - signal[t]) <= 1e-2 * (1 + Math.Abs(signal[t])));
            Assert.True(Math.Abs(recoveredIm[t]) <= 1e-2);
        }
    }

    [Fact]
    public void BatchedBitReverseEmitter_OffsetsByGridY()
    {
        string ptx = PtxBatchedBitReverseF32Kernel.EmitPtx(8, 6, 512);
        Assert.Contains("exact-shape n=512 log2n=9 block=256 op=batched-fft-bit-reverse", ptx);
        Assert.Contains("brev.b32 %r3, %r2", ptx);
        Assert.Contains("shr.u32 %r3, %r3, 23", ptx);
        Assert.Contains("mov.u32 %r4, %ctaid.y", ptx);          // batch index from gridY
        Assert.Contains("mul.lo.u32 %r5, %r4, 512", ptx);       // baseOffset = b*n
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("fma", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBatchedBitReverseF32Kernel.IsSupportedShape(512, 4));
        Assert.False(PtxBatchedBitReverseF32Kernel.IsSupportedShape(512, 0));   // batch must be >= 1
        Assert.False(PtxBatchedBitReverseF32Kernel.IsPromotedShape(512, 4));
    }

    [Fact]
    public void BatchedFftButterflyEmitter_OffsetsByGridY()
    {
        string ptx = PtxBatchedFftButterflyF32Kernel.EmitPtx(8, 6, 512);
        Assert.Contains("exact-shape n=512 block=256 op=batched-fft-butterfly", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u32"));            // stride and inverse
        Assert.Contains("mov.u32 %r10, %ctaid.y", ptx);         // batch index from gridY
        Assert.Contains("mul.lo.u32 %r11, %r10, 512", ptx);     // baseOffset = b*n
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBatchedFftButterflyF32Kernel.IsSupportedShape(512, 4));
        Assert.False(PtxBatchedFftButterflyF32Kernel.IsSupportedShape(256, 4));   // (n/2) not multiple of 256
        Assert.False(PtxBatchedFftButterflyF32Kernel.IsPromotedShape(512, 4));
    }

    [SkippableFact]
    public void DriverOnlyBatchedFft_MatchesPerRowDftOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 512, batch = 3;
        using var reverse = new PtxBatchedBitReverseF32Kernel(runtime, n, batch);
        using var butterfly = new PtxBatchedFftButterflyF32Kernel(runtime, n, batch);
        Assert.Equal(0, reverse.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, butterfly.Audit.Function.LocalBytesPerThread);

        var real = new float[batch * n];
        var imag = new float[batch * n];
        var random = RandomHelper.CreateSeededRandom(20261210);
        for (int i = 0; i < real.Length; i++) { real[i] = (float)(random.NextDouble() * 2 - 1); imag[i] = 0f; }

        var expectedRe = new double[batch * n];
        var expectedIm = new double[batch * n];
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < n; k++)
            {
                double sr = 0, si = 0;
                for (int t = 0; t < n; t++)
                {
                    double angle = -2.0 * Math.PI * k * t / n;
                    sr += real[b * n + t] * Math.Cos(angle);
                    si += real[b * n + t] * Math.Sin(angle);
                }
                expectedRe[b * n + k] = sr; expectedIm[b * n + k] = si;
            }

        using var reB = runtime.AllocateBytes(reverse.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(reverse.Blueprint.Tensors[1].RequiredBytes);
        reB.Upload<float>(real); imB.Upload<float>(imag);
        reverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, reverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, reverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= n; stride <<= 1)
            butterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, butterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, butterfly.Blueprint.Tensors[1]), stride, 0);
        runtime.Synchronize();

        var actualRe = new float[batch * n];
        var actualIm = new float[batch * n];
        reB.Download<float>(actualRe); imB.Download<float>(actualIm);
        for (int i = 0; i < actualRe.Length; i++)
        {
            Assert.True(Math.Abs(actualRe[i] - expectedRe[i]) <= 1e-2 * (1 + Math.Abs(expectedRe[i])));
            Assert.True(Math.Abs(actualIm[i] - expectedIm[i]) <= 1e-2 * (1 + Math.Abs(expectedIm[i])));
        }
    }

    [Fact]
    public void FftColsBitReverseEmitter_IsStridedGuardedSwap()
    {
        string ptx = PtxFftColsBitReverseF32Kernel.EmitPtx(8, 6, 4, 512);   // log2height=2, shift=30
        Assert.Contains("exact-shape height=4 width=512 log2h=2 block=256 op=fft-cols-bit-reverse", ptx);
        Assert.Contains("rem.u32 %r3, %r2, 512", ptx);         // col = gid % width
        Assert.Contains("div.u32 %r4, %r2, 512", ptx);         // row = gid / width
        Assert.Contains("brev.b32 %r5, %r4", ptx);
        Assert.Contains("shr.u32 %r5, %r5, 30", ptx);          // 32 - log2(4)
        Assert.Contains("mad.lo.u32 %r6, %r4, 512, %r3", ptx); // row*width + col
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("fma", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxFftColsBitReverseF32Kernel.IsSupportedShape(4, 512));
        Assert.False(PtxFftColsBitReverseF32Kernel.IsSupportedShape(3, 512));   // height not power of two
        Assert.False(PtxFftColsBitReverseF32Kernel.IsPromotedShape(4, 512));
    }

    [Fact]
    public void FftColsButterflyEmitter_IsStridedWingTwiddle()
    {
        string ptx = PtxFftColsButterflyF32Kernel.EmitPtx(8, 6, 4, 512);
        Assert.Contains("exact-shape height=4 width=512 block=256 op=fft-cols-butterfly", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u32"));
        Assert.Contains("rem.u32 %r3, %r2, 512", ptx);            // col
        Assert.Contains("mad.lo.u32 %r11, %r10, 512, %r3", ptx);  // topIdx = topRow*width + col
        Assert.Contains("mad.lo.u32 %r13, %r5, 512, %r11", ptx);  // botIdx = topIdx + halfStride*width
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxFftColsButterflyF32Kernel.IsSupportedShape(4, 512));
        Assert.False(PtxFftColsButterflyF32Kernel.IsSupportedShape(4, 500));   // width not power of two
        Assert.False(PtxFftColsButterflyF32Kernel.IsPromotedShape(4, 512));
    }

    [SkippableFact]
    public void DriverOnlyFft2D_MatchesDouble2DDftOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int height = 4, width = 512, total = height * width;
        using var rowReverse = new PtxBatchedBitReverseF32Kernel(runtime, width, height);
        using var rowButterfly = new PtxBatchedFftButterflyF32Kernel(runtime, width, height);
        using var colReverse = new PtxFftColsBitReverseF32Kernel(runtime, height, width);
        using var colButterfly = new PtxFftColsButterflyF32Kernel(runtime, height, width);

        var real = new float[total];
        var imag = new float[total];
        var random = RandomHelper.CreateSeededRandom(20261220);
        for (int i = 0; i < total; i++) { real[i] = (float)(random.NextDouble() * 2 - 1); imag[i] = 0f; }

        // Full 2D forward DFT oracle in double precision.
        var expectedRe = new double[total];
        var expectedIm = new double[total];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
            {
                double sr = 0, si = 0;
                for (int t = 0; t < height; t++)
                    for (int s = 0; s < width; s++)
                    {
                        double angle = -2.0 * Math.PI * ((double)r * t / height + (double)c * s / width);
                        double xr = real[t * width + s];
                        sr += xr * Math.Cos(angle);
                        si += xr * Math.Sin(angle);
                    }
                expectedRe[r * width + c] = sr; expectedIm[r * width + c] = si;
            }

        using var reB = runtime.AllocateBytes(rowReverse.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(rowReverse.Blueprint.Tensors[1].RequiredBytes);
        reB.Upload<float>(real); imB.Upload<float>(imag);

        // Row pass: batched FFT over height rows of length width.
        rowReverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, rowReverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, rowReverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= width; stride <<= 1)
            rowButterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, rowButterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, rowButterfly.Blueprint.Tensors[1]), stride, 0);
        // Column pass: strided FFT over width columns of length height.
        colReverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, colReverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, colReverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= height; stride <<= 1)
            colButterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, colButterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, colButterfly.Blueprint.Tensors[1]), stride, 0);
        runtime.Synchronize();

        var actualRe = new float[total];
        var actualIm = new float[total];
        reB.Download<float>(actualRe); imB.Download<float>(actualIm);
        for (int i = 0; i < total; i++)
        {
            Assert.True(Math.Abs(actualRe[i] - expectedRe[i]) <= 2e-2 * (1 + Math.Abs(expectedRe[i])));
            Assert.True(Math.Abs(actualIm[i] - expectedIm[i]) <= 2e-2 * (1 + Math.Abs(expectedIm[i])));
        }
    }

    [Fact]
    public void BatchedFftColsBitReverseEmitter_OffsetsByImage()
    {
        string ptx = PtxBatchedFftColsBitReverseF32Kernel.EmitPtx(8, 6, 4, 512);
        Assert.Contains("exact-shape height=4 width=512 log2h=2 block=256 op=batched-fft-cols-bit-reverse", ptx);
        Assert.Contains("mov.u32 %r8, %ctaid.y", ptx);            // image index
        Assert.Contains("mul.lo.u32 %r9, %r8, 2048", ptx);        // imgOffset = img*height*width
        Assert.Contains("brev.b32 %r5, %r4", ptx);
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("fma", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBatchedFftColsBitReverseF32Kernel.IsSupportedShape(4, 512, 2));
        Assert.False(PtxBatchedFftColsBitReverseF32Kernel.IsSupportedShape(4, 512, 0));   // images >= 1
        Assert.False(PtxBatchedFftColsBitReverseF32Kernel.IsPromotedShape(4, 512, 2));
    }

    [Fact]
    public void BatchedFftColsButterflyEmitter_OffsetsByImage()
    {
        string ptx = PtxBatchedFftColsButterflyF32Kernel.EmitPtx(8, 6, 4, 512);
        Assert.Contains("exact-shape height=4 width=512 block=256 op=batched-fft-cols-butterfly", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u32"));
        Assert.Contains("mov.u32 %r14, %ctaid.y", ptx);           // image index
        Assert.Contains("mul.lo.u32 %r15, %r14, 2048", ptx);      // imgOffset
        Assert.Contains("mad.lo.u32 %r13, %r5, 512, %r11", ptx);  // botIdx = topIdx + halfStride*width
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "fma.rn.f32"));
        Assert.Equal(4, Count(ptx, "ld.global.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBatchedFftColsButterflyF32Kernel.IsSupportedShape(4, 512, 2));
        Assert.False(PtxBatchedFftColsButterflyF32Kernel.IsSupportedShape(4, 500, 2));   // width not power of two
        Assert.False(PtxBatchedFftColsButterflyF32Kernel.IsPromotedShape(4, 512, 2));
    }

    [SkippableFact]
    public void DriverOnlyBatchedFft2D_MatchesPerImage2DDftOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexMultiply(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int images = 2, height = 4, width = 512, imageSize = height * width, total = images * imageSize;
        using var rowReverse = new PtxBatchedBitReverseF32Kernel(runtime, width, images * height);
        using var rowButterfly = new PtxBatchedFftButterflyF32Kernel(runtime, width, images * height);
        using var colReverse = new PtxBatchedFftColsBitReverseF32Kernel(runtime, height, width, images);
        using var colButterfly = new PtxBatchedFftColsButterflyF32Kernel(runtime, height, width, images);

        var real = new float[total];
        var imag = new float[total];
        var random = RandomHelper.CreateSeededRandom(20261225);
        for (int i = 0; i < total; i++) { real[i] = (float)(random.NextDouble() * 2 - 1); imag[i] = 0f; }

        // Independent 2D forward DFT per image.
        var expectedRe = new double[total];
        var expectedIm = new double[total];
        for (int img = 0; img < images; img++)
        {
            int b = img * imageSize;
            for (int r = 0; r < height; r++)
                for (int c = 0; c < width; c++)
                {
                    double sr = 0, si = 0;
                    for (int t = 0; t < height; t++)
                        for (int s = 0; s < width; s++)
                        {
                            double angle = -2.0 * Math.PI * ((double)r * t / height + (double)c * s / width);
                            double xr = real[b + t * width + s];
                            sr += xr * Math.Cos(angle);
                            si += xr * Math.Sin(angle);
                        }
                    expectedRe[b + r * width + c] = sr; expectedIm[b + r * width + c] = si;
                }
        }

        using var reB = runtime.AllocateBytes(rowReverse.Blueprint.Tensors[0].RequiredBytes);
        using var imB = runtime.AllocateBytes(rowReverse.Blueprint.Tensors[1].RequiredBytes);
        reB.Upload<float>(real); imB.Upload<float>(imag);

        rowReverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, rowReverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, rowReverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= width; stride <<= 1)
            rowButterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, rowButterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, rowButterfly.Blueprint.Tensors[1]), stride, 0);
        colReverse.Launch(
            DirectPtxTensorView.CreateOwned(reB, colReverse.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(imB, colReverse.Blueprint.Tensors[1]));
        for (int stride = 2; stride <= height; stride <<= 1)
            colButterfly.Launch(
                DirectPtxTensorView.CreateOwned(reB, colButterfly.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(imB, colButterfly.Blueprint.Tensors[1]), stride, 0);
        runtime.Synchronize();

        var actualRe = new float[total];
        var actualIm = new float[total];
        reB.Download<float>(actualRe); imB.Download<float>(actualIm);
        for (int i = 0; i < total; i++)
        {
            Assert.True(Math.Abs(actualRe[i] - expectedRe[i]) <= 2e-2 * (1 + Math.Abs(expectedRe[i])));
            Assert.True(Math.Abs(actualIm[i] - expectedIm[i]) <= 2e-2 * (1 + Math.Abs(expectedIm[i])));
        }
    }

    [Fact]
    public void OverlapAddEmitter_IsFrameLoopFma()
    {
        string ptx = PtxOverlapAddF32Kernel.EmitPtx(8, 6, 8, 512, 128, 1408);   // (8-1)*128 + 512 = 1408
        Assert.Contains("exact-shape numFrames=8 nFft=512 hop=128 outLen=1408 block=256 op=overlap-add", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Contains("setp.ge.u32 %p0, %r2, 1408", ptx);        // output guard
        Assert.Contains("$OLA_LOOP:", ptx);
        Assert.Contains("setp.ge.u32 %p1, %r3, 8", ptx);           // frame loop bound
        Assert.Contains("mul.lo.u32 %r4, %r3, 128", ptx);          // frameStart = frame*hop
        Assert.Contains("setp.ge.s32 %p3, %r5, 512", ptx);         // localIdx < nFft
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxOverlapAddF32Kernel.IsSupportedShape(8, 512, 128, 1408));
        Assert.False(PtxOverlapAddF32Kernel.IsSupportedShape(8, 512, 600, 1408));   // hop > nFft
        Assert.False(PtxOverlapAddF32Kernel.IsPromotedShape(8, 512, 128, 1408));
    }

    [Fact]
    public void WindowSumSquaresEmitter_IsFrameLoopSquare()
    {
        string ptx = PtxWindowSumSquaresF32Kernel.EmitPtx(8, 6, 512, 128, 1408);   // numFrames = (1408-512)/128+1 = 8
        Assert.Contains("exact-shape nFft=512 hop=128 outLen=1408 numFrames=8 block=256 op=window-sum-squares", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Contains("setp.ge.u32 %p1, %r3, 8", ptx);           // derived numFrames
        Assert.Contains("fma.rn.f32 %f0, %f1, %f1, %f0", ptx);     // sum += w*w
        Assert.Equal(1, Count(ptx, "ld.global.nc.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxWindowSumSquaresF32Kernel.IsSupportedShape(512, 128, 1408));
        Assert.False(PtxWindowSumSquaresF32Kernel.IsSupportedShape(512, 128, 1400));   // (out-nFft) not divisible by hop
        Assert.False(PtxWindowSumSquaresF32Kernel.IsPromotedShape(512, 128, 1408));
    }

    [SkippableFact]
    public void DriverOnlyOverlapAddAndWindowSumSquares_MatchDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int numFrames = 8, nFft = 512, hop = 128, outLen = (numFrames - 1) * hop + nFft;
        using var overlap = new PtxOverlapAddF32Kernel(runtime, numFrames, nFft, hop, outLen);
        using var winSq = new PtxWindowSumSquaresF32Kernel(runtime, nFft, hop, outLen);
        Assert.Equal(numFrames, winSq.NumFrames);

        var frames = new float[numFrames * nFft];
        var window = new float[nFft];
        var random = RandomHelper.CreateSeededRandom(20261230);
        for (int i = 0; i < frames.Length; i++) frames[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < nFft; i++) window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / nFft));   // Hann

        var expectedOut = new double[outLen];
        var expectedWss = new double[outLen];
        for (int idx = 0; idx < outLen; idx++)
        {
            double s = 0, w2 = 0;
            for (int frame = 0; frame < numFrames; frame++)
            {
                int local = idx - frame * hop;
                if (local >= 0 && local < nFft)
                {
                    s += (double)frames[frame * nFft + local] * window[local];
                    w2 += (double)window[local] * window[local];
                }
            }
            expectedOut[idx] = s; expectedWss[idx] = w2;
        }

        using var framesB = runtime.AllocateBytes(overlap.Blueprint.Tensors[0].RequiredBytes);
        using var windowB = runtime.AllocateBytes(overlap.Blueprint.Tensors[1].RequiredBytes);
        using var outB = runtime.AllocateBytes(overlap.Blueprint.Tensors[2].RequiredBytes);
        using var wssB = runtime.AllocateBytes(winSq.Blueprint.Tensors[0].RequiredBytes);
        framesB.Upload<float>(frames); windowB.Upload<float>(window);

        overlap.Launch(
            DirectPtxTensorView.CreateOwned(framesB, overlap.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(windowB, overlap.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outB, overlap.Blueprint.Tensors[2]));
        winSq.Launch(
            DirectPtxTensorView.CreateOwned(wssB, winSq.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(windowB, winSq.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actualOut = new float[outLen];
        var actualWss = new float[outLen];
        outB.Download<float>(actualOut); wssB.Download<float>(actualWss);
        for (int idx = 0; idx < outLen; idx++)
        {
            Assert.True(Math.Abs(actualOut[idx] - expectedOut[idx]) <= 1e-3 * (1 + Math.Abs(expectedOut[idx])));
            Assert.True(Math.Abs(actualWss[idx] - expectedWss[idx]) <= 1e-3 * (1 + Math.Abs(expectedWss[idx])));
        }
    }

    [Fact]
    public void ApplyWindowEmitter_IsSingleMul()
    {
        string ptx = PtxApplyWindowF32Kernel.EmitPtx(8, 6, 65536);
        Assert.Contains("exact-shape count=65536 block=256 op=apply-window", ptx);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "mul.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxApplyWindowF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxApplyWindowF32Kernel.IsSupportedShape(65500));   // not multiple of 256
        Assert.False(PtxApplyWindowF32Kernel.IsPromotedShape(65536));
    }

    [Fact]
    public void PowerToDbEmitter_IsLg2Approx()
    {
        string ptx = PtxPowerToDbF32Kernel.EmitPtx(8, 6, 65536);
        Assert.Contains("exact-shape count=65536 block=256 op=power-to-db", ptx);
        Assert.Contains("ld.param.f32 %f3, [ref_val]", ptx);
        Assert.Contains("ld.param.f32 %f4, [min_db]", ptx);
        Assert.Contains("lg2.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "max.f32"));    // max(power,1e-10) and max(db,minDb)
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPowerToDbF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxPowerToDbF32Kernel.IsPromotedShape(65536));
    }

    [Fact]
    public void DbToPowerEmitter_IsEx2Approx()
    {
        string ptx = PtxDbToPowerF32Kernel.EmitPtx(8, 6, 65536);
        Assert.Contains("exact-shape count=65536 block=256 op=db-to-power", ptx);
        Assert.Contains("ld.param.f32 %f3, [ref_val]", ptx);
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Equal(3, Count(ptx, "mul.rn.f32"));   // db*const, refSq, *refSq
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxDbToPowerF32Kernel.IsSupportedShape(65536));
        Assert.False(PtxDbToPowerF32Kernel.IsPromotedShape(65536));
    }

    [SkippableFact]
    public void DriverOnlyApplyWindow_MatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 65536;
        using var kernel = new PtxApplyWindowF32Kernel(runtime, n);
        var input = new float[n];
        var window = new float[n];
        var random = RandomHelper.CreateSeededRandom(20270101);
        for (int i = 0; i < n; i++) { input[i] = (float)(random.NextDouble() * 2 - 1); window[i] = (float)random.NextDouble(); }
        using var iB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var wB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var oB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        iB.Upload<float>(input); wB.Upload<float>(window);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(iB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(wB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(oB, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[n];
        oB.Download<float>(actual);
        for (int i = 0; i < n; i++) Assert.Equal(input[i] * window[i], actual[i]);   // bit-exact
    }

    [SkippableFact]
    public void DriverOnlyPowerDbRoundTrip_MatchesWithinTolerance()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int n = 65536;
        const float refValue = 1.0f, minDb = -80.0f;
        using var toDb = new PtxPowerToDbF32Kernel(runtime, n);
        using var toPower = new PtxDbToPowerF32Kernel(runtime, n);
        var power = new float[n];
        var random = RandomHelper.CreateSeededRandom(20270102);
        for (int i = 0; i < n; i++) power[i] = (float)(random.NextDouble() * 4 + 1e-3);   // above the floor
        var expectedDb = new double[n];
        for (int i = 0; i < n; i++)
            expectedDb[i] = Math.Max(10.0 * Math.Log10(Math.Max(power[i], 1e-10) / (refValue * (double)refValue)), minDb);

        using var pB = runtime.AllocateBytes(toDb.Blueprint.Tensors[0].RequiredBytes);
        using var dB = runtime.AllocateBytes(toDb.Blueprint.Tensors[1].RequiredBytes);
        using var p2B = runtime.AllocateBytes(toPower.Blueprint.Tensors[1].RequiredBytes);
        pB.Upload<float>(power);
        toDb.Launch(
            DirectPtxTensorView.CreateOwned(pB, toDb.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dB, toDb.Blueprint.Tensors[1]), refValue, minDb);
        toPower.Launch(
            DirectPtxTensorView.CreateOwned(dB, toPower.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(p2B, toPower.Blueprint.Tensors[1]), refValue);
        runtime.Synchronize();

        var actualDb = new float[n];
        var recovered = new float[n];
        dB.Download<float>(actualDb); p2B.Download<float>(recovered);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(actualDb[i] - expectedDb[i]) <= 1e-2 * (1 + Math.Abs(expectedDb[i])));
            Assert.True(Math.Abs(recovered[i] - power[i]) <= 1e-2 * (1 + power[i]));   // round trip
        }
    }

    [Fact]
    public void BuildSpectrumEmitter_IsThreeLoopHermitian()
    {
        string ptx = PtxBuildSpectrumF32Kernel.EmitPtx(8, 6, 4, 257, 8, 512);   // batch=4,numFreqs=257,numFrames=8,nFft=512
        Assert.Contains("exact-shape batch=4 numFreqs=257 numFrames=8 nFft=512 block=256 op=build-spectrum", ptx);
        Assert.Contains("$BS_ZERO:", ptx);
        Assert.Contains("$BS_FILL_LOOP:", ptx);
        Assert.Contains("$BS_MIRROR_LOOP:", ptx);
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Equal(1, Count(ptx, "neg.f32"));   // conjugate mirror
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxBuildSpectrumF32Kernel.IsSupportedShape(4, 257, 8, 512));
        Assert.False(PtxBuildSpectrumF32Kernel.IsSupportedShape(4, 400, 8, 512));   // numFreqs > nFft/2+1
        Assert.False(PtxBuildSpectrumF32Kernel.IsPromotedShape(4, 257, 8, 512));
    }

    [Fact]
    public void StftMagPhaseEmitter_IsWindowedDftWithAtan2()
    {
        string ptx = PtxStftMagPhaseF32Kernel.EmitPtx(8, 6, 4, 1408, 512, 128, 8, 257);
        Assert.Contains("op=stft-mag-phase", ptx);
        Assert.Contains("$STFT_LOOP:", ptx);
        Assert.Contains("cos.approx.f32", ptx);
        Assert.Contains("sin.approx.f32", ptx);
        Assert.Contains("sqrt.rn.f32", ptx);
        Assert.Contains("rcp.approx.f32", ptx);   // atan2 minimax
        Assert.Equal(2, Count(ptx, "fma.rn.f32 %f0, %f5, %f8, %f0") + Count(ptx, "fma.rn.f32 %f1, %f5, %f9, %f1"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxStftMagPhaseF32Kernel.IsSupportedShape(4, 1408, 512, 128, 8, 257));
        Assert.False(PtxStftMagPhaseF32Kernel.IsSupportedShape(4, 100, 512, 128, 8, 257));   // Lp too small
        Assert.False(PtxStftMagPhaseF32Kernel.IsPromotedShape(4, 1408, 512, 128, 8, 257));
    }

    [Fact]
    public void PhaseVocoderEmitter_IsLerpWithWrappedPhase()
    {
        string ptx = PtxPhaseVocoderF32Kernel.EmitPtx(8, 6, 4, 8, 257, 12);   // leading=4,nFramesV=8,nFreqV=257,outFrames=12
        Assert.Contains("op=phase-vocoder", ptx);
        Assert.Contains("$PV_LOOP:", ptx);
        Assert.Contains("ld.param.f32 %f0, [rate_val]", ptx);
        Assert.Contains("cvt.rmi.s32.f32", ptx);   // floor(t*rate)
        Assert.Contains("cvt.rni.f32.f32", ptx);   // round(dp/2pi)
        Assert.Contains("min.s32 %r9", ptx);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxPhaseVocoderF32Kernel.IsSupportedShape(4, 8, 257, 12));
        Assert.False(PtxPhaseVocoderF32Kernel.IsSupportedShape(4, 1, 257, 12));   // nFramesV < 2
        Assert.False(PtxPhaseVocoderF32Kernel.IsPromotedShape(4, 8, 257, 12));
    }

    [SkippableFact]
    public void DriverOnlyBuildSpectrum_MatchesDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int batch = 4, numFreqs = 257, numFrames = 8, nFft = 512;
        using var kernel = new PtxBuildSpectrumF32Kernel(runtime, batch, numFreqs, numFrames, nFft);
        var mag = new float[batch * numFreqs * numFrames];
        var phase = new float[batch * numFreqs * numFrames];
        var random = RandomHelper.CreateSeededRandom(20270201);
        for (int i = 0; i < mag.Length; i++) { mag[i] = (float)random.NextDouble(); phase[i] = (float)(random.NextDouble() * 2 * Math.PI - Math.PI); }

        var expRe = new double[batch * numFrames * nFft];
        var expIm = new double[batch * numFrames * nFft];
        for (int b = 0; b < batch; b++)
            for (int frame = 0; frame < numFrames; frame++)
            {
                int specOff = (b * numFrames + frame) * nFft;
                for (int k = 0; k < numFreqs; k++)
                {
                    double m = mag[b * numFreqs * numFrames + k * numFrames + frame];
                    double p = phase[b * numFreqs * numFrames + k * numFrames + frame];
                    expRe[specOff + k] = m * Math.Cos(p); expIm[specOff + k] = m * Math.Sin(p);
                }
                for (int k = 1; k < numFreqs - 1; k++)
                {
                    int dst = nFft - k;
                    expRe[specOff + dst] = expRe[specOff + k]; expIm[specOff + dst] = -expIm[specOff + k];
                }
            }

        using var magB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var phB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var reB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var imB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        magB.Upload<float>(mag); phB.Upload<float>(phase);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(magB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(phB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(reB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(imB, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actRe = new float[batch * numFrames * nFft];
        var actIm = new float[batch * numFrames * nFft];
        reB.Download<float>(actRe); imB.Download<float>(actIm);
        for (int i = 0; i < actRe.Length; i++)
        {
            Assert.True(Math.Abs(actRe[i] - expRe[i]) <= 2e-4 * (1 + Math.Abs(expRe[i])));
            Assert.True(Math.Abs(actIm[i] - expIm[i]) <= 2e-4 * (1 + Math.Abs(expIm[i])));
        }
    }

    [SkippableFact]
    public void DriverOnlyStftMagPhase_MatchesDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int batch = 2, nFft = 256, hop = 128, numFrames = 6, numFreqs = 129;
        int lp = (numFrames - 1) * hop + nFft;
        using var kernel = new PtxStftMagPhaseF32Kernel(runtime, batch, lp, nFft, hop, numFrames, numFreqs);
        var padded = new float[batch * lp];
        var window = new float[nFft];
        var random = RandomHelper.CreateSeededRandom(20270202);
        for (int i = 0; i < padded.Length; i++) padded[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < nFft; i++) window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / nFft));

        var expMag = new double[batch * numFreqs * numFrames];
        var expPhase = new double[batch * numFreqs * numFrames];
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < numFreqs; k++)
                for (int frame = 0; frame < numFrames; frame++)
                {
                    double re = 0, im = 0, bk = -2.0 * Math.PI * k / nFft;
                    for (int i = 0; i < nFft; i++)
                    {
                        double x = padded[b * lp + frame * hop + i] * window[i];
                        re += x * Math.Cos(bk * i); im += x * Math.Sin(bk * i);
                    }
                    int outOff = b * numFreqs * numFrames + k * numFrames + frame;
                    expMag[outOff] = Math.Sqrt(re * re + im * im); expPhase[outOff] = Math.Atan2(im, re);
                }

        using var pB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var wB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var mB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var phB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        pB.Upload<float>(padded); wB.Upload<float>(window);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(pB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(wB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(mB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(phB, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actMag = new float[batch * numFreqs * numFrames];
        var actPhase = new float[batch * numFreqs * numFrames];
        mB.Download<float>(actMag); phB.Download<float>(actPhase);
        for (int i = 0; i < actMag.Length; i++)
        {
            Assert.True(Math.Abs(actMag[i] - expMag[i]) <= 1e-2 * (1 + Math.Abs(expMag[i])));
            if (expMag[i] > 1e-3)   // phase is meaningless where magnitude vanishes
            {
                double d = Math.Abs(actPhase[i] - expPhase[i]);
                d = Math.Min(d, Math.Abs(d - 2 * Math.PI));   // wrap-around
                Assert.True(d <= 2e-3);
            }
        }
    }

    [SkippableFact]
    public void DriverOnlyPhaseVocoder_MatchesDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int leading = 4, nFramesV = 8, nFreqV = 257, outFrames = 12;
        const float rate = 0.6f;
        using var kernel = new PtxPhaseVocoderF32Kernel(runtime, leading, nFramesV, nFreqV, outFrames);
        var mag = new float[leading * nFramesV * nFreqV];
        var phase = new float[leading * nFramesV * nFreqV];
        var random = RandomHelper.CreateSeededRandom(20270203);
        for (int i = 0; i < mag.Length; i++) { mag[i] = (float)random.NextDouble(); phase[i] = (float)(random.NextDouble() * 2 * Math.PI - Math.PI); }

        var expMag = new double[leading * outFrames * nFreqV];
        var expPhase = new double[leading * outFrames * nFreqV];
        int stride = nFramesV * nFreqV, outStride = outFrames * nFreqV;
        for (int b = 0; b < leading; b++)
            for (int f = 0; f < nFreqV; f++)
            {
                double acc = 0;
                for (int t = 0; t < outFrames; t++)
                {
                    double srcT = t * (double)rate;
                    int t0 = (int)Math.Floor(srcT);
                    int t1 = Math.Min(t0 + 1, nFramesV - 1);
                    double frac = srcT - t0;
                    double m0 = mag[b * stride + t0 * nFreqV + f], m1 = mag[b * stride + t1 * nFreqV + f];
                    expMag[b * outStride + t * nFreqV + f] = (1 - frac) * m0 + frac * m1;
                    double dp = 0;
                    if (t0 + 1 < nFramesV)
                    {
                        dp = phase[b * stride + (t0 + 1) * nFreqV + f] - phase[b * stride + t0 * nFreqV + f];
                        dp -= 2 * Math.PI * Math.Round(dp / (2 * Math.PI));
                    }
                    acc += dp; expPhase[b * outStride + t * nFreqV + f] = acc;
                }
            }

        using var magB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var phB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var nmB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var npB = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        magB.Upload<float>(mag); phB.Upload<float>(phase);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(magB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(phB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(nmB, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(npB, kernel.Blueprint.Tensors[3]), rate);
        runtime.Synchronize();
        var actMag = new float[leading * outFrames * nFreqV];
        var actPhase = new float[leading * outFrames * nFreqV];
        nmB.Download<float>(actMag); npB.Download<float>(actPhase);
        for (int i = 0; i < actMag.Length; i++)
        {
            Assert.True(Math.Abs(actMag[i] - expMag[i]) <= 1e-3 * (1 + Math.Abs(expMag[i])));
            Assert.True(Math.Abs(actPhase[i] - expPhase[i]) <= 1e-2 * (1 + Math.Abs(expPhase[i])));
        }
    }

    [Fact]
    public void MelFilterbankApplyEmitter_IsGuardedFmaReduction()
    {
        string ptx = PtxMelFilterbankApplyF32Kernel.EmitPtx(8, 6, 100, 257, 80);   // 100*80 = 8000, not mult of 256
        Assert.Contains("exact-shape totalSegBatch=100 specBins=257 melBins=80 block=256 op=mel-filterbank-apply", ptx);
        Assert.Contains("setp.ge.u32 %p0, %r2, 8000", ptx);     // total guard
        Assert.Contains("div.u32 %r3, %r2, 80", ptx);           // seg = outIdx / melBins
        Assert.Contains("rem.u32 %r4, %r2, 80", ptx);           // m = outIdx % melBins
        Assert.Contains("$MFA_LOOP:", ptx);
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(2, Count(ptx, "ld.global.nc.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxMelFilterbankApplyF32Kernel.IsSupportedShape(100, 257, 80));
        Assert.False(PtxMelFilterbankApplyF32Kernel.IsSupportedShape(0, 257, 80));   // seg >= 1
        Assert.False(PtxMelFilterbankApplyF32Kernel.IsPromotedShape(100, 257, 80));
    }

    [SkippableFact]
    public void DriverOnlyMelFilterbankApply_MatchesDoubleOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedComplexUnary(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The candidate is admitted only on SM86.");
        const int seg = 100, specBins = 257, melBins = 80;
        using var kernel = new PtxMelFilterbankApplyF32Kernel(runtime, seg, specBins, melBins);
        var power = new float[seg * specBins];
        var filters = new float[melBins * specBins];
        var random = RandomHelper.CreateSeededRandom(20270301);
        for (int i = 0; i < power.Length; i++) power[i] = (float)random.NextDouble();
        for (int i = 0; i < filters.Length; i++) filters[i] = (float)random.NextDouble();
        var expected = new float[seg * melBins];
        for (int s = 0; s < seg; s++)
            for (int m = 0; m < melBins; m++)
            {
                double sum = 0;
                for (int i = 0; i < specBins; i++) sum += (double)power[s * specBins + i] * filters[m * specBins + i];
                expected[s * melBins + m] = (float)sum;
            }
        using var pB = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var fB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var eB = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        pB.Upload<float>(power); fB.Upload<float>(filters);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(pB, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(fB, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(eB, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[seg * melBins];
        eB.Download<float>(actual);
        for (int i = 0; i < actual.Length; i++) Assert.True(MathF.Abs(actual[i] - expected[i]) <= 3e-4f);
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
