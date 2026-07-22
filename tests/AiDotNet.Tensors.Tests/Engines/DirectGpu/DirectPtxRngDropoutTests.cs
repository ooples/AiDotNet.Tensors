#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxRngDropoutTests
{
    [Fact]
    public void EmitterBakesExactShapeAndHasNoTailOrStridePath()
    {
        string ptx = PtxFusedPhiloxDropoutF32Kernel.EmitPtx(8, 6, 65_536);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedPhiloxDropoutF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Contains("exact_elements=65536", ptx, StringComparison.Ordinal);
        Assert.Contains("no_tail_branch=1", ptx, StringComparison.Ordinal);
        Assert.Contains("ld.global.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("st.global.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("0xD2511F53", ptx, StringComparison.Ordinal);
        Assert.Contains("0xCD9E8D57", ptx, StringComparison.Ordinal);
        Assert.Equal(10, Count(ptx, "// Philox4x32-10 round"));
        Assert.DoesNotContain(".param .u32 size", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("size_ptr", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("setp.ge", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(4_096, true)]
    [InlineData(65_536, true)]
    [InlineData(1_048_576, true)]
    [InlineData(4_095, false)]
    [InlineData(65_540, false)]
    public void ExactShapeBucketsAreExplicit(int elements, bool expected) =>
        Assert.Equal(expected, PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elements));

    [Fact]
    public void EmitterRejectsEveryUnmeasuredSm()
    {
        Assert.Throws<PlatformNotSupportedException>(() =>
            PtxFusedPhiloxDropoutF32Kernel.EmitPtx(8, 9, 4_096));
    }

    [Fact]
    public void PhiloxReferenceMatchesPublishedZeroVector()
    {
        var actual = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(0, 0, 0);

        Assert.Equal(0x6627E8D5u, actual.X0);
        Assert.Equal(0xE169C58Du, actual.X1);
        Assert.Equal(0xBC57AC4Cu, actual.X2);
        Assert.Equal(0x9B00DBD8u, actual.X3);
    }

    [Fact]
    public void SeedSubsequenceAndCounterFormDeterministicIndependentDomains()
    {
        var first = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 29);
        var repeat = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 29);

        Assert.Equal(first, repeat);
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEEul, 17, 29));
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 18, 29));
        Assert.NotEqual(first, PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            0x0123_4567_89AB_CDEFul, 17, 30));
    }

    [Fact]
    public void AdmissionProvesExactExtentAlignmentAndNumericalContract()
    {
        const int elements = 4_096;
        const long bytes = elements * sizeof(float);
        bool admitted = DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, bytes,
            (IntPtr)0x20000, bytes,
            (IntPtr)0x30000, bytes,
            elements, 0.1f, 8, 6,
            out DirectPtxRngDropoutParameters parameters,
            out string? rejection);

        Assert.True(admitted, rejection);
        Assert.Null(rejection);
        Assert.NotEqual(0u, parameters.KeepThreshold);
        Assert.Equal(1.0f / 0.9f, parameters.InverseKeep, 5);
    }

    [Theory]
    [InlineData(8, 9, "rng-dropout-exact-sm-not-supported")]
    [InlineData(9, 0, "rng-dropout-exact-sm-not-supported")]
    public void AdmissionRejectsUnmeasuredArchitectures(
        int major, int minor, string expectedReason)
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x20000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, major, minor,
            out _, out string? rejection));
        Assert.Equal(expectedReason, rejection);
    }

    [Fact]
    public void AdmissionRejectsAliasingBeforeDispatch()
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x10000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, 8, 6,
            out _, out string? rejection));
        Assert.Equal("rng-dropout-alias-not-supported", rejection);
    }

    [Fact]
    public void AdmissionRejectsOversizedViewsInsteadOfSilentlyAcceptingThem()
    {
        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            (IntPtr)0x10000, 16_388,
            (IntPtr)0x20000, 16_384,
            (IntPtr)0x30000, 16_384,
            4_096, 0.1f, 8, 6,
            out _, out string? rejection));
        Assert.Equal("rng-dropout-physical-extent-mismatch", rejection);
    }

    [Theory]
    [InlineData("shape", "rng-dropout-exact-shape-not-supported")]
    [InlineData("rate", "rng-dropout-rate-not-supported")]
    [InlineData("pointer", "rng-dropout-invalid-device-pointer")]
    [InlineData("alignment", "rng-dropout-alignment-mismatch")]
    [InlineData("range", "rng-dropout-invalid-pointer-range")]
    public void AdmissionHasStableReasonsForEveryPhysicalRejection(
        string scenario, string expectedReason)
    {
        int elements = scenario == "shape" ? 4_100 : 4_096;
        long bytes = checked((long)elements * sizeof(float));
        IntPtr input = scenario switch
        {
            "pointer" => IntPtr.Zero,
            "alignment" => (IntPtr)0x10004,
            "range" => new IntPtr(-16),
            _ => (IntPtr)0x10000
        };
        float rate = scenario == "rate" ? 1.0f : 0.1f;

        Assert.False(DirectPtxRngDropoutAdmission.TryValidate(
            input, bytes,
            (IntPtr)0x20000, bytes,
            (IntPtr)0x30000, bytes,
            elements, rate, 8, 6,
            out _, out string? rejection));
        Assert.Equal(expectedReason, rejection);
    }

    [Fact]
    public void PinnedCaptureModuleCannotBeEvictedAndIsDisposedWithOwner()
    {
        using var cache = new DirectPtxKernelCache<int, DisposableProbe>(1);
        var pinned = new DisposableProbe();
        Assert.Same(pinned, cache.AddOrGetExisting(1, pinned));
        Assert.True(cache.Pin(1));
        var rejected = new DisposableProbe();

        Assert.Throws<InvalidOperationException>(() => cache.AddOrGetExisting(2, rejected));
        Assert.True(rejected.IsDisposed);
        Assert.False(pinned.IsDisposed);
        cache.Dispose();
        Assert.True(pinned.IsDisposed);
    }

    [Fact]
    public void CoverageManifestAssignsEachFamilyAndExcludesSecureRandom()
    {
        Assert.NotEmpty(DirectPtxRngCoverageManifest.All);
        Assert.Equal(
            DirectPtxRngCoverageManifest.All.Count,
            DirectPtxRngCoverageManifest.All.Select(cell => cell.Api).Distinct(StringComparer.Ordinal).Count());
        DirectPtxRngCoverageCell secure = DirectPtxRngCoverageManifest.Get(
            "Cryptographic and SecureSeededRandom APIs");
        Assert.Equal(DirectPtxRngCoverageStatus.ExplicitlyExcluded, secure.Status);
        Assert.Contains("fail closed", secure.DirectPtxAssignment, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FeatureGateIsKernelSpecificAndOptIn()
    {
        Assert.Equal("AIDOTNET_DIRECT_PTX_RNG_DROPOUT",
            DirectPtxFeatureGate.RngDropoutEnvironmentVariable);
        Assert.True(DirectPtxArchitecture.HasExperimentalRngDropout(8, 6));
        Assert.False(DirectPtxArchitecture.HasExperimentalRngDropout(8, 9));
    }

    private static int Count(string text, string value)
    {
        int count = 0;
        int start = 0;
        while ((start = text.IndexOf(value, start, StringComparison.Ordinal)) >= 0)
        {
            count++;
            start += value.Length;
        }
        return count;
    }

    private sealed class DisposableProbe : IDisposable
    {
        internal bool IsDisposed { get; private set; }
        public void Dispose() => IsDisposed = true;
    }
}
#endif
