#if NETCOREAPP
using System;
using System.Linq;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549 rate detection: the driver's SM issue-rate modifiers are read first, and the microbenchmark only when the driver
/// cannot answer. Routing is checked against a mocked driver query, so a full-rate (unlocked or datacenter) device and a
/// reduced-rate GeForce can both be exercised on whichever card runs the tests.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class MmaRateSourceTests
{
    private readonly ITestOutputHelper _output;

    public MmaRateSourceTests(ITestOutputHelper output) => _output = output;

    private sealed class FakeIssueRateQuery : ISmIssueRateQuery
    {
        private readonly SmIssueRateModifier? _answer;
        private readonly string _failure;

        internal FakeIssueRateQuery(SmIssueRateModifier? answer, string failure = "refused for tests")
        {
            _answer = answer;
            _failure = failure;
        }

        internal int Calls { get; private set; }
        internal (int Domain, int Bus, int Device) LastAddress { get; private set; }

        public bool TryQuery(int pciDomain, int pciBus, int pciDevice, out SmIssueRateModifier modifier, out string failure)
        {
            Calls++;
            LastAddress = (pciDomain, pciBus, pciDevice);
            modifier = _answer ?? default;
            failure = _answer is null ? _failure : string.Empty;
            return _answer is not null;
        }
    }

    // The speed selects map onto the same rate classes a measurement produces: FMLA32 relative to FMLA16.
    [Theory]
    [InlineData(0, 0, "fp32acc-full", true)]
    [InlineData(0, 1, "fp32acc-1/2", false)]
    [InlineData(0, 2, "fp32acc-1/4", false)]
    [InlineData(0, 3, "fp32acc-1/8", false)]   // RTX 5090 (gpu-unlock-ada#12)
    [InlineData(1, 1, "fp32acc-full", true)]   // both halved: FP16 accumulation cannot double anything
    public void DriverModifiers_MapToTheMeasuredRateClasses(byte fmla16, byte fmla32, string expectedClass, bool fullRate)
    {
        var rate = new SmIssueRateModifier(fmla16, fmla32, 0, 0).ToRateResult();
        Assert.Equal(expectedClass, rate.RateClass);
        Assert.Equal(fullRate, rate.Fp32AccumulateIsFullRate);
        Assert.Equal(PtxMmaRateProbe.RateSource.DriverQuery, rate.Source);
        Assert.Contains("full issue rate", rate.Describe());
    }

    [Fact]
    public void MeasuredRate_DescribesItselfInTflops()
    {
        var rate = new PtxMmaRateProbe.Result(61.9, 123.2);
        Assert.Equal(PtxMmaRateProbe.RateSource.Measured, rate.Source);
        Assert.Contains("TFLOP/s", rate.Describe());
    }

    // _IOWR('F', nr, size) as the driver decodes it (_IOC_NR, _IOC_SIZE): NVOS54 (32 bytes) for RM_CONTROL 0x2A and
    // NVOS64 (48 bytes) for RM_ALLOC 0x2B.
    [Fact]
    public void IoctlRequests_UseTheLinuxReadWriteEncoding()
    {
        Assert.Equal(0xC020462AUL, NvRmIssueRateQuery.IoctlRequest(0x2A, 32));
        Assert.Equal(0xC030462BUL, NvRmIssueRateQuery.IoctlRequest(0x2B, 48));
        Assert.Equal(0xC00446C9UL, NvRmIssueRateQuery.IoctlRequest(200 + 1, 4));   // NV_ESC_REGISTER_FD
        Assert.Equal(0x20801230U, NvRmIssueRateQuery.NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER);
    }

    // Off Linux there is no user-mode route to the control: the query declines, so the caller measures.
    [SkippableFact]
    public void DriverQuery_DeclinesOffLinux()
    {
        Skip.If(RuntimeInformation.IsOSPlatform(OSPlatform.Linux), "Linux has the driver interface.");
        Assert.False(NvRmIssueRateQuery.Instance.TryQuery(0, 1, 0, out _, out string failure));
        Assert.Contains("Linux", failure);
    }

    // On Linux without the NVIDIA device files (WSL2 exposes /dev/dxg, a container without the driver mounted), the
    // query goes through the real libc open/ioctl P/Invokes and declines with a reason; the caller then measures.
    [SkippableFact]
    public void DriverQuery_OnLinuxWithoutTheControlDevice_DeclinesWithAReason()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Linux), "Linux-only path.");
        Skip.If(System.IO.File.Exists("/dev/nvidiactl"), "This machine has the NVIDIA control device.");
        Assert.False(NvRmIssueRateQuery.Instance.TryQuery(0, 1, 0, out _, out string failure));
        Assert.Contains("/dev/nvidiactl", failure);
    }

    // Wherever the driver answers (a Linux box with the NVIDIA driver), its answer must be a valid speed select.
    [SkippableFact]
    public void DriverQuery_OnThisMachine_ReportsValidSpeedSelects()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Linux), "The RM ioctl interface is Linux-only.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "No CUDA device.");
        var profile = backend.TensorCoreAccumulateRateProfile();
        _output.WriteLine(profile is { } p
            ? $"rate source {p.Source}: FP32 acc {p.Fp32Accumulate:G4}, FP16 acc {p.Fp16Accumulate:G4} -> {p.RateClass}"
            : "no rate");
        if (profile is not { Source: "driver" } answered)
        {
            // A broken struct layout, ioctl number or refused client must not pass as a skip where the query is
            // expected to work: lanes with an NVIDIA Linux driver set AIDOTNET_REQUIRE_RM_QUERY=1 to make it a failure.
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_RM_QUERY") != "1",
                $"The driver did not answer: {backend.MmaRateQueryFailure}");
            Assert.Fail($"AIDOTNET_REQUIRE_RM_QUERY=1 but the driver query did not answer: {backend.MmaRateQueryFailure}");
            return;
        }
        Assert.InRange(answered.Fp32Accumulate, 1.0 / 64, 1.0);
        Assert.InRange(answered.Fp16Accumulate, 1.0 / 64, 1.0);
        // The class must be the one the same rates produce, so routing keys verdicts consistently.
        Assert.Equal(
            new PtxMmaRateProbe.Result(answered.Fp32Accumulate, answered.Fp16Accumulate, PtxMmaRateProbe.RateSource.DriverQuery).RateClass,
            answered.RateClass);
    }

    // Speed selects above REDUCED_SPEED_1_64 are not defined: the query declines them so the caller measures instead of
    // caching garbage (an all-zero answer is legitimate: a full-rate part).
    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 6, true)]
    [InlineData(6, 6, true)]
    [InlineData(0, 7, false)]
    [InlineData(7, 0, false)]
    [InlineData(255, 1, false)]
    public void SpeedSelects_OutsideTheDefinedRange_AreRejected(byte fmla16, byte fmla32, bool defined)
    {
        Assert.Equal(defined, SmIssueRateModifier.SpeedSelectsAreDefined(fmla16, fmla32));
    }

    [Theory]
    [InlineData(null, nameof(MmaRateSourcePolicy.Auto))]
    [InlineData("", nameof(MmaRateSourcePolicy.Auto))]
    [InlineData("auto", nameof(MmaRateSourcePolicy.Auto))]
    [InlineData(" Driver ", nameof(MmaRateSourcePolicy.Driver))]
    [InlineData("PROBE", nameof(MmaRateSourcePolicy.Probe))]
    [InlineData("bogus", nameof(MmaRateSourcePolicy.Auto))]
    public void RateSourceSetting_ParsesTheDocumentedValues(string? value, string expected)
    {
        // The policy enum is internal, so the theory names the expected member.
        Assert.Equal(expected, CudaBackend.ParseMmaRateSource(value).ToString());
    }

    // A persisted TF32 verdict applies only at the rate class it was measured at; verdicts recorded before the class was
    // stored came from stock (reduced-rate) cards.
    [Theory]
    [InlineData("fp32acc-1/2", "fp32acc-1/2", true)]    // recorded at the current class
    [InlineData("fp32acc-1/2", "fp32acc-full", false)]  // recorded at another class
    [InlineData(null, "fp32acc-1/2", true)]             // legacy verdict, reduced-rate device
    [InlineData(null, "fp32acc-full", false)]           // legacy verdict, full-rate device
    [InlineData("unknown", "unknown", false)]           // no rate either time: may differ, never reused
    [InlineData(null, "unknown", false)]                // legacy verdict, rate unreadable now
    public void TensorCoreVerdict_AppliesOnlyAtItsRateClass(string? recorded, string current, bool applies)
    {
        var parameters = new System.Collections.Generic.Dictionary<string, string>();
        if (recorded is not null) parameters["mma-rate"] = recorded;
        Assert.Equal(applies, CudaBackend.TensorCoreVerdictMatchesRate(parameters, current));
    }

    private static CudaBackend RequireAmpereBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || backend.GetDirectPtxRuntime().ComputeCapabilityMajor < 8)
        {
            backend.Dispose();
            Skip.If(true, "Needs a CUDA device of compute capability 8.0+.");
        }
        return backend;
    }

    private static byte[] HalfBytes(Half value, int count)
    {
        ushort bits = BitConverter.HalfToUInt16Bits(value);
        var bytes = new byte[count * 2];
        for (int i = 0; i < count; i++)
        {
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    // A driver that reports FMLA32 at full rate (an unlocked GeForce or a datacenter part) keeps cuBLAS without tuning,
    // from the driver's answer alone: no microbenchmark runs, and the verdict says where the rate came from.
    [SkippableFact]
    public void DriverReportsFullRate_KeepsCublas_WithoutMeasuring()
    {
        using var backend = RequireAmpereBackend();
        var query = new FakeIssueRateQuery(new SmIssueRateModifier(0, 0, 0, 0));
        backend.SmIssueRateQueryForTests = query;
        backend.PciAddressForTests = (0, 0x41, 0);
        backend.ResetMmaRateForTests();

        const int m = 1024, n = 1024, k = 1024;
        using var da = backend.AllocateByteBuffer(m * k * 2);
        using var db = backend.AllocateByteBuffer(k * n * 2);
        using var dc = backend.AllocateBuffer(m * n);
        backend.UploadBytes(da, HalfBytes((Half)0.5f, m * k));
        backend.UploadBytes(db, HalfBytes((Half)0.25f, k * n));
        long before = backend.Fp16AccumulateDispatchCount;
        backend.GemmFp16In32fOut(da, db, dc, m, n, k);

        Assert.Equal(1, query.Calls);
        Assert.Equal((0, 0x41, 0), query.LastAddress);
        var verdictOrNull = backend.Fp16AccumulateVerdict(m, n, k);
        Assert.True(verdictOrNull.HasValue, "the route recorded no verdict");
        var verdict = verdictOrNull.GetValueOrDefault();
        Assert.False(verdict.Routed);
        Assert.Contains("FP16 rate", verdict.Reason);
        Assert.Contains("driver speed selects", verdict.Reason);
        Assert.Equal(before, backend.Fp16AccumulateDispatchCount);
        Assert.True(backend.Fp32AccumulateIsFullRate());
        Assert.Equal(PtxMmaRateProbe.FullRateClass, backend.MmaRateClass());
        Assert.Equal("driver", backend.TensorCoreAccumulateRateProfile()?.Source);
        Assert.All(backend.DownloadBuffer(dc), v => Assert.Equal(0.125f * k, v));
    }

    // A reduced-rate report is used as is (and read once per backend).
    [SkippableFact]
    public void DriverReportsHalfRate_ClassIsHalf_AndIsReadOnce()
    {
        using var backend = RequireAmpereBackend();
        var query = new FakeIssueRateQuery(new SmIssueRateModifier(0, 1, 0, 0));
        backend.SmIssueRateQueryForTests = query;
        backend.PciAddressForTests = (0, 1, 0);
        backend.ResetMmaRateForTests();

        Assert.Equal("fp32acc-1/2", backend.MmaRateClass());
        Assert.False(backend.Fp32AccumulateIsFullRate());
        var profileOrNull = backend.TensorCoreAccumulateRateProfile();
        Assert.True(profileOrNull.HasValue, "no rate profile");
        var profile = profileOrNull.GetValueOrDefault();
        Assert.Equal("driver", profile.Source);
        Assert.Equal(0.5, profile.Fp32Accumulate);
        Assert.Equal(1.0, profile.Fp16Accumulate);
        Assert.Equal(1, query.Calls);
    }

    // When the driver declines, the probe measures instead, and the reason the driver gave is kept for diagnostics.
    [SkippableFact]
    public void DriverDeclines_FallsBackToTheMicrobenchmark()
    {
        using var backend = RequireAmpereBackend();
        var query = new FakeIssueRateQuery(null, "status 0x1f for tests");
        backend.SmIssueRateQueryForTests = query;
        backend.PciAddressForTests = (0, 1, 0);
        backend.ResetMmaRateForTests();

        var profile = backend.TensorCoreAccumulateRateProfile();
        Assert.Equal(1, query.Calls);
        Assert.Equal("status 0x1f for tests", backend.MmaRateQueryFailure);
        Assert.True(profile is { Source: "measured" }, $"expected a measured rate, got {profile}");
        var measured = profile.GetValueOrDefault();
        _output.WriteLine($"measured: FP32 acc {measured.Fp32Accumulate:F1}, FP16 acc {measured.Fp16Accumulate:F1} TFLOP/s -> {measured.RateClass}");
        Assert.InRange(measured.Fp32Accumulate, 1, 10_000);
    }

    // AIDOTNET_MMA_RATE_SOURCE=driver: the driver's answer or nothing; a declined query never falls back to measuring.
    [SkippableFact]
    public void DriverOnlySource_DriverDeclines_ReportsNoRate()
    {
        using var backend = RequireAmpereBackend();
        var query = new FakeIssueRateQuery(null, "declined for tests");
        backend.SmIssueRateQueryForTests = query;
        backend.PciAddressForTests = (0, 1, 0);
        backend.MmaRateSource = MmaRateSourcePolicy.Driver;
        backend.ResetMmaRateForTests();

        Assert.Null(backend.TensorCoreAccumulateRateProfile());
        Assert.Equal("unknown", backend.MmaRateClass());
        Assert.Equal("declined for tests", backend.MmaRateQueryFailure);
        Assert.True(query.Calls >= 1);
    }

    // AIDOTNET_MMA_RATE_SOURCE=probe: always measured; the driver is never asked, even when it would answer.
    [SkippableFact]
    public void ProbeOnlySource_NeverAsksTheDriver()
    {
        using var backend = RequireAmpereBackend();
        var query = new FakeIssueRateQuery(new SmIssueRateModifier(0, 0, 0, 0));
        backend.SmIssueRateQueryForTests = query;
        backend.PciAddressForTests = (0, 1, 0);
        backend.MmaRateSource = MmaRateSourcePolicy.Probe;
        backend.ResetMmaRateForTests();

        var profile = backend.TensorCoreAccumulateRateProfile();

        Assert.Equal(0, query.Calls);
        Assert.True(profile is { Source: "measured" }, $"expected a measured rate, got {profile}");
    }
}
#endif
