using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.NumericOperations;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

public sealed class GpuPrecisionPolicyTests
{
    [Fact]
    public void CublasDefaultAlgorithmMatchesTheNativeHeaderValue()
        => Assert.Equal(-1, CuBlasNative.CUBLAS_GEMM_DEFAULT);

    [Fact]
    public void SpeedFirst_DefaultConvertsEveryOrdinaryPublicTypeThroughFp32()
    {
        var backend = CreateBackend(Fp32());

        AssertGpuPlan<float>(backend, GpuScalarType.Float32);
        AssertGpuPlan<double>(backend, GpuScalarType.Float32);
        AssertGpuPlan<int>(backend, GpuScalarType.Float32);
        AssertGpuPlan<long>(backend, GpuScalarType.Float32);
        AssertGpuPlan<decimal>(backend, GpuScalarType.Float32);

        Assert.Contains("Float64", Plan<double>(backend).FallbackReason);
        Assert.Contains("Generic", Plan<int>(backend).FallbackReason);
    }

    [Fact]
    public void SpeedFirst_AutoHonorsReducedPublicTypesWhenSupported()
    {
        var backend = CreateBackend(Fp32(), Fp16(), Bf16(), Fp8E4M3(), Fp8E5M2());

        AssertGpuPlan<Half>(backend, GpuScalarType.Float16);
        AssertGpuPlan<BFloat16>(backend, GpuScalarType.BFloat16);
        AssertGpuPlan<Float8E4M3>(backend, GpuScalarType.Float8E4M3);
        AssertGpuPlan<Float8E5M2>(backend, GpuScalarType.Float8E5M2);
    }

    [Fact]
    public void PreserveInputType_UsesCpuWhenTheBackendCannotComputeTheDeclaredType()
    {
        var backend = CreateBackend(Fp32(), Fp16());
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var doublePlan = Plan<double>(backend);
        var integerPlan = Plan<int>(backend);

        Assert.Equal(GpuExecutionRoute.Cpu, doublePlan.Route);
        Assert.Equal(GpuScalarType.Float64, doublePlan.MultiplyType);
        Assert.Contains("PreserveInputType", doublePlan.FallbackReason);
        Assert.Equal(GpuExecutionRoute.Cpu, integerPlan.Route);
        Assert.Equal(GpuScalarType.Generic, integerPlan.MultiplyType);
    }

    [Fact]
    public void PreserveInputType_UsesNativeDeclaredTypeWhenAdvertised()
    {
        var backend = CreateBackend(Fp32(), Fp64());
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        AssertGpuPlan<double>(backend, GpuScalarType.Float64);
    }

    [Fact]
    public void UnsupportedBFloat16FallsBackToFp32InsteadOfNarrowingToFp16()
    {
        var backend = CreateBackend(Fp32(), Fp16());
        using var policy = new GpuExecutionPolicyScope(new GpuExecutionPolicy(
            computePreference: GpuComputePreference.BFloat16));

        var plan = Plan<float>(backend);

        Assert.Equal(GpuScalarType.Float32, plan.MultiplyType);
        Assert.Contains("BFloat16", plan.FallbackReason);
    }

    [Fact]
    public void UnsupportedFp8UsesTheFirstSupportedHigherPrecisionFormat()
    {
        var backend = CreateBackend(Fp32(), Fp16(), Bf16());
        using var policy = new GpuExecutionPolicyScope(new GpuExecutionPolicy(
            computePreference: GpuComputePreference.Float8E4M3));

        var plan = Plan<float>(backend);

        Assert.Equal(GpuScalarType.BFloat16, plan.MultiplyType);
        Assert.Contains("Float8E4M3", plan.FallbackReason);
    }

    [Fact]
    public void ExplicitFp64NeverSilentlyFallsBackToLowerGpuPrecision()
    {
        var backend = CreateBackend(Fp32(), Fp16());
        using var policy = new GpuExecutionPolicyScope(new GpuExecutionPolicy(
            computePreference: GpuComputePreference.Float64));

        var plan = Plan<double>(backend);

        Assert.Equal(GpuExecutionRoute.Cpu, plan.Route);
        Assert.Equal(GpuScalarType.Float64, plan.MultiplyType);
        Assert.Contains("no eligible GPU precision", plan.FallbackReason);
    }

    [Fact]
    public void StrictUnsupportedAutocastFailsBeforeDispatch()
    {
        var backend = CreateBackend(Fp32());
        using var policy = new GpuExecutionPolicyScope(new GpuExecutionPolicy(
            fallbackBehavior: GpuPrecisionFallbackBehavior.Throw));
        using var autocast = new AutocastScope(PrecisionMode.Float8E5M2);

        var error = Assert.Throws<NotSupportedException>(() => Plan<float>(backend));
        Assert.Contains("Float8E5M2", error.Message);
    }

    [Fact]
    public void ThirdPartyBackendWithoutPrecisionInterfaceRemainsFp32Compatible()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());

        var plan = Plan<double>(backend);

        Assert.Equal(GpuExecutionRoute.Gpu, plan.Route);
        Assert.Equal(GpuScalarType.Float32, plan.InputStorage);
        Assert.Equal(GpuScalarType.Float32, plan.AccumulatorType);
    }

    [Fact]
    public void Fp16ElementwiseCapabilityKeepsFp32AccumulatorAndFp16Output()
    {
        var capabilities = GpuPrecisionCapabilityCatalog.Create(
            supportsFp16: true,
            GpuPrecisionImplementation.Native,
            fp16ReducesStorageBytes: true,
            fp16OutputStorage: GpuScalarType.Float16,
            fp16MultiplyType: GpuScalarType.Float32);

        var fp16 = Assert.Single(capabilities, capability =>
            capability.InputStorage == GpuScalarType.Float16);
        Assert.Equal(GpuScalarType.Float32, fp16.AccumulatorType);
        Assert.Equal(GpuScalarType.Float16, fp16.OutputStorage);
        Assert.Equal(GpuScalarType.Float16, fp16.ComputeFormat);
        Assert.Equal(GpuScalarType.Float32, fp16.MultiplyType);
    }

    [Fact]
    public void TensorFloat32CapabilitySeparatesLogicalSelectionFromPhysicalMultiply()
    {
        var capabilities = GpuPrecisionCapabilityCatalog.Create(
            supportsFp16: false,
            GpuPrecisionImplementation.Native,
            fp16ReducesStorageBytes: false,
            fp32Implementation: GpuPrecisionImplementation.VendorLibrary,
            fp32MultiplyType: GpuScalarType.TensorFloat32,
            supportsTensorFloat32: true);
        var backend = CreateBackend(capabilities.ToArray());

        var automatic = Plan<float>(backend);
        Assert.Equal(GpuScalarType.Float32, automatic.ComputeFormat);
        Assert.Equal(GpuScalarType.TensorFloat32, automatic.MultiplyType);
        Assert.Equal(GpuScalarType.Float32, automatic.AccumulatorType);

        using var explicitTf32 = new GpuExecutionPolicyScope(new GpuExecutionPolicy(
            computePreference: GpuComputePreference.TensorFloat32));
        var requested = Plan<float>(backend);
        Assert.Equal(GpuScalarType.TensorFloat32, requested.ComputeFormat);
        Assert.Equal(GpuScalarType.Float32, requested.InputStorage);
        Assert.Equal(GpuScalarType.TensorFloat32, requested.MultiplyType);
    }

    [Fact]
    public void PreserveInputType_RejectsTf32MultiplyForFloat()
    {
        var capabilities = GpuPrecisionCapabilityCatalog.Create(
            supportsFp16: false,
            GpuPrecisionImplementation.Native,
            fp16ReducesStorageBytes: false,
            fp32Implementation: GpuPrecisionImplementation.VendorLibrary,
            fp32MultiplyType: GpuScalarType.TensorFloat32,
            supportsTensorFloat32: true);
        var backend = CreateBackend(capabilities.ToArray());
        using var preserve = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var plan = Plan<float>(backend);

        Assert.Equal(GpuExecutionRoute.Cpu, plan.Route);
        Assert.Equal(GpuScalarType.Float32, plan.MultiplyType);
        Assert.Contains("PreserveInputType", plan.FallbackReason);
    }

    [Fact]
    public async Task PolicyAndAutocastScopesFlowAcrossAwaitAndRestoreNestedValues()
    {
        Assert.Same(GpuExecutionPolicy.Default, GpuExecutionPolicyScope.CurrentPolicy);
        Assert.False(AutocastScope.IsEnabled);

        using (var preserve = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve))
        using (var fp16 = new AutocastScope(PrecisionMode.Float16))
        {
            await Task.Yield();
            Assert.Same(GpuExecutionPolicy.Preserve, GpuExecutionPolicyScope.CurrentPolicy);
            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);

            using (var speed = new GpuExecutionPolicyScope(GpuExecutionPolicy.Default))
            using (var bf16 = new AutocastScope(PrecisionMode.BFloat16))
            {
                await Task.Yield();
                Assert.Same(GpuExecutionPolicy.Default, GpuExecutionPolicyScope.CurrentPolicy);
                Assert.Equal(PrecisionMode.BFloat16, AutocastScope.ActivePrecision);
            }

            Assert.Same(GpuExecutionPolicy.Preserve, GpuExecutionPolicyScope.CurrentPolicy);
            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
        }

        Assert.Same(GpuExecutionPolicy.Default, GpuExecutionPolicyScope.CurrentPolicy);
        Assert.False(AutocastScope.IsEnabled);
    }

    [Theory]
    [InlineData(PrecisionMode.BFloat16, GpuComputePreference.BFloat16)]
    [InlineData(PrecisionMode.Float8E4M3, GpuComputePreference.Float8E4M3)]
    [InlineData(PrecisionMode.Float8E5M2, GpuComputePreference.Float8E5M2)]
    public void EveryAutocastModeIsAcceptedAndObservable(
        PrecisionMode mode,
        GpuComputePreference expected)
    {
        var backend = CreateBackend(Fp32());
        using var autocast = new AutocastScope(mode);

        var plan = Plan<float>(backend);

        Assert.Equal(expected, plan.RequestedPreference);
        Assert.Equal(GpuScalarType.Float32, plan.MultiplyType);
        Assert.NotNull(plan.FallbackReason);
    }

    private static GpuComputePlan Plan<T>(IDirectGpuBackend backend)
        => GpuPrecisionPlanner.CreatePlan<T>(backend, GpuPrecisionOperation.MatMul, "test-matmul");

    private static void AssertGpuPlan<T>(IDirectGpuBackend backend, GpuScalarType expected)
    {
        var plan = Plan<T>(backend);
        Assert.Equal(GpuExecutionRoute.Gpu, plan.Route);
        Assert.Equal(typeof(T), plan.PublicType);
        Assert.Equal(expected, plan.MultiplyType);
    }

    private static IDirectGpuBackend CreateBackend(params GpuPrecisionCapability[] capabilities)
        => new PrecisionBackend(
            MockDirectGpuBackend.Create(new MockBackendState()),
            capabilities);

    private static GpuPrecisionCapability Fp64() => Capability(GpuScalarType.Float64);
    private static GpuPrecisionCapability Fp32() => Capability(GpuScalarType.Float32);
    private static GpuPrecisionCapability Fp16() => Capability(GpuScalarType.Float16);
    private static GpuPrecisionCapability Bf16() => Capability(GpuScalarType.BFloat16);
    private static GpuPrecisionCapability Fp8E4M3() => Capability(GpuScalarType.Float8E4M3);
    private static GpuPrecisionCapability Fp8E5M2() => Capability(GpuScalarType.Float8E5M2);

    private static GpuPrecisionCapability Capability(GpuScalarType type) => new(
        type,
        type,
        type is GpuScalarType.Float16 or GpuScalarType.BFloat16
            or GpuScalarType.Float8E4M3 or GpuScalarType.Float8E5M2
            ? GpuScalarType.Float32
            : type,
        type == GpuScalarType.Float64 ? GpuScalarType.Float64 : GpuScalarType.Float32,
        GpuPrecisionImplementation.Native,
        type is GpuScalarType.Float16 or GpuScalarType.BFloat16
            or GpuScalarType.Float8E4M3 or GpuScalarType.Float8E5M2);

    private sealed class PrecisionBackend : DelegatingGpuBackend, IGpuPrecisionBackend
    {
        private readonly IReadOnlyList<GpuPrecisionCapability> _capabilities;

        internal PrecisionBackend(
            IDirectGpuBackend inner,
            IReadOnlyList<GpuPrecisionCapability> capabilities)
            : base(inner)
        {
            _capabilities = capabilities;
        }

        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(
            GpuPrecisionOperation operation) => _capabilities;
    }
}
