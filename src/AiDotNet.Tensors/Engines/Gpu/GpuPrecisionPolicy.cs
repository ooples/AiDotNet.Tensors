using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>Controls whether GPU execution prioritizes throughput or the declared tensor type.</summary>
public enum GpuAccuracyMode
{
    /// <summary>
    /// Prefer GPU execution and permit conversion through a backend-supported floating-point format.
    /// The returned tensor still has the caller's declared element type.
    /// </summary>
    SpeedFirst,

    /// <summary>
    /// Preserve the declared tensor type. A backend without a matching native route must use the CPU
    /// implementation rather than silently narrowing the computation.
    /// </summary>
    PreserveInputType,
}

/// <summary>
/// Preferred logical compute format for eligible GPU operations. The selected plan separately
/// reports physical input storage, multiply, accumulation, and output formats.
/// </summary>
public enum GpuComputePreference
{
    /// <summary>Let the precision planner select from the backend's measured and advertised capabilities.</summary>
    Auto,
    /// <summary>64-bit IEEE floating point.</summary>
    Float64,
    /// <summary>32-bit IEEE floating point.</summary>
    Float32,
    /// <summary>NVIDIA/AMD tensor-float compute over FP32 storage.</summary>
    TensorFloat32,
    /// <summary>Brain floating point with FP32-range exponent.</summary>
    BFloat16,
    /// <summary>IEEE binary16.</summary>
    Float16,
    /// <summary>8-bit E4M3 floating point.</summary>
    Float8E4M3,
    /// <summary>8-bit E5M2 floating point.</summary>
    Float8E5M2,
}

/// <summary>Behavior when an explicitly requested GPU format is unavailable.</summary>
public enum GpuPrecisionFallbackBehavior
{
    /// <summary>Use a safe, supported higher-precision format and report the decision.</summary>
    UseHigherPrecision,
    /// <summary>Reject the operation before allocating or launching a kernel.</summary>
    Throw,
}

/// <summary>Physical scalar formats understood by the GPU precision planner.</summary>
public enum GpuScalarType
{
    /// <summary>A generic/non-floating public type without a matching physical GPU representation.</summary>
    Generic,
    /// <summary>64-bit IEEE floating point.</summary>
    Float64,
    /// <summary>32-bit IEEE floating point.</summary>
    Float32,
    /// <summary>TensorFloat-32 multiply over FP32 storage.</summary>
    TensorFloat32,
    /// <summary>Brain floating point.</summary>
    BFloat16,
    /// <summary>IEEE binary16.</summary>
    Float16,
    /// <summary>8-bit E4M3 floating point.</summary>
    Float8E4M3,
    /// <summary>8-bit E5M2 floating point.</summary>
    Float8E5M2,
}

/// <summary>Operation families whose supported formats can differ on the same device.</summary>
public enum GpuPrecisionOperation
{
    /// <summary>An operation without a specialized mixed-precision contract.</summary>
    General,
    /// <summary>Element-wise arithmetic or activation.</summary>
    Elementwise,
    /// <summary>Element-wise addition with a typed reduced-precision route.</summary>
    Add,
    /// <summary>Rectified linear activation with a typed reduced-precision route.</summary>
    Relu,
    /// <summary>Gaussian error linear activation with a typed reduced-precision route.</summary>
    Gelu,
    /// <summary>Matrix multiplication.</summary>
    MatMul,
    /// <summary>Matrix multiplication with a logically transposed right operand.</summary>
    MatMulTransposed,
    /// <summary>Batched matrix multiplication.</summary>
    BatchMatMul,
    /// <summary>Convolution.</summary>
    Convolution,
    /// <summary>Reduction or numerically-sensitive normalization.</summary>
    Reduction,
}

/// <summary>How a backend realizes a reported precision capability.</summary>
public enum GpuPrecisionImplementation
{
    /// <summary>Native device instructions and native-width storage.</summary>
    Native,
    /// <summary>A vendor math library such as cuBLAS, hipBLAS, or MPS.</summary>
    VendorLibrary,
    /// <summary>A composed route using both a backend-native kernel and a vendor math library.</summary>
    Composite,
    /// <summary>Reduced-width packed storage decoded by a shader with wider arithmetic.</summary>
    Packed,
    /// <summary>Values are quantized but retained in wider physical storage.</summary>
    Emulated,
}

/// <summary>Whether a selected plan executes on the GPU or preserves semantics on the CPU.</summary>
public enum GpuExecutionRoute
{
    /// <summary>Execute through the selected GPU backend.</summary>
    Gpu,
    /// <summary>Execute through the CPU engine.</summary>
    Cpu,
}

/// <summary>Immutable user policy for generic GPU conversion and precision selection.</summary>
public sealed class GpuExecutionPolicy
{
    /// <summary>The default speed-first policy with automatic format selection.</summary>
    public static GpuExecutionPolicy Default { get; } = new(
        GpuAccuracyMode.SpeedFirst,
        GpuComputePreference.Auto,
        GpuPrecisionFallbackBehavior.UseHigherPrecision);

    /// <summary>A policy suitable for exactness-sensitive tests and numerical validation.</summary>
    public static GpuExecutionPolicy Preserve { get; } = new(
        GpuAccuracyMode.PreserveInputType,
        GpuComputePreference.Auto,
        GpuPrecisionFallbackBehavior.UseHigherPrecision);

    /// <summary>Creates a precision policy.</summary>
    public GpuExecutionPolicy(
        GpuAccuracyMode accuracyMode = GpuAccuracyMode.SpeedFirst,
        GpuComputePreference computePreference = GpuComputePreference.Auto,
        GpuPrecisionFallbackBehavior fallbackBehavior = GpuPrecisionFallbackBehavior.UseHigherPrecision)
    {
        AccuracyMode = accuracyMode;
        ComputePreference = computePreference;
        FallbackBehavior = fallbackBehavior;
    }

    /// <summary>Gets the requested accuracy behavior.</summary>
    public GpuAccuracyMode AccuracyMode { get; }

    /// <summary>Gets the preferred compute format.</summary>
    public GpuComputePreference ComputePreference { get; }

    /// <summary>Gets the unsupported-format behavior.</summary>
    public GpuPrecisionFallbackBehavior FallbackBehavior { get; }
}

/// <summary>
/// Async-flowing execution-policy scope. Preservation is deliberately independent from autocast so tests can
/// preserve <c>T</c> without pretending that preservation is another low-precision format.
/// </summary>
public sealed class GpuExecutionPolicyScope : IDisposable
{
    private static readonly AsyncLocal<GpuExecutionPolicyScope?> s_current = new();
    private readonly GpuExecutionPolicyScope? _previous;
    private bool _disposed;

    /// <summary>Creates and activates a scope.</summary>
    public GpuExecutionPolicyScope(GpuExecutionPolicy policy)
    {
        Policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _previous = s_current.Value;
        s_current.Value = this;
    }

    /// <summary>Gets the current scope, or <see langword="null"/>.</summary>
    public static GpuExecutionPolicyScope? Current => s_current.Value;

    /// <summary>Gets the active policy, defaulting to speed-first automatic selection.</summary>
    public static GpuExecutionPolicy CurrentPolicy => Current?.Policy ?? GpuExecutionPolicy.Default;

    /// <summary>Gets this scope's policy.</summary>
    public GpuExecutionPolicy Policy { get; }

    /// <summary>Restores the containing scope.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        s_current.Value = _previous;
    }
}

/// <summary>A physical precision combination supported for one operation family.</summary>
public sealed class GpuPrecisionCapability
{
    /// <summary>Creates a backend precision capability.</summary>
    public GpuPrecisionCapability(
        GpuScalarType inputStorage,
        GpuScalarType multiplyType,
        GpuScalarType accumulatorType,
        GpuScalarType outputStorage,
        GpuPrecisionImplementation implementation,
        bool reducesStorageBytes,
        GpuScalarType? computeFormat = null)
    {
        InputStorage = inputStorage;
        MultiplyType = multiplyType;
        AccumulatorType = accumulatorType;
        OutputStorage = outputStorage;
        Implementation = implementation;
        ReducesStorageBytes = reducesStorageBytes;
        ComputeFormat = computeFormat ?? inputStorage;
    }

    /// <summary>Gets the logical format selected by Auto or explicit autocast.</summary>
    public GpuScalarType ComputeFormat { get; }
    /// <summary>Gets the physical input storage format.</summary>
    public GpuScalarType InputStorage { get; }
    /// <summary>Gets the multiply format used by the kernel.</summary>
    public GpuScalarType MultiplyType { get; }
    /// <summary>Gets the accumulator format.</summary>
    public GpuScalarType AccumulatorType { get; }
    /// <summary>Gets the physical result storage format.</summary>
    public GpuScalarType OutputStorage { get; }
    /// <summary>Gets how the backend realizes this capability.</summary>
    public GpuPrecisionImplementation Implementation { get; }
    /// <summary>Gets whether this route actually reduces device storage or transfer bytes.</summary>
    public bool ReducesStorageBytes { get; }
}

/// <summary>Optional capability surface implemented by every in-tree DirectGPU backend.</summary>
public interface IGpuPrecisionBackend
{
    /// <summary>Returns the physical precision combinations supported for <paramref name="operation"/>.</summary>
    IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(GpuPrecisionOperation operation);
}

/// <summary>An immutable, observable precision decision for one operation.</summary>
public sealed class GpuComputePlan
{
    internal GpuComputePlan(
        GpuExecutionRoute route,
        string backend,
        string operation,
        Type publicType,
        GpuComputePreference requestedPreference,
        GpuScalarType computeFormat,
        GpuScalarType inputStorage,
        GpuScalarType multiplyType,
        GpuScalarType accumulatorType,
        GpuScalarType outputStorage,
        GpuPrecisionImplementation? implementation,
        bool reducesStorageBytes,
        string? fallbackReason)
    {
        Route = route;
        Backend = backend;
        Operation = operation;
        PublicType = publicType;
        RequestedPreference = requestedPreference;
        ComputeFormat = computeFormat;
        InputStorage = inputStorage;
        MultiplyType = multiplyType;
        AccumulatorType = accumulatorType;
        OutputStorage = outputStorage;
        Implementation = implementation;
        ReducesStorageBytes = reducesStorageBytes;
        FallbackReason = fallbackReason;
    }

    /// <summary>Gets the selected CPU/GPU route.</summary>
    public GpuExecutionRoute Route { get; }
    /// <summary>Gets the backend name considered by the planner.</summary>
    public string Backend { get; }
    /// <summary>Gets the operation name.</summary>
    public string Operation { get; }
    /// <summary>Gets the caller-visible tensor element type.</summary>
    public Type PublicType { get; }
    /// <summary>Gets the requested compute preference.</summary>
    public GpuComputePreference RequestedPreference { get; }
    /// <summary>Gets the selected logical autocast format.</summary>
    public GpuScalarType ComputeFormat { get; }
    /// <summary>Gets the selected physical input storage.</summary>
    public GpuScalarType InputStorage { get; }
    /// <summary>Gets the selected multiply type.</summary>
    public GpuScalarType MultiplyType { get; }
    /// <summary>Gets the selected accumulator type.</summary>
    public GpuScalarType AccumulatorType { get; }
    /// <summary>Gets the selected physical output storage.</summary>
    public GpuScalarType OutputStorage { get; }
    /// <summary>Gets how the backend implements the selected format.</summary>
    public GpuPrecisionImplementation? Implementation { get; }
    /// <summary>Gets whether the selected route actually reduces device storage bytes.</summary>
    public bool ReducesStorageBytes { get; }
    /// <summary>Gets why the requested format or route was not used.</summary>
    public string? FallbackReason { get; }
}

/// <summary>Per-async-flow diagnostics for the precision route that actually executed.</summary>
public static class GpuPrecisionDiagnostics
{
    private static readonly AsyncLocal<GpuComputePlan?> s_lastPlan = new();

    /// <summary>Raised after an operation selects and commits to an execution route.</summary>
    public static event Action<GpuComputePlan>? PlanExecuted;

    /// <summary>Gets the most recently executed plan in the current async flow.</summary>
    public static GpuComputePlan? LastPlan => s_lastPlan.Value;

    /// <summary>Clears the current async flow's last plan.</summary>
    public static void Clear() => s_lastPlan.Value = null;

    internal static void Publish(GpuComputePlan plan)
    {
        s_lastPlan.Value = plan;
        PlanExecuted?.Invoke(plan);
    }
}

/// <summary>Central precision planner shared by generic conversion and autocast dispatch.</summary>
public static class GpuPrecisionPlanner
{
    /// <summary>Creates a plan for a generic tensor operation without launching work.</summary>
    public static GpuComputePlan CreatePlan<T>(
        IDirectGpuBackend backend,
        GpuPrecisionOperation operation,
        string operationName)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (string.IsNullOrWhiteSpace(operationName)) throw new ArgumentException("Operation name required.", nameof(operationName));

        var policy = GpuExecutionPolicyScope.CurrentPolicy;
        var requested = AutocastScope.IsEnabled
            ? FromPrecisionMode(AutocastScope.ActivePrecision)
            : policy.ComputePreference;
        var sourceType = ScalarTypeFor(typeof(T));
        var capabilities = GetCapabilities(backend, operation);

        if (policy.AccuracyMode == GpuAccuracyMode.PreserveInputType)
        {
            var exact = FindPreserving(capabilities, sourceType);
            if (exact is not null)
                return FromCapability(backend, operationName, typeof(T), requested, exact, null);

            return CpuPlan(backend, operationName, typeof(T), requested,
                $"{backend.BackendName} has no {sourceType} {operation} route required by PreserveInputType.");
        }

        var desired = requested == GpuComputePreference.Auto
            ? AutomaticTypeFor(sourceType)
            : ScalarTypeFor(requested);
        var selected = Find(capabilities, desired);
        if (selected is not null)
        {
            string? conversionReason = requested == GpuComputePreference.Auto && sourceType != desired
                ? $"SpeedFirst Auto converts public {sourceType} through {desired}."
                : null;
            return FromCapability(backend, operationName, typeof(T), requested, selected, conversionReason);
        }

        if (policy.FallbackBehavior == GpuPrecisionFallbackBehavior.Throw)
            throw new NotSupportedException(
                $"{backend.BackendName} does not support requested {desired} for {operationName}.");

        foreach (var fallback in SafeFallbacks(desired))
        {
            selected = Find(capabilities, fallback);
            if (selected is not null)
            {
                return FromCapability(backend, operationName, typeof(T), requested, selected,
                    $"Requested {desired} is unavailable for {operation}; using {selected.ComputeFormat}.");
            }
        }

        return CpuPlan(backend, operationName, typeof(T), requested,
            $"{backend.BackendName} exposes no eligible GPU precision for {operation}.");
    }

    internal static IReadOnlyList<GpuPrecisionCapability> GetCapabilities(
        IDirectGpuBackend backend,
        GpuPrecisionOperation operation)
    {
        if (backend is IGpuPrecisionBackend precisionBackend)
            return precisionBackend.GetPrecisionCapabilities(operation);

        // Compatibility adapter for third-party backends compiled before this capability surface existed.
        // Their IDirectGpuBackend contract is FP32, so advertising more would be unsafe.
        return GpuPrecisionCapabilityCatalog.Float32Only;
    }

    internal static GpuComputePlan CpuFallback<T>(
        IDirectGpuBackend backend,
        string operationName,
        GpuComputePreference requested,
        string reason)
        => CpuPlan(backend, operationName, typeof(T), requested, reason);

    private static GpuComputePlan FromCapability(
        IDirectGpuBackend backend,
        string operationName,
        Type publicType,
        GpuComputePreference requested,
        GpuPrecisionCapability capability,
        string? fallbackReason)
        => new(
            GpuExecutionRoute.Gpu,
            backend.BackendName,
            operationName,
            publicType,
            requested,
            capability.ComputeFormat,
            capability.InputStorage,
            capability.MultiplyType,
            capability.AccumulatorType,
            capability.OutputStorage,
            capability.Implementation,
            capability.ReducesStorageBytes,
            fallbackReason);

    private static GpuComputePlan CpuPlan(
        IDirectGpuBackend backend,
        string operationName,
        Type publicType,
        GpuComputePreference requested,
        string reason)
        => new(
            GpuExecutionRoute.Cpu,
            backend.BackendName,
            operationName,
            publicType,
            requested,
            ScalarTypeFor(publicType),
            ScalarTypeFor(publicType),
            ScalarTypeFor(publicType),
            ScalarTypeFor(publicType),
            ScalarTypeFor(publicType),
            null,
            false,
            reason);

    private static GpuPrecisionCapability? Find(
        IReadOnlyList<GpuPrecisionCapability> capabilities,
        GpuScalarType computeFormat)
    {
        for (int i = 0; i < capabilities.Count; i++)
        {
            if (capabilities[i].ComputeFormat == computeFormat)
                return capabilities[i];
        }
        return null;
    }

    private static GpuPrecisionCapability? FindPreserving(
        IReadOnlyList<GpuPrecisionCapability> capabilities,
        GpuScalarType sourceType)
    {
        for (int i = 0; i < capabilities.Count; i++)
        {
            var capability = capabilities[i];
            if (capability.ComputeFormat == sourceType
                && capability.InputStorage == sourceType
                && DoesNotNarrow(sourceType, capability.MultiplyType)
                && DoesNotNarrow(sourceType, capability.AccumulatorType)
                && DoesNotNarrow(sourceType, capability.OutputStorage))
            {
                return capability;
            }
        }
        return null;
    }

    private static bool DoesNotNarrow(GpuScalarType sourceType, GpuScalarType physicalType)
        => sourceType switch
        {
            GpuScalarType.Float64 => physicalType == GpuScalarType.Float64,
            GpuScalarType.Float32 => physicalType is GpuScalarType.Float32 or GpuScalarType.Float64,
            GpuScalarType.BFloat16 => physicalType is GpuScalarType.BFloat16
                or GpuScalarType.Float32 or GpuScalarType.Float64,
            GpuScalarType.Float16 => physicalType is GpuScalarType.Float16
                or GpuScalarType.Float32 or GpuScalarType.Float64,
            GpuScalarType.Float8E4M3 => physicalType is GpuScalarType.Float8E4M3
                or GpuScalarType.BFloat16 or GpuScalarType.Float16
                or GpuScalarType.Float32 or GpuScalarType.Float64,
            GpuScalarType.Float8E5M2 => physicalType is GpuScalarType.Float8E5M2
                or GpuScalarType.BFloat16 or GpuScalarType.Float16
                or GpuScalarType.Float32 or GpuScalarType.Float64,
            GpuScalarType.TensorFloat32 => physicalType is GpuScalarType.TensorFloat32
                or GpuScalarType.Float32 or GpuScalarType.Float64,
            _ => physicalType == sourceType,
        };

    private static GpuScalarType AutomaticTypeFor(GpuScalarType sourceType)
    {
        // Preserve an explicitly reduced public format when the backend supports it. All other generic T values
        // retain the package's established FP32 GPU boundary until a hardware-specific measured profile opts into
        // a narrower automatic route. This avoids turning an unmeasured assumption into a global default.
        return sourceType is GpuScalarType.Float16 or GpuScalarType.BFloat16
            or GpuScalarType.Float8E4M3 or GpuScalarType.Float8E5M2
            ? sourceType
            : GpuScalarType.Float32;
    }

    private static IEnumerable<GpuScalarType> SafeFallbacks(GpuScalarType requested)
    {
        switch (requested)
        {
            case GpuScalarType.Float8E4M3:
            case GpuScalarType.Float8E5M2:
                yield return GpuScalarType.BFloat16;
                yield return GpuScalarType.Float16;
                yield return GpuScalarType.Float32;
                break;
            case GpuScalarType.BFloat16:
            case GpuScalarType.Float16:
            case GpuScalarType.TensorFloat32:
            case GpuScalarType.Generic:
                yield return GpuScalarType.Float32;
                break;
            case GpuScalarType.Float64:
                // There is no higher-precision floating fallback. An explicit FP64 request must
                // route to an exact CPU implementation when the device cannot execute it.
                break;
            case GpuScalarType.Float32:
                break;
        }
    }

    private static GpuComputePreference FromPrecisionMode(PrecisionMode mode) => mode switch
    {
        PrecisionMode.Float32 => GpuComputePreference.Float32,
        PrecisionMode.Float16 => GpuComputePreference.Float16,
        PrecisionMode.BFloat16 => GpuComputePreference.BFloat16,
        PrecisionMode.Float8E4M3 => GpuComputePreference.Float8E4M3,
        PrecisionMode.Float8E5M2 => GpuComputePreference.Float8E5M2,
        _ => GpuComputePreference.Auto,
    };

    private static GpuScalarType ScalarTypeFor(GpuComputePreference preference) => preference switch
    {
        GpuComputePreference.Float64 => GpuScalarType.Float64,
        GpuComputePreference.Float32 => GpuScalarType.Float32,
        GpuComputePreference.TensorFloat32 => GpuScalarType.TensorFloat32,
        GpuComputePreference.BFloat16 => GpuScalarType.BFloat16,
        GpuComputePreference.Float16 => GpuScalarType.Float16,
        GpuComputePreference.Float8E4M3 => GpuScalarType.Float8E4M3,
        GpuComputePreference.Float8E5M2 => GpuScalarType.Float8E5M2,
        _ => GpuScalarType.Float32,
    };

    private static GpuScalarType ScalarTypeFor(Type type)
    {
        if (type == typeof(double)) return GpuScalarType.Float64;
        if (type == typeof(float)) return GpuScalarType.Float32;
        if (type == typeof(Half)) return GpuScalarType.Float16;
        if (type == typeof(NumericOperations.BFloat16)) return GpuScalarType.BFloat16;
        if (type == typeof(NumericOperations.Float8E4M3)) return GpuScalarType.Float8E4M3;
        if (type == typeof(NumericOperations.Float8E5M2)) return GpuScalarType.Float8E5M2;
        return GpuScalarType.Generic;
    }
}

/// <summary>Shared immutable capability sets used by backend implementations.</summary>
internal static class GpuPrecisionCapabilityCatalog
{
    internal static readonly IReadOnlyList<GpuPrecisionCapability> Float32Only =
        new[]
        {
            new GpuPrecisionCapability(
                GpuScalarType.Float32,
                GpuScalarType.Float32,
                GpuScalarType.Float32,
                GpuScalarType.Float32,
                GpuPrecisionImplementation.Native,
                false),
        };

    internal static IReadOnlyList<GpuPrecisionCapability> Create(
        bool supportsFp16,
        GpuPrecisionImplementation fp16Implementation,
        bool fp16ReducesStorageBytes,
        GpuScalarType fp16OutputStorage = GpuScalarType.Float32,
        GpuScalarType fp16MultiplyType = GpuScalarType.Float16,
        GpuPrecisionImplementation fp32Implementation = GpuPrecisionImplementation.Native,
        GpuScalarType fp32MultiplyType = GpuScalarType.Float32,
        bool supportsTensorFloat32 = false)
    {
        var capabilities = new List<GpuPrecisionCapability>(supportsFp16 && supportsTensorFloat32 ? 3 : 2)
        {
            new(
                GpuScalarType.Float32,
                fp32MultiplyType,
                GpuScalarType.Float32,
                GpuScalarType.Float32,
                fp32Implementation,
                false,
                GpuScalarType.Float32),
        };
        if (supportsTensorFloat32)
        {
            capabilities.Add(new GpuPrecisionCapability(
                GpuScalarType.Float32,
                GpuScalarType.TensorFloat32,
                GpuScalarType.Float32,
                GpuScalarType.Float32,
                fp32Implementation,
                false,
                GpuScalarType.TensorFloat32));
        }
        if (supportsFp16)
        {
            capabilities.Add(new GpuPrecisionCapability(
                GpuScalarType.Float16,
                fp16MultiplyType,
                GpuScalarType.Float32,
                fp16OutputStorage,
                fp16Implementation,
                fp16ReducesStorageBytes,
                GpuScalarType.Float16));
        }
        return capabilities;
    }
}
