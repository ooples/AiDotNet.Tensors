// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). One op × one shape/config = one OpCase.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public enum GpuProbeExpectation
{
    Kernel,
    MetadataOnly,
    HostContract
}

/// <summary>
/// Type-safe classification of readbacks that are inherent in a synchronous public API contract.
/// This is deliberately semantic rather than operation-name based: fixed-shape tensor results must
/// remain readback-free, while a host-owned dynamic shape or legacy CLR output may require one
/// metadata transfer.
/// </summary>
public enum GpuReadbackContract
{
    None = 0,
    DynamicResultShape,
    LegacyHostOutput
}

/// <summary>
/// Type-safe graph-capture contract for registry cases. Cases opt into a replay or rejection
/// contract explicitly; the registry-wide output-shape invariant independently examines every
/// operation that actually records a node, including cases not yet assessed for live replay.
/// </summary>
public enum GraphCaptureExpectation
{
    Required = 0,
    InputIndependent,
    DataDependentOutputShape,
    HeterogeneousInput,
    HeterogeneousOutput,
    MixedElementTypes,
    NonDeterministic,
    HostBoundary,
    Stateful,
    BackwardKernel
}

/// <summary>
/// Element-type limitation of an IEngine signature. This is independent of execution phase: a
/// backward kernel can also have heterogeneous inputs, and both facts must remain expressible.
/// </summary>
public enum GraphCaptureSignatureConstraint
{
    None = 0,
    HeterogeneousInput,
    HeterogeneousOutput,
    MixedElementTypes
}

/// <summary>Whether an input contains replayable numeric data or fixed structural metadata.</summary>
public enum GraphInputRole
{
    MutableValue = 0,
    FixedMetadata
}

/// <summary>Type-safe deterministic probes used to prove that compiled replay reads live inputs.</summary>
internal enum GraphMutationProfile
{
    RotateValues,
    AlternatingScale,
    Affine,
    ContractTowardZero,
    FirstMutableInput,
    SentinelPattern
}

/// <summary>Complete tensor-valued result shape of an <see cref="IEngine"/> operation.</summary>
public enum TensorOutputContract
{
    SingleTensor,
    HomogeneousMultiple,
    HeterogeneousMultiple
}

/// <summary>Type-safe comparison semantics for an individual numeric tensor output.</summary>
public enum TensorOutputComparison
{
    Numeric,
    WrappedRadians
}

/// <summary>Whether one homogeneous result is expected to depend on mutable tensor inputs.</summary>
public enum GraphOutputDependency
{
    MutableInput = 0,
    InputIndependent
}

internal sealed class FloatInputSnapshot
{
    internal Tensor<float> Tensor { get; }
    internal float[] InitialValues { get; }
    internal GraphInputRole Role { get; }
    internal int MutableOrdinal { get; }

    internal FloatInputSnapshot(
        Tensor<float> tensor,
        float[] initialValues,
        GraphInputRole role,
        int mutableOrdinal)
    {
        Tensor = tensor;
        InitialValues = initialValues;
        Role = role;
        MutableOrdinal = mutableOrdinal;
    }
}

/// <summary>
/// Complete result of an operation that mixes floating outputs with typed metadata tensors.
/// Empty arrays are valid; at least one metadata array must be non-empty for a heterogeneous case.
/// </summary>
public sealed class HeterogeneousTensorOutputs<T>
{
    public Tensor<T>[] Numeric { get; }
    public Tensor<int>[] Integers { get; }
    public Tensor<bool>[] Booleans { get; }

    public HeterogeneousTensorOutputs(
        Tensor<T>[] numeric,
        Tensor<int>[]? integers = null,
        Tensor<bool>[]? booleans = null)
    {
        Numeric = numeric ?? throw new ArgumentNullException(nameof(numeric));
        Integers = integers ?? Array.Empty<Tensor<int>>();
        Booleans = booleans ?? Array.Empty<Tensor<bool>>();
    }
}

/// <summary>
/// A deterministic tensor input shared between the float run, the GPU float run, and the double
/// ORACLE run. The SAME numeric samples (generated once as double[]) feed all three, so any
/// difference is the op's numerics — not the input. <see cref="F"/> materializes a fresh
/// <c>Tensor&lt;float&gt;</c>, <see cref="D"/> a fresh <c>Tensor&lt;double&gt;</c> (fresh each call
/// so an in-place op on one engine can't corrupt a sibling run).
/// </summary>
public sealed class OpInput
{
    [ThreadStatic]
    private static List<Tensor<float>>? s_floatInputCapture;

    [ThreadStatic]
    private static List<FloatInputSnapshot>? s_floatInputSnapshotCapture;

    [ThreadStatic]
    private static GraphMutationContext? s_graphMutation;

    private readonly double[] _data;
    private readonly GraphInputRole _graphInputRole;
    public int[] Shape { get; }

    private OpInput(double[] data, int[] shape, GraphInputRole graphInputRole)
    {
        _data = data;
        Shape = shape;
        _graphInputRole = graphInputRole;
    }

    /// <summary>Uniform samples in [lo, hi] from a fixed seed. Deterministic across runs/engines.</summary>
    public static OpInput Rand(int seed, int[] shape, double lo = -1.0, double hi = 1.0)
    {
        shape = ApplyShapePolicy(shape);
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = lo + rng.NextDouble() * (hi - lo);
        return new OpInput(data, (int[])shape.Clone(), GraphInputRole.MutableValue);
    }

    [ThreadStatic]
    private static Func<int[], int[]>? s_shapePolicy;

    /// <summary>
    /// Rewrites the shape of every <see cref="Rand"/> / <see cref="RandPositive"/> input built while
    /// the returned scope is open.
    /// </summary>
    /// <remarks>
    /// <para>
    /// THE REGISTRY IS SHAPE-BLIND TO ITS OWN BLIND SPOT. Counting every shape literal across the
    /// spec registry, the dimensions used are 1, 2, 3, 4, 6, 8, 16, 32 and 64 — each of them either
    /// BELOW a SIMD vector (8 floats / 4 doubles) or an exact MULTIPLE of one. 536 of the 544
    /// tensor-returning IEngine ops have a spec, and not one of them is exercised at the single
    /// condition the vectorized tails need: a dimension larger than the vector width and not a
    /// multiple of it.
    /// </para>
    /// <para>
    /// That is not hypothetical. Two kernels shipped with unguarded column tails —
    /// <c>SgemmDirectParallelMIntoTransA</c> and friends returned wrong values for every row but the
    /// last of each block, and wrote past the end of C into the GC heap — and 536 covered ops said
    /// nothing, because none of them ever asked for 13 columns.
    /// </para>
    /// <para>
    /// Rewriting shapes centrally is what makes the extra coverage a DERIVATION rather than 536
    /// hand-written duplicates: the specs stay the single source of truth for how each op is called,
    /// and the shape axis is applied over them. Ops whose shapes must agree with an inline literal
    /// the policy cannot see will throw; the sweep records those as not-applicable rather than
    /// failing, and reports the count, so the gap stays visible instead of silently passing.
    /// </para>
    /// <para>
    /// Only <see cref="Rand"/> and <see cref="RandPositive"/> are rewritten. Explicit-value inputs
    /// carry values whose length and meaning are tied to the exact shape the spec chose, so resizing
    /// them would produce nonsense rather than coverage.
    /// </para>
    /// </remarks>
    internal static IDisposable UseShapePolicy(Func<int[], int[]> policy)
    {
        if (policy is null) throw new ArgumentNullException(nameof(policy));
        var previous = s_shapePolicy;
        s_shapePolicy = policy;
        return new ShapePolicyScope(previous);
    }

    private static int[] ApplyShapePolicy(int[] shape)
    {
        var policy = s_shapePolicy;
        if (policy is null) return shape;

        var rewritten = policy(shape);
        return rewritten ?? shape;
    }

    private sealed class ShapePolicyScope : IDisposable
    {
        private Func<int[], int[]>? _previous;
        private bool _disposed;

        public ShapePolicyScope(Func<int[], int[]>? previous) => _previous = previous;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            s_shapePolicy = _previous;
            _previous = null;
        }
    }

    /// <summary>Strictly-positive samples in [lo, hi] — for log / sqrt / rsqrt / pow domains.</summary>
    public static OpInput RandPositive(int seed, int[] shape, double lo = 0.1, double hi = 4.0)
        => Rand(seed, shape, lo, hi);

    /// <summary>Explicit mutable numeric data in row-major order.</summary>
    public static OpInput From(double[] data, int[] shape)
        => new OpInput((double[])data.Clone(), shape, GraphInputRole.MutableValue);

    /// <summary>
    /// Explicit structural metadata encoded in the operation's numeric element type. Graph probes
    /// preserve these values exactly; use this only for indices, masks, and fixed selectors.
    /// </summary>
    public static OpInput FixedFrom(double[] data, int[] shape)
        => new OpInput((double[])data.Clone(), shape, GraphInputRole.FixedMetadata);

    /// <summary>
    /// Explicit alias that emphasizes that graph replay probes may transform these numeric values.
    /// </summary>
    public static OpInput MutableFrom(double[] data, int[] shape)
        => From(data, shape);

    public Tensor<float> F()
    {
        var initial = new float[_data.Length];
        for (int i = 0; i < _data.Length; i++)
            initial[i] = (float)_data[i];

        int mutableOrdinal = -1;
        if (_graphInputRole == GraphInputRole.MutableValue)
        {
            if (s_graphMutation is not null)
                mutableOrdinal = s_graphMutation.NextMutableOrdinal++;
            else if (s_floatInputSnapshotCapture is not null)
            {
                mutableOrdinal = 0;
                for (int i = 0; i < s_floatInputSnapshotCapture.Count; i++)
                    if (s_floatInputSnapshotCapture[i].Role == GraphInputRole.MutableValue)
                        mutableOrdinal++;
            }
        }

        var f = (float[])initial.Clone();
        if (s_graphMutation is not null && mutableOrdinal >= 0)
            ApplyGraphMutation(f, initial, s_graphMutation.Profile, mutableOrdinal);

        var tensor = new Tensor<float>(f, (int[])Shape.Clone());
        s_floatInputCapture?.Add(tensor);
        s_floatInputSnapshotCapture?.Add(
            new FloatInputSnapshot(tensor, initial, _graphInputRole, mutableOrdinal));
        return tensor;
    }

    /// <summary>
    /// Captures the registry inputs materialized by <see cref="F"/> during one parity run. Their materialization
    /// order is stable across CPU and GPU runs, providing a contract identity that cannot be confused by two
    /// distinct leaves with identical contents.
    /// </summary>
    internal static IDisposable CaptureFloatInputs(List<Tensor<float>> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var previous = s_floatInputCapture;
        s_floatInputCapture = destination;
        return new FloatInputCaptureScope(previous);
    }

    internal static IDisposable CaptureFloatInputSnapshots(List<FloatInputSnapshot> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var previous = s_floatInputSnapshotCapture;
        s_floatInputSnapshotCapture = destination;
        return new FloatInputSnapshotCaptureScope(previous);
    }

    internal static IDisposable UseGraphMutation(GraphMutationProfile profile)
    {
        var previous = s_graphMutation;
        s_graphMutation = new GraphMutationContext(profile);
        return new GraphMutationScope(previous);
    }

    internal static void ApplyGraphMutation(
        float[] destination,
        float[] source,
        GraphMutationProfile profile,
        int mutableOrdinal)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            float value = source[i];
            destination[i] = profile switch
            {
                GraphMutationProfile.RotateValues =>
                    source.Length == 0 ? value : source[(i + 1) % source.Length],
                GraphMutationProfile.AlternatingScale =>
                    value * ((i & 1) == 0 ? 0.625f : 1.375f),
                GraphMutationProfile.Affine =>
                    value * 1.125f + 0.0625f + (i % 5) * 0.0078125f,
                GraphMutationProfile.ContractTowardZero => value * 0.5f,
                GraphMutationProfile.FirstMutableInput => mutableOrdinal == 0
                    ? value * 0.75f + 0.25f + (i % 3) * 0.125f
                    : value,
                GraphMutationProfile.SentinelPattern => (i % 3) == 0
                    ? 0.5f
                    : value * 0.875f - 0.125f,
                _ => throw new ArgumentOutOfRangeException(nameof(profile), profile, null)
            };
        }
    }

    private sealed class GraphMutationContext
    {
        internal GraphMutationProfile Profile { get; }
        internal int NextMutableOrdinal { get; set; }

        internal GraphMutationContext(GraphMutationProfile profile) => Profile = profile;
    }

    private sealed class FloatInputCaptureScope : IDisposable
    {
        private List<Tensor<float>>? _previous;

        public FloatInputCaptureScope(List<Tensor<float>>? previous) => _previous = previous;

        public void Dispose()
        {
            s_floatInputCapture = _previous;
            _previous = null;
        }
    }

    private sealed class FloatInputSnapshotCaptureScope : IDisposable
    {
        private List<FloatInputSnapshot>? _previous;

        internal FloatInputSnapshotCaptureScope(List<FloatInputSnapshot>? previous) => _previous = previous;

        public void Dispose()
        {
            s_floatInputSnapshotCapture = _previous;
            _previous = null;
        }
    }

    private sealed class GraphMutationScope : IDisposable
    {
        private GraphMutationContext? _previous;

        internal GraphMutationScope(GraphMutationContext? previous) => _previous = previous;

        public void Dispose()
        {
            s_graphMutation = _previous;
            _previous = null;
        }
    }

    public Tensor<double> D() => new Tensor<double>((double[])_data.Clone(), (int[])Shape.Clone());

    /// <summary>Complex float tensor whose real parts are this input's samples and imaginary parts
    /// are <paramref name="imag"/>'s samples (both must share this shape). For NativeComplex* ops.</summary>
    public Tensor<Complex<float>> CF(OpInput imag)
    {
        if (imag._data.Length != _data.Length)
            throw new ArgumentException($"imag length {imag._data.Length} must match this input's length {_data.Length}.", nameof(imag));
        var c = new Complex<float>[_data.Length];
        for (int i = 0; i < _data.Length; i++) c[i] = new Complex<float>((float)_data[i], (float)imag._data[i]);
        return new Tensor<Complex<float>>(c, (int[])Shape.Clone());
    }

    /// <summary>Complex double tensor (the oracle counterpart of <see cref="CF"/>).</summary>
    public Tensor<Complex<double>> CD(OpInput imag)
    {
        if (imag._data.Length != _data.Length)
            throw new ArgumentException($"imag length {imag._data.Length} must match this input's length {_data.Length}.", nameof(imag));
        var c = new Complex<double>[_data.Length];
        for (int i = 0; i < _data.Length; i++) c[i] = new Complex<double>(_data[i], imag._data[i]);
        return new Tensor<Complex<double>>(c, (int[])Shape.Clone());
    }
}

/// <summary>
/// One parity case: an op invoked on a given engine, in float and in double, plus tolerances.
/// The delegates take the <see cref="IEngine"/> so the harness runs the identical closure on the
/// CPU engine, the GPU engine, and (in double) the CPU oracle. Backward delegates are optional;
/// when present the harness parity-checks the gradient too.
/// </summary>
public sealed class OpCase
{
    /// <summary>Display id, e.g. "Softmax[2,8]". Doubles as the emitted test name.</summary>
    public string Name { get; }

    /// <summary>The IEngine method this case exercises, e.g. "GELU", "TensorMatMul", "Conv2D".
    /// Used by the inventory/coverage report to mark that op covered across the full surface.</summary>
    public string OpMethod { get; }

    /// <summary>Coarse bucket (arithmetic, matmul, activation, reduction, norm, conv, attention, loss, shape).</summary>
    public string Category { get; }

    public Func<IEngine, Tensor<float>> RunFloat { get; }
    public Func<IEngine, Tensor<double>> RunDouble { get; }
    public ParityTol Fwd { get; }

    /// <summary>
    /// Complete tensor-valued result contract for operations with multiple homogeneous outputs.
    /// The ordinary delegates remain the primary output used by coverage/residency probes; these
    /// delegates prevent auxiliary outputs or backward gradients from being silently discarded.
    /// </summary>
    public Func<IEngine, Tensor<float>[]>? RunFloatOutputs { get; init; }
    public Func<IEngine, Tensor<double>[]>? RunDoubleOutputs { get; init; }
    public bool HasMultipleOutputs => RunFloatOutputs is not null && RunDoubleOutputs is not null;

    public Func<IEngine, HeterogeneousTensorOutputs<float>>? RunFloatHeterogeneousOutputs { get; init; }
    public Func<IEngine, HeterogeneousTensorOutputs<double>>? RunDoubleHeterogeneousOutputs { get; init; }
    public bool HasHeterogeneousOutputs =>
        RunFloatHeterogeneousOutputs is not null && RunDoubleHeterogeneousOutputs is not null;

    /// <summary>
    /// Typed declaration of whether the operation returns one tensor, multiple tensors of the
    /// generic element type, or a heterogeneous tuple such as values plus integer indices.
    /// </summary>
    public TensorOutputContract TensorOutputContract { get; init; }

    /// <summary>
    /// Generated overload identity. Required only when an IEngine method has more than one
    /// multi-output overload; the default cannot satisfy an overload-specific invariant.
    /// </summary>
    public TensorOutputOverload TensorOutputOverload { get; init; }

    /// <summary>
    /// Generated overload identity for an IEngine signature that mixes tensor element types.
    /// Required when the method name is overloaded so a classification cannot be satisfied by a
    /// case exercising a different overload.
    /// </summary>
    public GraphCaptureSignatureOverload GraphCaptureSignatureOverload { get; init; }

    public GraphCaptureSignatureConstraint GraphCaptureSignatureConstraint { get; internal set; }

    /// <summary>
    /// Per-output graph dependency contract. Omitted entries default to
    /// <see cref="GraphOutputDependency.MutableInput"/>.
    /// </summary>
    public GraphOutputDependency[]? GraphOutputDependencies { get; init; }

    /// <summary>
    /// Optional per-numeric-output semantics. Omitted entries use ordinary numeric comparison;
    /// when supplied, the list must describe every numeric output in order.
    /// </summary>
    public IReadOnlyList<TensorOutputComparison>? TensorOutputComparisons { get; init; }

    public Func<IEngine, Tensor<float>>? RunFloatGrad { get; }
    public Func<IEngine, Tensor<double>>? RunDoubleGrad { get; }
    public ParityTol BwdTol { get; }
    public bool HasBackward => RunFloatGrad is not null && RunDoubleGrad is not null;

    /// <summary>Non-null marks a CONFIRMED, tracked CPU/GPU divergence (a real cross-engine bug the
    /// scaffold found that isn't fixed yet). The harness records it and SKIPS with this reason
    /// instead of failing the build — but if the op ever starts passing, the harness fails to prompt
    /// removing the marker, so the fix is noticed. Keeps CI green without hiding the finding.</summary>
    public string? KnownDivergence { get; init; }

    /// <summary>Set when the op's GPU kernel is not just numerically divergent but actively UNSAFE —
    /// it crashes / hangs the host process or corrupts GPU state for subsequent ops (e.g. an OpenCL
    /// kernel that over-allocates private memory and errors the command queue). Unlike
    /// <see cref="KnownDivergence"/>, which still executes both engines before skipping, a GpuUnsafe
    /// op is SKIPPED before the GPU is ever touched, so it can't poison the run. Records the finding
    /// (visible in the report) without executing the crashing kernel. Requires <see cref="KnownDivergence"/>
    /// to carry the reason.</summary>
    public bool GpuUnsafe { get; init; }

    /// <summary>
    /// Metadata-only operations intentionally launch no kernel. The residency probe
    /// tracks these separately; dedicated tests verify they preserve a resident buffer.
    /// </summary>
    public GpuProbeExpectation GpuProbeExpectation { get; init; }

    /// <summary>Graph-capture behavior required by the generated live-input replay sweep.
    /// Forward capture is the fail-closed default; non-forward contracts must opt out explicitly.</summary>
    private GraphCaptureExpectation _graphCaptureExpectation = GraphCaptureExpectation.Required;

    public GraphCaptureExpectation GraphCaptureExpectation
    {
        get => _graphCaptureExpectation;
        internal set
        {
            _graphCaptureExpectation = value;
            GraphCaptureSignatureConstraint inferred = value switch
            {
                GraphCaptureExpectation.HeterogeneousInput => GraphCaptureSignatureConstraint.HeterogeneousInput,
                GraphCaptureExpectation.HeterogeneousOutput => GraphCaptureSignatureConstraint.HeterogeneousOutput,
                GraphCaptureExpectation.MixedElementTypes => GraphCaptureSignatureConstraint.MixedElementTypes,
                _ => GraphCaptureSignatureConstraint.None
            };
            if (inferred != GraphCaptureSignatureConstraint.None &&
                GraphCaptureSignatureConstraint == GraphCaptureSignatureConstraint.None)
                GraphCaptureSignatureConstraint = inferred;
        }
    }

    /// <summary>
    /// Minimum dispatches required before the residency probe accepts this case as GPU-covered.
    /// Cases that project a non-floating or multi-output result through another GPU primitive use
    /// two so that the projection kernel cannot hide a CPU fallback in the operation under test.
    /// </summary>
    public int GpuMinimumKernelLaunches { get; }

    /// <summary>Required explanation when <see cref="GpuProbeExpectation"/> is
    /// <see cref="Engines.OpParity.GpuProbeExpectation.HostContract"/>.</summary>
    public string? GpuHostContractReason { get; init; }

    /// <summary>
    /// Typed public-contract reason for an unavoidable metadata transfer. Tensor-valued
    /// alternatives must use separate zero-readback registry cases.
    /// </summary>
    public GpuReadbackContract GpuReadbackContract { get; init; }

    public int GpuAllowedReadbacks => GpuReadbackContract switch
    {
        GpuReadbackContract.None => 0,
        GpuReadbackContract.DynamicResultShape => 1,
        GpuReadbackContract.LegacyHostOutput => 1,
        _ => throw new InvalidOperationException($"Unknown GPU readback contract: {GpuReadbackContract}.")
    };

    public string GpuReadbackDescription => GpuReadbackContract switch
    {
        GpuReadbackContract.None => "No internal readback is permitted.",
        GpuReadbackContract.DynamicResultShape =>
            "One metadata transfer is required to construct the synchronous host-owned Tensor.Shape.",
        GpuReadbackContract.LegacyHostOutput =>
            "One transfer is required by the legacy host-owned output; a tensor-valued overload is the resident alternative.",
        _ => throw new InvalidOperationException($"Unknown GPU readback contract: {GpuReadbackContract}.")
    };

    public OpCase(
        string name, string category,
        Func<IEngine, Tensor<float>> runFloat,
        Func<IEngine, Tensor<double>> runDouble,
        ParityTol fwd,
        Func<IEngine, Tensor<float>>? runFloatGrad = null,
        Func<IEngine, Tensor<double>>? runDoubleGrad = null,
        ParityTol bwdTol = default,
        string? opMethod = null,
        int gpuMinimumKernelLaunches = 1,
        GraphCaptureExpectation graphCaptureExpectation = GraphCaptureExpectation.Required,
        GraphCaptureSignatureConstraint graphCaptureSignatureConstraint = GraphCaptureSignatureConstraint.None,
        GraphCaptureSignatureOverload graphCaptureSignatureOverload = GraphCaptureSignatureOverload.Unspecified)
    {
        Name = name;
        // Default the covered op method to the leading identifier of the display name
        // (e.g. "Softmax[4,16]" -> "Softmax") when not given explicitly.
        OpMethod = opMethod ?? LeadingIdentifier(name);
        Category = category;
        RunFloat = runFloat;
        RunDouble = runDouble;
        Fwd = fwd;
        RunFloatGrad = runFloatGrad;
        RunDoubleGrad = runDoubleGrad;
        BwdTol = bwdTol;
        GpuMinimumKernelLaunches = gpuMinimumKernelLaunches;
        GraphCaptureExpectation = graphCaptureExpectation;
        if (graphCaptureSignatureConstraint != GraphCaptureSignatureConstraint.None)
            GraphCaptureSignatureConstraint = graphCaptureSignatureConstraint;
        GraphCaptureSignatureOverload = graphCaptureSignatureOverload;
    }

    private static string LeadingIdentifier(string name)
    {
        int i = 0;
        while (i < name.Length && (char.IsLetterOrDigit(name[i]) || name[i] == '_')) i++;
        return i > 0 ? name.Substring(0, i) : name;
    }
}
#endif
