using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;

namespace AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

/// <summary>
/// An operation family that the engine resolves through the tuned-kernel registry. Every engine op that has more
/// than one correct implementation (a vendor library algorithm, a generated or hand-written kernel, an externally
/// evolved kernel) dispatches through one <see cref="TunedKernelSlot{TArgs}"/> per family and device, so a
/// candidate proven correct and faster for a shape class is picked up by every model with no model or layer change.
/// </summary>
public enum TunedKernelOp
{
    /// <summary>2-D convolution forward, NCHW.</summary>
    Conv2DForward,
    /// <summary>2-D convolution gradient with respect to the input, NCHW.</summary>
    Conv2DBackwardData,
    /// <summary>2-D convolution gradient with respect to the filter, NCHW.</summary>
    Conv2DBackwardFilter,
    /// <summary>Dense matrix multiply, optionally with an epilogue.</summary>
    Gemm,
    /// <summary>Strided batched matrix multiply.</summary>
    BatchedGemm,
    /// <summary>Scaled dot-product attention forward.</summary>
    Attention,
    /// <summary>Scaled dot-product attention backward.</summary>
    AttentionBackward,
    /// <summary>Layer normalization forward.</summary>
    LayerNorm,
    /// <summary>Layer normalization backward.</summary>
    LayerNormBackward,
    /// <summary>Root-mean-square normalization.</summary>
    RmsNorm,
    /// <summary>LayerNorm affine-parameter gradients (column reduction over rows).</summary>
    LayerNormGradParameters,
    /// <summary>RMSNorm scale gradient (column reduction over rows).</summary>
    RmsNormGradGamma,
    /// <summary>Row softmax.</summary>
    Softmax,
    /// <summary>Row softmax backward.</summary>
    SoftmaxBackward,
    /// <summary>Fused elementwise chain.</summary>
    Elementwise,
    /// <summary>Fused optimizer update.</summary>
    OptimizerUpdate,
    /// <summary>Reduction.</summary>
    Reduction,
}

/// <summary>Where a registered candidate came from.</summary>
public enum TunedKernelOrigin
{
    /// <summary>The established in-tree implementation; the reference every other candidate is gated against.</summary>
    Builtin,
    /// <summary>A vendor library algorithm (cuDNN, cuBLAS, oneDNN).</summary>
    Vendor,
    /// <summary>A kernel produced by an in-tree generator or emitter.</summary>
    Generated,
    /// <summary>A kernel found by in-process evolutionary parameter search.</summary>
    Evolved,
    /// <summary>A kernel loaded from an externally evolved, versioned artifact.</summary>
    External,
}

/// <summary>How the registry chooses among candidates.</summary>
public enum TunedKernelMode
{
    /// <summary>Measure unseen shapes online (bounded by the tuning budget) and use profiles and persisted evidence.</summary>
    Tune,
    /// <summary>Never measure: use overrides, then the shipped/persisted profile, then the reference. Choices are pinned.</summary>
    Pinned,
    /// <summary>Always use the reference candidate (overrides still apply).</summary>
    Off,
}

/// <summary>Why a decision selected its candidate.</summary>
public enum TunedKernelDecisionReason
{
    /// <summary>Only one candidate applied to the shape.</summary>
    OnlyCandidate,
    /// <summary>An environment or host override named the candidate.</summary>
    Override,
    /// <summary>A shipped or persisted profile named the candidate.</summary>
    Profile,
    /// <summary>The candidate passed the correctness and paired-timing gate in this process.</summary>
    Tuned,
    /// <summary>No candidate beat the reference under the gate.</summary>
    ReferenceWon,
    /// <summary>The registry is pinned or off, so the reference was used.</summary>
    Pinned,
    /// <summary>The process tuning budget was exhausted before this shape was measured.</summary>
    BudgetExhausted,
}

/// <summary>
/// An allocation-free shape-class key: up to <see cref="Capacity"/> integer dimensions plus a dtype code. The
/// registry keys every decision by (op, device) on the slot and by this value inside it.
/// </summary>
public unsafe struct TunedShape : IEquatable<TunedShape>
{
    /// <summary>Maximum number of dimensions a shape key can hold.</summary>
    public const int Capacity = 20;

    private fixed int _dims[Capacity];
    private int _count;
    private int _dtype;
    private int _hash;

    /// <summary>Gets the number of dimensions.</summary>
    public int Count => _count;

    /// <summary>Gets the dtype code (see <see cref="TunedKernelDType"/>).</summary>
    public int DType => _dtype;

    /// <summary>Gets a dimension.</summary>
    public int this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_count) throw new ArgumentOutOfRangeException(nameof(index));
            fixed (int* d = _dims) return d[index];
        }
    }

    /// <summary>Creates a key from dimensions and a dtype code.</summary>
    public static TunedShape Create(int dtype, ReadOnlySpan<int> dims)
    {
        if (dims.Length > Capacity) throw new ArgumentOutOfRangeException(nameof(dims));
        var s = new TunedShape { _count = dims.Length, _dtype = dtype };
        unchecked
        {
            int h = (int)2166136261 ^ dtype;
            for (int i = 0; i < dims.Length; i++)
            {
                s._dims[i] = dims[i];
                h = (h ^ dims[i]) * 16777619;
            }
            s._hash = h ^ dims.Length;
        }
        return s;
    }

    /// <summary>Creates a two-dimension key (rows x columns) without a span.</summary>
    public static TunedShape Of2(int dtype, int d0, int d1)
    {
        int* d = stackalloc int[2];
        d[0] = d0; d[1] = d1;
        return Create(dtype, new ReadOnlySpan<int>(d, 2));
    }

    /// <summary>Parses the text form produced by <see cref="ToString"/>.</summary>
    public static bool TryParse(string text, out TunedShape shape)
    {
        shape = default;
        if (string.IsNullOrEmpty(text)) return false;
        int colon = text.IndexOf(':');
        if (colon <= 0) return false;
        int dtype = TunedKernelDType.Parse(text.Substring(0, colon));
        if (dtype == 0) return false;
        string body = text.Substring(colon + 1);
        string[] parts = body.Length == 0 ? new string[0] : body.Split('x');
        if (parts.Length > Capacity) return false;
        var dims = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            if (!int.TryParse(parts[i], System.Globalization.NumberStyles.AllowLeadingSign,
                    System.Globalization.CultureInfo.InvariantCulture, out dims[i])) return false;
        shape = Create(dtype, dims);
        return true;
    }

    /// <inheritdoc />
    public bool Equals(TunedShape other)
    {
        if (_hash != other._hash || _count != other._count || _dtype != other._dtype) return false;
        fixed (int* a = _dims)
        {
            int* b = other._dims;
            for (int i = 0; i < _count; i++) if (a[i] != b[i]) return false;
        }
        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is TunedShape s && Equals(s);

    /// <inheritdoc />
    public override int GetHashCode() => _hash;

    /// <summary>Copies the dimensions to a new array (diagnostics and persistence only).</summary>
    public int[] ToArray()
    {
        var a = new int[_count];
        fixed (int* d = _dims) for (int i = 0; i < _count; i++) a[i] = d[i];
        return a;
    }

    /// <summary>A stable text form, "f32:2x3x4", used by profiles, overrides and persistence.</summary>
    public override string ToString() => TunedKernelDType.Name(_dtype) + ":" + string.Join("x", ToArray());
}

/// <summary>Dtype codes for <see cref="TunedShape"/>.</summary>
public static class TunedKernelDType
{
    /// <summary>32-bit float.</summary>
    public const int Float32 = 1;
    /// <summary>64-bit float.</summary>
    public const int Float64 = 2;
    /// <summary>16-bit float.</summary>
    public const int Float16 = 3;
    /// <summary>bfloat16.</summary>
    public const int BFloat16 = 4;

    /// <summary>Text name of a dtype code.</summary>
    public static string Name(int code) => code switch
    {
        Float32 => "f32",
        Float64 => "f64",
        Float16 => "f16",
        BFloat16 => "bf16",
        _ => "t" + code.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };

    /// <summary>Dtype code for a text name, or 0.</summary>
    public static int Parse(string name) => name switch
    {
        "f32" => Float32,
        "f64" => Float64,
        "f16" => Float16,
        "bf16" => BFloat16,
        _ => 0,
    };
}

/// <summary>
/// One implementation of an op family. Contract: <see cref="Execute"/> reads its inputs, never writes them, and
/// fully overwrites its outputs, so the registry can run any candidate on the live arguments while measuring and
/// then run the winner once more for the real result.
/// </summary>
public interface ITunedKernelCandidate<TArgs> where TArgs : struct
{
    /// <summary>Stable identifier, e.g. "cudnn.fwd.winograd". Profiles and overrides refer to it.</summary>
    string Id { get; }

    /// <summary>Where the candidate came from.</summary>
    TunedKernelOrigin Origin { get; }

    /// <summary>True when two runs on the same inputs produce bit-identical outputs (no atomics or races).</summary>
    bool IsDeterministic { get; }

    /// <summary>Whether this candidate supports the shape on this device.</summary>
    bool IsApplicable(in TunedShape shape);

    /// <summary>Runs the candidate.</summary>
    void Execute(in TArgs args);
}

/// <summary>
/// Device-specific measurement services for one op family: capturing outputs for the correctness check and
/// timing a candidate on the device clock.
/// </summary>
public interface ITunedKernelHarness<TArgs> where TArgs : struct
{
    /// <summary>
    /// False when these arguments cannot be measured now: inside CUDA graph capture, or when an output aliases an
    /// input (running a candidate twice in place would corrupt the result).
    /// </summary>
    bool CanMeasure(in TArgs args);

    /// <summary>
    /// Allowed error for this op, as a fraction of the reference output's maximum magnitude. Candidates whose
    /// output differs from the reference by more than this are rejected before they are timed.
    /// </summary>
    double RelativeTolerance { get; }

    /// <summary>
    /// Overwrites the outputs with NaN before a candidate's correctness run, so a candidate that silently writes
    /// nothing cannot pass by leaving the reference's result in place.
    /// </summary>
    void PoisonOutput(in TArgs args);

    /// <summary>Copies the outputs the last execution wrote to host memory.</summary>
    float[] SnapshotOutput(in TArgs args);

    /// <summary>Device time, in milliseconds, of <paramref name="repetitions"/> back-to-back executions.</summary>
    double MeasureMilliseconds(ITunedKernelCandidate<TArgs> candidate, in TArgs args, int repetitions);
}

/// <summary>An immutable record of one registry decision and the evidence behind it.</summary>
public sealed class TunedKernelDecision
{
    internal TunedKernelDecision(TunedKernelOp op, string device, TunedShape shape, string candidateId,
        TunedKernelOrigin origin, TunedKernelDecisionReason reason, string referenceId,
        double referenceMs = double.NaN, double candidateMs = double.NaN, double medianSpeedup = double.NaN,
        double lowerSpeedup = double.NaN, double noiseRatio = double.NaN, double maxRelativeError = double.NaN,
        IReadOnlyList<string>? rejected = null)
    {
        Op = op; Device = device; Shape = shape; CandidateId = candidateId; Origin = origin; Reason = reason;
        ReferenceId = referenceId; ReferenceMilliseconds = referenceMs; CandidateMilliseconds = candidateMs;
        MedianSpeedup = medianSpeedup; LowerSpeedupBound = lowerSpeedup; NoiseRatio = noiseRatio;
        MaxRelativeError = maxRelativeError; Rejected = rejected ?? Array.Empty<string>();
    }

    /// <summary>The op family.</summary>
    public TunedKernelOp Op { get; }
    /// <summary>The device key, e.g. "cuda:sm75:NVIDIA GeForce GTX 1660 Ti".</summary>
    public string Device { get; }
    /// <summary>The shape class.</summary>
    public TunedShape Shape { get; }
    /// <summary>The selected candidate.</summary>
    public string CandidateId { get; }
    /// <summary>The selected candidate's origin.</summary>
    public TunedKernelOrigin Origin { get; }
    /// <summary>Why it was selected.</summary>
    public TunedKernelDecisionReason Reason { get; }
    /// <summary>The reference candidate it was gated against.</summary>
    public string ReferenceId { get; }
    /// <summary>Median reference device time per execution, when measured.</summary>
    public double ReferenceMilliseconds { get; }
    /// <summary>Median selected-candidate device time per execution, when measured.</summary>
    public double CandidateMilliseconds { get; }
    /// <summary>Median within-pair speedup over the reference, when measured.</summary>
    public double MedianSpeedup { get; }
    /// <summary>Lowest within-pair speedup observed (the gate requires it above one).</summary>
    public double LowerSpeedupBound { get; }
    /// <summary>Reference-versus-reference noise ratio measured before the comparison.</summary>
    public double NoiseRatio { get; }
    /// <summary>Selected candidate's error against the reference, relative to the reference's max magnitude.</summary>
    public double MaxRelativeError { get; }
    /// <summary>Candidates that were considered and rejected, with the reason.</summary>
    public IReadOnlyList<string> Rejected { get; }

    /// <inheritdoc />
    public override string ToString()
    {
        string head = $"{Op} {Shape} on {Device}: {CandidateId} ({Reason}";
        if (Reason == TunedKernelDecisionReason.Tuned)
            head += $", {MedianSpeedup:F2}x vs {ReferenceId} [{ReferenceMilliseconds:F4} -> {CandidateMilliseconds:F4} ms]" +
                    $", min pair {LowerSpeedupBound:F2}x, noise {NoiseRatio:F3}, err {MaxRelativeError:E1}";
        head += ")";
        if (Rejected.Count > 0) head += " rejected: " + string.Join("; ", Rejected);
        return head;
    }
}

/// <summary>Process-wide registry policy: mode, overrides, and the online tuning budget.</summary>
public static class TunedKernelPolicy
{
    /// <summary>Environment variable selecting the mode: <c>tune</c> (default), <c>pinned</c>, or <c>off</c>.</summary>
    public const string ModeEnvironmentVariable = "AIDOTNET_KERNEL_REGISTRY";

    /// <summary>Environment variable for the per-process online tuning budget in milliseconds (default 2000).</summary>
    public const string BudgetEnvironmentVariable = "AIDOTNET_KERNEL_TUNE_BUDGET_MS";

    /// <summary>Prefix of the per-op override variable, e.g. <c>AIDOTNET_KERNEL_CONV2DFORWARD=cudnn.fwd.winograd</c>.</summary>
    public const string OverridePrefix = "AIDOTNET_KERNEL_";

    /// <summary>Environment variable that disables reading and writing persisted decisions when set to 0.</summary>
    public const string PersistEnvironmentVariable = "AIDOTNET_KERNEL_REGISTRY_PERSIST";

    private static readonly TunedKernelMode s_envMode =
        ParseMode(Environment.GetEnvironmentVariable(ModeEnvironmentVariable));
    private static readonly ConcurrentDictionary<TunedKernelOp, string?> s_overrides = new();
    private static long s_spentTicks;

    /// <summary>Mode override for tests and hosts; null uses the environment.</summary>
    public static TunedKernelMode? ModeOverride { get; set; }

    /// <summary>The effective mode.</summary>
    public static TunedKernelMode Mode => ModeOverride ?? s_envMode;

    /// <summary>Whether persisted decisions are read and written.</summary>
    public static bool PersistDecisions { get; set; } =
        Environment.GetEnvironmentVariable(PersistEnvironmentVariable) != "0";

    /// <summary>The per-process online tuning budget in milliseconds.</summary>
    public static double BudgetMilliseconds { get; set; } =
        ParseBudget(Environment.GetEnvironmentVariable(BudgetEnvironmentVariable));

    /// <summary>Milliseconds of online tuning spent so far in this process.</summary>
    public static double SpentMilliseconds =>
        Interlocked.Read(ref s_spentTicks) * 1000.0 / Stopwatch.Frequency;

    /// <summary>The candidate id an override names for <paramref name="op"/>, if any.</summary>
    public static string? OverrideFor(TunedKernelOp op) =>
        s_overrides.GetOrAdd(op, o =>
        {
            string? v = Environment.GetEnvironmentVariable(OverridePrefix + o.ToString().ToUpperInvariant());
            return v is null || string.IsNullOrWhiteSpace(v) ? null : v.Trim();
        });

    /// <summary>Sets or clears an in-process override (tests and hosts). Affects only shapes not yet decided.</summary>
    public static void SetOverride(TunedKernelOp op, string? candidateId) => s_overrides[op] = candidateId;

    internal static bool BudgetRemaining => SpentMilliseconds < BudgetMilliseconds;

    internal static void Charge(long stopwatchTicks) => Interlocked.Add(ref s_spentTicks, stopwatchTicks);

    /// <summary>Resets the spent budget (tests).</summary>
    internal static void ResetBudget() => Interlocked.Exchange(ref s_spentTicks, 0);

    private static TunedKernelMode ParseMode(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "pinned" or "deterministic" => TunedKernelMode.Pinned,
        "off" or "0" or "false" => TunedKernelMode.Off,
        _ => TunedKernelMode.Tune,
    };

    private static double ParseBudget(string? value) =>
        double.TryParse(value, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double ms) && ms >= 0 ? ms : 2000.0;
}

/// <summary>Process-wide view of every slot and decision, for inventories and reports.</summary>
public static class TunedKernelRegistry
{
    private static readonly ConcurrentQueue<TunedKernelDecision> s_log = new();
    private static readonly List<WeakReference<ITunedKernelSlotInfo>> s_slots = new();
    private static readonly object s_slotsLock = new();

    /// <summary>Environment variable that writes every decision to the trace listeners when set to 1.</summary>
    public const string LogEnvironmentVariable = "AIDOTNET_KERNEL_REGISTRY_LOG";

    private static readonly bool s_logDecisions = Environment.GetEnvironmentVariable(LogEnvironmentVariable) == "1";

    /// <summary>Every decision made in this process, oldest first (bounded to the most recent 4096).</summary>
    public static IReadOnlyList<TunedKernelDecision> Decisions => s_log.ToArray();

    /// <summary>A snapshot of live slots: op, device and registered candidate ids.</summary>
    public static IReadOnlyList<ITunedKernelSlotInfo> Slots
    {
        get
        {
            lock (s_slotsLock)
            {
                var live = new List<ITunedKernelSlotInfo>();
                s_slots.RemoveAll(w => !w.TryGetTarget(out _));
                foreach (var w in s_slots) if (w.TryGetTarget(out var s)) live.Add(s);
                return live;
            }
        }
    }

    internal static void Record(TunedKernelDecision decision)
    {
        s_log.Enqueue(decision);
        while (s_log.Count > 4096 && s_log.TryDequeue(out _)) { }
        // Trace, not the console: the host routes it (a console or file listener, its logging framework).
        if (s_logDecisions) System.Diagnostics.Trace.WriteLine("[kernel-registry] " + decision);
    }

    internal static void Track(ITunedKernelSlotInfo slot)
    {
        lock (s_slotsLock) s_slots.Add(new WeakReference<ITunedKernelSlotInfo>(slot));
    }
}

/// <summary>Read-only description of a slot.</summary>
public interface ITunedKernelSlotInfo
{
    /// <summary>The op family.</summary>
    TunedKernelOp Op { get; }
    /// <summary>The device key.</summary>
    string Device { get; }
    /// <summary>Registered candidate ids, reference first.</summary>
    IReadOnlyList<string> CandidateIds { get; }
    /// <summary>Decisions currently cached in this slot.</summary>
    IReadOnlyList<TunedKernelDecision> CachedDecisions { get; }
}
