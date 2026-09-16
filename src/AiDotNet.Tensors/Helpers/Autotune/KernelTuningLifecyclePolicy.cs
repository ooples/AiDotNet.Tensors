namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Finite per-controller retuning and monitoring allowances, explicitly configured by the application.</summary>
public sealed class KernelTuningLifecyclePolicy
{
    /// <summary>The default number of raw timings required in each monitoring window.</summary>
    public const int DefaultMonitoringSamples = 9;

    /// <summary>The default number of consecutive violating windows required before rollback.</summary>
    public const int DefaultConsecutiveBreaches = 2;

    /// <summary>The default P95 ratio, relative to the deployed validation receipt, treated as a regression.</summary>
    public const double DefaultMaximumP95RegressionRatio = 1.1;

    // Validation maxima. These are contract bounds, not tuning knobs: they cap what a controller may
    // admit over its lifetime and how much monitoring evidence it can be made to retain.
    private const int MinimumRetunes = 1;
    private const int MaximumRetunesAllowed = 1024;
    private const int MinimumSearchOperations = 1;
    private const int MaximumSearchOperations = 1_000_000;
    private const int MinimumMonitoringSamples = 3;
    private const int MaximumMonitoringSamples = 4096;
    private const int MinimumConsecutiveBreaches = 1;
    private const int MaximumConsecutiveBreaches = 100;

    /// <summary>The largest number of retained raw timings across all monitoring windows combined.</summary>
    private const int MaximumRetainedMonitoringSamples = 65536;

    private static readonly TimeSpan MaximumTimeout = TimeSpan.FromDays(1);
    private static readonly TimeSpan MaximumGracePeriod = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan MaximumCooldown = TimeSpan.FromDays(30);

    /// <summary>Creates fixed lifetime admission limits and a predeclared regression-window policy.</summary>
    public KernelTuningLifecyclePolicy(int maximumRetunes, int evaluationsPerRetune, int proposalsPerRetune,
        TimeSpan timeout, TimeSpan gracePeriod, TimeSpan cooldown, int monitoringSamples = DefaultMonitoringSamples,
        int consecutiveBreaches = DefaultConsecutiveBreaches,
        double maximumP95RegressionRatio = DefaultMaximumP95RegressionRatio)
    {
        if (maximumRetunes < MinimumRetunes || maximumRetunes > MaximumRetunesAllowed)
            throw new ArgumentOutOfRangeException(nameof(maximumRetunes));
        if (evaluationsPerRetune < MinimumSearchOperations || evaluationsPerRetune > MaximumSearchOperations)
            throw new ArgumentOutOfRangeException(nameof(evaluationsPerRetune));
        if (proposalsPerRetune < evaluationsPerRetune || proposalsPerRetune > MaximumSearchOperations)
            throw new ArgumentOutOfRangeException(nameof(proposalsPerRetune));
        if (timeout <= TimeSpan.Zero || timeout > MaximumTimeout) throw new ArgumentOutOfRangeException(nameof(timeout));
        if (gracePeriod < TimeSpan.Zero || gracePeriod > MaximumGracePeriod) throw new ArgumentOutOfRangeException(nameof(gracePeriod));
        if (cooldown < TimeSpan.Zero || cooldown > MaximumCooldown) throw new ArgumentOutOfRangeException(nameof(cooldown));
        if (monitoringSamples < MinimumMonitoringSamples || monitoringSamples > MaximumMonitoringSamples)
            throw new ArgumentOutOfRangeException(nameof(monitoringSamples));
        if (consecutiveBreaches < MinimumConsecutiveBreaches || consecutiveBreaches > MaximumConsecutiveBreaches)
            throw new ArgumentOutOfRangeException(nameof(consecutiveBreaches));
        if (monitoringSamples * consecutiveBreaches > MaximumRetainedMonitoringSamples)
            throw new ArgumentException("Combined monitoring windows exceed the bounded evidence allowance.");
        if (!KernelTuningMeasurement.IsFinite(maximumP95RegressionRatio) || maximumP95RegressionRatio <= 1)
            throw new ArgumentOutOfRangeException(nameof(maximumP95RegressionRatio));
        MaximumRetunes = maximumRetunes;
        EvaluationsPerRetune = evaluationsPerRetune;
        ProposalsPerRetune = proposalsPerRetune;
        Timeout = timeout;
        GracePeriod = gracePeriod;
        Cooldown = cooldown;
        MonitoringSamples = monitoringSamples;
        ConsecutiveBreaches = consecutiveBreaches;
        MaximumP95RegressionRatio = maximumP95RegressionRatio;
    }
    /// <summary>Gets the controller-lifetime admission cap; failed/abandoned admissions are not refunded.</summary>
    public int MaximumRetunes { get; }
    /// <summary>Gets the search evaluation cap per admitted retune; finalist replay is a separate bounded provider operation.</summary>
    public int EvaluationsPerRetune { get; }
    /// <summary>Gets the proposal cap per admitted retune.</summary>
    public int ProposalsPerRetune { get; }
    /// <summary>Gets the cooperative timeout including idle wait, search and replay.</summary>
    public TimeSpan Timeout { get; }
    /// <summary>Gets the additional wait before reporting abandoned work; this does not kill an uncooperative backend.</summary>
    public TimeSpan GracePeriod { get; }
    /// <summary>Gets the minimum interval between admissions.</summary>
    public TimeSpan Cooldown { get; }
    /// <summary>Gets the required number of raw timings in each monitoring window.</summary>
    public int MonitoringSamples { get; }
    /// <summary>Gets the number of consecutive violating windows required for rollback.</summary>
    public int ConsecutiveBreaches { get; }
    /// <summary>Gets the maximum P95 ratio relative to the exact deployed validation receipt.</summary>
    public double MaximumP95RegressionRatio { get; }
}

/// <summary>Outcome of an explicit request to drain the controller's single coalesced retuning slot.</summary>
public enum KernelTuningRetuneStatus
{
    /// <summary>No retuning is pending.</summary>
    NotRequested,
    /// <summary>An earlier operation is still executing, including abandoned work.</summary>
    Busy,
    /// <summary>Lifetime admission or cooldown prevents starting more work.</summary>
    BudgetDenied,
    /// <summary>Completed validated tuning was adopted for the still-current envelope.</summary>
    Completed,
    /// <summary>The envelope changed before completion; results were not adopted by this controller.</summary>
    Stale,
    /// <summary>Fresh evidence or artifact durability did not satisfy explicit application promotion policy.</summary>
    Rejected,
    /// <summary>The caller stopped waiting after its deadline; capacity remains occupied until work settles.</summary>
    Abandoned
}
