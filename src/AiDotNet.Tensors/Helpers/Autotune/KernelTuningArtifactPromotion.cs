namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Explicit application authorization for artifact replay and promotion; never inferred from benchmark success.</summary>
public sealed class KernelTuningArtifactPromotionPolicy
{
    /// <summary>Creates a policy with explicit latency/noise gates and durability requirements.</summary>
    public KernelTuningArtifactPromotionPolicy(KernelTuningOptions options, bool requireDurableArtifact = true)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        // Only promotion thresholds belong to this authorization, not the search archive configuration.
        Options = new KernelTuningOptions
        {
            MinimumPromotionRatio = options.MinimumPromotionRatio,
            MaximumP95LatencyRatio = options.MaximumP95LatencyRatio
        };
        _ = Options.SnapshotAndValidate(KernelTuningDeviceKind.Cpu);
        RequireDurableArtifact = requireDurableArtifact;
    }
    internal KernelTuningOptions Options { get; }
    /// <summary>Gets whether promotion requires a successful artifact directory durability barrier in this invocation.</summary>
    public bool RequireDurableArtifact { get; }
}

/// <summary>A fresh promotion decision, separating artifact evidence, activation and winner-cache persistence.</summary>
public sealed class KernelTuningArtifactPromotion<TConfiguration> where TConfiguration : notnull
{
    internal KernelTuningArtifactPromotion(KernelTuningDeploymentSnapshot<TConfiguration>? active,
        KernelTuningDeploymentSnapshot<TConfiguration> candidate, KernelTuningArtifactReceipt evidence, bool promoted, bool persisted)
    { ActiveDeployment = active; Candidate = candidate; Evidence = evidence; WasPromoted = promoted; WasPersisted = persisted; }
    /// <summary>Gets the active implementation, or null when the caller must use its valid built-in fallback.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration>? ActiveDeployment { get; }
    /// <summary>Gets the freshly replayed candidate, including rejected/inconclusive promotion evidence.</summary>
    public KernelTuningDeploymentSnapshot<TConfiguration> Candidate { get; }
    /// <summary>Gets the retained replay artifact receipt.</summary>
    public KernelTuningArtifactReceipt Evidence { get; }
    /// <summary>Gets whether the explicit application policy activated this candidate.</summary>
    public bool WasPromoted { get; }
    /// <summary>Gets whether winner-cache persistence also succeeded.</summary>
    public bool WasPersisted { get; }
}
