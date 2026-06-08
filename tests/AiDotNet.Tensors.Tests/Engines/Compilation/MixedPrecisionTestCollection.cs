using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Serializes every test class that mutates the process-wide
/// <c>MixedPrecisionEmit.TestOverrideEnabled</c> static (the AIDOTNET_FP16_ACTIVATIONS
/// test override). The per-test try/finally restores the flag WITHIN a test, but xUnit
/// runs separate collections in parallel by default — two classes toggling the global
/// concurrently can observe each other's override state and flake. Putting all mutators
/// in one non-parallel collection makes the global mutation safe without locking in
/// every test body (PR #557 review).
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class MixedPrecisionTestCollection
{
    public const string Name = "MixedPrecisionTests";
}
