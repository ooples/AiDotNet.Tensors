// Copyright (c) AiDotNet. All rights reserved.

using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// xUnit collection definition for tests that touch the process-wide
/// <see cref="AiDotNet.Tensors.LinearAlgebra.WeightRegistry"/> singleton.
/// Both <see cref="WeightRegistryStreamingTests"/> and
/// <see cref="WeightLifetimeIntegrationTests"/> reference the
/// <c>"WeightRegistry"</c> collection via <c>[Collection("WeightRegistry")]</c>;
/// this class exists so xUnit registers the collection with
/// <see cref="CollectionDefinitionAttribute.DisableParallelization"/> = true,
/// matching the convention used by other shared-singleton suites in this
/// repo (PersistenceGuard, AutotuneCacheTests, etc.). Without
/// DisableParallelization, the collection's classes still serialize against
/// each other (xUnit default), but inside-class methods could parallelize
/// in future xUnit versions — opt out explicitly to be future-proof.
/// </summary>
[CollectionDefinition("WeightRegistry", DisableParallelization = true)]
public sealed class WeightRegistryCollection
{
    // No fixture needed — the collection definition exists purely to
    // disable parallelization for classes referencing it.
}
