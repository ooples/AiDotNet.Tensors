using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// xUnit serial-execution markers for BlasManaged test groups. Without these
// CollectionDefinition entries the [Collection("...")] attributes on
// individual test classes just group them together — they don't actually
// serialize execution. CodeRabbit #366 flagged ConvTranspose2DL2PerfTest and
// DeterminismTests as needing serialization; promoting the existing
// collection names to real definitions fixes both findings AND makes the
// pre-existing [Collection] usages on ScalarKernelTests, StatsCounterTests,
// NativeAotCompatibilityTest, and BlasManagedRegressionTest actually do
// what their authors intended.

/// <summary>
/// Tests that mutate process-wide global state (CpuParallelSettings,
/// BlasProvider.IsDeterministicMode, BlasManaged stats counters) must run
/// serially to avoid cross-class state contamination.
/// </summary>
[CollectionDefinition("BlasManaged-Stats-Serial", DisableParallelization = true)]
public sealed class BlasManagedStatsSerialCollection { }

/// <summary>
/// Performance tests need exclusive CPU time to keep measurement noise low.
/// Concurrent execution causes flaky perf-threshold checks.
/// </summary>
[CollectionDefinition("BlasManaged-Perf-Serial", DisableParallelization = true)]
public sealed class BlasManagedPerfSerialCollection { }

/// <summary>
/// StreamingWorkerPool tests measure dispatch latency and assert each chunk
/// runs exactly once — both are sensitive to concurrent CPU contention from
/// other collections. Serialize against all other collections.
/// </summary>
[CollectionDefinition("BlasManaged-Pool-Serial", DisableParallelization = true)]
public sealed class BlasManagedPoolSerialCollection { }

/// <summary>
/// AutotuneCacheV2 tests toggle the process-global <c>EnableAutotuneV2</c>
/// flag. If another GEMM collection ran concurrently while the flag is on, its
/// dispatch would change underneath it. Serialize against all other collections.
/// </summary>
[CollectionDefinition("BlasManaged-Autotune-Serial", DisableParallelization = true)]
public sealed class BlasManagedAutotuneSerialCollection { }
