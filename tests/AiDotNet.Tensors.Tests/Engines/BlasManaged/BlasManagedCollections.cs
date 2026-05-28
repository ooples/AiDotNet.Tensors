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
/// Single serial collection for every test that mutates process-wide global state
/// affecting the GEMM path — CpuParallelSettings, the OpenBLAS thread count,
/// BlasManaged stats/JIT caches, and especially BlasProvider.IsDeterministicMode
/// (which AiDotNetEngine.SetDeterministicMode also flips, and which switches the GEMM
/// reduction/packing path). These were previously split across three separate serial
/// collections (BlasGlobalState, AiDotNetEngineGlobalState, this one); cross-collection
/// DisableParallelization did NOT reliably isolate them on CI, so a flip of the
/// determinism flag in one collection drifted a concurrent pre-pack/GEMM test's output
/// away from its baseline (the intermittent "drift" failures in
/// PrePackedB_Output_BitMatches_LivePack, the cached-packed-buffer tests, and
/// SetDeterministicMode_Toggles). Membership in ONE collection guarantees sequential
/// execution (xUnit's core contract, independent of DisableParallelization semantics),
/// which is what actually fixes the flakiness.
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
