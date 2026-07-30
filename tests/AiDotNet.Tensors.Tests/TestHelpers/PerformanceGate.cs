using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.TestHelpers
{
    /// <summary>
    /// Shared opt-in gate for performance / benchmark tests. Centralizes the
    /// <c>AIDOTNET_RUN_PERF_TESTS</c> check that was previously duplicated verbatim across every perf
    /// test — and with inconsistent <c>== "1"</c> vs <c>string.Equals(..., Ordinal)</c> comparisons —
    /// so the enable flag, the ordinal comparison, and the skip message are all defined once.
    /// </summary>
    public static class PerformanceGate
    {
        /// <summary>True when <c>AIDOTNET_RUN_PERF_TESTS=1</c> (opt-in perf/benchmark runs).</summary>
        public static bool Enabled =>
            string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_TESTS"), "1", StringComparison.Ordinal);

        /// <summary>Skips the calling <c>[SkippableFact]</c>/<c>[SkippableTheory]</c> unless perf tests are enabled.</summary>
        public static void SkipUnlessEnabled() =>
            Skip.IfNot(Enabled, "Performance gate; set AIDOTNET_RUN_PERF_TESTS=1 to run.");
    }
}
