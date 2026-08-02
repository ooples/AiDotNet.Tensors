using System;

namespace AiDotNet.Tensors.Tests.Engines.OpParity
{
    /// <summary>
    /// Optional test-time partitioning for the data-driven parity theories, so a single
    /// <c>dotnet test</c> process only exercises a subset of ops. Set
    /// <c>AIDOTNET_PARITY_SHARD="k/N"</c> (1-based k) to keep only ops whose stable name hash falls
    /// in bucket k of N. Unset =&gt; the full set — so the normal build job, which never sets it, still
    /// runs every op. Used by the POCL parity lane to keep each shard under POCL's per-process
    /// kernel-compile ceiling (a single process that compiles too many distinct kernels loses the
    /// OpenCL context part-way through).
    /// </summary>
    internal static class ParityShard
    {
        // Parsed once at type-init from the process environment (the workflow sets it per matrix job).
        private static readonly (int Index, int Total)? Shard =
            Parse(Environment.GetEnvironmentVariable("AIDOTNET_PARITY_SHARD"));

        private static (int, int)? Parse(string spec)
        {
            if (string.IsNullOrWhiteSpace(spec)) return null;
            var parts = spec.Split('/');
            if (parts.Length == 2
                && int.TryParse(parts[0], out int k) && int.TryParse(parts[1], out int n)
                && n > 0 && k >= 1 && k <= n)
            {
                return (k - 1, n);
            }
            return null;
        }

        /// <summary>
        /// True if <paramref name="opName"/> belongs to the current shard. Always true when
        /// AIDOTNET_PARITY_SHARD is unset or malformed (full coverage).
        /// </summary>
        internal static bool Include(string opName)
        {
            if (Shard == null) return true;
            var s = Shard.Value;
            return (int)(StableHash(opName) % (uint)s.Total) == s.Index;
        }

        // FNV-1a 32-bit — deterministic across processes/runs, unlike string.GetHashCode() which is
        // randomized per-process on modern .NET (that would put an op in a different shard each run).
        private static uint StableHash(string s)
        {
            uint h = 2166136261u;
            foreach (char c in s)
            {
                h ^= c;
                h *= 16777619u;
            }
            return h;
        }
    }
}
