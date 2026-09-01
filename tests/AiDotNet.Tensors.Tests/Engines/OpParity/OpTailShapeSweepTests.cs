// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Runs the whole parity registry again at an inner dimension that is NOT a multiple of a SIMD
/// vector, on CPU only, against the double oracle.
/// </summary>
/// <remarks>
/// <para>
/// THE REGISTRY COVERS THE OPS AND MISSES THE SHAPES. 536 of the 544 tensor-returning IEngine ops
/// have a parity spec — 98.5% of the surface — and every shape literal in those specs uses a
/// dimension of 1, 2, 3, 4, 6, 8, 16, 32 or 64. Each of those is either BELOW a SIMD vector
/// (8 floats / 4 doubles) or an exact MULTIPLE of one. Nothing in the registry ever asks an op for
/// 13 columns, which is the only condition a vectorized column tail can be got wrong at.
/// </para>
/// <para>
/// That gap is what let two kernels ship with unguarded tails. The 4-row block kernels stored a
/// full vector on their final column step, which overwrote the start of the next row (wrong values
/// for every row but the last of each block) and, on the last block, ran past the end of C into the
/// GC heap. The process then died at an unrelated allocation with
/// <c>Internal CLR error (0x80131506)</c>. 536 covered ops said nothing, because none of them was
/// ever shaped to trigger it.
/// </para>
/// <para>
/// AND IT RUNS ON CPU. <see cref="OpParityHarness.CheckForward"/> skips outright when no DirectGpu
/// backend is present, so on an ordinary CPU-only CI shard the entire registry executes nothing.
/// This sweep compares the CPU float run against the CPU double oracle, so it runs everywhere the
/// tests run — which is where these kernels actually shipped broken.
/// </para>
/// <para>
/// The comparison bar is deliberately loose. This is not a ULP test; drift between float and double
/// is expected and uninteresting. It is looking for GROSS breakage — the measured failure was
/// 0.33678542 where the product is 0.694251873, a relative error near 1 — so a 1e-3 relative bar
/// clears real FP32 rounding by orders of magnitude while still catching a corrupted row outright.
/// </para>
/// </remarks>
public class OpTailShapeSweepTests
{
    /// <summary>Gross-breakage bar; see the class remarks for why it is loose on purpose.</summary>
    private static readonly ParityTol GrossBreakage = ParityTol.Accum(rel: 1e-3, ulps: 1L << 24, absFloor: 1e-3);

    /// <summary>
    /// Rewrites an inner dimension to one that is a multiple of neither vector width.
    /// </summary>
    /// <remarks>
    /// Only the LAST dimension moves: it is the one a row-major kernel strides over, so it is where
    /// a column tail lives, and leaving the outer dimensions alone keeps the tensors small. The
    /// result avoids multiples of 4 AND 8 so the FP64 kernels (4 doubles per vector) and the FP32
    /// kernels (8 floats) both get a tail from the same sweep. A dimension of 0 or 1 is left alone —
    /// it usually carries meaning (a singleton broadcast axis, an empty edge case) rather than width.
    /// </remarks>
    internal static int[] AwkwardShape(int[] shape)
    {
        if (shape is null || shape.Length == 0) return shape ?? Array.Empty<int>();

        var rewritten = (int[])shape.Clone();
        int last = rewritten.Length - 1;
        int d = rewritten[last];
        if (d <= 1) return rewritten;

        int t = d < 9 ? 13 : d + 5;
        while (t % 8 == 0 || t % 4 == 0) t++;
        rewritten[last] = t;
        return rewritten;
    }

    private sealed class SweepOutcome
    {
        public List<string> Mismatches { get; } = new();
        public List<string> NotApplicable { get; } = new();
        public List<string> Excluded { get; } = new();
        public List<string> Crashed { get; } = new();
        public int Compared { get; set; }
        public int Total { get; set; }
    }

    private static readonly Lazy<SweepOutcome> Sweep = new(RunSweep, isThreadSafe: true);

    [Fact]
    public void EveryOp_AtAnAwkwardInnerDimension_AgreesWithTheDoubleOracle()
    {
        var outcome = Sweep.Value;

        Assert.True(
            outcome.Crashed.Count == 0,
            $"{outcome.Crashed.Count} ops FAILED at an inner dimension that is not a multiple of a "
                + "SIMD vector, rather than rejecting it. An op that neither validates the shape nor "
                + "handles it walks off the end of its own operands.\n  "
                + string.Join("\n  ", outcome.Crashed.Take(40)));

        Assert.True(
            outcome.Mismatches.Count == 0,
            $"{outcome.Mismatches.Count} of {outcome.Compared} compared ops disagree with the double "
                + "oracle once their inner dimension is not a multiple of a SIMD vector. That is the "
                + "signature of an unguarded vector tail: the full-width path is right and the "
                + "remainder is not.\n  "
                + string.Join("\n  ", outcome.Mismatches.Take(40))
                + (outcome.Mismatches.Count > 40 ? $"\n  ... and {outcome.Mismatches.Count - 40} more" : string.Empty));
    }

    [Fact]
    public void AwkwardShapeSweep_ActuallyReachesMostOfTheRegistry()
    {
        var outcome = Sweep.Value;

        // A sweep that quietly compared nothing would pass the assertion above forever. This is the
        // guard against that: it fails if the reach COLLAPSES, not if it is merely imperfect. Ops
        // whose shapes must agree with a literal the policy cannot see will always be
        // not-applicable, and that is recorded rather than hidden.
        //
        // Raise the floor as reach improves; a drop means ops stopped being exercised at a tail.
        const int reachFloor = 400;   // measured reach is 453 of 556; this guards against collapse
        Assert.True(
            outcome.Compared >= reachFloor,
            $"The awkward-shape sweep only compared {outcome.Compared} of {outcome.Total} registry "
                + $"cases (floor {reachFloor}). {outcome.NotApplicable.Count} were not applicable "
                + $"(the op rejected the rewritten shape) and {outcome.Excluded.Count} were excluded. "
                + "If this dropped, ops stopped being exercised at a non-multiple-of-vector width.");
    }

    private static SweepOutcome RunSweep()
    {
        var outcome = new SweepOutcome();
        var cpu = new CpuEngine();
        string reportPath = BeginReport();

        // The policy must be live while the registry is ENUMERATED as well as while a case runs:
        // most specs build their OpInputs in the generator body, so the shapes are chosen at
        // enumeration time, not at invocation time.
        using (OpInput.UseShapePolicy(AwkwardShape))
        {
            List<OpCase> cases;
            try
            {
                cases = OpParityRegistry.All().ToList();
            }
            catch (Exception ex)
            {
                // A spec that cannot even be CONSTRUCTED at an awkward shape tells us nothing about
                // the kernels, and must not mask the sweep entirely.
                Append(reportPath, $"REGISTRY-ENUMERATION-FAILED\t{Describe(ex)}");
                outcome.NotApplicable.Add($"<registry enumeration>: {Describe(ex)}");
                return outcome;
            }

            outcome.Total = cases.Count;

            foreach (var op in cases)
            {
                // Written BEFORE the op runs. An op that corrupts the heap can take the process with
                // it at an unrelated allocation later; the last line here names the last op attempted.
                Append(reportPath, $"ATTEMPT\t{op.Name}\t{op.Category}");

                if (op.KnownDivergence is not null)
                {
                    outcome.Excluded.Add($"{op.Name}: known divergence");
                    continue;
                }

                // Random ops seed a generator; float and double runs draw different sequences by
                // construction, so an oracle comparison is meaningless for them. The GPU parity path
                // compares them float-to-float instead, which is why it can include them and this
                // cannot.
                if (string.Equals(op.Category, "random", StringComparison.Ordinal))
                {
                    outcome.Excluded.Add($"{op.Name}: random category");
                    continue;
                }

                float[] cpuF;
                double[] oracleD;
                try
                {
                    cpuF = op.RunFloat(cpu).ToArray();
                    oracleD = op.RunDouble(cpu).ToArray();
                }
                catch (Exception ex) when (!IsShapeRejection(ex))
                {
                    // NOT a rejection: the op FAILED at this shape. Collected rather than rethrown
                    // so one run reports every such op instead of stopping at the first, and
                    // asserted on below so it can never be mistaken for "unsupported".
                    outcome.Crashed.Add($"{op.Name} [{op.Category}]: {Describe(ex)}");
                    Append(reportPath, $"CRASH	{op.Name}	{Describe(ex)}");
                    continue;
                }
                catch (Exception ex) when (IsShapeRejection(ex))
                {
                    // The op rejected the rewritten shape — almost always because a spec pairs a
                    // policy-visible OpInput with an inline literal the policy cannot reach, so the
                    // two no longer agree. Not a kernel defect; recorded so the reach stays honest.
                    //
                    // ONLY a rejection is caught. Catching every Exception here made the sweep able
                    // to hide the very thing it exists to find: a tail path that walks off the end
                    // throws IndexOutOfRangeException, which would have been filed as "not
                    // applicable" and skipped, and with a reach of 453 against a floor of 400 up to
                    // 53 such failures could sit there while both tests passed. Anything that is
                    // not a documented precondition rejection now propagates and fails the run.
                    outcome.NotApplicable.Add($"{op.Name}: {Describe(ex)}");
                    Append(reportPath, $"N/A\t{op.Name}\t{Describe(ex)}");
                    continue;
                }

                if (cpuF.Length != oracleD.Length)
                {
                    outcome.NotApplicable.Add(
                        $"{op.Name}: float/double lengths differ ({cpuF.Length} vs {oracleD.Length})");
                    continue;
                }

                outcome.Compared++;

                var oracleF = ParityMath.ToFloat(oracleD);
                if (!ParityMath.Within(cpuF, oracleF, GrossBreakage, out var delta))
                {
                    string line = string.Format(
                        CultureInfo.InvariantCulture,
                        "{0} [{1}]: worst index {2}, float {3:R} vs oracle {4:R} (abs {5:R}, rel {6:R})",
                        op.Name, op.Category, delta.WorstIndex, delta.WorstA, delta.WorstB,
                        delta.MaxAbs, delta.MaxRel);
                    outcome.Mismatches.Add(line);
                    Append(reportPath, $"MISMATCH\t{line}");
                }

                // Heap corruption from an over-wide store is invisible until something collects. A
                // forced compacting collection here attributes it to the op that just ran instead of
                // to whatever allocates next, which in practice was a test two classes away. Off by
                // default because it is slow.
                if (GcVerifyEnabled)
                {
                    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                    GC.WaitForPendingFinalizers();
                }
            }
        }

        WriteSummary(reportPath, outcome);
        return outcome;
    }

    /// <summary>
    /// Whether an exception is an op REJECTING the rewritten shape, as opposed to failing on it.
    /// </summary>
    /// <remarks>
    /// A precondition rejection is an <see cref="ArgumentException"/> (FFT wanting a power of two,
    /// split wanting divisibility, a mask that must match its tensor) or a
    /// <see cref="NotSupportedException"/>. Everything else — an index walking off the end, a null
    /// deref, an invalid state — is the op FAILING at that shape, which is the finding this sweep
    /// exists to surface and must never be filed as "not applicable".
    /// <para>
    /// <see cref="IndexOutOfRangeException"/> deliberately does not qualify. It derives from
    /// <see cref="SystemException"/> rather than <see cref="ArgumentException"/>, so narrowing to
    /// argument exceptions lets exactly the interesting failure through.
    /// </para>
    /// </remarks>
    private static bool IsShapeRejection(Exception ex)
        => ex is ArgumentException or NotSupportedException;

    private static bool GcVerifyEnabled =>
        !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_OPSWEEP_GCVERIFY"));

    private static string Describe(Exception ex) =>
        $"{ex.GetType().Name}: {ex.Message.Replace('\n', ' ').Replace('\r', ' ')}";

    private static string BeginReport()
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, "op-tail-shape-sweep.txt");
            File.WriteAllText(path, "# awkward-inner-dimension sweep (CPU float vs CPU double oracle)\n");
            return path;
        }
        catch (IOException)
        {
            return string.Empty;
        }
        catch (UnauthorizedAccessException)
        {
            return string.Empty;
        }
    }

    private static void Append(string path, string line)
    {
        if (path.Length == 0) return;
        try { File.AppendAllText(path, line + "\n"); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static void WriteSummary(string path, SweepOutcome outcome)
    {
        if (path.Length == 0) return;
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine($"registry cases    : {outcome.Total}");
        sb.AppendLine($"compared          : {outcome.Compared}");
        sb.AppendLine($"mismatches        : {outcome.Mismatches.Count}");
        sb.AppendLine($"not applicable    : {outcome.NotApplicable.Count}");
        sb.AppendLine($"excluded          : {outcome.Excluded.Count}");
        sb.AppendLine();
        sb.AppendLine("## mismatches");
        foreach (var m in outcome.Mismatches) sb.AppendLine($"  {m}");
        sb.AppendLine();
        sb.AppendLine("## not applicable (op rejected the rewritten shape)");
        foreach (var n in outcome.NotApplicable) sb.AppendLine($"  {n}");
        Append(path, sb.ToString());
    }
}
#endif
