// Copyright (c) AiDotNet. All rights reserved.
// Cross-cutting guarantees for the conveyor: a static-metric ratchet, byte-for-byte
// determinism, and architecture parameterisation.
//
// The device metrics that actually gate a release -- registers, SASS instructions,
// spills -- come from ptxas and nvdisasm and therefore need a GPU. CI has none. So
// the ratchet runs on the metrics the EMITTER can produce without a device: PTX size,
// how many loads it issued, how many bounds guards interval analysis elided, and how
// the reduction was lowered. Those are the quantities a codegen regression moves
// first, and they move in the same direction as the device metrics.
//
// The baseline is checked in. A change that improves a metric fails too, with a
// message saying to update the baseline -- a ratchet that silently accepts
// improvements cannot tell an optimisation from a kernel that stopped doing work.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenRatchetTests
{
    private const int Major = 8, Minor = 6;

#if NET6_0_OR_GREATER
    private sealed record Metrics(int PtxLines, int Loads, int VectorLoads, int ElidedGuards, int LoopedAxes)
    {
        internal string ToRow(string kernel) => string.Join("\t",
            kernel,
            PtxLines.ToString(CultureInfo.InvariantCulture),
            Loads.ToString(CultureInfo.InvariantCulture),
            VectorLoads.ToString(CultureInfo.InvariantCulture),
            ElidedGuards.ToString(CultureInfo.InvariantCulture),
            LoopedAxes.ToString(CultureInfo.InvariantCulture));
    }
#else
    private sealed class Metrics : IEquatable<Metrics>
    {
        internal Metrics(int ptxLines, int loads, int vectorLoads, int elidedGuards, int loopedAxes)
        {
            PtxLines = ptxLines;
            Loads = loads;
            VectorLoads = vectorLoads;
            ElidedGuards = elidedGuards;
            LoopedAxes = loopedAxes;
        }

        internal int PtxLines { get; }
        internal int Loads { get; }
        internal int VectorLoads { get; }
        internal int ElidedGuards { get; }
        internal int LoopedAxes { get; }

        internal string ToRow(string kernel) => string.Join("\t",
            kernel,
            PtxLines.ToString(CultureInfo.InvariantCulture),
            Loads.ToString(CultureInfo.InvariantCulture),
            VectorLoads.ToString(CultureInfo.InvariantCulture),
            ElidedGuards.ToString(CultureInfo.InvariantCulture),
            LoopedAxes.ToString(CultureInfo.InvariantCulture));

        public bool Equals(Metrics? other) => other is not null &&
            PtxLines == other.PtxLines && Loads == other.Loads &&
            VectorLoads == other.VectorLoads && ElidedGuards == other.ElidedGuards &&
            LoopedAxes == other.LoopedAxes;

        public override bool Equals(object? obj) => Equals(obj as Metrics);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = PtxLines;
                hash = hash * 397 ^ Loads;
                hash = hash * 397 ^ VectorLoads;
                hash = hash * 397 ^ ElidedGuards;
                return hash * 397 ^ LoopedAxes;
            }
        }

        public override string ToString() => ToRow(string.Empty).TrimStart('\t');
    }
#endif

    private static Metrics Measure(CodegenKernelSpec spec)
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, Major, Minor);
        int lines = ptx.Split('\n').Length;
        return new Metrics(lines, emitter.EmittedLoads, emitter.VectorisedLoads,
                           emitter.ElidedGuards, emitter.LoopedAxes);
    }

    private static string BaselinePath()
    {
        // Walk up to the repository root so the baseline lives with the source, not
        // in bin/ where it would be regenerated on every clean build.
        //
        // .git is tested as a FILE as well as a directory: in a git worktree it is a
        // file containing a gitdir pointer, so a directory-only check walks past the
        // root and silently seeds the baseline under bin/.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null &&
               !Directory.Exists(Path.Combine(dir.FullName, ".git")) &&
               !File.Exists(Path.Combine(dir.FullName, ".git")))
            dir = dir.Parent;
        string root = dir?.FullName ?? AppContext.BaseDirectory;
        return Path.Combine(root, "tests", "AiDotNet.Tensors.Tests", "Engines", "Codegen",
                            "codegen-static-baseline.tsv");
    }

    /// <summary>
    /// Every catalog kernel's device-free metrics must match the checked-in baseline
    /// exactly. Regressions and unexplained improvements both fail.
    /// </summary>
    [Fact]
    public void StaticMetrics_MatchTheCheckedInBaseline()
    {
        var actual = new Dictionary<string, Metrics>(StringComparer.Ordinal);
        foreach (var entry in CodegenKernelCatalog.All)
            actual[entry.Name] = Measure(entry.Bench);

        string path = BaselinePath();
        if (!File.Exists(path))
        {
            var seed = new StringBuilder();
            seed.AppendLine("kernel\tptx_lines\tloads\tvector_loads\telided_guards\tlooped_axes");
            foreach (var entry in CodegenKernelCatalog.All)
                seed.AppendLine(actual[entry.Name].ToRow(entry.Name));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, seed.ToString());
            Assert.Fail("No baseline existed; one was written to " + path +
                        ". Review it and commit it, then re-run.");
        }

        var expected = new Dictionary<string, Metrics>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] f = line.Split('\t');
            expected[f[0]] = new Metrics(
                int.Parse(f[1], CultureInfo.InvariantCulture),
                int.Parse(f[2], CultureInfo.InvariantCulture),
                int.Parse(f[3], CultureInfo.InvariantCulture),
                int.Parse(f[4], CultureInfo.InvariantCulture),
                int.Parse(f[5], CultureInfo.InvariantCulture));
        }

        var problems = new List<string>();
        foreach (var pair in actual)
        {
            if (!expected.TryGetValue(pair.Key, out var want))
            {
                problems.Add(pair.Key + " is new; add it to the baseline: " + pair.Value.ToRow(pair.Key));
                continue;
            }
            if (!want.Equals(pair.Value))
                problems.Add(pair.Key + Environment.NewLine +
                             "    baseline " + want + Environment.NewLine +
                             "    actual   " + pair.Value);
        }
        foreach (string gone in expected.Keys.Where(k => !actual.ContainsKey(k)))
            problems.Add(gone + " is in the baseline but no longer in the catalog.");

        if (problems.Count > 0)
            Assert.Fail("Codegen static metrics moved. If the change is intended, update " +
                        path + ":" + Environment.NewLine + string.Join(Environment.NewLine, problems));
    }

    /// <summary>
    /// The same spec must produce byte-identical PTX every time and from every
    /// emitter instance. Content-addressed cubins are keyed on the PTX text, so any
    /// nondeterminism silently defeats the artifact cache and makes a released hash
    /// unreproducible.
    /// </summary>
    [Fact]
    public void Emission_IsByteIdenticalAcrossRunsAndInstances()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            var shared = new PtxAffineEmitter();
            string first = shared.Emit(entry.Bench, Major, Minor);
            string second = shared.Emit(entry.Bench, Major, Minor);
            Assert.True(string.Equals(first, second, StringComparison.Ordinal),
                entry.Name + ": re-emitting from the same instance changed the PTX.");

            string fromFresh = new PtxAffineEmitter().Emit(entry.Bench, Major, Minor);
            Assert.True(string.Equals(first, fromFresh, StringComparison.Ordinal),
                entry.Name + ": a fresh emitter produced different PTX.");
        }
    }

    /// <summary>
    /// The PTX ISA version must be able to name the target. It was previously fixed at
    /// 7.1 while the target was parameterised, so emitting for sm_90 produced PTX that
    /// ptxas rejects: sm_89 and sm_90 did not exist until ISA 7.8.
    /// </summary>
    [Theory]
    [InlineData(7, 5, "7.1")]
    [InlineData(8, 0, "7.1")]
    [InlineData(8, 6, "7.1")]
    [InlineData(8, 7, "7.4")]
    [InlineData(8, 9, "7.8")]
    [InlineData(9, 0, "7.8")]
    [InlineData(10, 0, "8.6")]
    [InlineData(12, 0, "8.7")]
    public void PtxVersion_CanNameTheTarget(int major, int minor, string expectedVersion)
    {
        var entry = CodegenKernelCatalog.Find("depthwise_conv2d_3x3");
        Assert.NotNull(entry);

        string ptx = new PtxAffineEmitter().Emit(entry!.Bench, major, minor);
        string[] lines = ptx.Split('\n');

        Assert.Equal(".version " + expectedVersion, lines[0].Trim());
        Assert.Equal(".target sm_" + major.ToString(CultureInfo.InvariantCulture) +
                     minor.ToString(CultureInfo.InvariantCulture), lines[1].Trim());
    }

    /// <summary>
    /// Changing the architecture must change ONLY the header. A kernel body that
    /// varied with the target would mean the arch parameter is silently steering
    /// codegen, and the sm_86 evidence would not transfer to any other card.
    /// </summary>
    [Fact]
    public void ArchitectureAffectsOnlyTheHeader()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            string[] ampere = new PtxAffineEmitter().Emit(entry.Bench, 8, 6).Split('\n');
            string[] hopper = new PtxAffineEmitter().Emit(entry.Bench, 9, 0).Split('\n');

            Assert.Equal(ampere.Length, hopper.Length);
            for (int i = 2; i < ampere.Length; i++)
                Assert.True(string.Equals(ampere[i], hopper[i], StringComparison.Ordinal),
                    entry.Name + ": line " + i + " differs between sm_86 and sm_90 -- " +
                    "the architecture is steering the kernel body, not just the header.");
        }
    }
}
