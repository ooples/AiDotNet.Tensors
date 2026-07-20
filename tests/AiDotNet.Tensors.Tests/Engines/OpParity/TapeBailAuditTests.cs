using System.Reflection;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Guards the class of bug the forward-parity harness is structurally blind to: a GPU override that bails
/// to the CPU base whenever a tape is active.
///
/// WHY THE EXISTING HARNESS CANNOT SEE THIS: OpParityTests compares CPU and GPU FORWARD results. A tape bail
/// falls back to the CPU implementation, which returns the CORRECT forward answer — so parity passes while
/// the GPU kernel never runs and every training step silently uses the CPU path. Nothing in
/// Engines/OpParity referenced GradientTape before this file.
///
/// WHAT A BAIL COSTS: DirectGpuTensorEngine.MissingKernels.cs opens ~105 overrides with
///   if (IsTapeActive&lt;T&gt;() || GraphMode.IsActive || ...) return base.Op(...);
/// Three cases, only one of which is a defect:
///   - COMPOSE-ONLY forward (no backend.* kernel): sub-ops record their own nodes, so gradients stay correct.
///     Bailing is a choice about tape weight, not correctness.
///   - KERNEL-BACKED forward with NO named CpuEngine backward: recording is impossible without writing one,
///     so bailing is REQUIRED — the GPU result would otherwise carry no gradient at all.
///   - KERNEL-BACKED forward WITH a named CpuEngine backward: the bail is a pure loss. The node can simply be
///     recorded, as RFFT, ScaledDotProductAttention, TensorCosh and TensorSinh now do.
///
/// The third case is what this test pins. It fails when a kernel-backed forward that COULD record a tape
/// node bails instead — either a regression on a fixed op, or a newly added op that repeated the pattern.
///
/// It is a source audit rather than a runtime check on purpose: it needs no GPU, so it runs in CI on any
/// machine, and it covers every op at once instead of requiring a hand-written case per op.
/// </summary>
public class TapeBailAuditTests
{
    private readonly ITestOutputHelper _out;
    public TapeBailAuditTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Walks up from the test binaries to the repository root.
    /// </summary>
    /// <remarks>
    /// Anchored on the source file this audit actually reads rather than on a solution filename — the
    /// solution here is AiDotNet.Tensors.slnx, not .sln, and hardcoding either couples the test to a name
    /// that can change. Checking for the engine source means the walk succeeds exactly when the audit has
    /// something to audit.
    /// </remarks>
    private static string RepoRoot()
    {
        var probe = Path.Combine("src", "AiDotNet.Tensors", "Engines", "CpuEngine.cs");
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, probe)))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException(
            $"could not locate repo root by walking up from {AppContext.BaseDirectory} looking for {probe}");
    }

    /// <summary>Ops known to bail while a reusable backward exists, pending individual verification.</summary>
    /// <remarks>
    /// Every entry is a GPU kernel sitting idle during training. Removing one from this list means: drop
    /// IsTapeActive from its forward bail, record the node with the SAME backward and saved state CpuEngine
    /// uses, and add a gradient-equivalence test. Verify the OVERLOAD matches — TensorMax/TensorMin were
    /// excluded from this list precisely because CpuEngine records only their (tensor, tensor) form while the
    /// GPU override implements (tensor, scalar), so reusing that backward would produce wrong gradients.
    /// This list must only ever shrink.
    /// </remarks>
    private static readonly string[] KnownUnfixed =
    {
        "AffineGrid",
        "MaxPool3DWithIndices",
        "RBFKernel",

        // Reverted after a gradient test FAILED it. Recording CpuEngine's ScatterBackward on the GPU result
        // gave d(values) exactly right but d(input) wrong by 3.14e-01. Scatter overwrites input at the
        // scattered positions, so d/d(input) must be zero there — a mask the reused recording did not
        // reproduce. Signature matching was not enough: only the gradient is wrong, the forward is perfect,
        // so forward parity could never surface it. Needs its own GPU-side backward, not a reused one.
        "Scatter",

        // Reverted after testing: removing its bail surfaced that the GPU path throws
        // InvalidOperationException("CUDA kernel not found: where_select") — CudaBackend.Where references a
        // kernel with no source and no registration anywhere, so backend.Where is non-functional on CUDA.
        // The bail was concealing that, since without it the path was only reachable outside a tape and had
        // no test. Fix where_select first, then re-apply the recording.
        "Sparsemax",

        // NOT fixable, and NOT pending — these are here because the audit matches on op NAME and cannot
        // tell overloads apart. CpuEngine records a backward for TensorMax/TensorMin(tensor, TENSOR), while
        // the GPU override implements TensorMax/TensorMin(tensor, SCALAR). The scalar overload has NO
        // recording on either engine, so reusing the tensor-overload backward would attribute gradient to a
        // second operand that does not exist — silently wrong gradients rather than a crash. Bailing changes
        // nothing here: the CPU path produces no gradient for the scalar form either. (That the scalar
        // overloads are non-differentiable at all is a separate question, not this audit's.)
        "TensorMax",
        "TensorMin",
    };

    /// <summary>Ops already fixed — they must never regress to bailing.</summary>
    private static readonly string[] MustNotBail =
    {
        "RFFT",
        "IRFFT",
        "ScaledDotProductAttention",
        "TensorCosh",
        "TensorSinh",
        "Upsample3D",
    };

    private static Dictionary<string, (bool HasKernel, bool Bails)> ScanGpuOverrides(string gpuSrc)
    {
        var lines = File.ReadAllLines(gpuSrc);
        var methodRe = new Regex(@"(?:Tensor<T> IEngine\.|public override Tensor<T> |void IEngine\.)([A-Za-z0-9_]+)<T>\s*\(");
        var result = new Dictionary<string, (bool, bool)>();

        for (int i = 0; i < lines.Length; i++)
        {
            var m = methodRe.Match(lines[i]);
            if (!m.Success) continue;

            string name = m.Groups[1].Value;
            int depth = 0; bool started = false;
            var body = new List<string>();
            for (int j = i; j < lines.Length && j < i + 400; j++)
            {
                body.Add(lines[j]);
                depth += lines[j].Count(c => c == '{') - lines[j].Count(c => c == '}');
                if (lines[j].Contains('{')) started = true;
                if (started && depth <= 0 && j > i) break;
            }

            // Strip line comments BEFORE analysing. A fixed op documents why its bail was removed, and
            // those comments name IsTapeActive — matching raw text therefore reported every fixed op as
            // still bailing. The audit must read code, not prose.
            string text = string.Join("\n",
                body.Select(l =>
                {
                    int c = l.IndexOf("//", StringComparison.Ordinal);
                    return c >= 0 ? l.Substring(0, c) : l;
                }));

            bool hasKernel = Regex.IsMatch(text, @"\bbackend\.[A-Za-z0-9_]+\(");
            // The generic call form, so a mention in a string or identifier cannot trip it.
            bool bails = Regex.IsMatch(text, @"\bIsTapeActive\s*<");
            // An op can appear more than once (overloads); kernel/bail status ORs across them.
            if (result.TryGetValue(name, out var prev))
                result[name] = (prev.Item1 || hasKernel, prev.Item2 || bails);
            else
                result[name] = (hasKernel, bails);
        }
        return result;
    }

    private static HashSet<string> OpsWithCpuBackward(string cpuSrc)
    {
        string cpu = File.ReadAllText(cpuSrc);
        var found = new HashSet<string>();
        foreach (Match m in Regex.Matches(cpu, @"DifferentiableOps\.Record[A-Za-z]*\(\s*""([A-Za-z0-9_]+)"""))
            found.Add(m.Groups[1].Value);
        return found;
    }

    /// <summary>
    /// No kernel-backed GPU forward with a reusable backward may bail on the tape, except those explicitly
    /// listed as pending. A new violation means a GPU kernel silently stopped running during training.
    /// </summary>
    [Fact]
    public void No_new_kernel_backed_op_bails_on_an_active_tape()
    {
        string root = RepoRoot();
        string gpuSrc = Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "DirectGpuTensorEngine.MissingKernels.cs");
        string cpuSrc = Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "CpuEngine.cs");
        Assert.True(File.Exists(gpuSrc), $"GPU override source not found: {gpuSrc}");
        Assert.True(File.Exists(cpuSrc), $"CpuEngine source not found: {cpuSrc}");

        var overrides = ScanGpuOverrides(gpuSrc);
        var withBackward = OpsWithCpuBackward(cpuSrc);

        var violations = new List<string>();
        foreach (var (name, info) in overrides)
        {
            // Backward ops are exempt: ComputeGradients nulls the current tape before replaying, so their
            // IsTapeActive check is already false during backprop and the bail never fires there.
            if (name.EndsWith("Backward", StringComparison.Ordinal)) continue;
            if (!info.Bails || !info.HasKernel) continue;
            if (!withBackward.Contains(name)) continue;          // no backward to record — bail is correct
            if (KnownUnfixed.Contains(name)) continue;           // tracked, pending verification
            violations.Add(name);
        }

        _out.WriteLine($"GPU overrides scanned      : {overrides.Count}");
        _out.WriteLine($"…kernel-backed and bailing : {overrides.Count(kv => kv.Value.Bails && kv.Value.HasKernel)}");
        _out.WriteLine($"known-unfixed (allowed)    : {KnownUnfixed.Length}");
        _out.WriteLine($"new violations             : {violations.Count}");

        Assert.True(violations.Count == 0,
            "These GPU forwards bail on an active tape even though CpuEngine names a backward that could be "
            + "recorded, so their kernels do not run during training: " + string.Join(", ", violations)
            + ". Either record the node (drop IsTapeActive, add RecordIfActive/RecordUnary with the SAME "
            + "backward and saved state CpuEngine uses, and add a gradient test), or add it to KnownUnfixed "
            + "with a reason.");
    }

    /// <summary>Ops already converted must not silently regress to bailing.</summary>
    [Fact]
    public void Already_fixed_ops_do_not_regress_to_bailing()
    {
        string root = RepoRoot();
        var overrides = ScanGpuOverrides(
            Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "DirectGpuTensorEngine.MissingKernels.cs"));

        var regressed = MustNotBail.Where(op => overrides.TryGetValue(op, out var i) && i.Bails).ToList();

        _out.WriteLine($"checked: {string.Join(", ", MustNotBail)}");
        Assert.True(regressed.Count == 0,
            "These ops previously recorded their tape node on the GPU result and have regressed to bailing, "
            + "which puts them back on the CPU for every training step: " + string.Join(", ", regressed));
    }

    /// <summary>
    /// The KnownUnfixed list is a debt ledger, so it must stay accurate: an entry that no longer bails (or
    /// no longer exists) should be removed, otherwise the list stops meaning anything.
    /// </summary>
    [Fact]
    public void Known_unfixed_list_has_no_stale_entries()
    {
        string root = RepoRoot();
        var overrides = ScanGpuOverrides(
            Path.Combine(root, "src", "AiDotNet.Tensors", "Engines", "DirectGpuTensorEngine.MissingKernels.cs"));

        var stale = KnownUnfixed
            .Where(op => !overrides.TryGetValue(op, out var i) || !i.Bails)
            .ToList();

        Assert.True(stale.Count == 0,
            "KnownUnfixed lists ops that no longer bail (or no longer exist). Remove them so the list keeps "
            + "reflecting real debt: " + string.Join(", ", stale));
    }
}
