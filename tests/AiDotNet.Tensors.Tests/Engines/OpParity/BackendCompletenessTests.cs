// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). Backend completeness guard.
//
// GOAL: every IEngine tensor operation must have a REAL implementation everywhere it can run —
// on the CPU engine AND as a genuine GPU path (not a silent `return base.Op(...)` CPU fallback),
// and no GPU backend may ship a kernel that is only a `throw new NotSupportedException` stub.
//
// This turns "which ops are missing a GPU/backend implementation" from tribal knowledge into an
// automatically enforced, always-visible worklist: the reports list every gap, and the floors
// ratchet so a regression (an op quietly losing its GPU override, or a backend gaining a stub)
// fails the build. Drive the floors to zero by writing the missing kernels for all backends.
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Enforces that the GPU dispatch surface keeps pace with the CPU op surface. Two tiers:
/// (1) engine-level — <see cref="DirectGpuTensorEngine"/> must override every tensor-returning
/// IEngine op, so nothing silently runs on the CPU when a caller asked for the GPU engine;
/// (2) backend-level — none of the six GPU backends may implement a kernel-surface method as a
/// bare NotSupported/NotImplemented stub. Both tiers write a full gap report and ratchet a floor.
/// </summary>
public sealed class BackendCompletenessTests
{
    /// <summary>The six concrete GPU backends. All must provide real kernels for every op.</summary>
    private static readonly Type[] GpuBackends =
    {
        typeof(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend),
        typeof(AiDotNet.Tensors.Engines.DirectGpu.HIP.HipBackend),
        typeof(AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend),
        typeof(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend),
        typeof(AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend),
        typeof(AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend),
    };

    // ---- Tier 1: engine-level GPU override completeness -------------------------------------

    /// <summary>Ops that legitimately have no distinct GPU kernel — they are pure shape/metadata
    /// manipulation that the CPU path already does with zero compute (reshape/view/transpose-of-
    /// metadata), so a GPU override would only add a needless round-trip. Everything NOT on this
    /// list must have a real DirectGpuTensorEngine override. Keep this list SMALL and justified.</summary>
    private static readonly HashSet<string> ZeroComputeOps = new(StringComparer.Ordinal)
    {
        // (intentionally empty for now — every gap is surfaced until proven zero-compute)
    };

    // Current count of tensor ops still lacking a GPU override. GOAL: 0. This floor only ratchets
    // DOWN — add a GPU kernel, lower the number. It must never be raised to hide a new gap.
    private const int GpuOverrideMissingFloor = 117;

    [Fact]
    public void EveryTensorOp_HasRealGpuEngineOverride()
    {
        var map = typeof(DirectGpuTensorEngine).GetInterfaceMap(typeof(IEngine));

        // Every IEngine method (any return type) that DirectGpuTensorEngine actually implements on
        // the GPU. Includes the allocate-free `<Name>Into` siblings — a tensor-returning op counts
        // as GPU-covered if EITHER it OR its `<Name>Into` variant has a real GPU override, since the
        // Into variant is where the GPU kernel lives for the compiled/resident dispatch path.
        var gpuOverridden = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < map.InterfaceMethods.Length; i++)
            if (map.TargetMethods[i].DeclaringType == typeof(DirectGpuTensorEngine))
                gpuOverridden.Add(map.InterfaceMethods[i].Name);

        var missing = new SortedSet<string>(StringComparer.Ordinal);
        var covered = new SortedSet<string>(StringComparer.Ordinal);

        for (int i = 0; i < map.InterfaceMethods.Length; i++)
        {
            var iface = map.InterfaceMethods[i];
            if (!IsTensorReturning(iface)) continue;
            bool hasGpuKernel = gpuOverridden.Contains(iface.Name) || gpuOverridden.Contains(iface.Name + "Into");
            if (hasGpuKernel) covered.Add(iface.Name);
            else if (!ZeroComputeOps.Contains(iface.Name)) missing.Add(iface.Name);
        }
        // An op can appear covered via one overload and missing via another; coverage wins.
        missing.ExceptWith(covered);

        // NOTE: "missing" here means "no DEDICATED GPU override". Many such ops still run on the GPU
        // via composition — e.g. CpuEngine.TensorSquare is literally TensorMultiply(x,x), and
        // TensorMultiply IS a GPU-resident override, so TensorSquare executes on-device even though it
        // has no override of its own. This metric therefore tracks dedicated-override coverage (a perf/
        // residency bar), NOT silent-CPU-fallback. The genuinely CPU-bound ops are the subset whose
        // CpuEngine body does raw scalar/array loops without reaching a GPU-resident primitive (e.g.
        // BatchNormBackward before it was wired) — those are the priority. Reflection can't tell the
        // two apart statically, so treat the report as a superset worklist.
        WriteReport("gpu-engine-missing.md",
            $"tensor-returning IEngine ops with a dedicated GPU override : {covered.Count}",
            $"tensor-returning IEngine ops WITHOUT a dedicated override   : {missing.Count} " +
            $"(goal 0; many still run on GPU via composed primitives)",
            covered, missing);

        Assert.True(missing.Count <= GpuOverrideMissingFloor,
            $"{missing.Count} tensor IEngine ops lack a dedicated GPU override (floor " +
            $"{GpuOverrideMissingFloor}). A new op lost/never had one. See gpu-engine-missing.md. " +
            $"Add a dedicated override (or verify it already composes onto GPU primitives); do not raise the floor.");
    }

    // ---- Tier 2: backend-level kernel stub detection ---------------------------------------

    // Per-backend count of kernel-surface methods currently implemented as a NotSupported/
    // NotImplemented stub. GOAL: 0 for every backend. Ratchets DOWN only.
    private const int BackendStubFloor = 2;

    [Fact]
    public void NoGpuBackend_LeavesAKernelAsANotImplementedStub()
    {
        var sb = new StringBuilder();
        sb.AppendLine("# GPU backend kernel-stub report (#775)");
        sb.AppendLine("Methods on each backend that are only `throw new NotSupported/NotImplementedException`.");
        sb.AppendLine();

        int worst = 0;
        string worstName = "";
        foreach (var backend in GpuBackends)
        {
            var stubs = KernelSurfaceMethods(backend)
                .Where(IsNotImplementedStub)
                .Select(m => Signature(m))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();
            sb.AppendLine($"## {backend.Name} — {stubs.Count} stub(s)");
            foreach (var s in stubs) sb.AppendLine($"  [ ] {s}");
            sb.AppendLine();
            if (stubs.Count > worst) { worst = stubs.Count; worstName = backend.Name; }
        }
        WriteRaw("gpu-backend-stubs.md", sb.ToString());

        Assert.True(worst <= BackendStubFloor,
            $"{worstName} has {worst} kernel-surface methods stubbed with NotSupported/NotImplemented " +
            $"(floor {BackendStubFloor}). Implement the real kernel; do not raise the floor. See gpu-backend-stubs.md.");
    }

    // ---- helpers ---------------------------------------------------------------------------

    private static bool IsTensorReturning(MethodInfo m)
    {
        var rt = m.ReturnType;
        return rt.IsGenericType && rt.Name == "Tensor`1";
    }

    /// <summary>All methods that satisfy one of the backend "kernel surface" interfaces the type
    /// implements (IDirectGpuBackend + the segmented IAudioBackend/IGeometryBackend/ILinalgBackend/
    /// IParity210Backend/... surfaces). These are the methods that MUST be real kernels.</summary>
    private static IEnumerable<MethodInfo> KernelSurfaceMethods(Type backend)
    {
        var seen = new HashSet<MethodInfo>();
        foreach (var iface in backend.GetInterfaces())
        {
            // Only the GPU/kernel backend interfaces (not IDisposable et al.).
            if (!iface.Name.Contains("Backend") && !iface.Name.Contains("Kernels")) continue;
            InterfaceMapping map;
            try { map = backend.GetInterfaceMap(iface); }
            catch { continue; }
            foreach (var t in map.TargetMethods)
                if (t.DeclaringType == backend && seen.Add(t))
                    yield return t;
        }
    }

    // Operand-size tables derived once from System.Reflection.Emit.OpCodes so IL stepping is exact.
    private static readonly System.Reflection.Emit.OperandType?[] s_oneByte = BuildOpTable(false);
    private static readonly System.Reflection.Emit.OperandType?[] s_twoByte = BuildOpTable(true);

    private static System.Reflection.Emit.OperandType?[] BuildOpTable(bool twoByte)
    {
        var table = new System.Reflection.Emit.OperandType?[256];
        foreach (var f in typeof(System.Reflection.Emit.OpCodes)
                     .GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.GetValue(null) is not System.Reflection.Emit.OpCode oc) continue;
            int v = (ushort)oc.Value;
            bool isTwo = (v & 0xFF00) == 0xFE00;
            if (isTwo != twoByte) continue;
            table[v & 0xFF] = oc.OperandType;
        }
        return table;
    }

    private static int OperandSize(System.Reflection.Emit.OperandType t, byte[] il, int operandPos)
    {
        switch (t)
        {
            case System.Reflection.Emit.OperandType.InlineNone: return 0;
            case System.Reflection.Emit.OperandType.ShortInlineBrTarget:
            case System.Reflection.Emit.OperandType.ShortInlineI:
            case System.Reflection.Emit.OperandType.ShortInlineVar: return 1;
            case System.Reflection.Emit.OperandType.InlineVar: return 2;
            case System.Reflection.Emit.OperandType.InlineI8:
            case System.Reflection.Emit.OperandType.InlineR: return 8;
            case System.Reflection.Emit.OperandType.InlineSwitch:
                int n = operandPos + 4 <= il.Length ? BitConverter.ToInt32(il, operandPos) : 0;
                return 4 + 4 * n;
            default: return 4; // InlineBrTarget/Field/I/Method/Sig/String/Tok/Type/ShortInlineR
        }
    }

    /// <summary>True if the method body is a short stub whose only real act is to throw
    /// NotImplementedException/NotSupportedException — i.e. an unimplemented kernel.</summary>
    private static bool IsNotImplementedStub(MethodInfo m)
    {
        MethodBody? body;
        try { body = m.GetMethodBody(); } catch { return false; }
        if (body is null) return false;
        var il = body.GetILAsByteArray();
        if (il is null || il.Length == 0 || il.Length > 64) return false; // real kernels are longer

        var module = m.Module;
        var typeArgs = m.DeclaringType is { IsGenericType: true } dt ? dt.GetGenericArguments() : null;
        int pos = 0;
        while (pos < il.Length)
        {
            int b = il[pos++];
            System.Reflection.Emit.OperandType? ot;
            bool isNewobj;
            if (b == 0xFE)
            {
                if (pos >= il.Length) break;
                int b2 = il[pos++];
                ot = s_twoByte[b2];
                isNewobj = false;
            }
            else
            {
                ot = s_oneByte[b];
                isNewobj = b == 0x73;
            }
            if (ot is null) return false; // unknown opcode — bail rather than mis-step

            if (isNewobj && pos + 4 <= il.Length)
            {
                int token = BitConverter.ToInt32(il, pos);
                try
                {
                    var ctor = module.ResolveMethod(token, typeArgs, null);
                    var dt2 = ctor?.DeclaringType;
                    if (dt2 == typeof(NotImplementedException) || dt2 == typeof(NotSupportedException))
                        return true;
                }
                catch { /* unresolvable token in a generic context — ignore */ }
            }
            pos += OperandSize(ot.Value, il, pos);
        }
        return false;
    }

    private static string Signature(MethodInfo m) =>
        $"{m.Name}({string.Join(",", m.GetParameters().Select(p => p.ParameterType.Name))})";

    private static void WriteReport(string file, string headerCovered, string headerMissing,
        IEnumerable<string> covered, IEnumerable<string> missing)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# GPU engine op-override completeness (#775)");
        sb.AppendLine(headerCovered);
        sb.AppendLine(headerMissing);
        sb.AppendLine();
        sb.AppendLine("## missing a GPU override (runs on CPU) — the worklist");
        foreach (var n in missing) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine("## has a real GPU override");
        foreach (var n in covered) sb.AppendLine($"  [x] {n}");
        WriteRaw(file, sb.ToString());
    }

    private static void WriteRaw(string file, string content)
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, file), content);
        }
        catch { /* best-effort */ }
    }
}
#endif
