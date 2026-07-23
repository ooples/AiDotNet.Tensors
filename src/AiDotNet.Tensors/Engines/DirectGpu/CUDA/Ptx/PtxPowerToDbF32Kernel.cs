using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Power-to-decibel conversion for issue #850, matching the NVRTC <c>power_to_db</c> kernel:
/// <c>db[i] = max(10*log10(max(power[i], 1e-10) / refValue^2), minDb)</c>. PTX has no log10 primitive, so
/// the base-10 log is computed as <c>lg2.approx(ratio) * (10*log10(2))</c>; the specialization is therefore
/// covered by a TOLERANCE-based parity spec, not bit-exact. The <c>refValue</c> and <c>minDb</c> scalars
/// are per-launch <c>.param .f32</c>. The element <c>count</c> is baked and the launch covers it exactly
/// (no bounds guard). Two pointers plus two f32 scalars reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxPowerToDbF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_power_to_db_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPowerToDbF32Kernel(DirectPtxRuntime runtime, int count, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in power-to-db specialization is admitted only on SM86.");
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView power, DirectPtxTensorView db, float refValue, float minDb)
    {
        Require(power, Blueprint.Tensors[0], nameof(power));
        Require(db, Blueprint.Tensors[1], nameof(db));

        IntPtr powerPointer = power.Pointer, dbPointer = db.Pointer;
        float refValueArg = refValue, minDbArg = minDb;
        void** arguments = stackalloc void*[4];
        arguments[0] = &powerPointer;
        arguments[1] = &dbPointer;
        arguments[2] = &refValueArg;
        arguments[3] = &minDbArg;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int count, int blockThreads = DefaultBlockThreads)
    {
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        string tiny = Hex(1e-10f);
        string tenLog10Of2 = Hex((float)(10.0 * Math.Log10(2.0)));   // 10*log10(x) = lg2(x) * (10*log10 2)

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape count={count} block={blockThreads} op=power-to-db");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 power_ptr,");
        ptx.AppendLine("    .param .u64 db_ptr,");
        ptx.AppendLine("    .param .f32 ref_val,");
        ptx.AppendLine("    .param .f32 min_db");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [power_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [db_ptr];");
        ptx.AppendLine("    ld.param.f32 %f3, [ref_val];");
        ptx.AppendLine("    ld.param.f32 %f4, [min_db];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");        // power
        ptx.AppendLine($"    max.f32 %f0, %f0, {tiny};");          // max(power, 1e-10)
        ptx.AppendLine("    mul.rn.f32 %f1, %f3, %f3;");            // refSq
        ptx.AppendLine("    div.rn.f32 %f0, %f0, %f1;");            // ratio
        ptx.AppendLine("    lg2.approx.f32 %f0, %f0;");             // log2(ratio)
        ptx.AppendLine($"    mul.rn.f32 %f0, %f0, {tenLog10Of2};"); // 10*log10(ratio)
        ptx.AppendLine("    max.f32 %f0, %f0, %f4;");               // max(dbVal, minDb)
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int count, int blockThreads)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "power-to-db-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{count}",
            Tensors:
            [
                new("power", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("db", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "db[i] = max(10*log10(max(power[i],1e-10)/refValue^2), minDb)",
                ["mode"] = "inference-forward-power-to-db",
                ["arithmetic"] = "lg2.approx scaled by 10*log10(2); tolerance-based parity, not bit-exact",
                ["scalars"] = "refValue and minDb are per-launch .param .f32",
                ["bounds-check"] = "none - the launch covers exactly the element count",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int count) =>
        count >= 256 && count % DefaultBlockThreads == 0 && count <= (1 << 24);

    internal static bool IsPromotedShape(int count) => false;

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The power-to-db family supports counts n>=256 that are a multiple of 256, up to 2^24.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Power-to-db block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
