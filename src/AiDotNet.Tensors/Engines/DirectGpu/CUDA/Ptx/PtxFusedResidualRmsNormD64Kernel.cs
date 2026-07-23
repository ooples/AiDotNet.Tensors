using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Real transformer boundary: residual addition followed by RMSNorm over D=64.
/// Each warp owns one row, reads both inputs exactly once, keeps the reduction
/// in registers, and writes only the normalized result plus one RMS scalar.
/// </summary>
internal sealed class PtxFusedResidualRmsNormD64Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_residual_rmsnorm_d64";
    internal const int Dimension = 64;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal float Epsilon { get; }

    /// <summary>The register the epsilon launch parameter is loaded into.</summary>
    private const string EpsilonRegister = "%eps";
    internal int WarpsPerBlock { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal nuint InputBytes => checked((nuint)Rows * Dimension * sizeof(float));
    internal nuint OutputBytes => InputBytes;
    internal const nuint GammaBytes = Dimension * sizeof(float);
    internal nuint RmsBytes => checked((nuint)Rows * sizeof(float));

    internal PtxFusedResidualRmsNormD64Kernel(
        DirectPtxRuntime runtime,
        int rows,
        float epsilon = 1e-5f,
        int warpsPerBlock = 4)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (rows <= 0 || rows > 65535) throw new ArgumentOutOfRangeException(nameof(rows));
        if (!PtxCompat.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (warpsPerBlock is not (1 or 2 or 4 or 8))
            throw new ArgumentOutOfRangeException(nameof(warpsPerBlock));
        if (!DirectPtxArchitecture.HasValidatedOnlineAttention(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Residual RMSNorm has no validated SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

        Rows = rows;
        Epsilon = epsilon;
        WarpsPerBlock = warpsPerBlock;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, warpsPerBlock);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, epsilon, rows, warpsPerBlock);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int blockThreads = warpsPerBlock * 32;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            blockThreads, activeBlocks, _module.JitInfoLog);
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int warpsPerBlock)
    {
        var matrix = new DirectPtxExtent(rows, Dimension);
        var vector = new DirectPtxExtent(Dimension);
        var rms = new DirectPtxExtent(rows);
        return new DirectPtxKernelBlueprint(
            Operation: "residual-rmsnorm-d64",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w{warpsPerBlock}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read),
                new("residual", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read),
                new("gamma", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write),
                new("rms", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    rms, rms, 16, DirectPtxTensorAccess.Write)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "rmsnorm(input+residual,gamma)",
                ["input"] = "fp32",
                ["accumulator"] = "fp32",
                ["output"] = "fp32",
                ["layout"] = "row-major-2d",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView residual,
        DirectPtxTensorView gamma,
        DirectPtxTensorView output,
        DirectPtxTensorView rms)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(residual, Blueprint.Tensors[1], nameof(residual));
        Require(gamma, Blueprint.Tensors[2], nameof(gamma));
        Require(output, Blueprint.Tensors[3], nameof(output));
        Require(rms, Blueprint.Tensors[4], nameof(rms));
        if (output.Pointer == input.Pointer || output.Pointer == residual.Pointer)
            throw new ArgumentException("The fused output must not alias either input.", nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr residualPointer = residual.Pointer;
        IntPtr gammaPointer = gamma.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr rmsPointer = rms.Pointer;
        float epsilon = Epsilon;
        void** arguments = stackalloc void*[6];
        arguments[0] = &inputPointer;
        arguments[1] = &residualPointer;
        arguments[2] = &gammaPointer;
        arguments[3] = &outputPointer;
        arguments[4] = &rmsPointer;
        arguments[5] = &epsilon;
        _module.Launch(
            _function, (uint)((Rows + WarpsPerBlock - 1) / WarpsPerBlock), 1, 1,
            (uint)(WarpsPerBlock * 32), 1, 1, 0, arguments);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength < contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        float epsilon,
        int rows = 1,
        int warpsPerBlock = 1)
    {
        if (!PtxCompat.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (warpsPerBlock is not (1 or 2 or 4 or 8))
            throw new ArgumentOutOfRangeException(nameof(warpsPerBlock));
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 residual_ptr,");
        ptx.AppendLine("    .param .u64 gamma_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 rms_ptr,");
        // epsilon is a launch parameter rather than a baked literal. Baking it
        // made the module key depend on a value the blueprint id does not carry,
        // so a caller using a non-default epsilon emitted PTX that no checked-in
        // cubin matched and silently fell back to driver JIT.
        ptx.AppendLine("    .param .f32 epsilon");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine($"    .reg .f32 {EpsilonRegister};");
        ptx.AppendLine($"    ld.param.f32 {EpsilonRegister}, [epsilon];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [residual_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gamma_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [rms_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r4, %r0, 5;");
        ptx.AppendLine("    and.b32 %r0, %r0, 31;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r1, %r1, {warpsPerBlock}, %r4;");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r1, {rows};");
        ptx.AppendLine("    @%p1 bra DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r1, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd5;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd7];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7+128];");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd8];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd8+128];");
        ptx.AppendLine("    add.rn.f32 %f4, %f0, %f2;");
        ptx.AppendLine("    add.rn.f32 %f5, %f1, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f4, %f4;");
        ptx.AppendLine("    fma.rn.f32 %f6, %f5, %f5, %f6;");
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine("    mov.b32 %r2, %f6;");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r3, %r2, {delta}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f7, %r3;");
            ptx.AppendLine("    add.rn.f32 %f6, %f6, %f7;");
        }
        ptx.AppendLine("    mul.rn.f32 %f8, %f6, 0f3C800000;"); // 1/64
        ptx.AppendLine($"    add.rn.f32 %f8, %f8, {EpsilonRegister};");
        ptx.AppendLine("    sqrt.rn.f32 %f9, %f8;");
        ptx.AppendLine("    rcp.approx.f32 %f10, %f9;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f11, [%rd9];");
        ptx.AppendLine("    ld.global.f32 %f12, [%rd9+128];");
        ptx.AppendLine("    mul.rn.f32 %f13, %f4, %f10;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f13, %f11;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f5, %f10;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f14, %f12;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd5;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f13;");
        ptx.AppendLine("    st.global.f32 [%rd10+128], %f14;");
        ptx.AppendLine("    setp.eq.u32 %p0, %r0, 0;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
        ptx.AppendLine("    @%p0 st.global.f32 [%rd11], %f9;");
        ptx.AppendLine("DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
