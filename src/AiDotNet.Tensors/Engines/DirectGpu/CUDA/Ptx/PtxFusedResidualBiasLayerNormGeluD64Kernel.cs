using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Inference transformer boundary over exact D=64 rows. One warp reads input,
/// residual, and pre-normalization bias once, retains both row values through
/// LayerNorm, affine, and tanh-GELU, then performs the sole output store.
/// </summary>
internal sealed class PtxFusedResidualBiasLayerNormGeluD64Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_residual_bias_layernorm_gelu_d64";
    internal const int Dimension = 64;
    internal const int WarpsPerBlock = 8;
    internal const int BlockThreads = WarpsPerBlock * 32;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal float Epsilon { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedResidualBiasLayerNormGeluD64Kernel(
        DirectPtxRuntime runtime,
        int rows,
        float epsilon = 1e-5f)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in residual+bias+LayerNorm+GELU specialization is measured only on GA10x/SM86.");
        Validate(rows, epsilon);
        Rows = rows;
        Epsilon = epsilon;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows, epsilon);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView residual,
        DirectPtxTensorView preNormBias,
        DirectPtxTensorView gamma,
        DirectPtxTensorView beta,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(residual, Blueprint.Tensors[1], nameof(residual));
        Require(preNormBias, Blueprint.Tensors[2], nameof(preNormBias));
        Require(gamma, Blueprint.Tensors[3], nameof(gamma));
        Require(beta, Blueprint.Tensors[4], nameof(beta));
        Require(output, Blueprint.Tensors[5], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, residual) ||
            Overlaps(output, preNormBias) || Overlaps(output, gamma) || Overlaps(output, beta))
            throw new ArgumentException("The fused normalization output may not alias an input tensor.");

        IntPtr inputPointer = input.Pointer;
        IntPtr residualPointer = residual.Pointer;
        IntPtr biasPointer = preNormBias.Pointer;
        IntPtr gammaPointer = gamma.Pointer;
        IntPtr betaPointer = beta.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &inputPointer;
        arguments[1] = &residualPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &gammaPointer;
        arguments[4] = &betaPointer;
        arguments[5] = &outputPointer;
        _module.Launch(
            _function, (uint)((Rows + WarpsPerBlock - 1) / WarpsPerBlock), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int rows,
        float epsilon)
    {
        Validate(rows, epsilon);
        string epsilonHex = FloatLiteral(epsilon);
        var ptx = new StringBuilder(10_240);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 residual_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 gamma_ptr,");
        ptx.AppendLine("    .param .u64 beta_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [residual_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [gamma_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [beta_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {WarpsPerBlock}, %r2;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
        ptx.AppendLine("    @%p0 bra LN_GELU_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 256;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd9];");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd9+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd10];");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd10+128];");
        ptx.AppendLine("    add.rn.f32 %f6, %f0, %f2;");
        ptx.AppendLine("    add.rn.f32 %f6, %f6, %f4;");
        ptx.AppendLine("    add.rn.f32 %f7, %f1, %f3;");
        ptx.AppendLine("    add.rn.f32 %f7, %f7, %f5;");
        ptx.AppendLine("    add.rn.f32 %f8, %f6, %f7;");
        EmitWarpSum(ptx, "%f8", "%f9");
        ptx.AppendLine("    mul.rn.f32 %f10, %f8, 0f3C800000;");
        ptx.AppendLine("    sub.rn.f32 %f11, %f6, %f10;");
        ptx.AppendLine("    sub.rn.f32 %f12, %f7, %f10;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f11, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f13, %f12, %f12, %f13;");
        EmitWarpSum(ptx, "%f13", "%f14");
        ptx.AppendLine("    mul.rn.f32 %f15, %f13, 0f3C800000;");
        ptx.AppendLine($"    add.rn.f32 %f15, %f15, {epsilonHex};");
        ptx.AppendLine("    sqrt.rn.f32 %f16, %f15;");
        ptx.AppendLine("    rcp.approx.f32 %f17, %f16;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd7;");
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f18, [%rd11];");
        ptx.AppendLine("    ld.global.nc.f32 %f19, [%rd11+128];");
        ptx.AppendLine("    ld.global.nc.f32 %f20, [%rd12];");
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd12+128];");
        ptx.AppendLine("    mul.rn.f32 %f22, %f11, %f17;");
        ptx.AppendLine("    fma.rn.f32 %f22, %f22, %f18, %f20;");
        ptx.AppendLine("    mul.rn.f32 %f23, %f12, %f17;");
        ptx.AppendLine("    fma.rn.f32 %f23, %f23, %f19, %f21;");
        EmitGelu(ptx, "%f22", "%f24", "%f26");
        EmitGelu(ptx, "%f23", "%f25", "%f26");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd6;");
        ptx.AppendLine("    add.u64 %rd13, %rd13, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f24;");
        ptx.AppendLine("    st.global.f32 [%rd13+128], %f25;");
        ptx.AppendLine("LN_GELU_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitWarpSum(StringBuilder ptx, string value, string temporary)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r8, {value};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r9, %r8, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {temporary}, %r9;");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {temporary};");
        }
    }

    private static void EmitGelu(
        StringBuilder ptx,
        string input,
        string output,
        string temporary)
    {
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {input}, {input};");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, {input};");
        ptx.AppendLine($"    fma.rn.f32 {temporary}, {temporary}, 0f3D372713, {input};");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, 0f3F4C422A;");
        ptx.AppendLine($"    tanh.approx.f32 {temporary}, {temporary};");
        ptx.AppendLine($"    add.rn.f32 {temporary}, {temporary}, 0f3F800000;");
        ptx.AppendLine($"    mul.rn.f32 {output}, {temporary}, {input};");
        ptx.AppendLine($"    mul.rn.f32 {output}, {output}, 0f3F000000;");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows)
    {
        var matrix = new DirectPtxExtent(rows, Dimension);
        var vector = new DirectPtxExtent(Dimension);
        return new DirectPtxKernelBlueprint(
            Operation: "residual-bias-layernorm-gelu-d64",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w8-r{rows}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("residual", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("pre_norm_bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gamma", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("beta", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vector, vector, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "gelu_tanh(layernorm(input+residual+pre_norm_bias,gamma,beta,epsilon))",
                ["mode"] = "inference-forward-no-saved-statistics",
                ["input"] = "fp32",
                ["accumulator"] = "warp-reduced-fp32-register",
                ["output"] = "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedRows(int rows) => rows is 256 or 2048 or 8192;

    internal static bool IsPromotedRows(int rows) => rows is 2048 or 8192;

    private static void Validate(int rows, float epsilon)
    {
        if (!IsSupportedRows(rows))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first residual LayerNorm+GELU family supports exact row buckets 256, 2048, and 8192.");
        if (!PtxCompat.IsFinite(epsilon) || epsilon <= 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
