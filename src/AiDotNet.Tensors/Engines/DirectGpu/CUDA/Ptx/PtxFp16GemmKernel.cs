using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtx16BitInputType
{
    Float16,
    BFloat16
}

internal enum DirectPtxGemmOutputType
{
    Float16,
    Float32
}

/// <summary>
/// Exact-shape direct PTX for the FP16/BF16 GEMM contracts in issue #836.
/// Transposition, batch strides, input dtype, accumulator semantics, and output
/// conversion are baked into the module; the launch ABI contains only pointers.
/// This correctness baseline stays experiment-only until a Tensor-Core variant
/// for the same cell passes the championship gate.
/// </summary>
internal sealed class PtxFp16GemmKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fp16_gemm";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal int K { get; }
    internal int Batch { get; }
    internal bool TransposeA { get; }
    internal bool TransposeB { get; }
    internal DirectPtx16BitInputType InputType { get; }
    internal DirectPtxGemmOutputType OutputType { get; }
    internal bool HalfAccumulate { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFp16GemmKernel(
        DirectPtxRuntime runtime,
        int m,
        int n,
        int k,
        int batch = 1,
        bool transposeA = false,
        bool transposeB = false,
        DirectPtx16BitInputType inputType = DirectPtx16BitInputType.Float16,
        DirectPtxGemmOutputType outputType = DirectPtxGemmOutputType.Float32,
        bool halfAccumulate = false)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The 16-bit GEMM PTX specializations are measured only on GA10x/SM86.");
        Validate(m, n, k, batch, inputType, outputType, halfAccumulate);

        M = m;
        N = n;
        K = k;
        Batch = batch;
        TransposeA = transposeA;
        TransposeB = transposeB;
        InputType = inputType;
        OutputType = outputType;
        HalfAccumulate = halfAccumulate;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, m, n, k, batch,
            transposeA, transposeB, inputType, outputType, halfAccumulate);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            m, n, k, batch, transposeA, transposeB,
            inputType, outputType, halfAccumulate);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView left,
        DirectPtxTensorView right,
        DirectPtxTensorView output)
    {
        Require(left, Blueprint.Tensors[0], nameof(left));
        Require(right, Blueprint.Tensors[1], nameof(right));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, left) || Overlaps(output, right))
            throw new ArgumentException("16-bit GEMM output may not alias an input.");

        IntPtr leftPointer = left.Pointer;
        IntPtr rightPointer = right.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &leftPointer;
        arguments[1] = &rightPointer;
        arguments[2] = &outputPointer;
        uint grid = checked((uint)(((long)M * N + BlockThreads - 1) / BlockThreads));
        _module.Launch(
            _function, grid, 1, checked((uint)Batch),
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int m,
        int n,
        int k,
        int batch = 1,
        bool transposeA = false,
        bool transposeB = false,
        DirectPtx16BitInputType inputType = DirectPtx16BitInputType.Float16,
        DirectPtxGemmOutputType outputType = DirectPtxGemmOutputType.Float32,
        bool halfAccumulate = false)
    {
        Validate(m, n, k, batch, inputType, outputType, halfAccumulate);
        int matrixElements = checked(m * n);
        int aElements = checked(m * k);
        int bElements = checked(k * n);
        int outputElementBytes = outputType == DirectPtxGemmOutputType.Float32 ? 4 : 2;
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($"// exact {(inputType == DirectPtx16BitInputType.Float16 ? "fp16" : "bf16")} GEMM B={batch} M={m} N={n} K={k} ta={transposeA} tb={transposeB}");
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b16 %h<4>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [left_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [right_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {matrixElements};");
        ptx.AppendLine("    @%p0 bra.uni GEMM_DONE;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {n};");
        ptx.AppendLine($"    rem.u32 %r4, %r2, {n};");
        ptx.AppendLine("    mov.u32 %r5, %ctaid.z;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r5, {aElements * sizeof(ushort)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r5, {bElements * sizeof(ushort)};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r5, {matrixElements * outputElementBytes};");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("GEMM_K_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r6, {k};");
        ptx.AppendLine("    @%p1 bra.uni GEMM_STORE;");
        if (transposeA)
            ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {m}, %r3;");
        else
            ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {k}, %r6;");
        if (transposeB)
            ptx.AppendLine($"    mad.lo.u32 %r8, %r4, {k}, %r6;");
        else
            ptx.AppendLine($"    mad.lo.u32 %r8, %r6, {n}, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r7, 2;");
        ptx.AppendLine("    add.u64 %rd10, %rd4, %rd9;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r8, 2;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd11;");
        Emit16BitLoad(ptx, "%rd10", "%h0", "%r10", "%f0", inputType);
        Emit16BitLoad(ptx, "%rd12", "%h1", "%r11", "%f1", inputType);
        ptx.AppendLine("    fma.rn.f32 %f2, %f0, %f1, %f2;");
        if (halfAccumulate)
        {
            ptx.AppendLine("    cvt.rn.f16.f32 %h2, %f2;");
            ptx.AppendLine("    cvt.f32.f16 %f2, %h2;");
        }
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra.uni GEMM_K_LOOP;");
        ptx.AppendLine("GEMM_STORE:");
        ptx.AppendLine($"    mul.wide.u32 %rd13, %r2, {outputElementBytes};");
        ptx.AppendLine("    add.u64 %rd14, %rd8, %rd13;");
        if (outputType == DirectPtxGemmOutputType.Float32)
            ptx.AppendLine("    st.global.f32 [%rd14], %f2;");
        else
        {
            ptx.AppendLine("    cvt.rn.f16.f32 %h3, %f2;");
            ptx.AppendLine("    st.global.u16 [%rd14], %h3;");
        }
        ptx.AppendLine("GEMM_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void Emit16BitLoad(
        StringBuilder ptx,
        string address,
        string halfRegister,
        string integerRegister,
        string floatRegister,
        DirectPtx16BitInputType inputType)
    {
        if (inputType == DirectPtx16BitInputType.Float16)
        {
            ptx.AppendLine($"    ld.global.nc.u16 {halfRegister}, [{address}];");
            ptx.AppendLine($"    cvt.f32.f16 {floatRegister}, {halfRegister};");
        }
        else
        {
            ptx.AppendLine($"    ld.global.nc.u16 {integerRegister}, [{address}];");
            ptx.AppendLine($"    shl.b32 {integerRegister}, {integerRegister}, 16;");
            ptx.AppendLine($"    mov.b32 {floatRegister}, {integerRegister};");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int m,
        int n,
        int k,
        int batch,
        bool transposeA,
        bool transposeB,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType,
        bool halfAccumulate)
    {
        var left = batch == 1
            ? new DirectPtxExtent(transposeA ? k : m, transposeA ? m : k)
            : new DirectPtxExtent(batch, transposeA ? k : m, transposeA ? m : k);
        var right = batch == 1
            ? new DirectPtxExtent(transposeB ? n : k, transposeB ? k : n)
            : new DirectPtxExtent(batch, transposeB ? n : k, transposeB ? k : n);
        var output = batch == 1
            ? new DirectPtxExtent(m, n)
            : new DirectPtxExtent(batch, m, n);
        DirectPtxPhysicalType physicalInput = inputType == DirectPtx16BitInputType.Float16
            ? DirectPtxPhysicalType.Float16 : DirectPtxPhysicalType.BFloat16;
        DirectPtxPhysicalType physicalOutput = outputType == DirectPtxGemmOutputType.Float32
            ? DirectPtxPhysicalType.Float32 : DirectPtxPhysicalType.Float16;
        DirectPtxPhysicalLayout layout = batch == 1
            ? DirectPtxPhysicalLayout.RowMajor2D : DirectPtxPhysicalLayout.RowMajor3D;
        return new DirectPtxKernelBlueprint(
            Operation: "16-bit-gemm",
            Version: 1,
            Architecture: architecture,
            Variant: $"{inputType}-to-{outputType}-b{batch}-m{m}-n{n}-k{k}-ta{transposeA}-tb{transposeB}-ha{halfAccumulate}",
            Tensors:
            [
                new("left", physicalInput, layout, left, left, 16,
                    DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("right", physicalInput, layout, right, right, 16,
                    DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", physicalOutput, layout, output, output, 16,
                    DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output=op(left)@op(right)",
                ["input-dtype"] = inputType.ToString(),
                ["accumulator"] = halfAccumulate ? "fp16-rounded-each-fma" : "fp32",
                ["output-dtype"] = outputType.ToString(),
                ["transpose-a"] = transposeA.ToString(),
                ["transpose-b"] = transposeB.ToString(),
                ["batch"] = batch.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none",
                ["temporary-device-allocation"] = "none",
                ["promotion"] = "correctness-baseline-only; Tensor-Core replacement required"
            });
    }

    private static void Validate(
        int m,
        int n,
        int k,
        int batch,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType,
        bool halfAccumulate)
    {
        if (m <= 0 || m > 65_536) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0 || n > 65_536) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0 || k > 65_536) throw new ArgumentOutOfRangeException(nameof(k));
        if (batch <= 0 || batch > 65_535) throw new ArgumentOutOfRangeException(nameof(batch));
        if (!Enum.IsDefined(typeof(DirectPtx16BitInputType), inputType))
            throw new ArgumentOutOfRangeException(nameof(inputType));
        if (!Enum.IsDefined(typeof(DirectPtxGemmOutputType), outputType))
            throw new ArgumentOutOfRangeException(nameof(outputType));
        if (halfAccumulate && (inputType != DirectPtx16BitInputType.Float16 ||
            outputType != DirectPtxGemmOutputType.Float16))
            throw new ArgumentException("FP16 accumulation requires FP16 inputs and output.");
        _ = checked(batch * checked(m * k));
        _ = checked(batch * checked(k * n));
        _ = checked(batch * checked(m * n));
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
}
