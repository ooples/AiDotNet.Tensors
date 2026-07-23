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
    internal const int BaselineBlockThreads = 256;
    internal const int TensorCoreBlockThreads = 64;
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
    internal int BlockThreads { get; }
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
        BlockThreads = IsTensorCoreShape(
            m, n, k, batch, transposeA, transposeB,
            inputType, outputType, halfAccumulate)
            ? TensorCoreBlockThreads
            : BaselineBlockThreads;
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
        uint grid = IsTensorCoreShape(
                M, N, K, Batch, TransposeA, TransposeB,
                InputType, OutputType, HalfAccumulate)
            ? 1u
            : checked((uint)(((long)M * N + BlockThreads - 1) / BlockThreads));
        _module.Launch(
            _function, grid, 1, checked((uint)Batch),
            checked((uint)BlockThreads), 1, 1, 0, arguments);
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
        if (IsTensorCoreShape(
                m, n, k, batch, transposeA, transposeB,
                inputType, outputType, halfAccumulate))
            return EmitTensorCoreM16N16K32(ccMajor, ccMinor);
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
        ptx.AppendLine($".maxntid {BaselineBlockThreads}, 1, 1");
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
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BaselineBlockThreads}, %r1;");
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

    private static string EmitTensorCoreM16N16K32(int ccMajor, int ccMinor)
    {
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine("// exact FP16 M16 N16 K32: two-warp async Tensor-Core specialization");
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {TensorCoreBlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b32 %a<4>;");
        ptx.AppendLine("    .reg .b32 %b<2>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %c<4>;");
        ptx.AppendLine("    .shared .align 16 .b8 smem[2048];");
        ptx.AppendLine("    ld.param.u64 %rd0, [left_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [right_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    shl.b32 %r3, %r0, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd3, %r3;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        // A is repacked from row-major [16,32] into two adjacent M16xK16
        // panels so one x4 ldmatrix supplies each MMA K fragment.
        ptx.AppendLine("    shr.u32 %r4, %r0, 2;");
        ptx.AppendLine("    and.b32 %r5, %r0, 3;");
        ptx.AppendLine("    shr.u32 %r6, %r5, 1;");
        ptx.AppendLine("    and.b32 %r7, %r5, 1;");
        ptx.AppendLine("    shl.b32 %r6, %r6, 9;");
        ptx.AppendLine("    shl.b32 %r7, %r7, 4;");
        ptx.AppendLine("    mad.lo.u32 %r8, %r4, 32, %r6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r7;");
        ptx.AppendLine("    mov.u64 %rd6, smem;");
        ptx.AppendLine("    cvt.u64.u32 %rd7, %r8;");
        ptx.AppendLine("    add.u64 %rd8, %rd6, %rd7;");
        ptx.AppendLine("    cp.async.ca.shared.global [%rd8], [%rd4], 16;");
        // B remains row-major [32,16]. The transposed x2 ldmatrix below
        // presents its KxN rows as the column-major MMA B fragment.
        ptx.AppendLine("    add.u64 %rd9, %rd6, %rd3;");
        ptx.AppendLine("    cp.async.ca.shared.global [%rd9+1024], [%rd5], 16;");
        ptx.AppendLine("    cp.async.commit_group;");
        ptx.AppendLine("    cp.async.wait_group 0;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mov.f32 %c0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c3, 0f00000000;");
        // Warp-collective A addresses for an M16xK16 row-major fragment.
        ptx.AppendLine("    and.b32 %r9, %r1, 7;");
        ptx.AppendLine("    shr.u32 %r10, %r1, 3;");
        ptx.AppendLine("    and.b32 %r11, %r10, 1;");
        ptx.AppendLine("    shl.b32 %r11, %r11, 8;");
        ptx.AppendLine("    shr.u32 %r10, %r10, 1;");
        ptx.AppendLine("    shl.b32 %r10, %r10, 4;");
        ptx.AppendLine("    mad.lo.u32 %r9, %r9, 32, %r11;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r10;");
        // Warp 0 consumes N0..7; warp 1 consumes N8..15. The second x2
        // matrix starts eight K rows later (8 * 32 bytes).
        ptx.AppendLine("    and.b32 %r12, %r1, 7;");
        ptx.AppendLine("    shr.u32 %r13, %r1, 3;");
        ptx.AppendLine("    and.b32 %r13, %r13, 1;");
        ptx.AppendLine("    shl.b32 %r13, %r13, 8;");
        ptx.AppendLine("    mad.lo.u32 %r12, %r12, 32, %r13;");
        ptx.AppendLine("    shl.b32 %r14, %r2, 4;");
        ptx.AppendLine("    add.u32 %r12, %r12, %r14;");
        ptx.AppendLine("    cvt.u64.u32 %rd10, %r9;");
        ptx.AppendLine("    add.u64 %rd11, %rd6, %rd10;");
        ptx.AppendLine("    cvt.u64.u32 %rd12, %r12;");
        ptx.AppendLine("    add.u64 %rd13, %rd6, %rd12;");
        for (int panel = 0; panel < 2; panel++)
        {
            int offset = panel * 512;
            ptx.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4.shared.b16 " +
                $"{{%a0,%a1,%a2,%a3}}, [%rd11+{offset}];");
            ptx.AppendLine($"    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 " +
                $"{{%b0,%b1}}, [%rd13+{1024 + offset}];");
            ptx.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
                "{%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};");
        }
        ptx.AppendLine("    shr.u32 %r15, %r1, 2;");
        ptx.AppendLine("    and.b32 %r16, %r1, 3;");
        ptx.AppendLine("    shl.b32 %r16, %r16, 1;");
        ptx.AppendLine("    mad.lo.u32 %r16, %r2, 8, %r16;");
        ptx.AppendLine("    mad.lo.u32 %r17, %r15, 16, %r16;");
        ptx.AppendLine("    shl.b32 %r17, %r17, 2;");
        ptx.AppendLine("    cvt.u64.u32 %rd10, %r17;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd10;");
        ptx.AppendLine("    st.global.v2.f32 [%rd11], {%c0,%c1};");
        ptx.AppendLine("    st.global.v2.f32 [%rd11+512], {%c2,%c3};");
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
        bool tensorCore = IsTensorCoreShape(
            m, n, k, batch, transposeA, transposeB,
            inputType, outputType, halfAccumulate);
        return new DirectPtxKernelBlueprint(
            Operation: "16-bit-gemm",
            Version: tensorCore ? 2 : 1,
            Architecture: architecture,
            Variant: $"{(tensorCore ? "tensorcore-async-" : string.Empty)}{inputType}-to-{outputType}-b{batch}-m{m}-n{n}-k{k}-ta{transposeA}-tb{transposeB}-ha{halfAccumulate}",
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
                MaxStaticSharedBytes: tensorCore ? 2_048 : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: tensorCore ? 8 : 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output=op(left)@op(right)",
                ["input-dtype"] = inputType.ToString(),
                ["accumulator"] = tensorCore
                    ? "mma-m16n8k16-fp32-register-fragment"
                    : halfAccumulate ? "fp16-rounded-each-fma" : "fp32",
                ["output-dtype"] = outputType.ToString(),
                ["transpose-a"] = transposeA.ToString(),
                ["transpose-b"] = transposeB.ToString(),
                ["batch"] = batch.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none",
                ["temporary-device-allocation"] = "none",
                ["pipeline"] = tensorCore
                    ? "two-warp-cp.async-ldmatrix-mma; register-only-output"
                    : "scalar-correctness-baseline",
                ["promotion"] = tensorCore
                    ? "exact M16 N16 K32 Tensor-Core experiment"
                    : "correctness-baseline-only; Tensor-Core replacement required"
            });
    }

    private static bool IsTensorCoreShape(
        int m,
        int n,
        int k,
        int batch,
        bool transposeA,
        bool transposeB,
        DirectPtx16BitInputType inputType,
        DirectPtxGemmOutputType outputType,
        bool halfAccumulate) =>
        m == 16 && n == 16 && k == 32 && batch == 1 &&
        !transposeA && !transposeB &&
        inputType == DirectPtx16BitInputType.Float16 &&
        outputType == DirectPtxGemmOutputType.Float32 && !halfAccumulate;

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
