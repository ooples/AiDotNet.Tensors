using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP16 Winograd F(2,3) input transform for the Tensor-Core GEMM path: for every
/// (input-channel c, output-tile p) gathers the 4x4 same-padded input patch d,
/// computes V = B^T d B in fp32, and writes the 16 positions as fp16 into a
/// tile-major buffer V[16, P, C] (V[xi*P*C + p*C + c]), where P = N*TH*TW is the
/// number of 2x2 output tiles. This is exactly the B operand layout the WMMA
/// batched GEMM consumes: per position xi, V[xi] is a row-major [P, C] matrix so
/// the col-major WMMA load yields V[xi]^T and the mma computes U[xi]*V[xi].
/// One thread per (c, p).
/// </summary>
internal sealed class PtxWinogradF23InputTransformFp16Kernel : IDisposable
{
    internal const int BlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;       // P
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long TransformedBytes => (long)16 * Tiles * InputChannels * sizeof(ushort);   // V[16,P,C] fp16

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_f23_input_transform_fp16_n{Batch}_c{InputChannels}_h{Height}_w{Width}");

    internal PtxWinogradF23InputTransformFp16Kernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int height, int width)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Winograd fp16 input transform has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if ((height & 1) != 0 || (width & 1) != 0)
            throw new ArgumentException("Even H and W (whole 2x2 tiling) required.");
        Batch = batch; InputChannels = inputChannels; Height = height; Width = width;
        if ((long)inputChannels * Tiles % BlockThreads != 0)
            throw new ArgumentException($"C*P must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, inputChannels, height, width);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, inputChannels, height, width);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int n, int c, int h, int w)
    {
        int tiles = n * (h / 2) * (w / 2);
        var input = new DirectPtxExtent(n, c, h, w);
        var v = new DirectPtxExtent(16, tiles, c);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-f23-input-transform-fp16",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{n}-c{c}-h{h}-w{w}-winograd-f23-fp16"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("transformed", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    v, v, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "V = B^T d B",
                ["layout"] = "nchw -> tile-major V[16,P,C]",
                ["padding"] = "same-1",
                ["output"] = "fp16",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView transformed)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(transformed, Blueprint.Tensors[1], nameof(transformed));
        IntPtr iPtr = input.Pointer, vPtr = transformed.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &iPtr; arguments[1] = &vPtr;
        int total = InputChannels * Tiles;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, int n, int c, int h, int w)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 fp16 input transform emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int th = h / 2, tw = w / 2;
        int p = n * th * tw;        // P
        int hw = h * w;
        int xiStride = p * c * sizeof(ushort);   // fp16 V[16,P,C] xi stride: P*C*2
        string entry = FormattableString.Invariant(
            $"aidotnet_winograd_f23_input_transform_fp16_n{n}_c{c}_h{h}_w{w}");

        var s = new StringBuilder(16384);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 transformed_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b16 %h<4>;");
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<20>;");
        s.AppendLine("    .reg .f32 %f<48>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [transformed_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // id = c*P + pp
        s.AppendLine($"    rem.u32 %r3, %r2, {I(p)};");                      // pp (tile index)
        s.AppendLine($"    div.u32 %r4, %r2, {I(p)};");                      // c
        // decode pp -> n, ti, tj
        s.AppendLine($"    rem.u32 %r5, %r3, {I(tw)};");                     // tj
        s.AppendLine($"    div.u32 %r6, %r3, {I(tw)};");
        s.AppendLine($"    rem.u32 %r7, %r6, {I(th)};");                     // ti
        s.AppendLine($"    div.u32 %r8, %r6, {I(th)};");                     // n
        s.AppendLine("    mul.lo.s32 %r9, %r7, 2;");
        s.AppendLine("    sub.s32 %r9, %r9, 1;");                            // ih0 = 2*ti-1
        s.AppendLine("    mul.lo.s32 %r10, %r5, 2;");
        s.AppendLine("    sub.s32 %r10, %r10, 1;");                          // iw0 = 2*tj-1
        // input channel base: input + (n*C + c)*H*W*4
        s.AppendLine($"    mad.lo.u32 %r11, %r8, {I(c)}, %r4;");             // n*C + c
        s.AppendLine($"    mul.wide.u32 %rd2, %r11, {I(hw)};");
        s.AppendLine("    shl.b64 %rd2, %rd2, 2;");
        s.AppendLine("    add.u64 %rd2, %rd0, %rd2;");                       // input channel base
        // load 4x4 patch d -> %f0..%f15 (zero-padded)
        for (int di = 0; di < 4; di++)
            for (int dj = 0; dj < 4; dj++)
            {
                int reg = di * 4 + dj;
                s.AppendLine($"    mov.f32 %f{I(reg)}, 0f00000000;");
                s.AppendLine($"    add.s32 %r12, %r9, {I(di)};");
                s.AppendLine($"    add.s32 %r13, %r10, {I(dj)};");
                s.AppendLine("    setp.ge.s32 %p1, %r12, 0;");
                s.AppendLine($"    setp.lt.s32 %p2, %r12, {I(h)};");
                s.AppendLine("    setp.ge.s32 %p3, %r13, 0;");
                s.AppendLine($"    setp.lt.s32 %p4, %r13, {I(w)};");
                s.AppendLine("    and.pred %p1, %p1, %p2;");
                s.AppendLine("    and.pred %p3, %p3, %p4;");
                s.AppendLine("    and.pred %p1, %p1, %p3;");
                s.AppendLine($"    mul.lo.u32 %r14, %r12, {I(w)};");
                s.AppendLine("    add.u32 %r14, %r14, %r13;");
                s.AppendLine("    mul.wide.s32 %rd3, %r14, 4;");
                s.AppendLine("    add.u64 %rd3, %rd2, %rd3;");
                s.AppendLine($"    @%p1 ld.global.nc.f32 %f{I(reg)}, [%rd3];");
            }
        // V = B^T d B : t = B^T d (%f16..31), V = t B (%f32..47)
        int D(int i, int j) => i * 4 + j;
        int T(int i, int j) => 16 + i * 4 + j;
        int V(int i, int j) => 32 + i * 4 + j;
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    sub.rn.f32 %f{I(T(0, j))}, %f{I(D(0, j))}, %f{I(D(2, j))};");
            s.AppendLine($"    add.rn.f32 %f{I(T(1, j))}, %f{I(D(1, j))}, %f{I(D(2, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(T(2, j))}, %f{I(D(2, j))}, %f{I(D(1, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(T(3, j))}, %f{I(D(1, j))}, %f{I(D(3, j))};");
        }
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine($"    sub.rn.f32 %f{I(V(i, 0))}, %f{I(T(i, 0))}, %f{I(T(i, 2))};");
            s.AppendLine($"    add.rn.f32 %f{I(V(i, 1))}, %f{I(T(i, 1))}, %f{I(T(i, 2))};");
            s.AppendLine($"    sub.rn.f32 %f{I(V(i, 2))}, %f{I(T(i, 2))}, %f{I(T(i, 1))};");
            s.AppendLine($"    sub.rn.f32 %f{I(V(i, 3))}, %f{I(T(i, 1))}, %f{I(T(i, 3))};");
        }
        // store V[xi*P*C + pp*C + c] for xi=0..15 as fp16.
        // base = V + (pp*C + c)*2 ; xi stride = P*C*2.
        s.AppendLine($"    mad.lo.u32 %r15, %r3, {I(c)}, %r4;");             // pp*C + c
        s.AppendLine("    mul.wide.u32 %rd4, %r15, 2;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");                       // &V[0][pp][c]
        for (int xi = 0; xi < 16; xi++)
        {
            s.AppendLine($"    cvt.rn.f16.f32 %h0, %f{I(32 + xi)};");
            s.AppendLine($"    st.global.b16 [%rd4+{I(xi * xiStride)}], %h0;");
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
