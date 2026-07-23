using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Winograd F(2,3) output transform: reads M[16,K,P] (the batched-GEMM result),
/// applies Y = A^T M A per (k, tile), adds bias, ReLUs, and scatters the 2x2
/// output tile to output[N,K,H,W]. One thread per (k, tile).
/// </summary>
internal sealed class PtxWinogradF23OutputTransformKernel : IDisposable
{
    internal const int BlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int OutputChannels { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int TileRows => Height / 2;
    internal int TileCols => Width / 2;
    internal int Tiles => Batch * TileRows * TileCols;
    internal long MBytes => (long)16 * OutputChannels * Tiles * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_winograd_f23_output_transform_n{Batch}_h{Height}_w{Width}_k{OutputChannels}");

    internal PtxWinogradF23OutputTransformKernel(
        DirectPtxRuntime runtime, int batch, int height, int width, int outputChannels)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Winograd output transform has no experimental non-SM86 specialization.");
        if ((height & 1) != 0 || (width & 1) != 0) throw new ArgumentException("Even H,W required.");
        Batch = batch; Height = height; Width = width; OutputChannels = outputChannels;
        if ((long)outputChannels * Tiles % BlockThreads != 0)
            throw new ArgumentException($"K*P must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var m = new DirectPtxExtent(16, OutputChannels, Tiles);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-f23-output-transform",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-h{Height}-w{Width}-k{OutputChannels}-fp32"),
            Tensors:
            [
                new("M", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    m, m, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "relu(A^T M A + bias)",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView m, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(m, Blueprint.Tensors[0], nameof(m));
        Require(bias, Blueprint.Tensors[1], nameof(bias));
        Require(output, Blueprint.Tensors[2], nameof(output));
        IntPtr mPtr = m.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &mPtr; arguments[1] = &bPtr; arguments[2] = &oPtr;
        int total = OutputChannels * Tiles;
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

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 output transform emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int k = OutputChannels, h = Height, w = Width, th = TileRows, tw = TileCols, pp = Tiles;
        int kp = k * pp, hw = h * w;
        string entry = EntryPoint;

        var s = new StringBuilder(16384);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 m_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<2>;");
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<32>;");
        s.AppendLine("    ld.param.u64 %rd0, [m_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // id = k*P + pp
        s.AppendLine($"    rem.u32 %r3, %r2, {I(pp)};");                     // p (tile)
        s.AppendLine($"    div.u32 %r4, %r2, {I(pp)};");                     // k
        // M base: M + (k*P + p)*4 ; xi stride = K*P*4
        s.AppendLine($"    mad.lo.u32 %r5, %r4, {I(pp)}, %r3;");            // k*P + p
        s.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // load M[0..15] -> %f0..15
        for (int xi = 0; xi < 16; xi++)
            s.AppendLine($"    ld.global.nc.f32 %f{I(xi)}, [%rd3+{I(xi * kp * 4)}];");
        // Y = A^T M A -> %f16..19 ; s in %f24..31
        int M(int i, int j) => i * 4 + j;
        int S(int i, int j) => 24 + i * 4 + j;
        int Y(int i, int j) => 16 + i * 2 + j;
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(M(0, j))}, %f{I(M(1, j))};");
            s.AppendLine($"    add.rn.f32 %f{I(S(0, j))}, %f{I(S(0, j))}, %f{I(M(2, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(M(1, j))}, %f{I(M(2, j))};");
            s.AppendLine($"    sub.rn.f32 %f{I(S(1, j))}, %f{I(S(1, j))}, %f{I(M(3, j))};");
        }
        for (int i = 0; i < 2; i++)
        {
            s.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(S(i, 0))}, %f{I(S(i, 1))};");
            s.AppendLine($"    add.rn.f32 %f{I(Y(i, 0))}, %f{I(Y(i, 0))}, %f{I(S(i, 2))};");
            s.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(S(i, 1))}, %f{I(S(i, 2))};");
            s.AppendLine($"    sub.rn.f32 %f{I(Y(i, 1))}, %f{I(Y(i, 1))}, %f{I(S(i, 3))};");
        }
        // decode p -> n, ti, tj
        s.AppendLine($"    rem.u32 %r6, %r3, {I(tw)};");   // tj
        s.AppendLine($"    div.u32 %r7, %r3, {I(tw)};");
        s.AppendLine($"    rem.u32 %r8, %r7, {I(th)};");   // ti
        s.AppendLine($"    div.u32 %r9, %r7, {I(th)};");   // n
        s.AppendLine("    mul.lo.u32 %r10, %r8, 2;");      // oh
        s.AppendLine("    mul.lo.u32 %r11, %r6, 2;");      // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r4, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f20, [%rd4];");  // bias[k]
        s.AppendLine($"    mad.lo.u32 %r12, %r9, {I(k)}, %r4;");   // n*K + k
        s.AppendLine($"    mul.wide.u32 %rd5, %r12, {I(hw)};");
        s.AppendLine("    shl.b64 %rd5, %rd5, 2;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        for (int oi = 0; oi < 2; oi++)
            for (int oj = 0; oj < 2; oj++)
            {
                int yreg = 16 + oi * 2 + oj;
                s.AppendLine($"    add.rn.f32 %f{I(yreg)}, %f{I(yreg)}, %f20;");
                s.AppendLine($"    max.f32 %f{I(yreg)}, %f{I(yreg)}, 0f00000000;");
                s.AppendLine($"    add.u32 %r13, %r10, {I(oi)};");
                s.AppendLine($"    mul.lo.u32 %r13, %r13, {I(w)};");
                s.AppendLine("    add.u32 %r13, %r13, %r11;");
                s.AppendLine($"    add.u32 %r13, %r13, {I(oj)};");
                s.AppendLine("    mul.wide.u32 %rd6, %r13, 4;");
                s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
                s.AppendLine($"    st.global.f32 [%rd6], %f{I(yreg)};");
            }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
