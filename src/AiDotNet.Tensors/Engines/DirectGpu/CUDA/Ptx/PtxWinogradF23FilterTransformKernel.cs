using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Precomputes the Winograd F(2,3) filter transform U = G g G^T for every
/// (output-channel k, input-channel c): reads OIHW weights[K,C,3,3] and writes
/// U[K,C,4,4]. One thread per (k,c). Run once (filters are constant); the main
/// Winograd kernel then reads U instead of recomputing G g G^T per output tile —
/// removing the dominant redundant work of the correctness-first layout.
/// </summary>
internal sealed class PtxWinogradF23FilterTransformKernel : IDisposable
{
    internal const int BlockThreads = 32;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OutputChannels { get; }
    internal int InputChannels { get; }
    internal bool PositionMajor { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal string EntryPoint =>
        FormattableString.Invariant($"aidotnet_winograd_f23_filter_transform_k{OutputChannels}_c{InputChannels}{(PositionMajor ? "_pm" : "")}");
    internal long WeightBytes => (long)OutputChannels * InputChannels * 9 * sizeof(float);
    internal long TransformedBytes => (long)OutputChannels * InputChannels * 16 * sizeof(float);

    internal PtxWinogradF23FilterTransformKernel(DirectPtxRuntime runtime, int outputChannels, int inputChannels, bool positionMajor = false)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Winograd filter transform has no experimental non-SM86 specialization.");
        if (outputChannels <= 0 || inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if ((long)outputChannels * inputChannels % BlockThreads != 0)
            throw new ArgumentException($"K*C must be a multiple of {BlockThreads}.");
        OutputChannels = outputChannels;
        InputChannels = inputChannels;
        PositionMajor = positionMajor;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outputChannels, inputChannels);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, outputChannels, inputChannels, positionMajor);
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
        DirectPtxArchitectureFamily architecture, int k, int c)
    {
        var g = new DirectPtxExtent(k, c, 3, 3);
        var u = new DirectPtxExtent(k, c, 4, 4);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-f23-filter-transform",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"k{k}-c{c}-r3-winograd-f23-fp32"),
            Tensors:
            [
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    g, g, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("transformed", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    u, u, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "U = G g G^T",
                ["input"] = "fp32", ["output"] = "fp32", ["layout"] = "oihw",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView weights, DirectPtxTensorView transformed)
    {
        Require(weights, Blueprint.Tensors[0], nameof(weights));
        Require(transformed, Blueprint.Tensors[1], nameof(transformed));
        IntPtr wPtr = weights.Pointer, uPtr = transformed.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &wPtr; arguments[1] = &uPtr;
        int total = OutputChannels * InputChannels;
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

    internal static string EmitPtx(
        int major, int minor, int k, int c, bool positionMajor = false,
        bool reductionMajor = false, bool invertFilter = false)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 filter transform emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        string entry = "aidotnet_winograd_f23_filter_transform_k" + I(k) +
            "_c" + I(c) + (positionMajor ? "_pm" : "") +
            (reductionMajor ? "_reduction_major" : "") +
            (invertFilter ? "_flip" : "");
        int total = k * c;

        var p = new StringBuilder(8192);
        p.AppendLine(".version 7.1");
        p.AppendLine($".target sm_{major}{minor}");
        p.AppendLine(".address_size 64");
        p.AppendLine();
        p.AppendLine($".visible .entry {entry}(");
        p.AppendLine("    .param .u64 weights_ptr,");
        p.AppendLine("    .param .u64 transformed_ptr");
        p.AppendLine(")");
        p.AppendLine("{");
        p.AppendLine("    .reg .pred %p<2>;");
        p.AppendLine("    .reg .b32 %r<8>;");
        p.AppendLine("    .reg .b64 %rd<8>;");
        p.AppendLine("    .reg .f32 %f<40>;");
        p.AppendLine("    ld.param.u64 %rd0, [weights_ptr];");
        p.AppendLine("    ld.param.u64 %rd1, [transformed_ptr];");
        p.AppendLine("    mov.u32 %r0, %tid.x;");
        p.AppendLine("    mov.u32 %r1, %ctaid.x;");
        p.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // id = k*C + c
        p.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        p.AppendLine("    @%p0 bra DONE;");
        if (reductionMajor)
        {
            if ((c & (c - 1)) == 0)
            {
                int shift = 0;
                while ((1 << shift) != c) shift++;
                p.AppendLine($"    shr.u32 %r3, %r2, {I(shift)};");          // logical k
                p.AppendLine($"    and.b32 %r4, %r2, {I(c - 1)};");          // logical c
            }
            else
            {
                p.AppendLine($"    div.u32 %r3, %r2, {I(c)};");             // logical k
                p.AppendLine($"    rem.u32 %r4, %r2, {I(c)};");             // logical c
            }
            p.AppendLine($"    mad.lo.u32 %r5, %r4, {I(k)}, %r3;");
            p.AppendLine("    mul.wide.u32 %rd2, %r5, 36;");
        }
        else
        {
            p.AppendLine("    mul.wide.u32 %rd2, %r2, 36;");                 // g base = weights + id*9*4
        }
        p.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        // U base. kc-major: U + id*16*4 (16 contiguous per (k,c)).
        // position-major: U + id*4 (the (k,c) slot in xi=0; xi stride = K*C*4).
        p.AppendLine($"    mul.wide.u32 %rd3, %r2, {I(positionMajor ? 4 : 64)};");
        p.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // load g[0..8] -> %f0..%f8. Consecutive filters are nine words
        // apart, so their bases are not generally vector-aligned.
        for (int gi = 0; gi < 3; gi++)
            for (int gj = 0; gj < 3; gj++)
            {
                int register = gi * 3 + gj;
                int sourceGi = invertFilter ? 2 - gi : gi;
                int sourceGj = invertFilter ? 2 - gj : gj;
                int source = sourceGi * 3 + sourceGj;
                p.AppendLine($"    ld.global.nc.f32 %f{I(register)}, " +
                    $"[%rd2+{I(source * 4)}];");
            }
        // u = G g (4x3) -> %f9..%f20 ; g[gi][gj] = %f(gi*3+gj)
        int G(int i, int j) => i * 3 + j;          // g at %f0..8
        int U3(int i, int j) => 9 + i * 3 + j;     // u at %f9..20
        for (int j = 0; j < 3; j++)
        {
            p.AppendLine($"    mov.f32 %f{I(U3(0, j))}, %f{I(G(0, j))};");
            p.AppendLine($"    add.rn.f32 %f37, %f{I(G(0, j))}, %f{I(G(1, j))};");
            p.AppendLine($"    add.rn.f32 %f37, %f37, %f{I(G(2, j))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U3(1, j))}, %f37, 0f3F000000;");   // /2
            p.AppendLine($"    sub.rn.f32 %f38, %f{I(G(0, j))}, %f{I(G(1, j))};");
            p.AppendLine($"    add.rn.f32 %f38, %f38, %f{I(G(2, j))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U3(2, j))}, %f38, 0f3F000000;");
            p.AppendLine($"    mov.f32 %f{I(U3(3, j))}, %f{I(G(2, j))};");
        }
        // U = u G^T (4x4) -> %f21..%f36 ; store
        int U(int i, int j) => 21 + i * 4 + j;
        for (int i = 0; i < 4; i++)
        {
            p.AppendLine($"    mov.f32 %f{I(U(i, 0))}, %f{I(U3(i, 0))};");
            p.AppendLine($"    add.rn.f32 %f37, %f{I(U3(i, 0))}, %f{I(U3(i, 1))};");
            p.AppendLine($"    add.rn.f32 %f37, %f37, %f{I(U3(i, 2))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U(i, 1))}, %f37, 0f3F000000;");
            p.AppendLine($"    sub.rn.f32 %f38, %f{I(U3(i, 0))}, %f{I(U3(i, 1))};");
            p.AppendLine($"    add.rn.f32 %f38, %f38, %f{I(U3(i, 2))};");
            p.AppendLine($"    mul.rn.f32 %f{I(U(i, 2))}, %f38, 0f3F000000;");
            p.AppendLine($"    mov.f32 %f{I(U(i, 3))}, %f{I(U3(i, 2))};");
        }
        int uStride = positionMajor ? total * 4 : 4;   // xi stride: K*C*4 (pm) or 4 (kc)
        if (uStride == sizeof(float))
        {
            for (int i = 0; i < 16; i += 4)
                p.AppendLine($"    st.global.v4.f32 [%rd3+{I(i * sizeof(float))}], " +
                    $"{{%f{I(21 + i)}, %f{I(22 + i)}, %f{I(23 + i)}, %f{I(24 + i)}}};");
        }
        else
        {
            for (int i = 0; i < 16; i++)
                p.AppendLine($"    st.global.f32 [%rd3+{I(i * uStride)}], %f{I(21 + i)};");
        }
        p.AppendLine("DONE:");
        p.AppendLine("    ret;");
        p.AppendLine("}");
        return p.ToString();
    }

    public void Dispose() => _module.Dispose();
}
