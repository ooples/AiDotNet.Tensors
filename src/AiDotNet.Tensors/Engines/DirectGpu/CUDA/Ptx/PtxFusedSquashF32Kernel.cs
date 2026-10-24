using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Which half of the capsule squash a <see cref="PtxFusedSquashF32Kernel"/> computes.</summary>
internal enum SquashDirection
{
    /// <summary><c>y = f(r) x</c>.</summary>
    Forward,

    /// <summary>The vector-Jacobian product <c>dx = f(r) g + (f'(r)/r) x dot(g, x)</c>.</summary>
    Backward
}

/// <summary>
/// Exact contiguous FP32 capsule squash, forward and backward (#833: the census's <c>squash</c> /
/// <c>squash_backward</c>, the two CUDA kernels no epic child had ported).
/// </summary>
/// <remarks>
/// <para>
/// For a capsule <c>x</c> with <c>r = |x|</c>, <c>f(r) = r^2 / ((1 + r^2)(r + eps))</c> and <c>y = f(r) x</c>. The
/// backward is <c>dx = f(r) g + (f'(r)/r) x dot(g, x)</c> with
/// <c>f'(r)/r = (r + 2 eps - r^3) / ((1 + r^2)^2 (r + eps)^2)</c>.
/// </para>
/// <para>
/// One thread owns one capsule, as in the CUDA kernels it replaces: the capsule is read once with aligned 16-byte
/// vector loads, every sum is accumulated in FP64 in the same order and with the same contractions nvcc gives the CUDA
/// source (<c>fma.rn.f64</c> for each <c>sum += a * b</c>, IEEE <c>sqrt.rn.f64</c> and <c>div.rn.f64</c>), and each
/// element is rounded to FP32 once, at the store. No shared memory, no global intermediates, no temporary allocation,
/// no scalar shape parameters: the capsule count, capsule width and epsilon are baked per specialization.
/// </para>
/// <para>
/// Disabled by default and fail-closed: it dispatches only when <see cref="DirectPtxFeatureGate.IsCapsuleEnabled"/>
/// and <see cref="IsPromotedShape"/> (or the experiment override) allow it, which stays false until the shapes clear
/// correctness and the &gt;= 1.10x promotion gate on an NVIDIA GPU (see <see cref="DirectPtxFeatureGate.CapsuleSquashPromoted"/>).
/// </para>
/// </remarks>
internal sealed class PtxFusedSquashF32Kernel : IDisposable
{
    internal const string ForwardEntryPoint = "aidotnet_fused_squash_f32";
    internal const string BackwardEntryPoint = "aidotnet_fused_squash_backward_f32";
    internal const int DefaultBlockThreads = 128;

    /// <summary>The only epsilon baked: the engine's default (<c>SquashGpu</c>, <c>TensorSquash</c>).</summary>
    internal const float Epsilon = 1e-8f;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal SquashDirection Direction { get; }
    internal int NumCapsules { get; }
    internal int CapsuleDim { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedSquashF32Kernel(
        DirectPtxRuntime runtime,
        SquashDirection direction,
        int numCapsules,
        int capsuleDim,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedCapsuleSquash(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 capsule-squash specialization targets GA10x/SM86 only.");
        Validate(numCapsules, capsuleDim);
        ValidateBlockThreads(numCapsules, blockThreads);
        Direction = direction;
        NumCapsules = numCapsules;
        CapsuleDim = capsuleDim;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, direction, numCapsules, capsuleDim, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, direction, numCapsules, capsuleDim,
            blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointOf(direction), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointOf(direction), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    /// <summary>Forward: <paramref name="input"/> to <paramref name="output"/>.</summary>
    internal unsafe void LaunchForward(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        if (Direction != SquashDirection.Forward)
            throw new InvalidOperationException("This specialization computes the squash backward.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(input, output))
            throw new ArgumentException("The squash specialization does not admit aliasing.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        Launch(arguments);
    }

    /// <summary>Backward: <paramref name="gradOutput"/> and <paramref name="input"/> to <paramref name="gradInput"/>.</summary>
    internal unsafe void LaunchBackward(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradInput)
    {
        if (Direction != SquashDirection.Backward)
            throw new InvalidOperationException("This specialization computes the squash forward.");
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        if (Overlaps(gradOutput, gradInput) || Overlaps(input, gradInput))
            throw new ArgumentException("The squash-backward specialization does not admit aliasing its output.");

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &gradInputPointer;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments) =>
        _module.Launch(
            _function,
            checked((uint)(NumCapsules / BlockThreads)),
            1, 1,
            checked((uint)BlockThreads),
            1, 1,
            0,
            arguments);

    public void Dispose() => _module.Dispose();

    internal static string EntryPointOf(SquashDirection direction) =>
        direction == SquashDirection.Forward ? ForwardEntryPoint : BackwardEntryPoint;

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        SquashDirection direction,
        int numCapsules,
        int capsuleDim,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(numCapsules, capsuleDim);
        ValidateBlockThreads(numCapsules, blockThreads);
        bool backward = direction == SquashDirection.Backward;
        int vectors = capsuleDim / 4;
        int rowBytes = capsuleDim * sizeof(float);
        // eps is the FP32 value widened to FP64, as the CUDA source's (double)epsilon.
        string eps = F64((double)Epsilon);

        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape capsules={numCapsules} dim={capsuleDim} block={blockThreads} " +
            $"strategy=thread-capsule op={(backward ? "squash-backward" : "squash")} accumulate=f64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPointOf(direction)}(");
        if (backward)
        {
            ptx.AppendLine("    .param .u64 grad_output_ptr,");
            ptx.AppendLine("    .param .u64 input_ptr,");
            ptx.AppendLine("    .param .u64 grad_input_ptr");
        }
        else
        {
            ptx.AppendLine("    .param .u64 input_ptr,");
            ptx.AppendLine("    .param .u64 output_ptr");
        }
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<1>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        // %f0..%f(dim-1): x; %f(dim)..: the gradient (backward). One FP32 register per element, widened on use.
        ptx.AppendLine($"    .reg .f32 %f<{(backward ? 2 : 1) * capsuleDim}>;");
        ptx.AppendLine("    .reg .f64 %fd<16>;");

        // Capsule index: the grid covers the capsules exactly (numCapsules % blockThreads == 0), so no bound check.
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine($"    mul.wide.u32 %rd0, %r2, {rowBytes};");

        if (backward)
        {
            ptx.AppendLine("    ld.param.u64 %rd1, [grad_output_ptr];");
            ptx.AppendLine("    ld.param.u64 %rd2, [input_ptr];");
            ptx.AppendLine("    ld.param.u64 %rd3, [grad_input_ptr];");
            ptx.AppendLine("    add.u64 %rd4, %rd1, %rd0;");   // g
            ptx.AppendLine("    add.u64 %rd5, %rd2, %rd0;");   // x
            ptx.AppendLine("    add.u64 %rd6, %rd3, %rd0;");   // dx
            EmitLoads(ptx, "%rd5", vectors, firstRegister: 0);
            EmitLoads(ptx, "%rd4", vectors, firstRegister: capsuleDim);
        }
        else
        {
            ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
            ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
            ptx.AppendLine("    add.u64 %rd5, %rd1, %rd0;");   // x
            ptx.AppendLine("    add.u64 %rd6, %rd2, %rd0;");   // y
            EmitLoads(ptx, "%rd5", vectors, firstRegister: 0);
        }

        // %fd0 = r2 = sum x*x, %fd1 = dot = sum x*g (backward); %fd2/%fd3 are the widened element scratch.
        ptx.AppendLine("    mov.f64 %fd0, 0d0000000000000000;");
        if (backward) ptx.AppendLine("    mov.f64 %fd1, 0d0000000000000000;");
        for (int j = 0; j < capsuleDim; j++)
        {
            ptx.AppendLine($"    cvt.f64.f32 %fd2, %f{j};");
            ptx.AppendLine("    fma.rn.f64 %fd0, %fd2, %fd2, %fd0;");
            if (backward)
            {
                ptx.AppendLine($"    cvt.f64.f32 %fd3, %f{capsuleDim + j};");
                ptx.AppendLine("    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;");
            }
        }

        ptx.AppendLine("    sqrt.rn.f64 %fd4, %fd0;");                               // norm
        ptx.AppendLine($"    add.rn.f64 %fd5, %fd4, {eps};");                         // norm + eps
        ptx.AppendLine("    add.rn.f64 %fd6, %fd0, 0d3FF0000000000000;");            // 1 + r2
        ptx.AppendLine("    mul.rn.f64 %fd7, %fd6, %fd5;");                          // denominator
        ptx.AppendLine("    setp.neu.f64 %p0, %fd7, 0d0000000000000000;");
        ptx.AppendLine("    div.rn.f64 %fd8, %fd0, %fd7;");
        ptx.AppendLine("    selp.f64 %fd8, %fd8, 0d0000000000000000, %p0;");         // scale = f(r)

        if (backward)
        {
            // coefficient = (norm + 2 eps - r2 norm) / denominator^2, with nvcc's contractions of the CUDA source:
            // norm + 2.0*eps -> fma(2, eps, norm), then (...) - r2*norm -> fma(-r2, norm, ...).
            ptx.AppendLine($"    fma.rn.f64 %fd9, 0d4000000000000000, {eps}, %fd4;");
            ptx.AppendLine("    neg.f64 %fd10, %fd0;");
            ptx.AppendLine("    fma.rn.f64 %fd9, %fd10, %fd4, %fd9;");
            ptx.AppendLine("    mul.rn.f64 %fd11, %fd7, %fd7;");
            ptx.AppendLine("    div.rn.f64 %fd12, %fd9, %fd11;");
            ptx.AppendLine("    selp.f64 %fd12, %fd12, 0d0000000000000000, %p0;");  // coefficient
            for (int j = 0; j < capsuleDim; j++)
            {
                // dx = scale*g + (coefficient*x)*dot -> fma(coefficient*x, dot, scale*g), rounded to FP32 once.
                ptx.AppendLine($"    cvt.f64.f32 %fd2, %f{j};");
                ptx.AppendLine($"    cvt.f64.f32 %fd3, %f{capsuleDim + j};");
                ptx.AppendLine("    mul.rn.f64 %fd13, %fd8, %fd3;");
                ptx.AppendLine("    mul.rn.f64 %fd14, %fd12, %fd2;");
                ptx.AppendLine("    fma.rn.f64 %fd15, %fd14, %fd1, %fd13;");
                ptx.AppendLine($"    cvt.rn.f32.f64 %f{j}, %fd15;");
            }
        }
        else
        {
            for (int j = 0; j < capsuleDim; j++)
            {
                ptx.AppendLine($"    cvt.f64.f32 %fd2, %f{j};");
                ptx.AppendLine("    mul.rn.f64 %fd13, %fd8, %fd2;");
                ptx.AppendLine($"    cvt.rn.f32.f64 %f{j}, %fd13;");
            }
        }

        for (int v = 0; v < vectors; v++)
            ptx.AppendLine(
                $"    st.global.v4.f32 [%rd6+{v * 16}], {{%f{4 * v}, %f{4 * v + 1}, %f{4 * v + 2}, %f{4 * v + 3}}};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Streamed once and never revisited, so the loads go through the read-only data cache.
    private static void EmitLoads(StringBuilder ptx, string address, int vectors, int firstRegister)
    {
        for (int v = 0; v < vectors; v++)
        {
            int r = firstRegister + 4 * v;
            ptx.AppendLine(
                $"    ld.global.nc.v4.f32 {{%f{r}, %f{r + 1}, %f{r + 2}, %f{r + 3}}}, [{address}+{v * 16}];");
        }
    }

    private static string F64(double value) =>
        "0d" + BitConverter.DoubleToInt64Bits(value).ToString("X16", CultureInfo.InvariantCulture);

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        SquashDirection direction,
        int numCapsules,
        int capsuleDim,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(numCapsules, capsuleDim);
        bool backward = direction == SquashDirection.Backward;
        DirectPtxTensorContract Contract(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                extent, extent, 16, access, DirectPtxExtentMode.Exact);
        return new DirectPtxKernelBlueprint(
            Operation: backward ? "capsule-squash-backward-f32" : "capsule-squash-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"thread-capsule-b{blockThreads}-n{numCapsules}-d{capsuleDim}",
            Tensors: backward
                ? new[]
                {
                    Contract("grad_output", DirectPtxTensorAccess.Read), Contract("input", DirectPtxTensorAccess.Read),
                    Contract("grad_input", DirectPtxTensorAccess.Write)
                }
                : new[] { Contract("input", DirectPtxTensorAccess.Read), Contract("output", DirectPtxTensorAccess.Write) },
            // FP64 values take register pairs; 64 registers at 128 threads still leaves eight blocks per SM.
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 65_536 / (64 * blockThreads)),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = backward
                    ? "dx = f(r) g + (r + 2 eps - r^3) / ((1 + r^2)^2 (r + eps)^2) x dot(g, x)"
                    : "y = r^2 / ((1 + r^2)(r + eps)) x, r = |x| over the capsule",
                ["axis"] = "last",
                ["epsilon"] = "1e-8f widened to fp64",
                ["accumulator"] = "fp64-thread-resident-fma-rn",
                ["rounding"] = "one cvt.rn.f32.f64 per element at the store",
                ["zero-denominator"] = "scale and coefficient are 0, as the CUDA kernel",
                ["input"] = "fp32",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-per-element",
                ["global-output-writes"] = "one-per-element",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["alias-policy"] = backward ? "grad_input-disjoint-from-both-inputs" : "input-output-disjoint",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>
    /// The exact (capsules, width) buckets: CapsNet's primary capsules (1152 x 8, at batch 1 and 8) and its digit
    /// capsules (10 per sample x 16, at batch 128 and 1024).
    /// </summary>
    internal static bool IsSupportedShape(int numCapsules, int capsuleDim) =>
        (numCapsules, capsuleDim) is (1152, 8) or (9216, 8) or (1280, 16) or (10240, 16);

    /// <summary>
    /// Whether a shape dispatches without the experiment override. False until the shapes are verified on an NVIDIA
    /// GPU; flipping <see cref="DirectPtxFeatureGate.CapsuleSquashPromoted"/> promotes every supported shape.
    /// </summary>
    internal static bool IsPromotedShape(int numCapsules, int capsuleDim) =>
        DirectPtxFeatureGate.CapsuleSquashPromoted && IsSupportedShape(numCapsules, capsuleDim);

    private static void Validate(int numCapsules, int capsuleDim)
    {
        if (!IsSupportedShape(numCapsules, capsuleDim))
            throw new ArgumentOutOfRangeException(nameof(numCapsules),
                "The capsule-squash family supports exact (capsules,width) buckets (1152,8), (9216,8), (1280,16) " +
                "and (10240,16).");
    }

    private static void ValidateBlockThreads(int numCapsules, int blockThreads)
    {
        if (blockThreads is not (64 or 128 or 256) || numCapsules % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Capsule-squash block threads must be 64, 128 or 256 and evenly tile the capsules.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
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
