using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Radial-basis-function activation <c>output[b,c] = exp(-epsilons[c] * ||input[b] - centers[c]||^2)</c>
/// (issue #854). One thread owns one (batch, center) output pair, walking the <c>inputDim</c> feature
/// axis serially in registers — no shared memory, no reduction — matching the NVRTC <c>rbf_forward</c>
/// kernel exactly. <c>expf</c> is reconstructed as <c>ex2.approx.f32(x * log2(e))</c>, the same
/// transcendental path used by the softmax family.
///
/// Shape is baked into the PTX (batchSize, numCenters, inputDim are compile-time constants), so the
/// launch takes buffer pointers only. 256 threads/block, grid = (batchSize*numCenters)/256, which is
/// required to divide evenly so there is no divergent bounds guard.
/// </summary>
internal sealed class PtxRbfForwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxPairs = 2048 * 4096;
    internal const int MaxInputDim = 4096;
    internal const string EntryPoint = "aidotnet_rbf_forward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int NumCenters { get; }
    internal int InputDim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRbfForwardKernel(DirectPtxRuntime runtime, int batchSize, int numCenters, int inputDim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in RBF-forward specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, numCenters, inputDim);
        BatchSize = batchSize;
        NumCenters = numCenters;
        InputDim = inputDim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, numCenters, inputDim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, numCenters, inputDim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input, DirectPtxTensorView centers,
        DirectPtxTensorView epsilons, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(centers, Blueprint.Tensors[1], nameof(centers));
        Require(epsilons, Blueprint.Tensors[2], nameof(epsilons));
        Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr centersPointer = centers.Pointer;
        IntPtr epsilonsPointer = epsilons.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &centersPointer;
        arguments[2] = &epsilonsPointer;
        arguments[3] = &outputPointer;
        uint grid = (uint)((BatchSize * NumCenters) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int numCenters, int inputDim)
    {
        ValidateShape(batchSize, numCenters, inputDim);
        string log2e = Hex(1.4426950408889634f);

        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// rbf-forward batch={batchSize} centers={numCenters} dim={inputDim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 centers_ptr,");
        ptx.AppendLine("    .param .u64 epsilons_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [centers_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [epsilons_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx = b*NC + c
        ptx.AppendLine($"    div.u32 %r3, %r2, {numCenters};");            // b
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numCenters};");            // c
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {inputDim};");           // b*ID
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {inputDim};");           // c*ID
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");                   // &input[b*ID]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // &centers[c*ID]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // distSq
        ptx.AppendLine("    mov.u32 %r7, 0;");                            // d = 0
        ptx.AppendLine("$RBF_DIM_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    sub.rn.f32 %f3, %f1, %f2;");                  // diff
        ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f3, %f0;");             // distSq += diff*diff
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r7, {inputDim};");
        ptx.AppendLine("    @%p0 bra $RBF_DIM_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd9];");             // eps = epsilons[c]
        ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f0;");                 // eps*distSq
        ptx.AppendLine("    neg.f32 %f5, %f5;");                         // -eps*distSq
        ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {log2e};");            // * log2(e)
        ptx.AppendLine("    ex2.approx.f32 %f6, %f5;");                  // exp(-eps*distSq)
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f6;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batchSize, int numCenters, int inputDim)
    {
        var inputExtent = new DirectPtxExtent(batchSize * inputDim);
        var centersExtent = new DirectPtxExtent(numCenters * inputDim);
        var epsilonsExtent = new DirectPtxExtent(numCenters);
        var outputExtent = new DirectPtxExtent(batchSize * numCenters);
        return new DirectPtxKernelBlueprint(
            Operation: "rbf-forward",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-c{numCenters}-d{inputDim}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("centers", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    centersExtent, centersExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("epsilons", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    epsilonsExtent, epsilonsExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[b,c] = exp(-epsilons[c] * sum_d (input[b,d]-centers[c,d])^2)",
                ["approximation"] = "expf via ex2.approx.f32(x*log2e)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int numCenters, int inputDim)
    {
        if (batchSize <= 0 || numCenters <= 0 || inputDim <= 0 || inputDim > MaxInputDim) return false;
        long pairs = (long)batchSize * numCenters;
        return pairs > 0 && pairs % BlockThreads == 0 && pairs <= MaxPairs;
    }

    internal static bool IsPromotedShape(int batchSize, int numCenters, int inputDim) => false;

    private static void ValidateShape(int batchSize, int numCenters, int inputDim)
    {
        if (!IsSupportedShape(batchSize, numCenters, inputDim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"RBF forward requires positive dims with inputDim<={MaxInputDim} and (batchSize*numCenters) a multiple of {BlockThreads} up to {MaxPairs}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
