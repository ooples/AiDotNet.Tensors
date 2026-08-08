using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Radial-basis-function activation <c>output[b,c] = exp(-epsilons[c] * ||input[b] - centers[c]||^2)</c>
/// (issue #854). One warp owns one (batch, center) output pair, loading the <c>inputDim</c>
/// feature axis in coalesced lane-strided chunks and reducing the squared distance with shuffles.
/// <c>expf</c> is reconstructed as <c>ex2.approx.f32(x * log2(e))</c>, the same
/// transcendental path used by the softmax family.
///
/// Shape is baked into the PTX (batchSize, numCenters, inputDim are compile-time constants), so the
/// launch takes buffer pointers only. 256 threads/block (8 cells/block), grid =
/// (batchSize*numCenters)/8, which is required to divide evenly so there is no bounds guard.
/// </summary>
internal sealed class PtxRbfForwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int WarpsPerBlock = BlockThreads / 32;
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
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                var audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input, DirectPtxTensorView centers,
        DirectPtxTensorView epsilons, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbi.Require(centers, Blueprint.Tensors[1], nameof(centers));
        DirectPtxAbi.Require(epsilons, Blueprint.Tensors[2], nameof(epsilons));
        DirectPtxAbi.Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr centersPointer = centers.Pointer;
        IntPtr epsilonsPointer = epsilons.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &centersPointer;
        arguments[2] = &epsilonsPointer;
        arguments[3] = &outputPointer;
        uint grid = (uint)((BatchSize * NumCenters) / WarpsPerBlock);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();


    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int numCenters, int inputDim)
    {
        ValidateShape(batchSize, numCenters, inputDim);
        string log2e = DirectPtxPtxText.Hex(1.4426950408889634f);

        var ptx = new StringBuilder(4_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// rbf-forward batch={batchSize} centers={numCenters} dim={inputDim}; one cell per warp");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 centers_ptr,");
        ptx.AppendLine("    .param .u64 epsilons_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [centers_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [epsilons_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                        // warp in block
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                       // lane
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {WarpsPerBlock}, %r2;"); // pair = b*NC + c
        ptx.AppendLine($"    div.u32 %r5, %r4, {numCenters};");            // b
        ptx.AppendLine($"    rem.u32 %r6, %r4, {numCenters};");            // c
        ptx.AppendLine($"    mad.lo.u32 %r7, %r5, {inputDim}, %r3;");      // b*ID + lane
        ptx.AppendLine($"    mad.lo.u32 %r8, %r6, {inputDim}, %r3;");      // c*ID + lane
        ptx.AppendLine("    mul.wide.u32 %rd4, %r7, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");                   // &input[b*ID]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // &centers[c*ID]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // distSq
        ptx.AppendLine("    mov.u32 %r9, %r3;");                          // d = lane
        ptx.AppendLine("$RBF_DIM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r9, {inputDim};");
        ptx.AppendLine("    @%p0 bra $RBF_REDUCE;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    sub.rn.f32 %f3, %f1, %f2;");                  // diff
        ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f3, %f0;");             // distSq += diff*diff
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r9, %r9, 32;");
        ptx.AppendLine("    bra $RBF_DIM_LOOP;");
        ptx.AppendLine("$RBF_REDUCE:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f0", "%r11", "%r12", "%f7");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra $RBF_END;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd9];");             // eps = epsilons[c]
        ptx.AppendLine("    mul.rn.f32 %f5, %f4, %f0;");                 // eps*distSq
        ptx.AppendLine("    neg.f32 %f5, %f5;");                         // -eps*distSq
        ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {log2e};");            // * log2(e)
        ptx.AppendLine("    ex2.approx.f32 %f6, %f5;");                  // exp(-eps*distSq)
        ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f6;");
        ptx.AppendLine("$RBF_END:");
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
            Version: 2,
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
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
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
        return pairs > 0 && pairs % WarpsPerBlock == 0 && pairs <= MaxPairs;
    }

    internal static bool IsPromotedShape(int batchSize, int numCenters, int inputDim) => false;

    private static void ValidateShape(int batchSize, int numCenters, int inputDim)
    {
        if (!IsSupportedShape(batchSize, numCenters, inputDim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"RBF forward requires positive dims with inputDim<={MaxInputDim} and (batchSize*numCenters) a multiple of {WarpsPerBlock} up to {MaxPairs}.");
    }

}
