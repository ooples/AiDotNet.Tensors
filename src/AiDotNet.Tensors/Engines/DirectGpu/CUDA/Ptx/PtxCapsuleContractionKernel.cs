using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which capsule contraction this kernel specializes. Both contract the input over its feature axis
/// (<c>inputDim</c>); they differ only in the weight tensor's physical layout.
/// </summary>
internal enum DirectPtxCapsuleOp
{
    /// <summary>output[b,i,c,d] = sum_k input[b,i,k] * weights[i,c,k,d] (weights [I,C,K,D]).</summary>
    Predictions,
    /// <summary>output[b,i,j,d] = sum_k input[b,i,k] * weights[i,k,j,d] (weights [I,K,J,D]).</summary>
    Transform
}

/// <summary>
/// Capsule contraction over the input feature axis (issue #854), matching the NVRTC
/// <c>capsule_predictions</c> and <c>capsule_transform</c> kernels. One thread owns one output
/// element and walks the contracted axis <c>k = 0..inputDim-1</c> serially in registers — no shared
/// memory, no reduction. The two variants share this emitter; only the baked weight strides and the
/// entry-point name differ (<see cref="DirectPtxCapsuleOp"/>).
///
/// Logical dims are (batchSize B, inputCapsules I, inputDim K, outputCount C, outputDim D). Input is
/// <c>[B,I,K]</c>; output is <c>[B,I,C,D]</c> row-major, so the flat output index is the thread index.
/// Shape is baked into the PTX; the launch takes buffer pointers only. 256 threads/block,
/// grid = (B*I*C*D)/256, required to divide evenly (no divergent bounds guard).
/// </summary>
internal sealed class PtxCapsuleContractionKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxOutputs = 2048 * 4096;
    internal const int MaxContraction = 4096;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCapsuleOp Op { get; }
    internal int BatchSize { get; }
    internal int InputCapsules { get; }
    internal int InputDim { get; }
    internal int OutputCount { get; }
    internal int OutputDim { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxCapsuleOp op) => op switch
    {
        DirectPtxCapsuleOp.Predictions => "aidotnet_capsule_predictions",
        DirectPtxCapsuleOp.Transform => "aidotnet_capsule_transform",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    // Weight strides over (i, c/j, k) for weightIdx = i*Wi + cj*Wcj + k*Wk + d.
    private static (int Wi, int Wcj, int Wk) WeightStrides(DirectPtxCapsuleOp op, int inputDim, int outputCount, int outputDim) => op switch
    {
        // weights [I,C,K,D]: Wi = C*K*D, Wc = K*D, Wk = D.
        DirectPtxCapsuleOp.Predictions => (outputCount * inputDim * outputDim, inputDim * outputDim, outputDim),
        // weights [I,K,J,D]: Wi = K*J*D, Wj = D, Wk = J*D.
        DirectPtxCapsuleOp.Transform => (inputDim * outputCount * outputDim, outputDim, outputCount * outputDim),
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxCapsuleContractionKernel(
        DirectPtxRuntime runtime, DirectPtxCapsuleOp op,
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in capsule-contraction specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, inputCapsules, inputDim, outputCount, outputDim);
        Op = op;
        BatchSize = batchSize;
        InputCapsules = inputCapsules;
        InputDim = inputDim;
        OutputCount = outputCount;
        OutputDim = outputDim;
        EntryPoint = EntryPointFor(op);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, batchSize, inputCapsules, inputDim, outputCount, outputDim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, op, batchSize, inputCapsules, inputDim, outputCount, outputDim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightsPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &weightsPointer;
        arguments[2] = &outputPointer;
        uint grid = (uint)((BatchSize * InputCapsules * OutputCount * OutputDim) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, DirectPtxCapsuleOp op,
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        ValidateShape(batchSize, inputCapsules, inputDim, outputCount, outputDim);
        string entry = EntryPointFor(op);
        (int wi, int wcj, int wk) = WeightStrides(op, inputDim, outputCount, outputDim);
        int inStrideB = inputCapsules * inputDim;   // input [B,I,K]

        var ptx = new StringBuilder(4_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// {entry} B={batchSize} I={inputCapsules} K={inputDim} C={outputCount} D={outputDim} (Wi={wi},Wcj={wcj},Wk={wk})");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx
        ptx.AppendLine($"    rem.u32 %r3, %r2, {outputDim};");             // d
        ptx.AppendLine($"    div.u32 %r4, %r2, {outputDim};");             // t = idx/D
        ptx.AppendLine($"    rem.u32 %r5, %r4, {outputCount};");           // cj
        ptx.AppendLine($"    div.u32 %r6, %r4, {outputCount};");           // t2 = t/C
        ptx.AppendLine($"    rem.u32 %r7, %r6, {inputCapsules};");         // i
        ptx.AppendLine($"    div.u32 %r8, %r6, {inputCapsules};");         // b
        // inputBaseElem = b*inStrideB + i*inputDim
        ptx.AppendLine($"    mul.lo.u32 %r9, %r8, {inStrideB};");
        ptx.AppendLine($"    mad.lo.u32 %r9, %r7, {inputDim}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &input[b,i,0]
        // weightBaseElem = i*Wi + cj*Wcj + d
        ptx.AppendLine($"    mul.lo.u32 %r10, %r7, {wi};");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r5, {wcj}, %r10;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");                   // &weights[i,cj,0,d]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        ptx.AppendLine("    mov.u32 %r11, 0;");                           // k = 0
        ptx.AppendLine("$CAPS_K_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");             // input[b,i,k]
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");             // weights[i,cj,k,d]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");                    // input k stride = 1 elem
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {wk * 4};");           // weight k stride = Wk elems
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r11, {inputDim};");
        ptx.AppendLine("    @%p0 bra $CAPS_K_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, DirectPtxCapsuleOp op,
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        var inputExtent = new DirectPtxExtent(batchSize * inputCapsules * inputDim);
        var weightExtent = new DirectPtxExtent(inputCapsules * outputCount * inputDim * outputDim);
        var outputExtent = new DirectPtxExtent(batchSize * inputCapsules * outputCount * outputDim);
        return new DirectPtxKernelBlueprint(
            Operation: op == DirectPtxCapsuleOp.Predictions ? "capsule-predictions" : "capsule-transform",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-i{inputCapsules}-k{inputDim}-c{outputCount}-d{outputDim}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    weightExtent, weightExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
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
                ["formula"] = op == DirectPtxCapsuleOp.Predictions
                    ? "output[b,i,c,d] = sum_k input[b,i,k] * weights[i,c,k,d]"
                    : "output[b,i,j,d] = sum_k input[b,i,k] * weights[i,k,j,d]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        if (batchSize <= 0 || inputCapsules <= 0 || inputDim <= 0 || outputCount <= 0 || outputDim <= 0) return false;
        if (inputDim > MaxContraction) return false;
        long outputs = (long)batchSize * inputCapsules * outputCount * outputDim;
        return outputs > 0 && outputs % BlockThreads == 0 && outputs <= MaxOutputs;
    }

    internal static bool IsPromotedShape(int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim) => false;

    private static void ValidateShape(int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        if (!IsSupportedShape(batchSize, inputCapsules, inputDim, outputCount, outputDim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Capsule contraction requires positive dims with inputDim<={MaxContraction} and (B*I*C*D) a multiple of {BlockThreads} up to {MaxOutputs}.");
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
