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
/// <c>capsule_predictions</c> and <c>capsule_transform</c> kernels. Shapes with 16-wide input and
/// output dimensions use 16x16x8 TF32 WMMA tiles with FP32 accumulation. Other divisible shapes
/// assign four or eight contiguous outputs to each thread so an input value is shared across
/// accumulators and weights use aligned vector loads; remaining shapes retain the scalar path.
///
/// Logical dims are (batchSize B, inputCapsules I, inputDim K, outputCount C, outputDim D). Input is
/// <c>[B,I,K]</c>; output is <c>[B,I,C,D]</c> row-major. Shape is baked into the PTX; the launch
/// takes buffer pointers only. Scalar paths use 256-thread blocks; the WMMA path uses four-warp
/// 128-thread blocks so its fixed warp count is distributed evenly across the GPU. Both launch
/// shapes divide their selected output tiles exactly and need no bounds guard.
/// </summary>
internal sealed class PtxCapsuleContractionKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int TensorCoreBlockThreads = 128;
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
    internal int OutputsPerThread { get; }
    internal bool UsesTensorCores { get; }
    internal int LaunchThreads { get; }
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
        UsesTensorCores = UseTensorCorePath(
            batchSize, inputCapsules, inputDim, outputCount, outputDim);
        LaunchThreads = UsesTensorCores ? TensorCoreBlockThreads : BlockThreads;
        OutputsPerThread = UsesTensorCores
            ? 16
            : GetOutputsPerThread(batchSize, inputCapsules, outputCount, outputDim);
        EntryPoint = EntryPointFor(op);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, batchSize, inputCapsules, inputDim, outputCount, outputDim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, op, batchSize, inputCapsules, inputDim, outputCount, outputDim);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, LaunchThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, LaunchThreads, activeBlocks);
                var audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, LaunchThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbi.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbi.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightsPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &weightsPointer;
        arguments[2] = &outputPointer;
        if (UsesTensorCores)
        {
            _module.Launch(
                _function,
                (uint)(OutputCount / (TensorCoreBlockThreads / 32)),
                (uint)InputCapsules,
                (uint)(BatchSize / 16),
                TensorCoreBlockThreads, 1, 1, 0, arguments);
        }
        else
        {
            uint grid = (uint)((BatchSize * InputCapsules * OutputCount * OutputDim) /
                (BlockThreads * OutputsPerThread));
            _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
        }
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, DirectPtxCapsuleOp op,
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        ValidateShape(batchSize, inputCapsules, inputDim, outputCount, outputDim);
        if (UseTensorCorePath(batchSize, inputCapsules, inputDim, outputCount, outputDim))
            return EmitTensorCorePtx(
                ccMajor, ccMinor, op,
                batchSize, inputCapsules, inputDim, outputCount, outputDim);
        string entry = EntryPointFor(op);
        (int wi, int wcj, int wk) = WeightStrides(op, inputDim, outputCount, outputDim);
        int inStrideB = inputCapsules * inputDim;   // input [B,I,K]
        int outputsPerThread = GetOutputsPerThread(batchSize, inputCapsules, outputCount, outputDim);

        var ptx = new StringBuilder(4_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: outputsPerThread > 1);
        ptx.AppendLine($"// {entry} B={batchSize} I={inputCapsules} K={inputDim} C={outputCount} D={outputDim} x{outputsPerThread} (Wi={wi},Wcj={wcj},Wk={wk})");
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
        ptx.AppendLine($"    .reg .f32 %f<{(outputsPerThread > 1 ? outputsPerThread * 2 + 2 : 4)}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx
        int dimensionGroups = outputDim / outputsPerThread;
        ptx.AppendLine($"    rem.u32 %r3, %r2, {dimensionGroups};");       // d or d-group
        ptx.AppendLine($"    div.u32 %r4, %r2, {dimensionGroups};");       // t = idx/dimensionGroups
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
        if (outputsPerThread > 1)
            ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {outputsPerThread}, %r10;");
        else
            ptx.AppendLine("    add.u32 %r10, %r10, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");                   // &weights[i,cj,0,d]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        for (int outputIndex = 1; outputIndex < outputsPerThread; outputIndex++)
            ptx.AppendLine($"    mov.f32 %f{outputIndex}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r11, 0;");                           // k = 0
        ptx.AppendLine("$CAPS_K_LOOP:");
        if (outputsPerThread > 1)
        {
            int inputRegister = outputsPerThread;
            int weightRegister = inputRegister + 1;
            ptx.AppendLine($"    ld.global.nc.f32 %f{inputRegister}, [%rd6];");
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex += 4)
            {
                int value = weightRegister + outputIndex;
                int byteOffset = outputIndex * sizeof(float);
                ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f{value}, %f{value + 1}, %f{value + 2}, %f{value + 3}}}, [%rd7+{byteOffset}];");
            }
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex++)
                ptx.AppendLine($"    fma.rn.f32 %f{outputIndex}, %f{inputRegister}, %f{weightRegister + outputIndex}, %f{outputIndex};");
        }
        else
        {
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
            ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");                    // input k stride = 1 elem
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {wk * 4};");           // weight k stride = Wk elems
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r11, {inputDim};");
        ptx.AppendLine("    @%p0 bra $CAPS_K_LOOP;");
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r2, {outputsPerThread * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        if (outputsPerThread > 1)
        {
            for (int outputIndex = 0; outputIndex < outputsPerThread; outputIndex += 4)
            {
                int byteOffset = outputIndex * sizeof(float);
                ptx.AppendLine($"    st.global.v4.f32 [%rd9+{byteOffset}], {{%f{outputIndex}, %f{outputIndex + 1}, %f{outputIndex + 2}, %f{outputIndex + 3}}};");
            }
        }
        else
            ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string EmitTensorCorePtx(
        int ccMajor, int ccMinor, DirectPtxCapsuleOp op,
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        string entry = EntryPointFor(op);
        int inputRowStride = inputCapsules * inputDim;
        int weightInputCapsuleStride = inputDim * outputCount * outputDim;
        int weightOutputStride = op == DirectPtxCapsuleOp.Predictions
            ? inputDim * outputDim
            : outputDim;
        int weightKStride = op == DirectPtxCapsuleOp.Predictions
            ? outputDim
            : outputCount * outputDim;
        int outputRowStride = inputCapsules * outputCount * outputDim;

        var ptx = new StringBuilder(3_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor);
        ptx.AppendLine($"// {entry} tensor-core B={batchSize} I={inputCapsules} K={inputDim} C={outputCount} D={outputDim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {TensorCoreBlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .b32 %a<4>;");
        ptx.AppendLine("    .reg .b32 %b<4>;");
        ptx.AppendLine("    .reg .f32 %d<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");                       // warp in block
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r2, {TensorCoreBlockThreads / 32}, %r1;"); // c/j
        ptx.AppendLine("    mov.u32 %r6, %ctaid.y;");                     // i
        ptx.AppendLine("    mov.u32 %r7, %ctaid.z;");                     // b tile

        // A = input[bTile*16, i, 0], row-major 16x16 with physical B-row stride I*K.
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {16 * inputRowStride};");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r6, {inputDim}, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");

        // B = weights[i,c/j,0,0], with the operation-specific K row stride.
        ptx.AppendLine($"    mul.lo.u32 %r9, %r6, {weightInputCapsuleStride};");
        ptx.AppendLine($"    mad.lo.u32 %r9, %r4, {weightOutputStride}, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");

        // C = output[bTile*16, i, c/j, 0], row-major with physical B-row stride I*C*D.
        ptx.AppendLine($"    mul.lo.u32 %r10, %r7, {16 * outputRowStride};");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r6, {outputCount * outputDim}, %r10;");
        ptx.AppendLine($"    mad.lo.u32 %r10, %r4, {outputDim}, %r10;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd5;");

        for (int accumulator = 0; accumulator < 8; accumulator++)
            ptx.AppendLine($"    mov.f32 %d{accumulator}, 0f00000000;");
        for (int kTile = 0; kTile < inputDim; kTile += 8)
        {
            ptx.AppendLine($"    add.u64 %rd9, %rd6, {kTile * sizeof(float)};");
            ptx.AppendLine($"    add.u64 %rd10, %rd7, {kTile * weightKStride * sizeof(float)};");
            ptx.AppendLine($"    wmma.load.a.sync.aligned.row.m16n16k8.global.tf32 " +
                $"{{%a0,%a1,%a2,%a3}}, [%rd9], {inputRowStride};");
            ptx.AppendLine($"    wmma.load.b.sync.aligned.row.m16n16k8.global.tf32 " +
                $"{{%b0,%b1,%b2,%b3}}, [%rd10], {weightKStride};");
            ptx.AppendLine("    wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32 " +
                "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, " +
                "{%a0,%a1,%a2,%a3}, {%b0,%b1,%b2,%b3}, " +
                "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7};");
        }
        ptx.AppendLine($"    wmma.store.d.sync.aligned.row.m16n16k8.global.f32 " +
            $"[%rd8], {{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}}, {outputRowStride};");
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
            Version: 4,
            Architecture: architecture,
            Variant: UseTensorCorePath(batchSize, inputCapsules, inputDim, outputCount, outputDim)
                ? $"tf32-wmma-t{TensorCoreBlockThreads}-b{batchSize}-i{inputCapsules}-k{inputDim}-c{outputCount}-d{outputDim}"
                : $"fp32-x{GetOutputsPerThread(batchSize, inputCapsules, outputCount, outputDim)}-b{batchSize}-i{inputCapsules}-k{inputDim}-c{outputCount}-d{outputDim}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    weightExtent, weightExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: UseTensorCorePath(batchSize, inputCapsules, inputDim, outputCount, outputDim)
                ? DirectPtxResourceBudget.FromDriverMeasurement(
                    measuredRegistersPerThread: 38,
                    maxStaticSharedBytes: 0,
                    maxLocalBytesPerThread: 0,
                    minBlocksPerMultiprocessor: 1)
                : DirectPtxResourceBudget.FromDriverMeasurement(
                    measuredRegistersPerThread: GetOutputsPerThread(batchSize, inputCapsules, outputCount, outputDim) switch
                    {
                        8 => 28,
                        4 => 18,
                        _ => 26
                    },
                    maxStaticSharedBytes: 0,
                    maxLocalBytesPerThread: 0,
                    minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = op == DirectPtxCapsuleOp.Predictions
                    ? "output[b,i,c,d] = sum_k input[b,i,k] * weights[i,c,k,d]"
                    : "output[b,i,j,d] = sum_k input[b,i,k] * weights[i,k,j,d]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["outputs-per-thread"] = UseTensorCorePath(batchSize, inputCapsules, inputDim, outputCount, outputDim)
                    ? "wmma-16x16"
                    : GetOutputsPerThread(batchSize, inputCapsules, outputCount, outputDim)
                        .ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["numeric-mode"] = UseTensorCorePath(batchSize, inputCapsules, inputDim, outputCount, outputDim)
                    ? "TF32 multiply with FP32 accumulation"
                    : "FP32 multiply with FP32 accumulation"
            });
    }

    private static int GetOutputsPerThread(
        int batchSize, int inputCapsules, int outputCount, int outputDim)
    {
        long outputs = (long)batchSize * inputCapsules * outputCount * outputDim;
        if (outputDim % 8 == 0 && outputs % (BlockThreads * 8L) == 0) return 8;
        return outputDim % 4 == 0 && outputs % (BlockThreads * 4L) == 0 ? 4 : 1;
    }

    private static bool UseTensorCorePath(
        int batchSize, int inputCapsules, int inputDim, int outputCount, int outputDim)
    {
        long tiles = (long)(batchSize / 16) * inputCapsules * outputCount;
        return batchSize % 16 == 0 && outputCount % (TensorCoreBlockThreads / 32) == 0 &&
            inputDim == 16 && outputDim == 16 && tiles > 0;
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

}
