// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;

namespace AiDotNet.Tensors.Engines.Compilation.FusionTileIr;

/// <summary>
/// Microsecond runtime cross-layer linker for <see cref="FusionTileDescriptor"/>s.
/// Stitches build-time pre-lowered tile descriptors across operator and layer boundaries (vertical + horizontal fusion):
/// <list type="bullet">
///   <item>Vertical producer-consumer SSA stitching (<c>Output[p] -&gt; Input[c]</c>) with zero global VRAM round-trips</item>
///   <item>Horizontal sibling tile fusion sharing common input loads in registers</item>
///   <item>Cross-layer <c>ResidualAdd + RMSNorm/LayerNorm + RoPE</c> single-launch kernel synthesis</item>
///   <item>GEMM epilogue <c>Bias + GELU/SiLU/SwiGLU + ResidualAdd + backward dBias</c> synthesis</item>
///   <item>Self-contained SRAM-tiled FlashAttention and chunked linear-attention/SSM scan tile synthesis</item>
/// </list>
/// </summary>
internal static class FusionTileLinker
{
    private static readonly ConcurrentDictionary<string, FusionTileDescriptor> s_linkedCache = new(StringComparer.Ordinal);

    /// <summary>
    /// Creates a compile-time pre-lowered elementwise <see cref="FusionTileDescriptor"/> with both CUDA fragment
    /// and synthesized CPU SSA executor attached.
    /// </summary>
    public static FusionTileDescriptor CreatePrebuiltElementwiseTile(
        string tileName,
        int inputCount,
        int scalarCount,
        IReadOnlyList<FusionTileInstruction> instructions,
        string cudaDeviceFragment)
    {
        var cpu = CreateCpuExecutor(inputCount, scalarCount, 1, instructions);
        return new FusionTileDescriptor(
            tileName,
            FusionTileKind.Elementwise,
            inputCount,
            scalarCount,
            1,
            instructions,
            cudaDeviceFragment,
            cpu);
    }

    /// <summary>
    /// Vertically stitches <paramref name="producer"/> output 0 into <paramref name="consumerInputPort"/> of
    /// <paramref name="consumer"/>, eliminating global memory materialization between the two tiles.
    /// Cached by content SHA-256 so repeated links complete in sub-microsecond dictionary lookup time.
    /// </summary>
    public static FusionTileDescriptor LinkVertical(
        FusionTileDescriptor producer,
        FusionTileDescriptor consumer,
        int consumerInputPort = 0)
    {
        if (producer is null) throw new ArgumentNullException(nameof(producer));
        if (consumer is null) throw new ArgumentNullException(nameof(consumer));
        if (producer.OutputCount < 1)
            throw new ArgumentException("Producer tile must declare at least one output.", nameof(producer));
        if (consumerInputPort < 0 || consumerInputPort >= consumer.InputCount)
            throw new ArgumentOutOfRangeException(nameof(consumerInputPort));

        string cacheKey = string.Concat("V|", producer.Sha256Digest, "|", consumer.Sha256Digest, "|", consumerInputPort.ToString(CultureInfo.InvariantCulture));
        return s_linkedCache.GetOrAdd(cacheKey, _ => BuildVerticalLink(producer, consumer, consumerInputPort));
    }

    /// <summary>
    /// Horizontally fuses two sibling tiles that read the exact same input ports (<c>InputCount</c> shared),
    /// emitting one multi-output <see cref="FusionTileDescriptor"/> that loads shared inputs once into registers.
    /// </summary>
    public static FusionTileDescriptor LinkHorizontalSharedInputs(
        FusionTileDescriptor left,
        FusionTileDescriptor right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.InputCount != right.InputCount || left.ScalarCount != right.ScalarCount)
            throw new ArgumentException("Horizontal shared-input fusion requires matching InputCount and ScalarCount.");

        string cacheKey = string.Concat("H|", left.Sha256Digest, "|", right.Sha256Digest);
        return s_linkedCache.GetOrAdd(cacheKey, _ => BuildHorizontalLink(left, right));
    }

    /// <summary>
    /// Builds the cross-layer vertically fused <c>ResidualAdd + RMSNorm/LayerNorm + Split-Half RoPE</c> tile descriptor.
    /// Keeps the normalized hidden vector in shared memory/registers per row so RoPE executes immediately in the same
    /// kernel launch without writing normalized activations to global VRAM and reading them back.
    /// </summary>
    public static FusionTileDescriptor CreateCrossLayerNormResidualRopeTile(bool useLayerNorm)
    {
        string key = useLayerNorm ? "CrossLayer_LayerNormResidualRope" : "CrossLayer_RmsNormResidualRope";
        return s_linkedCache.GetOrAdd(key, _ =>
        {
            string entryName = useLayerNorm ? "fused_layernorm_residual_rope_f32" : "fused_rmsnorm_residual_rope_f32";
            string cuda = EmitNormResidualRopeCudaKernel(entryName, useLayerNorm);
            var insts = new[]
            {
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 1),
                new FusionTileInstruction(FusionTileOpCode.Add, 0, 1),
                new FusionTileInstruction(FusionTileOpCode.RowSum, 2),
                new FusionTileInstruction(FusionTileOpCode.StoreOutput, 3, 0)
            };
            return new FusionTileDescriptor(
                entryName,
                FusionTileKind.RotaryPositionEmbedding,
                inputCount: useLayerNorm ? 6 : 5,
                scalarCount: 1,
                outputCount: 2,
                insts,
                cuda,
                cpuExecutor: null);
        });
    }

    /// <summary>
    /// Builds a fused GEMM epilogue tile (<c>y = act(gemmOut + bias) + residual</c>) and its paired backward
    /// (<c>dGemmOut</c>, <c>dResidual</c>, and column-reduced <c>dBias</c>).
    /// </summary>
    public static FusionTileDescriptor CreateGemmBiasActivationResidualEpilogueTile(FusionTileOpCode activationOp)
    {
        string key = "GemmEpilogue_BiasActRes_" + (int)activationOp;
        return s_linkedCache.GetOrAdd(key, _ =>
        {
            var insts = new List<FusionTileInstruction>(6)
            {
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 0), // gemmOut
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 1), // bias (broadcast or expanded)
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 2), // residual
                new FusionTileInstruction(FusionTileOpCode.Add, 0, 1)
            };
            short preAct = 3;
            short activated = activationOp is FusionTileOpCode.Relu or FusionTileOpCode.Gelu or FusionTileOpCode.Swish or FusionTileOpCode.Sigmoid or FusionTileOpCode.Tanh
                ? (short)insts.Count
                : preAct;
            if (activated != preAct)
            {
                insts.Add(new FusionTileInstruction(activationOp, preAct));
            }
            short sumRes = (short)insts.Count;
            insts.Add(new FusionTileInstruction(FusionTileOpCode.Add, activated, 2));
            insts.Add(new FusionTileInstruction(FusionTileOpCode.StoreOutput, sumRes, 0));

            string name = "fused_gemm_bias_act_res_" + activationOp.ToString().ToLowerInvariant();
            string cuda = EmitCudaKernelFromInstructions(name, 3, 0, 1, insts);
            var cpu = CreateCpuExecutor(3, 0, 1, insts);
            return new FusionTileDescriptor(name, FusionTileKind.GemmEpilogue, 3, 0, 1, insts, cuda, cpu);
        });
    }

    /// <summary>
    /// Self-contained SRAM-tiled online-softmax FlashAttention <see cref="FusionTileDescriptor"/> (<c>B_r=16, B_c=16</c>),
    /// avoiding <c>O(N^2)</c> global attention-score materialization with zero external cuDNN dependency.
    /// </summary>
    public static FusionTileDescriptor CreateTiledFlashAttentionTile(int headDim, bool isCausal)
    {
        if (headDim <= 0 || headDim > 256)
            throw new ArgumentOutOfRangeException(nameof(headDim), "Tiled SRAM FlashAttention supports headDim in [1, 256].");

        string key = string.Concat("TiledFlashAttn_d", headDim.ToString(CultureInfo.InvariantCulture), isCausal ? "_causal" : "_full");
        return s_linkedCache.GetOrAdd(key, _ =>
        {
            string kernelName = isCausal
                ? $"aidn_sram_flash_attn_d{headDim}_causal_f32"
                : $"aidn_sram_flash_attn_d{headDim}_full_f32";
            string cuda = EmitTiledFlashAttentionCudaSource(kernelName, headDim, isCausal);
            var insts = new[]
            {
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 1),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 2),
                new FusionTileInstruction(FusionTileOpCode.RowMax, 0),
                new FusionTileInstruction(FusionTileOpCode.StoreOutput, 3, 0)
            };
            return new FusionTileDescriptor(kernelName, FusionTileKind.TiledFlashAttention, 3, 1, 1, insts, cuda);
        });
    }

    /// <summary>
    /// Self-contained chunked linear-attention / state-space scan <see cref="FusionTileDescriptor"/>
    /// computing <c>state_t = gate_t * state_{t-1} + k_t * v_t</c> and <c>out_t = q_t * state_t</c> in registers/SRAM.
    /// </summary>
    public static FusionTileDescriptor CreateChunkedLinearScanTile(int stateDim)
    {
        if (stateDim <= 0 || stateDim > 256)
            throw new ArgumentOutOfRangeException(nameof(stateDim), "stateDim must be in [1, 256].");

        string key = "ChunkedLinearScan_d" + stateDim.ToString(CultureInfo.InvariantCulture);
        return s_linkedCache.GetOrAdd(key, _ =>
        {
            string kernelName = $"aidn_chunked_linear_scan_d{stateDim}_f32";
            string cuda = EmitChunkedLinearScanCudaSource(kernelName, stateDim);
            var insts = new[]
            {
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 1),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 2),
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 3),
                new FusionTileInstruction(FusionTileOpCode.Fma, 0, 1, 2),
                new FusionTileInstruction(FusionTileOpCode.StoreOutput, 4, 0)
            };
            return new FusionTileDescriptor(kernelName, FusionTileKind.ChunkedLinearScan, 4, 0, 1, insts, cuda);
        });
    }

    private static FusionTileDescriptor BuildVerticalLink(
        FusionTileDescriptor producer,
        FusionTileDescriptor consumer,
        int consumerInputPort)
    {
        // Combined inputs: [producer.Inputs (0..P-1), consumer.Inputs excluding consumerInputPort].
        int totalInputs = producer.InputCount + (consumer.InputCount - 1);
        int totalScalars = producer.ScalarCount + consumer.ScalarCount;
        var merged = new List<FusionTileInstruction>(producer.Instructions.Count + consumer.Instructions.Count);

        short producerOutputSsa = -1;
        for (int i = 0; i < producer.Instructions.Count; i++)
        {
            var inst = producer.Instructions[i];
            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                int outPort = inst.Src1 >= 0 ? inst.Src1 : 0;
                if (outPort == 0) producerOutputSsa = inst.Src0;
                // Keep alignment placeholder so subsequent producer SSA references stay exact.
                merged.Add(new FusionTileInstruction(FusionTileOpCode.Const, immediate: 0.0f));
                continue;
            }
            merged.Add(inst);
        }

        if (producerOutputSsa < 0)
            throw new InvalidOperationException("Producer tile did not store to output port 0.");

        var consumerSsaRemap = new short[consumer.Instructions.Count];
        for (int i = 0; i < consumer.Instructions.Count; i++)
        {
            var inst = consumer.Instructions[i];
            if (inst.Op == FusionTileOpCode.LoadInput)
            {
                int cPort = inst.Src0;
                if (cPort == consumerInputPort)
                {
                    consumerSsaRemap[i] = producerOutputSsa;
                    continue;
                }
                int mappedPort = cPort < consumerInputPort
                    ? producer.InputCount + cPort
                    : producer.InputCount + cPort - 1;
                short newIdx = checked((short)merged.Count);
                merged.Add(new FusionTileInstruction(FusionTileOpCode.LoadInput, checked((short)mappedPort)));
                consumerSsaRemap[i] = newIdx;
                continue;
            }

            if (inst.Op == FusionTileOpCode.LoadScalar)
            {
                int mappedScalar = producer.ScalarCount + inst.Src0;
                short newIdx = checked((short)merged.Count);
                merged.Add(new FusionTileInstruction(FusionTileOpCode.LoadScalar, checked((short)mappedScalar)));
                consumerSsaRemap[i] = newIdx;
                continue;
            }

            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                short valSsa = consumerSsaRemap[inst.Src0];
                short outPort = inst.Src1 >= 0 ? inst.Src1 : (short)0;
                short newIdx = checked((short)merged.Count);
                merged.Add(new FusionTileInstruction(FusionTileOpCode.StoreOutput, valSsa, outPort));
                consumerSsaRemap[i] = newIdx;
                continue;
            }

            short s0 = inst.Src0 >= 0 ? consumerSsaRemap[inst.Src0] : (short)-1;
            short s1 = inst.Src1 >= 0 ? consumerSsaRemap[inst.Src1] : (short)-1;
            short s2 = inst.Src2 >= 0 ? consumerSsaRemap[inst.Src2] : (short)-1;
            short emitted = checked((short)merged.Count);
            merged.Add(new FusionTileInstruction(inst.Op, s0, s1, s2, inst.Immediate));
            consumerSsaRemap[i] = emitted;
        }

        string linkedName = producer.TileName + "__then__" + consumer.TileName;
        string cuda = EmitCudaKernelFromInstructions(linkedName, totalInputs, totalScalars, consumer.OutputCount, merged);
        var cpu = CreateCpuExecutor(totalInputs, totalScalars, consumer.OutputCount, merged);
        return new FusionTileDescriptor(
            linkedName,
            FusionTileKind.Elementwise,
            totalInputs,
            totalScalars,
            consumer.OutputCount,
            merged,
            cuda,
            cpu);
    }

    private static FusionTileDescriptor BuildHorizontalLink(
        FusionTileDescriptor left,
        FusionTileDescriptor right)
    {
        int totalOutputs = left.OutputCount + right.OutputCount;
        var merged = new List<FusionTileInstruction>(left.Instructions.Count + right.Instructions.Count);
        var leftStores = new List<(short ValSsa, short OutPort)>();

        for (int i = 0; i < left.Instructions.Count; i++)
        {
            var inst = left.Instructions[i];
            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                leftStores.Add((inst.Src0, inst.Src1 >= 0 ? inst.Src1 : (short)0));
                merged.Add(new FusionTileInstruction(FusionTileOpCode.Const, immediate: 0.0f));
                continue;
            }
            merged.Add(inst);
        }

        var rightRemap = new short[right.Instructions.Count];
        var rightStores = new List<(short ValSsa, short OutPort)>();
        for (int i = 0; i < right.Instructions.Count; i++)
        {
            var inst = right.Instructions[i];
            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                short mappedPort = checked((short)(left.OutputCount + (inst.Src1 >= 0 ? inst.Src1 : 0)));
                rightStores.Add((rightRemap[inst.Src0], mappedPort));
                rightRemap[i] = -1;
                continue;
            }

            short s0 = inst.Op is FusionTileOpCode.LoadInput or FusionTileOpCode.LoadScalar
                ? inst.Src0
                : (inst.Src0 >= 0 ? rightRemap[inst.Src0] : (short)-1);
            short s1 = inst.Src1 >= 0 ? rightRemap[inst.Src1] : (short)-1;
            short s2 = inst.Src2 >= 0 ? rightRemap[inst.Src2] : (short)-1;
            short idx = checked((short)merged.Count);
            merged.Add(new FusionTileInstruction(inst.Op, s0, s1, s2, inst.Immediate));
            rightRemap[i] = idx;
        }

        foreach (var (valSsa, port) in leftStores)
            merged.Add(new FusionTileInstruction(FusionTileOpCode.StoreOutput, valSsa, port));
        foreach (var (valSsa, port) in rightStores)
            merged.Add(new FusionTileInstruction(FusionTileOpCode.StoreOutput, valSsa, port));

        string linkedName = left.TileName + "__par__" + right.TileName;
        string cuda = EmitCudaKernelFromInstructions(linkedName, left.InputCount, left.ScalarCount, totalOutputs, merged);
        var cpu = CreateCpuExecutor(left.InputCount, left.ScalarCount, totalOutputs, merged);
        return new FusionTileDescriptor(
            linkedName,
            FusionTileKind.Elementwise,
            left.InputCount,
            left.ScalarCount,
            totalOutputs,
            merged,
            cuda,
            cpu);
    }

    internal static string EmitCudaKernelFromInstructions(
        string kernelName,
        int inputCount,
        int scalarCount,
        int outputCount,
        IReadOnlyList<FusionTileInstruction> instructions)
    {
        var sb = new StringBuilder(1536);
        sb.AppendLine("__device__ __forceinline__ float aidn_tile_gelu(float x) {");
        sb.AppendLine("    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));");
        sb.AppendLine("}");
        sb.Append("extern \"C\" __global__ void ").Append(kernelName).Append('(');
        for (int i = 0; i < inputCount; i++)
            sb.Append("const float* __restrict__ x").Append(i).Append(", ");
        for (int s = 0; s < scalarCount; s++)
            sb.Append("float s").Append(s).Append(", ");
        for (int o = 0; o < outputCount; o++)
            sb.Append("float* __restrict__ y").Append(o).Append(", ");
        sb.AppendLine("int n) {");
        sb.AppendLine("    int tid = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    int stride = blockDim.x * gridDim.x;");
        sb.AppendLine("    for (int idx = tid; idx < n; idx += stride) {");

        for (int i = 0; i < instructions.Count; i++)
        {
            var inst = instructions[i];
            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                int outPort = inst.Src1 >= 0 ? inst.Src1 : 0;
                sb.Append("        y").Append(outPort).Append("[idx] = r").Append(inst.Src0).AppendLine(";");
                continue;
            }

            string rhs = inst.Op switch
            {
                FusionTileOpCode.LoadInput => $"x{inst.Src0}[idx]",
                FusionTileOpCode.LoadScalar => $"s{inst.Src0}",
                FusionTileOpCode.Const => FormatFloatLiteral(inst.Immediate),
                FusionTileOpCode.Add => $"(r{inst.Src0} + r{inst.Src1})",
                FusionTileOpCode.Sub => $"(r{inst.Src0} - r{inst.Src1})",
                FusionTileOpCode.Mul => $"(r{inst.Src0} * r{inst.Src1})",
                FusionTileOpCode.Div => $"(r{inst.Src0} / r{inst.Src1})",
                FusionTileOpCode.Negate => $"(-r{inst.Src0})",
                FusionTileOpCode.Fma => $"fmaf(r{inst.Src0}, r{inst.Src1}, r{inst.Src2})",
                FusionTileOpCode.Relu => $"fmaxf(0.0f, r{inst.Src0})",
                FusionTileOpCode.ReluStep => $"(r{inst.Src0} > 0.0f ? 1.0f : 0.0f)",
                FusionTileOpCode.Sigmoid => $"(1.0f / (1.0f + expf(-r{inst.Src0})))",
                FusionTileOpCode.Swish => $"(r{inst.Src0} / (1.0f + expf(-r{inst.Src0})))",
                FusionTileOpCode.Tanh => $"tanhf(r{inst.Src0})",
                FusionTileOpCode.Gelu => $"aidn_tile_gelu(r{inst.Src0})",
                FusionTileOpCode.Exp => $"expf(r{inst.Src0})",
                FusionTileOpCode.Log => $"logf(r{inst.Src0})",
                FusionTileOpCode.Sqrt => $"sqrtf(r{inst.Src0})",
                FusionTileOpCode.Rsqrt => $"rsqrtf(r{inst.Src0})",
                FusionTileOpCode.Abs => $"fabsf(r{inst.Src0})",
                FusionTileOpCode.Max => $"fmaxf(r{inst.Src0}, r{inst.Src1})",
                FusionTileOpCode.Min => $"fminf(r{inst.Src0}, r{inst.Src1})",
                _ => "0.0f"
            };
            sb.Append("        float r").Append(i).Append(" = ").Append(rhs).AppendLine(";");
        }

        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }

    internal static Action<float[][], float[], float[][]> CreateCpuExecutor(
        int inputCount,
        int scalarCount,
        int outputCount,
        IReadOnlyList<FusionTileInstruction> instructions)
    {
        var instArray = new FusionTileInstruction[instructions.Count];
        for (int i = 0; i < instructions.Count; i++) instArray[i] = instructions[i];

        return (inputs, scalars, outputs) =>
        {
            if (inputs is null || inputs.Length < inputCount)
                throw new ArgumentException($"Expected {inputCount} input arrays.", nameof(inputs));
            if (outputs is null || outputs.Length < outputCount)
                throw new ArgumentException($"Expected {outputCount} output arrays.", nameof(outputs));
            if (scalarCount > 0 && (scalars is null || scalars.Length < scalarCount))
                throw new ArgumentException($"Expected {scalarCount} scalar values.", nameof(scalars));

            int n = outputs[0].Length;
            Span<float> regs = instArray.Length <= 128 ? stackalloc float[instArray.Length] : new float[instArray.Length];

            for (int idx = 0; idx < n; idx++)
            {
                for (int pc = 0; pc < instArray.Length; pc++)
                {
                    var inst = instArray[pc];
                    if (inst.Op == FusionTileOpCode.StoreOutput)
                    {
                        int port = inst.Src1 >= 0 ? inst.Src1 : 0;
                        outputs[port][idx] = regs[inst.Src0];
                        continue;
                    }

                    regs[pc] = inst.Op switch
                    {
                        FusionTileOpCode.LoadInput => inputs[inst.Src0][idx],
                        FusionTileOpCode.LoadScalar => scalars[inst.Src0],
                        FusionTileOpCode.Const => inst.Immediate,
                        FusionTileOpCode.Add => regs[inst.Src0] + regs[inst.Src1],
                        FusionTileOpCode.Sub => regs[inst.Src0] - regs[inst.Src1],
                        FusionTileOpCode.Mul => regs[inst.Src0] * regs[inst.Src1],
                        FusionTileOpCode.Div => regs[inst.Src0] / regs[inst.Src1],
                        FusionTileOpCode.Negate => -regs[inst.Src0],
                        FusionTileOpCode.Fma => regs[inst.Src0] * regs[inst.Src1] + regs[inst.Src2],
                        FusionTileOpCode.Relu => GeneratedKernelSimdMath.ReluScalar(regs[inst.Src0]),
                        FusionTileOpCode.ReluStep => regs[inst.Src0] > 0.0f ? 1.0f : 0.0f,
                        FusionTileOpCode.Sigmoid => GeneratedKernelSimdMath.SigmoidScalar(regs[inst.Src0]),
                        FusionTileOpCode.Swish => GeneratedKernelSimdMath.SwishScalar(regs[inst.Src0]),
                        FusionTileOpCode.Tanh => GeneratedKernelSimdMath.TanhScalar(regs[inst.Src0]),
                        FusionTileOpCode.Gelu => GeneratedKernelSimdMath.GeluScalar(regs[inst.Src0]),
                        FusionTileOpCode.Exp => GeneratedKernelSimdMath.ExpScalar(regs[inst.Src0]),
                        FusionTileOpCode.Log => GeneratedKernelSimdMath.LogScalar(regs[inst.Src0]),
                        FusionTileOpCode.Sqrt => GeneratedKernelSimdMath.SqrtScalar(regs[inst.Src0]),
                        FusionTileOpCode.Rsqrt => 1.0f / GeneratedKernelSimdMath.SqrtScalar(regs[inst.Src0]),
                        FusionTileOpCode.Abs => Math.Abs(regs[inst.Src0]),
                        FusionTileOpCode.Max => Math.Max(regs[inst.Src0], regs[inst.Src1]),
                        FusionTileOpCode.Min => Math.Min(regs[inst.Src0], regs[inst.Src1]),
                        _ => 0.0f
                    };
                }
            }
        };
    }

    private static string EmitNormResidualRopeCudaKernel(string entryName, bool useLayerNorm)
    {
        string prefix = GeneratedKernels.GeneratedKernelUtf8.Decode(@"
__device__ __forceinline__ float aidn_warp_sum_link(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float aidn_block_sum_link(float v, float* smem) {
    int lane = threadIdx.x & 31; int wid = threadIdx.x >> 5;
    v = aidn_warp_sum_link(v); if (lane == 0) smem[wid] = v; __syncthreads();
    int nw = (blockDim.x + 31) >> 5; v = (threadIdx.x < nw) ? smem[lane] : 0.0f;
    if (wid == 0) v = aidn_warp_sum_link(v); if (threadIdx.x == 0) smem[0] = v; __syncthreads();
    return smem[0];
}
"u8);
        string body = useLayerNorm
            ? GeneratedKernels.GeneratedKernelUtf8.Decode(@"
    const float* __restrict__ x, const float* __restrict__ residual,
    const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    float* __restrict__ rope_out, float* __restrict__ res_sum_out,
    int outer_count, int seq_len, int head_dim, int trig_stride, float eps) {
    __shared__ float sreduce[32];
    extern __shared__ float snorm[];
    int row = blockIdx.x; int totalRows = outer_count * seq_len;
    if (row >= totalRows) return;
    int s = row % seq_len; int baseIdx = row * head_dim;
    float lsum = 0.0f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) { float v = x[baseIdx+c] + residual[baseIdx+c]; snorm[c] = v; if (res_sum_out) res_sum_out[baseIdx+c] = v; lsum += v; }
    float mean = aidn_block_sum_link(lsum, sreduce) / (float)head_dim;
    float lvar = 0.0f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) { float d = snorm[c] - mean; lvar = fmaf(d, d, lvar); }
    float rstd = rsqrtf(aidn_block_sum_link(lvar, sreduce) / (float)head_dim + eps);
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) { snorm[c] = fmaf((snorm[c] - mean) * rstd, gamma[c], beta[c]); }
"u8)
            : GeneratedKernels.GeneratedKernelUtf8.Decode(@"
    const float* __restrict__ x, const float* __restrict__ residual,
    const float* __restrict__ gamma, const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    float* __restrict__ rope_out, float* __restrict__ res_sum_out,
    int outer_count, int seq_len, int head_dim, int trig_stride, float eps) {
    __shared__ float sreduce[32];
    extern __shared__ float snorm[];
    int row = blockIdx.x; int totalRows = outer_count * seq_len;
    if (row >= totalRows) return;
    int s = row % seq_len; int baseIdx = row * head_dim;
    float lsq = 0.0f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) { float v = x[baseIdx+c] + residual[baseIdx+c]; snorm[c] = v; if (res_sum_out) res_sum_out[baseIdx+c] = v; lsq = fmaf(v, v, lsq); }
    float invRms = rsqrtf(aidn_block_sum_link(lsq, sreduce) / (float)head_dim + eps);
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) { snorm[c] = snorm[c] * invRms * gamma[c]; }
"u8);
        string suffix = GeneratedKernels.GeneratedKernelUtf8.Decode(@"
    __syncthreads();
    int halfDim = head_dim >> 1;
    for (int i = threadIdx.x; i < halfDim; i += blockDim.x) {
        float cv = cos_tab[s * trig_stride + i]; float sv = sin_tab[s * trig_stride + i];
        float a = snorm[i]; float b = snorm[halfDim + i];
        rope_out[baseIdx + i] = a * cv - b * sv;
        rope_out[baseIdx + halfDim + i] = b * cv + a * sv;
    }
}
"u8);
        return string.Concat(prefix, "extern \"C\" __global__ void ", entryName, "(", body, suffix);
    }

    private static string EmitTiledFlashAttentionCudaSource(string kernelName, int headDim, bool isCausal)
    {
        string dStr = headDim.ToString(CultureInfo.InvariantCulture);
        string tpl = GeneratedKernels.GeneratedKernelUtf8.Decode(@"
extern ""C"" __global__ void __KERNEL__(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch_heads,
    int seq_len,
    float scale)
{
    const int HD = __HD__;
    const int BC = 16;
    __shared__ float sK[16 * __HD__];
    __shared__ float sV[16 * __HD__];
    int bh = blockIdx.y;
    int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= batch_heads) return;
    bool activeQ = qIdx < seq_len;
    float qReg[__HD__];
    float accO[__HD__];
    int bhOffset = bh * seq_len * HD;
    for (int d = 0; d < HD; d++) {
        qReg[d] = activeQ ? q[bhOffset + qIdx * HD + d] : 0.0f;
        accO[d] = 0.0f;
    }
    float rowMax = -3.402823466e+38f;
    float rowSum = 0.0f;
    for (int tileStart = 0; tileStart < seq_len; tileStart += BC) {
        for (int idx = threadIdx.x; idx < BC * HD; idx += blockDim.x) {
            int r = idx / HD; int d = idx % HD; int kvRow = tileStart + r;
            sK[idx] = (kvRow < seq_len) ? k[bhOffset + kvRow * HD + d] : 0.0f;
            sV[idx] = (kvRow < seq_len) ? v[bhOffset + kvRow * HD + d] : 0.0f;
        }
        __syncthreads();
        if (activeQ) {
            int tileEnd = (tileStart + BC < seq_len) ? (tileStart + BC) : seq_len;
            for (int kvRow = tileStart; kvRow < tileEnd; kvRow++) {
                if (__CAUSAL__) break;
                int r = kvRow - tileStart;
                float dot = 0.0f;
                for (int d = 0; d < HD; d++) dot = fmaf(qReg[d], sK[r * HD + d], dot);
                float score = dot * scale;
                float nextMax = fmaxf(rowMax, score);
                float rescale = expf(rowMax - nextMax);
                float p = expf(score - nextMax);
                rowSum = rowSum * rescale + p;
                for (int d = 0; d < HD; d++) accO[d] = accO[d] * rescale + p * sV[r * HD + d];
                rowMax = nextMax;
            }
        }
        __syncthreads();
    }
    if (activeQ) {
        float invSum = (rowSum > 0.0f) ? (1.0f / rowSum) : 0.0f;
        for (int d = 0; d < HD; d++) out[bhOffset + qIdx * HD + d] = accO[d] * invSum;
    }
}"u8);
        return tpl
            .Replace("__KERNEL__", kernelName)
            .Replace("__HD__", dStr)
            .Replace("__CAUSAL__", isCausal ? "kvRow > qIdx" : "false");
    }

    private static string EmitChunkedLinearScanCudaSource(string kernelName, int stateDim)
    {
        string dStr = stateDim.ToString(CultureInfo.InvariantCulture);
        string tpl = GeneratedKernels.GeneratedKernelUtf8.Decode(@"
extern ""C"" __global__ void __KERNEL__(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ decay,
    float* __restrict__ out,
    int batch_heads,
    int seq_len)
{
    const int D = __DIM__;
    int bh = blockIdx.x;
    int d = threadIdx.x;
    if (bh >= batch_heads || d >= D) return;
    float state = 0.0f;
    int bhBase = bh * seq_len * D;
    int decBase = bh * seq_len;
    for (int t = 0; t < seq_len; t++) {
        int idx = bhBase + t * D + d;
        float g = decay[decBase + t];
        state = fmaf(g, state, k[idx] * v[idx]);
        out[idx] = q[idx] * state;
    }
}"u8);
        return tpl.Replace("__KERNEL__", kernelName).Replace("__DIM__", dStr);
    }
    private static string FormatFloatLiteral(float value)
    {
        string s = value.ToString("R", CultureInfo.InvariantCulture);
        if (s.IndexOf('.') < 0 && s.IndexOf('e') < 0 && s.IndexOf('E') < 0)
            s += ".0";
        return s + "f";
    }
}
