// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;

namespace AiDotNet.Tensors.Engines.Compilation.FusionTileIr;

/// <summary>
/// Reverse-mode automatic differentiator over <see cref="FusionTileDescriptor"/> SSA instruction streams.
/// Auto-generates fused backward <see cref="FusionTileDescriptor"/>s (computing all input gradients in one linked
/// tile pass) directly from forward <see cref="FusionTileDescriptor"/>s without hand-writing backward kernels.
/// </summary>
internal static class FusionTileAdjointBuilder
{
    /// <summary>
    /// Derives a multi-output backward <see cref="FusionTileDescriptor"/> whose inputs are
    /// <c>[forwardInputs..., gradOutput_0..gradOutput_{M-1}]</c> and whose outputs are
    /// <c>[gradInput_0..gradInput_{N-1}]</c>.
    /// </summary>
    public static FusionTileDescriptor BuildBackwardTile(FusionTileDescriptor forward)
    {
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        if (forward.Kind != FusionTileKind.Elementwise && forward.Kind != FusionTileKind.GemmEpilogue)
            throw new NotSupportedException($"Automatic SSA tile differentiation supports Elementwise/GemmEpilogue tiles; got {forward.Kind}.");

        var fwdInsts = forward.Instructions;
        var bwd = new List<FusionTileInstruction>(fwdInsts.Count * 3);

        // Replay forward SSA values into indices [0 .. fwdCount - 1].
        var inputSlotToSsa = new int[forward.InputCount];
        for (int i = 0; i < inputSlotToSsa.Length; i++) inputSlotToSsa[i] = -1;

        var outputStores = new List<(int StoreIndex, int ValueSsa, int OutputPort)>();
        for (int i = 0; i < fwdInsts.Count; i++)
        {
            var inst = fwdInsts[i];
            if (inst.Op == FusionTileOpCode.StoreOutput)
            {
                outputStores.Add((i, inst.Src0, inst.Src1 >= 0 ? inst.Src1 : 0));
                // Emit a harmless Const(0) placeholder so SSA indices align 1:1 with fwdInsts indices!
                bwd.Add(new FusionTileInstruction(FusionTileOpCode.Const, immediate: 0.0f));
                continue;
            }

            if (inst.Op == FusionTileOpCode.LoadInput && inst.Src0 >= 0 && inst.Src0 < forward.InputCount)
            {
                inputSlotToSsa[inst.Src0] = i;
            }
            bwd.Add(inst);
        }

        short Emit(FusionTileOpCode op, short s0 = -1, short s1 = -1, short s2 = -1, float immediate = 0.0f)
        {
            short idx = checked((short)bwd.Count);
            bwd.Add(new FusionTileInstruction(op, s0, s1, s2, immediate));
            return idx;
        }

        short AddAdj(short existing, short delta)
        {
            if (existing < 0) return delta;
            return Emit(FusionTileOpCode.Add, existing, delta);
        }

        // Map each forward SSA slot to its accumulated adjoint SSA index (-1 = zero).
        var adjoint = new short[fwdInsts.Count];
        for (int i = 0; i < adjoint.Length; i++) adjoint[i] = -1;

        // Seed upstream output gradients as additional LoadInput ports [forward.InputCount + outPort].
        for (int o = 0; o < outputStores.Count; o++)
        {
            var (_, valSsa, outPort) = outputStores[o];
            short gOut = Emit(FusionTileOpCode.LoadInput, checked((short)(forward.InputCount + outPort)));
            adjoint[valSsa] = AddAdj(adjoint[valSsa], gOut);
        }

        short cZero = Emit(FusionTileOpCode.Const, immediate: 0.0f);
        short cOne = Emit(FusionTileOpCode.Const, immediate: 1.0f);
        short cHalf = Emit(FusionTileOpCode.Const, immediate: 0.5f);

        // Walk forward instructions in reverse topological order.
        for (int i = fwdInsts.Count - 1; i >= 0; i--)
        {
            short dy = adjoint[i];
            if (dy < 0) continue;

            var inst = fwdInsts[i];
            short y = (short)i;
            short a = inst.Src0;
            short b = inst.Src1;
            short c = inst.Src2;

            switch (inst.Op)
            {
                case FusionTileOpCode.LoadInput:
                case FusionTileOpCode.LoadScalar:
                case FusionTileOpCode.Const:
                case FusionTileOpCode.StoreOutput:
                    break;

                case FusionTileOpCode.Add:
                    adjoint[a] = AddAdj(adjoint[a], dy);
                    adjoint[b] = AddAdj(adjoint[b], dy);
                    break;

                case FusionTileOpCode.Sub:
                    adjoint[a] = AddAdj(adjoint[a], dy);
                    adjoint[b] = AddAdj(adjoint[b], Emit(FusionTileOpCode.Negate, dy));
                    break;

                case FusionTileOpCode.Mul:
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, b));
                    adjoint[b] = AddAdj(adjoint[b], Emit(FusionTileOpCode.Mul, dy, a));
                    break;

                case FusionTileOpCode.Div:
                {
                    // d(a / b)/da = dy / b; d(a / b)/db = -dy * (a / b) / b = -dy * y / b
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Div, dy, b));
                    short dyY = Emit(FusionTileOpCode.Mul, dy, y);
                    short db = Emit(FusionTileOpCode.Negate, Emit(FusionTileOpCode.Div, dyY, b));
                    adjoint[b] = AddAdj(adjoint[b], db);
                    break;
                }

                case FusionTileOpCode.Negate:
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Negate, dy));
                    break;

                case FusionTileOpCode.Fma:
                    // fma(a, b, c) = a * b + c
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, b));
                    adjoint[b] = AddAdj(adjoint[b], Emit(FusionTileOpCode.Mul, dy, a));
                    adjoint[c] = AddAdj(adjoint[c], dy);
                    break;

                case FusionTileOpCode.Relu:
                {
                    // Subgradient: x > 0 ? dy : 0, expressed branchlessly when y > 0: since y = max(0, a),
                    // step(a) via Relu sign approximation or exact scalar/CUDA ternary emitted by linker!
                    // Using y > 0: max(0, a) == a for a > 0; we compute indicator via step helper in evaluator/CUDA:
                    // Actually, let's compute exact indicator using Abs: (a > 0) => Relu(a) / max(a, eps)? No —
                    // Relu derivative at a = 0 is 0; let's handle Relu exact derivative in FusionTileLinker's emitter
                    // or decompose cleanly: wait — let's check! If a != 0, step(a) = 0.5 * (1 + a / abs(a))? At a == 0, 0/0!
                    // Instead, let's evaluate Relu grad cleanly: if y > 0 -> dy else 0. Notice:
                    // In IEEE float, for y = relu(a) >= 0: let's add a tiny helper or evaluate via Relu!
                    // Even better: let's support exact derivative for Sigmoid/Swish/Tanh/Gelu/Exp/Log/Sqrt/Rsqrt/Relu!
                    // For Relu: if a <= 0, y == 0. Let's emit Div(Mul(dy, y), Add(y, Const(1e-38f)))? Wait: at y = 1e-38, 0.5!
                    // No approximations! Let's add a clean Const-tagged Fma or keep exact step:
                    // Let's check: we own FusionTileOpCode! Adding ReluGrad = 24 to FusionTileOpCode makes Relu backward 100% exact!
                    short dRelu = Emit(FusionTileOpCode.Mul, dy, EmitReluMask(Emit, a, cZero, cOne));
                    adjoint[a] = AddAdj(adjoint[a], dRelu);
                    break;
                }

                case FusionTileOpCode.Sigmoid:
                {
                    // y = sigmoid(a) => dy * y * (1 - y)
                    short oneMinusY = Emit(FusionTileOpCode.Sub, cOne, y);
                    short yPrime = Emit(FusionTileOpCode.Mul, y, oneMinusY);
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, yPrime));
                    break;
                }

                case FusionTileOpCode.Swish:
                {
                    // swish(a) = a * sig(a); d/da = sig(a) * (1 + a * (1 - sig(a)))
                    short sig = Emit(FusionTileOpCode.Sigmoid, a);
                    short oneMinusSig = Emit(FusionTileOpCode.Sub, cOne, sig);
                    short inner = Emit(FusionTileOpCode.Fma, a, oneMinusSig, cOne);
                    short swishPrime = Emit(FusionTileOpCode.Mul, sig, inner);
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, swishPrime));
                    break;
                }

                case FusionTileOpCode.Tanh:
                {
                    // y = tanh(a) => dy * (1 - y * y)
                    short ySq = Emit(FusionTileOpCode.Mul, y, y);
                    short oneMinusYSq = Emit(FusionTileOpCode.Sub, cOne, ySq);
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, oneMinusYSq));
                    break;
                }

                case FusionTileOpCode.Gelu:
                {
                    // Exact derivative of tanh-GELU:
                    // u = k0 * (a + k1 * a^3), t = tanh(u), sech^2(u) = 1 - t^2, du/da = k0 * (1 + 3 * k1 * a^2)
                    // dGelu/da = 0.5 * (1 + t) + 0.5 * a * sech^2(u) * du/da
                    short cK0 = Emit(FusionTileOpCode.Const, immediate: 0.7978845608f);
                    short cK1 = Emit(FusionTileOpCode.Const, immediate: 0.044715f);
                    short c3K1 = Emit(FusionTileOpCode.Const, immediate: 0.134145f);
                    short a2 = Emit(FusionTileOpCode.Mul, a, a);
                    short a3 = Emit(FusionTileOpCode.Mul, a2, a);
                    short u = Emit(FusionTileOpCode.Mul, cK0, Emit(FusionTileOpCode.Fma, cK1, a3, a));
                    short t = Emit(FusionTileOpCode.Tanh, u);
                    short sech2 = Emit(FusionTileOpCode.Sub, cOne, Emit(FusionTileOpCode.Mul, t, t));
                    short du = Emit(FusionTileOpCode.Mul, cK0, Emit(FusionTileOpCode.Fma, c3K1, a2, cOne));
                    short leftTerm = Emit(FusionTileOpCode.Mul, cHalf, Emit(FusionTileOpCode.Add, cOne, t));
                    short rightTerm = Emit(FusionTileOpCode.Mul, Emit(FusionTileOpCode.Mul, cHalf, a), Emit(FusionTileOpCode.Mul, sech2, du));
                    short geluPrime = Emit(FusionTileOpCode.Add, leftTerm, rightTerm);
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, geluPrime));
                    break;
                }

                case FusionTileOpCode.Exp:
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, y));
                    break;

                case FusionTileOpCode.Log:
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Div, dy, a));
                    break;

                case FusionTileOpCode.Sqrt:
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Div, Emit(FusionTileOpCode.Mul, dy, cHalf), y));
                    break;

                case FusionTileOpCode.Rsqrt:
                {
                    // y = a^(-1/2) => dy * (-0.5) * y^3
                    short cNegHalf = Emit(FusionTileOpCode.Const, immediate: -0.5f);
                    short y3 = Emit(FusionTileOpCode.Mul, Emit(FusionTileOpCode.Mul, y, y), y);
                    adjoint[a] = AddAdj(adjoint[a], Emit(FusionTileOpCode.Mul, dy, Emit(FusionTileOpCode.Mul, cNegHalf, y3)));
                    break;
                }

                default:
                    throw new NotSupportedException($"Opcode {inst.Op} does not have an auto-differentiation rule yet.");
            }
        }

        // Emit StoreOutput for each forward input port in order [0 .. forward.InputCount - 1].
        for (int p = 0; p < forward.InputCount; p++)
        {
            int ssa = inputSlotToSsa[p];
            short gradSsa = ssa >= 0 && adjoint[ssa] >= 0 ? adjoint[ssa] : cZero;
            Emit(FusionTileOpCode.StoreOutput, gradSsa, checked((short)p));
        }

        int totalBwdInputs = forward.InputCount + forward.OutputCount;
        string bwdName = forward.TileName + "_autograd_bwd";
        string cudaKernel = FusionTileLinker.EmitCudaKernelFromInstructions(
            bwdName, totalBwdInputs, forward.ScalarCount, forward.InputCount, bwd);
        var cpuExec = FusionTileLinker.CreateCpuExecutor(totalBwdInputs, forward.ScalarCount, forward.InputCount, bwd);

        return new FusionTileDescriptor(
            bwdName,
            FusionTileKind.Elementwise,
            totalBwdInputs,
            forward.ScalarCount,
            forward.InputCount,
            bwd,
            cudaKernel,
            cpuExec);
    }

    private static short EmitReluMask(
        Func<FusionTileOpCode, short, short, short, float, short> emit,
        short a,
        short cZero,
        short cOne)
    {
        // Exact indicator for normal IEEE floats via Clamp/Step: max(0, min(1, a * +Inf)) would NaN at a == 0!
        // Instead, compute pos = relu(a); for pos > 0, pos / pos = 1; to avoid 0/0 at pos == 0:
        // Add tiny positive subnormal/min and subtract: let's add ReluStep opcode to FusionTileOpCode for 100% exact IEEE semantics!
        _ = cZero;
        _ = cOne;
        return emit(FusionTileOpCode.ReluStep, a, -1, -1, 0.0f);
    }
}
