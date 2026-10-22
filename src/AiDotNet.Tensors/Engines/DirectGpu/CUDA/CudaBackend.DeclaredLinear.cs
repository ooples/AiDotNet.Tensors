// Copyright (c) AiDotNet. All rights reserved.
// Linear layers on declared programs (#890 phase 4c): CudaBackend.FusedLinear{ReLU,Sigmoid,Tanh,GELU,Swish} and their
// backwards are served by programs lowered from their specs (CodegenBackendPrograms.FusedLinear / FusedLinearBackward)
// instead of the hand-written thread-per-output kernels, in those kernels' contract. They run after the direct-PTX
// tiled linear routes and before the hand-written kernels, which still serve any shape the declared path declines.
//
// The backward's activation derivative is the function derived from the hand-written elementwise backward kernel
// (relu_backward, gelu_backward, ...: CudaMathLibrary), the same arithmetic the fused kernels inline.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private static readonly string[] s_fusedLinearNames = { "x", "w", "bias", "y" };
    private static readonly string[] s_fusedLinearBackwardNames = { "dy", "x", "w", "saved", "dx", "dw", "dbias" };

    private Func<DeclaredKey, CodegenProgram>? _buildFusedLinearBackward;
    private Func<CodegenKernelSpec, PtxPassLowering?>? _lowerFusedLinearUnsplit;

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildFusedLinear =
        key => CodegenBackendPrograms.FusedLinear(key.D0, key.D1, key.D2, (CodegenActivationKind)key.D3);

    internal static DeclaredKey FusedLinearKey(int batch, int inFeatures, int outFeatures, CodegenActivationKind activation) =>
        new(DeclaredFamily.FusedLinear, batch, inFeatures, outFeatures, (int)activation, 0, 0);

    internal static DeclaredKey FusedLinearBackwardKey(int batch, int inFeatures, int outFeatures,
        CodegenActivationKind activation) =>
        new(DeclaredFamily.FusedLinearBackward, batch, inFeatures, outFeatures, (int)activation, 0, 0);

    // The hand-written elementwise kernel whose derived function is the activation's backward, and what it is called
    // with besides the gradient: the pre-activation (ReLU, GELU, Swish) or the output (Sigmoid, Tanh), as the fused
    // backward kernels take them.
    private static string FusedLinearDerivative(CodegenActivationKind activation) => activation switch
    {
        CodegenActivationKind.ReLU => "relu_backward",
        CodegenActivationKind.Sigmoid => "sigmoid_backward",
        CodegenActivationKind.Tanh => "tanh_backward",
        CodegenActivationKind.Gelu => "gelu_backward",
        CodegenActivationKind.Swish => "swish_backward",
        _ => throw new ArgumentOutOfRangeException(nameof(activation), activation, "No fused linear backward."),
    };

    /// <summary>A fused linear forward through its declared program; false when the shape declines.</summary>
    private bool TryDeclaredFusedLinear(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        if (batch <= 0 || inFeatures <= 0 || outFeatures <= 0) return false;
        var lower = _lowerFusedLinearUnsplit ??= LowerFusedLinearUnsplit;
        return TryRunDeclaredProgram(FusedLinearKey(batch, inFeatures, outFeatures, activation), s_buildFusedLinear,
            s_fusedLinearNames, new DeclaredBindings(input, weight, bias, output), lower);
    }

    // The forward as ONE kernel. The modelled lowering splits a reduction whose outputs alone leave the device short
    // of four blocks a multiprocessor, which costs a second launch: on sm_75 a 64x128 -> 96 layer took 21.8 us split
    // against the hand-written kernel's 10.9, while unsplit layers from 256x512 -> 512 up already ran 2.5-3x faster.
    private PtxPassLowering? LowerFusedLinearUnsplit(CodegenKernelSpec pass)
    {
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(pass, _ccMajor, _ccMinor);
        if (emitter.CalledFunctions.Count != 0) return null;   // a pass that links a library keeps the default path
        return new PtxPassLowering(new[]
        {
            new PtxPassKernel(ptx, pass.Name, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY),
        }, null, "unsplit");
    }

    /// <summary>A fused linear layer as a resident stage, in CudaBackend.FusedLinear*'s contract.</summary>
    internal DeclaredResidentStage FusedLinearStage(IGpuBuffer input, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer output,
        int batch, int inFeatures, int outFeatures, CodegenActivationKind activation) =>
        new(FusedLinearKey(batch, inFeatures, outFeatures, activation), s_buildFusedLinear, s_fusedLinearNames,
            new DeclaredBindings(input, weight, bias, output), _lowerFusedLinearUnsplit ??= LowerFusedLinearUnsplit);

    /// <summary>A fused linear backward as a resident stage, in CudaBackend.FusedLinear*Backward's contract.</summary>
    internal DeclaredResidentStage FusedLinearBackwardStage(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batch, int inFeatures, int outFeatures, CodegenActivationKind activation) =>
        new(FusedLinearBackwardKey(batch, inFeatures, outFeatures, activation),
            _buildFusedLinearBackward ??= BuildFusedLinearBackward, s_fusedLinearBackwardNames,
            new DeclaredBindings(gradOutput, input, weight, saved, gradInput, gradWeight, gradBias));

    /// <summary>A fused linear backward through its declared program; false when the shape declines.</summary>
    private bool TryDeclaredFusedLinearBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer saved, IGpuBuffer gradInput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        if (batch <= 0 || inFeatures <= 0 || outFeatures <= 0 || !DeclaredKernelsEnabled) return false;
        // A cached delegate: the method group would allocate one per call.
        var build = _buildFusedLinearBackward ??= BuildFusedLinearBackward;
        return TryRunDeclaredProgram(FusedLinearBackwardKey(batch, inFeatures, outFeatures, activation),
            build, s_fusedLinearBackwardNames,
            new DeclaredBindings(gradOutput, input, weight, saved, gradInput, gradWeight, gradBias),
            residentWhenSmall: true);
    }

    // Only reached when a key is first built. A derivative that is not derivable declines the key (the hand-written
    // kernels then serve it), the same as a shape the lowering cannot take.
    private CodegenProgram BuildFusedLinearBackward(DeclaredKey key)
    {
        string kernel = FusedLinearDerivative((CodegenActivationKind)key.D3);
        var derivative = ResolveElementwise(kernel)
            ?? throw new NotSupportedException(kernel + " has no derived function: " + ElementwiseRefusal(kernel));
        return CodegenBackendPrograms.FusedLinearBackward(key.D0, key.D1, key.D2, derivative.Handle);
    }
}
