// Copyright (c) CrowdTrainer LLC. All rights reserved.
// cuDNN as candidates of the tuned convolution slots (#890 phase 4b). With cuDNN on (TensorCodecOptions.UseCudnn, the
// default where it loads), the forward used to call cuDNN's implicit-precomp-GEMM unconditionally -- whatever the
// shape, and bypassing every measured route -- and the backward never called cuDNN at all. Now each pass has a vendor
// slot: the established path is its reference (cuDNN's default algorithm for the forward, the backend's own routes
// for the gradients), and every cuDNN algorithm and every backend route competes under the slot's evidence gate, so a
// shape runs whichever measured fastest, as cudnn.benchmark does for PyTorch but against our kernels too.
//
// Precision: the descriptors use FMA math (strict fp32) unless the explicit reduced-precision opt-in
// (CudaDispatchPolicy.ConvReducedPrecisionTF32) allows tensor-core TF32, the policy the backend's own engine follows
// (the forward used to request tensor-op math, which is TF32 for fp32 data on Ampere, whatever the opt-in).
// Determinism: cuDNN's backward-data algorithm 0 and backward-filter algorithms 0 and 3 accumulate with atomics, so
// under GpuDeterminism the slot never considers them.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // A cuDNN algorithm needing more workspace than this is not offered (FFT algorithms can ask for gigabytes).
    private const ulong CudnnCandidateMaxWorkspace = 256UL << 20;

    private ITunedKernelCandidate<CudaConv2DArgs>[]? _conv2DForwardCandidateList;
    private ITunedKernelCandidate<TunedConv2DBackwardArgs>[]? _conv2DBackwardDataCandidateList, _conv2DBackwardFilterCandidateList;
    private TunedKernelSlot<CudaConv2DArgs>? _conv2DForwardVendorSlot;
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardDataVendorSlot, _conv2DBackwardFilterVendorSlot;

    private CuDnnNative.CudnnMathType CudnnConvMath =>
        CudaDispatchPolicy.ConvReducedPrecisionTF32 && ConvTensorCoresAvailable
            ? CuDnnNative.CudnnMathType.TensorOp
            : CuDnnNative.CudnnMathType.Fma;

    private static bool CudnnAlgorithmIsDeterministic(int pass, int algorithm) =>
        pass switch
        {
            1 => algorithm != (int)CuDnnNative.CudnnConvolutionBwdDataAlgo.Algo0,
            2 => algorithm != (int)CuDnnNative.CudnnConvolutionBwdFilterAlgo.Algo0 &&
                 algorithm != (int)CuDnnNative.CudnnConvolutionBwdFilterAlgo.Algo3,
            _ => true,
        };

    // Whether cuDNN supports this algorithm for the shape class within the workspace bound. -1 is the wrapper's own
    // choice, which always applies.
    private bool CudnnAdmits(int pass, int algorithm, TunedShape s)
    {
        if (algorithm < 0) return true;
        EnsureCudnnConv();
        if (_cudnnConv is not { } conv) return false;
        try
        {
            using var _ = PushContext();
            ulong? workspace = conv.AlgorithmWorkspace(pass, algorithm, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[13],
                s[14], s[9], s[10], s[7], s[8], s[11], s[12], CudnnConvMath);
            return workspace is { } bytes && bytes <= CudnnCandidateMaxWorkspace;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private void RunCudnnForward(in CudaConv2DArgs a, int algorithm)
    {
        using var _ = PushContext();
        EnsureCudnnConv();
        var conv = _cudnnConv ?? throw new InvalidOperationException("cuDNN convolution is unavailable.");
        var context = _cudnnContext ?? throw new InvalidOperationException("cuDNN is unavailable.");
        CuDnnContext.CheckStatus(CuDnnNative.cudnnSetStream(context.Handle, _stream), "cudnnSetStream");
        conv.Conv2DForwardGpu(a.Input.Handle, a.Kernel.Handle, a.Output.Handle, a.Batch, a.InChannels, a.InHeight,
            a.InWidth, a.OutChannels, a.KernelH, a.KernelW, a.OutHeight, a.OutWidth, a.PadH, a.PadW, a.StrideH,
            a.StrideW, a.DilationH, a.DilationW, algorithm: algorithm, mathType: CudnnConvMath);
        RunConvEpiloguePasses(a);
    }

    private void RunCudnnBackward(in TunedConv2DBackwardArgs a, int algorithm)
    {
        using var _ = PushContext();
        EnsureCudnnConv();
        var conv = _cudnnConv ?? throw new InvalidOperationException("cuDNN convolution is unavailable.");
        var context = _cudnnContext ?? throw new InvalidOperationException("cuDNN is unavailable.");
        CuDnnContext.CheckStatus(CuDnnNative.cudnnSetStream(context.Handle, _stream), "cudnnSetStream");
        // Weight gradient: First = input, Second = gradOutput. Data gradient: First = gradOutput, Second = kernel.
        if (a.WeightGradient)
            conv.Conv2DBackwardFilterGpu(a.First.Handle, a.Second.Handle, a.Output.Handle, a.Batch, a.InChannels,
                a.InHeight, a.InWidth, a.OutChannels, a.KernelH, a.KernelW, a.OutHeight, a.OutWidth, a.PadH, a.PadW,
                a.StrideH, a.StrideW, a.DilationH, a.DilationW, algorithm: algorithm, mathType: CudnnConvMath);
        else
            conv.Conv2DBackwardDataGpu(a.Second.Handle, a.First.Handle, a.Output.Handle, a.Batch, a.InChannels,
                a.InHeight, a.InWidth, a.OutChannels, a.KernelH, a.KernelW, a.OutHeight, a.OutWidth, a.PadH, a.PadW,
                a.StrideH, a.StrideW, a.DilationH, a.DilationW, algorithm: algorithm, mathType: CudnnConvMath);
    }

    /// <summary>The forward slot with cuDNN: cuDNN's default algorithm the reference, every route a candidate.</summary>
    internal TunedKernelSlot<CudaConv2DArgs> Conv2DForwardVendorSlot
    {
        get
        {
            if (_conv2DForwardVendorSlot is { } s) return s;
            _ = Conv2DForwardSlot;   // builds the backend's own candidate list
            lock (_tunedSlotsLock)
            {
                if (_conv2DForwardVendorSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<CudaConv2DArgs>>
                    {
                        new CudaTunedKernelCandidate<CudaConv2DArgs>("cudnn.conv2d.default", TunedKernelOrigin.Vendor, true,
                            _ => true, (in CudaConv2DArgs a) => RunCudnnForward(a, -1)),
                    };
                    candidates.AddRange(_conv2DForwardCandidateList ?? Array.Empty<ITunedKernelCandidate<CudaConv2DArgs>>());
                    for (int algorithm = 0; algorithm < (int)CuDnnNative.CudnnConvolutionFwdAlgo.Count; algorithm++)
                    {
                        int al = algorithm;
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DArgs>("cudnn.conv2d.algo" + al,
                            TunedKernelOrigin.Vendor, CudnnAlgorithmIsDeterministic(0, al), shape => CudnnAdmits(0, al, shape),
                            (in CudaConv2DArgs a) => RunCudnnForward(a, al)));
                    }
                    _conv2DForwardVendorSlot = new TunedKernelSlot<CudaConv2DArgs>(TunedKernelOp.Conv2DForward,
                        TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaConv2DArgs>(this, ConvolutionTolerance,
                            a => new[] { (a.Output, a.OutputCount) },
                            a => a.Output.Handle == a.Input.Handle || a.Output.Handle == a.Kernel.Handle),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_conv2DForwardVendorSlot);
            }
        }
    }

    /// <summary>A gradient slot with cuDNN: the backend's own reference first, then its routes, then cuDNN's.</summary>
    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardVendorSlot(bool weightGradient)
    {
        var existing = weightGradient ? _conv2DBackwardFilterVendorSlot : _conv2DBackwardDataVendorSlot;
        if (existing is { } s) return s;
        _ = weightGradient ? Conv2DBackwardFilterSlot : Conv2DBackwardDataSlot;   // builds the backend's own list
        lock (_tunedSlotsLock)
        {
            existing = weightGradient ? _conv2DBackwardFilterVendorSlot : _conv2DBackwardDataVendorSlot;
            if (existing is null)
            {
                int pass = weightGradient ? 2 : 1;
                string prefix = weightGradient ? "cudnn.conv2d_dw.algo" : "cudnn.conv2d_dx.algo";
                var candidates = new List<ITunedKernelCandidate<TunedConv2DBackwardArgs>>(
                    (weightGradient ? _conv2DBackwardFilterCandidateList : _conv2DBackwardDataCandidateList)
                    ?? Array.Empty<ITunedKernelCandidate<TunedConv2DBackwardArgs>>());
                int count = weightGradient ? (int)CuDnnNative.CudnnConvolutionBwdFilterAlgo.Count
                    : (int)CuDnnNative.CudnnConvolutionBwdDataAlgo.Count;
                for (int algorithm = 0; algorithm < count; algorithm++)
                {
                    int al = algorithm;
                    candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>(prefix + al, TunedKernelOrigin.Vendor,
                        CudnnAlgorithmIsDeterministic(pass, al), shape => CudnnAdmits(pass, al, shape),
                        (in TunedConv2DBackwardArgs a) => RunCudnnBackward(a, al)));
                }
                existing = new TunedKernelSlot<TunedConv2DBackwardArgs>(
                    weightGradient ? TunedKernelOp.Conv2DBackwardFilter : TunedKernelOp.Conv2DBackwardData,
                    TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, candidates.ToArray());
                if (weightGradient) _conv2DBackwardFilterVendorSlot = existing;
                else _conv2DBackwardDataVendorSlot = existing;
            }
            return Created(existing);
        }
    }
}
