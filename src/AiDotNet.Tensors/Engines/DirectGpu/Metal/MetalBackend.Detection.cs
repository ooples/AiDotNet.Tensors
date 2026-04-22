// Copyright (c) AiDotNet. All rights reserved.
// Metal launcher shims for the vision detection kernels (Issue #217).
// Mirrors MetalBackend.Parity210.cs — pipeline resolved via shader library
// handle, dispatched with 1-D thread grid sized to total output cells.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IDetectionBackend
{
    private const string DetectionLibName = "Detection";

    private MetalPipelineState GetDetectionPipeline(string kernelName)
    {
        if (_detectionLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal Detection library was not compiled. Falling back to CPU reference.");
        return GetPipeline(DetectionLibName, _detectionLibrary, kernelName);
    }

    private void DispatchPairwiseIou(string kernelName,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int m)
    {
        if (n <= 0 || m <= 0) return;
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer aBuf || b is not MetalGpuBuffer bBuf
            || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        int total = n * m;
        var pipeline = GetDetectionPipeline(kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuf, 0);
        encoder.SetBuffer(bBuf, 1);
        encoder.SetBuffer(outBuf, 2);
        encoder.SetBytes(n, 3);
        encoder.SetBytes(m, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void BoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_box_iou", boxesA, boxesB, output, n, m);

    public void GeneralizedBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_generalized_box_iou", boxesA, boxesB, output, n, m);

    public void DistanceBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_distance_box_iou", boxesA, boxesB, output, n, m);

    public void CompleteBoxIou(IGpuBuffer boxesA, IGpuBuffer boxesB, IGpuBuffer output, int n, int m)
        => DispatchPairwiseIou("detection_complete_box_iou", boxesA, boxesB, output, n, m);

    public void BoxArea(IGpuBuffer boxes, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        ThrowIfDisposed();
        if (boxes is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetDetectionPipeline("detection_box_area");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(n, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void BoxConvert(IGpuBuffer boxes, IGpuBuffer output, int n, int fromFormat, int toFormat)
    {
        if (n <= 0) return;
        if ((uint)fromFormat > 2 || (uint)toFormat > 2)
            throw new ArgumentException(
                $"fromFormat/toFormat must be 0/1/2; got {fromFormat}, {toFormat}.");
        ThrowIfDisposed();
        if (boxes is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetDetectionPipeline("detection_box_convert");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(fromFormat, 3);
        encoder.SetBytes(toFormat, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void IouFamilyBackward(
        IGpuBuffer gradOutput, IGpuBuffer boxesA, IGpuBuffer boxesB,
        IGpuBuffer gradA, IGpuBuffer gradB,
        int n, int m, int variant)
    {
        // See CudaBackend.Detection for rationale — dispatch each side
        // independently when only one dim is zero.
        if (n <= 0 && m <= 0) return;
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer goBuf
            || boxesA is not MetalGpuBuffer aBuf
            || boxesB is not MetalGpuBuffer bBuf
            || gradA is not MetalGpuBuffer gABuf
            || gradB is not MetalGpuBuffer gBBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        // A-side: N threads.
        if (n > 0)
        {
            var pipeA = GetDetectionPipeline("detection_iou_backward_a");
            var (tgrA, tpgA) = pipeA.Calculate1DDispatch(n);
            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeA.Handle);
                encoder.SetBuffer(goBuf, 0);
                encoder.SetBuffer(aBuf, 1);
                encoder.SetBuffer(bBuf, 2);
                encoder.SetBuffer(gABuf, 3);
                encoder.SetBytes(n, 4);
                encoder.SetBytes(m, 5);
                encoder.SetBytes(variant, 6);
                encoder.DispatchThreadgroups(tgrA, tpgA);
            }
        }

        // B-side: M threads.
        if (m > 0)
        {
            var pipeB = GetDetectionPipeline("detection_iou_backward_b");
            var (tgrB, tpgB) = pipeB.Calculate1DDispatch(m);
            using (var encoder = _commandQueue.CreateScopedComputeEncoder())
            {
                encoder.SetPipelineState(pipeB.Handle);
                encoder.SetBuffer(goBuf, 0);
                encoder.SetBuffer(aBuf, 1);
                encoder.SetBuffer(bBuf, 2);
                encoder.SetBuffer(gBBuf, 3);
                encoder.SetBytes(n, 4);
                encoder.SetBytes(m, 5);
                encoder.SetBytes(variant, 6);
                encoder.DispatchThreadgroups(tgrB, tpgB);
            }
        }
    }
}
