#if NET5_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxRgLruRequest(
    int ComputeCapabilityMajor,
    int ComputeCapabilityMinor,
    DirectPtxPhysicalType PhysicalType,
    DirectPtxPhysicalLayout Layout,
    int Batch,
    int SequenceLength,
    int RecurrentDimension,
    bool IsTraining);

internal readonly record struct DirectPtxRgLruBufferRequest(
    nuint ValuePointer, long ValueBytes,
    nuint RecurrenceGatePointer, long RecurrenceGateBytes,
    nuint InputGatePointer, long InputGateBytes,
    nuint DecayPointer, long DecayBytes,
    nuint OutputPointer, long OutputBytes);

/// <summary>Fail-closed semantic admission for the first recurrent PTX slice.</summary>
internal static class DirectPtxRecurrentEligibility
{
    internal static DirectPtxEligibilityResult Evaluate(in DirectPtxRgLruRequest request)
    {
        if (!DirectPtxArchitecture.HasExperimentalRgLruScan(
            request.ComputeCapabilityMajor, request.ComputeCapabilityMinor))
            return DirectPtxEligibilityResult.Reject("rglru-sm-version-not-implemented");
        if (request.PhysicalType != DirectPtxPhysicalType.Float32)
            return DirectPtxEligibilityResult.Reject("rglru-dtype-not-fp32");
        if (request.Layout != DirectPtxPhysicalLayout.BatchSequenceFeature)
            return DirectPtxEligibilityResult.Reject("rglru-layout-not-dense-bsf");
        if (request.Batch != PtxFusedRgLruScan128x256Kernel.Batch)
            return DirectPtxEligibilityResult.Reject("rglru-batch-not-1");
        if (request.SequenceLength != PtxFusedRgLruScan128x256Kernel.SequenceLength)
            return DirectPtxEligibilityResult.Reject("rglru-sequence-not-128");
        if (request.RecurrentDimension != PtxFusedRgLruScan128x256Kernel.RecurrentDimension)
            return DirectPtxEligibilityResult.Reject("rglru-dimension-not-256");
        if (request.IsTraining)
            return DirectPtxEligibilityResult.Reject("rglru-backward-not-implemented");
        return DirectPtxEligibilityResult.Eligible;
    }

    internal static DirectPtxEligibilityResult EvaluateBuffers(
        in DirectPtxRgLruBufferRequest request)
    {
        const long sequenceBytes =
            (long)PtxFusedRgLruScan128x256Kernel.Batch *
            PtxFusedRgLruScan128x256Kernel.SequenceLength *
            PtxFusedRgLruScan128x256Kernel.RecurrentDimension * sizeof(float);
        const long decayBytes =
            (long)PtxFusedRgLruScan128x256Kernel.RecurrentDimension * sizeof(float);

        if (request.ValueBytes != sequenceBytes ||
            request.RecurrenceGateBytes != sequenceBytes ||
            request.InputGateBytes != sequenceBytes ||
            request.DecayBytes != decayBytes ||
            request.OutputBytes != sequenceBytes)
            return DirectPtxEligibilityResult.Reject("rglru-physical-extent-mismatch");
        if (request.ValuePointer == 0 || request.RecurrenceGatePointer == 0 ||
            request.InputGatePointer == 0 || request.DecayPointer == 0 ||
            request.OutputPointer == 0)
            return DirectPtxEligibilityResult.Reject("rglru-invalid-device-pointer");
        if (((request.ValuePointer | request.RecurrenceGatePointer |
              request.InputGatePointer | request.DecayPointer |
              request.OutputPointer) & 127u) != 0)
            return DirectPtxEligibilityResult.Reject("rglru-alignment-mismatch");

        try
        {
            var value = new Range(request.ValuePointer, request.ValueBytes);
            var recurrenceGate = new Range(request.RecurrenceGatePointer, request.RecurrenceGateBytes);
            var inputGate = new Range(request.InputGatePointer, request.InputGateBytes);
            var decay = new Range(request.DecayPointer, request.DecayBytes);
            var output = new Range(request.OutputPointer, request.OutputBytes);
            if (value.Overlaps(recurrenceGate) || value.Overlaps(inputGate) ||
                value.Overlaps(decay) || value.Overlaps(output) ||
                recurrenceGate.Overlaps(inputGate) || recurrenceGate.Overlaps(decay) ||
                recurrenceGate.Overlaps(output) || inputGate.Overlaps(decay) ||
                inputGate.Overlaps(output) || decay.Overlaps(output))
                return DirectPtxEligibilityResult.Reject("rglru-alias-not-supported");
        }
        catch (OverflowException)
        {
            return DirectPtxEligibilityResult.Reject("rglru-address-range-overflow");
        }
        return DirectPtxEligibilityResult.Eligible;
    }

    private readonly record struct Range(nuint Start, nuint End)
    {
        internal Range(nuint start, long byteLength)
            : this(start, checked(start + (nuint)byteLength)) { }

        internal bool Overlaps(Range other) => Start < other.End && other.Start < End;
    }
}
#endif
