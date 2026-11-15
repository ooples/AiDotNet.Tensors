using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Shapes observed by eager recorders in one training trace. A guard owns a snapshot; probes only read it.
/// Entries contain metadata, never activation values, and are usable only for audited shape-stable operations.
/// </summary>
internal sealed class CompiledOpShapeCache
{
    private readonly Dictionary<int, Entry> _entries;

    internal CompiledOpShapeCache() => _entries = new();
    private CompiledOpShapeCache(Dictionary<int, Entry> entries) => _entries = new(entries);
    internal CompiledOpShapeCache Snapshot() => new(_entries);

    internal void Remember(int position, ObjectiveSignature key, int[] shape, int[]? auxiliaryShape = null)
    {
        if (key.Signable)
            _entries[position] = new Entry(key, (int[])shape.Clone(), auxiliaryShape is null ? null : (int[])auxiliaryShape.Clone());
    }

    internal Tensor<T> Lookup<T>(int position, ObjectiveSignature key, string opName, out int[]? auxiliaryShape)
    {
        if (!_entries.TryGetValue(position, out var entry) || !entry.Key.Equals(key))
            throw new ObjectiveProbeAbortedException(opName + " (no matching trace-time shape)");
        auxiliaryShape = entry.AuxiliaryShape is null ? null : (int[])entry.AuxiliaryShape.Clone();
        return Tensor<T>.CreateDeferred((int[])entry.Shape.Clone());
    }

    // Explicit opt-in: a new recorder is unsafe until its output shape is audited. In particular,
    // ScatterAdd/ScatterMax can infer size from index VALUES and must never use a shape-only cache.
    internal static bool Supports(string opName) => opName is
        "TensorMaskedFill" or "TensorWhere" or "TensorMax" or
        "FFTReal" or "FFTImag" or "IFFTReal" or "IFFTImag" or
        "FFT2DReal" or "FFT2DImag" or "IFFT2DReal" or "IFFT2DImag" or
        "STFTMagnitude" or "STFTPhase" or "ISTFT" or "MelSpectrogram" or
        "Spectrogram" or "StftPhase" or "AmplitudeToDB" or "ComputeDeltas" or "Resample" or
        "BoxIou" or "GeneralizedBoxIou" or "DistanceBoxIou" or "CompleteBoxIou" or
        "TensorAdaptiveAvgPool1D" or "TensorAdaptiveAvgPool3D" or
        "TensorAdaptiveMaxPool1D" or "TensorAdaptiveMaxPool3D" or "TensorLpPool" or "TensorMaxUnpool" or
        "MaxPool2DWithTensorIndices" or "MaxPool3DWithTensorIndices" or "ReduceMaxWithTensorIndices" or "TensorTopK" or
        "TensorFakeQuantize" or "TensorTopKSoftmax" or "TensorSinusoidalTimeEmbedding" or "TensorCrfForward" or
        "TensorMsort" or "TensorDiagflat" or "TensorDiagonalScatter" or "TensorGridSample3D" or
        "TensorQuantile" or "TensorNanQuantile" or "RoIAlign" or "RoIPool" or "PsRoIAlign" or "PsRoIPool" or
        "TensorDot" or "TensorCosineSimilarity" or "TensorPDist" or "TensorCDist" or "TensorKron" or "TensorInner" or
        "TensorCartesianProd" or "TensorLogCumSumExp" or "TensorClampTensor" or "TensorSelectScatter" or
        "TensorIndexPut" or "TensorBlockDiag" or "TensorSliceScatter" or "TensorIndexFill" or "TensorScatterReduce" or
        "TensorUnfold" or "TensorZeta" or "TensorMultiDot" or "TensorLdexp" or "TensorPut" or
        "TensorPolygamma" or "TensorDigamma" or "PadNd";

    private sealed class Entry
    {
        internal Entry(ObjectiveSignature key, int[] shape, int[]? auxiliaryShape)
        { Key = key; Shape = shape; AuxiliaryShape = auxiliaryShape; }
        internal ObjectiveSignature Key { get; }
        internal int[] Shape { get; }
        internal int[]? AuxiliaryShape { get; }
    }
}
