namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Converts the numeric-float representation of resident index tensors to native int32.</summary>
public interface IResidentIndexBackend
{
    void ConvertIndicesToInt32(IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length);

    void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize);

    void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize);

    void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows);

    void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices);

    void ScatterAddRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows);

    void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows);

    void ScatterSoftmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups);

    void ScatterAddBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows);

    void ScatterMeanBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows);

    void ScatterMaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows);

    void ScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int numGroups);
}
