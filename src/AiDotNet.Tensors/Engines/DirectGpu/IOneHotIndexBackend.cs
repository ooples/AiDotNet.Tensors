namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>A one-hot encoding read straight from a native int32 index buffer (a resident TopK / ArgMax result).</summary>
public interface IOneHotIndexBackend
{
    /// <summary>
    /// Writes <paramref name="output"/> as [<paramref name="rows"/>, <paramref name="depth"/>]: every row zero except a 1
    /// at its index, in one launch with no host read. An index outside [0, depth) gives a zero row.
    /// </summary>
    void OneHotInt32(IGpuBuffer int32Indices, IGpuBuffer output, int rows, int depth);
}
