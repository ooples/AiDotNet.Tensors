namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Resident topology traversal for mesh spiral index generation.</summary>
public interface ISpiralIndicesBackend
{
    void GenerateSpiralIndices(
        IGpuBuffer vertices,
        IGpuBuffer faces,
        IGpuBuffer visited,
        IGpuBuffer currentRing,
        IGpuBuffer nextRing,
        IGpuBuffer output,
        int numVertices,
        int numFaces,
        int spiralLength);
}
