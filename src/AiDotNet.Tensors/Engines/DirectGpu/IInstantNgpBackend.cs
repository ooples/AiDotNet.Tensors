namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Native GPU primitives used by Instant-NGP operations whose integer hash arithmetic cannot be
/// represented by the generic floating-point tensor algebra.
/// </summary>
/// <remarks>
/// One call encodes one level of tiny-cuda-nn's hash grid. Each coordinate is clamped to [0, 0.999999] and mapped to
/// <c>pos = fma(scale, x, positionOffset)</c> in fp32; the cell is <c>floor(pos)</c> and the trilinear weights come from
/// <c>pos - floor(pos)</c>. A corner is indexed densely (<c>x + y*R + z*R*R</c>, R = <c>gridResolution</c>
/// in the methods below) when <c>R^3</c> is at most the table size, and through the coherent prime hash
/// <c>x ^ y*2654435761 ^ z*805459861</c> otherwise, in wrapping uint32 arithmetic and modulo the table size. The host
/// computes the level's scale, offset and resolution (see <see cref="HashGridAlignment"/>).
/// </remarks>
public interface IInstantNgpBackend
{
    /// <summary>Encodes one hash-grid level into its columns of a shared output buffer.</summary>
    /// <param name="positions">Positions [numPoints, 3].</param>
    /// <param name="hashTable">The level's table [tableSize, featuresPerLevel].</param>
    /// <param name="output">The encoding [numPoints, outputStride]; only this level's columns are written.</param>
    /// <param name="numPoints">Number of positions.</param>
    /// <param name="scale">The level's grid scale.</param>
    /// <param name="positionOffset">The offset added after scaling (0.5 for tiny-cuda-nn's grid).</param>
    /// <param name="gridResolution">The level's vertices per axis, which decides dense or hashed indexing.</param>
    /// <param name="tableSize">Entries in the level's table.</param>
    /// <param name="featuresPerLevel">Features per entry.</param>
    /// <param name="levelOffset">The level's first column in the output.</param>
    /// <param name="outputStride">Columns per output row.</param>
    void HashGridEncodeLevel(
        IGpuBuffer positions,
        IGpuBuffer hashTable,
        IGpuBuffer output,
        int numPoints,
        float scale,
        float positionOffset,
        int gridResolution,
        int tableSize,
        int featuresPerLevel,
        int levelOffset,
        int outputStride);

    /// <summary>
    /// Computes one level's table gradient. Each output cell is owned by one thread, avoiding
    /// nondeterministic floating-point atomics when hash collisions occur.
    /// </summary>
    /// <param name="positions">Positions [numPoints, 3].</param>
    /// <param name="outputGradient">Gradient of the encoding [numPoints, outputStride].</param>
    /// <param name="tableGradient">The level's table gradient [tableSize, featuresPerLevel], fully written.</param>
    /// <param name="numPoints">Number of positions.</param>
    /// <param name="scale">The level's grid scale.</param>
    /// <param name="positionOffset">The offset added after scaling (0.5 for tiny-cuda-nn's grid).</param>
    /// <param name="gridResolution">The level's vertices per axis, which decides dense or hashed indexing.</param>
    /// <param name="tableSize">Entries in the level's table.</param>
    /// <param name="featuresPerLevel">Features per entry.</param>
    /// <param name="levelOffset">The level's first column in the output gradient.</param>
    /// <param name="outputStride">Columns per output-gradient row.</param>
    void HashGridEncodeLevelBackward(
        IGpuBuffer positions,
        IGpuBuffer outputGradient,
        IGpuBuffer tableGradient,
        int numPoints,
        float scale,
        float positionOffset,
        int gridResolution,
        int tableSize,
        int featuresPerLevel,
        int levelOffset,
        int outputStride);
}
