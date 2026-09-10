namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>An immutable, typed launch configuration for a square convolution tile.</summary>
public readonly record struct ConvTileConfiguration
{
    /// <summary>Creates a validated square-tile configuration.</summary>
    /// <param name="tileEdge">Positive width and height of the tile.</param>
    public ConvTileConfiguration(int tileEdge)
    {
        if (tileEdge <= 0) throw new ArgumentOutOfRangeException(nameof(tileEdge));
        TileEdge = tileEdge;
    }

    /// <summary>Gets the shared width and height of the square tile.</summary>
    public int TileEdge { get; }
}
