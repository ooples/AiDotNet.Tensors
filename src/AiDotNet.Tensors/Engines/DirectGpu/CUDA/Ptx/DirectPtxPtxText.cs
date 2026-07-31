namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Canonical text encodings shared by direct PTX emitters.</summary>
internal static class DirectPtxPtxText
{
    internal static string Hex(float value)
        => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");
}
