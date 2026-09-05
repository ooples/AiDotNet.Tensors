namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>Identifies the GPU vendor without requiring callers to branch on driver-provided strings.</summary>
public enum GpuVendorKind
{
    /// <summary>The vendor is absent, unknown, or not one of the explicitly supported families.</summary>
    Other = 0,

    /// <summary>NVIDIA CUDA-capable hardware.</summary>
    Nvidia = 1,

    /// <summary>AMD GPU hardware.</summary>
    Amd = 2,

    /// <summary>Intel GPU hardware.</summary>
    Intel = 3,

    /// <summary>Apple GPU hardware.</summary>
    Apple = 4
}
