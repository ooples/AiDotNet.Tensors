using System;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    internal static bool CompiledGradientBankIdentityEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_BANK_IDENTITY") == "1";
}
