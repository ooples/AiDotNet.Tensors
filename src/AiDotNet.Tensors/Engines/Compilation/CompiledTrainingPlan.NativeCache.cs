using System;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _compiledNativeCacheBestFit =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_NATIVE_CACHE_BEST_FIT") == "1";
}
