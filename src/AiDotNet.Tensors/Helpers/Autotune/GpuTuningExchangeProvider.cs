using System;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Process-wide exchange selected once for production dispatch. Community
/// tuning remains inert unless telemetry was explicitly enabled before first
/// use; .NET Framework retains the complete local autotune path without the
/// net10-only HTTP/JSON client.
/// </summary>
internal static class GpuTuningExchangeProvider
{
    private static readonly Lazy<IGpuTuningExchange> Exchange = new(Create);

    internal static IGpuTuningExchange Current => Exchange.Value;

    private static IGpuTuningExchange Create()
    {
#if NET5_0_OR_GREATER
        return new Engines.DirectGpu.Telemetry.SupabaseGpuTuningExchange(enabled: true);
#else
        return NullGpuTuningExchange.Instance;
#endif
    }
}
