// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Internal capability: widen bf16 device data to fp32 (exact: a bf16 value is the top 16 bits of the fp32 of the
    /// same value). The engine keeps a weight whose streaming store is bf16 on the device at 2 bytes per element and
    /// widens it into an fp32 temporary for each op that reads it.
    /// </summary>
    internal interface IBf16WidenBackend
    {
        /// <summary>True when the widening kernel is compiled on this device.</summary>
        bool CanWidenBf16 { get; }

        /// <summary>fp32[i] = bf16[i] for <paramref name="count"/> elements; <paramref name="bf16"/> holds packed 16-bit values.</summary>
        void WidenBf16ToFp32(IGpuBuffer bf16, IGpuBuffer fp32, int count);
    }
}
