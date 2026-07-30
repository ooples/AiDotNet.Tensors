using System;
using System.Security.Cryptography;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static class GpuRandomSeed
{
    // Stateless inverse-CDF sampling uses a fixed stream so CPU and every GPU backend select the
    // same stratified sample in a given output cell without sharing mutable RNG state.
    public const uint StatelessImportanceSampling = 0xA511E9B3u;

    public static ulong Create()
    {
        byte[] bytes = new byte[sizeof(ulong)];
        using (RandomNumberGenerator generator = RandomNumberGenerator.Create())
        {
            generator.GetBytes(bytes);
        }

        ulong seed = BitConverter.ToUInt64(bytes, 0);
        Array.Clear(bytes, 0, bytes.Length);
        return seed;
    }
}
