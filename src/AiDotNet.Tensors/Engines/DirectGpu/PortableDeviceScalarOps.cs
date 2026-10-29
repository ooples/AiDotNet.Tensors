// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IDeviceScalarOps.MultiTensorSumOfSquares"/> for backends without the fused launch: the same double
/// result (two float slots), reduced on the host from each tensor's download. Correct everywhere; the fused flag
/// stays false so the optimizer's per-step paths keep their per-tensor device route on these backends.
/// </summary>
internal static class PortableDeviceScalarOps
{
    public static void MultiTensorSumOfSquares(IDirectGpuBackend backend, IReadOnlyList<IGpuBuffer> tensors,
        IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        if (sumOfSquares is null) throw new ArgumentNullException(nameof(sumOfSquares));
        if (sumOfSquares.Size < 2) throw new ArgumentException("The double result needs two float slots.", nameof(sumOfSquares));
        double total = 0;
        if (tensors is not null)
        {
            if (sizes is null || sizes.Count != tensors.Count)
                throw new ArgumentException("One size per tensor is required.", nameof(sizes));
            for (int t = 0; t < tensors.Count; t++)
            {
                float[] host = backend.DownloadBuffer(tensors[t]);
                int n = Math.Min(sizes[t], host.Length);
                for (int i = 0; i < n; i++) total += (double)host[i] * host[i];
            }
        }
        byte[] bits = BitConverter.GetBytes(total);
        backend.UploadBufferInPlace(new[] { BitConverter.ToSingle(bits, 0), BitConverter.ToSingle(bits, 4) }, sumOfSquares);
    }
}