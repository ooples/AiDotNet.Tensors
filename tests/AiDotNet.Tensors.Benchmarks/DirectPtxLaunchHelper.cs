// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Builds the driver API pointer-to-pointer argument vector for a PTX launch.</summary>
internal static class DirectPtxLaunchHelper
{
    internal static unsafe void Launch(
        DirectPtxModule module, IntPtr function, IntPtr[] pointers,
        uint blocks, uint threads, uint dynamicSharedMemoryBytes = 0)
    {
        if (module is null) throw new ArgumentNullException(nameof(module));
        if (pointers is null) throw new ArgumentNullException(nameof(pointers));
        fixed (IntPtr* pinned = pointers)
        {
            void** arguments = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) arguments[i] = pinned + i;
            module.Launch(function, blocks, 1, 1, threads, 1, 1,
                dynamicSharedMemoryBytes, arguments);
        }
    }
}
