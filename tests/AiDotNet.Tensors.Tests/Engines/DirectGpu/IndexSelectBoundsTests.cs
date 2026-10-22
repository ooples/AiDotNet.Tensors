// Copyright (c) AiDotNet. All rights reserved.
// resident_index_select used an index without checking it, so a negative or past-the-axis index read another row of
// the source, or memory outside it. An index outside the source axis now yields zeros -- the contract #890's declared
// gather (CodegenBackendPrograms.IndexSelect) has, so the call's result does not depend on which kernel serves it.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class IndexSelectBoundsTests
{
    /// <summary>
    /// In-range indices copy their row; negative and past-the-axis ones give zeros. Output starts filled, so an element
    /// the kernel skipped cannot pass as a zero.
    /// </summary>
    [SkippableFact]
    public void OutOfRangeIndicesSelectZeros()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "The CUDA backend is unavailable (no NVRTC).");
        const int Outer = 3, SourceAxis = 5, Inner = 4;
        int[] indices = { 0, -1, 4, 5, 2, int.MinValue, int.MaxValue, 3 };
        var source = new float[Outer * SourceAxis * Inner];
        for (int i = 0; i < source.Length; i++) source[i] = 1 + i;
        var filled = new float[Outer * indices.Length * Inner];
        for (int i = 0; i < filled.Length; i++) filled[i] = -7f;
        using var bsource = backend.AllocateBuffer(source);
        using var bindices = backend.AllocateIntBuffer(indices);
        using var bout = backend.AllocateBuffer(filled);
        backend.IndexSelect(bsource, bindices, bout, Outer, SourceAxis, indices.Length, Inner);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(bout);
        for (int o = 0; o < Outer; o++)
            for (int j = 0; j < indices.Length; j++)
                for (int i = 0; i < Inner; i++)
                {
                    bool valid = indices[j] >= 0 && indices[j] < SourceAxis;
                    float expected = valid ? source[(o * SourceAxis + indices[j]) * Inner + i] : 0f;
                    Assert.Equal(expected, got[(o * indices.Length + j) * Inner + i]);
                }
    }
}
