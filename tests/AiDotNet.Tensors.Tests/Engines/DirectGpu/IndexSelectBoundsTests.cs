// Copyright (c) AiDotNet. All rights reserved.
// resident_index_select used an index without checking it, so a negative or past-the-axis index read another row of
// the source, or memory outside it. An index outside the source axis now yields zeros -- the contract #890's declared
// gather (CodegenBackendPrograms.IndexSelect) has, so the call's result does not depend on which kernel serves it.
// One arm per backend whose kernel source was changed: the bound is written three times (CUDA, Metal, OpenCL), so one
// arm proves only the backend it runs on, and OpenCL is the one that runs on a non-NVIDIA host.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class IndexSelectBoundsTests
{
    // The 7-argument entry point is on each concrete backend, not IDirectGpuBackend (whose IndexSelect takes a row
    // count), so the arms pass their own.
    private delegate void IndexSelectLaunch(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize);

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
        AssertZerosOutsideTheAxis(backend, backend.IndexSelect);
    }

    /// <summary>
    /// The same contract through the OpenCL kernel, which carries its own copy of the bound.
    /// </summary>
    [SkippableFact]
    public void OutOfRangeIndicesSelectZerosOnOpenCl()
    {
        var backend = new OpenClBackend();
        if (!backend.IsAvailable)
        {
            // Disposed before the skip: Skip throws, and the `using` below only takes ownership once this line is
            // past, so skipping with it still live would leak the context and command queue for the rest of the run.
            Exception? initializationException = backend.InitializationException;
            backend.Dispose();
            Skip.If(true, "OpenCL is unavailable: " + (initializationException?.GetType().Name ?? "no device"));
        }
        using (backend) AssertZerosOutsideTheAxis(backend, backend.IndexSelect);
    }

    private static void AssertZerosOutsideTheAxis(IDirectGpuBackend backend, IndexSelectLaunch indexSelect)
    {
        const int Outer = 3, SourceAxis = 5, Inner = 4;
        int[] indices = { 0, -1, 4, 5, 2, int.MinValue, int.MaxValue, 3 };
        var source = new float[Outer * SourceAxis * Inner];
        for (int i = 0; i < source.Length; i++) source[i] = 1 + i;
        var filled = new float[Outer * indices.Length * Inner];
        for (int i = 0; i < filled.Length; i++) filled[i] = -7f;
        using var bsource = backend.AllocateBuffer(source);
        using var bindices = backend.AllocateIntBuffer(indices);
        using var bout = backend.AllocateBuffer(filled);
        indexSelect(bsource, bindices, bout, Outer, SourceAxis, indices.Length, Inner);
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
