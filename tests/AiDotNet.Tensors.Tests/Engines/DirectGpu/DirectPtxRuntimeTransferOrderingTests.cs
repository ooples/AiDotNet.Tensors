using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxRuntimeTransferOrderingTests
{
    [SkippableFact]
    public unsafe void StandaloneHostTransfers_OrderTheNonBlockingLaunchStream()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        const int elements = 1 << 20;
        using var runtime = new DirectPtxRuntime();
        using var module = runtime.LoadModule(
            EmitCopyPtx(elements),
            allowExperimentalJitFallback: true);
        IntPtr function = module.GetFunction("aidotnet_transfer_order_copy", out _);
        using var input = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
        var expected = new float[elements];
        for (int index = 0; index < expected.Length; index++)
            expected[index] = index + 1;

        input.Upload<float>(expected);
        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        module.Launch(
            function, (uint)((elements + 255) / 256), 1, 1,
            256, 1, 1, 0, arguments);

        // Deliberately omit runtime.Synchronize(). The synchronous buffer APIs
        // must establish their own ordering edges to the non-blocking stream.
        var actual = new float[elements];
        output.Download<float>(actual);
        int mismatch = -1;
        for (int index = 0; index < actual.Length; index++)
        {
            if (actual[index] == expected[index]) continue;
            mismatch = index;
            break;
        }
        if (mismatch >= 0)
            Assert.Fail(
                $"transfer ordering failed at {mismatch}: expected {expected[mismatch]}, " +
                $"actual {actual[mismatch]}");
    }

    /// <summary>
    /// A standalone runtime retains the device's primary context, as every backend does, instead of creating a private
    /// one: a private context per runtime was destroyed on Dispose, and on the RTX 3090 runner that cuCtxDestroy stalled
    /// every other test thread's cuMemAllocHost and cublasCreate until the hang blame killed the host. Disposing the
    /// runtime must leave the shared context alive for everyone else holding it.
    /// </summary>
    [SkippableFact]
    public void StandaloneRuntime_SharesThePrimaryContext_AndLeavesItAliveOnDispose()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        Assert.Equal(CudaResult.Success, CuBlasNative.cuInit(0));
        Assert.Equal(CudaResult.Success, CuBlasNative.cuDevicePrimaryCtxRetain(out IntPtr primary, 0));
        try
        {
            var runtime = new DirectPtxRuntime();
            Assert.Equal(primary, runtime.Context);
            Assert.False(runtime.OwnsContext);
            runtime.Dispose();

            using var again = new DirectPtxRuntime();
            Assert.Equal(primary, again.Context);
            var expected = new[] { 1f, 2f, 3f, 4f };
            using var buffer = again.AllocateBytes((nuint)(expected.Length * sizeof(float)));
            buffer.Upload<float>(expected);
            var actual = new float[expected.Length];
            buffer.Download<float>(actual);
            Assert.Equal(expected, actual);
        }
        finally
        {
            CuBlasNative.cuDevicePrimaryCtxRelease(0);
        }
    }

    /// <summary>
    /// The primary context a standalone runtime retains outlives the runtime, so destroying a private context no longer
    /// frees whatever a caller forgot to dispose: the runtime's Dispose must free its own leftover buffers.
    /// </summary>
    [SkippableFact]
    public void StandaloneRuntime_Dispose_FreesUndisposedBuffers()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var runtime = new DirectPtxRuntime();
        var leftover = runtime.AllocateBytes((nuint)(1024 * sizeof(float)));
        Assert.NotEqual(IntPtr.Zero, leftover.Pointer);
        runtime.Dispose();
        Assert.Equal(IntPtr.Zero, leftover.Pointer);
        leftover.Dispose();   // already freed by the runtime: a no-op, not a double free
    }

    private static string EmitCopyPtx(int elements) =>
        ".version 7.1\n" +
        // This kernel uses only baseline instructions. A conservative virtual
        // target lets the driver JIT it for newer physical architectures too.
        ".target sm_50\n" +
        ".address_size 64\n\n" +
        ".visible .entry aidotnet_transfer_order_copy(\n" +
        "    .param .u64 input,\n" +
        "    .param .u64 output)\n" +
        "{\n" +
        "    .reg .pred %p<2>;\n" +
        "    .reg .b32 %r<4>;\n" +
        "    .reg .b64 %rd<6>;\n" +
        "    .reg .f32 %f<2>;\n" +
        "    ld.param.u64 %rd0, [input];\n" +
        "    ld.param.u64 %rd1, [output];\n" +
        "    mov.u32 %r0, %ctaid.x;\n" +
        "    mov.u32 %r1, %ntid.x;\n" +
        "    mov.u32 %r2, %tid.x;\n" +
        "    mad.lo.u32 %r3, %r0, %r1, %r2;\n" +
        "    setp.ge.u32 %p0, %r3, " + elements + ";\n" +
        "    @%p0 ret;\n" +
        "    mul.wide.u32 %rd2, %r3, 4;\n" +
        "    add.u64 %rd3, %rd0, %rd2;\n" +
        "    add.u64 %rd4, %rd1, %rd2;\n" +
        "    ld.global.f32 %f0, [%rd3];\n" +
        "    st.global.f32 [%rd4], %f0;\n" +
        "    ret;\n" +
        "}\n";
}
