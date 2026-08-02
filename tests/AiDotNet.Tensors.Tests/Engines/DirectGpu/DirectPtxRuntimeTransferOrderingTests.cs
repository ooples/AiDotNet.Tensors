using System;
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
