#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class OpenClPrecisionBufferOwnershipTests : IClassFixture<OpenClBackendTestFixture>
{
    public enum BufferWrapper { Float, Byte }
    public enum Operand { Left, Right, Output }
    private readonly OpenClBackendTestFixture _fixture;

    public OpenClPrecisionBufferOwnershipTests(OpenClBackendTestFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData(BufferWrapper.Float, Operand.Left)]
    [InlineData(BufferWrapper.Float, Operand.Right)]
    [InlineData(BufferWrapper.Float, Operand.Output)]
    [InlineData(BufferWrapper.Byte, Operand.Left)]
    [InlineData(BufferWrapper.Byte, Operand.Right)]
    [InlineData(BufferWrapper.Byte, Operand.Output)]
    public void HalfGemm_RejectsForeignContextBeforeDispatch(BufferWrapper wrapper, Operand operand)
    {
        OpenClBackend backend = RequireBackend();
        using var foreign = new OpenClBackend();
        Assert.True(foreign.IsAvailable, "The second OpenCL context failed to initialize.");
        using IGpuBuffer left = Allocate(operand == Operand.Left ? foreign : backend, wrapper);
        using IGpuBuffer right = Allocate(operand == Operand.Right ? foreign : backend, wrapper);
        using IGpuBuffer output = Allocate(operand == Operand.Output ? foreign : backend, wrapper);

        ArgumentException error = Assert.Throws<ArgumentException>(
            () => backend.GemmFp16In32fOut(left, right, output, 1, 1, 1));

        Assert.Contains("context", error.Message);
        Assert.NotEqual(IntPtr.Zero, left.Handle);
        Assert.NotEqual(IntPtr.Zero, right.Handle);
        Assert.NotEqual(IntPtr.Zero, output.Handle);
    }

    [SkippableTheory]
    [InlineData(BufferWrapper.Float)]
    [InlineData(BufferWrapper.Byte)]
    public void HalfGemm_AcceptsBothWrappersFromItsOwnContext(BufferWrapper wrapper)
    {
        OpenClBackend backend = RequireBackend();
        using IGpuBuffer left = Allocate(backend, wrapper);
        using IGpuBuffer right = Allocate(backend, wrapper);
        using IGpuBuffer output = backend.AllocateBuffer(new[] { -1f });

        backend.GemmFp16In32fOut(left, right, output, 1, 1, 1);
        backend.Synchronize();

        Assert.Equal(new[] { 0f }, backend.DownloadBuffer(output));
    }

    private OpenClBackend RequireBackend()
    {
        bool ready = _fixture.IsAvailable && _fixture.Backend?.SupportsHgemm == true;
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(ready, "OpenCL FP16 GEMM was required but is unavailable.");
        Skip.IfNot(ready, "OpenCL FP16 GEMM is unavailable.");
        return _fixture.Backend ?? throw new InvalidOperationException("The OpenCL backend is unavailable.");
    }

    private static IGpuBuffer Allocate(OpenClBackend backend, BufferWrapper wrapper)
    {
        if (wrapper == BufferWrapper.Float) return backend.AllocateBuffer(new[] { 0f });
        IGpuBuffer buffer = backend.AllocateByteBuffer(sizeof(float));
        try
        {
            backend.UploadByteBuffer(buffer, new byte[sizeof(float)]);
            return buffer;
        }
        catch { buffer.Dispose(); throw; }
    }
}

#endif
