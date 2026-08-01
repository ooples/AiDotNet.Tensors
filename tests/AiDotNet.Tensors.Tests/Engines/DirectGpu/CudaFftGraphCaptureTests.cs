using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class CudaFftGraphCaptureTests
{
    [SkippableFact]
    public void FftFamily_CapturesAndReplaysOnBackendStream()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available.");
        Skip.IfNot(CudaNativeBindings.SupportsGraphCapture, "CUDA graph capture is unavailable.");

        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend failed to initialize.");

        const int n = 8;
        const int batch = 2;
        const int height = 4;
        const int width = 4;
        float[] signal = Enumerable.Range(0, n).Select(i => (float)(i + 1)).ToArray();
        float[] batchedSignal = Enumerable.Range(0, batch * n).Select(i => (float)(i + 1)).ToArray();
        float[] image = Enumerable.Range(0, height * width).Select(i => (float)(i + 1)).ToArray();

        using var fftInputReal = backend.AllocateBuffer(signal);
        using var fftInputImag = backend.AllocateBuffer(new float[n]);
        using var fftOutputReal = backend.AllocateBuffer(n);
        using var fftOutputImag = backend.AllocateBuffer(n);
        using var rfftOutputReal = backend.AllocateBuffer(n / 2 + 1);
        using var rfftOutputImag = backend.AllocateBuffer(n / 2 + 1);
        using var irfftOutput = backend.AllocateBuffer(n);
        using var batchedInputReal = backend.AllocateBuffer(batchedSignal);
        using var batchedInputImag = backend.AllocateBuffer(new float[batch * n]);
        using var batchedOutputReal = backend.AllocateBuffer(batch * n);
        using var batchedOutputImag = backend.AllocateBuffer(batch * n);
        using var fft2DInputReal = backend.AllocateBuffer(image);
        using var fft2DInputImag = backend.AllocateBuffer(new float[height * width]);
        using var fft2DOutputReal = backend.AllocateBuffer(height * width);
        using var fft2DOutputImag = backend.AllocateBuffer(height * width);

        void LaunchFftFamily()
        {
            backend.FFT(fftInputReal, fftInputImag, fftOutputReal, fftOutputImag, n, inverse: false);
            backend.RFFT(fftInputReal, rfftOutputReal, rfftOutputImag, n);
            backend.IRFFT(rfftOutputReal, rfftOutputImag, irfftOutput, n);
            backend.BatchedFFT(
                batchedInputReal, batchedInputImag, batchedOutputReal, batchedOutputImag,
                batch, n, inverse: false);
            backend.FFT2D(
                fft2DInputReal, fft2DInputImag, fft2DOutputReal, fft2DOutputImag,
                height, width, inverse: false);
        }

        LaunchFftFamily();
        backend.Synchronize();
        var eager = SnapshotOutputs();

        IntPtr graph = backend.CaptureGraph(LaunchFftFamily);
        Assert.NotEqual(IntPtr.Zero, graph);
        try
        {
            backend.LaunchCapturedGraph(graph);
            var replay = SnapshotOutputs();
            Assert.Equal(eager, replay);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }

        float[] SnapshotOutputs()
        {
            var result = new float[4 * n + 2 + 2 * batch * n + 2 * height * width];
            int offset = 0;
            Copy(fftOutputReal, n);
            Copy(fftOutputImag, n);
            Copy(rfftOutputReal, n / 2 + 1);
            Copy(rfftOutputImag, n / 2 + 1);
            Copy(irfftOutput, n);
            Copy(batchedOutputReal, batch * n);
            Copy(batchedOutputImag, batch * n);
            Copy(fft2DOutputReal, height * width);
            Copy(fft2DOutputImag, height * width);
            return result;

            void Copy(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer source, int length)
            {
                var slice = new float[length];
                backend.DownloadBuffer(source, slice);
                slice.CopyTo(result, offset);
                offset += length;
            }
        }
    }
}
