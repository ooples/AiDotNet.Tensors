// Parity tests for the cuDNN BatchNorm dispatch (issue #1159 / PR #726). CudaBackend.BatchNorm now
// routes to cudnnBatchNormalizationForwardTraining|Inference when UseCudnnForBatchNorm is on
// (default), and falls through to the hand-written batchnorm_forward kernel otherwise. These tests
// run BOTH paths on the live CUDA backend from identical inputs and check each against a CPU
// double-precision reference — so the cuDNN wiring is verified correct, not merely equal to the
// hand-written kernel.
//
// One documented convention difference: cuDNN applies Bessel's correction (unbiased, /(N-1)) to the
// RUNNING variance it accumulates, while the hand-written kernel stores the biased (/N) batch
// variance. cuDNN's unbiased running-var is the paper-faithful choice (Ioffe & Szegedy 2015 §3.1 use
// the unbiased estimate for inference). The current-batch NORMALIZATION uses the biased variance in
// both paths, so the forward OUTPUT, saveMean, saveInvVar and runningMean are identical; only the
// accumulated runningVar differs by the N/(N-1) factor. Each path is asserted against its own
// convention. Runs only when a CUDA GPU is present.

#if !NETFRAMEWORK
#nullable disable

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CudaBatchNormCudnnParityTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _gpuReady;

    public CudaBatchNormCudnnParityTests()
    {
        try { _gpu = new DirectGpuTensorEngine(); _gpuReady = _gpu.IsGpuAvailable; }
        catch { _gpuReady = false; }
    }

    public void Dispose() => _gpu?.Dispose();

    private CudaBackend GetCuda()
    {
        if (!_gpuReady) return null;
        return _gpu.GetBackend() as CudaBackend;   // null on non-CUDA GPUs → test no-ops
    }

    private const float Eps = 1e-5f;   // == CUDNN_BN_MIN_EPSILON, so both paths use the same value
    private const float Momentum = 0.1f;

    // Result of one BatchNorm forward: everything the op writes.
    private struct BnResult
    {
        public float[] Output;
        public float[] SaveMean;      // batch mean (training only)
        public float[] SaveInvVar;    // 1/sqrt(biasedVar+eps) (training only)
        public float[] RunningMean;   // updated in place (training) / unchanged (inference)
        public float[] RunningVar;
    }

    // Runs CudaBackend.BatchNorm once with the given cuDNN gate, from fresh copies of every input,
    // and downloads all outputs. runningMean/runningVar are uploaded from runInMean/runInVar and read
    // back after the (in-place) update.
    private static BnResult RunBn(
        CudaBackend cb, bool useCudnn, bool training,
        float[] input, float[] gamma, float[] beta,
        float[] runInMean, float[] runInVar,
        int batch, int channels, int spatialSize)
    {
        var prev = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { UseCudnnBatchNorm = useCudnn });
        try
        {
            int n = batch * channels * spatialSize;
            var inBuf = cb.AllocateBuffer(input);
            var outBuf = cb.AllocateBuffer(n);
            var gammaBuf = cb.AllocateBuffer(gamma);
            var betaBuf = cb.AllocateBuffer(beta);
            var rmBuf = cb.AllocateBuffer((float[])runInMean.Clone());
            var rvBuf = cb.AllocateBuffer((float[])runInVar.Clone());
            var saveMeanBuf = cb.AllocateBuffer(channels);
            var saveInvVarBuf = cb.AllocateBuffer(channels);
            try
            {
                cb.BatchNorm(inBuf, outBuf, gammaBuf, betaBuf, rmBuf, rvBuf, saveMeanBuf, saveInvVarBuf,
                    batch, channels, spatialSize, Eps, Momentum, training);

                var r = new BnResult
                {
                    Output = new float[n],
                    SaveMean = new float[channels],
                    SaveInvVar = new float[channels],
                    RunningMean = new float[channels],
                    RunningVar = new float[channels],
                };
                cb.DownloadBuffer(outBuf, r.Output);
                cb.DownloadBuffer(saveMeanBuf, r.SaveMean);
                cb.DownloadBuffer(saveInvVarBuf, r.SaveInvVar);
                cb.DownloadBuffer(rmBuf, r.RunningMean);
                cb.DownloadBuffer(rvBuf, r.RunningVar);
                return r;
            }
            finally
            {
                (inBuf as IDisposable)?.Dispose();
                (outBuf as IDisposable)?.Dispose();
                (gammaBuf as IDisposable)?.Dispose();
                (betaBuf as IDisposable)?.Dispose();
                (rmBuf as IDisposable)?.Dispose();
                (rvBuf as IDisposable)?.Dispose();
                (saveMeanBuf as IDisposable)?.Dispose();
                (saveInvVarBuf as IDisposable)?.Dispose();
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prev);
        }
    }

    private static float[] Rand(int seed, int n, double mag)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * mag);
        return a;
    }

    private static void AssertClose(float[] actual, double[] expected, double tol, string what)
    {
        double maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(actual[i] - expected[i]));
        Assert.True(maxErr < tol, $"{what}: max_abs_err {maxErr:E3} exceeded {tol:E1}");
    }

    [SkippableFact]
    public void CudnnBatchNorm_Training_MatchesCpuReference_AndHandWrittenPath()
    {
        var cb = GetCuda();
        Skip.If(cb is null, "CUDA backend not available");
        Skip.If(!CuDnnContext.IsAvailable, "cuDNN not available on this host");

        const int batch = 4, channels = 3, spatialSize = 5;   // N=batch*spatial=20 → Bessel 20/19
        int n = batch * channels * spatialSize;
        var input = Rand(1, n, 2.0);
        var gamma = Rand(2, channels, 1.0);
        var beta = Rand(3, channels, 0.5);
        var runInMean = Rand(4, channels, 0.3);
        var runInVar = new float[channels];
        for (int c = 0; c < channels; c++) runInVar[c] = 0.5f + 0.5f * c;   // positive

        // --- CPU double-precision reference, per channel ---
        int N = batch * spatialSize;
        var mean = new double[channels];
        var biasedVar = new double[channels];
        var unbiasedVar = new double[channels];
        var expOut = new double[n];
        var expSaveMean = new double[channels];
        var expSaveInvVar = new double[channels];
        var expRunMean = new double[channels];
        var expRunVarBiased = new double[channels];
        var expRunVarUnbiased = new double[channels];
        for (int c = 0; c < channels; c++)
        {
            double sum = 0;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatialSize; s++)
                    sum += input[(b * channels + c) * spatialSize + s];
            mean[c] = sum / N;
            double vs = 0;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatialSize; s++)
                {
                    double d = input[(b * channels + c) * spatialSize + s] - mean[c];
                    vs += d * d;
                }
            biasedVar[c] = vs / N;
            unbiasedVar[c] = vs / (N - 1);
            double invStd = 1.0 / Math.Sqrt(biasedVar[c] + Eps);
            expSaveMean[c] = mean[c];
            expSaveInvVar[c] = invStd;
            expRunMean[c] = (1.0 - Momentum) * runInMean[c] + Momentum * mean[c];
            expRunVarBiased[c] = (1.0 - Momentum) * runInVar[c] + Momentum * biasedVar[c];
            expRunVarUnbiased[c] = (1.0 - Momentum) * runInVar[c] + Momentum * unbiasedVar[c];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = (b * channels + c) * spatialSize + s;
                    expOut[idx] = gamma[c] * (input[idx] - mean[c]) * invStd + beta[c];
                }
        }

        var hand = RunBn(cb, useCudnn: false, training: true, input, gamma, beta, runInMean, runInVar, batch, channels, spatialSize);
        var cudnn = RunBn(cb, useCudnn: true, training: true, input, gamma, beta, runInMean, runInVar, batch, channels, spatialSize);

        // Both paths: forward output, saveMean, saveInvVar, runningMean identical to reference.
        AssertClose(hand.Output, expOut, 1e-4, "hand output");
        AssertClose(cudnn.Output, expOut, 1e-4, "cuDNN output");
        AssertClose(hand.SaveMean, expSaveMean, 1e-4, "hand saveMean");
        AssertClose(cudnn.SaveMean, expSaveMean, 1e-4, "cuDNN saveMean");
        AssertClose(hand.SaveInvVar, expSaveInvVar, 1e-3, "hand saveInvVar");
        AssertClose(cudnn.SaveInvVar, expSaveInvVar, 1e-3, "cuDNN saveInvVar");
        AssertClose(hand.RunningMean, expRunMean, 1e-4, "hand runningMean");
        AssertClose(cudnn.RunningMean, expRunMean, 1e-4, "cuDNN runningMean");

        // Running variance: hand-written = biased; cuDNN = unbiased (Bessel). Each matches its own
        // convention. The two differ measurably (N/(N-1) = 20/19), which we also assert to lock the
        // documented behaviour in place.
        AssertClose(hand.RunningVar, expRunVarBiased, 2e-3, "hand runningVar (biased)");
        AssertClose(cudnn.RunningVar, expRunVarUnbiased, 2e-3, "cuDNN runningVar (unbiased/Bessel)");
        double maxDelta = 0;
        for (int c = 0; c < channels; c++)
            maxDelta = Math.Max(maxDelta, Math.Abs(cudnn.RunningVar[c] - hand.RunningVar[c]));
        Assert.True(maxDelta > 1e-4,
            $"expected a measurable biased-vs-unbiased runningVar difference, got {maxDelta:E3}");
    }

    [SkippableFact]
    public void CudnnBatchNorm_Inference_MatchesCpuReference_AndHandWrittenPath()
    {
        var cb = GetCuda();
        Skip.If(cb is null, "CUDA backend not available");
        Skip.If(!CuDnnContext.IsAvailable, "cuDNN not available on this host");

        const int batch = 2, channels = 4, spatialSize = 6;
        int n = batch * channels * spatialSize;
        var input = Rand(11, n, 1.5);
        var gamma = Rand(12, channels, 1.0);
        var beta = Rand(13, channels, 0.5);
        var runMean = Rand(14, channels, 0.4);
        var runVar = new float[channels];
        for (int c = 0; c < channels; c++) runVar[c] = 0.7f + 0.3f * c;

        // CPU reference: normalize with the provided running stats (no stat update in inference).
        var expOut = new double[n];
        for (int c = 0; c < channels; c++)
        {
            double invStd = 1.0 / Math.Sqrt(runVar[c] + Eps);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = (b * channels + c) * spatialSize + s;
                    expOut[idx] = gamma[c] * (input[idx] - runMean[c]) * invStd + beta[c];
                }
        }

        var hand = RunBn(cb, useCudnn: false, training: false, input, gamma, beta, runMean, runVar, batch, channels, spatialSize);
        var cudnn = RunBn(cb, useCudnn: true, training: false, input, gamma, beta, runMean, runVar, batch, channels, spatialSize);

        AssertClose(hand.Output, expOut, 1e-4, "hand inference output");
        AssertClose(cudnn.Output, expOut, 1e-4, "cuDNN inference output");
        // Inference must not mutate the running stats on either path.
        AssertClose(hand.RunningMean, Array.ConvertAll(runMean, x => (double)x), 1e-6, "hand runningMean unchanged");
        AssertClose(cudnn.RunningMean, Array.ConvertAll(runMean, x => (double)x), 1e-6, "cuDNN runningMean unchanged");
    }

    [SkippableFact]
    public void CudnnBatchNorm_DispatchTelemetry_RecordsVendorPath()
    {
        var cb = GetCuda();
        Skip.If(cb is null, "CUDA backend not available");
        Skip.If(!CuDnnContext.IsAvailable, "cuDNN not available on this host");

        var profiler = PerformanceProfiler.Instance;
        bool prevEnabled = profiler.Enabled;
        var prevOpts = TensorCodecOptions.Current;
        profiler.Enabled = true;
        profiler.Clear();
        try
        {
            const int batch = 2, channels = 3, spatialSize = 4;
            var input = Rand(21, batch * channels * spatialSize, 1.0);
            var gamma = Rand(22, channels, 1.0);
            var beta = new float[channels];
            var rm = new float[channels];
            var rv = new float[channels];
            for (int c = 0; c < channels; c++) rv[c] = 1f;

            RunBn(cb, useCudnn: true, training: true, input, gamma, beta, rm, rv, batch, channels, spatialSize);
            var vendor = profiler.GetStats("BatchNorm.cuDNN");
            Assert.True(vendor is not null && vendor.CallCount >= 1,
                "expected the cuDNN BatchNorm dispatch path to be recorded (BatchNorm.cuDNN)");

            profiler.Clear();
            RunBn(cb, useCudnn: false, training: true, input, gamma, beta, rm, rv, batch, channels, spatialSize);
            var generic = profiler.GetStats("BatchNorm.generic");
            Assert.True(generic is not null && generic.CallCount >= 1,
                "expected the hand-written BatchNorm path to be recorded (BatchNorm.generic)");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prevOpts);
            profiler.Enabled = prevEnabled;
            profiler.Clear();
        }
    }
}
#endif
