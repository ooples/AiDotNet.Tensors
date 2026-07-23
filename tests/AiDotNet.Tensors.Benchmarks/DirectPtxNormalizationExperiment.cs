using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #838 first-stage resident screen. Each cell invokes the same production backend API
/// against the same physical buffers with direct PTX disabled and enabled.
/// </summary>
internal static class DirectPtxNormalizationExperiment
{
    private const int Dimension = 64;
    private const float Epsilon = 1e-5f;
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 50;
    private static readonly int[] RowBuckets = [256, 2048, 8192];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);

    internal static void Run(int independentRuns = 3, string scope = "all")
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        if (scope != "all" && scope != "row" && scope != "row256" &&
            scope != "row2048" && scope != "row8192" && scope != "channel")
            throw new ArgumentOutOfRangeException(
                nameof(scope), scope, "Use all, row, row256, row2048, row8192, or channel.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        TensorCodecOptions previousOptions = TensorCodecOptions.Current;
        var benchmarkOptions = new TensorCodecOptions
        {
            UseCudnnBatchNorm = false,
            // PyTorch's fastest default lane permits unordered GPU reductions.
            // Benchmark the equivalent AiDotNet fast lane; deterministic mode
            // retains the fixed-order parameter-gradient specialization.
            Deterministic = false
        };
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        TensorCodecOptions.SetCurrent(benchmarkOptions);
        try
        {
            Console.WriteLine(
                $"Direct-PTX normalization production-route screen: {independentRuns} run(s), " +
                $"{Warmups} warmups + {Samples} samples, {DeviceLaunches} launches/device sample");
            Console.WriteLine("Numerical mode: fast unordered reductions (Deterministic=false), matching PyTorch default");
            Console.WriteLine(
                "Baseline and candidate use identical resident inputs, extents, semantics, stream, " +
                "and production entry point. Error is candidate vs current AiDotNet output.");
            Console.WriteLine(
                "COMPARE is only the first-stage resident screen; production promotion still requires " +
                "three clean wins against the fastest corrected PyTorch eager/compiled lane.");
            Console.WriteLine(
                "cuDNN BatchNorm is ineligible on this host (cudnn_graph64_9.dll unavailable); " +
                "channel baselines use AiDotNet's resident CUDA kernels.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            {
                using var backend = new CudaBackend();
                if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
                if (scope != "channel")
                {
                    foreach (int rows in RowBuckets)
                    {
                        if (scope == "all" || scope == "row" || scope == "row" + rows)
                            RunRowRoutes(run, rows, backend);
                    }
                }
                if (scope is "all" or "channel") RunChannelRoutes(run, backend);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
            TensorCodecOptions.SetCurrent(previousOptions);
        }
    }

    private static void RunRowRoutes(int run, int rows, CudaBackend backend)
    {
        int elements = checked(rows * Dimension);
        float[] inputHost = Values(elements, 1000 + run * 100 + rows, 0.75f);
        float[] gradHost = Values(elements, 2000 + run * 100 + rows, 0.25f);
        float[] gammaHost = Enumerable.Range(0, Dimension).Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Values(Dimension, 3000 + run * 100 + rows, 0.05f);
        float[] meanHost = Values(rows, 4000 + run * 100 + rows, 0.05f);
        float[] invStdHost = Enumerable.Repeat(1.125f, rows).ToArray();
        float[] rmsHost = Enumerable.Repeat(1.25f, rows).ToArray();
        float[] normHost = Enumerable.Repeat(2.5f, rows).ToArray();
        float[] scalarGradientHost = Values(rows, 5000 + run * 100 + rows, 0.25f);

        using var input = backend.AllocateBuffer(inputHost);
        using var gradOutput = backend.AllocateBuffer(gradHost);
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var mean = backend.AllocateBuffer(meanHost);
        using var invStd = backend.AllocateBuffer(invStdHost);
        using var rms = backend.AllocateBuffer(rmsHost);
        using var norm = backend.AllocateBuffer(normHost);
        using var scalarGradient = backend.AllocateBuffer(scalarGradientHost);

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(rows))
        using (var directMean = backend.AllocateBuffer(rows))
        using (var baselineInvStd = backend.AllocateBuffer(rows))
        using (var directInvStd = backend.AllocateBuffer(rows))
        {
            Action baseline = () => backend.LayerNorm(
                input, baselineOutput, gamma, beta, baselineMean, baselineInvStd,
                rows, Dimension, Epsilon);
            Action direct = () => backend.LayerNorm(
                input, directOutput, gamma, beta, directMean, directInvStd,
                rows, Dimension, Epsilon);
            RunCell(run, rows, "LayerNorm forward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineOutput, directOutput),
                    (baselineMean, directMean),
                    (baselineInvStd, directInvStd)),
                DirectPtxRowNormalizationOperation.LayerNormForward);
        }

        using (var baselineGradInput = backend.AllocateBuffer(elements))
        using (var directGradInput = backend.AllocateBuffer(elements))
        using (var baselineGradGamma = backend.AllocateBuffer(Dimension))
        using (var directGradGamma = backend.AllocateBuffer(Dimension))
        using (var baselineGradBeta = backend.AllocateBuffer(Dimension))
        using (var directGradBeta = backend.AllocateBuffer(Dimension))
        {
            Action baseline = () => backend.LayerNormBackward(
                gradOutput, input, gamma, mean, invStd,
                baselineGradInput, baselineGradGamma, baselineGradBeta,
                rows, Dimension, Epsilon);
            Action direct = () => backend.LayerNormBackward(
                gradOutput, input, gamma, mean, invStd,
                directGradInput, directGradGamma, directGradBeta,
                rows, Dimension, Epsilon);
            RunCell(run, rows, "LayerNorm backward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineGradInput, directGradInput),
                    (baselineGradGamma, directGradGamma),
                    (baselineGradBeta, directGradBeta)),
                rows == 8_192
                    ? DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic
                    : DirectPtxRowNormalizationOperation.LayerNormBackwardInput,
                secondaryAuditOperation: rows == 8_192
                    ? null
                    : DirectPtxRowNormalizationOperation.LayerNormGradParameters,
                directPersistentWorkspaceBytes: rows == 8_192
                    ? PtxRowNormalizationD64Kernel.NormalizationWorkspaceBytes : 0,
                directPersistentWorkspaceBoundedAndReusable: rows == 8_192,
                dispatchesPerAction: rows == 8_192 ? 1 : 2);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineRms = backend.AllocateBuffer(rows))
        using (var directRms = backend.AllocateBuffer(rows))
        {
            Action baseline = () => backend.RmsNorm(
                input, baselineOutput, gamma, baselineRms, rows, Dimension, Epsilon);
            Action direct = () => backend.RmsNorm(
                input, directOutput, gamma, directRms, rows, Dimension, Epsilon);
            RunCell(run, rows, "RMSNorm forward", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput), (baselineRms, directRms)),
                DirectPtxRowNormalizationOperation.RmsNormForward);
        }

        using (var baselineGradInput = backend.AllocateBuffer(elements))
        using (var directGradInput = backend.AllocateBuffer(elements))
        using (var baselineGradGamma = backend.AllocateBuffer(Dimension))
        using (var directGradGamma = backend.AllocateBuffer(Dimension))
        {
            Action baseline = () => backend.RmsNormBackward(
                gradOutput, input, gamma, rms,
                baselineGradInput, baselineGradGamma, rows, Dimension, Epsilon);
            Action direct = () => backend.RmsNormBackward(
                gradOutput, input, gamma, rms,
                directGradInput, directGradGamma, rows, Dimension, Epsilon);
            RunCell(run, rows, "RMSNorm backward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineGradInput, directGradInput),
                    (baselineGradGamma, directGradGamma)),
                rows == 8_192
                    ? DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic
                    : DirectPtxRowNormalizationOperation.RmsNormBackwardInput,
                secondaryAuditOperation: rows == 8_192
                    ? null
                    : DirectPtxRowNormalizationOperation.RmsNormGradGamma,
                directPersistentWorkspaceBytes: rows == 8_192
                    ? PtxRowNormalizationD64Kernel.NormalizationWorkspaceBytes : 0,
                directPersistentWorkspaceBoundedAndReusable: rows == 8_192,
                dispatchesPerAction: rows == 8_192 ? 1 : 2);
        }

        using (var baselineOutput = backend.AllocateBuffer(rows))
        using (var directOutput = backend.AllocateBuffer(rows))
        {
            Action baseline = () => backend.NormAxis(input, baselineOutput, rows, Dimension);
            Action direct = () => backend.NormAxis(input, directOutput, rows, Dimension);
            RunCell(run, rows, "L2 norm axis", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxRowNormalizationOperation.NormAxis);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () => backend.NormBackward(
                scalarGradient, input, norm, baselineOutput, rows, Dimension);
            Action direct = () => backend.NormBackward(
                scalarGradient, input, norm, directOutput, rows, Dimension);
            RunCell(run, rows, "L2 norm backward", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxRowNormalizationOperation.NormBackward);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () => backend.NormalizeL2(input, baselineOutput, rows, Dimension);
            Action direct = () => backend.NormalizeL2(input, directOutput, rows, Dimension);
            RunCell(run, rows, "NormalizeL2", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxRowNormalizationOperation.NormalizeL2);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () => backend.NormalizeRowsFused(input, baselineOutput, rows, Dimension);
            Action direct = () => backend.NormalizeRowsFused(input, directOutput, rows, Dimension);
            RunCell(run, rows, "NormalizeRowsFused", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxRowNormalizationOperation.NormalizeL2);
        }

        using (var baselineOutput = backend.AllocateBuffer(1))
        using (var directOutput = backend.AllocateBuffer(1))
        {
            Action baseline = () => backend.ReduceNormL2(input, baselineOutput, elements);
            Action direct = () => backend.ReduceNormL2(input, directOutput, elements);
            RunCell(run, rows, "ReduceNormL2", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxRowNormalizationOperation.ReduceNormL2,
                baselineCaptureCompatible: false);
        }

        RunFp16RowRoutes(run, rows, backend, inputHost, gradHost,
            gammaHost, betaHost, meanHost, invStdHost);
    }

    private static void RunFp16RowRoutes(
        int run, int rows, CudaBackend backend,
        float[] inputHost, float[] gradHost,
        float[] gammaHost, float[] betaHost,
        float[] meanHost, float[] invStdHost)
    {
        int elements = checked(rows * Dimension);
        byte[] inputBytes = HalfBytes(inputHost);
        byte[] gradBytes = HalfBytes(gradHost);
        byte[] gammaBytes = HalfBytes(gammaHost);
        byte[] betaBytes = HalfBytes(betaHost);
        using var input = backend.AllocateByteBuffer(inputBytes.Length);
        using var gradOutput = backend.AllocateByteBuffer(gradBytes.Length);
        using var gamma = backend.AllocateByteBuffer(gammaBytes.Length);
        using var beta = backend.AllocateByteBuffer(betaBytes.Length);
        backend.UploadBytes(input, inputBytes);
        backend.UploadBytes(gradOutput, gradBytes);
        backend.UploadBytes(gamma, gammaBytes);
        backend.UploadBytes(beta, betaBytes);
        using var mean = backend.AllocateBuffer(meanHost);
        using var invStd = backend.AllocateBuffer(invStdHost);

        using (var baselineOutput = backend.AllocateByteBuffer(elements * 2))
        using (var directOutput = backend.AllocateByteBuffer(elements * 2))
        using (var baselineMean = backend.AllocateBuffer(rows))
        using (var directMean = backend.AllocateBuffer(rows))
        using (var baselineVariance = backend.AllocateBuffer(rows))
        using (var directVariance = backend.AllocateBuffer(rows))
        {
            Action baseline = () => backend.Fp16LayerNorm(
                input, gamma, beta, baselineOutput, baselineMean, baselineVariance,
                rows, Dimension, Epsilon);
            Action direct = () => backend.Fp16LayerNorm(
                input, gamma, beta, directOutput, directMean, directVariance,
                rows, Dimension, Epsilon);
            RunCell(run, rows, "FP16 LayerNorm forward", backend, baseline, direct,
                () => PtxCompat.Max(
                    MaxHalfError(backend, baselineOutput, directOutput, elements),
                    MaxError(backend,
                        (baselineMean, directMean),
                        (baselineVariance, directVariance))),
                DirectPtxRowNormalizationOperation.Fp16LayerNormForward);
        }

        using (var baselineGradInput = backend.AllocateByteBuffer(elements * 2))
        using (var directGradInput = backend.AllocateByteBuffer(elements * 2))
        {
            Action baseline = () => backend.Fp16LayerNormBackward(
                gradOutput, input, gamma, mean, invStd,
                baselineGradInput, rows, Dimension);
            Action direct = () => backend.Fp16LayerNormBackward(
                gradOutput, input, gamma, mean, invStd,
                directGradInput, rows, Dimension);
            RunCell(run, rows, "FP16 LayerNorm backward", backend, baseline, direct,
                () => MaxHalfError(backend, baselineGradInput, directGradInput, elements),
                DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput);
        }

        using (var baselineGradGamma = backend.AllocateByteBuffer(Dimension * 2))
        using (var directGradGamma = backend.AllocateByteBuffer(Dimension * 2))
        using (var baselineGradBeta = backend.AllocateByteBuffer(Dimension * 2))
        using (var directGradBeta = backend.AllocateByteBuffer(Dimension * 2))
        {
            Action baseline = () => backend.Fp16LayerNormGradParams(
                gradOutput, input, mean, invStd,
                baselineGradGamma, baselineGradBeta, rows, Dimension);
            Action direct = () => backend.Fp16LayerNormGradParams(
                gradOutput, input, mean, invStd,
                directGradGamma, directGradBeta, rows, Dimension);
            RunCell(run, rows, "FP16 LayerNorm params", backend, baseline, direct,
                () => PtxCompat.Max(
                    MaxHalfError(backend, baselineGradGamma, directGradGamma, Dimension),
                    MaxHalfError(backend, baselineGradBeta, directGradBeta, Dimension)),
                DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters);
        }
    }

    private static void RunChannelRoutes(int run, CudaBackend backend)
    {
        RunBatchNormRoutes(run, backend);
        RunGroupNormRoutes(run, backend);
        RunInstanceNormRoutes(run, backend);
    }

    private static void RunBatchNormRoutes(int run, CudaBackend backend)
    {
        const int batch = PtxChannelNormalizationD64Kernel.BatchNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.BatchNormChannels;
        const int spatial = PtxChannelNormalizationD64Kernel.BatchNormSpatial;
        const int elements = batch * channels * spatial;
        const float momentum = 0.1f;
        float[] inputHost = Values(elements, 6000 + run, 0.75f);
        float[] gradHost = Values(elements, 6100 + run, 0.25f);
        float[] residualHost = Values(elements, 6200 + run, 0.20f);
        float[] gammaHost = Enumerable.Range(0, channels).Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Values(channels, 6300 + run, 0.05f);
        float[] meanHost = Values(channels, 6400 + run, 0.05f);
        float[] varianceHost = Enumerable.Range(0, channels).Select(i => 0.75f + i / 512f).ToArray();
        float[] invStdHost = Enumerable.Repeat(1.125f, channels).ToArray();
        using var input = backend.AllocateBuffer(inputHost);
        using var gradOutput = backend.AllocateBuffer(gradHost);
        using var residual = backend.AllocateBuffer(residualHost);
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var fixedMean = backend.AllocateBuffer(meanHost);
        using var fixedVariance = backend.AllocateBuffer(varianceHost);
        using var fixedInvStd = backend.AllocateBuffer(invStdHost);

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineRunningMean = backend.AllocateBuffer(meanHost))
        using (var directRunningMean = backend.AllocateBuffer(meanHost))
        using (var baselineRunningVariance = backend.AllocateBuffer(varianceHost))
        using (var directRunningVariance = backend.AllocateBuffer(varianceHost))
        using (var baselineMean = backend.AllocateBuffer(channels))
        using (var directMean = backend.AllocateBuffer(channels))
        using (var baselineInvStd = backend.AllocateBuffer(channels))
        using (var directInvStd = backend.AllocateBuffer(channels))
        {
            Action baseline = () => backend.BatchNorm(
                input, baselineOutput, gamma, beta,
                baselineRunningMean, baselineRunningVariance, baselineMean, baselineInvStd,
                batch, channels, spatial, Epsilon, momentum, training: true);
            Action direct = () => backend.BatchNorm(
                input, directOutput, gamma, beta,
                directRunningMean, directRunningVariance, directMean, directInvStd,
                batch, channels, spatial, Epsilon, momentum, training: true);
            RunChannelCell(run, elements, "BatchNorm training", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineOutput, directOutput),
                    (baselineMean, directMean),
                    (baselineInvStd, directInvStd)),
                DirectPtxChannelNormalizationOperation.BatchNormTraining, momentum: momentum);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(channels))
        using (var directMean = backend.AllocateBuffer(channels))
        using (var baselineInvStd = backend.AllocateBuffer(channels))
        using (var directInvStd = backend.AllocateBuffer(channels))
        {
            Action baseline = () => backend.BatchNorm(
                input, baselineOutput, gamma, beta,
                fixedMean, fixedVariance, baselineMean, baselineInvStd,
                batch, channels, spatial, Epsilon, momentum, training: false);
            Action direct = () => backend.BatchNorm(
                input, directOutput, gamma, beta,
                fixedMean, fixedVariance, directMean, directInvStd,
                batch, channels, spatial, Epsilon, momentum, training: false);
            RunChannelCell(run, elements, "BatchNorm inference", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxChannelNormalizationOperation.BatchNormInference, momentum: momentum);
        }

        (FusedActivationType Activation, DirectPtxChannelNormalizationOperation Operation, string Name)[] activations =
        [
            (FusedActivationType.ReLU, DirectPtxChannelNormalizationOperation.BatchNormRelu, "BatchNorm+ReLU"),
            (FusedActivationType.GELU, DirectPtxChannelNormalizationOperation.BatchNormGelu, "BatchNorm+GELU"),
            (FusedActivationType.Sigmoid, DirectPtxChannelNormalizationOperation.BatchNormSigmoid, "BatchNorm+Sigmoid"),
            (FusedActivationType.Tanh, DirectPtxChannelNormalizationOperation.BatchNormTanh, "BatchNorm+Tanh")
        ];
        foreach (var activation in activations)
        {
            using var baselineOutput = backend.AllocateBuffer(elements);
            using var directOutput = backend.AllocateBuffer(elements);
            using var saveMean = backend.AllocateBuffer(channels);
            using var saveInvStd = backend.AllocateBuffer(channels);
            Action baseline = () =>
            {
                if (!backend.TryFusedBatchNormActivation(
                        input, baselineOutput, gamma, beta, fixedMean, fixedVariance,
                        saveMean, saveInvStd, batch, channels, spatial,
                        Epsilon, momentum, training: false, activation.Activation))
                    throw new InvalidOperationException("Current CUDA fused BatchNorm activation is unavailable.");
            };
            Action direct = () =>
            {
                if (!backend.TryFusedBatchNormActivation(
                        input, directOutput, gamma, beta, fixedMean, fixedVariance,
                        saveMean, saveInvStd, batch, channels, spatial,
                        Epsilon, momentum, training: false, activation.Activation))
                    throw new InvalidOperationException(backend.DirectPtxLastError);
            };
            RunChannelCell(run, elements, activation.Name, backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)), activation.Operation);
        }

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () =>
            {
                if (!backend.TryResidualBatchNormRelu(
                        input, residual, baselineOutput, gamma, beta, fixedMean, fixedVariance,
                        batch, channels, spatial, Epsilon))
                    throw new InvalidOperationException("Current residual BatchNorm ReLU is unavailable.");
            };
            Action direct = () =>
            {
                if (!backend.TryResidualBatchNormRelu(
                        input, residual, directOutput, gamma, beta, fixedMean, fixedVariance,
                        batch, channels, spatial, Epsilon))
                    throw new InvalidOperationException(backend.DirectPtxLastError);
            };
            RunChannelCell(run, elements, "Residual+BN+ReLU", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu);
        }

        using (var baselineGradInput = backend.AllocateBuffer(elements))
        using (var directGradInput = backend.AllocateBuffer(elements))
        using (var baselineGradGamma = backend.AllocateBuffer(channels))
        using (var directGradGamma = backend.AllocateBuffer(channels))
        using (var baselineGradBeta = backend.AllocateBuffer(channels))
        using (var directGradBeta = backend.AllocateBuffer(channels))
        {
            Action baseline = () => backend.BatchNormBackward(
                gradOutput, input, gamma, fixedMean, fixedInvStd,
                baselineGradInput, baselineGradGamma, baselineGradBeta,
                batch, channels, spatial, Epsilon);
            Action direct = () => backend.BatchNormBackward(
                gradOutput, input, gamma, fixedMean, fixedInvStd,
                directGradInput, directGradGamma, directGradBeta,
                batch, channels, spatial, Epsilon);
            RunChannelCell(run, elements, "BatchNorm backward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineGradInput, directGradInput),
                    (baselineGradGamma, directGradGamma),
                    (baselineGradBeta, directGradBeta)),
                DirectPtxChannelNormalizationOperation.BatchNormBackward);
        }
    }

    private static void RunGroupNormRoutes(int run, CudaBackend backend)
    {
        const int batch = PtxChannelNormalizationD64Kernel.GroupNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.GroupNormChannels;
        const int groups = PtxChannelNormalizationD64Kernel.GroupNormGroups;
        const int spatial = PtxChannelNormalizationD64Kernel.GroupNormSpatial;
        const int elements = batch * channels * spatial;
        const int stats = batch * groups;
        float[] inputHost = Values(elements, 7000 + run, 0.75f);
        float[] rightHost = Values(elements, 7100 + run, 0.20f);
        float[] gradHost = Values(elements, 7150 + run, 0.25f);
        float[] gammaHost = Enumerable.Range(0, channels).Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Values(channels, 7200 + run, 0.05f);
        GroupNormStats(inputHost, batch, channels, groups, spatial,
            out float[] meanHost, out float[] varianceHost);
        using var input = backend.AllocateBuffer(inputHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var gradOutput = backend.AllocateBuffer(gradHost);
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var fixedMean = backend.AllocateBuffer(meanHost);
        using var fixedVariance = backend.AllocateBuffer(varianceHost);

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(stats))
        using (var directMean = backend.AllocateBuffer(stats))
        using (var baselineSecond = backend.AllocateBuffer(stats))
        using (var directSecond = backend.AllocateBuffer(stats))
        {
            Action baseline = () => backend.GroupNorm(
                input, baselineOutput, gamma, beta, baselineMean, baselineSecond,
                batch, groups, channels, spatial, Epsilon);
            Action direct = () => backend.GroupNorm(
                input, directOutput, gamma, beta, directMean, directSecond,
                batch, groups, channels, spatial, Epsilon);
            RunChannelCell(run, elements, "GroupNorm forward", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput), (baselineMean, directMean)),
                DirectPtxChannelNormalizationOperation.GroupNormForward);
        }

        using (var normalized = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(stats))
        using (var baselineSecond = backend.AllocateBuffer(stats))
        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () =>
            {
                backend.GroupNorm(input, normalized, gamma, beta, baselineMean, baselineSecond,
                    batch, groups, channels, spatial, Epsilon);
                backend.Swish(normalized, baselineOutput, elements);
            };
            Action direct = () =>
            {
                if (!backend.TryDirectPtxGroupNormSwishUnit64(
                        input, directOutput, gamma, beta,
                        batch, groups, channels, spatial, Epsilon))
                    throw new InvalidOperationException(backend.DirectPtxLastError);
            };
            long baselineTemporary = checked((long)(elements + 2 * stats) * sizeof(float));
            RunChannelCell(run, elements, "GroupNorm+Swish", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxChannelNormalizationOperation.GroupNormSwish,
                baselineTemporaryBytes: baselineTemporary);
        }

        using (var sum = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(stats))
        using (var baselineSecond = backend.AllocateBuffer(stats))
        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        {
            Action baseline = () =>
            {
                backend.Add(input, right, sum, elements);
                backend.GroupNorm(sum, baselineOutput, gamma, beta, baselineMean, baselineSecond,
                    batch, groups, channels, spatial, Epsilon);
            };
            Action direct = () =>
            {
                if (!backend.TryDirectPtxAddGroupNormUnit64(
                        input, right, directOutput, gamma, beta,
                        batch, groups, channels, spatial, Epsilon))
                    throw new InvalidOperationException(backend.DirectPtxLastError);
            };
            long baselineTemporary = checked((long)(elements + 2 * stats) * sizeof(float));
            RunChannelCell(run, elements, "Add+GroupNorm", backend, baseline, direct,
                () => MaxError(backend, (baselineOutput, directOutput)),
                DirectPtxChannelNormalizationOperation.AddGroupNorm,
                baselineTemporaryBytes: baselineTemporary);
        }

        byte[] inputBytes = HalfBytes(inputHost);
        using (var inputHalf = backend.AllocateByteBuffer(inputBytes.Length))
        using (var baselineHalf = backend.AllocateByteBuffer(inputBytes.Length))
        using (var directHalf = backend.AllocateByteBuffer(inputBytes.Length))
        {
            backend.UploadBytes(inputHalf, inputBytes);
            Action baseline = () => backend.Fp16GroupNormSwish(
                inputHalf, gamma, beta, baselineHalf,
                batch, groups, channels, spatial, Epsilon);
            Action direct = () => backend.Fp16GroupNormSwish(
                inputHalf, gamma, beta, directHalf,
                batch, groups, channels, spatial, Epsilon);
            RunChannelCell(run, elements, "FP16 GroupNorm+Swish", backend, baseline, direct,
                () => MaxHalfError(backend, baselineHalf, directHalf, elements),
                DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish);
        }

        RunGroupNormBackwardRoute(
            run, backend, gradOutput, input, gamma, fixedMean, fixedVariance,
            batch, channels, groups, spatial);
    }

    private static void RunGroupNormBackwardRoute(
        int run, CudaBackend backend,
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer mean, IGpuBuffer variance,
        int batch, int channels, int groups, int spatial)
    {
        int elements = checked(batch * channels * spatial);
        int groupSize = checked((channels / groups) * spatial);
        int groupCount = checked(batch * groups);

        using var onesGroup = backend.AllocateBuffer(groupSize);
        using var onesSpatial = backend.AllocateBuffer(spatial);
        using var onesBatch = backend.AllocateBuffer(batch);
        using var meanExpanded = backend.AllocateBuffer(elements);
        using var centered = backend.AllocateBuffer(elements);
        using var safeVariance = backend.AllocateBuffer(groupCount);
        using var standardDeviation = backend.AllocateBuffer(groupCount);
        using var inverseStandardDeviation = backend.AllocateBuffer(groupCount);
        using var inverseStandardDeviationExpanded = backend.AllocateBuffer(elements);
        using var normalized = backend.AllocateBuffer(elements);
        using var gammaSpatial = backend.AllocateBuffer(channels * spatial);
        using var gammaExpanded = backend.AllocateBuffer(elements);
        using var scaledGradient = backend.AllocateBuffer(elements);
        using var scaledGradientNorm = backend.AllocateBuffer(elements);
        using var sumGradient = backend.AllocateBuffer(groupCount);
        using var sumGradientNorm = backend.AllocateBuffer(groupCount);
        using var sumGradientExpanded = backend.AllocateBuffer(elements);
        using var sumGradientNormExpanded = backend.AllocateBuffer(elements);
        using var groupScaledGradient = backend.AllocateBuffer(elements);
        using var firstDifference = backend.AllocateBuffer(elements);
        using var normalizedProjection = backend.AllocateBuffer(elements);
        using var groupDifference = backend.AllocateBuffer(elements);
        using var inputScale = backend.AllocateBuffer(elements);
        using var gammaProducts = backend.AllocateBuffer(elements);
        using var gammaProductsPermuted = backend.AllocateBuffer(elements);
        using var gradientPermuted = backend.AllocateBuffer(elements);
        using var baselineGradInput = backend.AllocateBuffer(elements);
        using var directGradInput = backend.AllocateBuffer(elements);
        using var baselineGradGamma = backend.AllocateBuffer(channels);
        using var directGradGamma = backend.AllocateBuffer(channels);
        using var baselineGradBeta = backend.AllocateBuffer(channels);
        using var directGradBeta = backend.AllocateBuffer(channels);
        int[] collapsedShape = [batch, channels, spatial];
        int[] channelFirst = [1, 0, 2];

        Action baseline = () =>
        {
            backend.Fill(onesGroup, 1f, groupSize);
            backend.Fill(onesSpatial, 1f, spatial);
            backend.Fill(onesBatch, 1f, batch);
            backend.OuterProduct(mean, onesGroup, meanExpanded, groupCount, groupSize);
            backend.Subtract(input, meanExpanded, centered, elements);
            backend.AddScalar(variance, safeVariance, Epsilon, groupCount);
            backend.Sqrt(safeVariance, standardDeviation, groupCount);
            backend.Reciprocal(standardDeviation, inverseStandardDeviation, groupCount);
            backend.OuterProduct(inverseStandardDeviation, onesGroup,
                inverseStandardDeviationExpanded, groupCount, groupSize);
            backend.Multiply(centered, inverseStandardDeviationExpanded, normalized, elements);
            backend.OuterProduct(gamma, onesSpatial, gammaSpatial, channels, spatial);
            backend.OuterProduct(onesBatch, gammaSpatial, gammaExpanded, batch, channels * spatial);
            backend.Multiply(gradOutput, gammaExpanded, scaledGradient, elements);
            backend.Multiply(scaledGradient, normalized, scaledGradientNorm, elements);
            backend.SumAxis(scaledGradient, sumGradient, groupCount, groupSize);
            backend.SumAxis(scaledGradientNorm, sumGradientNorm, groupCount, groupSize);
            backend.OuterProduct(sumGradient, onesGroup, sumGradientExpanded, groupCount, groupSize);
            backend.OuterProduct(sumGradientNorm, onesGroup,
                sumGradientNormExpanded, groupCount, groupSize);
            backend.Scale(scaledGradient, groupScaledGradient, groupSize, elements);
            backend.Subtract(groupScaledGradient, sumGradientExpanded, firstDifference, elements);
            backend.Multiply(normalized, sumGradientNormExpanded, normalizedProjection, elements);
            backend.Subtract(firstDifference, normalizedProjection, groupDifference, elements);
            backend.Scale(inverseStandardDeviationExpanded, inputScale, 1f / groupSize, elements);
            backend.Multiply(inputScale, groupDifference, baselineGradInput, elements);
            backend.Multiply(gradOutput, normalized, gammaProducts, elements);
            backend.Permute(gammaProducts, gammaProductsPermuted, collapsedShape, channelFirst);
            backend.Permute(gradOutput, gradientPermuted, collapsedShape, channelFirst);
            backend.SumAxis(gammaProductsPermuted, baselineGradGamma, channels, batch * spatial);
            backend.SumAxis(gradientPermuted, baselineGradBeta, channels, batch * spatial);
        };
        Action direct = () =>
        {
            if (!backend.TryDirectPtxGroupNormBackwardUnit64(
                    gradOutput, input, gamma, mean, variance,
                    directGradInput, directGradGamma, directGradBeta,
                    batch, groups, channels, spatial, Epsilon))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        };
        long temporaryFloats = checked(
            groupSize + spatial + batch + 5L * groupCount +
            (long)channels * spatial + 17L * elements);
        RunChannelCell(run, elements, "GroupNorm backward", backend, baseline, direct,
            () => MaxError(backend,
                (baselineGradInput, directGradInput),
                (baselineGradGamma, directGradGamma),
                (baselineGradBeta, directGradBeta)),
            DirectPtxChannelNormalizationOperation.GroupNormBackwardInput,
            secondaryAuditOperation: DirectPtxChannelNormalizationOperation.GroupNormGradParameters,
            baselineTemporaryBytes: checked(temporaryFloats * sizeof(float)),
            baselineCaptureCompatible: false,
            dispatchesPerAction: 2);
    }

    private static void RunInstanceNormRoutes(int run, CudaBackend backend)
    {
        const int batch = PtxChannelNormalizationD64Kernel.InstanceNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.InstanceNormChannels;
        const int spatial = PtxChannelNormalizationD64Kernel.InstanceNormSpatial;
        const int elements = batch * channels * spatial;
        const int stats = batch * channels;
        float[] inputHost = Values(elements, 8000 + run, 0.75f);
        float[] gradHost = Values(elements, 8100 + run, 0.25f);
        float[] gammaHost = Enumerable.Range(0, channels).Select(i => 0.75f + i / 256f).ToArray();
        float[] betaHost = Values(channels, 8200 + run, 0.05f);
        float[] meanHost = Values(stats, 8300 + run, 0.05f);
        float[] invStdHost = Enumerable.Repeat(1.125f, stats).ToArray();
        using var input = backend.AllocateBuffer(inputHost);
        using var gradOutput = backend.AllocateBuffer(gradHost);
        using var gamma = backend.AllocateBuffer(gammaHost);
        using var beta = backend.AllocateBuffer(betaHost);
        using var fixedMean = backend.AllocateBuffer(meanHost);
        using var fixedInvStd = backend.AllocateBuffer(invStdHost);

        using (var baselineOutput = backend.AllocateBuffer(elements))
        using (var directOutput = backend.AllocateBuffer(elements))
        using (var baselineMean = backend.AllocateBuffer(stats))
        using (var directMean = backend.AllocateBuffer(stats))
        using (var baselineInvStd = backend.AllocateBuffer(stats))
        using (var directInvStd = backend.AllocateBuffer(stats))
        {
            Action baseline = () => backend.InstanceNorm(
                input, baselineOutput, gamma, beta, baselineMean, baselineInvStd,
                batch, channels, spatial, Epsilon);
            Action direct = () => backend.InstanceNorm(
                input, directOutput, gamma, beta, directMean, directInvStd,
                batch, channels, spatial, Epsilon);
            RunChannelCell(run, elements, "InstanceNorm forward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineOutput, directOutput),
                    (baselineMean, directMean),
                    (baselineInvStd, directInvStd)),
                DirectPtxChannelNormalizationOperation.InstanceNormForward);
        }

        using (var baselineGradInput = backend.AllocateBuffer(elements))
        using (var directGradInput = backend.AllocateBuffer(elements))
        using (var baselineGradGamma = backend.AllocateBuffer(channels))
        using (var directGradGamma = backend.AllocateBuffer(channels))
        using (var baselineGradBeta = backend.AllocateBuffer(channels))
        using (var directGradBeta = backend.AllocateBuffer(channels))
        {
            Action baseline = () => backend.InstanceNormBackward(
                gradOutput, input, gamma, fixedMean, fixedInvStd,
                baselineGradInput, baselineGradGamma, baselineGradBeta,
                batch, channels, spatial, Epsilon);
            Action direct = () => backend.InstanceNormBackward(
                gradOutput, input, gamma, fixedMean, fixedInvStd,
                directGradInput, directGradGamma, directGradBeta,
                batch, channels, spatial, Epsilon);
            RunChannelCell(run, elements, "InstanceNorm backward", backend, baseline, direct,
                () => MaxError(backend,
                    (baselineGradInput, directGradInput),
                    (baselineGradGamma, directGradGamma),
                    (baselineGradBeta, directGradBeta)),
                DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput,
                secondaryAuditOperation: DirectPtxChannelNormalizationOperation.InstanceNormGradParameters,
                baselineTemporaryBytes: checked((long)2 * stats * sizeof(float)),
                baselineCaptureCompatible: false,
                dispatchesPerAction: 2);
        }
    }

    private static void RunCell(
        int run, int rows, string name, CudaBackend backend,
        Action baseline, Action direct, Func<float> compare,
        DirectPtxRowNormalizationOperation auditOperation,
        DirectPtxRowNormalizationOperation? secondaryAuditOperation = null,
        long baselineTemporaryBytes = 0, long directTemporaryBytes = 0,
        long directPersistentWorkspaceBytes = 0,
        bool directPersistentWorkspaceBoundedAndReusable = false,
        bool baselineCaptureCompatible = true,
        bool directCaptureCompatible = true,
        int dispatchesPerAction = 1)
    {
        RunCellCore(
            run, rows, name, backend, baseline, direct, compare,
            () => backend.TryGetDirectPtxRowNormalizationAudit(
                PtxRowNormalizationD64Kernel.SelectFastOperation(
                    auditOperation, deterministic: false, rows), rows, Epsilon,
                out DirectPtxKernelAudit audit) ? audit : null,
            secondaryAuditOperation.HasValue
                ? () => backend.TryGetDirectPtxRowNormalizationAudit(
                    PtxRowNormalizationD64Kernel.SelectFastOperation(
                        secondaryAuditOperation.Value, deterministic: false, rows), rows, Epsilon,
                    out DirectPtxKernelAudit audit) ? audit : null
                : null,
            () => backend.DirectPtxRowNormalizationDispatchCount,
            baselineTemporaryBytes, directTemporaryBytes,
            directPersistentWorkspaceBytes,
            directPersistentWorkspaceBoundedAndReusable,
            baselineCaptureCompatible, directCaptureCompatible,
            dispatchesPerAction);
    }

    private static void RunChannelCell(
        int run, int extent, string name, CudaBackend backend,
        Action baseline, Action direct, Func<float> compare,
        DirectPtxChannelNormalizationOperation auditOperation,
        DirectPtxChannelNormalizationOperation? secondaryAuditOperation = null,
        float momentum = 0f,
        long baselineTemporaryBytes = 0, long directTemporaryBytes = 0,
        long directPersistentWorkspaceBytes = 0,
        bool directPersistentWorkspaceBoundedAndReusable = false,
        bool baselineCaptureCompatible = true,
        bool directCaptureCompatible = true,
        int dispatchesPerAction = 1)
    {
        RunCellCore(
            run, extent, name, backend, baseline, direct, compare,
            () => backend.TryGetDirectPtxChannelNormalizationAudit(
                auditOperation, Epsilon, momentum, out DirectPtxKernelAudit audit) ? audit : null,
            secondaryAuditOperation.HasValue
                ? () => backend.TryGetDirectPtxChannelNormalizationAudit(
                    secondaryAuditOperation.Value, Epsilon, momentum,
                    out DirectPtxKernelAudit audit) ? audit : null
                : null,
            () => backend.DirectPtxChannelNormalizationDispatchCount,
            baselineTemporaryBytes, directTemporaryBytes,
            directPersistentWorkspaceBytes,
            directPersistentWorkspaceBoundedAndReusable,
            baselineCaptureCompatible, directCaptureCompatible,
            dispatchesPerAction);
    }

    private static void RunCellCore(
        int run, int extent, string name, CudaBackend backend,
        Action baseline, Action direct, Func<float> compare,
        Func<DirectPtxKernelAudit?> getAudit,
        Func<DirectPtxKernelAudit?>? getSecondaryAudit,
        Func<long> getDirectDispatchCount,
        long baselineTemporaryBytes, long directTemporaryBytes,
        long directPersistentWorkspaceBytes,
        bool directPersistentWorkspaceBoundedAndReusable,
        bool baselineCaptureCompatible, bool directCaptureCompatible,
        int dispatchesPerAction)
    {
        DirectPtxFeatureGate.TestOverride = false;
        baseline();
        backend.Synchronize();
        DirectPtxFeatureGate.TestOverride = true;
        long dispatchBefore = getDirectDispatchCount();
        direct();
        backend.Synchronize();
        RequireDispatchDelta(
            backend, name, getDirectDispatchCount, dispatchBefore, dispatchesPerAction);
        float error = compare();
        DirectPtxKernelAudit? audit = getAudit() ?? throw new InvalidOperationException(
            $"{name}: the direct-PTX audit is missing after an admitted dispatch.");
        DirectPtxKernelAudit? secondaryAudit = getSecondaryAudit?.Invoke();
        if (getSecondaryAudit is not null && secondaryAudit is null)
            throw new InvalidOperationException(
                $"{name}: the secondary direct-PTX audit is missing after an admitted dispatch.");

        DirectPtxFeatureGate.TestOverride = false;
        Distribution baselineDevice = MeasureDevice(
            backend, baseline, baselineCaptureCompatible);
        Distribution baselineE2e = MeasureEndToEnd(backend, baseline);
        long baselineAllocation = MeasureAllocation(backend, baseline);

        DirectPtxFeatureGate.TestOverride = true;
        Distribution directDevice = MeasureDevice(
            backend, direct, directCaptureCompatible, getDirectDispatchCount, name,
            dispatchesPerAction);
        Distribution directE2e = MeasureEndToEnd(
            backend, direct, getDirectDispatchCount, name, dispatchesPerAction);
        long directAllocation = MeasureAllocation(
            backend, direct, getDirectDispatchCount, name, dispatchesPerAction);

        double speedup = baselineDevice.Median / directDevice.Median;
        bool tail = directDevice.P95 <= baselineDevice.P95 * 1.10;
        bool workspaceEligible = directPersistentWorkspaceBytes >= 0 &&
            (directPersistentWorkspaceBytes == 0 ||
             directPersistentWorkspaceBoundedAndReusable);
        bool advanceToCompetitor = speedup >= 1.10 && tail && directAllocation == 0 &&
            directTemporaryBytes == 0 && workspaceEligible &&
            directCaptureCompatible && error <= 2e-3f &&
            audit.Function.LocalBytesPerThread == 0 &&
            (secondaryAudit is null || secondaryAudit.Function.LocalBytesPerThread == 0);
        Print(run, extent, name, "AiDotNet", baselineDevice, baselineE2e,
            baselineAllocation, baselineTemporaryBytes, 0, error,
            null, null, 1.0, false);
        Print(run, extent, name, "Direct PTX", directDevice, directE2e,
            directAllocation, directTemporaryBytes, directPersistentWorkspaceBytes,
            error, audit, secondaryAudit, speedup, advanceToCompetitor);
    }

    private static Distribution MeasureDevice(
        CudaBackend backend, Action action, bool captureCompatible,
        Func<long>? getDispatchCount = null, string? operation = null,
        int dispatchesPerAction = 1)
    {
        long dispatchBefore = getDispatchCount?.Invoke() ?? 0;
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        if (!captureCompatible)
        {
            for (int sample = 0; sample < values.Length; sample++)
            {
                backend.RecordEvent(start, backend.DefaultStream);
                for (int launch = 0; launch < DeviceLaunches; launch++) action();
                backend.RecordEvent(end, backend.DefaultStream);
                end.Synchronize();
                values[sample] = backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
            }
            if (getDispatchCount is not null)
                RequireDispatchDelta(backend, operation!, getDispatchCount, dispatchBefore,
                    dispatchesPerAction * (Warmups + Samples * DeviceLaunches));
            return Summarize(values);
        }

        IntPtr graph = backend.CaptureGraph(() =>
        {
            for (int launch = 0; launch < DeviceLaunches; launch++) action();
        });
        if (graph == IntPtr.Zero)
            throw new InvalidOperationException("Normalization device benchmark could not capture its launch batch.");
        try
        {
            for (int i = 0; i < Warmups; i++) backend.LaunchCapturedGraph(graph);
            backend.Synchronize();
            for (int sample = 0; sample < values.Length; sample++)
            {
                backend.RecordEvent(start, backend.DefaultStream);
                backend.LaunchCapturedGraph(graph);
                backend.RecordEvent(end, backend.DefaultStream);
                end.Synchronize();
                values[sample] = backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
            }
        }
        finally { backend.DestroyCapturedGraph(graph); }
        if (getDispatchCount is not null)
            RequireDispatchDelta(backend, operation!, getDispatchCount, dispatchBefore,
                dispatchesPerAction * (Warmups + DeviceLaunches));
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(
        CudaBackend backend, Action action,
        Func<long>? getDispatchCount = null, string? operation = null,
        int dispatchesPerAction = 1)
    {
        long dispatchBefore = getDispatchCount?.Invoke() ?? 0;
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < values.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        if (getDispatchCount is not null)
            RequireDispatchDelta(backend, operation!, getDispatchCount, dispatchBefore,
                dispatchesPerAction * (Warmups + Samples));
        return Summarize(values);
    }

    private static long MeasureAllocation(
        CudaBackend backend, Action action,
        Func<long>? getDispatchCount = null, string? operation = null,
        int dispatchesPerAction = 1)
    {
        long dispatchBefore = getDispatchCount?.Invoke() ?? 0;
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
        long before = PtxCompat.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long allocation = (PtxCompat.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        if (getDispatchCount is not null)
            RequireDispatchDelta(backend, operation!, getDispatchCount, dispatchBefore,
                dispatchesPerAction * (8 + Samples));
        return allocation;
    }

    private static void RequireDispatchDelta(
        CudaBackend backend, string operation, Func<long> getDispatchCount,
        long before, int expected)
    {
        long actual = getDispatchCount() - before;
        if (actual != expected)
            throw new InvalidOperationException(
                $"{operation}: expected {expected} direct-PTX dispatches but observed {actual}; " +
                $"fallback is not benchmark-eligible. Last error: {backend.DirectPtxLastError ?? "<none>"}");
    }

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            "run rows operation                 method      dev med/p95/p99 us  " +
            "e2e med/p95/p99 us allocB tmpB persistB error    speed  screen R/S/L/B");
        Console.WriteLine(new string('-', 163));
    }

    private static void Print(
        int run, int rows, string name, string method,
        Distribution device, Distribution e2e,
        long allocation, long temporaryBytes, long persistentWorkspaceBytes, float error,
        DirectPtxKernelAudit? audit, DirectPtxKernelAudit? secondaryAudit,
        double speedup, bool advanceToCompetitor)
    {
        string resources = audit is null ? "-/-/-/-" :
            secondaryAudit is null
                ? $"{audit.Function.RegistersPerThread}/{audit.Function.StaticSharedBytes}/" +
                  $"{audit.Function.LocalBytesPerThread}/{audit.ActiveBlocksPerMultiprocessor}"
                : $"{audit.Function.RegistersPerThread}+{secondaryAudit.Function.RegistersPerThread}/" +
                  $"{Math.Max(audit.Function.StaticSharedBytes, secondaryAudit.Function.StaticSharedBytes)}/" +
                  $"{Math.Max(audit.Function.LocalBytesPerThread, secondaryAudit.Function.LocalBytesPerThread)}/" +
                  $"{Math.Min(audit.ActiveBlocksPerMultiprocessor, secondaryAudit.ActiveBlocksPerMultiprocessor)}";
        string screen = method == "Direct PTX"
            ? advanceToCompetitor ? "COMPARE" : "HOLD"
            : "BASE";
        Console.WriteLine(
            $"{run,3} {rows,5} {name,-25} {method,-10} " +
            $"{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2} " +
            $"{e2e.Median,6:F2}/{e2e.P95,6:F2}/{e2e.P99,6:F2} " +
            $"{allocation,6} {temporaryBytes,4} {persistentWorkspaceBytes,8} " +
            $"{error,8:E1} {speedup,5:F2}x " +
            $"{screen,7} {resources}");
    }

    private static float MaxError(
        CudaBackend backend,
        params (IGpuBuffer Baseline, IGpuBuffer Direct)[] pairs)
    {
        float maximum = 0f;
        foreach ((IGpuBuffer baseline, IGpuBuffer direct) in pairs)
        {
            float[] expected = backend.DownloadBuffer(baseline);
            float[] actual = backend.DownloadBuffer(direct);
            for (int i = 0; i < expected.Length; i++)
                maximum = PtxCompat.Max(maximum, PtxCompat.Abs(actual[i] - expected[i]));
        }
        return maximum;
    }

    private static float MaxHalfError(
        CudaBackend backend, IGpuBuffer baseline, IGpuBuffer direct, int elements)
    {
        byte[] expected = backend.DownloadBytes(baseline, checked(elements * 2));
        byte[] actual = backend.DownloadBytes(direct, checked(elements * 2));
        float maximum = 0f;
        for (int i = 0; i < elements; i++)
        {
            ushort expectedBits = (ushort)(expected[i * 2] | expected[i * 2 + 1] << 8);
            ushort actualBits = (ushort)(actual[i * 2] | actual[i * 2 + 1] << 8);
            float difference = PtxCompat.Abs(
                (float)AiDotNet.Tensors.NumericOperations.HalfBits.FromBits(actualBits) -
                (float)AiDotNet.Tensors.NumericOperations.HalfBits.FromBits(expectedBits));
            maximum = PtxCompat.Max(maximum, difference);
        }
        return maximum;
    }

    private static void GroupNormStats(
        float[] input, int batch, int channels, int groups, int spatial,
        out float[] mean, out float[] variance)
    {
        int channelsPerGroup = channels / groups;
        int valuesPerGroup = checked(channelsPerGroup * spatial);
        mean = new float[batch * groups];
        variance = new float[batch * groups];
        for (int n = 0; n < batch; n++)
        {
            for (int group = 0; group < groups; group++)
            {
                double sum = 0d;
                double sumSquares = 0d;
                int firstChannel = group * channelsPerGroup;
                for (int groupChannel = 0; groupChannel < channelsPerGroup; groupChannel++)
                {
                    int channel = firstChannel + groupChannel;
                    int offset = (n * channels + channel) * spatial;
                    for (int element = 0; element < spatial; element++)
                    {
                        double value = input[offset + element];
                        sum += value;
                        sumSquares += value * value;
                    }
                }
                int stat = n * groups + group;
                double average = sum / valuesPerGroup;
                mean[stat] = (float)average;
                variance[stat] = (float)PtxCompat.Max(
                    0f, (float)(sumSquares / valuesPerGroup - average * average));
            }
        }
    }

    private static byte[] HalfBytes(float[] values)
    {
        var bytes = new byte[checked(values.Length * 2)];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = AiDotNet.Tensors.NumericOperations.HalfBits.GetBits((Half)values[i]);
            bytes[i * 2] = (byte)bits;
            bytes[i * 2 + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    private static float[] Values(int count, int seed, float magnitude)
    {
        Random random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = ((float)random.NextDouble() * 2f - 1f) * magnitude;
        return values;
    }
}
