using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Full issue-#849 resident NVIDIA experiment. Every row is a complete public
/// operation, never an isolated sub-kernel timed against a composed peer.
/// </summary>
internal static class DirectPtxRngStochasticExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 10;
    private const ulong Seed = 0x8490_1234_5678_9ABCul;
    private const float DropoutRate = 0.1f;
    private const float Keep = 1.0f - DropoutRate;
    private const float Temperature = 0.7f;

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    private readonly record struct Cell(
        int Run,
        string Operation,
        string Shape,
        string Method,
        Distribution Device,
        Distribution EndToEnd,
        double GbPerSecond,
        long ManagedBytes,
        long TemporaryDeviceBytes,
        double MaximumSemanticError,
        DirectPtxKernelAudit? Audit);
    private sealed record PythonCell(
        string Status,
        int Run,
        string Operation,
        string Shape,
        string Method,
        Distribution Device,
        Distribution EndToEnd,
        double GbPerSecond,
        long PeakDeviceBytes,
        double MaximumSemanticError,
        string TorchVersion,
        string CudaVersion,
        string DeviceName,
        string Error);
    private readonly record struct GroupKey(int Run, string Operation, string Shape);

    internal static void Run(int independentRuns = 3, bool includeExternal = true)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-rng-stochastic-start");
        Console.WriteLine(
            $"Direct PTX stochastic suite: {independentRuns} clean run(s), {Warmups} warmups + " +
            $"{Samples} samples/cell; {DeviceLaunches} resident launches/device sample.");

        var cells = new List<Cell>();
        for (int run = 1; run <= independentRuns; run++)
        {
            (CudaBackend direct, CudaBackend established) = CreateIsolatedBackends();
            using (direct)
            using (established)
            {
                if (run == 1) Console.WriteLine($"GPU: {direct.DeviceName}");
                RunVectorFamilies(cells, direct, established, run);
                RunRowFamilies(cells, direct, established, run);
                RunImportanceFamilies(cells, direct, established, run);
                RunBiasDropoutFamilies(cells, direct, established, run);
            }
        }

        IReadOnlyList<PythonCell> python = includeExternal
            ? RunPython(independentRuns)
            : Array.Empty<PythonCell>();
        Print(cells, python);
        GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-rng-stochastic-end");
    }

    private static (CudaBackend Direct, CudaBackend Established) CreateIsolatedBackends()
    {
        CudaBackend direct;
        CudaBackend established;
        try
        {
            DirectPtxFeatureGate.TestOverride = true;
            direct = new CudaBackend();
            DirectPtxFeatureGate.TestOverride = false;
            established = new CudaBackend();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = null;
        }

        if (!direct.IsDirectPtxRngDropoutEnabled)
        {
            direct.Dispose();
            established.Dispose();
            throw new InvalidOperationException(
                "The exact admitted SM is required for the direct-PTX contender.");
        }
        if (established.IsDirectPtxRngDropoutEnabled)
        {
            direct.Dispose();
            established.Dispose();
            throw new InvalidOperationException(
                "The AiDotNet baseline backend unexpectedly admitted direct PTX.");
        }
        return (direct, established);
    }

    private static void RunVectorFamilies(
        List<Cell> cells,
        CudaBackend direct,
        CudaBackend established,
        int run)
    {
        foreach (int elements in new[] { 4_096, 65_536, 1_048_576 })
        {
            string shape = $"N={elements}";
            RunUniform(cells, direct, established, run, shape, elements);
            RunNormal(cells, direct, established, run, shape, elements, gaussianNoise: false);
            RunNormal(cells, direct, established, run, shape, elements, gaussianNoise: true);
            RunDropoutMask(cells, direct, established, run, shape, elements, stateless: false);
            RunDropoutMask(cells, direct, established, run, shape, elements, stateless: true);
            RunDropoutForward(cells, direct, established, run, shape, elements);
            RunDropoutBackward(cells, direct, established, run, shape, elements);
            RunDdim(cells, direct, established, run, shape, elements);
            RunRrelu(cells, direct, established, run, shape, elements);
        }
    }

    private static void RunDropoutForward(
        List<Cell> cells,
        CudaBackend direct,
        CudaBackend established,
        int run,
        string shape,
        int elements)
    {
        float[] input = Enumerable.Range(0, elements)
            .Select(index => (index % 257 - 128) / 128.0f).ToArray();
        using var directInput = direct.AllocateBuffer(input);
        using var directOutput = direct.AllocateBuffer(elements);
        using var directMask = direct.AllocateBuffer(elements);
        using var establishedInput = established.AllocateBuffer(input);
        using var establishedOutput = established.AllocateBuffer(elements);
        using var establishedMask = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxRngDropoutF32(elements), direct,
            "dropout forward prewarm");
        Require(direct.TryGetDirectPtxRngDropoutAudit(
            elements, out DirectPtxKernelAudit audit), direct, "dropout forward audit");

        void Direct() => direct.Dropout(
            directInput, directOutput, directMask, elements, DropoutRate, Seed, training: true);
        void Established() => established.Dropout(
            establishedInput, establishedOutput, establishedMask,
            elements, DropoutRate, Seed, training: true);
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        float[] directMaskHost = direct.DownloadBuffer(directMask);
        uint keepThreshold = (uint)Math.Floor((double)Keep * 4_294_967_296.0);
        float[] maskOracle = PhiloxMaskOracle(
            elements, keepThreshold, 1.0f / Keep, dropWhenLess: false);
        double directError = Math.Max(
            MaximumError(directMaskHost, maskOracle),
            MaximumError(
                direct.DownloadBuffer(directOutput),
                input.Select((value, index) => value * maskOracle[index]).ToArray()));
        double establishedError = DropoutForwardError(
            established.DownloadBuffer(establishedOutput),
            established.DownloadBuffer(establishedMask), input);
        AddPair(cells, direct, established, run, "dropout-forward", shape,
            Direct, Established, usefulBytes: 12L * elements,
            directError, establishedError, audit,
            establishedTemporaryBytes: 0);
    }

    private static void RunRrelu(
        List<Cell> cells,
        CudaBackend direct,
        CudaBackend established,
        int run,
        string shape,
        int elements)
    {
        const float lower = 0.125f;
        const float upper = 1.0f / 3.0f;
        float[] input = Enumerable.Range(0, elements)
            .Select(index => (index % 257 - 128) / 128.0f).ToArray();
        float[] savedNoise = Enumerable.Range(0, elements)
            .Select(index => lower + (upper - lower) * ((index % 251) / 250.0f)).ToArray();
        float[] gradient = Enumerable.Range(0, elements)
            .Select(index => (index % 193 - 96) / 96.0f).ToArray();

        using var directInput = direct.AllocateBuffer(input);
        using var directNoise = direct.AllocateBuffer(elements);
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedInput = established.AllocateBuffer(input);
        using var establishedNoise = established.AllocateBuffer(elements);
        using var establishedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxFusedRreluF32(elements), direct,
            "fused RReLU prewarm");
        Require(direct.TryGetDirectPtxFusedRreluAudit(
            elements, out DirectPtxKernelAudit fusedAudit), direct, "fused RReLU audit");

        void DirectFused() => Require(direct.TryFusedPhiloxRRelu(
            directInput, directNoise, directOutput, elements,
            lower, upper, Seed), direct, "fused RReLU dispatch");
        void EstablishedFused()
        {
            established.GenerateRandomUniform(
                establishedNoise, elements, lower, upper, Seed);
            established.RRelu(
                establishedInput, establishedNoise, establishedOutput, elements);
        }
        DirectFused();
        EstablishedFused();
        direct.Synchronize();
        established.Synchronize();
        float[] directNoiseHost = direct.DownloadBuffer(directNoise);
        float[] rreluNoiseOracle = PhiloxUniformOracle(elements, lower, upper);
        float[] rreluOutputOracle = input.Select((value, index) =>
            value >= 0f ? value : value * rreluNoiseOracle[index]).ToArray();
        double directFusedError = Math.Max(
            MaximumError(directNoiseHost, rreluNoiseOracle),
            MaximumError(direct.DownloadBuffer(directOutput), rreluOutputOracle));
        double establishedFusedError = RreluError(
            established.DownloadBuffer(establishedOutput),
            established.DownloadBuffer(establishedNoise), input, lower, upper);
        AddPair(cells, direct, established, run, "rrelu-training", shape,
            DirectFused, EstablishedFused, usefulBytes: 12L * elements,
            directFusedError, establishedFusedError, fusedAudit,
            establishedTemporaryBytes: 0);

        using var directSavedNoise = direct.AllocateBuffer(savedNoise);
        using var directSavedOutput = direct.AllocateBuffer(elements);
        using var establishedSavedNoise = established.AllocateBuffer(savedNoise);
        using var establishedSavedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxRreluF32(DirectPtxRreluKind.Forward, elements),
            direct, "saved-noise RReLU forward prewarm");
        Require(direct.TryGetDirectPtxRreluAudit(
            DirectPtxRreluKind.Forward, elements, out DirectPtxKernelAudit forwardAudit),
            direct, "saved-noise RReLU forward audit");
        void DirectForward() => direct.RRelu(
            directInput, directSavedNoise, directSavedOutput, elements);
        void EstablishedForward() => established.RRelu(
            establishedInput, establishedSavedNoise, establishedSavedOutput, elements);
        DirectForward();
        EstablishedForward();
        direct.Synchronize();
        established.Synchronize();
        double directForwardError = RreluError(
            direct.DownloadBuffer(directSavedOutput), savedNoise, input, lower, upper);
        double establishedForwardError = RreluError(
            established.DownloadBuffer(establishedSavedOutput), savedNoise, input, lower, upper);
        AddPair(cells, direct, established, run, "rrelu-saved-noise-forward", shape,
            DirectForward, EstablishedForward, usefulBytes: 12L * elements,
            directForwardError, establishedForwardError, forwardAudit,
            establishedTemporaryBytes: 0);

        using var directGradient = direct.AllocateBuffer(gradient);
        using var directGradientInput = direct.AllocateBuffer(elements);
        using var establishedGradient = established.AllocateBuffer(gradient);
        using var establishedGradientInput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxRreluF32(DirectPtxRreluKind.Backward, elements),
            direct, "saved-noise RReLU backward prewarm");
        Require(direct.TryGetDirectPtxRreluAudit(
            DirectPtxRreluKind.Backward, elements, out DirectPtxKernelAudit backwardAudit),
            direct, "saved-noise RReLU backward audit");
        void DirectBackward() => direct.RReluBackward(
            directGradient, directInput, directSavedNoise, directGradientInput, elements);
        void EstablishedBackward() => established.RReluBackward(
            establishedGradient, establishedInput, establishedSavedNoise,
            establishedGradientInput, elements);
        DirectBackward();
        EstablishedBackward();
        direct.Synchronize();
        established.Synchronize();
        float[] backwardOracle = new float[elements];
        for (int index = 0; index < elements; index++)
            backwardOracle[index] = gradient[index] * (input[index] >= 0f ? 1f : savedNoise[index]);
        double directBackwardError = MaximumError(
            direct.DownloadBuffer(directGradientInput), backwardOracle);
        double establishedBackwardError = MaximumError(
            established.DownloadBuffer(establishedGradientInput), backwardOracle);
        AddPair(cells, direct, established, run, "rrelu-backward", shape,
            DirectBackward, EstablishedBackward, usefulBytes: 16L * elements,
            directBackwardError, establishedBackwardError, backwardAudit,
            establishedTemporaryBytes: 0);
    }

    private static void RunUniform(
        List<Cell> cells, CudaBackend direct, CudaBackend established,
        int run, string shape, int elements)
    {
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxRngFillF32(
            DirectPtxPhiloxFillKind.Uniform, elements), direct, "uniform prewarm");
        Require(direct.TryGetDirectPtxRngFillAudit(
            DirectPtxPhiloxFillKind.Uniform, elements, out DirectPtxKernelAudit audit),
            direct, "uniform audit");

        void Direct() => direct.GenerateRandomUniform(
            directOutput, elements, -1.0f, 1.0f, Seed);
        void Established() => established.GenerateRandomUniform(
            establishedOutput, elements, -1.0f, 1.0f, Seed);
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        double directError = MaximumError(
            direct.DownloadBuffer(directOutput),
            PhiloxUniformOracle(elements, -1.0f, 1.0f));
        double establishedError = RangeViolation(
            established.DownloadBuffer(establishedOutput), -1f, 1f);
        AddPair(cells, direct, established, run, "uniform", shape,
            Direct, Established, usefulBytes: 4L * elements,
            directError, establishedError, audit, establishedTemporaryBytes: 0);
    }

    private static void RunNormal(
        List<Cell> cells, CudaBackend direct, CudaBackend established,
        int run, string shape, int elements, bool gaussianNoise)
    {
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxRngFillF32(
            DirectPtxPhiloxFillKind.Normal, elements), direct, "normal prewarm");
        Require(direct.TryGetDirectPtxRngFillAudit(
            DirectPtxPhiloxFillKind.Normal, elements, out DirectPtxKernelAudit audit),
            direct, "normal audit");
        const float mean = 0.25f;
        const float stdDev = 1.5f;

        void Direct()
        {
            if (gaussianNoise)
                direct.GaussianNoise(directOutput, elements, mean, stdDev, Seed);
            else
                direct.GenerateRandomNormal(directOutput, elements, mean, stdDev, Seed);
        }
        void Established()
        {
            if (gaussianNoise)
                established.GaussianNoise(establishedOutput, elements, mean, stdDev, Seed);
            else
                established.GenerateRandomNormal(establishedOutput, elements, mean, stdDev, Seed);
        }
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        double directError = MaximumError(
            direct.DownloadBuffer(directOutput),
            PhiloxNormalOracle(elements, mean, stdDev));
        double establishedError = NormalMomentError(
            established.DownloadBuffer(establishedOutput), mean, stdDev);
        AddPair(cells, direct, established, run,
            gaussianNoise ? "gaussian-noise" : "normal", shape,
            Direct, Established, usefulBytes: 4L * elements,
            directError, establishedError, audit, establishedTemporaryBytes: 0);
    }

    private static void RunDropoutMask(
        List<Cell> cells, CudaBackend direct, CudaBackend established,
        int run, string shape, int elements, bool stateless)
    {
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedOutput = established.AllocateBuffer(elements);
        DirectPtxPhiloxFillKind kind = stateless
            ? DirectPtxPhiloxFillKind.DropThresholdMask
            : DirectPtxPhiloxFillKind.BernoulliMask;
        Require(direct.PrewarmDirectPtxRngFillF32(kind, elements), direct, "mask prewarm");
        Require(direct.TryGetDirectPtxRngFillAudit(kind, elements, out DirectPtxKernelAudit audit),
            direct, "mask audit");
        double thresholdProbability = stateless ? DropoutRate : Keep;
        uint threshold = (uint)Math.Floor(thresholdProbability * 4_294_967_296.0);
        float scale = 1.0f / Keep;

        void Direct()
        {
            if (stateless)
                direct.GenerateStatelessDropoutMask(
                    directOutput, elements, threshold, scale, unchecked((uint)Seed));
            else
                direct.DropoutMask(directOutput, elements, Keep, Seed);
        }
        void Established()
        {
            if (stateless)
                established.GenerateStatelessDropoutMask(
                    establishedOutput, elements, threshold, scale, unchecked((uint)Seed));
            else
                established.DropoutMask(establishedOutput, elements, Keep, Seed);
        }
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        double directError = MaximumError(
            direct.DownloadBuffer(directOutput),
            PhiloxMaskOracle(elements, threshold, scale, dropWhenLess: stateless));
        double establishedError = BinaryScaleViolation(
            established.DownloadBuffer(establishedOutput), scale);
        AddPair(cells, direct, established, run,
            stateless ? "stateless-dropout-mask" : "dropout-mask", shape,
            Direct, Established, usefulBytes: 4L * elements,
            directError, establishedError, audit, establishedTemporaryBytes: 0);
    }

    private static void RunDropoutBackward(
        List<Cell> cells, CudaBackend direct, CudaBackend established,
        int run, string shape, int elements)
    {
        float[] gradient = Enumerable.Range(0, elements)
            .Select(index => (index % 257 - 128) / 128.0f).ToArray();
        float[] mask = Enumerable.Range(0, elements)
            .Select(index => index % 10 == 0 ? 0.0f : 1.0f / Keep).ToArray();
        using var directGradient = direct.AllocateBuffer(gradient);
        using var directMask = direct.AllocateBuffer(mask);
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedGradient = established.AllocateBuffer(gradient);
        using var establishedMask = established.AllocateBuffer(mask);
        using var establishedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxDropoutBackwardF32(elements), direct, "dropout backward prewarm");
        Require(direct.TryGetDirectPtxDropoutBackwardAudit(elements, out DirectPtxKernelAudit audit),
            direct, "dropout backward audit");

        void Direct() => direct.DropoutBackward(
            directGradient, directMask, directOutput, elements, DropoutRate);
        void Established() => established.DropoutBackward(
            establishedGradient, establishedMask, establishedOutput, elements, DropoutRate);
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        float[] oracle = new float[elements];
        for (int i = 0; i < elements; i++) oracle[i] = gradient[i] * mask[i];
        double directError = MaximumError(direct.DownloadBuffer(directOutput), oracle);
        double establishedError = MaximumError(
            established.DownloadBuffer(establishedOutput), oracle);
        AddPair(cells, direct, established, run, "dropout-backward", shape,
            Direct, Established, usefulBytes: 12L * elements,
            directError, establishedError, audit, establishedTemporaryBytes: 0);
    }

    private static void RunDdim(
        List<Cell> cells, CudaBackend direct, CudaBackend established,
        int run, string shape, int elements)
    {
        float[] x = Enumerable.Range(0, elements)
            .Select(index => (index % 251 - 125) / 125.0f).ToArray();
        float[] epsilon = Enumerable.Range(0, elements)
            .Select(index => (index % 127 - 63) / 63.0f).ToArray();
        using var directX = direct.AllocateBuffer(x);
        using var directEpsilon = direct.AllocateBuffer(epsilon);
        using var directOutput = direct.AllocateBuffer(elements);
        using var establishedX = established.AllocateBuffer(x);
        using var establishedEpsilon = established.AllocateBuffer(epsilon);
        using var establishedOutput = established.AllocateBuffer(elements);
        Require(direct.PrewarmDirectPtxFusedDdimStepF32(elements), direct, "DDIM prewarm");
        Require(direct.TryGetDirectPtxFusedDdimStepAudit(elements, out DirectPtxKernelAudit audit),
            direct, "DDIM audit");
        const float alphaT = 0.8f;
        const float alphaPrevious = 0.9f;

        void Direct() => direct.FusedDDIMStep(
            directX, directEpsilon, directOutput, elements, alphaT, alphaPrevious);
        void Established() => established.FusedDDIMStep(
            establishedX, establishedEpsilon, establishedOutput, elements,
            alphaT, alphaPrevious);
        Direct();
        Established();
        direct.Synchronize();
        established.Synchronize();
        double sqrtT = Math.Sqrt(alphaT);
        double sqrtPrevious = Math.Sqrt(alphaPrevious);
        double xCoefficient = sqrtPrevious / sqrtT;
        double epsilonCoefficient = Math.Sqrt(1.0 - alphaPrevious) -
            Math.Sqrt(1.0 - alphaT) * xCoefficient;
        var oracle = new float[elements];
        for (int i = 0; i < elements; i++)
            oracle[i] = (float)(xCoefficient * x[i] + epsilonCoefficient * epsilon[i]);
        double directError = MaximumError(direct.DownloadBuffer(directOutput), oracle);
        double establishedError = MaximumError(
            established.DownloadBuffer(establishedOutput), oracle);
        AddPair(cells, direct, established, run, "ddim-step", shape,
            Direct, Established, usefulBytes: 12L * elements,
            directError, establishedError, audit, establishedTemporaryBytes: 0);
    }

    private static void RunRowFamilies(
        List<Cell> cells, CudaBackend direct, CudaBackend established, int run)
    {
        foreach (int rows in new[] { 128, 2_048, 32_768 })
        {
            int classes = PtxFusedGumbelSoftmax32F32Kernel.InnerSize;
            int elements = checked(rows * classes);
            string shape = $"[{rows},32]";
            float[] logits = Enumerable.Range(0, elements)
                .Select(index => (index % classes - 15.5f) / 8.0f).ToArray();
            using var directLogits = direct.AllocateBuffer(logits);
            using var directSoft = direct.AllocateBuffer(elements);
            using var establishedLogits = established.AllocateBuffer(logits);
            using var establishedSoft = established.AllocateBuffer(elements);
            Require(direct.PrewarmDirectPtxGumbelSoftmaxF32(rows, classes), direct, "Gumbel prewarm");
            Require(direct.TryGetDirectPtxGumbelSoftmaxAudit(rows, out DirectPtxKernelAudit gumbelAudit),
                direct, "Gumbel audit");

            void DirectGumbel() => direct.GumbelSoftmax(
                directLogits, directSoft, rows, classes, Temperature, Seed);
            void EstablishedGumbel() => established.GumbelSoftmax(
                establishedLogits, establishedSoft, rows, classes, Temperature, Seed);
            DirectGumbel();
            EstablishedGumbel();
            direct.Synchronize();
            established.Synchronize();
            double directGumbelError = MaximumError(
                direct.DownloadBuffer(directSoft),
                GumbelSoftmaxOracle(logits, rows, classes, Temperature));
            double establishedGumbelError = SimplexViolation(
                established.DownloadBuffer(establishedSoft), rows, classes);
            AddPair(cells, direct, established, run, "gumbel-softmax", shape,
                DirectGumbel, EstablishedGumbel, usefulBytes: 8L * elements,
                directGumbelError, establishedGumbelError, gumbelAudit,
                establishedTemporaryBytes: 0);

            float[] probabilities = Enumerable.Repeat(1.0f / classes, elements).ToArray();
            using var directProbabilities = direct.AllocateBuffer(probabilities);
            using var directOneHot = direct.AllocateBuffer(elements);
            using var establishedProbabilities = established.AllocateBuffer(probabilities);
            using var establishedLogProbabilities = established.AllocateBuffer(elements);
            using var establishedCategoricalSoft = established.AllocateBuffer(elements);
            using var establishedSelectedIndices = established.AllocateBuffer(rows);
            using var establishedOneHot = established.AllocateBuffer(elements);
            Require(direct.PrewarmDirectPtxCategoricalF32(rows, classes), direct,
                "categorical prewarm");
            Require(direct.TryGetDirectPtxCategoricalAudit(rows, out DirectPtxKernelAudit categoricalAudit),
                direct, "categorical audit");
            void DirectCategorical()
            {
                Require(direct.TryCategoricalSample(
                    directProbabilities, directOneHot, rows, classes, Seed),
                    direct, "categorical dispatch");
            }
            void EstablishedCategorical()
            {
                established.Log(
                    establishedProbabilities, establishedLogProbabilities, elements);
                established.GumbelSoftmax(
                    establishedLogProbabilities, establishedCategoricalSoft,
                    rows, classes, 1f, Seed);
                established.ArgMaxAxis(
                    establishedCategoricalSoft, establishedSelectedIndices, rows, classes);
                established.OneHotKernel(
                    establishedSelectedIndices, establishedOneHot, rows, classes);
            }
            DirectCategorical();
            EstablishedCategorical();
            direct.Synchronize();
            established.Synchronize();
            double directCategoricalError = MaximumError(
                direct.DownloadBuffer(directOneHot),
                UniformCategoricalOracle(rows, classes));
            double establishedCategoricalError = OneHotViolation(
                established.DownloadBuffer(establishedOneHot), rows, classes);
            AddPair(cells, direct, established, run, "categorical-one-hot", shape,
                DirectCategorical, EstablishedCategorical, usefulBytes: 8L * elements,
                directCategoricalError, establishedCategoricalError, categoricalAudit,
                establishedTemporaryBytes: checked(8L * elements + 4L * rows));

            float[] grad = Enumerable.Range(0, elements)
                .Select(index => (index % classes - 15.5f) / 16.0f).ToArray();
            float[] soft = Enumerable.Repeat(1.0f / classes, elements).ToArray();
            using var directGrad = direct.AllocateBuffer(grad);
            using var directSoftInput = direct.AllocateBuffer(soft);
            using var directGradInput = direct.AllocateBuffer(elements);
            using var establishedGrad = established.AllocateBuffer(grad);
            using var establishedSoftInput = established.AllocateBuffer(soft);
            using var establishedTemporary = established.AllocateBuffer(elements);
            using var establishedGradInput = established.AllocateBuffer(elements);
            Require(direct.PrewarmDirectPtxGumbelSoftmaxBackwardF32(rows, classes), direct,
                "Gumbel backward prewarm");
            Require(direct.TryGetDirectPtxGumbelSoftmaxBackwardAudit(
                rows, out DirectPtxKernelAudit backwardAudit), direct, "Gumbel backward audit");

            void DirectBackward()
            {
                Require(direct.TryGumbelSoftmaxBackward(
                    directGrad, directSoftInput, directGradInput,
                    rows, classes, Temperature), direct, "Gumbel backward dispatch");
            }
            void EstablishedBackward()
            {
                established.SoftmaxBackward(
                    establishedGrad, establishedSoftInput, establishedTemporary, rows, classes);
                established.Scale(
                    establishedTemporary, establishedGradInput, 1.0f / Temperature, elements);
            }
            DirectBackward();
            EstablishedBackward();
            direct.Synchronize();
            established.Synchronize();
            float[] backwardOracle = SoftmaxBackwardOracle(grad, soft, rows, classes, Temperature);
            double directBackwardError = MaximumError(
                direct.DownloadBuffer(directGradInput), backwardOracle);
            double establishedBackwardError = MaximumError(
                established.DownloadBuffer(establishedGradInput), backwardOracle);
            AddPair(cells, direct, established, run, "gumbel-softmax-backward", shape,
                DirectBackward, EstablishedBackward, usefulBytes: 12L * elements,
                directBackwardError, establishedBackwardError, backwardAudit,
                establishedTemporaryBytes: 4L * elements);
        }
    }

    private static void RunImportanceFamilies(
        List<Cell> cells, CudaBackend direct, CudaBackend established, int run)
    {
        foreach (int rays in new[] { 64, 1_024, 16_384 })
        {
            const int samples = PtxFusedImportanceSampling64F32Kernel.Samples;
            int elements = checked(rays * samples);
            string shape = $"[{rays},64,64]";
            float[] tValues = Enumerable.Range(0, elements)
                .Select(index => (index % samples) / 63.0f).ToArray();
            float[] weights = Enumerable.Range(0, elements)
                .Select(index => 0.25f + (index % samples) / 64.0f).ToArray();
            using var directT = direct.AllocateBuffer(tValues);
            using var directWeights = direct.AllocateBuffer(weights);
            using var directOutput = direct.AllocateBuffer(elements);
            using var establishedT = established.AllocateBuffer(tValues);
            using var establishedWeights = established.AllocateBuffer(weights);
            using var establishedOutput = established.AllocateBuffer(elements);
            Require(direct.PrewarmDirectPtxImportanceSamplingF32(rays, samples, samples),
                direct, "importance prewarm");
            Require(direct.TryGetDirectPtxImportanceSamplingAudit(
                rays, out DirectPtxKernelAudit audit), direct, "importance audit");

            void Direct() => direct.ImportanceSampling(
                directT, directWeights, directOutput, rays, samples, samples,
                unchecked((uint)Seed));
            void Established() => established.ImportanceSampling(
                establishedT, establishedWeights, establishedOutput,
                rays, samples, samples, unchecked((uint)Seed));
            Direct();
            Established();
            direct.Synchronize();
            established.Synchronize();
            double directError = MaximumError(
                direct.DownloadBuffer(directOutput),
                ImportanceSamplingOracle(tValues, weights, rays, samples));
            double establishedError = ImportanceRangeViolation(
                established.DownloadBuffer(establishedOutput), rays, samples);
            AddPair(cells, direct, established, run, "importance-sampling", shape,
                Direct, Established, usefulBytes: 12L * elements,
                directError, establishedError, audit, establishedTemporaryBytes: 0);
        }
    }

    private static void RunBiasDropoutFamilies(
        List<Cell> cells, CudaBackend direct, CudaBackend established, int run)
    {
        foreach (int rows in new[] { 16, 256, 4_096 })
        {
            const int cols = PtxFusedBiasPhiloxDropout256F32Kernel.Columns;
            int elements = checked(rows * cols);
            string shape = $"[{rows},256]";
            float[] input = Enumerable.Range(0, elements)
                .Select(index => (index % cols - 127.5f) / 128.0f).ToArray();
            float[] bias = Enumerable.Range(0, cols)
                .Select(index => (index - 127.5f) / 256.0f).ToArray();
            using var directInput = direct.AllocateBuffer(input);
            using var directBias = direct.AllocateBuffer(bias);
            using var directOutput = direct.AllocateBuffer(elements);
            using var directMask = direct.AllocateBuffer(elements);
            using var establishedInput = established.AllocateBuffer(input);
            using var establishedBias = established.AllocateBuffer(bias);
            using var establishedOutput = established.AllocateBuffer(elements);
            using var establishedMask = established.AllocateBuffer(elements);
            Require(direct.PrewarmDirectPtxBiasDropoutF32(rows, cols), direct,
                "bias-dropout prewarm");
            Require(direct.TryGetDirectPtxBiasDropoutAudit(rows, out DirectPtxKernelAudit audit),
                direct, "bias-dropout audit");

            void Direct()
            {
                Require(direct.TryFusedBiasPhiloxDropout(
                    directInput, directOutput, directBias, directMask,
                    rows, cols, DropoutRate, Seed), direct, "bias-dropout dispatch");
            }
            void Established()
            {
                established.DropoutMask(establishedMask, elements, Keep, Seed);
                Require(established.TryFusedBiasDropout(
                    establishedInput, establishedOutput, establishedBias, establishedMask,
                    rows, cols, DropoutRate, 1.0f / Keep), established,
                    "established bias-dropout dispatch");
            }
            Direct();
            Established();
            direct.Synchronize();
            established.Synchronize();
            float[] directMaskHost = direct.DownloadBuffer(directMask);
            uint keepThreshold = (uint)Math.Floor((double)Keep * 4_294_967_296.0);
            float[] directMaskOracle = PhiloxMaskOracle(
                elements, keepThreshold, 1.0f / Keep, dropWhenLess: false);
            float[] directOutputOracle = new float[elements];
            for (int index = 0; index < elements; index++)
                directOutputOracle[index] =
                    (input[index] + bias[index % cols]) * directMaskOracle[index];
            double directError = Math.Max(
                MaximumError(directMaskHost, directMaskOracle),
                MaximumError(direct.DownloadBuffer(directOutput), directOutputOracle));
            double establishedError = BiasDropoutError(
                established.DownloadBuffer(establishedOutput),
                established.DownloadBuffer(establishedMask), input, bias, rows, cols,
                maskIncludesScale: false);
            AddPair(cells, direct, established, run, "bias-dropout", shape,
                Direct, Established, usefulBytes: 12L * elements + 4L * cols,
                directError, establishedError, audit, establishedTemporaryBytes: 0);
        }
    }

    private static void AddPair(
        List<Cell> cells,
        CudaBackend directBackend,
        CudaBackend establishedBackend,
        int run,
        string operation,
        string shape,
        Action direct,
        Action established,
        long usefulBytes,
        double directError,
        double establishedError,
        DirectPtxKernelAudit audit,
        long establishedTemporaryBytes)
    {
        IntPtr graph = directBackend.CaptureGraph(direct);
        if (graph == IntPtr.Zero)
            throw new InvalidOperationException($"Could not capture prewarmed {operation} direct PTX.");
        try
        {
            cells.Add(Measure(directBackend, run, operation, shape,
                "Direct PTX CUDA graph", () => directBackend.EnqueueCapturedGraph(graph),
                usefulBytes, directError, audit, temporaryDeviceBytes: 0));
            cells.Add(Measure(directBackend, run, operation, shape,
                "Direct PTX", direct, usefulBytes, directError, audit,
                temporaryDeviceBytes: 0));
            cells.Add(Measure(establishedBackend, run, operation, shape,
                "AiDotNet established CUDA", established, usefulBytes,
                establishedError, null, establishedTemporaryBytes));
        }
        finally
        {
            directBackend.DestroyCapturedGraph(graph);
        }
    }

    private static Cell Measure(
        CudaBackend backend,
        int run,
        string operation,
        string shape,
        string method,
        Action action,
        long usefulBytes,
        double maximumError,
        DirectPtxKernelAudit? audit,
        long temporaryDeviceBytes)
    {
        Distribution device = MeasureDevice(backend, action);
        Distribution endToEnd = MeasureEndToEnd(backend, action);
        long managedBytes = MeasureAllocation(backend, action);
        double gbps = usefulBytes / (device.Median * 1e-6) / 1e9;
        return new Cell(run, operation, shape, method, device, endToEnd,
            gbps, managedBytes, temporaryDeviceBytes, maximumError, audit);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < Samples; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < DeviceLaunches; launch++) action();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            values[sample] = backend.GetEventElapsedTime(start, end) * 1_000.0 / DeviceLaunches;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        double tickUs = 1_000_000.0 / Stopwatch.Frequency;
        for (int sample = 0; sample < Samples; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * tickUs;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
#if NET6_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
#else
        long bytes = -1;
        for (int i = 0; i < Samples; i++) action();
#endif
        backend.Synchronize();
        return bytes;
    }

    private static IReadOnlyList<PythonCell> RunPython(int runs)
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_rng_stochastic_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(
                AppContext.BaseDirectory, "run_direct_ptx_rng_stochastic_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("PyTorch stochastic peer is missing.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--runs");
        start.ArgumentList.Add(runs.ToString(CultureInfo.InvariantCulture));
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch CUDA stochastic peer.");
        var cells = new List<PythonCell>();
        while (process.StandardOutput.ReadLine() is { } line)
        {
            using JsonDocument json = JsonDocument.Parse(line);
            JsonElement root = json.RootElement;
            string status = ReadString(root, "status");
            if (status == "ok")
            {
                cells.Add(new PythonCell(
                    status,
                    ReadInt(root, "run"),
                    ReadString(root, "operation"),
                    ReadString(root, "shape"),
                    ReadString(root, "method"),
                    new Distribution(
                        ReadDouble(root, "device_mean_us"),
                        ReadDouble(root, "device_median_us"),
                        ReadDouble(root, "device_p95_us"),
                        ReadDouble(root, "device_p99_us")),
                    new Distribution(
                        ReadDouble(root, "e2e_mean_us"),
                        ReadDouble(root, "e2e_median_us"),
                        ReadDouble(root, "e2e_p95_us"),
                        ReadDouble(root, "e2e_p99_us")),
                    ReadDouble(root, "gb_per_second"),
                    ReadLong(root, "peak_device_bytes"),
                    ReadDouble(root, "max_error"),
                    ReadString(root, "torch_version"),
                    ReadString(root, "cuda_version"),
                    ReadString(root, "device_name"),
                    string.Empty));
            }
            else
            {
                cells.Add(new PythonCell(
                    status,
                    ReadInt(root, "run"),
                    ReadString(root, "operation"),
                    ReadString(root, "shape"),
                    ReadString(root, "method"),
                    default, default, 0, 0, 0,
                    string.Empty, string.Empty, string.Empty,
                    ReadString(root, "error")));
            }
        }
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch stochastic peer failed ({process.ExitCode}): {error}");
        return cells;
    }

    private static void Print(
        IReadOnlyList<Cell> cells,
        IReadOnlyList<PythonCell> python)
    {
        Console.WriteLine(
            $"environment | dotnet={Environment.Version} | os={Environment.OSVersion} | " +
            $"process={System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}");
        PythonCell? firstPython = python.FirstOrDefault(cell => cell.Status == "ok");
        if (firstPython is not null)
            Console.WriteLine(
                $"external-environment | torch={firstPython.TorchVersion} | " +
                $"cuda={firstPython.CudaVersion} | gpu={firstPython.DeviceName}");
        foreach (DirectPtxKernelAudit audit in cells
            .Where(cell => cell.Audit is not null)
            .Select(cell => cell.Audit!)
            .GroupBy(audit => audit.PtxSha256, StringComparer.Ordinal)
            .Select(group => group.First()))
            Console.WriteLine($"audit | {audit.ToJson()}");
        Console.WriteLine(
            "run | operation | shape | method | device mean/median/p95/p99 us | " +
            "e2e mean/median/p95/p99 us | GB/s | managed B/call | temp device B | " +
            "max semantic error | regs/shared/local/blocks-SM");
        GroupKey[] groups = cells
            .Select(cell => new GroupKey(cell.Run, cell.Operation, cell.Shape))
            .Concat(python.Select(cell => new GroupKey(
                cell.Run, cell.Operation, cell.Shape)))
            .Distinct()
            .OrderBy(key => key.Run)
            .ThenBy(key => key.Operation, StringComparer.Ordinal)
            .ThenBy(key => key.Shape, StringComparer.Ordinal)
            .ToArray();
        foreach (GroupKey group in groups)
        {
            Console.WriteLine($"--- run {group.Run}: {group.Operation} {group.Shape} ---");
            foreach (Cell cell in cells.Where(cell =>
                         cell.Run == group.Run && cell.Operation == group.Operation &&
                         cell.Shape == group.Shape))
            {
                string resources = cell.Audit is null ? "n/a" :
                    $"{cell.Audit.Function.RegistersPerThread}/" +
                    $"{cell.Audit.Function.StaticSharedBytes}/" +
                    $"{cell.Audit.Function.LocalBytesPerThread}/" +
                    $"{cell.Audit.ActiveBlocksPerMultiprocessor}";
                Console.WriteLine(
                    $"{cell.Run} | {cell.Operation} | {cell.Shape} | {cell.Method} | " +
                    $"{Format(cell.Device)} | {Format(cell.EndToEnd)} | {cell.GbPerSecond:F2} | " +
                    $"{cell.ManagedBytes} | {cell.TemporaryDeviceBytes} | " +
                    $"{cell.MaximumSemanticError:E3} | {resources}");
            }
            foreach (PythonCell cell in python.Where(cell =>
                         cell.Status == "ok" && cell.Run == group.Run &&
                         cell.Operation == group.Operation && cell.Shape == group.Shape))
                Console.WriteLine(
                    $"{cell.Run} | {cell.Operation} | {cell.Shape} | {cell.Method} | " +
                    $"{Format(cell.Device)} | {Format(cell.EndToEnd)} | {cell.GbPerSecond:F2} | " +
                    $"n/a | {cell.PeakDeviceBytes} | {cell.MaximumSemanticError:E3} | n/a");
            foreach (PythonCell cell in python.Where(cell =>
                         cell.Status != "ok" && cell.Run == group.Run &&
                         cell.Operation == group.Operation && cell.Shape == group.Shape))
                Console.WriteLine(
                    $"SKIPPED | {cell.Run} | {cell.Operation} | {cell.Shape} | " +
                    $"{cell.Method} | {cell.Error}");
        }
    }

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.50),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] values, double percentile)
    {
        double position = (values.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, values.Length - 1);
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    private static string Format(Distribution value) =>
        $"{value.Mean:F3}/{value.Median:F3}/{value.P95:F3}/{value.P99:F3}";

    private static string ReadString(JsonElement root, string name) =>
        root.TryGetProperty(name, out JsonElement value)
            ? value.GetString() ?? string.Empty
            : string.Empty;
    private static int ReadInt(JsonElement root, string name) =>
        root.GetProperty(name).GetInt32();
    private static long ReadLong(JsonElement root, string name) =>
        root.GetProperty(name).GetInt64();
    private static double ReadDouble(JsonElement root, string name) =>
        root.GetProperty(name).GetDouble();

    private static void Require(bool condition, CudaBackend backend, string operation)
    {
        if (!condition)
            throw new InvalidOperationException(
                $"{operation} failed: {backend.DirectPtxLastError ?? "no diagnostic"}");
    }

    private static double RangeViolation(float[] values, float minimum, float maximum)
    {
        double error = 0;
        foreach (float value in values)
        {
            if (!float.IsFinite(value)) return double.PositiveInfinity;
            if (value < minimum) error = Math.Max(error, minimum - value);
            if (value >= maximum) error = Math.Max(error, value - maximum);
        }
        return error;
    }

    private static double NormalMomentError(float[] values, float mean, float stdDev)
    {
        double observedMean = values.Average(value => (double)value);
        double variance = values.Average(value =>
        {
            double delta = value - observedMean;
            return delta * delta;
        });
        return Math.Max(Math.Abs(observedMean - mean), Math.Abs(Math.Sqrt(variance) - stdDev));
    }

    private static double BinaryScaleViolation(float[] values, float scale)
    {
        double error = 0;
        foreach (float value in values)
            error = Math.Max(error, Math.Min(Math.Abs(value), Math.Abs(value - scale)));
        return error;
    }

    private static double MaximumError(float[] actual, float[] expected)
    {
        if (actual.Length != expected.Length) return double.PositiveInfinity;
        double error = 0;
        for (int i = 0; i < actual.Length; i++)
            error = Math.Max(error, Math.Abs(actual[i] - expected[i]));
        return error;
    }

    private static float[] PhiloxUniformOracle(
        int elements,
        float minimum,
        float maximum)
    {
        var result = new float[elements];
        for (int index = 0; index < elements; index++)
        {
            uint word = PhiloxWordForFloat4Index(index);
            float unit = Math.Min(
                (float)word * BitConverter.Int32BitsToSingle(unchecked((int)0x2F800000)),
                BitConverter.Int32BitsToSingle(unchecked((int)0x3F7FFFFF)));
            result[index] = MathF.FusedMultiplyAdd(unit, maximum - minimum, minimum);
        }
        return result;
    }

    private static float[] PhiloxNormalOracle(
        int elements,
        float mean,
        float standardDeviation)
    {
        var result = new float[elements];
        for (int group = 0; group < elements / 4; group++)
        {
            var words = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(Seed, 0, (ulong)group);
            double u0 = Math.Clamp(words.X0 / 4_294_967_296.0,
                1.0 / 4_294_967_296.0, Math.BitDecrement(1.0));
            double u1 = Math.Clamp(words.X1 / 4_294_967_296.0,
                1.0 / 4_294_967_296.0, Math.BitDecrement(1.0));
            double u2 = Math.Clamp(words.X2 / 4_294_967_296.0,
                1.0 / 4_294_967_296.0, Math.BitDecrement(1.0));
            double u3 = Math.Clamp(words.X3 / 4_294_967_296.0,
                1.0 / 4_294_967_296.0, Math.BitDecrement(1.0));
            double radius0 = Math.Sqrt(-2.0 * Math.Log(u0));
            double radius1 = Math.Sqrt(-2.0 * Math.Log(u2));
            result[group * 4] = (float)(mean + standardDeviation * radius0 * Math.Cos(2.0 * Math.PI * u1));
            result[group * 4 + 1] = (float)(mean + standardDeviation * radius0 * Math.Sin(2.0 * Math.PI * u1));
            result[group * 4 + 2] = (float)(mean + standardDeviation * radius1 * Math.Cos(2.0 * Math.PI * u3));
            result[group * 4 + 3] = (float)(mean + standardDeviation * radius1 * Math.Sin(2.0 * Math.PI * u3));
        }
        return result;
    }

    private static float[] PhiloxMaskOracle(
        int elements,
        uint threshold,
        float scale,
        bool dropWhenLess)
    {
        var result = new float[elements];
        for (int index = 0; index < elements; index++)
        {
            bool less = PhiloxWordForFloat4Index(index) < threshold;
            bool keep = dropWhenLess ? !less : less;
            result[index] = keep ? scale : 0.0f;
        }
        return result;
    }

    private static uint PhiloxWordForFloat4Index(int index)
    {
        var words = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            Seed, 0, checked((ulong)(index / 4)));
        return (index & 3) switch
        {
            0 => words.X0,
            1 => words.X1,
            2 => words.X2,
            _ => words.X3
        };
    }

    private static double PhiloxUnitForCounter(ulong counter)
    {
        uint word = PtxFusedPhiloxDropoutF32Kernel.GenerateUInt4(
            Seed, 0, counter).X0;
        return Math.Clamp(
            word / 4_294_967_296.0,
            1.0 / 4_294_967_296.0,
            Math.BitDecrement(1.0));
    }

    private static float[] GumbelSoftmaxOracle(
        float[] logits,
        int rows,
        int columns,
        float temperature)
    {
        var result = new float[logits.Length];
        var perturbed = new double[columns];
        for (int row = 0; row < rows; row++)
        {
            double maximum = double.NegativeInfinity;
            for (int column = 0; column < columns; column++)
            {
                int index = row * columns + column;
                double uniform = PhiloxUnitForCounter((ulong)index);
                double gumbel = -Math.Log(-Math.Log(uniform));
                perturbed[column] = (logits[index] + gumbel) / temperature;
                maximum = Math.Max(maximum, perturbed[column]);
            }
            double sum = 0;
            for (int column = 0; column < columns; column++)
            {
                perturbed[column] = Math.Exp(perturbed[column] - maximum);
                sum += perturbed[column];
            }
            for (int column = 0; column < columns; column++)
                result[row * columns + column] = (float)(perturbed[column] / sum);
        }
        return result;
    }

    private static float[] UniformCategoricalOracle(int rows, int columns)
    {
        var result = new float[checked(rows * columns)];
        for (int row = 0; row < rows; row++)
        {
            int selected = Math.Min(
                columns - 1,
                (int)(PhiloxUnitForCounter((ulong)row) * columns));
            result[row * columns + selected] = 1.0f;
        }
        return result;
    }

    private static float[] ImportanceSamplingOracle(
        float[] tValues,
        float[] weights,
        int rows,
        int samples)
    {
        var result = new float[checked(rows * samples)];
        var cumulative = new double[samples];
        for (int row = 0; row < rows; row++)
        {
            int rowOffset = row * samples;
            double total = 0;
            for (int sample = 0; sample < samples; sample++)
                total += Math.Max(0.0, weights[rowOffset + sample]);
            if (total <= 1.0e-10)
            {
                double first = tValues[rowOffset];
                double range = tValues[rowOffset + samples - 1] - first;
                for (int sample = 0; sample < samples; sample++)
                {
                    double u = (sample + PhiloxUnitForCounter(
                        checked((ulong)(rowOffset + sample)))) / samples;
                    result[rowOffset + sample] = (float)(first + u * range);
                }
                continue;
            }

            double running = 0;
            for (int sample = 0; sample < samples; sample++)
            {
                running += Math.Max(0.0, weights[rowOffset + sample]) / total;
                cumulative[sample] = running;
            }
            for (int fine = 0; fine < samples; fine++)
            {
                double u = (fine + PhiloxUnitForCounter(
                    checked((ulong)(rowOffset + fine)))) / samples;
                int upper = 0;
                while (upper < samples - 1 && u > cumulative[upper]) upper++;
                if (upper == 0)
                {
                    result[rowOffset + fine] = tValues[rowOffset];
                    continue;
                }
                double lowerCdf = cumulative[upper - 1];
                double upperCdf = cumulative[upper];
                double lowerT = tValues[rowOffset + upper - 1];
                double upperT = tValues[rowOffset + upper];
                double denominator = upperCdf - lowerCdf;
                result[rowOffset + fine] = denominator > 1.0e-10
                    ? (float)(lowerT + (u - lowerCdf) / denominator * (upperT - lowerT))
                    : (float)lowerT;
            }
        }
        return result;
    }

    private static double SimplexViolation(float[] values, int rows, int columns)
    {
        double error = 0;
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
            {
                float value = values[row * columns + column];
                if (!float.IsFinite(value)) return double.PositiveInfinity;
                error = Math.Max(error, Math.Max(0.0, -value));
                sum += value;
            }
            error = Math.Max(error, Math.Abs(sum - 1.0));
        }
        return error;
    }

    private static double OneHotViolation(float[] values, int rows, int columns)
    {
        double error = 0;
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
            {
                float value = values[row * columns + column];
                error = Math.Max(error, Math.Min(Math.Abs(value), Math.Abs(value - 1.0f)));
                sum += value;
            }
            error = Math.Max(error, Math.Abs(sum - 1.0));
        }
        return error;
    }

    private static double ImportanceRangeViolation(float[] values, int rows, int samples)
    {
        double error = 0;
        for (int row = 0; row < rows; row++)
        for (int sample = 0; sample < samples; sample++)
        {
            float value = values[row * samples + sample];
            if (!float.IsFinite(value)) return double.PositiveInfinity;
            error = Math.Max(error, Math.Max(0.0, -value));
            error = Math.Max(error, Math.Max(0.0, value - 1.0));
        }
        return error;
    }

    private static float[] SoftmaxBackwardOracle(
        float[] gradient, float[] soft, int rows, int columns, float temperature)
    {
        var result = new float[gradient.Length];
        for (int row = 0; row < rows; row++)
        {
            double dot = 0;
            for (int column = 0; column < columns; column++)
            {
                int index = row * columns + column;
                dot += gradient[index] * soft[index];
            }
            for (int column = 0; column < columns; column++)
            {
                int index = row * columns + column;
                result[index] = (float)(soft[index] * (gradient[index] - dot) / temperature);
            }
        }
        return result;
    }

    private static double BiasDropoutError(
        float[] output,
        float[] mask,
        float[] input,
        float[] bias,
        int rows,
        int columns,
        bool maskIncludesScale)
    {
        double error = 0;
        for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
        {
            int index = row * columns + column;
            float maskScale = mask[index] == 0.0f
                ? 0.0f
                : maskIncludesScale ? mask[index] : 1.0f / Keep;
            float expected = (input[index] + bias[column]) * maskScale;
            error = Math.Max(error, Math.Abs(output[index] - expected));
        }
        return error;
    }

    private static double RreluError(
        float[] output,
        float[] savedNoise,
        float[] input,
        float lower,
        float upper)
    {
        if (output.Length != input.Length || savedNoise.Length != input.Length)
            return double.PositiveInfinity;
        double error = 0;
        for (int index = 0; index < input.Length; index++)
        {
            float slope = savedNoise[index];
            if (!float.IsFinite(slope) || slope < lower || slope > upper)
                return double.PositiveInfinity;
            float expected = input[index] >= 0f ? input[index] : input[index] * slope;
            error = Math.Max(error, Math.Abs(output[index] - expected));
        }
        return error;
    }

    private static double DropoutForwardError(
        float[] output,
        float[] mask,
        float[] input)
    {
        if (output.Length != input.Length || mask.Length != input.Length)
            return double.PositiveInfinity;
        double error = 0;
        for (int index = 0; index < input.Length; index++)
        {
            float expected = input[index] * mask[index];
            error = Math.Max(error, Math.Abs(output[index] - expected));
        }
        return error;
    }
}
