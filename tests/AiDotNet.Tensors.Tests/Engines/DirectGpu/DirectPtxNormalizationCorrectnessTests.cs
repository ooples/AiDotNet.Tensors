using System.Globalization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DirectPtxFeatureGateCollection
{
    public const string Name = "DirectPtxFeatureGate";
}

[Collection(DirectPtxFeatureGateCollection.Name)]
public sealed class DirectPtxNormalizationCorrectnessTests
{
    private const int Rows = 256;
    private const int Width = 64;
    private const float Epsilon = 1e-5f;

    [Fact]
    public void CubinIdentity_IsInvariantAcrossPlatformLineEndings()
    {
        const string lf = ".version 7.1\n.target sm_86\n.address_size 64\n";
        string crlf = lf.Replace("\n", "\r\n");
        string cr = lf.Replace('\n', '\r');

        Assert.Equal(lf, DirectPtxCubinArtifactCache.CanonicalizePtx(crlf));
        Assert.Equal(lf, DirectPtxCubinArtifactCache.CanonicalizePtx(cr));
        Assert.Equal(
            DirectPtxCubinArtifactCache.ComputePtxSha256(lf),
            DirectPtxCubinArtifactCache.ComputePtxSha256(crlf));
        Assert.Equal(
            DirectPtxCubinArtifactCache.ComputeSourceKey(lf, 8, 6),
            DirectPtxCubinArtifactCache.ComputeSourceKey(crlf, 8, 6));
    }

    [Fact]
    public void FastModeRouting_UsesOnlyMeasuredAtomicWinner()
    {
        Assert.Equal(DirectPtxRowNormalizationOperation.LayerNormGradParameters,
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.LayerNormGradParameters,
                deterministic: false, rows: 8_192));
        Assert.Equal(DirectPtxRowNormalizationOperation.RmsNormGradGamma,
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.RmsNormGradGamma,
                deterministic: false, rows: 2_048));
        Assert.Equal(DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic,
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.RmsNormGradGamma,
                deterministic: false, rows: 8_192));
        Assert.Equal(DirectPtxRowNormalizationOperation.RmsNormGradGamma,
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.RmsNormGradGamma,
                deterministic: true, rows: 8_192));
        Assert.Equal(DirectPtxRowNormalizationOperation.ReduceNormL2,
            PtxRowNormalizationD64Kernel.SelectFastOperation(
                DirectPtxRowNormalizationOperation.ReduceNormL2,
                deterministic: false, rows: 8_192));
    }

    [Fact]
    public void BankedL2Emitter_UsesReusableRedAccumulatorsAndRemainsUnpromoted()
    {
        string ptx = PtxRowNormalizationD64Kernel.EmitPtx(
            8, 6, DirectPtxRowNormalizationOperation.ReduceNormL2Atomic, 8_192);
        DirectPtxKernelBlueprint blueprint =
            PtxRowNormalizationD64Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere,
                DirectPtxRowNormalizationOperation.ReduceNormL2Atomic,
                8_192);

        Assert.Contains("red.global.add.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("setp.ge.u32 %p5, %r2, 16", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("REDUCE_NORM_LOOP", ptx, StringComparison.Ordinal);
        Assert.Equal(2, ptx.Split(
            new[] { "ld.global.nc.v4.f32" }, StringSplitOptions.None).Length - 1);
        Assert.Contains("shfl.sync.bfly.b32 %r9, %r8, 8, 15, 0x0000ffff", ptx,
            StringComparison.Ordinal);
        Assert.Contains("ld.global.f32 %f0, [%rd4]", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("REDUCE_NORM_LAST_FINAL", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("st.global.f32 [%rd4], %f3", ptx, StringComparison.Ordinal);
        Assert.Equal("16", blueprint.Semantics["accumulator-banks"]);
        Assert.Equal("false", blueprint.Semantics["deterministic"]);
        Assert.False(PtxRowNormalizationD64Kernel.IsPromoted(
            DirectPtxRowNormalizationOperation.ReduceNormL2Atomic, 8_192));
    }

    [Fact]
    public void HalfBits_UsesIeeeBinary16AcrossTargetFrameworks()
    {
        (float Value, ushort Bits)[] cases =
        [
            (0f, 0x0000), (-0f, 0x8000), (1f, 0x3c00), (-2f, 0xc000),
            (0.5f, 0x3800), (65504f, 0x7bff), (float.PositiveInfinity, 0x7c00),
            (float.NegativeInfinity, 0xfc00)
        ];
        foreach ((float value, ushort bits) in cases)
        {
            Assert.Equal(bits, HalfBits.GetBits((Half)value));
            float roundTrip = (float)HalfBits.FromBits(bits);
            Assert.Equal(PtxCompat.SingleToInt32Bits(value), PtxCompat.SingleToInt32Bits(roundTrip));
        }
        Assert.True(float.IsNaN((float)HalfBits.FromBits(0x7e00)));
        Assert.True((HalfBits.GetBits((Half)float.NaN) & 0x7c00) == 0x7c00);
    }

    [Fact]
    public void ReleaseCubins_AreCompleteAndMatchEmbeddedManifest()
    {
        var assembly = typeof(PtxRowNormalizationD64Kernel).Assembly;
        string[] resources = assembly.GetManifestResourceNames();
        // The driver-linked artifact set this test used to read was retired: every
        // row in it recorded the hash of PTX that stopped existing when epsilon
        // and momentum became launch parameters, and it could not be rebuilt
        // without an admitted device. The offline set that replaced it covers the
        // same operations and more - eighty-six modules against seventy-one - so
        // retargeting here strengthens the check rather than relaxing it.
        string manifestResource = Assert.Single(resources, name =>
            name.EndsWith(".Artifacts.sm86.normalization-offline-cubins.tsv", StringComparison.Ordinal));
        var expected = new Dictionary<string, string>(StringComparer.Ordinal);
        var blueprintIds = new HashSet<string>(StringComparer.Ordinal);
        int manifestRows = 0;
        using (Stream stream = Assert.IsAssignableFrom<Stream>(
                   assembly.GetManifestResourceStream(manifestResource)))
        using (var reader = new StreamReader(stream))
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#' ||
                    line.StartsWith("blueprint-id", StringComparison.Ordinal))
                    continue;
                string[] columns = line.Split('\t');
                Assert.Equal(9, columns.Length);
                manifestRows++;
                Assert.True(blueprintIds.Add(columns[0]), $"Duplicate blueprint identity: {columns[0]}");

                // The artifact is named by its own content, so the file name and
                // the recorded cubin hash have to be the same string. Checking it
                // here is what makes the lookup below meaningful.
                string cubinHash = columns[3];
                Assert.Equal(cubinHash + ".cubin", columns[8]);

                // ptxas measured these. A zero register count would mean the
                // manifest was written without the -v report it claims to carry,
                // which would silently disarm the resource-budget evidence.
                Assert.True(int.Parse(columns[5], CultureInfo.InvariantCulture) > 0,
                    $"No register measurement for {columns[0]}.");
                Assert.True(int.Parse(columns[7], CultureInfo.InvariantCulture) > 0,
                    $"No occupancy figure for {columns[0]}.");

                if (expected.TryGetValue(cubinHash, out string? existingHash))
                    Assert.Equal(existingHash, cubinHash);
                else
                    expected[cubinHash] = cubinHash;
            }
        }
        Assert.Equal(86, manifestRows);
        Assert.Equal(86, blueprintIds.Count);
        Assert.Equal(81, expected.Count);

        string[] cubins = resources.Where(name =>
            name.IndexOf(".Artifacts.sm86.", StringComparison.Ordinal) >= 0 &&
            name.EndsWith(".cubin", StringComparison.Ordinal)).ToArray();
        Assert.Equal(expected.Count, cubins.Length);
        foreach (string resource in cubins)
        {
            int end = resource.Length - ".cubin".Length;
            int start = resource.LastIndexOf('.', end - 1) + 1;
            string sourceKey = resource.Substring(start, end - start);
            using Stream stream = Assert.IsAssignableFrom<Stream>(
                assembly.GetManifestResourceStream(resource));
            using var memory = new MemoryStream();
            stream.CopyTo(memory);
            using SHA256 sha = SHA256.Create();
            string actual = PtxCompat.ToHexString(sha.ComputeHash(memory.ToArray())).ToLowerInvariant();
            Assert.True(expected.TryGetValue(sourceKey, out string? hash), sourceKey);
            Assert.Equal(hash, actual);
        }
    }

    [SkippableFact]
    public void FusionManager_ExecutesResidualBatchNormReluThroughCudaProductionRoute()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        string? previousBackendOrder = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "cuda");
        try
        {
            const int batch = PtxChannelNormalizationD64Kernel.BatchNormBatch;
            const int channels = PtxChannelNormalizationD64Kernel.BatchNormChannels;
            const int spatial = PtxChannelNormalizationD64Kernel.BatchNormSpatial;
            const int elements = batch * channels * spatial;
            float[] input = Values(elements, 43, 0.015625f);
            float[] residual = Values(elements, 47, 0.0078125f);
            float[] gamma = Values(channels, 29, 0.01f, 1f);
            float[] beta = Values(channels, 31, 0.005f);
            float[] runningMean = Values(channels, 23, 0.00390625f);
            float[] runningVariance = Values(channels, 19, 0.002f, 0.75f)
                .Select(PtxCompat.Abs).ToArray();
            var expected = new float[elements];
            for (int n = 0; n < batch; n++)
                for (int channel = 0; channel < channels; channel++)
                    for (int s = 0; s < spatial; s++)
                    {
                        int index = (n * channels + channel) * spatial + s;
                        float normalized = (input[index] - runningMean[channel]) /
                            PtxCompat.Sqrt(runningVariance[channel] + Epsilon);
                        expected[index] = PtxCompat.Max(0f,
                            normalized * gamma[channel] + beta[channel] + residual[index]);
                    }

            using var engine = new DirectGpuEngine();
            Skip.IfNot(engine.IsAvailable && engine.DeviceVendor.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0,
                "Requires the CUDA backend.");
            IReadOnlyList<FusableOperation> operations = new[]
            {
                new FusableOperation(GpuOperationType.Residual),
                new FusableOperation(GpuOperationType.BatchNorm),
                new FusableOperation(GpuOperationType.ReLU)
            };
            IReadOnlyDictionary<string, float[]> buffers = new Dictionary<string, float[]>
            {
                ["residual"] = residual,
                ["gamma"] = gamma,
                ["beta"] = beta,
                ["runningMean"] = runningMean,
                ["runningVariance"] = runningVariance
            };
            IReadOnlyDictionary<string, int> dimensions = new Dictionary<string, int>
            {
                ["batch"] = batch,
                ["channels"] = channels,
                ["spatialSize"] = spatial
            };

            float[]? actual = engine.ExecuteWithFusion(operations, input, buffers, dimensions);
            Assert.NotNull(actual);
            AssertClose(actual!, expected, 1.5e-3f, "production residual BatchNorm ReLU");
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", previousBackendOrder);
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void RowBackend_AdmissionTransactionCacheCaptureAndAllocationContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxNormalizationEnabled,
                "Requires the validated SM86 normalization backend.");
            int elements = Rows * Width;
            using var input = backend.AllocateBuffer(Values(elements, 37, 0.03125f));
            using var gamma = backend.AllocateBuffer(Values(Width, 29, 0.015625f, 1f));
            using var beta = backend.AllocateBuffer(Values(Width, 31, 0.0078125f));
            using var output = backend.AllocateBuffer(elements);
            using var mean = backend.AllocateBuffer(Rows);
            using var invStd = backend.AllocateBuffer(Rows);
            using var gradOutput = backend.AllocateBuffer(Values(elements, 41, 0.015625f));
            using var gradInput = backend.AllocateBuffer(elements);
            using var gradGamma = backend.AllocateBuffer(Width);
            using var gradBeta = backend.AllocateBuffer(Width);

            long dispatchBefore = backend.DirectPtxRowNormalizationDispatchCount;
            DirectPtxFeatureGate.TestOverride = false;
            Assert.False(backend.TryDirectPtxLayerNormD64(
                input, output, gamma, beta, mean, invStd, Rows, Epsilon));
            Assert.Equal("normalization-feature-disabled", backend.DirectPtxLastError);
            Assert.Equal(dispatchBefore, backend.DirectPtxRowNormalizationDispatchCount);

            DirectPtxFeatureGate.TestOverride = true;
            DirectPtxFeatureGate.NormalizationExperimentOverride = false;
            Assert.False(backend.TryDirectPtxLayerNormD64(
                input, output, gamma, beta, mean, invStd, Rows, Epsilon));
            Assert.Equal("normalization-performance-gate-not-met", backend.DirectPtxLastError);
            Assert.Equal(dispatchBefore, backend.DirectPtxRowNormalizationDispatchCount);

            DirectPtxFeatureGate.NormalizationExperimentOverride = true;
            Assert.False(backend.TryDirectPtxLayerNormD64(
                input, output, gamma, beta, mean, invStd, 32, Epsilon));
            Assert.Equal("normalization-shape-not-implemented", backend.DirectPtxLastError);
            Assert.Equal(dispatchBefore, backend.DirectPtxRowNormalizationDispatchCount);

            using (var oversizedOutput = backend.AllocateBuffer(elements + 1))
            {
                Assert.False(backend.TryDirectPtxLayerNormD64(
                    input, oversizedOutput, gamma, beta, mean, invStd, Rows, Epsilon));
                Assert.StartsWith("normalization-ArgumentException:", backend.DirectPtxLastError);
            }
            Assert.False(backend.TryDirectPtxLayerNormD64(
                input, output, gamma, beta, mean, mean, Rows, Epsilon));
            Assert.StartsWith("normalization-ArgumentException:", backend.DirectPtxLastError);
            Assert.Equal(dispatchBefore, backend.DirectPtxRowNormalizationDispatchCount);

            using (var oversizedGradBeta = backend.AllocateBuffer(Width + 1))
            {
                Assert.False(backend.TryDirectPtxLayerNormBackwardD64(
                    gradOutput, input, gamma, mean, invStd,
                    gradInput, gradGamma, oversizedGradBeta, Rows, Epsilon));
                Assert.StartsWith("normalization-ArgumentException:", backend.DirectPtxLastError);
            }
            Assert.Equal(dispatchBefore, backend.DirectPtxRowNormalizationDispatchCount);

            Assert.True(backend.PrewarmDirectPtxRowNormalization(
                DirectPtxRowNormalizationOperation.LayerNormForward, Rows, Epsilon),
                backend.DirectPtxLastError);
            Assert.True(backend.TryDirectPtxLayerNormD64(
                input, output, gamma, beta, mean, invStd, Rows, Epsilon),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.True(backend.TryGetDirectPtxRowNormalizationAudit(
                DirectPtxRowNormalizationOperation.LayerNormForward,
                Rows, Epsilon, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(0, audit.Function.StaticSharedBytes);
            Assert.NotEmpty(audit.CubinSha256);
            Assert.NotEmpty(audit.CubinSourceKey);
            Assert.Equal(DirectPtxModuleImageKind.EmbeddedCubin, audit.ImageKind);

            const int batchNormElements =
                PtxChannelNormalizationD64Kernel.BatchNormBatch *
                PtxChannelNormalizationD64Kernel.BatchNormChannels *
                PtxChannelNormalizationD64Kernel.BatchNormSpatial;
            using (var batchNormInput = backend.AllocateBuffer(batchNormElements))
            using (var residual = backend.AllocateBuffer(batchNormElements))
            using (var batchNormOutput = backend.AllocateBuffer(batchNormElements))
            using (var batchNormGamma = backend.AllocateBuffer(PtxChannelNormalizationD64Kernel.BatchNormChannels))
            using (var batchNormBeta = backend.AllocateBuffer(PtxChannelNormalizationD64Kernel.BatchNormChannels))
            using (var runningMean = backend.AllocateBuffer(PtxChannelNormalizationD64Kernel.BatchNormChannels))
            using (var runningVariance = backend.AllocateBuffer(PtxChannelNormalizationD64Kernel.BatchNormChannels))
            {
                long residualDispatchBefore = backend.DirectPtxChannelNormalizationDispatchCount;
                Assert.True(backend.TryResidualBatchNormRelu(
                    batchNormInput, residual, batchNormOutput,
                    batchNormGamma, batchNormBeta, runningMean, runningVariance,
                    PtxChannelNormalizationD64Kernel.BatchNormBatch,
                    PtxChannelNormalizationD64Kernel.BatchNormChannels,
                    PtxChannelNormalizationD64Kernel.BatchNormSpatial,
                    Epsilon), backend.DirectPtxLastError);
                backend.Synchronize();
                Assert.Equal(residualDispatchBefore + 1,
                    backend.DirectPtxChannelNormalizationDispatchCount);
            }

            for (int i = 0; i < 64; i++)
                Assert.True(backend.TryDirectPtxLayerNormD64(
                    input, output, gamma, beta, mean, invStd, Rows, Epsilon),
                    backend.DirectPtxLastError);
            backend.Synchronize();

            long captureProbeBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            bool captureProbe = false;
            for (int i = 0; i < 32; i++) captureProbe |= backend.IsStreamCapturing();
            long captureProbeAllocated = PtxCompat.GetAllocatedBytesForCurrentThread() - captureProbeBefore;
            Assert.False(captureProbe);
            Assert.True(captureProbeAllocated == 0,
                $"capture probe allocated {captureProbeAllocated} bytes");

            long contextBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++) backend.EnsureContextCurrent();
            long contextAllocated = PtxCompat.GetAllocatedBytesForCurrentThread() - contextBefore;
            Assert.True(contextAllocated == 0,
                $"context assertion allocated {contextAllocated} bytes");

            long cacheBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            bool cacheHits = true;
            for (int i = 0; i < 32; i++)
                cacheHits &= backend.TryGetDirectPtxRowNormalizationAudit(
                    DirectPtxRowNormalizationOperation.LayerNormForward,
                    Rows, Epsilon, out _);
            long cacheAllocated = PtxCompat.GetAllocatedBytesForCurrentThread() - cacheBefore;
            Assert.True(cacheHits);
            Assert.True(cacheAllocated == 0,
                $"cache lookup allocated {cacheAllocated} bytes");

            DirectPtxKernelBlueprint hotBlueprint =
                PtxRowNormalizationD64Kernel.CreateBlueprint(
                    DirectPtxArchitectureFamily.Ampere,
                    DirectPtxRowNormalizationOperation.LayerNormForward, Rows);
            Span<DirectPtxTensorView> hotViews = stackalloc DirectPtxTensorView[6];
            long viewBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
            {
                hotViews[0] = DirectPtxTensorView.Create(input, hotBlueprint.Tensors[0]);
                hotViews[1] = DirectPtxTensorView.Create(gamma, hotBlueprint.Tensors[1]);
                hotViews[2] = DirectPtxTensorView.Create(beta, hotBlueprint.Tensors[2]);
                hotViews[3] = DirectPtxTensorView.Create(output, hotBlueprint.Tensors[3]);
                hotViews[4] = DirectPtxTensorView.Create(mean, hotBlueprint.Tensors[4]);
                hotViews[5] = DirectPtxTensorView.Create(invStd, hotBlueprint.Tensors[5]);
            }
            long viewAllocated = PtxCompat.GetAllocatedBytesForCurrentThread() - viewBefore;
            Assert.True(viewAllocated == 0,
                $"view construction allocated {viewAllocated} bytes");

            long validationBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
            {
                PtxRowNormalizationD64Kernel.ValidateTensors(
                    hotBlueprint, hotViews,
                    PtxRowNormalizationD64Kernel.GetEntryPoint(
                        DirectPtxRowNormalizationOperation.LayerNormForward));
            }
            long validationAllocated = PtxCompat.GetAllocatedBytesForCurrentThread() - validationBefore;
            Assert.True(validationAllocated == 0,
                $"view validation allocated {validationAllocated} bytes");

            long allocationBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            bool everyLaunchSucceeded = true;
            for (int i = 0; i < 32; i++)
                everyLaunchSucceeded &= backend.TryDirectPtxLayerNormD64(
                    input, output, gamma, beta, mean, invStd, Rows, Epsilon);
            long allocated = PtxCompat.GetAllocatedBytesForCurrentThread() - allocationBefore;
            backend.Synchronize();
            Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxLayerNormD64(
                    input, output, gamma, beta, mean, invStd, Rows, Epsilon),
                    backend.DirectPtxLastError));
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxRowNormalizationPinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            Assert.Equal(0, backend.DirectPtxRowNormalizationPinnedKernelCount);
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void FusedBackwardWorkspace_ReplaysWithoutAllocationOrStaleAccumulators()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        TensorCodecOptions previousOptions = TensorCodecOptions.Current;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        TensorCodecOptions.SetCurrent(new TensorCodecOptions { Deterministic = false });
        try
        {
            const int rows = 8_192;
            int elements = rows * Width;
            float[] inputHost = Values(elements, 37, 0.03125f);
            float[] gradHost = Values(elements, 41, 0.00390625f);
            float[] gammaHost = Values(Width, 29, 0.015625f, 1f);
            float[] meanHost = Values(rows, 43, 0.015625f);
            float[] invStdHost = Enumerable.Repeat(1.125f, rows).ToArray();
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxNormalizationEnabled,
                "Requires the validated SM86 normalization backend.");
            using var input = backend.AllocateBuffer(inputHost);
            using var gradOutput = backend.AllocateBuffer(gradHost);
            using var gamma = backend.AllocateBuffer(gammaHost);
            using var mean = backend.AllocateBuffer(meanHost);
            using var invStd = backend.AllocateBuffer(invStdHost);
            using var gradInput = backend.AllocateBuffer(elements);
            using var gradGamma = backend.AllocateBuffer(Width);
            using var gradBeta = backend.AllocateBuffer(Width);

            Assert.True(backend.PrewarmDirectPtxRowNormalization(
                DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic,
                rows, Epsilon), backend.DirectPtxLastError);
            Assert.True(backend.TryDirectPtxLayerNormBackwardD64(
                gradOutput, input, gamma, mean, invStd,
                gradInput, gradGamma, gradBeta, rows, Epsilon),
                backend.DirectPtxLastError);
            backend.Synchronize();
            float[] expectedGamma = backend.DownloadBuffer(gradGamma);
            float[] expectedBeta = backend.DownloadBuffer(gradBeta);

            long allocationBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            bool everyLaunchSucceeded = true;
            for (int i = 0; i < 32; i++)
                everyLaunchSucceeded &= backend.TryDirectPtxLayerNormBackwardD64(
                    gradOutput, input, gamma, mean, invStd,
                    gradInput, gradGamma, gradBeta, rows, Epsilon);
            long allocated = PtxCompat.GetAllocatedBytesForCurrentThread() - allocationBefore;
            backend.Synchronize();
            Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertClose(backend.DownloadBuffer(gradGamma), expectedGamma, 4e-3f,
                "reused workspace grad gamma");
            AssertClose(backend.DownloadBuffer(gradBeta), expectedBeta, 4e-3f,
                "reused workspace grad beta");

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxLayerNormBackwardD64(
                    gradOutput, input, gamma, mean, invStd,
                    gradInput, gradGamma, gradBeta, rows, Epsilon),
                    backend.DirectPtxLastError));
            try
            {
                backend.LaunchCapturedGraph(graph);
                backend.LaunchCapturedGraph(graph);
                backend.Synchronize();
                AssertClose(backend.DownloadBuffer(gradGamma), expectedGamma, 4e-3f,
                    "graph workspace grad gamma");
                AssertClose(backend.DownloadBuffer(gradBeta), expectedBeta, 4e-3f,
                    "graph workspace grad beta");
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(previousOptions);
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void ChannelBackend_AdmissionTransactionCacheCaptureAndAllocationContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxNormalizationEnabled,
                "Requires the validated SM86 normalization backend.");
            const int batch = PtxChannelNormalizationD64Kernel.GroupNormBatch;
            const int channels = PtxChannelNormalizationD64Kernel.GroupNormChannels;
            const int groups = PtxChannelNormalizationD64Kernel.GroupNormGroups;
            const int spatial = PtxChannelNormalizationD64Kernel.GroupNormSpatial;
            const int elements = batch * channels * spatial;
            const int stats = batch * groups;
            using var input = backend.AllocateBuffer(Values(elements, 43, 0.015625f));
            using var gamma = backend.AllocateBuffer(Values(channels, 29, 0.01f, 1f));
            using var beta = backend.AllocateBuffer(Values(channels, 31, 0.005f));
            using var output = backend.AllocateBuffer(elements);
            using var mean = backend.AllocateBuffer(stats);
            using var variance = backend.AllocateBuffer(stats);
            using var gradOutput = backend.AllocateBuffer(Values(elements, 47, 0.0078125f));
            using var gradInput = backend.AllocateBuffer(elements);
            using var gradGamma = backend.AllocateBuffer(channels);

            long dispatchBefore = backend.DirectPtxChannelNormalizationDispatchCount;
            Assert.False(backend.TryDirectPtxGroupNormUnit64(
                input, output, gamma, beta, mean, variance,
                batch - 1, groups, channels, spatial, Epsilon));
            Assert.Equal("normalization-shape-not-implemented", backend.DirectPtxLastError);

            using (var oversizedOutput = backend.AllocateBuffer(elements + 1))
            {
                Assert.False(backend.TryDirectPtxGroupNormUnit64(
                    input, oversizedOutput, gamma, beta, mean, variance,
                    batch, groups, channels, spatial, Epsilon));
                Assert.StartsWith("channel-normalization-ArgumentException:", backend.DirectPtxLastError);
            }
            Assert.False(backend.TryDirectPtxGroupNormUnit64(
                input, output, gamma, beta, mean, mean,
                batch, groups, channels, spatial, Epsilon));
            Assert.StartsWith("channel-normalization-ArgumentException:", backend.DirectPtxLastError);
            Assert.Equal(dispatchBefore, backend.DirectPtxChannelNormalizationDispatchCount);

            using (var oversizedGradBeta = backend.AllocateBuffer(channels + 1))
            {
                Assert.False(backend.TryDirectPtxGroupNormBackwardUnit64(
                    gradOutput, input, gamma, mean, variance,
                    gradInput, gradGamma, oversizedGradBeta,
                    batch, groups, channels, spatial, Epsilon));
                Assert.StartsWith("channel-normalization-ArgumentException:", backend.DirectPtxLastError);
            }
            Assert.Equal(dispatchBefore, backend.DirectPtxChannelNormalizationDispatchCount);

            Assert.True(backend.PrewarmDirectPtxChannelNormalization(
                DirectPtxChannelNormalizationOperation.GroupNormForward, Epsilon, 0f),
                backend.DirectPtxLastError);
            Assert.True(backend.TryDirectPtxGroupNormUnit64(
                input, output, gamma, beta, mean, variance,
                batch, groups, channels, spatial, Epsilon), backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.True(backend.TryGetDirectPtxChannelNormalizationAudit(
                DirectPtxChannelNormalizationOperation.GroupNormForward,
                Epsilon, 0f, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(0, audit.Function.StaticSharedBytes);
            Assert.NotEmpty(audit.CubinSha256);
            Assert.NotEmpty(audit.CubinSourceKey);
            Assert.Equal(DirectPtxModuleImageKind.EmbeddedCubin, audit.ImageKind);

            for (int i = 0; i < 64; i++)
                Assert.True(backend.TryDirectPtxGroupNormUnit64(
                    input, output, gamma, beta, mean, variance,
                    batch, groups, channels, spatial, Epsilon), backend.DirectPtxLastError);
            backend.Synchronize();
            long allocationBefore = PtxCompat.GetAllocatedBytesForCurrentThread();
            bool everyLaunchSucceeded = true;
            for (int i = 0; i < 32; i++)
                everyLaunchSucceeded &= backend.TryDirectPtxGroupNormUnit64(
                    input, output, gamma, beta, mean, variance,
                    batch, groups, channels, spatial, Epsilon);
            long allocated = PtxCompat.GetAllocatedBytesForCurrentThread() - allocationBefore;
            backend.Synchronize();
            Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxGroupNormUnit64(
                    input, output, gamma, beta, mean, variance,
                    batch, groups, channels, spatial, Epsilon), backend.DirectPtxLastError));
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxChannelNormalizationPinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            Assert.Equal(0, backend.DirectPtxChannelNormalizationPinnedKernelCount);
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void RowForwardAndL2Inventory_MatchesCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        float[] input = Values(Rows * Width, 37, 0.03125f);
        float[] gamma = Values(Width, 29, 0.015625f, 1f);
        float[] beta = Values(Width, 31, 0.0078125f);
        RowStats(input, out float[] mean, out float[] variance, out float[] invStd,
            out float[] rms, out float[] norm);

        var layerNorm = new float[input.Length];
        var rmsNorm = new float[input.Length];
        var normalized = new float[input.Length];
        var normGradient = new float[input.Length];
        var scalarGradient = Values(Rows, 17, 0.01f);
        double totalSquares = 0;
        for (int row = 0; row < Rows; row++)
        {
            int rowBase = row * Width;
            for (int feature = 0; feature < Width; feature++)
            {
                int index = rowBase + feature;
                layerNorm[index] = (input[index] - mean[row]) * invStd[row] *
                    gamma[feature] + beta[feature];
                rmsNorm[index] = input[index] / rms[row] * gamma[feature];
                normalized[index] = input[index] / (norm[row] + 1e-7f);
                normGradient[index] = scalarGradient[row] * input[index] / norm[row];
                totalSquares += (double)input[index] * input[index];
            }
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.LayerNormForward, Rows, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, null, null, null);
            AssertClose((float[])result[3], layerNorm, 4e-4f, "LayerNorm output");
            AssertClose((float[])result[4], mean, 2e-5f, "LayerNorm mean");
            AssertClose((float[])result[5], invStd, 4e-4f, "LayerNorm inverse stddev");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.RmsNormForward, Rows, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, null, null);
            AssertClose((float[])result[2], rmsNorm, 4e-4f, "RMSNorm output");
            AssertClose((float[])result[3], rms, 4e-4f, "RMSNorm rms");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.NormAxis, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, null);
            AssertClose((float[])result[1], norm, 3e-4f, "row L2 norm");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.NormalizeL2, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, null);
            AssertClose((float[])result[1], normalized, 4e-4f, "L2 normalized rows");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.NormBackward, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                scalarGradient, input, norm, null);
            AssertClose((float[])result[3], normGradient, 4e-4f, "L2 norm backward");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.ReduceNormL2, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, null);
            AssertClose((float[])result[1], [(float)totalSquares], 5e-2f,
                "whole-tensor sum of squares");
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.ReduceNormL2Atomic, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views =>
                { kernel.Launch(views); kernel.Launch(views); },
                input, new float[1],
                new float[PtxRowNormalizationD64Kernel.NormalizationWorkspaceElements]);
            AssertClose((float[])result[1], [(float)totalSquares], 5e-2f,
                "atomic whole-tensor sum of squares");
        }
    }

    [SkippableFact]
    public void RowBackwardInventory_MatchesCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        float[] input = Values(Rows * Width, 37, 0.03125f);
        float[] gamma = Values(Width, 29, 0.015625f, 1f);
        float[] gradOutput = Values(Rows * Width, 41, 0.00390625f);
        RowStats(input, out float[] mean, out _, out float[] invStd,
            out float[] rms, out _);

        var layerGradInput = new float[input.Length];
        var layerGradGamma = new float[Width];
        var layerGradBeta = new float[Width];
        var rmsGradInput = new float[input.Length];
        var rmsGradGamma = new float[Width];
        for (int row = 0; row < Rows; row++)
        {
            int rowBase = row * Width;
            double sumDyGamma = 0;
            double sumDyGammaXhat = 0;
            double rmsDot = 0;
            for (int feature = 0; feature < Width; feature++)
            {
                int index = rowBase + feature;
                double xhat = (input[index] - mean[row]) * invStd[row];
                double dyGamma = gradOutput[index] * gamma[feature];
                sumDyGamma += dyGamma;
                sumDyGammaXhat += dyGamma * xhat;
                rmsDot += dyGamma * input[index];
                layerGradGamma[feature] += (float)(gradOutput[index] * xhat);
                layerGradBeta[feature] += gradOutput[index];
                rmsGradGamma[feature] += gradOutput[index] * input[index] / rms[row];
            }
            for (int feature = 0; feature < Width; feature++)
            {
                int index = rowBase + feature;
                double xhat = (input[index] - mean[row]) * invStd[row];
                double dyGamma = gradOutput[index] * gamma[feature];
                layerGradInput[index] = (float)(invStd[row] / Width *
                    (Width * dyGamma - sumDyGamma - xhat * sumDyGammaXhat));
                rmsGradInput[index] = (float)(dyGamma / rms[row] - input[index] *
                    rmsDot / (Width * rms[row] * rms[row] * rms[row]));
            }
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.LayerNormBackwardInput, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, gamma, mean, invStd, null);
            AssertClose((float[])result[5], layerGradInput, 8e-4f, "LayerNorm grad input");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.LayerNormGradParameters, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, mean, invStd, null, null);
            AssertClose((float[])result[4], layerGradGamma, 4e-3f, "LayerNorm grad gamma");
            AssertClose((float[])result[5], layerGradBeta, 4e-3f, "LayerNorm grad beta");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.LayerNormGradParametersAtomic, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, mean, invStd, new float[Width], new float[Width]);
            AssertClose((float[])result[4], layerGradGamma, 4e-3f,
                "atomic LayerNorm grad gamma");
            AssertClose((float[])result[5], layerGradBeta, 4e-3f,
                "atomic LayerNorm grad beta");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.LayerNormBackwardFusedAtomic, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views =>
                { kernel.Launch(views); kernel.Launch(views); },
                gradOutput, input, gamma, mean, invStd, null,
                new float[Width], new float[Width],
                new float[PtxRowNormalizationD64Kernel.NormalizationWorkspaceElements]);
            AssertClose((float[])result[5], layerGradInput, 8e-4f,
                "fused LayerNorm grad input");
            AssertClose((float[])result[6], layerGradGamma, 4e-3f,
                "fused LayerNorm grad gamma");
            AssertClose((float[])result[7], layerGradBeta, 4e-3f,
                "fused LayerNorm grad beta");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.RmsNormBackwardInput, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, gamma, rms, null);
            AssertClose((float[])result[4], rmsGradInput, 8e-4f, "RMSNorm grad input");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.RmsNormGradGamma, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, rms, null);
            AssertClose((float[])result[3], rmsGradGamma, 4e-3f, "RMSNorm grad gamma");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.RmsNormGradGammaAtomic, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, rms, new float[Width]);
            AssertClose((float[])result[3], rmsGradGamma, 4e-3f,
                "atomic RMSNorm grad gamma");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.RmsNormBackwardFusedAtomic, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views =>
                { kernel.Launch(views); kernel.Launch(views); },
                gradOutput, input, gamma, rms, null, new float[Width],
                new float[PtxRowNormalizationD64Kernel.NormalizationWorkspaceElements]);
            AssertClose((float[])result[4], rmsGradInput, 8e-4f,
                "fused RMSNorm grad input");
            AssertClose((float[])result[5], rmsGradGamma, 4e-3f,
                "fused RMSNorm grad gamma");
        }
    }

    [SkippableFact]
    public void Fp16LayerNormInventory_MatchesQuantizedCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        ushort[] inputBits = ToHalf(Values(Rows * Width, 37, 0.03125f));
        ushort[] gammaBits = ToHalf(Values(Width, 29, 0.015625f, 1f));
        ushort[] betaBits = ToHalf(Values(Width, 31, 0.0078125f));
        ushort[] gradBits = ToHalf(Values(Rows * Width, 41, 0.0009765625f));
        float[] input = FromHalf(inputBits);
        float[] gamma = FromHalf(gammaBits);
        float[] beta = FromHalf(betaBits);
        float[] gradOutput = FromHalf(gradBits);
        RowStats(input, out float[] mean, out float[] variance, out float[] invStd,
            out _, out _);

        var output = new float[input.Length];
        var gradInput = new float[input.Length];
        var gradGamma = new float[Width];
        var gradBeta = new float[Width];
        for (int row = 0; row < Rows; row++)
        {
            int rowBase = row * Width;
            double sumDyGamma = 0, sumDyGammaXhat = 0;
            for (int feature = 0; feature < Width; feature++)
            {
                int index = rowBase + feature;
                double xhat = (input[index] - mean[row]) * invStd[row];
                output[index] = (float)(xhat * gamma[feature] + beta[feature]);
                double dyGamma = gradOutput[index] * gamma[feature];
                sumDyGamma += dyGamma;
                sumDyGammaXhat += dyGamma * xhat;
                gradGamma[feature] += (float)(gradOutput[index] * xhat);
                gradBeta[feature] += gradOutput[index];
            }
            for (int feature = 0; feature < Width; feature++)
            {
                int index = rowBase + feature;
                double xhat = (input[index] - mean[row]) * invStd[row];
                double dyGamma = gradOutput[index] * gamma[feature];
                gradInput[index] = (float)(invStd[row] / Width *
                    (Width * dyGamma - sumDyGamma - xhat * sumDyGammaXhat));
            }
        }

        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.Fp16LayerNormForward, Rows, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                inputBits, gammaBits, betaBits, null, null, null);
            AssertClose(FromHalf((ushort[])result[3]), output, 4e-3f, "FP16 LayerNorm output");
            AssertClose((float[])result[4], mean, 3e-4f, "FP16 LayerNorm mean");
            AssertClose((float[])result[5], variance, 5e-4f, "FP16 LayerNorm variance");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.Fp16LayerNormBackwardInput, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradBits, inputBits, gammaBits, mean, invStd, null);
            AssertClose(FromHalf((ushort[])result[5]), gradInput, 4e-3f,
                "FP16 LayerNorm grad input");
        }
        using (var kernel = new PtxRowNormalizationD64Kernel(
                   runtime, DirectPtxRowNormalizationOperation.Fp16LayerNormGradParameters, Rows))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradBits, inputBits, mean, invStd, null, null);
            AssertClose(FromHalf((ushort[])result[4]), gradGamma, 8e-3f,
                "FP16 LayerNorm grad gamma");
            AssertClose(FromHalf((ushort[])result[5]), gradBeta, 8e-3f,
                "FP16 LayerNorm grad beta");
        }
    }

    [SkippableFact]
    public void BatchNormAndFusedInventory_MatchesCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        const int batch = PtxChannelNormalizationD64Kernel.BatchNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.BatchNormChannels;
        const int spatial = PtxChannelNormalizationD64Kernel.BatchNormSpatial;
        const float momentum = 0.1f;
        float[] input = Values(batch * channels * spatial, 43, 0.015625f);
        float[] residual = Values(input.Length, 47, 0.0078125f);
        float[] gamma = Values(channels, 29, 0.015625f, 1f);
        float[] beta = Values(channels, 31, 0.0078125f);
        float[] runningMean = Values(channels, 23, 0.005f);
        float[] runningVariance = Values(channels, 19, 0.01f, 1.25f);
        BatchStats(input, batch, channels, spatial,
            out float[] batchMean, out float[] batchVariance, out float[] batchInvStd);
        float[] trainingOutput = BatchAffine(
            input, gamma, beta, batchMean, batchInvStd, batch, channels, spatial);
        float[] inferenceInvStd = new float[channels];
        for (int channel = 0; channel < channels; channel++)
            inferenceInvStd[channel] = 1f / PtxCompat.Sqrt(runningVariance[channel] + Epsilon);
        float[] inference = BatchAffine(
            input, gamma, beta, runningMean, inferenceInvStd, batch, channels, spatial);

        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.BatchNormTraining,
                   Epsilon, momentum))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, runningMean, runningVariance, null, null, null);
            AssertClose((float[])result[6], batchMean, 3e-5f, "BatchNorm saved mean");
            AssertClose((float[])result[7], batchInvStd, 5e-4f, "BatchNorm saved inverse stddev");
            AssertClose((float[])result[5], trainingOutput, 7e-4f, "BatchNorm training output");
            var expectedRunningMean = new float[channels];
            var expectedRunningVariance = new float[channels];
            for (int channel = 0; channel < channels; channel++)
            {
                expectedRunningMean[channel] =
                    (1 - momentum) * runningMean[channel] + momentum * batchMean[channel];
                expectedRunningVariance[channel] = (1 - momentum) * runningVariance[channel] +
                    momentum * batchVariance[channel] * 64f / 63f;
            }
            AssertClose((float[])result[3], expectedRunningMean, 4e-5f,
                "BatchNorm running mean");
            AssertClose((float[])result[4], expectedRunningVariance, 7e-5f,
                "BatchNorm running variance");
        }

        DirectPtxChannelNormalizationOperation[] inferenceOperations =
        [
            DirectPtxChannelNormalizationOperation.BatchNormInference,
            DirectPtxChannelNormalizationOperation.BatchNormRelu,
            DirectPtxChannelNormalizationOperation.BatchNormGelu,
            DirectPtxChannelNormalizationOperation.BatchNormSigmoid,
            DirectPtxChannelNormalizationOperation.BatchNormTanh
        ];
        foreach (DirectPtxChannelNormalizationOperation operation in inferenceOperations)
        {
            float[] expected = Activate(inference, operation);
            using var kernel = new PtxChannelNormalizationD64Kernel(runtime, operation, Epsilon);
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, runningMean, runningVariance, null);
            AssertClose((float[])result[5], expected, 1.5e-3f, operation.ToString());
        }

        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.ResidualBatchNormRelu, Epsilon))
        {
            var expected = new float[inference.Length];
            for (int i = 0; i < expected.Length; i++)
                expected[i] = PtxCompat.Max(0, inference[i] + residual[i]);
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, residual, gamma, beta, runningMean, runningVariance, null);
            AssertClose((float[])result[6], expected, 8e-4f, "residual BatchNorm ReLU");
        }

        float[] gradOutput = Values(input.Length, 41, 0.00390625f);
        BatchBackward(input, gradOutput, gamma, batchMean, batchInvStd,
            batch, channels, spatial,
            out float[] gradInput, out float[] gradGamma, out float[] gradBeta);
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.BatchNormBackward, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, gamma, batchMean, batchInvStd, null, null, null);
            AssertClose((float[])result[5], gradInput, 9e-4f, "BatchNorm grad input");
            AssertClose((float[])result[6], gradGamma, 3e-3f, "BatchNorm grad gamma");
            AssertClose((float[])result[7], gradBeta, 3e-3f, "BatchNorm grad beta");
        }
    }

    [SkippableFact]
    public void GroupNormAndFusedInventory_MatchesCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        const int batch = PtxChannelNormalizationD64Kernel.GroupNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.GroupNormChannels;
        const int groups = PtxChannelNormalizationD64Kernel.GroupNormGroups;
        const int spatial = PtxChannelNormalizationD64Kernel.GroupNormSpatial;
        float[] input = Values(batch * channels * spatial, 43, 0.015625f);
        float[] right = Values(input.Length, 47, 0.0078125f);
        float[] gamma = Values(channels, 29, 0.015625f, 1f);
        float[] beta = Values(channels, 31, 0.0078125f);
        GroupStats(input, batch, channels, groups, spatial,
            out float[] mean, out float[] variance, out float[] invStd);
        float[] affine = GroupAffine(input, gamma, beta, mean, invStd,
            batch, channels, groups, spatial);

        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.GroupNormForward, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, null, null, null);
            AssertClose((float[])result[3], affine, 8e-4f, "GroupNorm output");
            AssertClose((float[])result[4], mean, 3e-5f, "GroupNorm mean");
            AssertClose((float[])result[5], variance, 5e-5f, "GroupNorm variance");
        }

        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.GroupNormSwish, Epsilon))
        {
            float[] expected = Activate(affine,
                DirectPtxChannelNormalizationOperation.GroupNormSwish);
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, null);
            AssertClose((float[])result[3], expected, 1.5e-3f, "GroupNorm Swish");
        }

        var sum = new float[input.Length];
        for (int i = 0; i < sum.Length; i++) sum[i] = input[i] + right[i];
        GroupStats(sum, batch, channels, groups, spatial,
            out float[] sumMean, out _, out float[] sumInvStd);
        float[] addExpected = GroupAffine(sum, gamma, beta, sumMean, sumInvStd,
            batch, channels, groups, spatial);
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.AddGroupNorm, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, right, gamma, beta, null);
            AssertClose((float[])result[4], addExpected, 9e-4f, "AddGroupNorm");
        }

        float[] gradOutput = Values(input.Length, 41, 0.00390625f);
        NormalizationBackward(input, gradOutput, gamma, mean, invStd,
            batch, channels, spatial, groups,
            out float[] gradInput, out float[] gradGamma, out float[] gradBeta);
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.GroupNormBackwardInput, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, gamma, mean, variance, null);
            AssertClose((float[])result[5], gradInput, 9e-4f, "GroupNorm grad input");
        }
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.GroupNormGradParameters, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, mean, variance, null, null);
            AssertClose((float[])result[4], gradGamma, 4e-3f, "GroupNorm grad gamma");
            AssertClose((float[])result[5], gradBeta, 4e-3f, "GroupNorm grad beta");
        }

        ushort[] inputBits = ToHalf(input);
        float[] quantizedInput = FromHalf(inputBits);
        GroupStats(quantizedInput, batch, channels, groups, spatial,
            out float[] halfMean, out _, out float[] halfInvStd);
        float[] halfExpected = GroupAffine(quantizedInput, gamma, beta,
            halfMean, halfInvStd, batch, channels, groups, spatial);
        halfExpected = Activate(halfExpected,
            DirectPtxChannelNormalizationOperation.GroupNormSwish);
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.Fp16GroupNormSwish, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                inputBits, gamma, beta, null);
            AssertClose(FromHalf((ushort[])result[3]), halfExpected, 5e-3f,
                "FP16 GroupNorm Swish");
        }
    }

    [SkippableFact]
    public void InstanceNormInventory_MatchesCpuOracles()
    {
        using var runtime = RequireSm86Runtime();
        const int batch = PtxChannelNormalizationD64Kernel.InstanceNormBatch;
        const int channels = PtxChannelNormalizationD64Kernel.InstanceNormChannels;
        const int spatial = PtxChannelNormalizationD64Kernel.InstanceNormSpatial;
        float[] input = Values(batch * channels * spatial, 43, 0.015625f);
        float[] gamma = Values(channels, 29, 0.015625f, 1f);
        float[] beta = Values(channels, 31, 0.0078125f);
        InstanceStats(input, batch, channels, spatial,
            out float[] mean, out float[] invStd);
        float[] affine = InstanceAffine(input, gamma, beta, mean, invStd,
            batch, channels, spatial);

        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.InstanceNormForward, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                input, gamma, beta, null, null, null);
            AssertClose((float[])result[3], affine, 8e-4f, "InstanceNorm output");
            AssertClose((float[])result[4], mean, 3e-5f, "InstanceNorm mean");
            AssertClose((float[])result[5], invStd, 5e-4f, "InstanceNorm inverse stddev");
        }

        float[] gradOutput = Values(input.Length, 41, 0.00390625f);
        NormalizationBackward(input, gradOutput, gamma, mean, invStd,
            batch, channels, spatial, groupCount: channels,
            out float[] gradInput, out float[] gradGamma, out float[] gradBeta);
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.InstanceNormBackwardInput, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, gamma, mean, invStd, null);
            AssertClose((float[])result[5], gradInput, 9e-4f, "InstanceNorm grad input");
        }
        using (var kernel = new PtxChannelNormalizationD64Kernel(
                   runtime, DirectPtxChannelNormalizationOperation.InstanceNormGradParameters, Epsilon))
        {
            object[] result = Run(runtime, kernel.Blueprint, views => kernel.Launch(views),
                gradOutput, input, mean, invStd, null, null);
            AssertClose((float[])result[4], gradGamma, 5e-3f, "InstanceNorm grad gamma");
            AssertClose((float[])result[5], gradBeta, 5e-3f, "InstanceNorm grad beta");
        }
    }

    private static DirectPtxRuntime RequireSm86Runtime()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasValidatedResidualLayerNormGelu(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            runtime.Dispose();
            Skip.If(true, "The checked-in normalization family is validated on GA10x/SM86.");
        }
        return runtime;
    }

    private static object[] Run(
        DirectPtxRuntime runtime,
        DirectPtxKernelBlueprint blueprint,
        Action<DirectPtxTensorView[]> launch,
        params object?[] host)
    {
        Assert.Equal(blueprint.Tensors.Count, host.Length);
        var buffers = new DirectPtxBuffer[host.Length];
        var views = new DirectPtxTensorView[host.Length];
        var result = new object[host.Length];
        try
        {
            for (int i = 0; i < host.Length; i++)
            {
                DirectPtxTensorContract contract = blueprint.Tensors[i];
                buffers[i] = runtime.AllocateBytes(contract.RequiredBytes);
                views[i] = DirectPtxTensorView.CreateOwned(buffers[i], contract);
                if (host[i] is float[] floats) buffers[i].Upload<float>(floats);
                else if (host[i] is ushort[] halves) buffers[i].Upload<ushort>(halves);
                else Assert.False((contract.Access & DirectPtxTensorAccess.Read) != 0,
                    $"Missing host input for {contract.Name}.");
            }
            launch(views);
            runtime.Synchronize();
            for (int i = 0; i < host.Length; i++)
            {
                int elements = checked((int)blueprint.Tensors[i].PhysicalExtent.ElementCount);
                if (blueprint.Tensors[i].PhysicalType == DirectPtxPhysicalType.Float16)
                {
                    var values = new ushort[elements];
                    buffers[i].Download<ushort>(values);
                    result[i] = values;
                }
                else
                {
                    var values = new float[elements];
                    buffers[i].Download<float>(values);
                    result[i] = values;
                }
            }
            return result;
        }
        finally
        {
            for (int i = buffers.Length - 1; i >= 0; i--) buffers[i]?.Dispose();
        }
    }

    private static void RowStats(
        float[] input,
        out float[] mean,
        out float[] variance,
        out float[] invStd,
        out float[] rms,
        out float[] norm)
    {
        mean = new float[Rows];
        variance = new float[Rows];
        invStd = new float[Rows];
        rms = new float[Rows];
        norm = new float[Rows];
        for (int row = 0; row < Rows; row++)
        {
            int rowBase = row * Width;
            double sum = 0, sumSquares = 0;
            for (int feature = 0; feature < Width; feature++)
            {
                double value = input[rowBase + feature];
                sum += value;
                sumSquares += value * value;
            }
            double rowMean = sum / Width;
            double varianceSum = 0;
            for (int feature = 0; feature < Width; feature++)
            {
                double difference = input[rowBase + feature] - rowMean;
                varianceSum += difference * difference;
            }
            mean[row] = (float)rowMean;
            variance[row] = (float)(varianceSum / Width);
            invStd[row] = (float)(1 / Math.Sqrt(variance[row] + Epsilon));
            rms[row] = (float)Math.Sqrt(sumSquares / Width + Epsilon);
            norm[row] = (float)Math.Sqrt(sumSquares);
        }
    }

    private static void BatchStats(
        float[] input, int batch, int channels, int spatial,
        out float[] mean, out float[] variance, out float[] invStd)
    {
        mean = new float[channels];
        variance = new float[channels];
        invStd = new float[channels];
        int count = batch * spatial;
        for (int channel = 0; channel < channels; channel++)
        {
            double sum = 0;
            for (int n = 0; n < batch; n++)
                for (int s = 0; s < spatial; s++)
                    sum += input[(n * channels + channel) * spatial + s];
            mean[channel] = (float)(sum / count);
            double squareSum = 0;
            for (int n = 0; n < batch; n++)
                for (int s = 0; s < spatial; s++)
                {
                    double difference = input[(n * channels + channel) * spatial + s] - mean[channel];
                    squareSum += difference * difference;
                }
            variance[channel] = (float)(squareSum / count);
            invStd[channel] = 1f / PtxCompat.Sqrt(variance[channel] + Epsilon);
        }
    }

    private static float[] BatchAffine(
        float[] input, float[] gamma, float[] beta, float[] mean, float[] invStd,
        int batch, int channels, int spatial)
    {
        var output = new float[input.Length];
        for (int n = 0; n < batch; n++)
            for (int channel = 0; channel < channels; channel++)
                for (int s = 0; s < spatial; s++)
                {
                    int index = (n * channels + channel) * spatial + s;
                    output[index] = (input[index] - mean[channel]) * invStd[channel] *
                        gamma[channel] + beta[channel];
                }
        return output;
    }

    private static void GroupStats(
        float[] input, int batch, int channels, int groups, int spatial,
        out float[] mean, out float[] variance, out float[] invStd)
    {
        int channelsPerGroup = channels / groups;
        int count = channelsPerGroup * spatial;
        mean = new float[batch * groups];
        variance = new float[batch * groups];
        invStd = new float[batch * groups];
        for (int n = 0; n < batch; n++)
            for (int group = 0; group < groups; group++)
            {
                int unit = n * groups + group;
                double sum = 0;
                for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++)
                    for (int s = 0; s < spatial; s++)
                        sum += input[(n * channels + group * channelsPerGroup + localChannel) * spatial + s];
                mean[unit] = (float)(sum / count);
                double squareSum = 0;
                for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++)
                    for (int s = 0; s < spatial; s++)
                    {
                        int index = (n * channels + group * channelsPerGroup + localChannel) * spatial + s;
                        double difference = input[index] - mean[unit];
                        squareSum += difference * difference;
                    }
                variance[unit] = (float)(squareSum / count);
                invStd[unit] = 1f / PtxCompat.Sqrt(variance[unit] + Epsilon);
            }
    }

    private static float[] GroupAffine(
        float[] input, float[] gamma, float[] beta, float[] mean, float[] invStd,
        int batch, int channels, int groups, int spatial)
    {
        int channelsPerGroup = channels / groups;
        var output = new float[input.Length];
        for (int n = 0; n < batch; n++)
            for (int channel = 0; channel < channels; channel++)
            {
                int unit = n * groups + channel / channelsPerGroup;
                for (int s = 0; s < spatial; s++)
                {
                    int index = (n * channels + channel) * spatial + s;
                    output[index] = (input[index] - mean[unit]) * invStd[unit] *
                        gamma[channel] + beta[channel];
                }
            }
        return output;
    }

    private static void InstanceStats(
        float[] input, int batch, int channels, int spatial,
        out float[] mean, out float[] invStd)
    {
        mean = new float[batch * channels];
        invStd = new float[batch * channels];
        for (int n = 0; n < batch; n++)
            for (int channel = 0; channel < channels; channel++)
            {
                int unit = n * channels + channel;
                int offset = unit * spatial;
                double sum = 0;
                for (int s = 0; s < spatial; s++) sum += input[offset + s];
                mean[unit] = (float)(sum / spatial);
                double squareSum = 0;
                for (int s = 0; s < spatial; s++)
                {
                    double difference = input[offset + s] - mean[unit];
                    squareSum += difference * difference;
                }
                invStd[unit] = 1f / PtxCompat.Sqrt((float)(squareSum / spatial) + Epsilon);
            }
    }

    private static float[] InstanceAffine(
        float[] input, float[] gamma, float[] beta, float[] mean, float[] invStd,
        int batch, int channels, int spatial)
    {
        var output = new float[input.Length];
        for (int n = 0; n < batch; n++)
            for (int channel = 0; channel < channels; channel++)
            {
                int unit = n * channels + channel;
                for (int s = 0; s < spatial; s++)
                {
                    int index = unit * spatial + s;
                    output[index] = (input[index] - mean[unit]) * invStd[unit] *
                        gamma[channel] + beta[channel];
                }
            }
        return output;
    }

    private static void NormalizationBackward(
        float[] input, float[] gradOutput, float[] gamma, float[] mean, float[] invStd,
        int batch, int channels, int spatial, int groupCount,
        out float[] gradInput, out float[] gradGamma, out float[] gradBeta)
    {
        int channelsPerGroup = channels / groupCount;
        int count = channelsPerGroup * spatial;
        gradInput = new float[input.Length];
        gradGamma = new float[channels];
        gradBeta = new float[channels];
        for (int n = 0; n < batch; n++)
            for (int group = 0; group < groupCount; group++)
            {
                int unit = n * groupCount + group;
                double sumDyGamma = 0, sumDyGammaXhat = 0;
                for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++)
                {
                    int channel = group * channelsPerGroup + localChannel;
                    for (int s = 0; s < spatial; s++)
                    {
                        int index = (n * channels + channel) * spatial + s;
                        double xhat = (input[index] - mean[unit]) * invStd[unit];
                        double dyGamma = gradOutput[index] * gamma[channel];
                        sumDyGamma += dyGamma;
                        sumDyGammaXhat += dyGamma * xhat;
                        gradGamma[channel] += (float)(gradOutput[index] * xhat);
                        gradBeta[channel] += gradOutput[index];
                    }
                }
                for (int localChannel = 0; localChannel < channelsPerGroup; localChannel++)
                {
                    int channel = group * channelsPerGroup + localChannel;
                    for (int s = 0; s < spatial; s++)
                    {
                        int index = (n * channels + channel) * spatial + s;
                        double xhat = (input[index] - mean[unit]) * invStd[unit];
                        double dyGamma = gradOutput[index] * gamma[channel];
                        gradInput[index] = (float)(invStd[unit] / count *
                            (count * dyGamma - sumDyGamma - xhat * sumDyGammaXhat));
                    }
                }
            }
    }

    private static void BatchBackward(
        float[] input, float[] gradOutput, float[] gamma, float[] mean, float[] invStd,
        int batch, int channels, int spatial,
        out float[] gradInput, out float[] gradGamma, out float[] gradBeta)
    {
        int count = batch * spatial;
        gradInput = new float[input.Length];
        gradGamma = new float[channels];
        gradBeta = new float[channels];
        for (int channel = 0; channel < channels; channel++)
        {
            double sumDyGamma = 0, sumDyGammaXhat = 0;
            for (int n = 0; n < batch; n++)
                for (int s = 0; s < spatial; s++)
                {
                    int index = (n * channels + channel) * spatial + s;
                    double xhat = (input[index] - mean[channel]) * invStd[channel];
                    double dyGamma = gradOutput[index] * gamma[channel];
                    sumDyGamma += dyGamma;
                    sumDyGammaXhat += dyGamma * xhat;
                    gradGamma[channel] += (float)(gradOutput[index] * xhat);
                    gradBeta[channel] += gradOutput[index];
                }
            for (int n = 0; n < batch; n++)
                for (int s = 0; s < spatial; s++)
                {
                    int index = (n * channels + channel) * spatial + s;
                    double xhat = (input[index] - mean[channel]) * invStd[channel];
                    double dyGamma = gradOutput[index] * gamma[channel];
                    gradInput[index] = (float)(invStd[channel] / count *
                        (count * dyGamma - sumDyGamma - xhat * sumDyGammaXhat));
                }
        }
    }

    private static float[] Activate(
        float[] input, DirectPtxChannelNormalizationOperation operation)
    {
        var output = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            float value = input[i];
            output[i] = operation switch
            {
                DirectPtxChannelNormalizationOperation.BatchNormRelu => PtxCompat.Max(0, value),
                DirectPtxChannelNormalizationOperation.BatchNormGelu =>
                    0.5f * value * (1f + PtxCompat.Tanh(0.7978845608f *
                        (value + 0.044715f * value * value * value))),
                DirectPtxChannelNormalizationOperation.BatchNormSigmoid =>
                    1f / (1f + PtxCompat.Exp(-value)),
                DirectPtxChannelNormalizationOperation.BatchNormTanh => PtxCompat.Tanh(value),
                DirectPtxChannelNormalizationOperation.GroupNormSwish =>
                    value / (1f + PtxCompat.Exp(-value)),
                _ => value
            };
        }
        return output;
    }

    private static float[] Values(int length, int period, float scale, float offset = 0)
    {
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (i % period - period / 2) * scale + offset;
        return values;
    }

    private static ushort[] ToHalf(float[] values)
    {
        var result = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = HalfBits.GetBits((Half)values[i]);
        return result;
    }

    private static float[] FromHalf(ushort[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            result[i] = (float)HalfBits.FromBits(values[i]);
        return result;
    }

    private static void AssertClose(float[] actual, float[] expected, float tolerance, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maximum = 0;
        int maximumIndex = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            float error = PtxCompat.Abs(actual[i] - expected[i]);
            if (error > maximum) { maximum = error; maximumIndex = i; }
        }
        Assert.True(maximum <= tolerance,
            $"{name} max absolute error {maximum:G9} at {maximumIndex} " +
            $"(actual {actual[maximumIndex]:G9}, expected {expected[maximumIndex]:G9}); " +
            $"tolerance {tolerance:G9}.");
    }
}
