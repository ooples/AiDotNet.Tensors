using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Small deterministic targets for Nsight Compute release evidence.</summary>
internal static class DirectPtxProfileTarget
{
    internal static void RunNormalization()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-normalization-start");
        bool previousExperiment = DirectPtxFeatureGate.NormalizationExperimentOverride;
        DirectPtxFeatureGate.NormalizationExperimentOverride = true;
        try
        {
            using var runtime = new DirectPtxRuntime();
            int launches = 0;
            foreach (int rows in new[] { 256, 2_048, 8_192 })
            foreach (DirectPtxRowNormalizationOperation operation in
                Enum.GetValues<DirectPtxRowNormalizationOperation>())
            {
                using var kernel = new PtxRowNormalizationD64Kernel(
                    runtime, operation, rows, 1e-5f);
                LaunchNormalizationKernel(runtime, kernel.Blueprint, kernel.Audit, kernel.Launch);
                launches++;
            }

            foreach (DirectPtxChannelNormalizationOperation operation in
                Enum.GetValues<DirectPtxChannelNormalizationOperation>())
            {
                using var kernel = new PtxChannelNormalizationD64Kernel(
                    runtime, operation, 1e-5f, 0.1f);
                LaunchNormalizationKernel(runtime, kernel.Blueprint, kernel.Audit, kernel.Launch);
                launches++;
            }

            runtime.Synchronize();
            Console.WriteLine($"PROFILED_NORMALIZATION_CUBINS={launches}");
        }
        finally
        {
            DirectPtxFeatureGate.NormalizationExperimentOverride = previousExperiment;
        }
        // Do not re-run the idle-GPU gate here: writing 71 audit records can make
        // a hardware-accelerated terminal briefly appear as a compute+graphics
        // process. The pre-launch gate protects the actual profiled interval.
    }

    private static void LaunchNormalizationKernel(
        DirectPtxRuntime runtime,
        DirectPtxKernelBlueprint blueprint,
        DirectPtxKernelAudit audit,
        Action<ReadOnlySpan<DirectPtxTensorView>> launch)
    {
        if (audit.ImageKind != DirectPtxModuleImageKind.EmbeddedCubin)
            throw new InvalidOperationException(
                $"Nsight target requires an embedded cubin, but {blueprint.Id} loaded {audit.ImageKind}.");

        var buffers = new DirectPtxBuffer[blueprint.Tensors.Count];
        var views = new DirectPtxTensorView[blueprint.Tensors.Count];
        try
        {
            for (int i = 0; i < blueprint.Tensors.Count; i++)
            {
                DirectPtxTensorContract contract = blueprint.Tensors[i];
                buffers[i] = runtime.AllocateBytes(contract.RequiredBytes);
                buffers[i].Upload<byte>(new byte[checked((int)contract.RequiredBytes)]);
                views[i] = DirectPtxTensorView.CreateOwned(buffers[i], contract);
            }
            launch(views);
            Console.WriteLine(audit.ToJson());
        }
        finally
        {
            for (int i = buffers.Length - 1; i >= 0; i--)
                buffers[i]?.Dispose();
        }
    }

    internal static void RunAttention()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-attention-start");
        using var runtime = new DirectPtxRuntime();
        foreach (int sequence in new[] { 16, 32, 64, 128 })
        foreach (bool causal in new[] { false, true })
        foreach (bool fused in new[] { false, true })
        {
            using var kernel = new PtxOnlineFusedAttention128x64Kernel(
                runtime, 128, causal, fused,
                sequenceLength: sequence, emitSoftmaxStats: false);
            using var q = runtime.AllocateBytes(kernel.QBytes);
            using var k = runtime.AllocateBytes(kernel.KBytes);
            using var v = runtime.AllocateBytes(kernel.VBytes);
            using var gamma = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
            using var beta = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);
            var input = new ushort[128 * sequence * 64];
            q.Upload<ushort>(input);
            k.Upload<ushort>(input);
            v.Upload<ushort>(input);
            gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
            beta.Upload<float>(new float[64]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]),
                default);
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        runtime.Synchronize();
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-attention-end");
    }

    internal static void RunResidualRmsNorm()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-residual-rmsnorm-start");
        using var runtime = new DirectPtxRuntime();
        foreach (int rows in new[] { 32, 256, 2048, 8192 })
        {
            using var kernel = new PtxFusedResidualRmsNormD64Kernel(runtime, rows);
            using var input = runtime.AllocateBytes(kernel.InputBytes);
            using var residual = runtime.AllocateBytes(kernel.InputBytes);
            using var gamma = runtime.AllocateBytes(PtxFusedResidualRmsNormD64Kernel.GammaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);
            using var rms = runtime.AllocateBytes(kernel.RmsBytes);
            input.Upload<float>(new float[rows * 64]);
            residual.Upload<float>(new float[rows * 64]);
            gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(rms, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-residual-rmsnorm-end");
    }

    internal static void RunDecode()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-decode-start");
        using var runtime = new DirectPtxRuntime();
        const int heads = 8, kvHeads = 1, sequence = 128, blockSize = 16, poolBlocks = 10;
        using var dense = new PtxFusedDecodeAttentionD64Kernel(
            runtime, false, heads, kvHeads, sequence, 0, 0, 0.125f);
        using var paged = new PtxFusedDecodeAttentionD64Kernel(
            runtime, true, heads, kvHeads, sequence, blockSize, poolBlocks, 0.125f);
        using var q = runtime.AllocateBytes(dense.QueryBytes);
        using var denseK = runtime.AllocateBytes(dense.KeyValueBytes);
        using var denseV = runtime.AllocateBytes(dense.KeyValueBytes);
        using var denseOutput = runtime.AllocateBytes(dense.OutputBytes);
        using var pagedK = runtime.AllocateBytes(paged.KeyValueBytes);
        using var pagedV = runtime.AllocateBytes(paged.KeyValueBytes);
        using var table = runtime.AllocateBytes(paged.BlockTableBytes);
        using var pagedOutput = runtime.AllocateBytes(paged.OutputBytes);
        q.Upload<float>(new float[heads * 64]);
        denseK.Upload<float>(new float[sequence * kvHeads * 64]);
        denseV.Upload<float>(new float[sequence * kvHeads * 64]);
        pagedK.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        pagedV.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        table.Upload<int>(Enumerable.Range(0, sequence / blockSize).Reverse().ToArray());
        Action launchDense = () => dense.LaunchDense(
            DirectPtxTensorView.CreateOwned(q, dense.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(denseK, dense.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(denseV, dense.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(denseOutput, dense.Blueprint.Tensors[3]));
        Action launchPaged = () => paged.LaunchPaged(
            DirectPtxTensorView.CreateOwned(q, paged.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(pagedK, paged.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(pagedV, paged.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(table, paged.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(pagedOutput, paged.Blueprint.Tensors[4]));
        launchDense();
        launchPaged();
        runtime.Synchronize();
        Console.WriteLine(dense.Audit.ToJson());
        Console.WriteLine(paged.Audit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-decode-end");
    }

    internal static void RunPagedPrefill()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-paged-prefill-start");
        using var runtime = new DirectPtxRuntime();
        const int heads = 8, kvHeads = 2, queries = 16, start = 112;
        const int blockSize = 16, poolBlocks = 10;
        using var kernel = new PtxFusedPagedPrefillAttentionD64Kernel(
            runtime, heads, kvHeads, queries, start,
            blockSize, poolBlocks, 0.125f);
        using var q = runtime.AllocateBytes(kernel.QueryBytes);
        using var k = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var v = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var table = runtime.AllocateBytes(kernel.BlockTableBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        q.Upload<float>(new float[queries * heads * 64]);
        k.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        v.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        table.Upload<int>(Enumerable.Range(2, 8).Reverse().ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(table, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-paged-prefill-end");
    }

    internal static void RunAttentionBackward()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-attention-backward-start");
        using var runtime = new DirectPtxRuntime();
        const int batch = 1, heads = 8, kvHeads = 2, querySequence = 64, keySequence = 64;
        using var kernel = new PtxFusedAttentionBackwardD64Kernel(
            runtime, batch, heads, kvHeads, querySequence, keySequence, 0.125f);
        using var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var key = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var value = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var probabilities = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var gradQuery = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        using var gradKey = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
        using var gradValue = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
        gradOutput.Upload<float>(new float[batch * heads * querySequence * 64]);
        query.Upload<float>(new float[batch * heads * querySequence * 64]);
        key.Upload<float>(new float[batch * kvHeads * keySequence * 64]);
        value.Upload<float>(new float[batch * kvHeads * keySequence * 64]);
        probabilities.Upload<float>(Enumerable.Repeat(
            1f / keySequence, batch * heads * querySequence * keySequence).ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(probabilities, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[7]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.RowDeltaAudit.ToJson());
        Console.WriteLine(kernel.GradKeyValueAudit.ToJson());
        Console.WriteLine(kernel.GradQueryAudit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-attention-backward-end");
    }

    internal static void RunFlashAttentionBackward()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-flash-attention-backward-start");
        using var runtime = new DirectPtxRuntime();
        const int batch = 1, heads = 8, querySequence = 64, keySequence = 64;
        using var kernel = new PtxFlashAttentionBackwardD64Kernel(
            runtime, batch, heads, querySequence, keySequence,
            0.125f, isCausal: true);
        using var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var key = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var value = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var stats = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        using var gradQuery = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
        using var gradKey = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
        using var gradValue = runtime.AllocateBytes(kernel.Blueprint.Tensors[8].RequiredBytes);
        gradOutput.Upload<float>(new float[batch * heads * querySequence * 64]);
        query.Upload<float>(new float[batch * heads * querySequence * 64]);
        key.Upload<float>(new float[batch * heads * keySequence * 64]);
        value.Upload<float>(new float[batch * heads * keySequence * 64]);
        output.Upload<float>(new float[batch * heads * querySequence * 64]);
        stats.Upload<float>(Enumerable.Repeat(
            MathF.Log(keySequence), batch * heads * querySequence).ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(stats, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[7]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[8]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.GradQueryAudit.ToJson());
        Console.WriteLine(kernel.GradKeyValueAudit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-flash-attention-backward-end");
    }

    internal static void RunQkvRopeCache()
    {
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-qkv-rope-cache-start");
        using var runtime = new DirectPtxRuntime();
        (int Heads, int Capacity, int Position)[] shapes =
        [
            (4, 16, 0),
            (8, 64, 17),
            (16, 128, 127)
        ];
        foreach ((int heads, int capacity, int position) in shapes)
        {
            using var kernel = new PtxFusedQkvRopeCacheD64Kernel(
                runtime, heads, capacity, position);
            int model = heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
            using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            using var cosine = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
            using var sine = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
            using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
            using var keyCache = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
            using var valueCache = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
            input.Upload<float>(new float[model]);
            weights.Upload<float>(new float[3 * model * model]);
            bias.Upload<float>(new float[3 * model]);
            cosine.Upload<float>(Enumerable.Repeat(1f, capacity * 32).ToArray());
            sine.Upload<float>(new float[capacity * 32]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(cosine, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(sine, kernel.Blueprint.Tensors[4]),
                DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[5]),
                DirectPtxTensorView.CreateOwned(keyCache, kernel.Blueprint.Tensors[6]),
                DirectPtxTensorView.CreateOwned(valueCache, kernel.Blueprint.Tensors[7]));
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-qkv-rope-cache-end");
    }

    internal static void RunResidualLayerNormGelu()
    {
        using var runtime = new DirectPtxRuntime();
        const int rows = 8192;
        using var kernel = new PtxFusedResidualBiasLayerNormGeluD64Kernel(runtime, rows);
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var residual = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var gamma = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var beta = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        input.Upload<float>(new float[rows * 64]);
        residual.Upload<float>(new float[rows * 64]);
        bias.Upload<float>(new float[64]);
        gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
        beta.Upload<float>(new float[64]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunFusedLinear()
    {
        using var runtime = new DirectPtxRuntime();
        const int inputFeatures = 512, outputFeatures = 2048;
        using var kernel = new PtxFusedLinearGeluM1Kernel(
            runtime, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        input.Upload<float>(new float[inputFeatures]);
        weights.Upload<float>(new float[inputFeatures * outputFeatures]);
        bias.Upload<float>(new float[outputFeatures]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunMixedLinear()
    {
        using var runtime = new DirectPtxRuntime();
        const int inputFeatures = 512, outputFeatures = 2048;
        using var kernel = new PtxFusedLinearGeluFp16M1Kernel(
            runtime, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        input.Upload<ushort>(new ushort[inputFeatures]);
        weights.Upload<ushort>(new ushort[inputFeatures * outputFeatures]);
        bias.Upload<float>(new float[outputFeatures]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunMixedLinearM16()
    {
        using var runtime = new DirectPtxRuntime();
        const int rows = PtxFusedLinearGeluFp16M16Kernel.Rows;
        const int inputFeatures = 1024, outputFeatures = 4096;
        using var kernel = new PtxFusedLinearGeluFp16M16Kernel(
            runtime, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        input.Upload<ushort>(new ushort[rows * inputFeatures]);
        weights.Upload<ushort>(new ushort[inputFeatures * outputFeatures]);
        bias.Upload<float>(new float[outputFeatures]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunW8A8Linear()
    {
        using var runtime = new DirectPtxRuntime();
        const int inputFeatures = 1024, outputFeatures = 4096;
        using var kernel = new PtxFusedLinearGeluW8A8M1Kernel(
            runtime, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var activationScale = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var weightScales = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        input.Upload<sbyte>(new sbyte[inputFeatures]);
        weights.Upload<sbyte>(new sbyte[inputFeatures * outputFeatures]);
        activationScale.Upload<float>([0.01f]);
        weightScales.Upload<float>(Enumerable.Repeat(0.005f, outputFeatures).ToArray());
        bias.Upload<float>(new float[outputFeatures]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(activationScale, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(weightScales, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void VerifyNcuCsv(string path)
    {
        DirectPtxProfilerEvidence evidence = DirectPtxProfilerEvidence.FromNcuCsv(path);
        Console.WriteLine($"source: {evidence.Source}");
        Console.WriteLine($"register-spill instructions: {evidence.RegisterSpillInstructions}");
        Console.WriteLine($"local-load instructions: {evidence.LocalLoadInstructions}");
        Console.WriteLine($"local-store instructions: {evidence.LocalStoreInstructions}");
        Console.WriteLine($"required metric groups observed: {evidence.ObservedMetricGroups}/3");
        Console.WriteLine(evidence.ProvesZeroExecutedSpills ? "PASS: zero executed spills" : "FAIL: spill/local traffic detected");
        if (!evidence.ProvesZeroExecutedSpills) Environment.ExitCode = 2;
    }
}
