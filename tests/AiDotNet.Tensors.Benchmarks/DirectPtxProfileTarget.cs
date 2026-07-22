using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Small deterministic targets for Nsight Compute release evidence.</summary>
internal static class DirectPtxProfileTarget
{
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

    internal static void RunRngDropout()
    {
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-rng-dropout-start");
        using var runtime = new DirectPtxRuntime();
        foreach (int elements in new[] { 4_096, 65_536, 1_048_576 })
        {
            using var kernel = new PtxFusedPhiloxDropoutF32Kernel(runtime, elements);
            using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var mask = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            input.Upload<float>(new float[elements]);
            const float dropoutRate = 0.1f;
            float keep = 1.0f - dropoutRate;
            uint threshold = (uint)Math.Floor((double)keep * 4_294_967_296.0);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(mask, kernel.Blueprint.Tensors[2]),
                seed: 0x8490_1234_5678_9ABCul,
                subsequence: 0,
                counterOffset: 0,
                keepThreshold: threshold,
                inverseKeep: 1.0f / keep);
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-rng-dropout-end");
    }

    /// <summary>
    /// Emits exactly one launch for every issue-#849 direct-PTX specialization.
    /// This is deliberately separate from the dropout-only compatibility target:
    /// promotion evidence must cover the complete stochastic-kernel set.
    /// </summary>
    internal static void RunRngStochastic()
    {
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-rng-stochastic-start");
        using var runtime = new DirectPtxRuntime();
        const ulong seed = 0x8490_1234_5678_9ABCul;
        const float dropoutRate = 0.1f;
        float keep = 1.0f - dropoutRate;
        uint threshold = (uint)Math.Floor((double)keep * 4_294_967_296.0);

        foreach (int elements in new[] { 4_096, 65_536, 1_048_576 })
        {
            using (var kernel = new PtxFusedPhiloxDropoutF32Kernel(runtime, elements))
            using (var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var mask = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                input.Upload<float>(new float[elements]);
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(mask, kernel.Blueprint.Tensors[2]),
                    seed, 0, 0, threshold, 1.0f / keep);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            foreach (DirectPtxPhiloxFillKind kind in Enum.GetValues<DirectPtxPhiloxFillKind>())
            {
                using var kernel = new PtxPhiloxFillF32Kernel(runtime, kind, elements);
                using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
                DirectPtxTensorView outputView =
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[0]);
                if (kind is DirectPtxPhiloxFillKind.BernoulliMask or
                    DirectPtxPhiloxFillKind.DropThresholdMask)
                    kernel.LaunchMask(outputView, seed, 0, 0, threshold, 1.0f);
                else
                    kernel.LaunchRange(outputView, seed, 0, 0,
                        kind == DirectPtxPhiloxFillKind.Uniform ? -1.0f : 0.0f,
                        kind == DirectPtxPhiloxFillKind.Uniform ? 1.0f : 1.0f);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxDropoutBackwardF32Kernel(runtime, elements))
            using (var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var mask = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var gradInput = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                gradOutput.Upload<float>(new float[elements]);
                mask.Upload<float>(Enumerable.Repeat(1.0f, elements).ToArray());
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(mask, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(gradInput, kernel.Blueprint.Tensors[2]));
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxFusedDdimStepF32Kernel(runtime, elements))
            using (var xT = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var epsilon = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                xT.Upload<float>(new float[elements]);
                epsilon.Upload<float>(new float[elements]);
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(xT, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(epsilon, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]),
                    xCoefficient: 0.95f, epsilonCoefficient: -0.05f);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxFusedPhiloxRreluF32Kernel(runtime, elements))
            using (var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var noise = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                input.Upload<float>(new float[elements]);
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(noise, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]),
                    seed, 0, 0, 0.125f, 1.0f / 3.0f);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxRreluF32Kernel(
                       runtime, DirectPtxRreluKind.Forward, elements))
            using (var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var noise = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                input.Upload<float>(new float[elements]);
                noise.Upload<float>(Enumerable.Repeat(0.2f, elements).ToArray());
                kernel.LaunchForward(
                    DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(noise, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxRreluF32Kernel(
                       runtime, DirectPtxRreluKind.Backward, elements))
            using (var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var noise = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            using (var gradInput = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes))
            {
                gradOutput.Upload<float>(Enumerable.Repeat(1.0f, elements).ToArray());
                input.Upload<float>(new float[elements]);
                noise.Upload<float>(Enumerable.Repeat(0.2f, elements).ToArray());
                kernel.LaunchBackward(
                    DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(noise, kernel.Blueprint.Tensors[2]),
                    DirectPtxTensorView.CreateOwned(gradInput, kernel.Blueprint.Tensors[3]));
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }
        }

        foreach (int rows in new[] { 128, 2_048, 32_768 })
        {
            int elements = checked(rows * PtxFusedGumbelSoftmax32F32Kernel.InnerSize);
            using (var kernel = new PtxFusedGumbelSoftmax32F32Kernel(runtime, rows))
            using (var logits = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            {
                logits.Upload<float>(new float[elements]);
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(logits, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]),
                    seed, 0, 0, temperature: 0.7f);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxPhiloxCategorical32F32Kernel(runtime, rows))
            using (var probabilities = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var oneHot = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            {
                probabilities.Upload<float>(Enumerable.Repeat(1.0f / 32.0f, elements).ToArray());
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(probabilities, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(oneHot, kernel.Blueprint.Tensors[1]),
                    seed, 0, 0);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }

            using (var kernel = new PtxGumbelSoftmaxBackward32F32Kernel(runtime, rows))
            using (var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes))
            using (var softOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes))
            using (var gradInput = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes))
            {
                gradOutput.Upload<float>(Enumerable.Repeat(1.0f, elements).ToArray());
                softOutput.Upload<float>(Enumerable.Repeat(1.0f / 32.0f, elements).ToArray());
                kernel.Launch(
                    DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.CreateOwned(softOutput, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.CreateOwned(gradInput, kernel.Blueprint.Tensors[2]),
                    temperature: 0.7f);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }
        }

        foreach (int rays in new[] { 64, 1_024, 16_384 })
        {
            int elements = checked(rays * PtxFusedImportanceSampling64F32Kernel.Samples);
            using var kernel = new PtxFusedImportanceSampling64F32Kernel(runtime, rays);
            using var tValues = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            tValues.Upload<float>(Enumerable.Range(0, elements)
                .Select(index => (index % 64) / 63.0f).ToArray());
            weights.Upload<float>(Enumerable.Repeat(1.0f, elements).ToArray());
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(tValues, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]),
                seed, 0, 0);
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }

        foreach (int rows in new[] { 16, 256, 4_096 })
        {
            int elements = checked(rows * PtxFusedBiasPhiloxDropout256F32Kernel.Columns);
            using var kernel = new PtxFusedBiasPhiloxDropout256F32Kernel(runtime, rows);
            using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            using var mask = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
            input.Upload<float>(new float[elements]);
            bias.Upload<float>(new float[PtxFusedBiasPhiloxDropout256F32Kernel.Columns]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(mask, kernel.Blueprint.Tensors[3]),
                seed, 0, 0, threshold, 1.0f / keep);
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }

        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-rng-stochastic-end");
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
