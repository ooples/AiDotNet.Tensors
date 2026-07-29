// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Times a generated kernel against the hand-written backend kernel a caller gets today.
/// </summary>
/// <remarks>
/// <para>
/// THIS IS WHAT UNBLOCKS PROMOTION, and it is not the same as wiring. Every capability closed
/// on this branch already has an incumbent: gather and scatter have
/// <c>embedding_forward</c> and <c>embedding_backward</c>, the optimizer family has
/// <c>adam_update</c> and friends, reductions have their own kernels. Dispatching a generated
/// kernel without this number replaces a known quantity with an unknown one.
/// </para>
/// <para>
/// Blueprint section 3 says the competitor is the best thing a user could otherwise call, and
/// on this repository that is frequently us. PR #874 measured a hand-written fused SGD against
/// the existing kernel and found a TIE at 0.73-1.05x, because that kernel was already
/// single-pass. The work was real; the win was not available. That outcome is only visible
/// from a head-to-head.
/// </para>
/// <para>
/// Both sides go through <see cref="StableTimer.MeasureHostPair"/> as adjacent batches, so a
/// ratio is only printed when both measurements AND the within-sample ratio converged. A
/// promotion decision taken on numbers that move 50% between runs is a coin flip wearing a
/// table.
/// </para>
/// </remarks>
internal static class HeadToHeadTool
{
    internal static void Run(string[] args)
    {
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("CUDA backend unavailable; nothing to compare against.");
            return;
        }

        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;

        Console.WriteLine();
        Console.WriteLine("HEAD TO HEAD - generated kernel against the kernel a caller gets today");
        Console.WriteLine("device: {0}", backend.DeviceName);
        Console.WriteLine("protocol: {0}", CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("{0,-44} {1,15} {2,15} {3,9}  {4}",
            "operator", "existing", "generated", "ratio", "verdict");

        ElementwiseBinary(backend, runtime, major, minor,
            "vector add 4M (vec4 incumbent)", "h2h_add", multiply: false);
        ElementwiseBinary(backend, runtime, major, minor,
            "vector multiply 4M (vec4 incumbent)", "h2h_multiply", multiply: true);
        ElementwiseUnary(backend, runtime, major, minor,
            "ReLU 4M (vec4 incumbent)", "h2h_relu", CodegenActivationKind.ReLU,
            static (b, input, output, count) => b.Relu(input, output, count));
        ElementwiseUnary(backend, runtime, major, minor,
            "sigmoid 4M (vec4 incumbent)", "h2h_sigmoid", CodegenActivationKind.Sigmoid,
            static (b, input, output, count) => b.Sigmoid(input, output, count));
        ElementwiseUnary(backend, runtime, major, minor,
            "tanh 4M (vec4 incumbent)", "h2h_tanh", CodegenActivationKind.Tanh,
            static (b, input, output, count) => b.Tanh(input, output, count));
        ElementwiseUnary(backend, runtime, major, minor,
            "GELU 4M (vec4 incumbent)", "h2h_gelu", CodegenActivationKind.Gelu,
            static (b, input, output, count) => b.Gelu(input, output, count));
        EmbeddingForward(backend, runtime, major, minor);
        EmbeddingBackward(backend, runtime, major, minor);
        SgdMomentum(backend, runtime, major, minor);
        RowReduction(backend, runtime, major, minor,
            "row sum 4096x1024", "h2h_rowsum", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.SumAxis(input, output, rows, inner));
        RowReduction(backend, runtime, major, minor,
            "row mean 4096x1024", "h2h_rowmean", CodegenReduceKind.Sum,
            static (b, input, output, rows, inner) => b.MeanAxis(input, output, rows, inner),
            reduceScale: 1.0 / 1024.0);
        RowReduction(backend, runtime, major, minor,
            "row max 4096x1024", "h2h_rowmax", CodegenReduceKind.Max,
            static (b, input, output, rows, inner) => b.MaxAxis(input, output, rows, inner));

        ReportUnavailable("Adam / AdamW",
            "generator cannot express grad^2 plus m/v/param updates yet");

        Console.WriteLine();
        Console.WriteLine("ratio > 1 means the generated kernel is faster. A promotion needs BOTH");
        Console.WriteLine("sides stable and a ratio clear of the noise floor -- 1.05x from samples");
        Console.WriteLine("that disagree by 5% is not a win, it is a tie with a rounding error.");
    }

    private static void ElementwiseUnary(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor,
        string label, string specName, CodegenActivationKind activation,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer, int> launchExisting)
    {
        const int Count = 1 << 22;
        var input = Values(Count, 61, 0.03125f);
        using var inputBuffer = backend.AllocateBuffer(input);
        using var outputBuffer = backend.AllocateBuffer(Count);

        void Existing() => launchExisting(backend, inputBuffer, outputBuffer, Count);
        Existing();
        backend.Synchronize();
        var existingOnce = backend.DownloadBuffer(outputBuffer);

        long workUnits = (long)Count * sizeof(float) * 2;
        var timing = MeasureGenerated(
            runtime, major, minor,
            IncumbentSemanticSpecs.Unary(specName, Count, activation),
            indexData: Array.Empty<int>(),
            floatInputs: new Dictionary<int, float[]> { [0] = input },
            launchExisting: Existing,
            synchronizeExisting: backend.Synchronize,
            existingWorkUnits: workUnits,
            generatedWorkUnits: workUnits,
            zeroOutput: false,
            out var generatedOnce);

        double error = RelativeError(existingOnce, generatedOnce);
        Report(label, timing,
            qualification: error <= 2e-5
                ? null
                : "NOT EQUIVALENT -- relative error " +
                  error.ToString("E2", CultureInfo.InvariantCulture));
    }

    private static void ElementwiseBinary(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor,
        string label, string specName, bool multiply)
    {
        const int Count = 1 << 22;
        var left = Values(Count, 67, 0.03125f);
        var right = Values(Count, 71, 0.015625f);
        using var leftBuffer = backend.AllocateBuffer(left);
        using var rightBuffer = backend.AllocateBuffer(right);
        using var outputBuffer = backend.AllocateBuffer(Count);

        void Existing()
        {
            if (multiply) backend.Multiply(leftBuffer, rightBuffer, outputBuffer, Count);
            else backend.Add(leftBuffer, rightBuffer, outputBuffer, Count);
        }

        Existing();
        backend.Synchronize();
        var existingOnce = backend.DownloadBuffer(outputBuffer);

        long workUnits = (long)Count * sizeof(float) * 3;
        CodegenKernelSpec spec = multiply
            ? IncumbentSemanticSpecs.Multiply(specName, Count)
            : IncumbentSemanticSpecs.Add(specName, Count);
        var timing = MeasureGenerated(
            runtime, major, minor, spec,
            indexData: Array.Empty<int>(),
            floatInputs: new Dictionary<int, float[]> { [0] = left, [1] = right },
            launchExisting: Existing,
            synchronizeExisting: backend.Synchronize,
            existingWorkUnits: workUnits,
            generatedWorkUnits: workUnits,
            zeroOutput: false,
            out var generatedOnce);

        double error = RelativeError(existingOnce, generatedOnce);
        Report(label, timing,
            qualification: error <= 2e-6
                ? null
                : "NOT EQUIVALENT -- relative error " +
                  error.ToString("E2", CultureInfo.InvariantCulture));
    }

    /// <summary>Embedding lookup: the generated gather against <c>embedding_forward</c>.</summary>
    private static void EmbeddingForward(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor)
    {
        const int Tokens = 1 << 20, Vocabulary = 4096, Width = 64;

        // TRUE int32 INDICES. embedding_forward takes const int*, and its own source carries a
        // comment about the bug from getting this wrong: float-interpreting an int32 bit
        // pattern turns small token IDs into denormals that truncate to 0, and other patterns
        // into huge or negative offsets that read outside the table. The first run of this
        // harness did exactly that and tripped a sticky CUDA-700.
        var ids = new int[Tokens];
        // The long multiply is also load-bearing: int overflow here creates negative IDs.
        for (int t = 0; t < Tokens; t++) ids[t] = (int)((t * 7919L) % Vocabulary);
        var table = new float[(long)Vocabulary * Width];
        for (int e = 0; e < table.Length; e++) table[e] = ((e * 37) % 97 - 48) / 16.0f;

        using var idsBuffer = backend.AllocateIntBuffer(ids);
        using var tableBuffer = backend.AllocateBuffer(table);
        using var outBuffer = backend.AllocateBuffer(Tokens * Width);

        void LaunchExisting() => backend.Embedding(
            idsBuffer, tableBuffer, outBuffer, Tokens, Width);

        // A fast kernel that reads uninitialised memory proves nothing. Check one full-size
        // invocation before timing, then time the same initialized inputs on both sides.
        LaunchExisting();
        backend.Synchronize();
        var existingOnce = backend.DownloadBuffer(outBuffer);

        long workUnits = ((long)Tokens * Width * 2 + Tokens) * sizeof(float);
        var timing = MeasureGenerated(
            runtime, major, minor,
            IncumbentSemanticSpecs.Gather("h2h_gather", Tokens, Vocabulary, Width),
            indexData: ids,
            floatInputs: new Dictionary<int, float[]> { [1] = table },
            launchExisting: LaunchExisting,
            synchronizeExisting: backend.Synchronize,
            existingWorkUnits: workUnits,
            generatedWorkUnits: workUnits,
            zeroOutput: false,
            out var generatedOnce);

        double error = RelativeError(existingOnce, generatedOnce);
        Report("embedding forward 1M x 64", timing,
            qualification: error <= 1e-7
                ? null
                : "NOT EQUIVALENT -- relative error "
                  + error.ToString("E2", CultureInfo.InvariantCulture));
    }

    /// <summary>The backward: the generated scatter against <c>embedding_backward</c>.</summary>
    private static void EmbeddingBackward(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor)
    {
        const int Tokens = 1 << 20, Vocabulary = 4096, Width = 64;

        // TRUE int32 INDICES. embedding_forward takes const int*, and its own source carries a
        // comment about the bug from getting this wrong: float-interpreting an int32 bit
        // pattern turns small token IDs into denormals that truncate to 0, and other patterns
        // into huge or negative offsets that read outside the table. The first run of this
        // harness did exactly that and tripped a sticky CUDA-700.
        var ids = new int[Tokens];
        // The long multiply is also load-bearing: int overflow here creates negative IDs.
        for (int t = 0; t < Tokens; t++) ids[t] = (int)((t * 7919L) % Vocabulary);
        var grad = new float[(long)Tokens * Width];
        for (int e = 0; e < grad.Length; e++) grad[e] = ((e * 37) % 97 - 48) / 16.0f;

        using var idsBuffer = backend.AllocateIntBuffer(ids);
        using var gradBuffer = backend.AllocateBuffer(grad);
        using var tableBuffer = backend.AllocateBuffer(Vocabulary * Width);

        void LaunchExisting() => backend.EmbeddingBackward(
            gradBuffer, idsBuffer, tableBuffer, Tokens, Width, Vocabulary);

        LaunchExisting();
        backend.Synchronize();
        var existingOnce = backend.DownloadBuffer(tableBuffer);

        // The deterministic incumbent performs V*D*N predicate checks. Use that actual work
        // to select its batch size; pretending it is O(ND) made a single harness run spend
        // five minutes repeating a 320 ms kernel almost 300 times per sample.
        long existingWorkUnits = (long)Vocabulary * Width * Tokens;
        long generatedWorkUnits =
            ((long)Tokens * Width * 2 + Tokens + (long)Vocabulary * Width) * sizeof(float);
        var timing = MeasureGenerated(
            runtime, major, minor,
            IncumbentSemanticSpecs.Scatter("h2h_scatter", Tokens, Vocabulary, Width),
            indexData: ids,
            floatInputs: new Dictionary<int, float[]> { [1] = grad },
            launchExisting: LaunchExisting,
            synchronizeExisting: backend.Synchronize,
            existingWorkUnits,
            generatedWorkUnits,
            zeroOutput: true,
            out var generatedOnce);

        double error = RelativeError(existingOnce, generatedOnce);

        // NOT THE SAME OPERATOR, and the ratio must say so. CudaBackend.EmbeddingBackward
        // dispatches to embedding_backward_DETERMINISTIC, which scans every one of the
        // numIndices for each (vocab, dim) cell -- O(V*D*N), or 2.7e11 operations at this
        // shape, which is why it takes 303 ms rather than the hundreds of microseconds the
        // memory alone would need.
        //
        // The generated kernel uses atomic accumulation. It is faster by three orders of
        // magnitude and it BUYS THAT WITH TWO GUARANTEES the incumbent provides:
        //
        //   - Bit-reproducibility. fp32 atomics commit in scheduling order, so repeated runs
        //     can differ in the last bits. The deterministic kernel cannot.
        //   - Self-containment. Its comment is explicit that plain assignment means the caller
        //     need not zero the destination. The atomic form ADDS, so a caller who does not
        //     zero gets the previous contents folded into the gradient.
        //
        // So this row is not a promotion verdict. It says the shipped default is
        // algorithmically expensive and that a faster form exists at a stated cost -- which is
        // a decision about determinism, not about kernels.
        Report("embedding backward (vs DETERMINISTIC scan)", timing,
               qualification: error <= 2e-5
                   ? "different guarantees; atomic timing excludes required clear"
                   : "NOT EQUIVALENT -- relative error "
                     + error.ToString("E2", CultureInfo.InvariantCulture));
    }

    /// <summary>
    /// SGD with momentum: the smallest optimizer whose entire state transition the current
    /// spec algebra can express. Weight decay is zero because the spec currently has one bias
    /// input, while momentum with L2 decay needs both gradient and parameter in the velocity
    /// update.
    /// </summary>
    private static void SgdMomentum(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor)
    {
        const int Count = 1 << 22;
        const float LearningRate = 0.01f, Momentum = 0.9f;

        var parameter = Values(Count, 11, 0.25f);
        var gradient = Values(Count, 23, 0.03125f);
        var velocity = Values(Count, 37, 0.015625f);

        using var existingParameter = backend.AllocateBuffer(parameter);
        using var existingGradient = backend.AllocateBuffer(gradient);
        using var existingVelocity = backend.AllocateBuffer(velocity);

        void LaunchExisting() => backend.SgdMomentumUpdate(
            existingParameter, existingGradient, existingVelocity,
            LearningRate, Momentum, weightDecay: 0f, Count);

        // Agreement is checked on exactly one step. Timing runs mutate optimizer state, and
        // the two stability loops can take different sample counts; comparing their final
        // buffers would therefore compare different numbers of optimizer steps.
        LaunchExisting();
        backend.Synchronize();
        var existingParameterOnce = backend.DownloadBuffer(existingParameter);
        var existingVelocityOnce = backend.DownloadBuffer(existingVelocity);

        backend.UploadBufferInPlace(parameter, existingParameter);
        backend.UploadBufferInPlace(velocity, existingVelocity);

        long workUnits = (long)Count * sizeof(float) * 5; // 3 reads + 2 writes
        var timing = MeasureGeneratedMomentum(
            runtime, major, minor, parameter, gradient, velocity,
            LearningRate, Momentum,
            LaunchExisting, backend.Synchronize, workUnits,
            out var generatedParameterOnce, out var generatedVelocityOnce);

        double error = Math.Max(
            RelativeError(existingParameterOnce, generatedParameterOnce),
            RelativeError(existingVelocityOnce, generatedVelocityOnce));

        Report("SGD momentum 4M (weight decay 0)", timing,
            qualification: error <= 2e-6
                ? null
                : "NOT EQUIVALENT -- relative error "
                  + error.ToString("E2", CultureInfo.InvariantCulture));
    }

    /// <summary>
    /// A hand-written axis reduction against the generated planner's complete program. When
    /// the planner chooses split-K, the generated timing includes both partial and combine
    /// launches; timing only the partial would move cost out of the row and manufacture a win.
    /// </summary>
    private static void RowReduction(
        CudaBackend backend, DirectPtxRuntime runtime, int major, int minor,
        string label, string kernelName, CodegenReduceKind reduce,
        Action<CudaBackend, IGpuBuffer, IGpuBuffer, int, int> launchExisting,
        double reduceScale = 1.0)
    {
        const int Rows = 4096, Inner = 1024;
        var input = Values(Rows * Inner, 53, 0.015625f);

        using var existingInput = backend.AllocateBuffer(input);
        using var existingOutput = backend.AllocateBuffer(Rows);

        void Existing() => launchExisting(backend, existingInput, existingOutput, Rows, Inner);

        Existing();
        backend.Synchronize();
        var existingValues = backend.DownloadBuffer(existingOutput);

        long workUnits = (long)input.Length * sizeof(float) + (long)Rows * sizeof(float);
        var spec = IncumbentSemanticSpecs.RowReduction(
            kernelName, Rows, Inner, reduce, reduceScale);
        var timing = MeasureGeneratedReduction(
            runtime, major, minor, spec, input,
            Existing, backend.Synchronize, workUnits,
            out var generatedValues, out bool usedSplit);

        double error = RelativeError(existingValues, generatedValues);
        string measuredLabel = usedSplit ? label + " (planned split)" : label + " (direct)";
        Report(measuredLabel, timing,
            qualification: error <= 2e-5
                ? null
                : "NOT EQUIVALENT -- relative error "
                  + error.ToString("E2", CultureInfo.InvariantCulture));
    }

    /// <summary>Emits and times a generated spec on its own buffers.</summary>
    private static StableTimer.PairResult MeasureGenerated(
        DirectPtxRuntime runtime, int major, int minor, CodegenKernelSpec spec,
        int[] indexData, IReadOnlyDictionary<int, float[]> floatInputs,
        Action launchExisting, Action synchronizeExisting,
        long existingWorkUnits, long generatedWorkUnits,
        bool zeroOutput, out float[] outputOnce)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < spec.Inputs.Count; i++)
            {
                var binding = spec.Inputs[i];
                DirectPtxBuffer buffer;

                if (binding.IsIndexTensor)
                {
                    var indices = new int[binding.ElementCount];
                    Array.Copy(indexData, indices, indices.Length);
                    buffer = runtime.AllocateBytes((nuint)(indices.Length * sizeof(int)));
                    buffer.Upload<int>(indices);
                }
                else
                {
                    if (!floatInputs.TryGetValue(binding.ParameterIndex, out var values)
                        || values.Length != binding.ElementCount)
                    {
                        throw new ArgumentException(
                            $"Missing {binding.ElementCount}-element fp32 input for " +
                            $"parameter {binding.ParameterIndex} ({binding.Name}).",
                            nameof(floatInputs));
                    }

                    buffer = runtime.AllocateBytes(
                        (nuint)(binding.ElementCount * binding.ElementBytes));
                    buffer.Upload<float>(values);
                }

                buffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }

            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            buffers.Add(output);
            pointers[spec.Output.ParameterIndex] = output.Pointer;

            float[]? zeros = zeroOutput ? new float[spec.Output.ElementCount] : null;
            if (zeros is not null) output.Upload<float>(zeros);

            void LaunchGenerated() => Launch(
                module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);

            LaunchGenerated();
            runtime.Synchronize();
            outputOnce = new float[spec.Output.ElementCount];
            output.Download<float>(outputOnce);

            // Scatter uses atomic add, so reset once after the agreement check. Deliberately
            // do not hide a memset inside every timed launch: this row is the atomic kernel's
            // wall clock, not an equivalent self-contained operator, and Report says so.
            if (zeros is not null) output.Upload<float>(zeros);

            // SAME TIMING METHOD, ADJACENT SAMPLES. The first version used CUDA events here
            // and a host Stopwatch for CudaBackend; the second measured all incumbent samples
            // before all generated samples. Both let the protocol or clock drift become part
            // of the ratio. The protocol requires A/B inside each paired sample.
            return StableTimer.MeasureHostPair(
                launchExisting, synchronizeExisting, existingWorkUnits,
                LaunchGenerated, runtime.Synchronize, generatedWorkUnits);
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static void Report(
        string label, StableTimer.PairResult timing,
        string? qualification = null)
    {
        StableTimer.Result existing = timing.A;
        StableTimer.Result generated = timing.B;
        // A RATIO NEEDS BOTH SIDES AND THE PAIRED RATIO STABLE. One converged measurement over
        // one that did not is not a comparison, and neither are two individually stable
        // distributions whose within-sample ratio still moves.
        string ratio, verdict;
        if (!timing.Stable)
        {
            ratio = "-";
            verdict = qualification ?? "NOT MEASURABLE at this size";
        }
        else
        {
            double r = timing.Ratio;
            ratio = timing.DescribeRatio();

            // The noise floor is measured directly from the paired ratios.
            double floor = 1.0 + timing.RelativeSpread;
            verdict = r > floor ? "generated faster"
                    : r < 1.0 / floor ? "existing wins -- withhold"
                    : "TIE within noise -- withhold";

            // A QUALIFIED ROW IS NEVER A PROMOTION. Two kernels that compute the same values
            // under different guarantees are not interchangeable however the ratio reads.
            if (qualification is not null) verdict = qualification;
        }

        Console.WriteLine("{0,-44} {1,15} {2,15} {3,9}  {4}",
            label, existing.Describe(), generated.Describe(), ratio, verdict);
    }

    private static void ReportUnavailable(string label, string reason)
    {
        Console.WriteLine("{0,-44} {1,15} {2,15} {3,9}  {4}",
            label, "existing", "-", "-", "NOT EXPRESSIBLE -- " + reason);
    }

    private static StableTimer.PairResult MeasureGeneratedMomentum(
        DirectPtxRuntime runtime, int major, int minor,
        float[] parameter, float[] gradient, float[] velocity,
        float learningRate, float momentum,
        Action launchExisting, Action synchronizeExisting, long workUnits,
        out float[] parameterOnce, out float[] velocityOnce)
    {
        var spec = IncumbentSemanticSpecs.Momentum(
            "h2h_momentum", parameter.Length, momentum, learningRate);
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, major, minor);
        using var module = runtime.LoadModule(ptx);
        IntPtr fn = module.GetFunction(spec.Name, out _);

        using var v = runtime.AllocateBytes((nuint)(velocity.Length * sizeof(float)));
        using var g = runtime.AllocateBytes((nuint)(gradient.Length * sizeof(float)));
        using var p = runtime.AllocateBytes((nuint)(parameter.Length * sizeof(float)));
        v.Upload<float>(velocity);
        g.Upload<float>(gradient);
        p.Upload<float>(parameter);

        // Inputs and outputs intentionally alias. The incumbent updates p and v in place, and
        // each generated thread loads its own element before storing it, so this is the same
        // memory contract rather than a separate-output approximation.
        var pointers = new[] { v.Pointer, g.Pointer, p.Pointer, v.Pointer, p.Pointer };
        void LaunchMomentum() => Launch(
            module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);

        LaunchMomentum();
        runtime.Synchronize();
        parameterOnce = new float[parameter.Length];
        velocityOnce = new float[velocity.Length];
        p.Download<float>(parameterOnce);
        v.Download<float>(velocityOnce);

        p.Upload<float>(parameter);
        v.Upload<float>(velocity);
        return StableTimer.MeasureHostPair(
            launchExisting, synchronizeExisting, workUnits,
            LaunchMomentum, runtime.Synchronize, workUnits);
    }

    private static StableTimer.PairResult MeasureGeneratedReduction(
        DirectPtxRuntime runtime, int major, int minor, CodegenKernelSpec spec,
        float[] input, Action launchExisting, Action synchronizeExisting, long workUnits,
        out float[] output, out bool usedSplit)
    {
        var plan = CodegenSplitReduction.TryPlan(spec);
        usedSplit = plan is not null;

        if (plan is null)
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(spec, major, minor);
            using var module = runtime.LoadModule(ptx);
            IntPtr fn = module.GetFunction(spec.Name, out _);
            using var inputBuffer = runtime.AllocateBytes((nuint)(input.Length * sizeof(float)));
            using var outputBuffer = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * sizeof(float)));
            inputBuffer.Upload<float>(input);
            var pointers = new[] { inputBuffer.Pointer, outputBuffer.Pointer };
            void LaunchDirect() => Launch(
                module, fn, pointers, (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX);

            LaunchDirect();
            runtime.Synchronize();
            output = new float[spec.Output.ElementCount];
            outputBuffer.Download<float>(output);
            return StableTimer.MeasureHostPair(
                launchExisting, synchronizeExisting, workUnits,
                LaunchDirect, runtime.Synchronize, workUnits);
        }

        var partialEmitter = new PtxAffineEmitter();
        var combineEmitter = new PtxAffineEmitter();
        string partialPtx = partialEmitter.Emit(plan.Partial, major, minor);
        string combinePtx = combineEmitter.Emit(plan.Combine, major, minor);
        using var partialModule = runtime.LoadModule(partialPtx);
        using var combineModule = runtime.LoadModule(combinePtx);
        IntPtr partialFn = partialModule.GetFunction(plan.Partial.Name, out _);
        IntPtr combineFn = combineModule.GetFunction(plan.Combine.Name, out _);

        using var inputGpu = runtime.AllocateBytes((nuint)(input.Length * sizeof(float)));
        using var temporary = runtime.AllocateBytes(
            (nuint)(plan.Partial.Output.ElementCount * sizeof(float)));
        using var outputGpu = runtime.AllocateBytes(
            (nuint)(spec.Output.ElementCount * sizeof(float)));
        inputGpu.Upload<float>(input);

        var partialArgs = new[] { inputGpu.Pointer, temporary.Pointer };
        var combineArgs = new[] { temporary.Pointer, outputGpu.Pointer };
        void LaunchSplit()
        {
            Launch(partialModule, partialFn, partialArgs,
                (uint)partialEmitter.LaunchBlocks, (uint)partialEmitter.LaunchBlockX);
            Launch(combineModule, combineFn, combineArgs,
                (uint)combineEmitter.LaunchBlocks, (uint)combineEmitter.LaunchBlockX);
        }

        LaunchSplit();
        runtime.Synchronize();
        output = new float[spec.Output.ElementCount];
        outputGpu.Download<float>(output);
        return StableTimer.MeasureHostPair(
            launchExisting, synchronizeExisting, workUnits,
            LaunchSplit, runtime.Synchronize, workUnits);
    }

    private static float[] Values(int count, int salt, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (((i * 37L + salt) % 97) - 48) * scale;
        return values;
    }

    private static double RelativeError(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length) return double.PositiveInfinity;

        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs((double)expected[i] - actual[i]));
            scale = Math.Max(scale, Math.Abs(expected[i]));
        }
        return worst / Math.Max(1.0, scale);
    }

    private static unsafe void Launch(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint threads)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, threads, 1, 1, 0, argv);
        }
    }
}
