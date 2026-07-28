// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
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
/// Both sides go through <see cref="StableTimer"/>, so a ratio is only printed when both
/// measurements converged. A promotion decision taken on numbers that move 50% between runs
/// is a coin flip wearing a table.
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
        Console.WriteLine();
        Console.WriteLine("{0,-38} {1,15} {2,15} {3,9}  {4}",
            "operator", "existing", "generated", "ratio", "verdict");

        EmbeddingForward(backend, runtime, major, minor);
        EmbeddingBackward(backend, runtime, major, minor);

        Console.WriteLine();
        Console.WriteLine("ratio > 1 means the generated kernel is faster. A promotion needs BOTH");
        Console.WriteLine("sides stable and a ratio clear of the noise floor -- 1.05x from samples");
        Console.WriteLine("that disagree by 5% is not a win, it is a tie with a rounding error.");
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
        for (int t = 0; t < Tokens; t++) ids[t] = (t * 7919) % Vocabulary;
        var table = new float[(long)Vocabulary * Width];
        for (int e = 0; e < table.Length; e++) table[e] = ((e * 37) % 97 - 48) / 16.0f;

        using var idsBuffer = backend.AllocateIntBuffer(ids);
        using var tableBuffer = backend.AllocateBuffer(table);
        using var outBuffer = backend.AllocateBuffer(Tokens * Width);

        var existing = StableTimer.MeasureHost(
            () => backend.Embedding(idsBuffer, tableBuffer, outBuffer, Tokens, Width),
            backend.Synchronize,
            workUnits: (long)Tokens * Width);

        var generated = MeasureGenerated(
            runtime, major, minor, GatherSpec(Tokens, Vocabulary, Width),
            indexData: ids, workUnits: (long)Tokens * Width);

        Report("embedding forward 1M x 64", existing, generated);
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
        for (int t = 0; t < Tokens; t++) ids[t] = (t * 7919) % Vocabulary;
        var grad = new float[(long)Tokens * Width];
        for (int e = 0; e < grad.Length; e++) grad[e] = ((e * 37) % 97 - 48) / 16.0f;

        using var idsBuffer = backend.AllocateIntBuffer(ids);
        using var gradBuffer = backend.AllocateBuffer(grad);
        using var tableBuffer = backend.AllocateBuffer(Vocabulary * Width);

        var existing = StableTimer.MeasureHost(
            () => backend.EmbeddingBackward(
                gradBuffer, idsBuffer, tableBuffer, Tokens, Width, Vocabulary),
            backend.Synchronize,
            workUnits: (long)Tokens * Width);

        var generated = MeasureGenerated(
            runtime, major, minor, ScatterSpec(Tokens, Vocabulary, Width),
            indexData: ids, workUnits: (long)Tokens * Width);

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
        Report("embedding backward (vs DETERMINISTIC scan)", existing, generated,
               qualification: "different guarantees -- see source");
    }

    /// <summary>Emits and times a generated spec on its own buffers.</summary>
    private static StableTimer.Result MeasureGenerated(
        DirectPtxRuntime runtime, int major, int minor, CodegenKernelSpec spec,
        int[] indexData, long workUnits)
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
                    buffer = runtime.AllocateBytes(
                        (nuint)(binding.ElementCount * binding.ElementBytes));
                }

                buffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }

            var output = runtime.AllocateBytes(
                (nuint)(spec.Output.ElementCount * spec.Output.ElementBytes));
            buffers.Add(output);
            pointers[spec.Output.ParameterIndex] = output.Pointer;

            return StableTimer.Measure(
                runtime,
                () => Launch(module, fn, pointers,
                             (uint)emitter.LaunchBlocks, (uint)emitter.LaunchBlockX),
                workUnits);
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    private static void Report(
        string label, StableTimer.Result existing, StableTimer.Result generated,
        string? qualification = null)
    {
        // A RATIO NEEDS BOTH SIDES STABLE. One converged measurement over one that did not is
        // not a comparison, and printing it as though it were is how a promotion gets decided
        // on noise.
        string ratio, verdict;
        if (!existing.Stable || !generated.Stable)
        {
            ratio = "-";
            verdict = "NOT MEASURABLE at this size";
        }
        else
        {
            double r = existing.Microseconds / generated.Microseconds;
            ratio = r.ToString("0.00", CultureInfo.InvariantCulture) + "x";

            // The noise floor is the two spreads combined: a ratio inside it is a tie.
            double floor = 1.0 + existing.RelativeSpread + generated.RelativeSpread;
            verdict = r > floor ? "generated faster"
                    : r < 1.0 / floor ? "existing wins -- withhold"
                    : "TIE within noise -- withhold";

            // A QUALIFIED ROW IS NEVER A PROMOTION. Two kernels that compute the same values
            // under different guarantees are not interchangeable however the ratio reads.
            if (qualification is not null) verdict = qualification;
        }

        Console.WriteLine("{0,-38} {1,15} {2,15} {3,9}  {4}",
            label, existing.Describe(), generated.Describe(), ratio, verdict);
    }

    private static CodegenKernelSpec GatherSpec(int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);
        var table = new CodegenTensorBinding(1, "table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });
        var output = new CodegenTensorBinding(2, "out", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) }, isOutput: true);

        return new CodegenKernelSpec("h2h_gather", space, new[] { ids, table }, output,
            new[] { 1 }, CodegenReduceKind.None);
    }

    private static CodegenKernelSpec ScatterSpec(int tokens, int vocabulary, int width)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("t", tokens), CodegenAxis.Parallel("e", width));

        var ids = new CodegenTensorBinding(0, "ids", new[] { tokens },
            new[] { CodegenAffineExpr.Axis(0) }, elementType: CodegenElementType.Int32);
        var grad = new CodegenTensorBinding(1, "grad", new[] { tokens, width },
            new[] { CodegenAffineExpr.Axis(0), CodegenAffineExpr.Axis(1) });
        var table = new CodegenTensorBinding(2, "grad_table", new[] { vocabulary, width },
            new[] { CodegenAffineExpr.Const(0), CodegenAffineExpr.Axis(1) },
            isOutput: true,
            indirect: new CodegenIndirectIndex?[]
            {
                new CodegenIndirectIndex(0, CodegenAffineExpr.Axis(0), vocabulary),
                null,
            });

        return new CodegenKernelSpec("h2h_scatter", space, new[] { ids, grad }, table,
            new[] { 1 }, CodegenReduceKind.None);
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
