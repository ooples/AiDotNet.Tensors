// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>Emits a shared-memory tiled, true-FP32 SIMT contraction.</summary>
/// <remarks>
/// This is separate from <see cref="PtxAffineEmitter"/> because a CTA cooperatively owns an
/// output tile here, while the affine path gives every thread an independent scalar/lane
/// tile. The schedule stages both operands, overlaps the next asynchronous copy with the
/// current FP32 FMA tile, and never changes multiplication semantics to TF32 or tensor cores.
/// </remarks>
public sealed class PtxTiledContractionEmitter
{
    private const int FixedR = 16;
    private const int FixedRd = 8;
    private const int FixedP = 4;
    private readonly StringBuilder _body = new();
    private readonly CodegenTiledContractionSchedule? _schedule;
    private int _r, _rd, _p, _f;

    public PtxTiledContractionEmitter() { }

    public PtxTiledContractionEmitter(CodegenTiledContractionSchedule schedule)
    {
        _schedule = schedule ?? throw new ArgumentNullException(nameof(schedule));
    }

    /// <summary>Plan used by the last emission.</summary>
    public CodegenTiledContractionPlan? Plan { get; private set; }

    /// <summary>Blocks the host must launch.</summary>
    public uint LaunchBlocks => checked((uint)(Plan?.Blocks ?? 0));

    /// <summary>Threads per block.</summary>
    public int LaunchBlockThreads => Plan?.BlockThreads ?? 0;

    /// <summary>Static shared-memory bytes declared by the emitted kernel.</summary>
    public int SharedMemoryBytes => Plan?.SharedMemoryBytes ?? 0;

    /// <summary>Plans and emits one exact FP32 contraction.</summary>
    public string Emit(CodegenKernelSpec spec, int computeMajor, int computeMinor)
    {
        if (computeMajor < 8)
            throw new NotSupportedException("The double-buffered tiled path requires cp.async on sm_80+.");
        if (!CodegenTiledContractionPlan.TryCreate(
                spec, _schedule, out var possible, out string reason))
            throw new NotSupportedException("This spec cannot use the tiled contraction: " + reason);
        var plan = possible!;
        AssertAsyncCopyInvariants(plan);

        Plan = plan;
        _body.Clear();
        _r = FixedR;
        _rd = FixedRd;
        _p = FixedP;
        _f = 0;

        int matrixParam = spec.Inputs[plan.MatrixInput].ParameterIndex;
        int streamParam = spec.Inputs[plan.StreamInput].ParameterIndex;
        int outputParam = spec.Output.ParameterIndex;
        int tilesM = plan.M / plan.TileM;
        int tilesN = plan.N / plan.TileN;

        L($"ld.param.u64 %rd0, [p{I(matrixParam)}];");
        L($"ld.param.u64 %rd1, [p{I(streamParam)}];");
        L($"ld.param.u64 %rd2, [p{I(outputParam)}];");
        L("mov.u64 %rd3, stage;");
        if (plan.BiasInput.HasValue)
        {
            int biasParam = spec.Inputs[plan.BiasInput.Value].ParameterIndex;
            L($"ld.param.u64 %rd4, [p{I(biasParam)}];");
        }
        if (plan.ScaleInput.HasValue)
        {
            int scaleParam = spec.Inputs[plan.ScaleInput.Value].ParameterIndex;
            L($"ld.param.u64 %rd5, [p{I(scaleParam)}];");
        }
        L("mov.u32 %r0, %ctaid.x;");
        L("mov.u32 %r1, %tid.x;");
        L($"rem.u32 %r2, %r0, {I(tilesN)};             // N tile");
        L($"div.u32 %r3, %r0, {I(tilesN)};");
        L($"rem.u32 %r4, %r3, {I(tilesM)};             // M tile");
        L($"div.u32 %r5, %r3, {I(tilesM)};             // batch");
        L($"rem.u32 %r6, %r1, {I(plan.ThreadsN)};       // thread N");
        L($"div.u32 %r7, %r1, {I(plan.ThreadsN)};       // thread M");

        // Block origins, in elements. Keeping these live turns every later address into a
        // compile-time K-step plus a small shared offset.
        L($"mul.lo.u32 %r8, %r4, {I(plan.TileM)};");
        L($"mul.lo.u32 %r9, %r2, {I(plan.TileN)};");
        L($"mul.lo.u32 %r10, %r5, {I(plan.K)};");
        L($"mul.lo.u32 %r10, %r10, {I(plan.N)};");
        L("add.u32 %r10, %r10, %r9;                   // stream base");
        L($"mul.lo.u32 %r11, %r5, {I(plan.M)};");
        L("add.u32 %r11, %r11, %r8;");
        L($"mul.lo.u32 %r11, %r11, {I(plan.N)};");
        L("add.u32 %r11, %r11, %r9;                   // output base");

        var accumulators = new string[plan.ThreadTileM, plan.ThreadTileN];
        for (int i = 0; i < plan.ThreadTileM; i++)
            for (int j = 0; j < plan.ThreadTileN; j++)
            {
                accumulators[i, j] = NextF();
                L($"mov.f32 {accumulators[i, j]}, 0f00000000;");
            }

        // Prologue. Every slab is a whole tile, so each 16-byte async copy is valid and no
        // zero-fill convention is needed.
        EmitAsyncStage(plan, step: 0, buffer: 0);
        L("cp.async.commit_group;");
        L("cp.async.wait_group 0;");
        L("bar.sync 0;");

        for (int step = 0; step < plan.Steps; step++)
        {
            int current = step & 1;
            if (step + 1 < plan.Steps)
            {
                EmitAsyncStage(plan, step + 1, 1 - current);
                L("cp.async.commit_group;");
            }

            EmitCompute(plan, current, accumulators);

            if (step + 1 < plan.Steps)
            {
                // The barrier both publishes the next slab and proves every warp has
                // stopped reading the buffer that will be reused two steps later.
                L("cp.async.wait_group 0;");
                L("bar.sync 0;");
            }
        }

        EmitStores(plan, spec.Activation, accumulators);
        L("ret;");

        var text = new StringBuilder();
        text.Append(".version ")
            .Append(PtxAffineEmitter.PtxIsaVersionFor(computeMajor, computeMinor)).Append('\n')
            .Append(".target sm_").Append(I(computeMajor)).Append(I(computeMinor)).Append('\n')
            .Append(".address_size 64\n\n")
            .Append("// generated by PtxTiledContractionEmitter -- true-fp32 SIMT\n")
            .Append("// tile ").Append(I(plan.TileM)).Append('x').Append(I(plan.TileN))
            .Append('x').Append(I(plan.TileK)).Append(", thread tile ")
            .Append(I(plan.ThreadTileM)).Append('x').Append(I(plan.ThreadTileN))
            .Append(", double-buffered\n")
            .Append("// ").Append(spec.Describe().Replace("\n", "\n// ")).Append('\n')
            .Append(".visible .entry ").Append(spec.Name).Append("(\n");
        for (int i = 0; i < spec.ParameterCount; i++)
            text.Append("    .param .u64 p").Append(I(i))
                .Append(i == spec.ParameterCount - 1 ? "\n" : ",\n");
        text.Append(")\n{\n")
            .Append("    .shared .align 16 .b8 stage[").Append(I(plan.SharedMemoryBytes))
            .Append("];\n")
            .Append("    .reg .pred %p<").Append(I(_p + 2)).Append(">;\n")
            .Append("    .reg .b32 %r<").Append(I(_r + 2)).Append(">;\n")
            .Append("    .reg .b64 %rd<").Append(I(_rd + 2)).Append(">;\n")
            .Append("    .reg .f32 %f<").Append(I(_f + 2)).Append(">;\n\n")
            .Append(_body)
            .Append("}\n");
        return text.ToString();
    }

    private void EmitAsyncStage(CodegenTiledContractionPlan plan, int step, int buffer)
    {
        const int ValuesPerCopy = 4;
        int matrixChunks = plan.TileK * plan.TileM / ValuesPerCopy;
        int streamChunks = plan.TileK * plan.TileN / ValuesPerCopy;
        int matrixChunksPerRow = (plan.MatrixReductionMajor ? plan.TileM : plan.TileK) /
            ValuesPerCopy;
        int streamChunksPerRow = plan.TileN / ValuesPerCopy;
        int matrixPasses = (matrixChunks + plan.BlockThreads - 1) / plan.BlockThreads;
        int streamPasses = (streamChunks + plan.BlockThreads - 1) / plan.BlockThreads;
        int bufferBase = buffer * plan.StageBytes;
        int streamSharedBase = bufferBase + plan.TileK * plan.TileM * sizeof(float);

        for (int pass = 0; pass < matrixPasses; pass++)
        {
            string index = NextR(), row = NextR(), column = NextR(), element = NextR();
            string bytes = NextRd(), source = NextRd(), sharedBytes = NextRd(), destination = NextRd();
            string valid = NextP();
            L($"add.u32 {index}, %r1, {I(pass * plan.BlockThreads)};");
            L($"setp.lt.u32 {valid}, {index}, {I(matrixChunks)};");
            L($"div.u32 {row}, {index}, {I(matrixChunksPerRow)};");
            L($"rem.u32 {column}, {index}, {I(matrixChunksPerRow)};");
            if (plan.MatrixReductionMajor)
            {
                L($"add.u32 {row}, {row}, {I(step * plan.TileK)};");
                L($"mad.lo.u32 {element}, {row}, {I(plan.M)}, %r8;");
                L($"mad.lo.u32 {element}, {column}, {I(ValuesPerCopy)}, {element};");
            }
            else
            {
                L($"add.u32 {row}, {row}, %r8;");
                L($"mad.lo.u32 {element}, {row}, {I(plan.K)}, {I(step * plan.TileK)};");
                L($"mad.lo.u32 {element}, {column}, {I(ValuesPerCopy)}, {element};");
            }
            L($"mul.wide.u32 {bytes}, {element}, 4;");
            L($"add.u64 {source}, %rd0, {bytes};");
            L($"mul.wide.u32 {sharedBytes}, {index}, 16;");
            L($"add.u64 {destination}, %rd3, {sharedBytes};");
            L($"@{valid} cp.async.ca.shared.global [{destination}+{I(bufferBase)}], [{source}], 16;");
        }

        for (int pass = 0; pass < streamPasses; pass++)
        {
            string index = NextR(), row = NextR(), column = NextR(), element = NextR();
            string bytes = NextRd(), source = NextRd(), sharedBytes = NextRd(), destination = NextRd();
            string valid = NextP();
            L($"add.u32 {index}, %r1, {I(pass * plan.BlockThreads)};");
            L($"setp.lt.u32 {valid}, {index}, {I(streamChunks)};");
            L($"div.u32 {row}, {index}, {I(streamChunksPerRow)};");
            L($"rem.u32 {column}, {index}, {I(streamChunksPerRow)};");
            L($"add.u32 {row}, {row}, {I(step * plan.TileK)};");
            L($"mad.lo.u32 {element}, {row}, {I(plan.N)}, %r10;");
            L($"mad.lo.u32 {element}, {column}, {I(ValuesPerCopy)}, {element};");
            L($"mul.wide.u32 {bytes}, {element}, 4;");
            L($"add.u64 {source}, %rd1, {bytes};");
            L($"mul.wide.u32 {sharedBytes}, {index}, 16;");
            L($"add.u64 {destination}, %rd3, {sharedBytes};");
            L($"@{valid} cp.async.ca.shared.global [{destination}+{I(streamSharedBase)}], [{source}], 16;");
        }
    }

    private void EmitCompute(
        CodegenTiledContractionPlan plan, int buffer, string[,] accumulators)
    {
        int bufferBase = buffer * plan.StageBytes;
        int streamBase = bufferBase + plan.TileK * plan.TileM * sizeof(float);
        for (int k = 0; k < plan.TileK; k++)
        {
            var matrix = new string[plan.ThreadTileM];
            var stream = new string[plan.ThreadTileN];
            for (int i = 0; i < matrix.Length; i++) matrix[i] = NextF();
            for (int j = 0; j < stream.Length; j++) stream[j] = NextF();

            if (plan.MatrixReductionMajor)
            {
                // A [K,M] stage gives each thread's M fragment contiguous, naturally
                // aligned values. Load the fragment as vectors: this removes up to seven
                // shared-memory instructions and their address chains from every K step.
                int vectorWidth = plan.ThreadTileM % 4 == 0
                    ? 4
                    : plan.ThreadTileM % 2 == 0 ? 2 : 1;
                for (int i = 0; i < matrix.Length; i += vectorWidth)
                {
                    int constant = bufferBase + k * plan.TileM * sizeof(float) +
                        i * sizeof(float);
                    string local = NextR();
                    string local64 = NextRd(), address = NextRd();
                    L($"mad.lo.u32 {local}, %r7, {I(plan.ThreadTileM * sizeof(float))}, {I(constant)};");
                    L($"cvt.u64.u32 {local64}, {local};");
                    L($"add.u64 {address}, %rd3, {local64};");
                    EmitSharedLoad(matrix, i, vectorWidth, address);
                }
            }
            else
            {
                // [M,K] values for fixed K are strided, so they remain scalar.
                for (int i = 0; i < matrix.Length; i++)
                {
                    int constant = bufferBase + i * plan.TileK * sizeof(float) +
                        k * sizeof(float);
                    string local = NextR();
                    string local64 = NextRd(), address = NextRd();
                    L($"mad.lo.u32 {local}, %r7, {I(plan.ThreadTileM * plan.TileK * sizeof(float))}, {I(constant)};");
                    L($"cvt.u64.u32 {local64}, {local};");
                    L($"add.u64 {address}, %rd3, {local64};");
                    L($"ld.shared.f32 {matrix[i]}, [{address}];");
                }
            }

            // N is contiguous for both operand layouts. Every retained schedule owns
            // two or four values, so one vector instruction replaces the scalar loads.
            int streamVectorWidth = plan.ThreadTileN % 4 == 0
                ? 4
                : plan.ThreadTileN % 2 == 0 ? 2 : 1;
            for (int j = 0; j < stream.Length; j += streamVectorWidth)
            {
                int constant = streamBase + k * plan.TileN * sizeof(float) +
                    j * sizeof(float);
                string local = NextR();
                string local64 = NextRd(), address = NextRd();
                L($"mad.lo.u32 {local}, %r6, {I(plan.ThreadTileN * sizeof(float))}, {I(constant)};");
                L($"cvt.u64.u32 {local64}, {local};");
                L($"add.u64 {address}, %rd3, {local64};");
                EmitSharedLoad(stream, j, streamVectorWidth, address);
            }
            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < stream.Length; j++)
                    L($"fma.rn.f32 {accumulators[i, j]}, {matrix[i]}, {stream[j]}, {accumulators[i, j]};");
        }
    }

    private void EmitSharedLoad(string[] values, int start, int width, string address)
    {
        if (width == 1)
        {
            L($"ld.shared.f32 {values[start]}, [{address}];");
            return;
        }

        L($"ld.shared.v{I(width)}.f32 {{" +
          string.Join(", ", values, start, width) + $"}}, [{address}];");
    }

    private void EmitStores(
        CodegenTiledContractionPlan plan, CodegenActivationKind activation,
        string[,] accumulators)
    {
        for (int i = 0; i < plan.ThreadTileM; i++)
        {
            string localM = NextR();
            L($"mad.lo.u32 {localM}, %r7, {I(plan.ThreadTileM)}, {I(i)};");
            string? rowBytes = null;
            if (plan.BiasInput.HasValue || plan.ScaleInput.HasValue)
            {
                string globalM = NextR();
                rowBytes = NextRd();
                L($"add.u32 {globalM}, %r8, {localM};");
                L($"mul.wide.u32 {rowBytes}, {globalM}, 4;");
            }
            string? bias = null;
            if (plan.BiasInput.HasValue)
            {
                string biasAddress = NextRd();
                bias = NextF();
                L($"add.u64 {biasAddress}, %rd4, {rowBytes!};");
                L($"ld.global.f32 {bias}, [{biasAddress}];");
            }
            string? scale = null;
            if (plan.ScaleInput.HasValue)
            {
                string scaleAddress = NextRd();
                scale = NextF();
                L($"add.u64 {scaleAddress}, %rd5, {rowBytes!};");
                L($"ld.global.f32 {scale}, [{scaleAddress}];");
            }
            for (int j = 0; j < plan.ThreadTileN; j++)
            {
                string n = NextR(), element = NextR();
                string bytes = NextRd(), address = NextRd();
                if (bias is not null)
                    L($"add.rn.f32 {accumulators[i, j]}, {accumulators[i, j]}, {bias};");
                if (scale is not null)
                    L($"mul.rn.f32 {accumulators[i, j]}, {accumulators[i, j]}, {scale};");
                if (activation == CodegenActivationKind.ReLU)
                    L($"max.f32 {accumulators[i, j]}, {accumulators[i, j]}, 0f00000000;");
                L($"mad.lo.u32 {n}, %r6, {I(plan.ThreadTileN)}, {I(j)};");
                L($"mad.lo.u32 {element}, {localM}, {I(plan.N)}, %r11;");
                L($"add.u32 {element}, {element}, {n};");
                L($"mul.wide.u32 {bytes}, {element}, 4;");
                L($"add.u64 {address}, %rd2, {bytes};");
                L($"st.global.f32 [{address}], {accumulators[i, j]};");
            }
        }
    }

    private string NextR() => "%r" + I(_r++);
    private string NextRd() => "%rd" + I(_rd++);
    private string NextP() => "%p" + I(_p++);
    private string NextF() => "%f" + I(_f++);
    private void L(string line) => _body.Append("    ").Append(line).Append('\n');
    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static void AssertAsyncCopyInvariants(CodegenTiledContractionPlan plan)
    {
        // TryCreate selects divisors, rather than merely bounded tile sizes. These checks
        // make that contract explicit at the boundary that truncates the tile counts and
        // emits 16-byte copies. A failure is an internal planner bug, never a partial tile.
        bool matrixCopyAligned = plan.MatrixReductionMajor
            ? plan.TileM % 4 == 0
            : plan.TileK % 4 == 0;
        if (plan.M % plan.TileM != 0 || plan.N % plan.TileN != 0 ||
            plan.K % plan.TileK != 0 || plan.TileN % 4 != 0 ||
            !matrixCopyAligned || plan.StageBytes % 16 != 0)
        {
            throw new InvalidOperationException(
                "The tiled contraction plan must contain whole, 16-byte-aligned slabs.");
        }
    }
}
