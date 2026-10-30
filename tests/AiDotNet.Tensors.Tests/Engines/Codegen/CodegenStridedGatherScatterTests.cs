// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Strided gather and scatter as declared programs (#890 phase 4e, #1441): the interpreter against the hand-written
// kernels' contract, and the emitted PTX against what the contract allows it to touch.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenStridedGatherScatterTests
{
    [Theory]
    [InlineData(5, 20, 0, 3)]
    [InlineData(7, 50, 4, 7)]
    [InlineData(1, 9, 8, 5)]      // one element at the last valid offset
    [InlineData(6, 6, 0, 1)]      // stride 1: a plain copy
    public void StridedGather_ReadsEveryStridedElement(int count, int sourceLength, int offset, int stride)
    {
        var source = new double[sourceLength];
        for (int i = 0; i < source.Length; i++) source[i] = (i * 13 % 11) - 5 + 0.25 * i;

        var outputs = CodegenBackendPrograms.StridedGather(count, sourceLength, offset, stride)
            .Interpret(new Dictionary<string, double[]> { ["src"] = source });

        Assert.Equal(count, outputs["dst"].Length);
        for (int i = 0; i < count; i++)
            Assert.Equal(source[offset + i * stride], outputs["dst"][i]);
    }

    [Theory]
    [InlineData(5, 20, 0, 3)]
    [InlineData(7, 50, 4, 7)]
    [InlineData(6, 6, 0, 1)]
    public void StridedScatter_WritesEveryStridedCellAndNothingElse(int count, int destinationLength, int offset, int stride)
    {
        var source = new double[count];
        for (int i = 0; i < count; i++) source[i] = 10 + i;

        var outputs = CodegenBackendPrograms.StridedScatter(count, destinationLength, offset, stride)
            .Interpret(new Dictionary<string, double[]> { ["src"] = source });

        var dst = outputs["dst"];
        Assert.Equal(destinationLength, dst.Length);
        var written = new bool[destinationLength];
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(source[i], dst[offset + i * stride]);
            written[offset + i * stride] = true;
        }
        // Cells between the strided ones are not written: the interpreter's output starts at zero, so they read 0.
        for (int j = 0; j < destinationLength; j++)
            if (!written[j]) Assert.Equal(0.0, dst[j]);
    }

    [Fact]
    public void StridedPrograms_EmitOneLoadOneStoreAndNoAtomic()
    {
        foreach (var program in new[]
                 {
                     CodegenBackendPrograms.StridedGather(64, 640, 3, 10),
                     CodegenBackendPrograms.StridedScatter(64, 640, 3, 10),
                 })
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(program.Passes[0], 7, 5);
            Assert.Equal(0, emitter.AtomicStores);
            Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
            Assert.Equal(0, emitter.IndirectIndexLoads);   // affine: no index tensor
            Assert.Equal(1, Count(ptx, "st.global"));
        }
    }

    [Theory]
    [InlineData(0, 10, 0, 1)]     // empty
    [InlineData(4, 10, -1, 1)]    // negative offset
    [InlineData(4, 10, 0, 0)]     // zero stride
    [InlineData(4, 10, 2, 3)]     // last element 11 lies past a 10-element tensor
    public void StridedPrograms_RefuseShapesThatReadOrWritePastTheTensor(int count, int length, int offset, int stride)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CodegenBackendPrograms.StridedGather(count, length, offset, stride));
        Assert.Throws<ArgumentOutOfRangeException>(() => CodegenBackendPrograms.StridedScatter(count, length, offset, stride));
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
