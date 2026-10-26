// Copyright (c) AiDotNet. All rights reserved.
// The grouped-query attention backward (#1440) on the fp64 interpreter: equal to the per-head backward of the same
// problem with every key/value head repeated for its query heads, dk and dv summed back over each group.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenGroupedAttentionBackwardTests
{
    [Theory]
    [InlineData(2, 4, 2, 3, 5, 4)]
    [InlineData(1, 6, 3, 4, 4, 2)]
    [InlineData(1, 3, 1, 2, 3, 3)]
    public void GroupedBackward_IsThePerHeadBackwardOfTheRepeatedHeadsSummedOverEachGroup(int batch, int heads,
        int kvHeads, int seqQ, int seqK, int headDim)
    {
        int g = heads / kvHeads;
        var random = new Random(1440 + heads);
        double[] R(int n) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = random.NextDouble() * 2 - 1; return a; }
        var q = R(batch * heads * seqQ * headDim);
        var k = R(batch * kvHeads * seqK * headDim);
        var v = R(batch * kvHeads * seqK * headDim);
        var dout = R(batch * heads * seqQ * headDim);
        var weights = R(batch * heads * seqQ * seqK);

        // The same problem with key/value head h / g repeated for every query head h.
        double[] Repeat(double[] kv)
        {
            var full = new double[batch * heads * seqK * headDim];
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < heads; h++)
                    Array.Copy(kv, ((b * kvHeads + h / g) * seqK) * headDim, full, ((b * heads + h) * seqK) * headDim, seqK * headDim);
            return full;
        }
        const double scale = 0.37;
        var mha = CodegenBackendPrograms.ScaledDotProductAttentionBackward(batch, heads, seqQ, seqK, headDim, scale)
            .Interpret(new Dictionary<string, double[]>
            {
                ["weights"] = weights, ["dout"] = dout, ["v"] = Repeat(v), ["k"] = Repeat(k), ["q"] = q,
            });
        var gqa = CodegenBackendPrograms.ScaledDotProductAttentionBackward(batch, heads, kvHeads, seqQ, seqK, headDim, scale)
            .Interpret(new Dictionary<string, double[]>
            {
                ["weights"] = weights, ["dout"] = dout, ["v"] = v, ["k"] = k, ["q"] = q,
            });

        for (int i = 0; i < q.Length; i++) Assert.Equal(mha["dq"][i], gqa["dq"][i], 12);
        foreach (string name in new[] { "dk", "dv" })
        {
            var summed = new double[k.Length];
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < heads; h++)
                    for (int e = 0; e < seqK * headDim; e++)
                        summed[((b * kvHeads + h / g) * seqK) * headDim + e] += mha[name][((b * heads + h) * seqK) * headDim + e];
            for (int i = 0; i < summed.Length; i++) Assert.Equal(summed[i], gqa[name][i], 12);
        }
    }
}