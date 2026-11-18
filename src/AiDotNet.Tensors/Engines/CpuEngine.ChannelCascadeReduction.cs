using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    [ThreadStatic] private static int t_channelCascadeReductionRuns;
    internal static int ChannelCascadeReductionRunsOnThisThread => t_channelCascadeReductionRuns;

    private static bool TryChannelCascadeSum<T>(Tensor<T> tensor, int[] axes, bool keepDims,
        [NotNullWhen(true)] out Tensor<T>? result)
    {
        result = null;
#if NET5_0_OR_GREATER
        if (typeof(T) != typeof(float) || tensor.Rank < 3 || !tensor.IsContiguous || axes.Length != tensor.Rank - 1)
            return false;
        for (int axis = 0; axis < tensor.Rank; axis++)
            if (axis != 1 && Array.IndexOf(axes, axis) < 0) return false;
        if (tensor.GetCpuBackingForStridedRead(out int sourceOffset) is not { } input) return false;
        int channels = tensor.Shape[1], batch = tensor.Shape[0];
        var shape = keepDims ? new int[tensor.Rank] : new[] { channels };
        if (keepDims) { for (int axis = 0; axis < shape.Length; axis++) shape[axis] = axis == 1 ? channels : 1; }
        using var keepInput = new KeepAliveScope(tensor);
        var reduced = AutoTensorCache.RentOrAllocate<T>(shape);
        using var keepOutput = new KeepAliveScope(reduced);
        if (reduced.GetCpuBackingForContiguousWrite(out int destinationOffset) is not { } output)
        {
            reduced.Dispose();
            return false;
        }
        var source = (float[])(object)input;
        var destination = (float[])(object)output;
        int spatial = batch == 0 || channels == 0 ? 0 : tensor.Length / batch / channels;
        CpuParallelSettings.ParallelForOrSerial(0, channels, tensor.Length, channel =>
        {
            float sum = 0f;
            if (channels == 1)
                sum = CascadeChannelSum(source.AsSpan(sourceOffset, tensor.Length));
            else
                for (int image = 0; image < batch; image++)
                    sum += CascadeChannelSum(source.AsSpan(sourceOffset + (image * channels + channel) * spatial, spatial));
            destination[destinationOffset + channel] = sum;
        }, deterministicSafe: true);
        t_channelCascadeReductionRuns++;
        result = reduced;
        return true;
#else
        return false;
#endif
    }

#if NET5_0_OR_GREATER
    /// <summary>Four-level SIMD cascade with four independent streams, followed by a fixed lane-order sum.</summary>
    /// <remarks>
    /// The cascade grouping follows PyTorch SumKernel.cpp at 5c4886908584029761b579af026dcfb627c84070.
    /// The following BSD notice is retained in source and in the generated library XML documentation.
    /// From PyTorch:
    /// 
    /// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
    /// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
    /// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
    /// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
    /// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
    /// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
    /// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
    /// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
    /// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
    /// 
    /// From Caffe2:
    /// 
    /// Copyright (c) 2016-present, Facebook Inc. All rights reserved.
    /// 
    /// All contributions by Facebook:
    /// Copyright (c) 2016 Facebook Inc.
    /// 
    /// All contributions by Google:
    /// Copyright (c) 2015 Google Inc.
    /// All rights reserved.
    /// 
    /// All contributions by Yangqing Jia:
    /// Copyright (c) 2015 Yangqing Jia
    /// All rights reserved.
    /// 
    /// All contributions by Kakao Brain:
    /// Copyright 2019-2020 Kakao Brain
    /// 
    /// All contributions by Cruise LLC:
    /// Copyright (c) 2022 Cruise LLC.
    /// All rights reserved.
    /// 
    /// All contributions by Tri Dao:
    /// Copyright (c) 2024 Tri Dao.
    /// All rights reserved.
    /// 
    /// All contributions by Arm:
    /// Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
    /// 
    /// All contributions from Caffe:
    /// Copyright(c) 2013, 2014, 2015, the respective contributors
    /// All rights reserved.
    /// 
    /// All other contributions:
    /// Copyright(c) 2015, 2016 the respective contributors
    /// All rights reserved.
    /// 
    /// Caffe2 uses a copyright model similar to Caffe: each contributor holds
    /// copyright over their contributions to Caffe2. The project versioning records
    /// all such contribution and copyright details. If a contributor wants to further
    /// mark their specific copyright on a particular contribution, they should
    /// indicate their copyright solely in the commit message of the change when it is
    /// committed.
    /// 
    /// All rights reserved.
    /// 
    /// Redistribution and use in source and binary forms, with or without
    /// modification, are permitted provided that the following conditions are met:
    /// 
    /// 1. Redistributions of source code must retain the above copyright
    ///    notice, this list of conditions and the following disclaimer.
    /// 
    /// 2. Redistributions in binary form must reproduce the above copyright
    ///    notice, this list of conditions and the following disclaimer in the
    ///    documentation and/or other materials provided with the distribution.
    /// 
    /// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
    ///    and IDIAP Research Institute nor the names of its contributors may be
    ///    used to endorse or promote products derived from this software without
    ///    specific prior written permission.
    /// 
    /// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    /// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    /// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    /// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    /// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    /// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    /// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    /// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    /// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    /// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    /// POSSIBILITY OF SUCH DAMAGE.
    /// </remarks>
    private static float CascadeChannelSum(ReadOnlySpan<float> values)
    {
        int width = System.Numerics.Vector<float>.Count, vectors = values.Length / width, groups = vectors / 4;
        int log = 0;
        for (int remaining = Math.Max(0, groups - 1); remaining != 0; remaining >>= 1) log++;
        int power = Math.Max(4, log / 4), step = 1 << power, mask = step - 1;
        Span<System.Numerics.Vector<float>> accumulators = stackalloc System.Numerics.Vector<float>[16];
        accumulators.Clear();
        int index = 0;
        while (index + step <= groups)
        {
            int stop = index + step;
            for (; index < stop; index++)
                for (int stream = 0; stream < 4; stream++)
                    accumulators[stream] += new System.Numerics.Vector<float>(values.Slice((index * 4 + stream) * width, width));
            for (int level = 1; level < 4; level++)
            {
                for (int stream = 0; stream < 4; stream++)
                {
                    accumulators[level * 4 + stream] += accumulators[(level - 1) * 4 + stream];
                    accumulators[(level - 1) * 4 + stream] = System.Numerics.Vector<float>.Zero;
                }
                if ((index & ((long)mask << (level * power))) != 0) break;
            }
        }
        for (; index < groups; index++)
            for (int stream = 0; stream < 4; stream++)
                accumulators[stream] += new System.Numerics.Vector<float>(values.Slice((index * 4 + stream) * width, width));
        for (int level = 1; level < 4; level++)
            for (int stream = 0; stream < 4; stream++) accumulators[stream] += accumulators[level * 4 + stream];
        for (int vector = groups * 4; vector < vectors; vector++)
            accumulators[0] += new System.Numerics.Vector<float>(values.Slice(vector * width, width));
        for (int stream = 1; stream < 4; stream++) accumulators[0] += accumulators[stream];
        float sum = 0f;
        for (int tail = vectors * width; tail < values.Length; tail++) sum += values[tail];
        for (int lane = 0; lane < width; lane++) sum += accumulators[0][lane];
        return sum;
    }
#endif
}

