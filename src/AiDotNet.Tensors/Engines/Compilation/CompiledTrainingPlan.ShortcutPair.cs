// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    /// <summary>
    /// Finds ResNet's downsampling pairs: two Conv2D steps reading the same input <c>h</c>, a 3 x 3 stride-2
    /// padding-1 convolution and a 1 x 1 stride-2 padding-0 projection shortcut, whose input gradient the GPU engine
    /// computes in one call (<see cref="IConvShortcutPairEngine"/>). Each pair is
    /// keyed by its earlier step's index, where the reverse backward reaches both output gradients complete; the later
    /// step's index goes to <paramref name="consumed"/>. Steps in <paramref name="excluded"/> (another fusion's) are
    /// left alone.
    /// </summary>
    private static void DetectShortcutConvPairs(
        List<CompiledStep<T>> steps,
        HashSet<Tensor<T>> requiresGrad,
        HashSet<int> excluded,
        IConvShortcutPairEngine gpu,
        Dictionary<int, (int Main, int Shortcut)> pairs,
        HashSet<int> consumed)
    {
        if (typeof(T) != typeof(float)) return;
        var conv2DBackward = ((BackwardFunction<T>)BackwardFunctions<T>.Conv2DBackward).Method;
        // Kind 3: a 3x3/s2/p1/d1 step; kind 1: a 1x1/s2/p0/d1 step; 0 otherwise.
        int Kind(CompiledStep<T> s)
        {
            if (s.OpType != OpType.Conv2D || s.Inputs.Length != 2 || s.BackwardFn?.Method != conv2DBackward
                || s.SavedState is not { Length: >= 3 } st || st[0] is not int[] stride || st[1] is not int[] pad
                || st[2] is not int[] dil || stride.Length != 2 || pad.Length != 2 || dil.Length != 2
                || stride[0] != 2 || stride[1] != 2 || dil[0] != 1 || dil[1] != 1 || s.Inputs[1].Rank != 4)
                return 0;
            int kh = s.Inputs[1]._shape[2], kw = s.Inputs[1]._shape[3];
            if (kh == 3 && kw == 3 && pad[0] == 1 && pad[1] == 1) return 3;
            if (kh == 1 && kw == 1 && pad[0] == 0 && pad[1] == 0) return 1;
            return 0;
        }

        var firstByInput = new Dictionary<Tensor<T>, int>();
        for (int j = 0; j < steps.Count; j++)
        {
            if (excluded.Contains(j)) continue;
            int kind = Kind(steps[j]);
            if (kind == 0) continue;
            var h = steps[j].Inputs[0];
            if (!firstByInput.TryGetValue(h, out int i))
            {
                firstByInput[h] = j;
                continue;
            }
            if (i < 0 || Kind(steps[i]) + kind != 4) continue;   // one of each kind
            int main = kind == 3 ? j : i, shortcut = kind == 3 ? i : j;
            var y = steps[main].OutputBuffer;
            if (!requiresGrad.Contains(h) || y.Rank != 4 || steps[shortcut].OutputBuffer.Rank != 4
                || !gpu.SupportsConv2DBackwardInputShortcutPair(h._shape, steps[main].Inputs[1]._shape,
                    steps[shortcut].Inputs[1]._shape, y._shape[2], y._shape[3]))
                continue;
            pairs[i] = (main, shortcut);
            consumed.Add(j);
            firstByInput[h] = -1;   // one pair per input
        }
    }

    /// <summary>
    /// The backward action of one shortcut pair, run at the earlier step's position: both convolutions' kernel
    /// gradients through their own backward functions with <c>h</c> taken out of the gradient relevance, and
    /// <c>h</c>'s gradient from the pair launch, accumulated once. When the pair declines (or an output gradient is
    /// missing) the two steps' own backwards run, later step first, exactly as the unfused plan does.
    /// </summary>
    private static Action<IEngine> BuildShortcutPairBackward(
        CompiledStep<T> main,
        CompiledStep<T> shortcut,
        bool shortcutIsLater,
        Dictionary<Tensor<T>, Tensor<T>> gradMap,
        HashSet<Tensor<T>>? relevance,
        HashSet<Tensor<T>> requiresGrad)
    {
        var h = main.Inputs[0];
        var withoutH = new HashSet<Tensor<T>>(relevance ?? requiresGrad);
        withoutH.Remove(h);
        var later = shortcutIsLater ? shortcut : main;
        var earlier = shortcutIsLater ? main : shortcut;

        void Run(CompiledStep<T> step, Tensor<T> gradOut, HashSet<Tensor<T>>? scope, IEngine eng)
        {
            if (scope is null)
            {
                step.BackwardFn!(gradOut, step.Inputs, step.OutputBuffer, step.SavedState ?? Array.Empty<object>(), eng, gradMap);
                return;
            }
            using (DifferentiableOps.PushGradientRelevance(scope))
                step.BackwardFn!(gradOut, step.Inputs, step.OutputBuffer, step.SavedState ?? Array.Empty<object>(), eng, gradMap);
        }

        return eng =>
        {
            DirectGpuTensorEngine.s_currentBackwardOp = "Conv2D+ShortcutPair";
            bool hasMain = gradMap.TryGetValue(main.OutputBuffer, out var gradMain);
            bool hasShortcut = gradMap.TryGetValue(shortcut.OutputBuffer, out var gradShortcut);
            Tensor<float>? gradH = null;
            if (hasMain && hasShortcut && eng is IConvShortcutPairEngine gpu && (relevance ?? requiresGrad).Contains(h))
                gradH = gpu.TryConv2DBackwardInputShortcutPair((Tensor<float>)(object)gradMain!, (Tensor<float>)(object)main.Inputs[1],
                    (Tensor<float>)(object)gradShortcut!, (Tensor<float>)(object)shortcut.Inputs[1], h._shape);
            if (gradH is null)
            {
                if (gradMap.TryGetValue(later.OutputBuffer, out var gl)) Run(later, gl, relevance, eng);
                if (gradMap.TryGetValue(earlier.OutputBuffer, out var ge)) Run(earlier, ge, relevance, eng);
                return;
            }
            Run(later, later == main ? gradMain! : gradShortcut!, withoutH, eng);
            Run(earlier, earlier == main ? gradMain! : gradShortcut!, withoutH, eng);
            DifferentiableOps.AccumulateGrad(gradMap, h, (Tensor<T>)(object)gradH, eng);
        };
    }
}
