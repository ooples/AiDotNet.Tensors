using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

internal static class GradientShadowSavedState
{
    internal static bool TryProtect<T>(object? initial, ISet<IGpuBuffer> roots,
        Stack<object> pending, HashSet<object> seen)
    {
        pending.Clear();
        seen.Clear();
        if (initial is not null) pending.Push(initial);
        void Protect(IGpuBuffer? buffer) { if (buffer is not null) roots.Add(GpuBufferViewRoot.Of(buffer)); }
        while (pending.Count != 0)
        {
            var value = pending.Pop();
            if (!seen.Add(value)) continue;
            if (value is Tensor<T> tensor) { Protect(tensor._gpuBuffer); continue; }
            if (value is Tensor<int> ints) { Protect(ints._gpuBuffer); continue; }
            if (value is Tensor<long> longs) { Protect(longs._gpuBuffer); continue; }
            if (value is IGpuBuffer buffer) { Protect(buffer); continue; }
            if (value is object[] nested)
            {
                foreach (var item in nested) if (item is not null) pending.Push(item);
                continue;
            }
            if (value is ISavedStateTensorContainer container)
            {
                foreach (var item in container.SavedStateValues) if (item is not null) pending.Push(item);
                continue;
            }
            var type = value.GetType();
            if (!(type.IsPrimitive || type.IsEnum || value is string || value is int[] || value is long[]
                || value is float[] || value is double[] || value is bool[] || value is int[][])) return false;
        }
        return true;
    }
}
