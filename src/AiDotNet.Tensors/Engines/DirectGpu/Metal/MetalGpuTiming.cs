// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Device-clock timing for the tuned-kernel registry on Metal (#1116).
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Records, on the calling thread, the GPU interval that the command buffers it waits for span: from the earliest
/// <c>GPUStartTime</c> to the latest <c>GPUEndTime</c>. Every Metal op commits its command buffer and waits for it, so
/// the span over a run of ops is what those ops cost a caller on the device clock, gaps between buffers included
/// (Vulkan measures the same interval with timestamp queries).
/// </summary>
internal static class MetalGpuTiming
{
    [ThreadStatic] private static int t_depth;
    [ThreadStatic] private static double t_start;
    [ThreadStatic] private static double t_end;
    [ThreadStatic] private static int t_buffers;
    [ThreadStatic] private static int t_unreported;

    /// <summary>Starts recording on this thread.</summary>
    internal static void Begin()
    {
        if (t_depth++ > 0) return;
        t_start = double.MaxValue;
        t_end = double.MinValue;
        t_buffers = 0;
        t_unreported = 0;
    }

    /// <summary>
    /// Stops recording and returns the span in milliseconds, or <c>null</c> when no buffer completed or one reported no
    /// GPU interval (the caller then times on the host).
    /// </summary>
    internal static double? End()
    {
        if (t_depth == 0) throw new InvalidOperationException("MetalGpuTiming.End without Begin.");
        if (--t_depth > 0) return null;
        if (t_buffers == 0 || t_unreported > 0 || t_end <= t_start) return null;
        return (t_end - t_start) * 1000.0;
    }

    /// <summary>Called after the host waited for <paramref name="commandBuffer"/> to complete.</summary>
    internal static void OnCompleted(IntPtr commandBuffer)
    {
        if (t_depth == 0 || commandBuffer == IntPtr.Zero) return;
        double start = SendMessageDoubleReturn(commandBuffer, Selectors.GpuStartTime);
        double end = SendMessageDoubleReturn(commandBuffer, Selectors.GpuEndTime);
        if (start <= 0 || end < start)
        {
            t_unreported++;
            return;
        }
        t_buffers++;
        if (start < t_start) t_start = start;
        if (end > t_end) t_end = end;
    }
}
