using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The engine's single-trainer rule for whole-step CUDA-graph capture
/// (<see cref="DirectGpuTensorEngine.NoteCompiledStepThread"/>). A second plan training from another thread must turn
/// capture off. A plan that only changed threads must not: an async training loop resumes on another pool thread after
/// each await, and treating that as a second trainer left HreLm training uncaptured after its first validation.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledStepCaptureOwnershipTests
{
    private static void OnNewThread(Action action)
    {
        Exception? failure = null;
        var t = new Thread(() => { try { action(); } catch (Exception ex) { failure = ex; } });
        t.Start();
        t.Join();
        if (failure is not null) throw new InvalidOperationException("worker failed", failure);
    }

    private static DirectGpuTensorEngine NewEngine()
    {
        try { return new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "GPU engine could not be constructed.");
            throw;
        }
    }

    [SkippableFact]
    public void The_same_plan_on_successive_threads_keeps_capture()
    {
        using var engine = NewEngine();
        OnNewThread(() => engine.NoteCompiledStepThread(1));
        OnNewThread(() => engine.NoteCompiledStepThread(1));
        engine.NoteCompiledStepThread(1);
        Assert.True(engine.AllowsWholeStepCapture);
    }

    [SkippableFact]
    public void A_new_plan_on_the_owners_thread_then_a_new_thread_keeps_capture()
    {
        using var engine = NewEngine();
        engine.NoteCompiledStepThread(1);
        engine.NoteCompiledStepThread(2);   // a recompile on the training thread
        OnNewThread(() => engine.NoteCompiledStepThread(2));
        Assert.True(engine.AllowsWholeStepCapture);
    }

    [SkippableFact]
    public void A_different_plan_on_a_different_thread_turns_capture_off()
    {
        using var engine = NewEngine();
        engine.NoteCompiledStepThread(1);
        OnNewThread(() => engine.NoteCompiledStepThread(2));
        Assert.False(engine.AllowsWholeStepCapture);
        engine.NoteCompiledStepThread(1);   // stays off
        Assert.False(engine.AllowsWholeStepCapture);
    }
}
