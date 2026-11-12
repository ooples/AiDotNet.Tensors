using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The create-graph (higher-order) paths of the backward functions are lambdas over the op's tensors and saved state
/// (TryCompositeBackward, LinearGrad, DiagonalLinearGrad, TryLinearBackward, ...). A lambda that captures a parameter
/// or a top-level local makes the compiler allocate its closure on ENTRY to the method, so every ordinary first-order
/// backward paid a closure and its delegates (two for ReLU) before it even checked <c>CreatesGraph</c>. The
/// create-graph paths now live in their own methods, entered only when a graph is being created.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class FirstOrderBackwardAllocationTests
{
    private readonly ITestOutputHelper _output;

    public FirstOrderBackwardAllocationTests(ITestOutputHelper output) => _output = output;

    // Backward functions whose FIRST-ORDER work itself runs a capturing lambda (a parallel row loop, or a host loop that
    // builds its result through a per-element callback). Their closure is part of the kernel, not of a create-graph
    // path, and is unrelated to the regression this test pins.
    private static readonly HashSet<string> FirstOrderKernelClosures = new(StringComparer.Ordinal)
    {
        "BackwardFunctions`1.ELUBackward",               // BackwardParallel.ForRows over chunks
        "BackwardFunctions`1.SliceBackward",             // CpuParallelSettings.ParallelForOrSerial scatter
        "BackwardFunctions`1.MelSpectrogramBackward",    // CpuParallelSettings.ParallelForOrSerial over rows
        "CpuEngine.AttentionCoreBackwardFloat",          // parallel heads
        "CpuEngine.FusedAttentionBackwardFloat",         // parallel heads
        "CpuEngine.GruSequenceBackwardFloat",            // parallel batch rows
        "CpuEngine.MultiHeadAttentionBackwardFloat",     // parallel heads
        "CpuEngine.ReduceMeanBackwardCompiledFloat",     // parallel rows
        "CpuEngine.SoftmaxCrossEntropyBackwardFloat",    // parallel rows
        "CpuEngine.QuantileBackward",                    // FromDoubles(shape, i => dx[i])
        "CpuEngine.GridSample3DBackward",                // FromDoubles(shape, i => dx[i])
        "CpuEngine.RreluBackward",                       // FromDoubles(shape, i => ...)
        "CpuEngine.WindowPoolBackward",                  // host loop through FromDoubles
    };

    [Fact]
    public void BackwardFunctions_AllocateNoClosureOutsideTheirCreateGraphPaths()
    {
        var assembly = typeof(CpuEngine).Assembly;
        var offenders = new List<string>();
        int scanned = 0;
        foreach (var type in new[] { typeof(BackwardFunctions<>), typeof(CpuEngine) })
        {
            foreach (var method in type.GetMethods(BindingFlags.Static | BindingFlags.Instance | BindingFlags.Public
                                                   | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!IsBackwardFunction(method)) continue;
                scanned++;
                var allocations = ClosureAllocations(method);
                string name = type.Name + "." + method.Name;
                if (allocations.Count > 0 && !FirstOrderKernelClosures.Contains(name))
                    offenders.Add($"{name}: {string.Join(", ", allocations)}");
            }
        }
        Assert.True(scanned > 300, $"only {scanned} backward functions found; the signature filter is stale");
        Assert.True(offenders.Count == 0,
            "These backward functions allocate a closure or delegate on every call, including first-order ones. Move the " +
            "create-graph lambda into its own method entered under CreatesGraph:\n" + string.Join("\n", offenders));
    }

#if NET5_0_OR_GREATER
    // The ReLU backward is the hottest of them: it allocated a closure and two delegates per call before the fix.
    [Fact]
    public void ReluFirstOrderBackward_AllocatesNothingBeyondItsKernel()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 64 });
        var g = new Tensor<float>(new[] { 64 });
        for (int i = 0; i < x.Length; i++) { x[i] = i - 32; g[i] = 1; }
        var y = engine.ReLU(x);
        var inputs = new[] { x };
        var grads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        var none = Array.Empty<object>();

        double kernel = BytesPerCall(() =>
        {
            grads.Clear();
            DifferentiableOps.AccumulateGrad(grads, x, engine.ReluBackward(g, x), engine);
        });
        double backward = BytesPerCall(() =>
        {
            grads.Clear();
            BackwardFunctions<float>.ReLUBackward(g, inputs, y, none, engine, grads);
        });
        _output.WriteLine($"ReLU first-order backward: {backward:F1} B/call; kernel + accumulate alone: {kernel:F1} B/call");
        Assert.True(backward - kernel < 16,
            $"ReLU first-order backward allocates {backward - kernel:F1} B/call more than its kernel (a closure and delegates)");
    }

    private static double BytesPerCall(Action body)
    {
        for (int i = 0; i < 50; i++) body();
        const int calls = 2000;
        long best = long.MaxValue;
        for (int round = 0; round < 3; round++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < calls; i++) body();
            best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
        }
        return (double)best / calls;
    }
#endif

    private static bool IsBackwardFunction(MethodInfo method)
    {
        if (method.ReturnType != typeof(void)) return false;
        ParameterInfo[] ps;
        try { ps = method.GetParameters(); }
        catch (Exception) { return false; }
        return ps.Length == 6
               && ps[1].ParameterType.IsArray
               && ps[3].ParameterType == typeof(object[])
               && ps[4].ParameterType == typeof(IEngine)
               && ps[5].ParameterType.IsGenericType
               && ps[5].ParameterType.GetGenericTypeDefinition() == typeof(Dictionary<,>);
    }

    private static readonly Dictionary<short, OpCode> OpCodesByValue = typeof(OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(f => f.GetValue(null))
        .OfType<OpCode>()
        .GroupBy(o => o.Value)
        .ToDictionary(g => g.Key, g => g.First());

    // Every `newobj` of a compiler-generated closure class, and of a delegate that is not the one-time fill of a cached
    // static lambda (`newobj; dup; stsfld`).
    private static List<string> ClosureAllocations(MethodInfo method)
    {
        var found = new List<string>();
        var il = method.GetMethodBody()?.GetILAsByteArray();
        if (il is null) return found;
        var typeArgs = method.DeclaringType!.IsGenericType ? method.DeclaringType.GetGenericArguments() : null;
        var methodArgs = method.IsGenericMethod ? method.GetGenericArguments() : null;
        var ops = new List<(OpCode Op, int Token)>();
        for (int i = 0; i < il.Length;)
        {
            short value = il[i] == 0xFE ? unchecked((short)(0xFE00 | il[i + 1])) : il[i];
            var op = OpCodesByValue[value];
            i += op.Size;
            int token = 0;
            switch (op.OperandType)
            {
                case OperandType.InlineNone: break;
                case OperandType.ShortInlineBrTarget:
                case OperandType.ShortInlineI:
                case OperandType.ShortInlineVar: i += 1; break;
                case OperandType.InlineVar: i += 2; break;
                case OperandType.InlineI8:
                case OperandType.InlineR: i += 8; break;
                case OperandType.InlineSwitch: i += 4 + 4 * BitConverter.ToInt32(il, i); break;
                default: token = BitConverter.ToInt32(il, i); i += 4; break;
            }
            ops.Add((op, token));
        }
        for (int k = 0; k < ops.Count; k++)
        {
            if (ops[k].Op != OpCodes.Newobj) continue;
            MethodBase? ctor;
            try { ctor = method.Module.ResolveMethod(ops[k].Token, typeArgs, methodArgs); }
            catch (Exception) { continue; }
            var type = ctor?.DeclaringType;
            if (type is null) continue;
            if (type.Name.Contains("DisplayClass")) found.Add("closure " + type.Name);
            else if (typeof(Delegate).IsAssignableFrom(type)
                     && !(k + 2 < ops.Count && ops[k + 1].Op == OpCodes.Dup && ops[k + 2].Op == OpCodes.Stsfld))
                found.Add("delegate " + type.Name);
        }
        return found.Distinct().ToList();
    }
}
