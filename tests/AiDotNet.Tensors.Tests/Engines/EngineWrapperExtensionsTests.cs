using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// EngineWrapperExtensions: the wrapper-aware is/as/cast/GetType helpers behave like the language operators on a plain
/// engine and see through any chain of IEngineWrapper decorators (nested wrappers, null, cycles).
/// </summary>
public sealed class EngineWrapperExtensionsTests
{
    [Fact]
    public void PlainEngine_BehavesLikeTheLanguageOperators()
    {
        IEngine engine = new CpuEngine();

        Assert.True(engine.Is<CpuEngine>());
        Assert.True(engine.Is<ITensorLevelEngine>());
        Assert.False(engine.Is<DirectGpuTensorEngine>());
        Assert.False(engine.Is<IEngineWrapper>());
        Assert.Same(engine, engine.As<CpuEngine>());
        Assert.Null(engine.As<DirectGpuTensorEngine>());
        Assert.True(engine.TryGet<CpuEngine>(out var cpu));
        Assert.Same(engine, cpu);
        Assert.False(engine.TryGet<DirectGpuTensorEngine>(out var gpu));
        Assert.Null(gpu);
        Assert.Same(engine, engine.Cast<CpuEngine>());
        Assert.Throws<InvalidCastException>(() => engine.Cast<DirectGpuTensorEngine>());
        Assert.Same(engine, engine.Unwrap());
    }

    [Fact]
    public void NullEngine_MatchesNothing()
    {
        IEngine? engine = null;

        Assert.Null(engine.Unwrap());
        Assert.False(engine.Is<CpuEngine>());
        Assert.Null(engine.As<CpuEngine>());
        Assert.False(engine.TryGet<CpuEngine>(out var cpu));
        Assert.Null(cpu);
        Assert.Null(engine.Cast<CpuEngine>()); // like (CpuEngine)null
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void WrappedEngine_IsFoundThroughEveryWrapper(int depth)
    {
        var inner = new CpuEngine();
        IEngine engine = inner;
        for (int i = 0; i < depth; i++) engine = WrapperProxy.Wrap(engine);

        Assert.False(engine is CpuEngine); // the reason for the helpers
        Assert.True(engine.Is<CpuEngine>());
        Assert.True(engine.Is<ITensorLevelEngine>());
        Assert.False(engine.Is<DirectGpuTensorEngine>());
        Assert.Same(inner, engine.As<CpuEngine>());
        Assert.Same(inner, engine.As<ITensorLevelEngine>());
        Assert.True(engine.TryGet<CpuEngine>(out var cpu));
        Assert.Same(inner, cpu);
        Assert.Same(inner, engine.Cast<CpuEngine>());
        var error = Assert.Throws<InvalidCastException>(() => engine.Cast<DirectGpuTensorEngine>());
        Assert.Contains(nameof(CpuEngine), error.Message, StringComparison.Ordinal);
        Assert.Same(inner, engine.Unwrap());
        Assert.Equal(typeof(CpuEngine), engine.Unwrap().GetType());
    }

    [Fact]
    public void Lookup_StopsAtTheOutermostMatch()
    {
        var inner = new CpuEngine();
        var middle = WrapperProxy.Wrap(inner);
        var outer = WrapperProxy.Wrap(middle);

        Assert.Same(outer, outer.As<IEngineWrapper>());
        Assert.Same(middle, ((IEngineWrapper)outer).Inner);
        Assert.Same(inner, middle.Unwrap());
    }

    [Fact]
    public void CyclicChain_Throws()
    {
        var wrapper = WrapperProxy.Wrap(new CpuEngine());
        WrapperProxy.Retarget(wrapper, wrapper);

        Assert.Throws<InvalidOperationException>(() => wrapper.Unwrap());
        Assert.Throws<InvalidOperationException>(() => wrapper.Is<DirectGpuTensorEngine>());
        Assert.True(wrapper.Is<IEngineWrapper>()); // the outermost engine matches before the chain is walked
    }

    [Fact]
    public void WrappedEngine_ForwardsCalls()
    {
        IEngine engine = WrapperProxy.Wrap(new CpuEngine());

        Assert.False(engine.SupportsGpu);
    }

    /// <summary>A wrapper engine: implements IEngine (and IEngineWrapper) only, forwarding every call to Inner.</summary>
    public interface ITestWrapperEngine : IEngine, IEngineWrapper
    {
    }

    public class WrapperProxy : DispatchProxy
    {
        private IEngine _inner = null!;

        public static ITestWrapperEngine Wrap(IEngine inner)
        {
            var proxy = DispatchProxy.Create<ITestWrapperEngine, WrapperProxy>();
            ((WrapperProxy)(object)proxy)._inner = inner;
            return proxy;
        }

        public static void Retarget(ITestWrapperEngine wrapper, IEngine inner) => ((WrapperProxy)(object)wrapper)._inner = inner;

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null) throw new ArgumentNullException(nameof(targetMethod));
            if (targetMethod.Name == "get_Inner" && targetMethod.DeclaringType == typeof(IEngineWrapper)) return _inner;
            try
            {
                return targetMethod.Invoke(_inner, args);
            }
            catch (TargetInvocationException e) when (e.InnerException is not null)
            {
                throw e.InnerException;
            }
        }
    }
#endif
}
