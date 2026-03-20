using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Layer-level benchmarks simulating DiffusionResBlock and MultiHeadAttention patterns.
/// Compares workspace-backed zero-allocation path vs standard allocation path.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class LayerLevelBenchmarks
{
    private readonly CpuEngine _engine = new();

    // DiffusionResBlock dimensions (SD15-like)
    private const int Batch = 1;
    private const int Channels = 256;
    private const int Height = 32;
    private const int Width = 32;
    private const int Groups = 32;

    // Attention dimensions
    private const int SeqLen = 64;
    private const int HeadDim = 64;
    private const int NumHeads = 8;

    // ResBlock tensors
    private Tensor<float> _resInput = new(new[] { Batch, Channels, Height, Width });
    private Tensor<float> _resKernel1 = new(new[] { Channels, Channels, 3, 3 });
    private Tensor<float> _resKernel2 = new(new[] { Channels, Channels, 3, 3 });
    private Tensor<float> _resGamma = new(new[] { Channels });
    private Tensor<float> _resBeta = new(new[] { Channels });

    // Attention tensors
    private Tensor<float> _query = new(new[] { Batch, SeqLen, NumHeads * HeadDim });
    private Tensor<float> _key = new(new[] { Batch, SeqLen, NumHeads * HeadDim });
    private Tensor<float> _value = new(new[] { Batch, SeqLen, NumHeads * HeadDim });

    // Workspace for zero-alloc path
    private TensorWorkspace<float> _resWorkspace = new();
    private int _resSlotGN1, _resSlotConv1, _resSlotGN2, _resSlotConv2;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new SimdRandom(42);

        // Initialize ResBlock data
        rng.NextFloats(_resInput.AsWritableSpan());
        rng.NextFloats(_resKernel1.AsWritableSpan());
        rng.NextFloats(_resKernel2.AsWritableSpan());
        rng.NextFloats(_resGamma.AsWritableSpan());
        rng.NextFloats(_resBeta.AsWritableSpan());

        // Scale gamma to reasonable values
        for (int i = 0; i < Channels; i++)
        {
            _resGamma.AsWritableSpan()[i] = 1f + _resGamma.AsSpan()[i] * 0.1f;
            _resBeta.AsWritableSpan()[i] = _resBeta.AsSpan()[i] * 0.1f;
        }

        // Initialize attention data
        rng.NextFloats(_query.AsWritableSpan());
        rng.NextFloats(_key.AsWritableSpan());
        rng.NextFloats(_value.AsWritableSpan());

        // Pre-allocate workspace for ResBlock
        _resWorkspace = new TensorWorkspace<float>();
        _resSlotGN1 = _resWorkspace.Register([Batch, Channels, Height, Width]);
        _resSlotConv1 = _resWorkspace.Register([Batch, Channels, Height, Width]);
        _resSlotGN2 = _resWorkspace.Register([Batch, Channels, Height, Width]);
        _resSlotConv2 = _resWorkspace.Register([Batch, Channels, Height, Width]);
        _resWorkspace.Allocate();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _resWorkspace.Dispose();
    }

    /// <summary>
    /// DiffusionResBlock with standard allocation (allocates per layer).
    /// Pattern: GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> Add
    /// </summary>
    [Benchmark]
    public Tensor<float> ResBlock_StandardAllocation()
    {
        // GroupNorm1 + SiLU
        var gn1 = _engine.GroupNorm(_resInput, Groups, _resGamma, _resBeta, 1e-5, out _, out _);
        var act1 = _engine.Swish(gn1);

        // Conv1
        var conv1 = _engine.Conv2D(act1, _resKernel1, stride: 1, padding: 1);

        // GroupNorm2 + SiLU
        var gn2 = _engine.GroupNorm(conv1, Groups, _resGamma, _resBeta, 1e-5, out _, out _);
        var act2 = _engine.Swish(gn2);

        // Conv2
        var conv2 = _engine.Conv2D(act2, _resKernel2, stride: 1, padding: 1);

        // Residual add
        return _engine.TensorAdd(_resInput, conv2);
    }

    /// <summary>
    /// DiffusionResBlock with TensorWorkspace (zero allocation for intermediates).
    /// Same computation, but intermediates are pre-allocated workspace slots.
    /// </summary>
    [Benchmark]
    public Tensor<float> ResBlock_WorkspaceZeroAlloc()
    {
        // Get pre-allocated tensors from workspace
        var gn1Out = _resWorkspace.Get(_resSlotGN1);
        var conv1Out = _resWorkspace.Get(_resSlotConv1);
        var gn2Out = _resWorkspace.Get(_resSlotGN2);
        var conv2Out = _resWorkspace.Get(_resSlotConv2);

        // GroupNorm1 + SiLU into workspace slot
        _engine.GroupNormSwishInto(gn1Out, _resInput, Groups, _resGamma, _resBeta, 1e-5);

        // Conv1 into workspace slot
        _engine.Conv2DInto(conv1Out, gn1Out, _resKernel1, stride: 1, padding: 1);

        // GroupNorm2 + SiLU into workspace slot
        _engine.GroupNormSwishInto(gn2Out, conv1Out, Groups, _resGamma, _resBeta, 1e-5);

        // Conv2 into workspace slot
        _engine.Conv2DInto(conv2Out, gn2Out, _resKernel2, stride: 1, padding: 1);

        // Residual add (this allocates the output — the only allocation)
        return _engine.TensorAdd(_resInput, conv2Out);
    }

    /// <summary>
    /// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Standard allocation path.
    /// </summary>
    [Benchmark]
    public Tensor<float> Attention_StandardAllocation()
    {
        // Q @ K^T
        var kt = _engine.TensorTranspose(_key);
        var scores = _engine.TensorMatMul(_query, kt);

        // Scale by 1/sqrt(d)
        var scale = 1f / MathF.Sqrt(HeadDim);
        var scaled = _engine.TensorMultiplyScalar(scores, scale);

        // Softmax
        var attn = _engine.TensorSoftmax(scaled, axis: -1);

        // Attn @ V
        return _engine.TensorMatMul(attn, _value);
    }
}
