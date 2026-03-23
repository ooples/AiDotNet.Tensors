namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    // All methods use Metal GPU dispatch via GetPipeline + DispatchThreadgroups.
    // CPU fallback only for ops where MSL kernel source doesn't exist yet.

    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("Reduction", _reductionLibrary, "mean_axis", i, o, sz); }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { SimpleUnary("Activation", _activationLibrary, "clamp", i, o, sz); }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "pow_scalar", i, o, sz); }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "frac_kernel", i, o, sz); }
    public void EyeKernel(IGpuBuffer o, int n) { SimpleGenerate("Reduction", _reductionLibrary, "eye_kernel", o, n*n); }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { SimpleUnary("Reduction", _reductionLibrary, "one_hot_kernel", idx, o, bs*nc); }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "masked_fill_kernel", i, m, o, sz); }
    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "equals_kernel", a, b, o, sz); }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "not_equals_kernel", a, b, o, sz); }
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) { SimpleBinary("Reduction", _reductionLibrary, "outer_product", a, b, o, M*N); }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) { SimpleBinary("Reduction", _reductionLibrary, "batch_dot_product", a, b, o, bs); }
    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "glu_forward", i, o, os*hd); }
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "geglu_forward", i, o, os*hd); }
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "reglu_forward", i, o, os*hd); }
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "swiglu_forward", i, o, os*hd); }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { SimpleBinary("Loss", _lossLibrary, "bce_loss", p, t, l, sz); }
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "sub_scalar", i, o, sz); }
    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_add_last", a, b, o, os*isz); }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_sub_last", a, b, o, os*isz); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_mul_last", a, b, o, os*isz); }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_div_last", a, b, o, os*isz); }

    private void SimpleUnary(string lib, System.IntPtr library, string kernel, IGpuBuffer i, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (i is not MetalGpuBuffer ib || o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ib, 0);
        encoder.SetBuffer(ob, 1);
        encoder.SetBytes((uint)sz, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void SimpleBinary(string lib, System.IntPtr library, string kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer ab || b is not MetalGpuBuffer bb || o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ab, 0);
        encoder.SetBuffer(bb, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)sz, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void SimpleGenerate(string lib, System.IntPtr library, string kernel, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffer must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ob, 0);
        encoder.SetBytes((uint)sz, 1);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
