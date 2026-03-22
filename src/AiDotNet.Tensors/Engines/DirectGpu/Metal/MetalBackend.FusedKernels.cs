namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("Reduction", _reductionLibrary, "mean_axis", i, o, sz); }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { SimpleUnary("Activation", _activationLibrary, "clamp", i, o, sz); }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "pow_scalar", i, o, sz); }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "frac_kernel", i, o, sz); }
    public void EyeKernel(IGpuBuffer o, int n) { float[] r = new float[n*n]; for(int j=0;j<n;j++) r[j*n+j]=1f; UploadToExisting(r, o); }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { float[] id = DownloadBuffer(idx); float[] r = new float[bs*nc]; for(int b=0;b<bs;b++) r[b*nc+(int)id[b]]=1f; UploadToExisting(r, o); }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { float[] d=DownloadBuffer(i); float[] md=DownloadBuffer(m); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=md[j]!=0?fv:d[j]; UploadToExisting(r,o); }
    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "equals_kernel", a, b, o, sz); }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "not_equals_kernel", a, b, o, sz); }
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[M*N]; for(int i2=0;i2<M;i2++) for(int j=0;j<N;j++) r[i2*N+j]=ad[i2]*bd[j]; UploadToExisting(r,o); }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[bs]; for(int i2=0;i2<bs;i2++){float s=0;for(int j=0;j<dim;j++)s+=ad[i2*dim+j]*bd[i2*dim+j];r[i2]=s;} UploadToExisting(r,o); }
    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "glu_forward", i, o, os*hd); }
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "geglu_forward", i, o, os*hd); }
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "reglu_forward", i, o, os*hd); }
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "swiglu_forward", i, o, os*hd); }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { SimpleBinary("Loss", _lossLibrary, "bce_loss", p, t, l, sz); }

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

    private void UploadToExisting(float[] data, IGpuBuffer buffer)
    {
        if (buffer is MetalGpuBuffer mb)
            System.Runtime.InteropServices.Marshal.Copy(data, 0, mb.Handle, data.Length);
    }
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]-sc; UploadToExisting(r,o); }
    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]+bd[j%isz]; UploadToExisting(r,o); }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]-bd[j%isz]; UploadToExisting(r,o); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]*bd[j%isz]; UploadToExisting(r,o); }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]/(bd[j%isz]+1e-12f); UploadToExisting(r,o); }
}
