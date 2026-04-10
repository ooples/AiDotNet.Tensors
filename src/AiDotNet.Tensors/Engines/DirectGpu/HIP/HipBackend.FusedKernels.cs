using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    // Every method tries GPU kernel first via _kernelCache, CPU fallback only if kernel not compiled.

    public unsafe void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { if (_kernelCache.TryGetValue("reduce_mean", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[3]; a[0]=&ip;a[1]=&op;a[2]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++)s+=d[j]; UploadToExistingHip(new[]{s/sz},o); }
    public unsafe void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { if (_kernelCache.TryGetValue("clip_kernel", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ip;a[1]=&op;a[2]=&mn;a[3]=&mx;a[4]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=Math.Min(Math.Max(d[j],mn),mx);UploadToExistingHip(r,o); }
    public unsafe void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { if (_kernelCache.TryGetValue("pow_scalar", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&ex;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=MathF.Pow(d[j],ex);UploadToExistingHip(r,o); }
    public unsafe void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { if (_kernelCache.TryGetValue("frac_kernel", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[3]; a[0]=&ip;a[1]=&op;a[2]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]-MathF.Floor(d[j]);UploadToExistingHip(r,o); }
    public unsafe void EyeKernel(IGpuBuffer o, int n) { if (_kernelCache.TryGetValue("eye_kernel", out var k)) { IntPtr op=o.Handle; void** a=stackalloc void*[2]; a[0]=&op;a[1]=&n; LaunchKernel(k,(uint)((n*n+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] r=new float[n*n]; for(int j=0;j<n;j++) r[j*n+j]=1f; UploadToExistingHip(r,o); }
    public unsafe void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { if (_kernelCache.TryGetValue("one_hot_kernel", out var k)) { IntPtr ip=idx.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&bs;a[3]=&nc; LaunchKernel(k,(uint)((bs*nc+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] id=DownloadBuffer(idx); float[] r=new float[bs*nc]; for(int b=0;b<bs;b++) r[b*nc+(int)id[b]]=1f; UploadToExistingHip(r,o); }
    public unsafe void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { if (_kernelCache.TryGetValue("masked_fill_kernel", out var k)) { IntPtr ip=i.Handle,mp=m.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ip;a[1]=&mp;a[2]=&op;a[3]=&fv;a[4]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i); float[] md=DownloadBuffer(m); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=md[j]!=0?fv:d[j]; UploadToExistingHip(r,o); }
    public unsafe void EqualsKernel(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int sz) { if (_kernelCache.TryGetValue("equals_kernel", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=ad[j]==bd[j]?1f:0f;UploadToExistingHip(r,o); }
    public unsafe void NotEqualsKernel(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int sz) { if (_kernelCache.TryGetValue("not_equals_kernel", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=ad[j]!=bd[j]?1f:0f;UploadToExistingHip(r,o); }
    public unsafe void OuterProduct(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int M, int N) { if (_kernelCache.TryGetValue("outer_product", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&M;a[4]=&N; LaunchKernel(k,(uint)((M*N+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[M*N];for(int i2=0;i2<M;i2++)for(int j=0;j<N;j++)r[i2*N+j]=ad[i2]*bd[j];UploadToExistingHip(r,o); }
    public unsafe void BatchDotProduct(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int bs, int dim) { if (_kernelCache.TryGetValue("batch_dot_product", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&bs;a[4]=&dim; LaunchKernel(k,(uint)((bs+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[bs];for(int i2=0;i2<bs;i2++){float s=0;for(int j=0;j<dim;j++)s+=ad[i2*dim+j]*bd[i2*dim+j];r[i2]=s;}UploadToExistingHip(r,o); }
    public unsafe void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { if (_kernelCache.TryGetValue("glu_forward", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&os;a[3]=&hd; LaunchKernel(k,(uint)((os*hd+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int x=0;x<hd;x++){float v=d[j*fd+x];float g=d[j*fd+hd+x];r[j*hd+x]=v*(1f/(1f+MathF.Exp(-g)));}UploadToExistingHip(r,o); }
    public unsafe void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { if (_kernelCache.TryGetValue("geglu_forward", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&os;a[3]=&hd; LaunchKernel(k,(uint)((os*hd+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int x=0;x<hd;x++){float v=d[j*fd+x];float g=d[j*fd+hd+x];float x3=v*v*v;r[j*hd+x]=0.5f*v*(1f+MathF.Tanh(0.7978845608f*(v+0.044715f*x3)))*g;}UploadToExistingHip(r,o); }
    public unsafe void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { if (_kernelCache.TryGetValue("reglu_forward", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&os;a[3]=&hd; LaunchKernel(k,(uint)((os*hd+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int x=0;x<hd;x++)r[j*hd+x]=Math.Max(d[j*fd+x],0f)*d[j*fd+hd+x];UploadToExistingHip(r,o); }
    public unsafe void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { if (_kernelCache.TryGetValue("swiglu_forward", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&os;a[3]=&hd; LaunchKernel(k,(uint)((os*hd+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int x=0;x<hd;x++){float v=d[j*fd+x];float sig=1f/(1f+MathF.Exp(-v));r[j*hd+x]=v*sig*d[j*fd+hd+x];}UploadToExistingHip(r,o); }
    public unsafe void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { if (_kernelCache.TryGetValue("bce_loss", out var k)) { IntPtr pp=p.Handle,tp=t.Handle,lp=l.Handle; void** a=stackalloc void*[4]; a[0]=&pp;a[1]=&tp;a[2]=&lp;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] pd=DownloadBuffer(p);float[] td=DownloadBuffer(t);float[] r=new float[sz];for(int j=0;j<sz;j++){float pc=Math.Min(Math.Max(pd[j],1e-7f),1f-1e-7f);r[j]=-(td[j]*MathF.Log(pc)+(1f-td[j])*MathF.Log(1f-pc));}UploadToExistingHip(r,l); }
    public unsafe void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { if (_kernelCache.TryGetValue("add_scalar", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&sc;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]+sc;UploadToExistingHip(r,o); }
    public unsafe void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { if (_kernelCache.TryGetValue("sub_scalar", out var k)) { IntPtr ip=i.Handle,op=o.Handle; void** a=stackalloc void*[4]; a[0]=&ip;a[1]=&op;a[2]=&sc;a[3]=&sz; LaunchKernel(k,(uint)((sz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]-sc;UploadToExistingHip(r,o); }
    public unsafe void BroadcastAddLast(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int os, int isz) { if (_kernelCache.TryGetValue("broadcast_add_last", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&os;a[4]=&isz; LaunchKernel(k,(uint)((os*isz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]+bd[j%isz];UploadToExistingHip(r,o); }
    public unsafe void BroadcastSubLast(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int os, int isz) { if (_kernelCache.TryGetValue("broadcast_sub_last", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&os;a[4]=&isz; LaunchKernel(k,(uint)((os*isz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]-bd[j%isz];UploadToExistingHip(r,o); }
    public unsafe void BroadcastMulLast(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int os, int isz) { if (_kernelCache.TryGetValue("broadcast_mul_last", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&os;a[4]=&isz; LaunchKernel(k,(uint)((os*isz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]*bd[j%isz];UploadToExistingHip(r,o); }
    public unsafe void BroadcastDivLast(IGpuBuffer a1, IGpuBuffer b1, IGpuBuffer o, int os, int isz) { if (_kernelCache.TryGetValue("broadcast_div_last", out var k)) { IntPtr ap=a1.Handle,bp=b1.Handle,op=o.Handle; void** a=stackalloc void*[5]; a[0]=&ap;a[1]=&bp;a[2]=&op;a[3]=&os;a[4]=&isz; LaunchKernel(k,(uint)((os*isz+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,a); return; } float[] ad=DownloadBuffer(a1);float[] bd=DownloadBuffer(b1);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]/(bd[j%isz]+1e-12f);UploadToExistingHip(r,o); }
    public unsafe void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) { if (_kernelCache.TryGetValue("dot_product", out var k)) { IntPtr ap=a.Handle,bp=b.Handle,op=output.Handle; void** ar=stackalloc void*[4]; ar[0]=&ap;ar[1]=&bp;ar[2]=&op;ar[3]=&size; LaunchKernel(k,(uint)((size+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,ar); return; } float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float s=0;for(int i=0;i<size;i++)s+=ad[i]*bd[i]; UploadToExistingHip(new[]{s},output); }
    public unsafe void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size, int strideA, int strideB, int count) { if (_kernelCache.TryGetValue("strided_dot_product", out var k)) { IntPtr ap=a.Handle,bp=b.Handle,op=output.Handle; void** ar=stackalloc void*[7]; ar[0]=&ap;ar[1]=&bp;ar[2]=&op;ar[3]=&size;ar[4]=&strideA;ar[5]=&strideB;ar[6]=&count; LaunchKernel(k,(uint)((count+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,ar); return; } float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[count];for(int c2=0;c2<count;c2++){float s=0;for(int i=0;i<size;i++)s+=ad[c2*strideA+i]*bd[c2*strideB+i];r[c2]=s;}UploadToExistingHip(r,output); }
    public unsafe void BatchedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim) { if (_kernelCache.TryGetValue("batched_dot_product", out var k)) { IntPtr ap=a.Handle,bp=b.Handle,op=output.Handle; void** ar=stackalloc void*[5]; ar[0]=&ap;ar[1]=&bp;ar[2]=&op;ar[3]=&batchSize;ar[4]=&dim; LaunchKernel(k,(uint)((batchSize+DefaultBlockSize-1)/DefaultBlockSize),DefaultBlockSize,ar); return; } float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[batchSize];for(int i=0;i<batchSize;i++){float s=0;for(int j=0;j<dim;j++)s+=ad[i*dim+j]*bd[i*dim+j];r[i]=s;}UploadToExistingHip(r,output); }

    private void UploadToExistingHip(float[] data, IGpuBuffer buffer)
    {
        unsafe
        {
            fixed (float* src = data)
            {
                var sizeBytes = (UIntPtr)(data.Length * sizeof(float));
                HipNativeBindings.hipMemcpy(buffer.Handle, (IntPtr)src, sizeBytes, HipMemcpyKind.HostToDevice);
            }
        }
    }

    // --- Split-buffer native Complex<T> operations (HIP dispatch) ---

    private static void ValidateHipSplitBuffers(int n, string opName, params IGpuBuffer[] buffers)
    {
        foreach (var buf in buffers)
        {
            if (buf is null) throw new ArgumentNullException(opName, "GPU buffer cannot be null.");
            if (n > buf.Size) throw new ArgumentException($"{opName}: n ({n}) exceeds buffer size ({buf.Size}).");
        }
    }

    public unsafe void SplitComplexMultiply(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMultiply), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_multiply", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_multiply. Register HipComplexKernels.");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexConjugate(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexConjugate), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_conjugate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_conjugate");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5]; args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitude(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMagnitude), inReal, inImag, outMag);
        if (!_kernelCache.TryGetValue("split_complex_magnitude", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_magnitude");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMag.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexMagnitudeSquared(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outMagSq, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexMagnitudeSquared), inReal, inImag, outMagSq);
        if (!_kernelCache.TryGetValue("split_complex_magnitude_squared", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_magnitude_squared");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outMagSq.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexPhase(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outPhase, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexPhase), inReal, inImag, outPhase);
        if (!_kernelCache.TryGetValue("split_complex_phase", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_phase");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pO = outPhase.Handle;
        void** args = stackalloc void*[4]; args[0] = &pIR; args[1] = &pII; args[2] = &pO; args[3] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexFromPolar(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexFromPolar), mag, phase, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_from_polar", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_from_polar");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pM = mag.Handle, pP = phase.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[5]; args[0] = &pM; args[1] = &pP; args[2] = &pOR; args[3] = &pOI; args[4] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexScale(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, float scalar, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexScale), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_scale", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_scale");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[6]; args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &scalar; args[5] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexAdd(IGpuBuffer aReal, IGpuBuffer aImag, IGpuBuffer bReal, IGpuBuffer bImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexAdd), aReal, aImag, bReal, bImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_add", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_add");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pAR = aReal.Handle, pAI = aImag.Handle, pBR = bReal.Handle, pBI = bImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pAR; args[1] = &pAI; args[2] = &pBR; args[3] = &pBI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexCrossSpectral(IGpuBuffer xReal, IGpuBuffer xImag, IGpuBuffer yReal, IGpuBuffer yImag, IGpuBuffer outReal, IGpuBuffer outImag, int n)
    {
        if (n <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexCrossSpectral), xReal, xImag, yReal, yImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_cross_spectral", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_cross_spectral");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pXR = xReal.Handle, pXI = xImag.Handle, pYR = yReal.Handle, pYI = yImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
        void** args = stackalloc void*[7]; args[0] = &pXR; args[1] = &pXI; args[2] = &pYR; args[3] = &pYI; args[4] = &pOR; args[5] = &pOI; args[6] = &n;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexTopK(IGpuBuffer inReal, IGpuBuffer inImag, IGpuBuffer outReal, IGpuBuffer outImag, int n, int k)
    {
        if (n <= 0 || k <= 0) return;
        ValidateHipSplitBuffers(n, nameof(SplitComplexTopK), inReal, inImag, outReal, outImag);
        if (!_kernelCache.TryGetValue("split_complex_topk", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: split_complex_topk");
        // Compute threshold on CPU
        var magBuf = AllocateBuffer(n);
        try
        {
            SplitComplexMagnitudeSquared(inReal, inImag, magBuf, n);
            var magData = DownloadBuffer(magBuf);
            Array.Sort(magData); Array.Reverse(magData);
            float threshold = k <= n ? magData[Math.Min(k, n) - 1] : 0f;
            uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
            IntPtr pIR = inReal.Handle, pII = inImag.Handle, pOR = outReal.Handle, pOI = outImag.Handle;
            void** args = stackalloc void*[6];
            args[0] = &pIR; args[1] = &pII; args[2] = &pOR; args[3] = &pOI; args[4] = &threshold; args[5] = &n;
            LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
        }
        finally { magBuf.Dispose(); }
    }

    public unsafe void SoftmaxRows(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0) return;
        if (!_kernelCache.TryGetValue("softmax_rows", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: softmax_rows");
        uint grid = (uint)rows;
        int blockSize = Math.Min(256, cols);
        int sharedMem = blockSize * sizeof(float);
        IntPtr pI = input.Handle, pO = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pI; args[1] = &pO; args[2] = &rows; args[3] = &cols;
        HipNativeBindings.hipModuleLaunchKernel(kernel, grid, 1, 1, (uint)blockSize, 1, 1,
            (uint)sharedMem, IntPtr.Zero, (IntPtr)args, IntPtr.Zero);
    }
}
