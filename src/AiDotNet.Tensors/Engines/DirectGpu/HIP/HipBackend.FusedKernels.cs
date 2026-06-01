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
    public unsafe void DotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        // Issue #382: deterministic variant fixes accumulation order by
        // launching a single block (kernel itself ignores blockIdx.x — only
        // threadIdx.x stride is used) and writing `*result = sum` instead of
        // atomicAdd.
        string kname = GpuDeterminism.IsActive ? "dot_product_deterministic" : "dot_product";
        if (_kernelCache.TryGetValue(kname, out var k))
        {
            IntPtr ap=a.Handle,bp=b.Handle,op=output.Handle;
            void** ar=stackalloc void*[4];
            ar[0]=&ap;ar[1]=&bp;ar[2]=&op;ar[3]=&size;
            uint grid = GpuDeterminism.IsActive ? 1u : (uint)((size+DefaultBlockSize-1)/DefaultBlockSize);
            LaunchKernel(k, grid, DefaultBlockSize, ar);
            return;
        }
        float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float s=0;for(int i=0;i<size;i++)s+=ad[i]*bd[i]; UploadToExistingHip(new[]{s},output);
    }
    public unsafe void StridedDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size, int strideA, int strideB, int count)
    {
        // Issue #382: deterministic variant launched single-block (kernel uses
        // only threadIdx.x stride). Note: the kernel signature is
        // (a, b, result, aSize, bSize, bOffset, bStride) — preserve the existing
        // wire-up; argument-name mismatch is a separate concern.
        string kname = GpuDeterminism.IsActive ? "strided_dot_product_deterministic" : "strided_dot_product";
        if (_kernelCache.TryGetValue(kname, out var k))
        {
            IntPtr ap=a.Handle,bp=b.Handle,op=output.Handle;
            void** ar=stackalloc void*[7];
            ar[0]=&ap;ar[1]=&bp;ar[2]=&op;ar[3]=&size;ar[4]=&strideA;ar[5]=&strideB;ar[6]=&count;
            uint grid = GpuDeterminism.IsActive ? 1u : (uint)((count+DefaultBlockSize-1)/DefaultBlockSize);
            LaunchKernel(k, grid, DefaultBlockSize, ar);
            return;
        }
        float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[count];for(int c2=0;c2<count;c2++){float s=0;for(int i=0;i<size;i++)s+=ad[c2*strideA+i]*bd[c2*strideB+i];r[c2]=s;}UploadToExistingHip(r,output);
    }
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
            if (buf is null) throw new ArgumentNullException(nameof(buffers), $"{opName}: GPU buffer cannot be null.");
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

    // ─── HRR binding primitives (issue #248) ────────────────────────

    public unsafe void SplitComplexUnitPhaseCodebook(
        IGpuBuffer outReal, IGpuBuffer outImag, int seed, int V, int D, bool kPsk, int k)
    {
        // Reject negative dims up front — if both are negative, their
        // product is positive and would slip past a naive "total <= 0"
        // check, then launch the kernel with invalid shape.
        if (V < 0) throw new ArgumentOutOfRangeException(nameof(V), "V must be >= 0.");
        if (D < 0) throw new ArgumentOutOfRangeException(nameof(D), "D must be >= 0.");
        if (V == 0 || D == 0) return;
        long total = (long)V * D;
        if (total > int.MaxValue) throw new ArgumentException($"V*D = {total} exceeds int.MaxValue.");
        if (kPsk && k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        int n = (int)total;
        ValidateHipSplitBuffers(n, nameof(SplitComplexUnitPhaseCodebook), outReal, outImag);
        if (!_kernelCache.TryGetValue("hrr_unit_phase_codebook", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_unit_phase_codebook");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pOR = outReal.Handle, pOI = outImag.Handle;
        int kPskI = kPsk ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &pOR; args[1] = &pOI; args[2] = &seed;
        args[3] = &V; args[4] = &D; args[5] = &kPskI; args[6] = &k;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    public unsafe void SplitComplexPhaseCoherenceDecode(
        IGpuBuffer codesReal, IGpuBuffer codesImag,
        IGpuBuffer queryReal, IGpuBuffer queryImag,
        IGpuBuffer outScores, int V, int D)
    {
        if (V <= 0 || D <= 0) return;
        long codeCount = (long)V * D;
        if (codesReal is null || codesImag is null || queryReal is null || queryImag is null || outScores is null)
            throw new ArgumentNullException(nameof(codesReal),
                "All five GPU buffers must be non-null for SplitComplexPhaseCoherenceDecode.");
        if (codesReal.Size < codeCount || codesImag.Size < codeCount)
            throw new ArgumentException(
                $"codes buffers must each hold at least V*D = {codeCount} elements " +
                $"(got {codesReal.Size}, {codesImag.Size}).");
        if (queryReal.Size < D || queryImag.Size < D)
            throw new ArgumentException(
                $"query buffers must each hold at least D = {D} elements " +
                $"(got {queryReal.Size}, {queryImag.Size}).");
        if (outScores.Size < V)
            throw new ArgumentException(
                $"outScores must hold at least V = {V} elements (got {outScores.Size}).");
        if (!_kernelCache.TryGetValue("hrr_phase_coherence_decode", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_phase_coherence_decode");
        uint grid = (uint)V;
        int blockSize = 1;
        while (blockSize < D && blockSize < 1024) blockSize <<= 1;
        blockSize = Math.Max(32, blockSize);
        uint sharedMem = (uint)blockSize * sizeof(float);
        IntPtr pCR = codesReal.Handle, pCI = codesImag.Handle;
        IntPtr pQR = queryReal.Handle, pQI = queryImag.Handle;
        IntPtr pOS = outScores.Handle;
        void** args = stackalloc void*[7];
        args[0] = &pCR; args[1] = &pCI;
        args[2] = &pQR; args[3] = &pQI;
        args[4] = &pOS; args[5] = &V; args[6] = &D;
        // Route through the shared helper so this runs on _stream with
        // the same error-surfacing as the rest of the backend.
        LaunchKernelWithSharedMem(kernel, grid, (uint)blockSize, sharedMem, args);
    }

    public unsafe void SplitComplexHrrBindAccumulate(
        IGpuBuffer keyCodeReal, IGpuBuffer keyCodeImag,
        IGpuBuffer valPermCodeReal, IGpuBuffer valPermCodeImag,
        IGpuBuffer keyIds, IGpuBuffer valIds,
        IGpuBuffer memoryReal, IGpuBuffer memoryImag,
        int N, int D)
    {
        if (N <= 0 || D <= 0) return;
        if (keyCodeReal is null || keyCodeImag is null || valPermCodeReal is null || valPermCodeImag is null
            || keyIds is null || valIds is null || memoryReal is null || memoryImag is null)
            throw new ArgumentNullException(nameof(keyCodeReal),
                "All eight GPU buffers must be non-null for SplitComplexHrrBindAccumulate.");
        if (keyIds.Size < N || valIds.Size < N)
            throw new ArgumentException(
                $"ID buffers must each hold at least N = {N} elements " +
                $"(got keyIds={keyIds.Size}, valIds={valIds.Size}).");
        if (memoryReal.Size < D || memoryImag.Size < D)
            throw new ArgumentException(
                $"Memory buffers must each hold at least D = {D} elements " +
                $"(got memoryReal={memoryReal.Size}, memoryImag={memoryImag.Size}).");
        if (keyCodeReal.Size < D || keyCodeImag.Size < D
            || valPermCodeReal.Size < D || valPermCodeImag.Size < D)
            throw new ArgumentException(
                $"Each codebook buffer must hold at least one full row of D = {D} elements.");
        // Derive vocabulary sizes from codebook capacities so the
        // kernel can reject out-of-range ids without OOB reads —
        // mirrors the CUDA backend's nKeys/nVals guard and the CPU
        // path's range check in NativeHRRBindAccumulate.
        long nKeysL = keyCodeReal.Size / D;
        long nValsL = valPermCodeReal.Size / D;
        if (nKeysL <= 0 || nValsL <= 0)
            throw new ArgumentException(
                $"Codebook buffers must hold at least one full row of D = {D} elements.");
        if (nKeysL > int.MaxValue || nValsL > int.MaxValue)
            throw new ArgumentException("Codebook vocabulary size exceeds int.MaxValue.");
        int nKeys = (int)nKeysL;
        int nVals = (int)nValsL;
        if (!_kernelCache.TryGetValue("hrr_bind_accumulate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: hrr_bind_accumulate");
        uint grid = (uint)((D + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pKR = keyCodeReal.Handle, pKI = keyCodeImag.Handle;
        IntPtr pVR = valPermCodeReal.Handle, pVI = valPermCodeImag.Handle;
        IntPtr pKId = keyIds.Handle, pVId = valIds.Handle;
        IntPtr pMR = memoryReal.Handle, pMI = memoryImag.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pKR; args[1] = &pKI;
        args[2] = &pVR; args[3] = &pVI;
        args[4] = &pKId; args[5] = &pVId;
        args[6] = &pMR; args[7] = &pMI;
        args[8] = &N; args[9] = &D;
        args[10] = &nKeys; args[11] = &nVals;
        LaunchKernel(kernel, grid, (uint)DefaultBlockSize, args);
    }

    // ── Fused GLA (Gated Linear Attention) scan (#1464) ────────────────────────────────────
    // q/k/v: [batch, seqLen, modelDim]; gate: [batch, seqLen, numHeads]; output: [batch, seqLen, modelDim].
    public unsafe void GlaScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (headDim > Kernels.HipGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.HipGlaKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gla_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gla_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pg = gate.Handle, po = output.Handle;
        void** args = stackalloc void*[10];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pg; args[4] = &po;
        args[5] = &batch; args[6] = &seqLen; args[7] = &modelDim; args[8] = &numHeads; args[9] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // dQ/dK/dV (each [batch, seqLen, modelDim]) and dG ([batch, seqLen, numHeads]) MUST be
    // pre-zeroed — the backward kernel accumulates into them via atomicAdd.
    public unsafe void GlaScanBackward(
        IGpuBuffer dOut, IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer gate,
        IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, IGpuBuffer dG,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (headDim > Kernels.HipGlaKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GLA headDim ({headDim}) exceeds max ({Kernels.HipGlaKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gla_scan_recompute", out var recompute) ||
            !_kernelCache.TryGetValue("gla_scan_backward", out var backward))
            throw new InvalidOperationException("HIP kernel not found: gla_scan_recompute / gla_scan_backward");

        // State trajectory scratch: [batch * numHeads * seqLen * headDim * headDim].
        int trajLen = batch * numHeads * seqLen * headDim * headDim;
        using var trajBuf = AllocateBuffer(trajLen);
        uint total = (uint)(batch * numHeads * headDim);
        uint grid = (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize;

        IntPtr pk = k.Handle, pv = v.Handle, pg = gate.Handle, ptr = trajBuf.Handle;
        void** ra = stackalloc void*[9];
        ra[0] = &pk; ra[1] = &pv; ra[2] = &pg; ra[3] = &ptr;
        ra[4] = &batch; ra[5] = &seqLen; ra[6] = &modelDim; ra[7] = &numHeads; ra[8] = &headDim;
        LaunchKernel(recompute, grid, (uint)DefaultBlockSize, ra);

        IntPtr pdo = dOut.Handle, pq = q.Handle;
        IntPtr pdq = dQ.Handle, pdk = dK.Handle, pdv = dV.Handle, pdg = dG.Handle;
        void** ba = stackalloc void*[15];
        ba[0] = &pdo; ba[1] = &pq; ba[2] = &pk; ba[3] = &pv; ba[4] = &pg; ba[5] = &ptr;
        ba[6] = &pdq; ba[7] = &pdk; ba[8] = &pdv; ba[9] = &pdg;
        ba[10] = &batch; ba[11] = &seqLen; ba[12] = &modelDim; ba[13] = &numHeads; ba[14] = &headDim;
        LaunchKernel(backward, grid, (uint)DefaultBlockSize, ba);
    }

    // ── Fused xLSTM (mLSTM) scan forward (#1464) ───────────────────────────────────────────
    public unsafe void XLstmScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v,
        IGpuBuffer iGate, IGpuBuffer fGate, IGpuBuffer oGate, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (headDim > Kernels.HipXLstmKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"xLSTM headDim ({headDim}) exceeds max ({Kernels.HipXLstmKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("xlstm_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: xlstm_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle;
        IntPtr pi = iGate.Handle, pf = fGate.Handle, po = oGate.Handle, pout = output.Handle;
        void** args = stackalloc void*[12];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pi; args[4] = &pf; args[5] = &po; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &modelDim; args[10] = &numHeads; args[11] = &headDim;
        uint total = (uint)(batch * numHeads);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Gated DeltaNet scan forward (#1464) ──────────────────────────────────────────
    public unsafe void GatedDeltaNetScanForward(
        IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer alpha, IGpuBuffer beta, IGpuBuffer output,
        int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        if (headDim > Kernels.HipGatedDeltaNetKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"GatedDeltaNet headDim ({headDim}) exceeds max ({Kernels.HipGatedDeltaNetKernels.MaxHeadDim}).");
        if (!_kernelCache.TryGetValue("gated_delta_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: gated_delta_scan_forward");

        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, pa = alpha.Handle, pb = beta.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &pa; args[4] = &pb; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim; args[9] = &numHeads; args[10] = &headDim;
        uint total = (uint)(batch * numHeads * headDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RG-LRU scan forward (#1464) ──────────────────────────────────────────────────
    public unsafe void RgLruScanForward(
        IGpuBuffer value, IGpuBuffer recGate, IGpuBuffer inpGate, IGpuBuffer decay, IGpuBuffer output,
        int batch, int seqLen, int recDim)
    {
        if (!_kernelCache.TryGetValue("rglru_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rglru_scan_forward");

        IntPtr pv = value.Handle, pr = recGate.Handle, pi = inpGate.Handle, pd = decay.Handle, pout = output.Handle;
        void** args = stackalloc void*[8];
        args[0] = &pv; args[1] = &pr; args[2] = &pi; args[3] = &pd; args[4] = &pout;
        args[5] = &batch; args[6] = &seqLen; args[7] = &recDim;
        uint total = (uint)(batch * recDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused RWKV-4 WKV scan forward (#1464) ──────────────────────────────────────────────
    public unsafe void Rwkv4WkvForward(
        IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer timeDecay, IGpuBuffer timeFirst, IGpuBuffer output,
        int batch, int seqLen, int modelDim)
    {
        if (!_kernelCache.TryGetValue("rwkv4_wkv_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: rwkv4_wkv_forward");

        IntPtr pr = r.Handle, pk = k.Handle, pv = v.Handle, pd = timeDecay.Handle, pf = timeFirst.Handle, pout = output.Handle;
        void** args = stackalloc void*[9];
        args[0] = &pr; args[1] = &pk; args[2] = &pv; args[3] = &pd; args[4] = &pf; args[5] = &pout;
        args[6] = &batch; args[7] = &seqLen; args[8] = &modelDim;
        uint total = (uint)(batch * modelDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba selective scan forward (#1464) ─────────────────────────────────────────
    public unsafe void MambaSelectiveScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int stateDim)
    {
        if (stateDim > Kernels.HipMambaKernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba stateDim ({stateDim}) exceeds max ({Kernels.HipMambaKernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba_selective_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mamba_selective_scan_forward");

        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[11];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused Mamba-2 SSD scan forward (#1464) ─────────────────────────────────────────────
    public unsafe void Mamba2SsdScanForward(
        IGpuBuffer x, IGpuBuffer delta, IGpuBuffer aLog, IGpuBuffer bParam, IGpuBuffer cParam, IGpuBuffer dParam,
        IGpuBuffer output, int batch, int seqLen, int innerDim, int numHeads, int headDim, int stateDim)
    {
        if (stateDim > Kernels.HipMamba2Kernels.MaxStateDim)
            throw new InvalidOperationException(
                $"Mamba-2 stateDim ({stateDim}) exceeds max ({Kernels.HipMamba2Kernels.MaxStateDim}).");
        if (!_kernelCache.TryGetValue("mamba2_ssd_scan_forward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: mamba2_ssd_scan_forward");

        IntPtr px = x.Handle, pdt = delta.Handle, pa = aLog.Handle, pb = bParam.Handle, pc = cParam.Handle, pd = dParam.Handle, pout = output.Handle;
        void** args = stackalloc void*[13];
        args[0] = &px; args[1] = &pdt; args[2] = &pa; args[3] = &pb; args[4] = &pc; args[5] = &pd; args[6] = &pout;
        args[7] = &batch; args[8] = &seqLen; args[9] = &innerDim; args[10] = &numHeads; args[11] = &headDim; args[12] = &stateDim;
        uint total = (uint)(batch * innerDim);
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
    }

    // ── Fused linear (LM head) + cross-entropy (#1464) ─────────────────────────────────────
    // Returns the mean cross-entropy over the N rows. targetIds holds the class ids as floats.
    public unsafe float FusedLinearCrossEntropyIndex(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer targetIds, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_index", hidden, weight, bias, targetIds, n, d, vocab);

    public unsafe float FusedLinearCrossEntropyDense(
        IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer target, int n, int d, int vocab)
        => FusedCeLaunch("fused_linear_ce_dense", hidden, weight, bias, target, n, d, vocab);

    private unsafe float FusedCeLaunch(
        string kernelName, IGpuBuffer hidden, IGpuBuffer weight, IGpuBuffer bias, IGpuBuffer tgt, int n, int d, int vocab)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {kernelName}");
        using var lossBuf = AllocateBuffer(1); // zeroed
        IntPtr ph = hidden.Handle, pw = weight.Handle, pb = bias.Handle, pt = tgt.Handle, pl = lossBuf.Handle;
        void** args = stackalloc void*[8];
        args[0] = &ph; args[1] = &pw; args[2] = &pb; args[3] = &pt; args[4] = &pl;
        args[5] = &n; args[6] = &d; args[7] = &vocab;
        uint total = (uint)n;
        LaunchKernel(kernel, (total + (uint)DefaultBlockSize - 1) / (uint)DefaultBlockSize, (uint)DefaultBlockSize, args);
        var loss = DownloadBuffer(lossBuf);
        return loss[0] / n;
    }
}
