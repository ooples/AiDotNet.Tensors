#if NET8_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { var d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++) s+=d[j]; UploadWebGpu(new[]{s/sz}, o); }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { var d=DownloadBuffer(i); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=Math.Min(Math.Max(d[j],mn),mx); UploadWebGpu(r,o); }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { var d=DownloadBuffer(i); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=MathF.Pow(d[j],ex); UploadWebGpu(r,o); }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { var d=DownloadBuffer(i); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]-MathF.Floor(d[j]); UploadWebGpu(r,o); }
    public void EyeKernel(IGpuBuffer o, int n) { var r=new float[n*n]; for(int j=0;j<n;j++) r[j*n+j]=1f; UploadWebGpu(r,o); }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { var id=DownloadBuffer(idx); var r=new float[bs*nc]; for(int b=0;b<bs;b++) r[b*nc+(int)id[b]]=1f; UploadWebGpu(r,o); }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { var d=DownloadBuffer(i); var md=DownloadBuffer(m); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=md[j]!=0?fv:d[j]; UploadWebGpu(r,o); }
    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { var ad=DownloadBuffer(a); var bd=DownloadBuffer(b); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=ad[j]==bd[j]?1f:0f; UploadWebGpu(r,o); }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { var ad=DownloadBuffer(a); var bd=DownloadBuffer(b); var r=new float[sz]; for(int j=0;j<sz;j++) r[j]=ad[j]!=bd[j]?1f:0f; UploadWebGpu(r,o); }
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) { var ad=DownloadBuffer(a); var bd=DownloadBuffer(b); var r=new float[M*N]; for(int i2=0;i2<M;i2++) for(int j=0;j<N;j++) r[i2*N+j]=ad[i2]*bd[j]; UploadWebGpu(r,o); }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) { var ad=DownloadBuffer(a); var bd=DownloadBuffer(b); var r=new float[bs]; for(int i2=0;i2<bs;i2++){float s=0;for(int j=0;j<dim;j++)s+=ad[i2*dim+j]*bd[i2*dim+j];r[i2]=s;} UploadWebGpu(r,o); }
    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { var d=DownloadBuffer(i); var r=new float[os*hd]; int fd=hd*2; for(int j=0;j<os;j++) for(int k=0;k<hd;k++){float v=d[j*fd+k];float g=d[j*fd+hd+k];r[j*hd+k]=v*(1f/(1f+MathF.Exp(-g)));} UploadWebGpu(r,o); }
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { var d=DownloadBuffer(i); var r=new float[os*hd]; int fd=hd*2; for(int j=0;j<os;j++) for(int k=0;k<hd;k++){float v=d[j*fd+k];float g=d[j*fd+hd+k];float x3=v*v*v;r[j*hd+k]=0.5f*v*(1f+MathF.Tanh(0.7978845608f*(v+0.044715f*x3)))*g;} UploadWebGpu(r,o); }
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { var d=DownloadBuffer(i); var r=new float[os*hd]; int fd=hd*2; for(int j=0;j<os;j++) for(int k=0;k<hd;k++) r[j*hd+k]=Math.Max(d[j*fd+k],0f)*d[j*fd+hd+k]; UploadWebGpu(r,o); }
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { var d=DownloadBuffer(i); var r=new float[os*hd]; int fd=hd*2; for(int j=0;j<os;j++) for(int k=0;k<hd;k++){float v=d[j*fd+k];float sig=1f/(1f+MathF.Exp(-v));r[j*hd+k]=v*sig*d[j*fd+hd+k];} UploadWebGpu(r,o); }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { var pd=DownloadBuffer(p); var td=DownloadBuffer(t); var r=new float[sz]; for(int j=0;j<sz;j++){float pc=Math.Min(Math.Max(pd[j],1e-7f),1f-1e-7f);r[j]=-(td[j]*MathF.Log(pc)+(1f-td[j])*MathF.Log(1f-pc));} UploadWebGpu(r,l); }

    private void UploadWebGpu(float[] data, IGpuBuffer buffer)
    {
        if (buffer is WebGpuBuffer wb)
            wb.CopyFrom(data);
    }
    public void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]+sc; UploadWebGpu(r,o); }
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]-sc; UploadWebGpu(r,o); }
    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]+bd[j%isz]; UploadWebGpu(r,o); }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]-bd[j%isz]; UploadWebGpu(r,o); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]*bd[j%isz]; UploadWebGpu(r,o); }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]/(bd[j%isz]+1e-12f); UploadWebGpu(r,o); }
}
#endif
