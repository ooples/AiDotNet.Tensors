using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Helpers;

// A bounded thin-N transpose product. Each 192-term partial is rounded before
// accumulation; the final eight columns use two independent K chains.
internal static class ConvTransposeThinGemm
{
 internal static unsafe bool TryMultiply(float[] a,float[] b,float[] c,int m,int n,int k)
 {
#if NET5_0_OR_GREATER
  if(!Avx2.IsSupported || !Fma.IsSupported || m<1024 || m%4!=0 || n!=32 || k%2!=0 || k>512 || k<192) return false;
  CpuParallelSettings.ParallelForOrSerial(0,m/4,(long)m*n*k,row => RowTile(a,b,c,m,n,k,row*4), deterministicSafe: true);
  return true;
#else
  return false;
#endif
 }
#if NET5_0_OR_GREATER
 [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
 private static unsafe void RowTile(float[] a,float[] b,float[] c,int m,int n,int k,int row)
 {
  float* packed=stackalloc float[4*k];
  fixed(float* ap=a) fixed(float* bp=b) fixed(float* cp=c)
  {
   for(int p=0;p<k;p++) for(int r=0;r<4;r++) packed[p*4+r]=ap[p*m+row+r];
   for(int pc=0;pc<k;pc+=192)
   {
    int count=Math.Min(192,k-pc);int col=0;
    for(;col+24<=n;col+=24) Kernel24(packed+pc*4,bp+pc*n+col,cp+row*n+col,n,count,pc==0);
    if(col+16<=n){Kernel16(packed+pc*4,bp+pc*n+col,cp+row*n+col,n,count,pc==0);col+=16;}
    if(col+8<=n)Kernel8(packed+pc*4,bp+pc*n+col,cp+row*n+col,n,count,pc==0);
   }
  }
 }

 [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
 private static unsafe void Kernel24(float* a,float* b,float* c,int n,int k,bool first)
 {
  var c000=Vector256<float>.Zero;
  var c010=Vector256<float>.Zero;
  var c020=Vector256<float>.Zero;
  var c100=Vector256<float>.Zero;
  var c110=Vector256<float>.Zero;
  var c120=Vector256<float>.Zero;
  var c200=Vector256<float>.Zero;
  var c210=Vector256<float>.Zero;
  var c220=Vector256<float>.Zero;
  var c300=Vector256<float>.Zero;
  var c310=Vector256<float>.Zero;
  var c320=Vector256<float>.Zero;
  for(int p=0;p<k;p+=1)
  {
   var b0_0=Avx.LoadVector256(b+(p+0)*n+0);
   var b1_0=Avx.LoadVector256(b+(p+0)*n+8);
   var b2_0=Avx.LoadVector256(b+(p+0)*n+16);
   Vector256<float> av0;
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+0);
   c000=Fma.MultiplyAdd(av0,b0_0,c000);
   c010=Fma.MultiplyAdd(av0,b1_0,c010);
   c020=Fma.MultiplyAdd(av0,b2_0,c020);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+1);
   c100=Fma.MultiplyAdd(av0,b0_0,c100);
   c110=Fma.MultiplyAdd(av0,b1_0,c110);
   c120=Fma.MultiplyAdd(av0,b2_0,c120);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+2);
   c200=Fma.MultiplyAdd(av0,b0_0,c200);
   c210=Fma.MultiplyAdd(av0,b1_0,c210);
   c220=Fma.MultiplyAdd(av0,b2_0,c220);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+3);
   c300=Fma.MultiplyAdd(av0,b0_0,c300);
   c310=Fma.MultiplyAdd(av0,b1_0,c310);
   c320=Fma.MultiplyAdd(av0,b2_0,c320);
  }
  var v00=c000;
  if(!first)v00=Avx.Add(Avx.LoadVector256(c+0*n+0),v00);
  Avx.Store(c+0*n+0,v00);
  var v01=c010;
  if(!first)v01=Avx.Add(Avx.LoadVector256(c+0*n+8),v01);
  Avx.Store(c+0*n+8,v01);
  var v02=c020;
  if(!first)v02=Avx.Add(Avx.LoadVector256(c+0*n+16),v02);
  Avx.Store(c+0*n+16,v02);
  var v10=c100;
  if(!first)v10=Avx.Add(Avx.LoadVector256(c+1*n+0),v10);
  Avx.Store(c+1*n+0,v10);
  var v11=c110;
  if(!first)v11=Avx.Add(Avx.LoadVector256(c+1*n+8),v11);
  Avx.Store(c+1*n+8,v11);
  var v12=c120;
  if(!first)v12=Avx.Add(Avx.LoadVector256(c+1*n+16),v12);
  Avx.Store(c+1*n+16,v12);
  var v20=c200;
  if(!first)v20=Avx.Add(Avx.LoadVector256(c+2*n+0),v20);
  Avx.Store(c+2*n+0,v20);
  var v21=c210;
  if(!first)v21=Avx.Add(Avx.LoadVector256(c+2*n+8),v21);
  Avx.Store(c+2*n+8,v21);
  var v22=c220;
  if(!first)v22=Avx.Add(Avx.LoadVector256(c+2*n+16),v22);
  Avx.Store(c+2*n+16,v22);
  var v30=c300;
  if(!first)v30=Avx.Add(Avx.LoadVector256(c+3*n+0),v30);
  Avx.Store(c+3*n+0,v30);
  var v31=c310;
  if(!first)v31=Avx.Add(Avx.LoadVector256(c+3*n+8),v31);
  Avx.Store(c+3*n+8,v31);
  var v32=c320;
  if(!first)v32=Avx.Add(Avx.LoadVector256(c+3*n+16),v32);
  Avx.Store(c+3*n+16,v32);
 }
 [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
 private static unsafe void Kernel16(float* a,float* b,float* c,int n,int k,bool first)
 {
  var c000=Vector256<float>.Zero;
  var c010=Vector256<float>.Zero;
  var c100=Vector256<float>.Zero;
  var c110=Vector256<float>.Zero;
  var c200=Vector256<float>.Zero;
  var c210=Vector256<float>.Zero;
  var c300=Vector256<float>.Zero;
  var c310=Vector256<float>.Zero;
  for(int p=0;p<k;p+=1)
  {
   var b0_0=Avx.LoadVector256(b+(p+0)*n+0);
   var b1_0=Avx.LoadVector256(b+(p+0)*n+8);
   Vector256<float> av0;
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+0);
   c000=Fma.MultiplyAdd(av0,b0_0,c000);
   c010=Fma.MultiplyAdd(av0,b1_0,c010);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+1);
   c100=Fma.MultiplyAdd(av0,b0_0,c100);
   c110=Fma.MultiplyAdd(av0,b1_0,c110);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+2);
   c200=Fma.MultiplyAdd(av0,b0_0,c200);
   c210=Fma.MultiplyAdd(av0,b1_0,c210);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+3);
   c300=Fma.MultiplyAdd(av0,b0_0,c300);
   c310=Fma.MultiplyAdd(av0,b1_0,c310);
  }
  var v00=c000;
  if(!first)v00=Avx.Add(Avx.LoadVector256(c+0*n+0),v00);
  Avx.Store(c+0*n+0,v00);
  var v01=c010;
  if(!first)v01=Avx.Add(Avx.LoadVector256(c+0*n+8),v01);
  Avx.Store(c+0*n+8,v01);
  var v10=c100;
  if(!first)v10=Avx.Add(Avx.LoadVector256(c+1*n+0),v10);
  Avx.Store(c+1*n+0,v10);
  var v11=c110;
  if(!first)v11=Avx.Add(Avx.LoadVector256(c+1*n+8),v11);
  Avx.Store(c+1*n+8,v11);
  var v20=c200;
  if(!first)v20=Avx.Add(Avx.LoadVector256(c+2*n+0),v20);
  Avx.Store(c+2*n+0,v20);
  var v21=c210;
  if(!first)v21=Avx.Add(Avx.LoadVector256(c+2*n+8),v21);
  Avx.Store(c+2*n+8,v21);
  var v30=c300;
  if(!first)v30=Avx.Add(Avx.LoadVector256(c+3*n+0),v30);
  Avx.Store(c+3*n+0,v30);
  var v31=c310;
  if(!first)v31=Avx.Add(Avx.LoadVector256(c+3*n+8),v31);
  Avx.Store(c+3*n+8,v31);
 }
 [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
 private static unsafe void Kernel8(float* a,float* b,float* c,int n,int k,bool first)
 {
  var c000=Vector256<float>.Zero;
  var c001=Vector256<float>.Zero;
  var c100=Vector256<float>.Zero;
  var c101=Vector256<float>.Zero;
  var c200=Vector256<float>.Zero;
  var c201=Vector256<float>.Zero;
  var c300=Vector256<float>.Zero;
  var c301=Vector256<float>.Zero;
  for(int p=0;p<k;p+=2)
  {
   var b0_0=Avx.LoadVector256(b+(p+0)*n+0);
   Vector256<float> av0;
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+0);
   c000=Fma.MultiplyAdd(av0,b0_0,c000);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+1);
   c100=Fma.MultiplyAdd(av0,b0_0,c100);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+2);
   c200=Fma.MultiplyAdd(av0,b0_0,c200);
   av0=Avx.BroadcastScalarToVector256(a+(p+0)*4+3);
   c300=Fma.MultiplyAdd(av0,b0_0,c300);
   var b0_1=Avx.LoadVector256(b+(p+1)*n+0);
   Vector256<float> av1;
   av1=Avx.BroadcastScalarToVector256(a+(p+1)*4+0);
   c001=Fma.MultiplyAdd(av1,b0_1,c001);
   av1=Avx.BroadcastScalarToVector256(a+(p+1)*4+1);
   c101=Fma.MultiplyAdd(av1,b0_1,c101);
   av1=Avx.BroadcastScalarToVector256(a+(p+1)*4+2);
   c201=Fma.MultiplyAdd(av1,b0_1,c201);
   av1=Avx.BroadcastScalarToVector256(a+(p+1)*4+3);
   c301=Fma.MultiplyAdd(av1,b0_1,c301);
  }
  var v00=Avx.Add(c000,c001);
  if(!first)v00=Avx.Add(Avx.LoadVector256(c+0*n+0),v00);
  Avx.Store(c+0*n+0,v00);
  var v10=Avx.Add(c100,c101);
  if(!first)v10=Avx.Add(Avx.LoadVector256(c+1*n+0),v10);
  Avx.Store(c+1*n+0,v10);
  var v20=Avx.Add(c200,c201);
  if(!first)v20=Avx.Add(Avx.LoadVector256(c+2*n+0),v20);
  Avx.Store(c+2*n+0,v20);
  var v30=Avx.Add(c300,c301);
  if(!first)v30=Avx.Add(Avx.LoadVector256(c+3*n+0),v30);
  Avx.Store(c+3*n+0,v30);
 }
#endif
}
