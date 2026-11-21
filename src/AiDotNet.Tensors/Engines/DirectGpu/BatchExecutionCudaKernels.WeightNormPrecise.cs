// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    // CUDA's normal fast-math module contracts gain*(1/sqrt(sum)) into approximate rsqrt.
    // Keep the explicit g/sqrt(sum+epsilon) scale and floating-point reduction in a precise module.
    // Small rows use contiguous vectors and a warp butterfly; larger rows stream over the
    // input with bounded shared storage. Each block computes its row scale only once.
    internal const string WeightNormPreciseSource = @"
__device__ __forceinline__ float bx_wn_read(const float* v, int rowOffset, int j, int columns) {
    return j < columns ? v[rowOffset+j] : 0.0f;
}
extern ""C"" __global__ __launch_bounds__(256) void bx_weight_norm(const float* direction,const float* gain,float* output,int rows,int columns,float epsilon) {
    int row=(int)blockIdx.x, thread=(int)threadIdx.x;
    if(row>=rows)return;
    bool streaming=columns>4096;
    int width=1;
    if(!streaming)while(width<columns)width*=2;
    int vectorWidth=width<=64?1:width<=128?2:4;
    int virtualThreads=streaming?256:width<=256?64:width<=512?128:width<=1024?256:512;
    __shared__ float warpSums[16];
    for(int pass=0;pass<2;pass++) {
        if(pass==1&&virtualThreads<=256)break;
        int virtualThread=thread+pass*256, base=row*columns;
        float partial;
        if(streaming) {
            partial=0.0f;
            for(long long j=thread;j<columns;j+=256) {
                float value=direction[base+j];
                partial=__fmaf_rn(value,value,partial);
            }
        } else {
            int j=virtualThread*vectorWidth;
            float a=bx_wn_read(direction,base,j,columns);
            float b=vectorWidth>1?bx_wn_read(direction,base,j+1,columns):0.0f;
            float c=vectorWidth>2?bx_wn_read(direction,base,j+2,columns):0.0f;
            float d=vectorWidth>2?bx_wn_read(direction,base,j+3,columns):0.0f;
            if(columns>2048) {
                float aa=bx_wn_read(direction,base,j+2048,columns), bb=bx_wn_read(direction,base,j+2049,columns);
                float cc=bx_wn_read(direction,base,j+2050,columns), dd=bx_wn_read(direction,base,j+2051,columns);
                // A masked tail accumulates into the rounded complete-tile products; a full
                // second tile permits the symmetric pair contraction without a mask merge.
                bool maskedTail=(columns%2048)!=0;
                float p0=maskedTail?__fmaf_rn(aa,aa,__fmul_rn(a,a)):__fmaf_rn(a,a,__fmul_rn(aa,aa));
                float p1=maskedTail?__fmaf_rn(bb,bb,__fmul_rn(b,b)):__fmaf_rn(b,b,__fmul_rn(bb,bb));
                float p2=maskedTail?__fmaf_rn(cc,cc,__fmul_rn(c,c)):__fmaf_rn(c,c,__fmul_rn(cc,cc));
                float p3=maskedTail?__fmaf_rn(dd,dd,__fmul_rn(d,d)):__fmaf_rn(d,d,__fmul_rn(dd,dd));
                partial=__fadd_rn(__fadd_rn(p0,p1),__fadd_rn(p2,p3));
            } else if(vectorWidth==4) {
                partial=__fadd_rn(__fmaf_rn(a,a,__fmul_rn(b,b)),__fmaf_rn(c,c,__fmul_rn(d,d)));
            } else if(vectorWidth==2)partial=__fmaf_rn(a,a,__fmul_rn(b,b));
            else partial=__fmul_rn(a,a);
            if(virtualThread>=virtualThreads)partial=0.0f;
        }
        for(int delta=16;delta>0;delta/=2)partial=__fadd_rn(partial,__shfl_xor_sync(0xffffffff,partial,delta));
        if((thread&31)==0)warpSums[pass*8+(thread>>5)]=partial;
    }
    __syncthreads();
    if(thread<32) {
        int warps=virtualThreads/32;
        float sum=thread<warps?warpSums[thread]:0.0f;
        for(int delta=warps/2;delta>0;delta/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,delta));
        if(thread==0) {
            float radius=__fsqrt_rn(__fadd_rn(sum,epsilon));
            float scale;
            asm(""div.full.f32 %0, %1, %2;"" : ""=f""(scale) : ""f""(gain[row]), ""f""(radius));
            warpSums[0]=scale;
        }
    }
    __syncthreads();
    float scale=warpSums[0];
    for(long long j=thread;j<columns;j+=256)output[(long long)row*columns+j]=__fmul_rn(direction[(long long)row*columns+j],scale);
}
";
}
