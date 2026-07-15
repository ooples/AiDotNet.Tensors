namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #775: WebGPU (WGSL) mirrors of the OpenCL extended-conv/geometry kernels. Each kernel is a standalone
// WGSL module string. Scalar params travel in a uniform struct (WGSL has no push constants), padded to a
// 16-byte multiple; the uniform binds at @binding(N) where N = number of storage buffers. WGSL differs
// from C/GLSL on most lines (let/var, f32/i32 casts, params.field access, select() instead of ?:), so the
// source-parity tests carry per-row WGSL markers.
internal static class WebGpuExtendedConvKernels
{
    public const string TrilinearInterpolate = @"
@group(0) @binding(0) var<storage,read> grid:array<f32>;
@group(0) @binding(1) var<storage,read> positions:array<f32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
struct Params{D:i32,H:i32,W:i32,C:i32,P:i32,upperEps:f32,pad0:i32,pad1:i32};
@group(0) @binding(3) var<uniform> pm:Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.P*pm.C){return;}
    let c=idx%pm.C;
    let n=idx/pm.C;
    let z=max(0.0,min(f32(pm.D-1)-pm.upperEps,positions[n*3+0]));
    let y=max(0.0,min(f32(pm.H-1)-pm.upperEps,positions[n*3+1]));
    let x=max(0.0,min(f32(pm.W-1)-pm.upperEps,positions[n*3+2]));
    let z0=i32(floor(z)); let y0=i32(floor(y)); let x0=i32(floor(x));
    let z1=min(z0+1,pm.D-1); let y1=min(y0+1,pm.H-1); let x1=min(x0+1,pm.W-1);
    let fz=z-f32(z0); let fy=y-f32(y0); let fx=x-f32(x0);
    let w000=(1.0-fz)*(1.0-fy)*(1.0-fx); let w001=(1.0-fz)*(1.0-fy)*fx;
    let w010=(1.0-fz)*fy*(1.0-fx); let w011=(1.0-fz)*fy*fx;
    let w100=fz*(1.0-fy)*(1.0-fx); let w101=fz*(1.0-fy)*fx;
    let w110=fz*fy*(1.0-fx); let w111=fz*fy*fx;
    output_[n*pm.C+c]=
        w000*grid[(((z0*pm.H+y0)*pm.W+x0)*pm.C)+c]+w001*grid[(((z0*pm.H+y0)*pm.W+x1)*pm.C)+c]+
        w010*grid[(((z0*pm.H+y1)*pm.W+x0)*pm.C)+c]+w011*grid[(((z0*pm.H+y1)*pm.W+x1)*pm.C)+c]+
        w100*grid[(((z1*pm.H+y0)*pm.W+x0)*pm.C)+c]+w101*grid[(((z1*pm.H+y0)*pm.W+x1)*pm.C)+c]+
        w110*grid[(((z1*pm.H+y1)*pm.W+x0)*pm.C)+c]+w111*grid[(((z1*pm.H+y1)*pm.W+x1)*pm.C)+c];
}";

    public const string TrilinearInterpolateBackward = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> positions:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradGrid:array<f32>;
struct Params{D:i32,H:i32,W:i32,C:i32,P:i32,upperEps:f32,pad0:i32,pad1:i32};
@group(0) @binding(3) var<uniform> pm:Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.D*pm.H*pm.W*pm.C){return;}
    let c=idx%pm.C;
    let gx=(idx/pm.C)%pm.W;
    let gy=(idx/(pm.C*pm.W))%pm.H;
    let gz=idx/(pm.C*pm.W*pm.H);
    var sum=0.0;
    for(var n=0;n<pm.P;n=n+1){
        let z=max(0.0,min(f32(pm.D-1)-pm.upperEps,positions[n*3+0]));
        let y=max(0.0,min(f32(pm.H-1)-pm.upperEps,positions[n*3+1]));
        let x=max(0.0,min(f32(pm.W-1)-pm.upperEps,positions[n*3+2]));
        let z0=i32(floor(z)); let y0=i32(floor(y)); let x0=i32(floor(x));
        let z1=min(z0+1,pm.D-1); let y1=min(y0+1,pm.H-1); let x1=min(x0+1,pm.W-1);
        let fz=z-f32(z0); let fy=y-f32(y0); let fx=x-f32(x0);
        let wz=select(0.0,1.0-fz,gz==z0)+select(0.0,fz,gz==z1);
        if(wz==0.0){continue;}
        let wy=select(0.0,1.0-fy,gy==y0)+select(0.0,fy,gy==y1);
        if(wy==0.0){continue;}
        let wx=select(0.0,1.0-fx,gx==x0)+select(0.0,fx,gx==x1);
        sum=sum+wz*wy*wx*gradOutput[n*pm.C+c];
    }
    gradGrid[idx]=sum;
}";

    private const string ConvT3DStruct =
        "struct P3{N:i32,inC:i32,iD:i32,iH:i32,iW:i32,outC:i32,outD:i32,outH:i32,outW:i32,kD:i32,kH:i32,kW:i32,strideD:i32,strideH:i32,strideW:i32,padD:i32,padH:i32,padW:i32,pad0:i32,pad1:i32};";

    public static readonly string ConvTranspose3D = @"
@group(0) @binding(0) var<storage,read> input_:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
" + ConvT3DStruct + @"
@group(0) @binding(3) var<uniform> pm:P3;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.N*pm.outC*pm.outD*pm.outH*pm.outW){return;}
    let ow=idx%pm.outW;
    let oh=(idx/pm.outW)%pm.outH;
    let od=(idx/(pm.outW*pm.outH))%pm.outD;
    let oc=(idx/(pm.outW*pm.outH*pm.outD))%pm.outC;
    let n=idx/(pm.outW*pm.outH*pm.outD*pm.outC);
    var sum=0.0;
    for(var kd=0;kd<pm.kD;kd=kd+1){
        let td=od+pm.padD-kd;
        if(td<0||(td%pm.strideD)!=0){continue;}
        let id=td/pm.strideD;
        if(id<0||id>=pm.iD){continue;}
        for(var kh=0;kh<pm.kH;kh=kh+1){
            let th=oh+pm.padH-kh;
            if(th<0||(th%pm.strideH)!=0){continue;}
            let ih=th/pm.strideH;
            if(ih<0||ih>=pm.iH){continue;}
            for(var kw=0;kw<pm.kW;kw=kw+1){
                let tw=ow+pm.padW-kw;
                if(tw<0||(tw%pm.strideW)!=0){continue;}
                let iw=tw/pm.strideW;
                if(iw<0||iw>=pm.iW){continue;}
                for(var ic=0;ic<pm.inC;ic=ic+1){
                    sum=sum+input_[(((n*pm.inC+ic)*pm.iD+id)*pm.iH+ih)*pm.iW+iw]*weights[((((ic*pm.outC+oc)*pm.kD+kd)*pm.kH+kh)*pm.kW+kw)];
                }
            }
        }
    }
    output_[idx]=sum;
}";

    public static readonly string ConvTranspose3DBackwardInput = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradInput:array<f32>;
" + ConvT3DStruct + @"
@group(0) @binding(3) var<uniform> pm:P3;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.N*pm.inC*pm.iD*pm.iH*pm.iW){return;}
    let iw=idx%pm.iW;
    let ih=(idx/pm.iW)%pm.iH;
    let id=(idx/(pm.iW*pm.iH))%pm.iD;
    let ic=(idx/(pm.iW*pm.iH*pm.iD))%pm.inC;
    let n=idx/(pm.iW*pm.iH*pm.iD*pm.inC);
    var sum=0.0;
    for(var kd=0;kd<pm.kD;kd=kd+1){
        let od=id*pm.strideD-pm.padD+kd;
        if(od<0||od>=pm.outD){continue;}
        for(var kh=0;kh<pm.kH;kh=kh+1){
            let oh=ih*pm.strideH-pm.padH+kh;
            if(oh<0||oh>=pm.outH){continue;}
            for(var kw=0;kw<pm.kW;kw=kw+1){
                let ow=iw*pm.strideW-pm.padW+kw;
                if(ow<0||ow>=pm.outW){continue;}
                for(var oc=0;oc<pm.outC;oc=oc+1){
                    sum=sum+gradOutput[(((n*pm.outC+oc)*pm.outD+od)*pm.outH+oh)*pm.outW+ow]*weights[((((ic*pm.outC+oc)*pm.kD+kd)*pm.kH+kh)*pm.kW+kw)];
                }
            }
        }
    }
    gradInput[idx]=sum;
}";

    public static readonly string ConvTranspose3DBackwardWeights = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> input_:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradWeights:array<f32>;
" + ConvT3DStruct + @"
@group(0) @binding(3) var<uniform> pm:P3;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.inC*pm.outC*pm.kD*pm.kH*pm.kW){return;}
    let kw=idx%pm.kW;
    let kh=(idx/pm.kW)%pm.kH;
    let kd=(idx/(pm.kW*pm.kH))%pm.kD;
    let oc=(idx/(pm.kW*pm.kH*pm.kD))%pm.outC;
    let ic=idx/(pm.kW*pm.kH*pm.kD*pm.outC);
    var sum=0.0;
    for(var n=0;n<pm.N;n=n+1){
        for(var id=0;id<pm.iD;id=id+1){
            let od=id*pm.strideD-pm.padD+kd;
            if(od<0||od>=pm.outD){continue;}
            for(var ih=0;ih<pm.iH;ih=ih+1){
                let oh=ih*pm.strideH-pm.padH+kh;
                if(oh<0||oh>=pm.outH){continue;}
                for(var iw=0;iw<pm.iW;iw=iw+1){
                    let ow=iw*pm.strideW-pm.padW+kw;
                    if(ow<0||ow>=pm.outW){continue;}
                    sum=sum+input_[(((n*pm.inC+ic)*pm.iD+id)*pm.iH+ih)*pm.iW+iw]*gradOutput[(((n*pm.outC+oc)*pm.outD+od)*pm.outH+oh)*pm.outW+ow];
                }
            }
        }
    }
    gradWeights[idx]=sum;
}";

    private const string SpiralStruct = "struct PS{V:i32,inC:i32,spiralLength:i32,outC:i32};";

    public static readonly string SpiralConv = @"
@group(0) @binding(0) var<storage,read> vertexFeatures:array<f32>;
@group(0) @binding(1) var<storage,read> spiralIndices:array<i32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read> biases:array<f32>;
@group(0) @binding(4) var<storage,read_write> output_:array<f32>;
" + SpiralStruct + @"
@group(0) @binding(5) var<uniform> pm:PS;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.V*pm.outC){return;}
    let oc=idx%pm.outC;
    let v=idx/pm.outC;
    let gatheredSize=pm.inC*pm.spiralLength;
    var sum=biases[oc];
    for(var s=0;s<pm.spiralLength;s=s+1){
        let neighborIdx=spiralIndices[v*pm.spiralLength+s];
        if(neighborIdx<0||neighborIdx>=pm.V){continue;}
        let gatherOffset=s*pm.inC;
        for(var c=0;c<pm.inC;c=c+1){
            sum=sum+vertexFeatures[neighborIdx*pm.inC+c]*weights[oc*gatheredSize+gatherOffset+c];
        }
    }
    output_[idx]=sum;
}";

    public static readonly string SpiralConvBackwardInput = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> spiralIndices:array<i32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read_write> gradVertexFeatures:array<f32>;
" + SpiralStruct + @"
@group(0) @binding(4) var<uniform> pm:PS;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.V*pm.inC){return;}
    let ic=idx%pm.inC;
    let nbr=idx/pm.inC;
    let gatheredSize=pm.inC*pm.spiralLength;
    var sum=0.0;
    for(var v=0;v<pm.V;v=v+1){
        for(var s=0;s<pm.spiralLength;s=s+1){
            if(spiralIndices[v*pm.spiralLength+s]!=nbr){continue;}
            for(var oc=0;oc<pm.outC;oc=oc+1){
                sum=sum+gradOutput[v*pm.outC+oc]*weights[oc*gatheredSize+s*pm.inC+ic];
            }
        }
    }
    gradVertexFeatures[idx]=sum;
}";

    public static readonly string SpiralConvBackwardWeights = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> vertexFeatures:array<f32>;
@group(0) @binding(2) var<storage,read> spiralIndices:array<i32>;
@group(0) @binding(3) var<storage,read_write> gradWeights:array<f32>;
" + SpiralStruct + @"
@group(0) @binding(4) var<uniform> pm:PS;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    let gatheredSize=pm.inC*pm.spiralLength;
    if(idx>=pm.outC*gatheredSize){return;}
    let g=idx%gatheredSize;
    let oc=idx/gatheredSize;
    let s=g/pm.inC;
    let ic=g%pm.inC;
    var sum=0.0;
    for(var v=0;v<pm.V;v=v+1){
        let neighborIdx=spiralIndices[v*pm.spiralLength+s];
        if(neighborIdx<0||neighborIdx>=pm.V){continue;}
        sum=sum+gradOutput[v*pm.outC+oc]*vertexFeatures[neighborIdx*pm.inC+ic];
    }
    gradWeights[idx]=sum;
}";
}
