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
}
