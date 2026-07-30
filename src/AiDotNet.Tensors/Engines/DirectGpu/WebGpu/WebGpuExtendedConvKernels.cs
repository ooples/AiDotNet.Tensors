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

    public const string AdaptiveMaxPool2D = @"
@group(0) @binding(0) var<storage,read> input_:array<f32>;
@group(0) @binding(1) var<storage,read_write> output_:array<f32>;
struct PA{batch:i32,channels:i32,inHeight:i32,inWidth:i32,outHeight:i32,outWidth:i32,pad0:i32,pad1:i32};
@group(0) @binding(2) var<uniform> pm:PA;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.batch*pm.channels*pm.outHeight*pm.outWidth){return;}
    let ow=idx%pm.outWidth;
    let oh=(idx/pm.outWidth)%pm.outHeight;
    let c=(idx/(pm.outWidth*pm.outHeight))%pm.channels;
    let b=idx/(pm.outWidth*pm.outHeight*pm.channels);
    let hStart=(oh*pm.inHeight)/pm.outHeight;
    let hEnd=((oh+1)*pm.inHeight)/pm.outHeight;
    let wStart=(ow*pm.inWidth)/pm.outWidth;
    let wEnd=((ow+1)*pm.inWidth)/pm.outWidth;
    var maxV=-3.402823466e38;
    for(var ih=hStart;ih<hEnd;ih=ih+1){
        for(var iw=wStart;iw<wEnd;iw=iw+1){
            let v=input_[((b*pm.channels+c)*pm.inHeight+ih)*pm.inWidth+iw];
            if(v>maxV){maxV=v;}
        }
    }
    output_[((b*pm.channels+c)*pm.outHeight+oh)*pm.outWidth+ow]=maxV;
}";

    // conv3d_backward reuses the 18-int Conv3DUniforms packing but names the input spatial dims D/H/W
    // (matching the OpenCL/GLSL body) rather than iD/iH/iW.
    private const string Conv3DBackwardStruct =
        "struct P3B{N:i32,inC:i32,D:i32,H:i32,W:i32,outC:i32,outD:i32,outH:i32,outW:i32,kD:i32,kH:i32,kW:i32,strideD:i32,strideH:i32,strideW:i32,padD:i32,padH:i32,padW:i32,pad0:i32,pad1:i32};";

    public static readonly string Conv3DBackwardInput = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradInput:array<f32>;
" + Conv3DBackwardStruct + @"
@group(0) @binding(3) var<uniform> pm:P3B;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.N*pm.inC*pm.D*pm.H*pm.W){return;}
    let w=idx%pm.W;
    let h=(idx/pm.W)%pm.H;
    let d=(idx/(pm.W*pm.H))%pm.D;
    let ic=(idx/(pm.W*pm.H*pm.D))%pm.inC;
    let n=idx/(pm.W*pm.H*pm.D*pm.inC);
    var sum=0.0;
    for(var oc=0;oc<pm.outC;oc=oc+1){
        for(var kd=0;kd<pm.kD;kd=kd+1){
            for(var kh=0;kh<pm.kH;kh=kh+1){
                for(var kw=0;kw<pm.kW;kw=kw+1){
                    var od=(d+pm.padD-kd);
                    var oh=(h+pm.padH-kh);
                    var ow=(w+pm.padW-kw);
                    if(od%pm.strideD==0&&oh%pm.strideH==0&&ow%pm.strideW==0){
                        od=od/pm.strideD;
                        oh=oh/pm.strideH;
                        ow=ow/pm.strideW;
                        if(od>=0&&od<pm.outD&&oh>=0&&oh<pm.outH&&ow>=0&&ow<pm.outW){
                            let gradOutIdx=((n*pm.outC+oc)*pm.outD+od)*pm.outH*pm.outW+oh*pm.outW+ow;
                            let kernelIdx=((oc*pm.inC+ic)*pm.kD+kd)*pm.kH*pm.kW+kh*pm.kW+kw;
                            sum=sum+gradOutput[gradOutIdx]*weights[kernelIdx];
                        }
                    }
                }
            }
        }
    }
    gradInput[idx]=sum;
}";

    public static readonly string Conv3DBackwardWeights = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> input_:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradKernel:array<f32>;
" + Conv3DBackwardStruct + @"
@group(0) @binding(3) var<uniform> pm:P3B;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.outC*pm.inC*pm.kD*pm.kH*pm.kW){return;}
    let kw=idx%pm.kW;
    let kh=(idx/pm.kW)%pm.kH;
    let kd=(idx/(pm.kW*pm.kH))%pm.kD;
    let ic=(idx/(pm.kW*pm.kH*pm.kD))%pm.inC;
    let oc=idx/(pm.kW*pm.kH*pm.kD*pm.inC);
    var sum=0.0;
    for(var n=0;n<pm.N;n=n+1){
        for(var od=0;od<pm.outD;od=od+1){
            for(var oh=0;oh<pm.outH;oh=oh+1){
                for(var ow=0;ow<pm.outW;ow=ow+1){
                    let d=od*pm.strideD+kd-pm.padD;
                    let h=oh*pm.strideH+kh-pm.padH;
                    let w=ow*pm.strideW+kw-pm.padW;
                    if(d>=0&&d<pm.D&&h>=0&&h<pm.H&&w>=0&&w<pm.W){
                        let gradOutIdx=((n*pm.outC+oc)*pm.outD+od)*pm.outH*pm.outW+oh*pm.outW+ow;
                        let inputIdx=((n*pm.inC+ic)*pm.D+d)*pm.H*pm.W+h*pm.W+w;
                        sum=sum+gradOutput[gradOutIdx]*input_[inputIdx];
                    }
                }
            }
        }
    }
    gradKernel[idx]=sum;
}";

    // Real spherical-harmonics basis up to degree 3 (16 coeffs). WGSL zero-initializes the local array, so
    // entries above the active degree stay 0 as in the C reference. Uses pm.degree from the PSH uniform.
    private const string ShBasisWgsl = @"
    var basis:array<f32,16>;
    basis[0]=0.282095;
    if(pm.degree>=1){basis[1]=0.488603*dy;basis[2]=0.488603*dz;basis[3]=0.488603*dx;}
    if(pm.degree>=2){
        basis[4]=1.092548*dx*dy;basis[5]=1.092548*dy*dz;
        basis[6]=0.315392*(3.0*dz*dz-1.0);
        basis[7]=1.092548*dx*dz;basis[8]=0.546274*(dx*dx-dy*dy);
    }
    if(pm.degree>=3){
        basis[9]=0.590044*dy*(3.0*dx*dx-dy*dy);
        basis[10]=2.890611*dx*dy*dz;
        basis[11]=0.457046*dy*(5.0*dz*dz-1.0);
        basis[12]=0.373176*dz*(5.0*dz*dz-3.0);
        basis[13]=0.457046*dx*(5.0*dz*dz-1.0);
        basis[14]=1.445306*dz*(dx*dx-dy*dy);
        basis[15]=0.590044*dx*(dx*dx-3.0*dy*dy);
    }";

    private const string ShStruct =
        "struct PSH{numPoints:i32,basisCount:i32,numChannels:i32,degree:i32,broadcastDir:i32,pad0:i32,pad1:i32,pad2:i32};";

    public static readonly string GaussianCovariance = @"
@group(0) @binding(0) var<storage,read> rotations:array<f32>;
@group(0) @binding(1) var<storage,read> scales:array<f32>;
@group(0) @binding(2) var<storage,read_write> covariances:array<f32>;
struct PG{numGaussians:i32,pad0:i32,pad1:i32,pad2:i32};
@group(0) @binding(3) var<uniform> pm:PG;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let i=i32(gid_.x);
    if(i>=pm.numGaussians){return;}
    var qw=rotations[i*4]; var qx=rotations[i*4+1]; var qy=rotations[i*4+2]; var qz=rotations[i*4+3];
    let qNorm=sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
    if(qNorm>0.0){let inv=1.0/qNorm; qw=qw*inv; qx=qx*inv; qy=qy*inv; qz=qz*inv;}
    let r00=1.0-2.0*(qy*qy+qz*qz);
    let r01=2.0*(qx*qy-qw*qz);
    let r02=2.0*(qx*qz+qw*qy);
    let r10=2.0*(qx*qy+qw*qz);
    let r11=1.0-2.0*(qx*qx+qz*qz);
    let r12=2.0*(qy*qz-qw*qx);
    let r20=2.0*(qx*qz-qw*qy);
    let r21=2.0*(qy*qz+qw*qx);
    let r22=1.0-2.0*(qx*qx+qy*qy);
    let sx=max(1e-6,abs(scales[i*3])); let sx2=sx*sx;
    let sy=max(1e-6,abs(scales[i*3+1])); let sy2=sy*sy;
    let sz=max(1e-6,abs(scales[i*3+2])); let sz2=sz*sz;
    let m00=r00*sx2; let m01=r01*sy2; let m02=r02*sz2;
    let m10=r10*sx2; let m11=r11*sy2; let m12=r12*sz2;
    let m20=r20*sx2; let m21=r21*sy2; let m22=r22*sz2;
    let o=i*6;
    covariances[o]=m00*r00+m01*r01+m02*r02;
    covariances[o+1]=m00*r10+m01*r11+m02*r12;
    covariances[o+2]=m00*r20+m01*r21+m02*r22;
    covariances[o+3]=m10*r10+m11*r11+m12*r12;
    covariances[o+4]=m10*r20+m11*r21+m12*r22;
    covariances[o+5]=m20*r20+m21*r21+m22*r22;
}";

    public static readonly string SphericalHarmonics = @"
@group(0) @binding(0) var<storage,read> shCoefficients:array<f32>;
@group(0) @binding(1) var<storage,read> viewDirections:array<f32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
" + ShStruct + @"
@group(0) @binding(3) var<uniform> pm:PSH;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.numPoints*pm.numChannels){return;}
    let ch=idx%pm.numChannels;
    let i=idx/pm.numChannels;
    let dirIdx=select(i,0,pm.broadcastDir!=0);
    var dx=viewDirections[dirIdx*3]; var dy=viewDirections[dirIdx*3+1]; var dz=viewDirections[dirIdx*3+2];
    let norm=sqrt(dx*dx+dy*dy+dz*dz);
    if(norm>0.0){let inv=1.0/norm; dx=dx*inv; dy=dy*inv; dz=dz*inv;}" + ShBasisWgsl + @"
    var color=0.0;
    for(var b=0;b<pm.basisCount;b=b+1){
        color=color+shCoefficients[i*pm.basisCount*pm.numChannels+b*pm.numChannels+ch]*basis[b];
    }
    output_[i*pm.numChannels+ch]=min(max(color,0.0),1.0);
}";

    public static readonly string SphericalHarmonicsBackward = @"
@group(0) @binding(0) var<storage,read> shCoefficients:array<f32>;
@group(0) @binding(1) var<storage,read> viewDirections:array<f32>;
@group(0) @binding(2) var<storage,read> outputGradient:array<f32>;
@group(0) @binding(3) var<storage,read_write> shGrad:array<f32>;
" + ShStruct + @"
@group(0) @binding(4) var<uniform> pm:PSH;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.numPoints*pm.basisCount*pm.numChannels){return;}
    let ch=idx%pm.numChannels;
    let b=(idx/pm.numChannels)%pm.basisCount;
    let i=idx/(pm.basisCount*pm.numChannels);
    let dirIdx=select(i,0,pm.broadcastDir!=0);
    var dx=viewDirections[dirIdx*3]; var dy=viewDirections[dirIdx*3+1]; var dz=viewDirections[dirIdx*3+2];
    let norm=sqrt(dx*dx+dy*dy+dz*dz);
    if(norm>0.0){let inv=1.0/norm; dx=dx*inv; dy=dy*inv; dz=dz*inv;}" + ShBasisWgsl + @"
    var preclamp=0.0;
    for(var bb=0;bb<pm.basisCount;bb=bb+1){
        preclamp=preclamp+shCoefficients[i*pm.basisCount*pm.numChannels+bb*pm.numChannels+ch]*basis[bb];
    }
    var colorGrad=outputGradient[i*pm.numChannels+ch];
    if(preclamp<0.0||preclamp>1.0){colorGrad=0.0;}
    shGrad[idx]=colorGrad*basis[b];
}";

    private const string Pool3DStruct =
        "struct PP{batch:i32,channels:i32,inDepth:i32,inHeight:i32,inWidth:i32,outDepth:i32,outHeight:i32,outWidth:i32,kernelD:i32,kernelH:i32,kernelW:i32,strideD:i32,strideH:i32,strideW:i32,countIncludePad:i32,pad0:i32};";

    public static readonly string AvgPool3D = @"
@group(0) @binding(0) var<storage,read> input_:array<f32>;
@group(0) @binding(1) var<storage,read_write> output_:array<f32>;
" + Pool3DStruct + @"
@group(0) @binding(2) var<uniform> pm:PP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.batch*pm.channels*pm.outDepth*pm.outHeight*pm.outWidth){return;}
    let ow=idx%pm.outWidth;
    let oh=(idx/pm.outWidth)%pm.outHeight;
    let od=(idx/(pm.outWidth*pm.outHeight))%pm.outDepth;
    let c=(idx/(pm.outWidth*pm.outHeight*pm.outDepth))%pm.channels;
    let b=idx/(pm.outWidth*pm.outHeight*pm.outDepth*pm.channels);
    var sum=0.0;
    var count=0;
    for(var kd=0;kd<pm.kernelD;kd=kd+1){
        let id=od*pm.strideD+kd;
        if(id>=pm.inDepth){continue;}
        for(var kh=0;kh<pm.kernelH;kh=kh+1){
            let ih=oh*pm.strideH+kh;
            if(ih>=pm.inHeight){continue;}
            for(var kw=0;kw<pm.kernelW;kw=kw+1){
                let iw=ow*pm.strideW+kw;
                if(iw>=pm.inWidth){continue;}
                let inputIdx=((b*pm.channels+c)*pm.inDepth+id)*pm.inHeight*pm.inWidth+ih*pm.inWidth+iw;
                sum=sum+input_[inputIdx];
                count=count+1;
            }
        }
    }
    let divisor=select(count,pm.kernelD*pm.kernelH*pm.kernelW,pm.countIncludePad!=0);
    let outIdx=((b*pm.channels+c)*pm.outDepth+od)*pm.outHeight*pm.outWidth+oh*pm.outWidth+ow;
    output_[outIdx]=sum/f32(max(divisor,1));
}";

    public static readonly string AvgPool3DBackward = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read_write> gradInput:array<f32>;
" + Pool3DStruct + @"
@group(0) @binding(2) var<uniform> pm:PP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.batch*pm.channels*pm.inDepth*pm.inHeight*pm.inWidth){return;}
    let iw=idx%pm.inWidth;
    let ih=(idx/pm.inWidth)%pm.inHeight;
    let id=(idx/(pm.inWidth*pm.inHeight))%pm.inDepth;
    let c=(idx/(pm.inWidth*pm.inHeight*pm.inDepth))%pm.channels;
    let b=idx/(pm.inWidth*pm.inHeight*pm.inDepth*pm.channels);
    var sum=0.0;
    for(var od=0;od<pm.outDepth;od=od+1){
        let dStart=od*pm.strideD;
        let dEnd=dStart+pm.kernelD;
        if(id<dStart||id>=dEnd){continue;}
        for(var oh=0;oh<pm.outHeight;oh=oh+1){
            let hStart=oh*pm.strideH;
            let hEnd=hStart+pm.kernelH;
            if(ih<hStart||ih>=hEnd){continue;}
            for(var ow=0;ow<pm.outWidth;ow=ow+1){
                let wStart=ow*pm.strideW;
                let wEnd=wStart+pm.kernelW;
                if(iw<wStart||iw>=wEnd){continue;}
                var poolSize=0;
                if(pm.countIncludePad!=0){
                    poolSize=pm.kernelD*pm.kernelH*pm.kernelW;
                }else{
                    let dEndClamp=min(dEnd,pm.inDepth);
                    let hEndClamp=min(hEnd,pm.inHeight);
                    let wEndClamp=min(wEnd,pm.inWidth);
                    poolSize=(dEndClamp-dStart)*(hEndClamp-hStart)*(wEndClamp-wStart);
                }
                let outIdx=((b*pm.channels+c)*pm.outDepth+od)*pm.outHeight*pm.outWidth+oh*pm.outWidth+ow;
                sum=sum+gradOutput[outIdx]/f32(max(poolSize,1));
            }
        }
    }
    let inputIdx=((b*pm.channels+c)*pm.inDepth+id)*pm.inHeight*pm.inWidth+ih*pm.inWidth+iw;
    gradInput[inputIdx]=sum;
}";

    private const string DepthwiseStruct =
        "struct PD{N:i32,inC:i32,H:i32,W:i32,M:i32,outH:i32,outW:i32,kH:i32,kW:i32,strideH:i32,strideW:i32,padH:i32,padW:i32,pad0:i32,pad1:i32,pad2:i32};";

    public static readonly string DepthwiseConv2DBackwardInput = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradInput:array<f32>;
" + DepthwiseStruct + @"
@group(0) @binding(3) var<uniform> pm:PD;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    if(idx>=pm.N*pm.inC*pm.H*pm.W){return;}
    let iw=idx%pm.W;
    let ih=(idx/pm.W)%pm.H;
    let ic=(idx/(pm.W*pm.H))%pm.inC;
    let b=idx/(pm.W*pm.H*pm.inC);
    let outC=pm.inC*pm.M;
    var sum=0.0;
    for(var m=0;m<pm.M;m=m+1){
        let oc=ic*pm.M+m;
        for(var kh=0;kh<pm.kH;kh=kh+1){
            let t=ih+pm.padH-kh;
            if(t<0||(t%pm.strideH)!=0){continue;}
            let oh=t/pm.strideH;
            if(oh<0||oh>=pm.outH){continue;}
            for(var kw=0;kw<pm.kW;kw=kw+1){
                let tw=iw+pm.padW-kw;
                if(tw<0||(tw%pm.strideW)!=0){continue;}
                let ow=tw/pm.strideW;
                if(ow<0||ow>=pm.outW){continue;}
                sum=sum+weights[(oc*pm.kH+kh)*pm.kW+kw]*gradOutput[((b*outC+oc)*pm.outH+oh)*pm.outW+ow];
            }
        }
    }
    gradInput[idx]=sum;
}";

    public static readonly string DepthwiseConv2DBackwardWeights = @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;
@group(0) @binding(1) var<storage,read> input_:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradKernel:array<f32>;
" + DepthwiseStruct + @"
@group(0) @binding(3) var<uniform> pm:PD;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid_:vec3<u32>){
    let idx=i32(gid_.x);
    let outC=pm.inC*pm.M;
    if(idx>=outC*pm.kH*pm.kW){return;}
    let kw=idx%pm.kW;
    let kh=(idx/pm.kW)%pm.kH;
    let oc=idx/(pm.kW*pm.kH);
    let ic=oc/pm.M;
    var sum=0.0;
    for(var b=0;b<pm.N;b=b+1){
        for(var oh=0;oh<pm.outH;oh=oh+1){
            let ih=oh*pm.strideH-pm.padH+kh;
            if(ih<0||ih>=pm.H){continue;}
            for(var ow=0;ow<pm.outW;ow=ow+1){
                let iw=ow*pm.strideW-pm.padW+kw;
                if(iw<0||iw>=pm.W){continue;}
                sum=sum+input_[((b*pm.inC+ic)*pm.H+ih)*pm.W+iw]*gradOutput[((b*outC+oc)*pm.outH+oh)*pm.outW+ow];
            }
        }
    }
    gradKernel[idx]=sum;
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
