// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanOptimizerKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    public static string TwoBuffer => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };
layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };
layout(push_constant) uniform Params { uint size; uint operation; float learningRate; float p0; };
void main(){uint i=gl_GlobalInvocationID.x;if(i>=size)return;if(operation==0u)param[i]-=learningRate*(gradient[i]+p0*param[i]);else{float value=param[i]-learningRate*gradient[i],magnitude=abs(value)-learningRate*p0;param[i]=magnitude>0.0?sign(value)*magnitude:0.0;}}
";

    public static string ThreeBuffer => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };
layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };
layout(set=0,binding=2) buffer State { float state[]; };
layout(push_constant) uniform Params { uint size; uint operation; float learningRate; float p0; float p1; float p2; };
void main(){uint i=gl_GlobalInvocationID.x;if(i>=size)return;float g=gradient[i];if(operation==0u){state[i]=p0*state[i]+g+p1*param[i];param[i]-=learningRate*state[i];}else if(operation==1u){g+=p2*param[i];state[i]=p0*state[i]+(1.0-p0)*g*g;param[i]-=learningRate*g/(sqrt(state[i])+p1);}else if(operation==2u){g+=p1*param[i];state[i]+=g*g;param[i]-=learningRate*g/(sqrt(state[i])+p0);}else if(operation==3u){float previous=state[i];state[i]=p0*state[i]-learningRate*(g+p1*param[i]);param[i]+=-p0*previous+(1.0+p0)*state[i];}else{float update=p0*state[i]+(1.0-p0)*g;param[i]-=learningRate*(sign(update)+p2*param[i]);state[i]=p1*state[i]+(1.0-p1)*g;}}
";

    public static string FourBuffer => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };
layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };
layout(set=0,binding=2) buffer State1 { float state1[]; };
layout(set=0,binding=3) buffer State2 { float state2[]; };
layout(push_constant) uniform Params { uint size; uint operation; uint step; float learningRate; float p0; float p1; float p2; float p3; };
void main(){uint i=gl_GlobalInvocationID.x;if(i>=size)return;float g=gradient[i];if(operation<=1u){float beta1=p0,beta2=p1,epsilon=p2,decay=p3;if(operation==0u)g+=decay*param[i];else param[i]*=1.0-learningRate*decay;state1[i]=beta1*state1[i]+(1.0-beta1)*g;state2[i]=beta2*state2[i]+(1.0-beta2)*g*g;float mh=state1[i]/(1.0-pow(beta1,float(step))),vh=state2[i]/(1.0-pow(beta2,float(step)));param[i]-=learningRate*mh/(sqrt(vh)+epsilon);}else if(operation==2u){float rho=p0,epsilon=p1,decay=p2;state1[i]=rho*state1[i]+(1.0-rho)*g*g;float update=sqrt(state2[i]+epsilon)/sqrt(state1[i]+epsilon)*g;state2[i]=rho*state2[i]+(1.0-rho)*update*update;param[i]-=update+decay*param[i];}else if(operation==3u){float beta1=p0,beta2=p1,epsilon=p2,decay=p3;g+=decay*param[i];state1[i]=beta1*state1[i]+(1.0-beta1)*g;state2[i]=max(beta2*state2[i],abs(g));param[i]-=learningRate/(1.0-pow(beta1,float(step)))*state1[i]/(state2[i]+epsilon);}else if(operation==4u){float beta1=p0,beta2=p1,epsilon=p2,decay=p3;g+=decay*param[i];state1[i]=beta1*state1[i]+(1.0-beta1)*g;state2[i]=beta2*state2[i]+(1.0-beta2)*g*g;float mh=state1[i]/(1.0-pow(beta1,float(step))),vh=state2[i]/(1.0-pow(beta2,float(step)));float nesterov=beta1*mh+(1.0-beta1)*g/(1.0-pow(beta1,float(step+1u)));param[i]-=learningRate*nesterov/(sqrt(vh)+epsilon);}else{float l1=p0,l2=p1,beta=p2;float sigma=(sqrt(state2[i]+g*g)-sqrt(state2[i]))/learningRate;state1[i]+=g-sigma*param[i];state2[i]+=g*g;param[i]=abs(state1[i])<=l1?0.0:-(state1[i]-l1*sign(state1[i]))/(l2+(beta+sqrt(state2[i]))/learningRate);}}
";

    public static string Amsgrad => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };layout(set=0,binding=2) buffer M { float m[]; };layout(set=0,binding=3) buffer V { float v[]; };layout(set=0,binding=4) buffer VMax { float vmax[]; };
layout(push_constant) uniform Params { uint size; uint step; float learningRate; float beta1; float beta2; float epsilon; float weightDecay; };
void main(){uint i=gl_GlobalInvocationID.x;if(i>=size)return;float g=gradient[i]+weightDecay*param[i];m[i]=beta1*m[i]+(1.0-beta1)*g;v[i]=beta2*v[i]+(1.0-beta2)*g*g;vmax[i]=max(vmax[i],v[i]);float mh=m[i]/(1.0-pow(beta1,float(step)));param[i]-=learningRate*mh/(sqrt(vmax[i]/(1.0-pow(beta2,float(step))))+epsilon);}
";

    public static string Lars => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };layout(set=0,binding=2) buffer Velocity { float velocity[]; };
layout(push_constant) uniform Params { uint size; float learningRate; float momentum; float weightDecay; float trustCoefficient; };
void main(){if(gl_GlobalInvocationID.x!=0u)return;float pn=0.0,gn=0.0;for(uint i=0u;i<size;++i){pn+=param[i]*param[i];gn+=gradient[i]*gradient[i];}pn=sqrt(pn);gn=sqrt(gn);float local=pn>0.0&&gn>0.0?trustCoefficient*pn/(gn+weightDecay*pn):1.0;for(uint i=0u;i<size;++i){velocity[i]=momentum*velocity[i]+local*(gradient[i]+weightDecay*param[i]);param[i]-=learningRate*velocity[i];}}
";

    public static string Lamb => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };layout(set=0,binding=1) readonly buffer Gradient { float gradient[]; };layout(set=0,binding=2) buffer M { float m[]; };layout(set=0,binding=3) buffer V { float v[]; };
layout(push_constant) uniform Params { uint size; uint step; float learningRate; float beta1; float beta2; float epsilon; float weightDecay; };
void main(){if(gl_GlobalInvocationID.x!=0u)return;float bc1=1.0-pow(beta1,float(step)),bc2=1.0-pow(beta2,float(step));for(uint i=0u;i<size;++i){float g=gradient[i];m[i]=beta1*m[i]+(1.0-beta1)*g;v[i]=beta2*v[i]+(1.0-beta2)*g*g;}float pn=0.0,un=0.0;for(uint i=0u;i<size;++i){float update=m[i]/bc1/(sqrt(v[i]/bc2)+epsilon)+weightDecay*param[i];pn+=param[i]*param[i];un+=update*update;}float ratio=un>0.0?sqrt(pn)/sqrt(un):1.0;for(uint i=0u;i<size;++i){float update=m[i]/bc1/(sqrt(v[i]/bc2)+epsilon)+weightDecay*param[i];param[i]-=learningRate*ratio*update;}}
";

    public static string ConvertToFp16 => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };layout(set=0,binding=1) writeonly buffer Output { uint outputData[]; };layout(push_constant) uniform Params { uint size; };void main(){uint word=gl_GlobalInvocationID.x;if(word>=(size+1u)/2u)return;uint first=word*2u;float second=first+1u<size?inputData[first+1u]:0.0;outputData[word]=packHalf2x16(vec2(inputData[first],second));}
";

    public static string ConvertToFp32 => Header + @"
layout(set=0,binding=0) readonly buffer Input { uint inputData[]; };layout(set=0,binding=1) writeonly buffer Output { float outputData[]; };layout(push_constant) uniform Params { uint size; };void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=size)return;vec2 pair=unpackHalf2x16(inputData[idx/2u]);outputData[idx]=(idx&1u)==0u?pair.x:pair.y;}
";

    public static string Sparse => Header + @"
layout(set=0,binding=0) buffer Param { float param[]; };
layout(set=0,binding=1) buffer State1 { float state1[]; };
layout(set=0,binding=2) buffer State2 { float state2[]; };
layout(set=0,binding=3) buffer State3 { float state3[]; };
layout(set=0,binding=4) readonly buffer Indices { float indices[]; };
layout(set=0,binding=5) readonly buffer Values { float values[]; };
layout(push_constant) uniform Params { uint size; uint nnz; uint operation; uint step; float learningRate; float p0; float p1; float p2; float p3; };
int decodeIndex(float raw){int bits=int(floatBitsToUint(raw));if(bits>=0&&uint(bits)<size)return bits;int numeric=int(raw);return numeric>=0&&uint(numeric)<size&&raw==float(numeric)?numeric:-1;}
void main(){if(gl_GlobalInvocationID.x!=0u)return;float bc1=1.0-pow(p0,float(step)),bc2=1.0-pow(p1,float(step));
    for(uint k=0u;k<nnz;++k){int decoded=decodeIndex(indices[k]);if(decoded<0)continue;uint i=uint(decoded);float g=values[k];
        if(operation==0u){if(p0>0.0)g+=p0*param[i];param[i]-=learningRate*g;}
        else if(operation==1u){if(p1>0.0)g+=p1*param[i];float value=p0*state1[i]+g;state1[i]=value;param[i]-=learningRate*value;}
        else if(operation==2u||operation==3u){float original=param[i];if(operation==3u&&p3>0.0)original-=learningRate*p3*original;float m=p0*state1[i]+(1.0-p0)*g,v=p1*state2[i]+(1.0-p1)*g*g;state1[i]=m;state2[i]=v;float update=learningRate*(m/bc1)/(sqrt(v/bc2)+p2);if(operation==2u&&p3>0.0)update+=learningRate*p3*param[i];param[i]=original-update;}
        else if(operation==4u){if(p2>0.0)g+=p2*param[i];float sq=p0*state1[i]+(1.0-p0)*g*g;state1[i]=sq;param[i]-=learningRate*g/(sqrt(sq)+p1);}
        else if(operation==5u){if(p1>0.0)g+=p1*param[i];float accum=state1[i]+g*g;state1[i]=accum;param[i]-=learningRate*g/(sqrt(accum)+p0);}
        else if(operation==6u){if(p1>0.0)g+=p1*param[i];float old=state1[i],value=p0*old+g;state1[i]=value;param[i]-=learningRate*((1.0+p0)*value-p0*old);}
        else if(operation==7u){if(p2>0.0)g+=p2*param[i];float ag=p0*state1[i]+(1.0-p0)*g*g;state1[i]=ag;float dx=sqrt(state2[i]+p1)/sqrt(ag+p1)*g;state2[i]=p0*state2[i]+(1.0-p0)*dx*dx;param[i]-=dx;}
        else if(operation==8u){if(p3>0.0)g+=p3*param[i];float m=p0*state1[i]+(1.0-p0)*g,v=p1*state2[i]+(1.0-p1)*g*g;state1[i]=m;state2[i]=v;state3[i]=max(state3[i],v);param[i]-=learningRate*(m/bc1)/(sqrt(state3[i]/bc2)+p2);}
        else if(operation==9u){if(p3>0.0)g+=p3*param[i];float m=p0*state1[i]+(1.0-p0)*g,u=max(p1*state2[i],abs(g));state1[i]=m;state2[i]=u;param[i]-=(learningRate/bc1)*m/(u+p2);}
        else if(operation==10u){float value=p0*state1[i]+(1.0-p0)*g,update=sign(value);if(p2>0.0)update+=p2*param[i];param[i]-=learningRate*update;state1[i]=p1*state1[i]+(1.0-p1)*g;}
        else if(operation==11u){if(p3>0.0)g+=p3*param[i];float m=p0*state1[i]+(1.0-p0)*g,v=p1*state2[i]+(1.0-p1)*g*g;state1[i]=m;state2[i]=v;float mh=(p0*m+(1.0-p0)*g)/bc1;param[i]-=learningRate*mh/(sqrt(v/bc2)+p2);}
        else if(operation==12u){float old=state2[i],current=old+g*g;state2[i]=current;float sigma=(sqrt(current)-sqrt(old))/learningRate;state1[i]+=g-sigma*param[i];float z=state1[i];param[i]=abs(z)<=p0?0.0:((z>0.0?1.0:-1.0)*p0-z)/((p2+sqrt(current))/learningRate+p1);}
        else {float value=param[i]-learningRate*g,threshold=learningRate*p0;param[i]=value>threshold?value-threshold:(value<(-threshold)?value+threshold:0.0);}
    }
}";
}
