// Copyright (c) AiDotNet. All rights reserved.
// Correctness-first resident Vulkan kernels for operations that do not yet have tuned SPIR-V.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanResidentKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    public static string RandomGenerate => Header + @"
layout(set=0,binding=0) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint count; uint seedLow; uint seedHigh; uint mode; uint minimumBits; uint maximumBits; uint meanBits; uint stdDevBits; };
uint mulHigh(uint a,uint b){uint al=a&65535u,ah=a>>16u,bl=b&65535u,bh=b>>16u;uint ll=al*bl,lh=al*bh,hl=ah*bl,hh=ah*bh,mid=(ll>>16u)+(lh&65535u)+(hl&65535u);return hh+(lh>>16u)+(hl>>16u)+(mid>>16u);}
uvec4 philoxRound(uint c0,uint c1,uint k0,uint k1){const uint m0=0xd2511f53u,m1=0xcd9e8d57u;uint lo0=c0*m0,hi0=mulHigh(c0,m0),lo1=c1*m1,hi1=mulHigh(c1,m1);return uvec4(hi1^k0,lo1,hi0^k1,lo0);}
uvec4 philox(uvec4 counter,uvec2 key){uvec4 c=counter;uvec2 k=key;for(uint i=0u;i<10u;++i){c=philoxRound(c.x,c.z,k.x,k.y);k.x+=0x9e3779b9u;k.y+=0xbb67ae85u;}return c;}
float uniform01(uint value){return float(value>>8u)*(1.0/16777216.0);}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=count)return;uvec2 key=uvec2(seedLow,seedHigh);
    if(mode==0u){uint seed=seedLow^seedHigh;uint state=idx*747796405u+seed+2891336453u;uint word=((state>>((state>>28u)+4u))^state)*277803737u;uint sample=(word>>22u)^word;float u=float(sample>>8u)*(1.0/16777216.0);float minimum=uintBitsToFloat(minimumBits),maximum=uintBitsToFloat(maximumBits);outputData[idx]=fma(u,maximum-minimum,minimum);return;}
    uint pairIndex=idx/2u,pairBlock=pairIndex/2u,pairLane=pairIndex%2u;uvec4 randomPair=philox(uvec4(pairBlock,1u,0u,0u),key);float u1=max(uniform01(randomPair[pairLane*2u]),1.0e-10),u2=uniform01(randomPair[pairLane*2u+1u]);float magnitude=uintBitsToFloat(stdDevBits)*sqrt(-2.0*log(u1)),angle=6.283185307*u2,mean=uintBitsToFloat(meanBits);outputData[idx]=mean+magnitude*((idx&1u)==0u?cos(angle):sin(angle));
}";

    public static string StatelessDropoutMask => Header + @"
layout(set=0,binding=0) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint count; uint threshold; uint scaleBits; uint seed; };
void main(){
    uint idx=gl_GlobalInvocationID.x;if(idx>=count)return;
    uint state=idx*747796405u+seed+2891336453u;
    uint word=((state>>((state>>28u)+4u))^state)*277803737u;
    uint sample=(word>>22u)^word;
    outputData[idx]=sample<threshold?0.0:uintBitsToFloat(scaleBits);
}";

    public static string Enforce2x4 => Header + @"
layout(set=0,binding=0) readonly buffer Dense { float dense[]; };
layout(set=0,binding=1) writeonly buffer Values { float values[]; };
layout(set=0,binding=2) writeonly buffer Indices { uint indices[]; };
layout(push_constant) uniform Params { uint M; uint K; uint groups; };
void main(){
    uint word=gl_GlobalInvocationID.x,first=word*4u;if(first>=groups)return;uint packed=0u;
    for(uint slot=0u;slot<4u;++slot){uint group=first+slot;if(group>=groups)break;uint row=group/(K/4u),block=group%(K/4u),base=(row*K)+(block*4u);uint i0=0u,i1=1u;
        if(abs(dense[base+i1])>abs(dense[base+i0])){uint t=i0;i0=i1;i1=t;}
        for(uint j=2u;j<4u;++j){float a=abs(dense[base+j]);if(a>abs(dense[base+i0])){i1=i0;i0=j;}else if(a>abs(dense[base+i1]))i1=j;}
        values[group*2u]=dense[base+i0];values[group*2u+1u]=dense[base+i1];packed|=(i0|(i1<<2u))<<(slot*8u);
    }indices[word]=packed;
}";

    public static string Decompress2x4 => Header + @"
layout(set=0,binding=0) readonly buffer Values { float values[]; };
layout(set=0,binding=1) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=2) writeonly buffer Dense { float dense[]; };
layout(push_constant) uniform Params { uint M; uint K; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=M*K)return;uint group=idx/4u,local=idx%4u,byteValue=(indices[group/4u]>>((group%4u)*8u))&255u;uint i0=byteValue&3u,i1=(byteValue>>2u)&3u;dense[idx]=local==i0?values[group*2u]:(local==i1?values[group*2u+1u]:0.0);}";

    public static string ScatterAddEdges => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer Sources { float sources[]; };
layout(set=0,binding=2) readonly buffer Targets { float targets[]; };
layout(set=0,binding=3) readonly buffer EdgeValues { float edgeValues[]; };
layout(set=0,binding=4) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint nodes; uint edges; uint features; uint weighted; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=nodes*features)return;uint node=idx/features,feature=idx%features;float sum=inputData[idx];for(uint e=0u;e<edges;++e){uint target=floatBitsToUint(targets[e]);if(target==node){uint source=floatBitsToUint(sources[e]);if(source<nodes)sum+=(weighted!=0u?edgeValues[e]:1.0)*inputData[source*features+feature];}}outputData[idx]=sum;}";

    public static string MaxPool2DBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint channels; uint inHeight; uint inWidth; uint outHeight; uint outWidth; };
void main(){uint idx=gl_GlobalInvocationID.x,inputSpatial=inHeight*inWidth,total=batch*channels*inputSpatial;if(idx>=total)return;uint bc=idx/inputSpatial,local=idx%inputSpatial,outputOffset=bc*outHeight*outWidth;float sum=0.0;for(uint oh=0u;oh<outHeight;++oh)for(uint ow=0u;ow<outWidth;++ow){uint outIndex=outputOffset+oh*outWidth+ow;if(indices[outIndex]==int(local))sum+=gradOutput[outIndex];}gradInput[idx]=sum;}";

    public static string MaxPool3DBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint channels; uint inDepth; uint inHeight; uint inWidth; uint outDepth; uint outHeight; uint outWidth; };
void main(){uint idx=gl_GlobalInvocationID.x,inputSpatial=inDepth*inHeight*inWidth,total=batch*channels*inputSpatial;if(idx>=total)return;uint bc=idx/inputSpatial,local=idx%inputSpatial,outputSpatial=outDepth*outHeight*outWidth,outputOffset=bc*outputSpatial;float sum=0.0;for(uint od=0u;od<outDepth;++od)for(uint oh=0u;oh<outHeight;++oh)for(uint ow=0u;ow<outWidth;++ow){uint outIndex=outputOffset+(od*outHeight+oh)*outWidth+ow;if(indices[outIndex]==int(local))sum+=gradOutput[outIndex];}gradInput[idx]=sum;}";

    public static string CsrSegmented => Header + @"
layout(set=0,binding=0) readonly buffer Columns { float columns[]; };
layout(set=0,binding=1) readonly buffer Rows { float rows[]; };
layout(set=0,binding=2) readonly buffer Input { float inputData[]; };
layout(set=0,binding=3) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint M; uint K; uint N; uint op; float epsilon; };
void main(){uint outIndex=gl_GlobalInvocationID.x;if(outIndex>=M*N)return;uint row=outIndex/N,feature=outIndex%N,start=floatBitsToUint(rows[row]),end=floatBitsToUint(rows[row+1u]);if(start>=end){outputData[outIndex]=0.0;return;}if(op<2u){float value=op==0u?-3.402823466e38:3.402823466e38;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K)value=op==0u?max(value,inputData[col*N+feature]):min(value,inputData[col*N+feature]);}outputData[outIndex]=value;return;}float mean=0.0;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K)mean+=inputData[col*N+feature];}mean/=float(end-start);float variance=0.0;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K){float d=inputData[col*N+feature]-mean;variance+=d*d;}}outputData[outIndex]=sqrt(variance/float(end-start)+epsilon);}";

    public static string RbfForward => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer Centers { float centers[]; };
layout(set=0,binding=2) readonly buffer Epsilons { float epsilons[]; };
layout(set=0,binding=3) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint batch; uint centerCount; uint inputDim; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*centerCount)return;uint b=idx/centerCount,c=idx%centerCount;float distance=0.0;for(uint d=0u;d<inputDim;++d){float x=inputData[b*inputDim+d]-centers[c*inputDim+d];distance+=x*x;}outputData[idx]=exp(-epsilons[c]*distance);}";

    public static string StdpUpdate => Header + @"
layout(set=0,binding=0) buffer Weights { float weights[]; };
layout(set=0,binding=1) readonly buffer PreTrace { float preTrace[]; };
layout(set=0,binding=2) readonly buffer PostTrace { float postTrace[]; };
layout(set=0,binding=3) readonly buffer PreSpike { float preSpike[]; };
layout(set=0,binding=4) readonly buffer PostSpike { float postSpike[]; };
layout(push_constant) uniform Params { uint numPre; uint numPost; float ltp; float ltd; float minWeight; float maxWeight; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=numPre*numPost)return;uint i=idx/numPost,j=idx%numPost;float value=weights[idx];if(postSpike[j]>0.5)value+=ltp*preTrace[i];if(preSpike[i]>0.5)value-=ltd*postTrace[j];weights[idx]=clamp(value,minWeight,maxWeight);}";

    public static string UpdateTraces => Header + @"
layout(set=0,binding=0) buffer Traces { float traces[]; };
layout(set=0,binding=1) readonly buffer Spikes { float spikes[]; };
layout(push_constant) uniform Params { uint size; float decay; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size)traces[idx]=decay*traces[idx]+(spikes[idx]>0.5?1.0:0.0);}";

    public static string HyperbolicBackwardInput => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint inputFeatures; uint outputFeatures; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*inputFeatures)return;uint b=idx/inputFeatures,i=idx%inputFeatures;float sum=0.0;for(uint o=0u;o<outputFeatures;++o)sum+=gradOutput[b*outputFeatures+o]*weights[o*inputFeatures+i];gradInput[idx]=sum;}";

    public static string HyperbolicBackwardWeights => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Input { float inputData[]; };
layout(set=0,binding=2) writeonly buffer GradWeights { float gradWeights[]; };
layout(push_constant) uniform Params { uint batch; uint inputFeatures; uint outputFeatures; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=outputFeatures*inputFeatures)return;uint o=idx/inputFeatures,i=idx%inputFeatures;float sum=0.0;for(uint b=0u;b<batch;++b)sum+=gradOutput[b*outputFeatures+o]*inputData[b*inputFeatures+i];gradWeights[idx]=sum;}";

    public static string HyperbolicBackwardBiases => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) writeonly buffer GradBiases { float gradBiases[]; };
layout(push_constant) uniform Params { uint batch; uint outputFeatures; };
void main(){uint o=gl_GlobalInvocationID.x;if(o>=outputFeatures)return;float sum=0.0;for(uint b=0u;b<batch;++b)sum+=gradOutput[b*outputFeatures+o];gradBiases[o]=sum;}";

    public static string QuantumMeasurement => Header + @"
layout(set=0,binding=0) readonly buffer Real { float realData[]; };
layout(set=0,binding=1) readonly buffer Imag { float imagData[]; };
layout(set=0,binding=2) writeonly buffer Probabilities { float probabilities[]; };
layout(push_constant) uniform Params { uint size; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size)probabilities[idx]=realData[idx]*realData[idx]+imagData[idx]*imagData[idx];}";

    public static string NormalizeProbabilities => Header + @"
layout(set=0,binding=0) buffer Probabilities { float probabilities[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; };
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint offset=b*stateSize;float sum=0.0;for(uint s=0u;s<stateSize;++s)sum+=probabilities[offset+s];if(sum>0.0)for(uint s=0u;s<stateSize;++s)probabilities[offset+s]/=sum;}";

    public static string ComplexMatVec => Header + @"
layout(set=0,binding=0) readonly buffer MatrixReal { float matrixReal[]; };
layout(set=0,binding=1) readonly buffer MatrixImag { float matrixImag[]; };
layout(set=0,binding=2) readonly buffer VectorReal { float vectorReal[]; };
layout(set=0,binding=3) readonly buffer VectorImag { float vectorImag[]; };
layout(set=0,binding=4) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=5) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint dim; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*dim)return;uint b=idx/dim,row=idx%dim,matrixOffset=b*dim*dim,vectorOffset=b*dim;float realSum=0.0,imagSum=0.0;for(uint j=0u;j<dim;++j){uint m=matrixOffset+row*dim+j;realSum+=matrixReal[m]*vectorReal[vectorOffset+j]-matrixImag[m]*vectorImag[vectorOffset+j];imagSum+=matrixReal[m]*vectorImag[vectorOffset+j]+matrixImag[m]*vectorReal[vectorOffset+j];}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string QuantumRotation => Header + @"
layout(set=0,binding=0) readonly buffer StateReal { float stateReal[]; };
layout(set=0,binding=1) readonly buffer StateImag { float stateImag[]; };
layout(set=0,binding=2) readonly buffer Angles { float angles[]; };
layout(set=0,binding=3) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=4) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; uint angleCount; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*stateSize)return;uint b=idx/stateSize;float angle=angles[b%angleCount],c=cos(angle),s=sin(angle);outputReal[idx]=c*stateReal[idx]-s*stateImag[idx];outputImag[idx]=s*stateReal[idx]+c*stateImag[idx];}";

    public static string MeasurementForward => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; };
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint offset=b*stateSize;float sum=0.0;for(uint s=0u;s<stateSize;++s){float value=inputData[offset+s]*inputData[offset+s];outputData[offset+s]=value;sum+=value;}if(sum>0.0)for(uint s=0u;s<stateSize;++s)outputData[offset+s]/=sum;}";

    public static string SplitDft => Header + @"
layout(set=0,binding=0) readonly buffer InputReal { float inputReal[]; };
layout(set=0,binding=1) readonly buffer InputImag { float inputImag[]; };
layout(set=0,binding=2) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=3) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint n; uint inverse; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*n)return;uint b=idx/n,k=idx%n,offset=b*n;float realSum=0.0,imagSum=0.0,sign=inverse!=0u?1.0:-1.0;for(uint j=0u;j<n;++j){float angle=sign*6.283185307179586*float(k*j)/float(n),c=cos(angle),s=sin(angle);realSum+=inputReal[offset+j]*c-inputImag[offset+j]*s;imagSum+=inputReal[offset+j]*s+inputImag[offset+j]*c;}if(inverse!=0u){realSum/=float(n);imagSum/=float(n);}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string SplitDft2D => Header + @"
layout(set=0,binding=0) readonly buffer InputReal { float inputReal[]; };
layout(set=0,binding=1) readonly buffer InputImag { float inputImag[]; };
layout(set=0,binding=2) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=3) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint height; uint width; uint inverse; };
void main(){uint idx=gl_GlobalInvocationID.x,slice=height*width;if(idx>=batch*slice)return;uint b=idx/slice,local=idx%slice,ky=local/width,kx=local%width,offset=b*slice;float realSum=0.0,imagSum=0.0,sign=inverse!=0u?1.0:-1.0;for(uint y=0u;y<height;++y)for(uint x=0u;x<width;++x){float angle=sign*6.283185307179586*(float(ky*y)/float(height)+float(kx*x)/float(width)),c=cos(angle),s=sin(angle);uint source=offset+y*width+x;realSum+=inputReal[source]*c-inputImag[source]*s;imagSum+=inputReal[source]*s+inputImag[source]*c;}if(inverse!=0u){realSum/=float(slice);imagSum/=float(slice);}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string PolarToComplex => Header + @"
layout(set=0,binding=0) readonly buffer Magnitude { float magnitude[]; };
layout(set=0,binding=1) readonly buffer Phase { float phase[]; };
layout(set=0,binding=2) writeonly buffer Real { float realData[]; };
layout(set=0,binding=3) writeonly buffer Imag { float imagData[]; };
layout(push_constant) uniform Params { uint size; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size){realData[idx]=magnitude[idx]*cos(phase[idx]);imagData[idx]=magnitude[idx]*sin(phase[idx]);}}";

    public static string MelFilterbank => Header + @"
layout(set=0,binding=0) readonly buffer Power { float power[]; };
layout(set=0,binding=1) readonly buffer Filterbank { float filterbank[]; };
layout(set=0,binding=2) writeonly buffer Mel { float mel[]; };
layout(push_constant) uniform Params { uint frames; uint frequencies; uint bands; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=frames*bands)return;uint frame=idx/bands,band=idx%bands;float sum=0.0;for(uint k=0u;k<frequencies;++k)sum+=power[frame*frequencies+k]*filterbank[band*frequencies+k];mel[idx]=sum;}";

    public static string LstmForwardSequence => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=3) readonly buffer BiasIh { float biasIh[]; };
layout(set=0,binding=4) readonly buffer BiasHh { float biasHh[]; };
layout(set=0,binding=5) writeonly buffer Output { float outputData[]; };
layout(set=0,binding=6) buffer AllH { float allH[]; };
layout(set=0,binding=7) buffer AllC { float allC[]; };
layout(set=0,binding=8) writeonly buffer CacheGates { float cacheGates[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint gateSize=4u*hiddenSize,cellSize=batch*hiddenSize;
    for(uint t=0u;t<seqLen;++t){uint hOffset=t*cellSize,nextOffset=(t+1u)*cellSize,gateOffset=(t*batch+b)*gateSize,inputOffset=(t*batch+b)*inputSize;
        for(uint g=0u;g<gateSize;++g){float sum=biasIh[g]+biasHh[g];for(uint k=0u;k<inputSize;++k)sum+=inputData[inputOffset+k]*weightsIh[g*inputSize+k];for(uint k=0u;k<hiddenSize;++k)sum+=allH[hOffset+b*hiddenSize+k]*weightsHh[g*hiddenSize+k];cacheGates[gateOffset+g]=sum;}
        for(uint i=0u;i<hiddenSize;++i){float ig=sigmoid(cacheGates[gateOffset+i]),fg=sigmoid(cacheGates[gateOffset+hiddenSize+i]),gg=tanh(cacheGates[gateOffset+2u*hiddenSize+i]),og=sigmoid(cacheGates[gateOffset+3u*hiddenSize+i]);uint state=b*hiddenSize+i;float ct=fg*allC[hOffset+state]+ig*gg,ht=og*tanh(ct);allC[nextOffset+state]=ct;allH[nextOffset+state]=ht;outputData[t*cellSize+state]=ht;}
    }
}";

    public static string LstmBackwardSequence => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer AllH { float allH[]; };
layout(set=0,binding=2) readonly buffer AllC { float allC[]; };
layout(set=0,binding=3) readonly buffer CacheGates { float cacheGates[]; };
layout(set=0,binding=4) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=5) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=6) readonly buffer Input { float inputData[]; };
layout(set=0,binding=7) buffer GradInput { float gradInput[]; };
layout(set=0,binding=8) buffer GradHInit { float gradHInit[]; };
layout(set=0,binding=9) buffer GradCInit { float gradCInit[]; };
layout(set=0,binding=10) buffer GradWeightsIh { float gradWeightsIh[]; };
layout(set=0,binding=11) buffer GradWeightsHh { float gradWeightsHh[]; };
layout(set=0,binding=12) buffer GradBias { float gradBias[]; };
layout(set=0,binding=13) buffer NextDh { float nextDh[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){if(gl_GlobalInvocationID.x!=0u)return;uint gateSize=4u*hiddenSize,cellSize=batch*hiddenSize;
    for(uint i=0u;i<seqLen*batch*inputSize;++i)gradInput[i]=0.0;for(uint i=0u;i<cellSize;++i){gradHInit[i]=0.0;gradCInit[i]=0.0;nextDh[i]=0.0;}for(uint i=0u;i<gateSize*inputSize;++i)gradWeightsIh[i]=0.0;for(uint i=0u;i<gateSize*hiddenSize;++i)gradWeightsHh[i]=0.0;for(uint i=0u;i<gateSize;++i)gradBias[i]=0.0;
    for(int ti=int(seqLen)-1;ti>=0;--ti){uint t=uint(ti),hOffset=t*cellSize,nextOffset=(t+1u)*cellSize;
        for(uint b=0u;b<batch;++b){uint stateOffset=b*hiddenSize,gateOffset=(t*batch+b)*gateSize,inputOffset=(t*batch+b)*inputSize;
            for(uint i=0u;i<hiddenSize;++i)gradHInit[stateOffset+i]+=gradOutput[t*cellSize+stateOffset+i];
            for(uint i=0u;i<hiddenSize;++i){float ig=sigmoid(cacheGates[gateOffset+i]),fg=sigmoid(cacheGates[gateOffset+hiddenSize+i]),gg=tanh(cacheGates[gateOffset+2u*hiddenSize+i]),og=sigmoid(cacheGates[gateOffset+3u*hiddenSize+i]),ct=allC[nextOffset+stateOffset+i],tanhCt=tanh(ct),localDh=gradHInit[stateOffset+i];gradCInit[stateOffset+i]+=localDh*og*(1.0-tanhCt*tanhCt);float localDc=gradCInit[stateOffset+i],dIg=localDc*gg*ig*(1.0-ig),dFg=localDc*allC[hOffset+stateOffset+i]*fg*(1.0-fg),dGg=localDc*ig*(1.0-gg*gg),dOg=localDh*tanhCt*og*(1.0-og);
                for(uint gi=0u;gi<4u;++gi){float dg=gi==0u?dIg:(gi==1u?dFg:(gi==2u?dGg:dOg));uint gate=gi*hiddenSize+i;gradBias[gate]+=dg;for(uint k=0u;k<inputSize;++k)gradWeightsIh[gate*inputSize+k]+=dg*inputData[inputOffset+k];for(uint k=0u;k<hiddenSize;++k)gradWeightsHh[gate*hiddenSize+k]+=dg*allH[hOffset+stateOffset+k];}
                for(uint k=0u;k<inputSize;++k)for(uint gi=0u;gi<4u;++gi){float dg=gi==0u?dIg:(gi==1u?dFg:(gi==2u?dGg:dOg));gradInput[inputOffset+k]+=dg*weightsIh[(gi*hiddenSize+i)*inputSize+k];}gradCInit[stateOffset+i]*=fg;
            }
            for(uint k=0u;k<hiddenSize;++k)nextDh[stateOffset+k]=0.0;
            for(uint i=0u;i<hiddenSize;++i){float ig=sigmoid(cacheGates[gateOffset+i]),fg=sigmoid(cacheGates[gateOffset+hiddenSize+i]),gg=tanh(cacheGates[gateOffset+2u*hiddenSize+i]),og=sigmoid(cacheGates[gateOffset+3u*hiddenSize+i]),ct=allC[nextOffset+stateOffset+i],tanhCt=tanh(ct),localDh=gradHInit[stateOffset+i],localDc=gradCInit[stateOffset+i],dOg=localDh*tanhCt*og*(1.0-og),dIg=localDc*gg*ig*(1.0-ig),dFg=localDc*allC[hOffset+stateOffset+i]*fg*(1.0-fg),dGg=localDc*ig*(1.0-gg*gg);for(uint k=0u;k<hiddenSize;++k)nextDh[stateOffset+k]+=dIg*weightsHh[i*hiddenSize+k]+dFg*weightsHh[(hiddenSize+i)*hiddenSize+k]+dGg*weightsHh[(2u*hiddenSize+i)*hiddenSize+k]+dOg*weightsHh[(3u*hiddenSize+i)*hiddenSize+k];}
            for(uint k=0u;k<hiddenSize;++k)gradHInit[stateOffset+k]=nextDh[stateOffset+k];
        }
    }
}";

    public static string GruForwardSequence => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=3) readonly buffer BiasIh { float biasIh[]; };
layout(set=0,binding=4) readonly buffer BiasHh { float biasHh[]; };
layout(set=0,binding=5) writeonly buffer Output { float outputData[]; };
layout(set=0,binding=6) buffer AllH { float allH[]; };
layout(set=0,binding=7) writeonly buffer CacheGates { float cacheGates[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint gateSize=3u*hiddenSize,cellSize=batch*hiddenSize;
    for(uint t=0u;t<seqLen;++t){uint hOffset=t*cellSize,nextOffset=(t+1u)*cellSize,gateOffset=(t*batch+b)*gateSize,inputOffset=(t*batch+b)*inputSize;
        for(uint g=0u;g<2u*hiddenSize;++g){float sum=biasIh[g]+biasHh[g];for(uint k=0u;k<inputSize;++k)sum+=inputData[inputOffset+k]*weightsIh[g*inputSize+k];for(uint k=0u;k<hiddenSize;++k)sum+=allH[hOffset+b*hiddenSize+k]*weightsHh[g*hiddenSize+k];cacheGates[gateOffset+g]=sigmoid(sum);}
        for(uint i=0u;i<hiddenSize;++i){uint gate=2u*hiddenSize+i;float sum=biasIh[gate]+biasHh[gate];for(uint k=0u;k<inputSize;++k)sum+=inputData[inputOffset+k]*weightsIh[gate*inputSize+k];for(uint k=0u;k<hiddenSize;++k)sum+=cacheGates[gateOffset+k]*allH[hOffset+b*hiddenSize+k]*weightsHh[gate*hiddenSize+k];cacheGates[gateOffset+gate]=tanh(sum);}
        for(uint i=0u;i<hiddenSize;++i){float z=cacheGates[gateOffset+hiddenSize+i],n=cacheGates[gateOffset+2u*hiddenSize+i],previous=allH[hOffset+b*hiddenSize+i],value=(1.0-z)*n+z*previous;allH[nextOffset+b*hiddenSize+i]=value;outputData[t*cellSize+b*hiddenSize+i]=value;}
    }
}";

    public static string GruBackwardSequence => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer AllH { float allH[]; };
layout(set=0,binding=2) readonly buffer CacheGates { float cacheGates[]; };
layout(set=0,binding=3) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=4) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=5) readonly buffer Input { float inputData[]; };
layout(set=0,binding=6) buffer GradInput { float gradInput[]; };
layout(set=0,binding=7) buffer GradHInit { float gradHInit[]; };
layout(set=0,binding=8) buffer GradWeightsIh { float gradWeightsIh[]; };
layout(set=0,binding=9) buffer GradWeightsHh { float gradWeightsHh[]; };
layout(set=0,binding=10) buffer GradBias { float gradBias[]; };
layout(set=0,binding=11) buffer NextDh { float nextDh[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){if(gl_GlobalInvocationID.x!=0u)return;uint gateSize=3u*hiddenSize,cellSize=batch*hiddenSize;
    for(uint i=0u;i<seqLen*batch*inputSize;++i)gradInput[i]=0.0;for(uint i=0u;i<cellSize;++i){gradHInit[i]=0.0;nextDh[i]=0.0;}for(uint i=0u;i<gateSize*inputSize;++i)gradWeightsIh[i]=0.0;for(uint i=0u;i<gateSize*hiddenSize;++i)gradWeightsHh[i]=0.0;for(uint i=0u;i<gateSize;++i)gradBias[i]=0.0;
    for(int ti=int(seqLen)-1;ti>=0;--ti){uint t=uint(ti),hOffset=t*cellSize;
        for(uint b=0u;b<batch;++b){uint stateOffset=b*hiddenSize,gateOffset=(t*batch+b)*gateSize,inputOffset=(t*batch+b)*inputSize;for(uint i=0u;i<hiddenSize;++i)gradHInit[stateOffset+i]+=gradOutput[t*cellSize+stateOffset+i];for(uint i=0u;i<hiddenSize;++i){float z=cacheGates[gateOffset+hiddenSize+i];nextDh[stateOffset+i]=gradHInit[stateOffset+i]*z;}
            for(uint i=0u;i<hiddenSize;++i){float r=cacheGates[gateOffset+i],z=cacheGates[gateOffset+hiddenSize+i],n=cacheGates[gateOffset+2u*hiddenSize+i],previous=allH[hOffset+stateOffset+i],localDh=gradHInit[stateOffset+i],dN=localDh*(1.0-z)*(1.0-n*n),dZ=localDh*(previous-n)*z*(1.0-z),dRPre=0.0;for(uint j=0u;j<hiddenSize;++j){float zj=cacheGates[gateOffset+hiddenSize+j],nj=cacheGates[gateOffset+2u*hiddenSize+j],dNj=gradHInit[stateOffset+j]*(1.0-zj)*(1.0-nj*nj);dRPre+=dNj*weightsHh[(2u*hiddenSize+j)*hiddenSize+i]*previous;}float dR=dRPre*r*(1.0-r);
                for(uint gi=0u;gi<3u;++gi){float dg=gi==0u?dR:(gi==1u?dZ:dN);uint gate=gi*hiddenSize+i;gradBias[gate]+=dg;for(uint k=0u;k<inputSize;++k)gradWeightsIh[gate*inputSize+k]+=dg*inputData[inputOffset+k];for(uint k=0u;k<hiddenSize;++k)gradWeightsHh[gate*hiddenSize+k]+=dg*allH[hOffset+stateOffset+k];}
                for(uint k=0u;k<inputSize;++k)gradInput[inputOffset+k]+=dR*weightsIh[i*inputSize+k]+dZ*weightsIh[(hiddenSize+i)*inputSize+k]+dN*weightsIh[(2u*hiddenSize+i)*inputSize+k];for(uint k=0u;k<hiddenSize;++k)nextDh[stateOffset+k]+=dR*weightsHh[i*hiddenSize+k]+dZ*weightsHh[(hiddenSize+i)*hiddenSize+k]+dN*weightsHh[(2u*hiddenSize+i)*hiddenSize+k];
            }for(uint k=0u;k<hiddenSize;++k)gradHInit[stateOffset+k]=nextDh[stateOffset+k];
        }
    }
}";

    public static string GruCellBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradH { float gradH[]; };
layout(set=0,binding=1) readonly buffer GateR { float gateR[]; };
layout(set=0,binding=2) readonly buffer GateZ { float gateZ[]; };
layout(set=0,binding=3) readonly buffer GateN { float gateN[]; };
layout(set=0,binding=4) readonly buffer PrevH { float prevH[]; };
layout(set=0,binding=5) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=6) writeonly buffer GradPrevH { float gradPrevH[]; };
layout(set=0,binding=7) writeonly buffer GradGateR { float gradGateR[]; };
layout(set=0,binding=8) writeonly buffer GradGateZ { float gradGateZ[]; };
layout(set=0,binding=9) writeonly buffer GradGateN { float gradGateN[]; };
layout(push_constant) uniform Params { uint batch; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint offset=(idx/hiddenSize)*hiddenSize,i=idx%hiddenSize;float r=gateR[idx],z=gateZ[idx],n=gateN[idx],previous=prevH[idx],localDh=gradH[idx],dRPre=0.0;for(uint j=0u;j<hiddenSize;++j){float dN=gradH[offset+j]*(1.0-gateZ[offset+j])*(1.0-gateN[offset+j]*gateN[offset+j]);dRPre+=dN*weightsHh[(2u*hiddenSize+j)*hiddenSize+i]*previous;}gradGateZ[idx]=localDh*(previous-n)*z*(1.0-z);gradGateN[idx]=localDh*(1.0-z)*(1.0-n*n);gradGateR[idx]=dRPre*r*(1.0-r);gradPrevH[idx]=localDh*z;}";
}
