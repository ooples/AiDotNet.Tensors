#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float frameGrad[]; };
layout(set=0, binding=1) buffer B1 { float window[]; };
layout(set=0, binding=2) buffer B2 { float gradSignal[]; };
layout(push_constant) uniform PC {
    int batch;
    int sigLen;
    int pad;
    int nFft;
    int hop;
    int numFrames;
};
#define P210_PI 3.14159265358979323846
float stftOverlap(int b, int q) {
    int fHi = q / hop; if (fHi > numFrames - 1) fHi = numFrames - 1;
    int fLo = q - nFft + 1 > 0 ? (q - nFft + hop) / hop : 0;
    float acc = 0.0;
    for (int f = fLo; f <= fHi; f++) { int i = q - f*hop; acc += frameGrad[(b*numFrames + f)*nFft + i] * window[i]; }
    return acc;
}
void main() {
    int idx = int(gl_GlobalInvocationID.x); if (idx >= batch*sigLen) return;
    int s = idx % sigLen; int b = idx / sigLen;
    float acc = stftOverlap(b, pad + s);
    if (s >= 1 && s <= pad) acc += stftOverlap(b, pad - s);
    if (s >= sigLen - 1 - pad && s <= sigLen - 2) acc += stftOverlap(b, 2*sigLen - 2 + pad - s);
    gradSignal[idx] = acc;
}
