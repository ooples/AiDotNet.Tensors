// STFT analysis adjoint (backward of IEngine.STFT): frames, waveform (overlap-add + reflection fold), window.
// Phase-gradient magnitude floor: BackwardFunctions.PhaseGradientMagnitudeFloor. DC and the even-length Nyquist bin
// are real for a real frame, so their phase is piecewise constant (0 or pi) and its gradient is exactly zero.
#define STFT_BWD_PHASE_FLOOR 1e-12f
extern "C" __global__ void parity210_stft_backward_frames(
    const float* __restrict__ grad, const float* __restrict__ magnitude, const float* __restrict__ phase,
    float* __restrict__ frameGrad, int batch, int nFft, int numFrames, int numFreqs, int phaseTarget)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*numFrames*nFft) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames;
    int specOff = b*numFreqs*numFrames + frame;
    // Twiddle e^(+2*pi*i*m/nFft) from the integer phase m = k*i mod nFft, quarter turns exact, as in the forward.
    float tw = 2.0f * (float)M_PI / (float)nFft; float acc = 0.0f; int m = 0;
    for (int k = 0; k < numFreqs; k++) {
        int bin = specOff + k*numFrames; float g = grad[bin]; float p = phase[bin];
        float cp = cosf(p); float sp = sinf(p); float sr; float si;
        if (phaseTarget != 0) { float mg = magnitude[bin]; float r = (k != 0 && 2 * k != nFft && mg > STFT_BWD_PHASE_FLOOR) ? g / mg : 0.0f; sr = -r*sp; si = r*cp; }
        else { sr = g*cp; si = g*sp; }
        float c; float s;
        if (m == 0) { c = 1.0f; s = 0.0f; }
        else if (2 * m == nFft) { c = -1.0f; s = 0.0f; }
        else if (4 * m == nFft) { c = 0.0f; s = 1.0f; }
        else if (4 * m == 3 * nFft) { c = 0.0f; s = -1.0f; }
        else { float a = tw * (float)(2 * m > nFft ? m - nFft : m); c = cosf(a); s = sinf(a); }
        acc += sr*c - si*s;
        m += i; if (m >= nFft) m -= nFft;
    }
    frameGrad[idx] = acc;
}
__device__ float parity210_stft_overlap(const float* __restrict__ frameGrad, const float* __restrict__ window,
    int b, int q, int nFft, int hop, int numFrames)
{
    int fHi = q / hop; if (fHi > numFrames - 1) fHi = numFrames - 1;
    int fLo = q - nFft + 1 > 0 ? (q - nFft + hop) / hop : 0;
    float acc = 0.0f;
    for (int f = fLo; f <= fHi; f++) { int i = q - f*hop; acc += frameGrad[(b*numFrames + f)*nFft + i] * window[i]; }
    return acc;
}
extern "C" __global__ void parity210_stft_backward_signal(
    const float* __restrict__ frameGrad, const float* __restrict__ window, float* __restrict__ gradSignal,
    int batch, int sigLen, int pad, int nFft, int hop, int numFrames)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*sigLen) return;
    int s = idx % sigLen; int b = idx / sigLen;
    float acc = parity210_stft_overlap(frameGrad, window, b, pad + s, nFft, hop, numFrames);
    if (s >= 1 && s <= pad) acc += parity210_stft_overlap(frameGrad, window, b, pad - s, nFft, hop, numFrames);
    if (s >= sigLen - 1 - pad && s <= sigLen - 2) acc += parity210_stft_overlap(frameGrad, window, b, 2*sigLen - 2 + pad - s, nFft, hop, numFrames);
    gradSignal[idx] = acc;
}
extern "C" __global__ void parity210_stft_backward_window(
    const float* __restrict__ frameGrad, const float* __restrict__ signal, float* __restrict__ gradWindow,
    int batch, int sigLen, int pad, int nFft, int hop, int numFrames)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= nFft) return;
    int i = idx; float acc = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int f = 0; f < numFrames; f++) {
            int j = f*hop + i;
            int src = j < pad ? pad - j : (j < pad + sigLen ? j - pad : sigLen - 2 - (j - pad - sigLen));
            acc += frameGrad[(b*numFrames + f)*nFft + i] * signal[b*sigLen + src];
        }
    }
    gradWindow[i] = acc;
}
