"""Evaluate the saved denoising outputs in D:/project/output."""
import sys, os, math, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def getvars(path):
    return {m['name']: list(m['data']) for m in load(path)}


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((v - ma) ** 2 for v in a) / n
    vb = sum((v - mb) ** 2 for v in b) / n
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n / math.sqrt(va * vb)


def rmse(a, b):
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)


def snr_db(sig, noise):
    ps = sum(v * v for v in sig) / len(sig)
    pn = sum(v * v for v in noise) / len(noise)
    if pn <= 0:
        return float('inf')
    return 10 * math.log10(ps / pn)


print('%-30s %8s %8s %8s %8s %8s %8s' % ('file', 'corr_N_C', 'corr_D_C', 'rms_N', 'rms_D', 'rms_C', 'RMSE_D_C'))
rows = []
for p in sorted(glob.glob('D:/project/output/final_denoised_signal_*.mat'),
                key=lambda x: int(''.join(c for c in os.path.basename(x) if c.isdigit()))):
    d = getvars(p)
    N, D, C = d['noisy_signal'], d['denoised_signal'], d['clean_signal']
    def rms(v):
        return math.sqrt(sum(t * t for t in v) / len(v))
    cnc, cdc = corr(N, C), corr(D, C)
    rows.append((os.path.basename(p), cnc, cdc, rms(N), rms(D), rms(C), rmse(D, C)))
    print('%-30s %8.4f %8.4f %8.3f %8.3f %8.3f %8.4f' % rows[-1])

print()
print('平均: corr(noisy,clean)=%.4f   corr(denoised,clean)=%.4f' %
      (sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)))

# 真实 SNR 评估 (以 clean 为参考)
print()
print('%-30s %10s %10s %10s' % ('file', 'SNR_N(dB)', 'SNR_D(dB)', 'gain(dB)'))
g = []
for p in sorted(glob.glob('D:/project/output/final_denoised_signal_*.mat'),
                key=lambda x: int(''.join(c for c in os.path.basename(x) if c.isdigit()))):
    d = getvars(p)
    N, D, C = d['noisy_signal'], d['denoised_signal'], d['clean_signal']
    sN = snr_db(C, [N[i] - C[i] for i in range(len(C))])
    sD = snr_db(C, [D[i] - C[i] for i in range(len(C))])
    g.append(sD - sN)
    print('%-30s %10.2f %10.2f %10.2f' % (os.path.basename(p), sN, sD, sD - sN))
print('平均真实SNR增益 = %.2f dB' % (sum(g) / len(g)))
print('对比: 代码自报平均增益 = %.2f dB' % 13.35)
