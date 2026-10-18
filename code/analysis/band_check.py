"""测量 chirp / 信道冲激响应 / 含噪信号 各自的频带，判断是否匹配。纯 Python FFT。"""
import sys, os, math, cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def gv(p, key=None):
    d = {m['name']: list(m['data']) for m in load(p)}
    return d[key] if key else list(d.values())[0]


def fft(a):
    n = len(a)
    if n == 1:
        return a
    ev = fft(a[0::2])
    od = fft(a[1::2])
    T = [cmath.exp(-2j * math.pi * k / n) * od[k] for k in range(n // 2)]
    return [ev[k] + T[k] for k in range(n // 2)] + [ev[k] - T[k] for k in range(n // 2)]


def spectrum(v, N):
    """返回前 N/2+1 个频点的功率谱（归一化频率 k/N）"""
    x = [complex(t) for t in v[:N]]
    x += [0j] * (N - len(x))
    X = fft(x)
    return [abs(X[k]) ** 2 for k in range(N // 2 + 1)]


def band99(P, N):
    """99% 能量所在的归一化频带"""
    tot = sum(P)
    if tot <= 0:
        return 0, 0, 0
    acc = 0.0
    lo = None
    for k, p in enumerate(P):
        acc += p
        if lo is None and acc >= 0.005 * tot:
            lo = k
        if acc >= 0.995 * tot:
            hi = k
            break
    else:
        hi = len(P) - 1
    peak = max(range(len(P)), key=lambda k: P[k])
    return lo / N, hi / N, peak / N


def report(name, v, N):
    P = spectrum(v, N)
    lo, hi, pk = band99(P, N)
    print('  %-26s 长度=%6d  归一化频带(99%%能量) %.5f ~ %.5f  峰值 %.5f' % (name, len(v), lo, hi, pk))
    return lo, hi, pk


print('=' * 78)
print('1) 三个信号各自的归一化频带 (cycles/sample)')
print('=' * 78)
chirp = gv('D:/project/chirp_signal.mat', 'chirp_signal')
H = gv('E:/project/data/信道产生/data/channel_H_1.mat', 'channel_H')
noisy = gv('D:/project/te_signal/test_sample_1.mat', 'noisy_signal')

c_lo, c_hi, c_pk = report('chirp_signal', chirp, 16384)
h_lo, h_hi, h_pk = report('channel_H_1', H, 8192)
n_lo, n_hi, n_pk = report('test_sample_1 (含噪)', noisy, 32768)

print()
print('=' * 78)
print('2) 换算成 Hz：不同采样率假设下')
print('=' * 78)
print('%-10s %-22s %-22s %s' % ('假设 fs', 'chirp 频带(Hz)', '信道 H 频带(Hz)', '是否重叠'))
for fs in (9600, 12000, 24000, 48000):
    a0, a1 = c_lo * fs, c_hi * fs
    b0, b1 = h_lo * fs, h_hi * fs
    ov = max(0.0, min(a1, b1) - max(a0, b0))
    print('%-10d %-22s %-22s %s' % (fs, '%.0f ~ %.0f' % (a0, a1), '%.0f ~ %.0f' % (b0, b1),
                                    '重叠 %.0f Hz' % ov if ov > 0 else '**不重叠**'))

print()
print('=' * 78)
print('3) 教师原始代码的频率栅格反推')
print('=' * 78)
print('  PEsim_main_parfor.m: fl=798, fh=1500, fct=3, Len=8000')
print('  F(fl/fct:fh/fct) = F(%d:%d)  → 占 8000 点 DFT 的第 %d~%d 个 bin' % (798 // 3, 1500 // 3, 798 // 3, 1500 // 3))
print('  归一化频带 = %.5f ~ %.5f cycles/sample' % (266 / 8000, 500 / 8000))
print('  若对应 798~1500 Hz，则 fs = 798*8000/266 = %.0f Hz' % (798 * 8000 / 266))
print()
print('  实测 channel_H 频带 = %.5f ~ %.5f cycles/sample' % (h_lo, h_hi))
print('  实测 chirp      频带 = %.5f ~ %.5f cycles/sample' % (c_lo, c_hi))
print('  → 两者归一化频带是否重叠：%s' % ('是' if min(c_hi, h_hi) > max(c_lo, h_lo) else '否'))
