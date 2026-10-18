"""Numerical verification of the denoising dataset (pure python, no numpy)."""
import sys, os, math, cmath, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def getvar(path, name=None):
    for m in load(path):
        if name is None or m.get('name') == name:
            return m['name'], list(m['data'])
    return None, None


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(a):
    n = len(a)
    if n == 1:
        return a
    ev = fft(a[0::2])
    od = fft(a[1::2])
    T = [cmath.exp(-2j * math.pi * k / n) * od[k] for k in range(n // 2)]
    return [ev[k] + T[k] for k in range(n // 2)] + [ev[k] - T[k] for k in range(n // 2)]


def ifft(a):
    n = len(a)
    conj = [x.conjugate() for x in a]
    y = fft(conj)
    return [x.conjugate() / n for x in y]


def conv(x, h):
    n = next_pow2(len(x) + len(h) - 1)
    X = fft([complex(v) for v in x] + [0j] * (n - len(x)))
    H = fft([complex(v) for v in h] + [0j] * (n - len(h)))
    Y = [X[i] * H[i] for i in range(n)]
    y = ifft(Y)
    return [z.real for z in y[:len(x) + len(h) - 1]]


def rms(v):
    return math.sqrt(sum(t * t for t in v) / len(v))


def peakfreq(v, fs, lo, hi):
    """Coarse DFT scan to find dominant frequency in [lo,hi]."""
    best, bf = -1.0, 0.0
    f = lo
    while f <= hi:
        w = 2 * math.pi * f / fs
        re = sum(v[i] * math.cos(w * i) for i in range(0, len(v), 4))
        im = sum(v[i] * math.sin(w * i) for i in range(0, len(v), 4))
        p = re * re + im * im
        if p > best:
            best, bf = p, f
        f += 2.0
    return bf


print('#' * 78)
print('# 1) 训练集样本幅度分布 (May 批 chrip_with_noise*.mat)')
print('#' * 78)
files = sorted(glob.glob('D:/project/tr_signal(1)/tr_signal/chrip_with_noise*.mat'),
               key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
random.seed(0)
samp = random.sample(files, 60)
stds = []
for p in samp:
    n, d = getvar(p)
    stds.append(rms(d))
stds.sort()
print('  样本数=%d  长度=%d' % (len(samp), len(d)))
print('  std: min=%.3e  p25=%.3e  median=%.3e  p75=%.3e  max=%.3e' %
      (stds[0], stds[len(stds)//4], stds[len(stds)//2], stds[3*len(stds)//4], stds[-1]))
print('  max/min 动态范围 = %.1e  (即 %.0f 个数量级)' % (stds[-1]/max(stds[0],1e-300), math.log10(stds[-1]/max(stds[0],1e-300))))

print()
print('#' * 78)
print('# 2) 干净目标信号 vs 含噪信号')
print('#' * 78)
cn, cd_ = getvar('D:/project/chirp_signal.mat')
print('  chirp_signal.mat: 变量=%s 长度=%d  rms=%.4f  min=%.3f max=%.3f' %
      (cn, len(cd_), rms(cd_), min(cd_), max(cd_)))
fs = 9600
print('  在 fs=9600Hz 下时长 = %.4f s' % (len(cd_) / fs))
print('  主频搜索(200-2000Hz) = %.0f Hz' % peakfreq(cd_, fs, 200, 2000))

tn, td = getvar('D:/project/te_signal/test_sample_1.mat')
print('  te_signal/test_sample_1.mat: 变量=%s 长度=%d  rms=%.3e' % (tn, len(td), rms(td)))

print()
print('#' * 78)
print('# 3) 验证假设: noisy = conv(chirp, channel_H) + noise ?')
print('#' * 78)
hp = 'E:/project/data/信道产生/data/channel_H_1.mat'
hn, hd = getvar(hp)
print('  channel_H_1.mat: 变量=%s 长度=%d rms=%.3e max|.|=%.3e' % (hn, len(hd), rms(hd), max(abs(v) for v in hd)))
print('  卷积长度 = %d + %d - 1 = %d ;  test_sample 长度 = %d' % (len(cd_), len(hd), len(cd_) + len(hd) - 1, len(td)))
y = conv(cd_, hd)
print('  conv(chirp,H) rms = %.3e  ; test_sample rms = %.3e' % (rms(y), rms(td)))
# 归一化后相关
sy = rms(y)
st = rms(td)
if sy > 0:
    yn = [v / sy for v in y]
    tn2 = [v / st for v in td]
    m = min(len(yn), len(tn2))
    cc = sum(yn[i] * tn2[i] for i in range(m)) / m
    print('  归一化互相关(零延迟) = %.4f' % cc)
    # 找最佳延迟
    best, bl = -2, 0
    for lag in range(-200, 201, 1):
        s = 0.0
        c = 0
        for i in range(0, m, 7):
            j = i + lag
            if 0 <= j < m:
                s += yn[i] * tn2[j]
                c += 1
        if c and s / c > best:
            best, bl = s / c, lag
    print('  最佳延迟=%d 处的归一化相关 = %.4f' % (bl, best))

print()
print('#' * 78)
print('# 4) 各数据集变量名 / 长度 汇总')
print('#' * 78)
checks = [
    ('tr_signal(1)/tr_signal/chirp_with_noise_1.mat', 'noisy_signal'),
    ('tr_signal(1)/tr_signal/chrip_with_noise1000.mat', 'noisy_signal'),
    ('te_signal/test_sample_1.mat', 'noisy_signal'),
    ('finetune_signal/training_sample_147.mat', 'chirp_with_noise'),
]
for rel, expect in checks:
    p = 'D:/project/' + rel
    names = [m.get('name') for m in load(p)]
    ln = [len(m.get('data', ())) for m in load(p)]
    print('  %-46s 变量=%s 长度=%s  -> 代码读取["chirp_with_noise"] %s' %
          (rel.split('/')[-1], names, ln, 'OK' if expect in names else 'KeyError!'))
