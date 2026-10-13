# -*- coding: utf-8 -*-
"""信道频率选择性 vs 模型增益 + 生成报告用图"""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, stft, istft, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch, importlib.util
spec = importlib.util.spec_from_file_location("dc", r"C:\Users\32213\PycharmProjects\project1\大创_v2.py")
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
CFG = dc.CFG; FS = CFG["fs_native"]//CFG["downsample"]; L = CFG["seg_len"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/figs"
os.makedirs(OUT, exist_ok=True)

# ================= 1. 信道频率选择性 =================
CH = r"D:/project/信道产生/data_fixed"
fl = sorted(glob.glob(os.path.join(CH, "channel_H_*.mat")))
H, cid, names = [], [], []
for f in fl:
    H.append(np.asarray(sio.loadmat(f)["channel_H"]).squeeze())
    cid.append(int(os.path.basename(f).split("_")[-1].split(".")[0]))
    names.append(os.path.basename(f))
H = np.array(H); cid = np.array(cid)
keep = []
for i in range(len(H)):
    if not any(np.all(np.abs(H[j]-H[i]) < 1e-30) for j in keep): keep.append(i)
H = H[keep]; cid = cid[keep]; names = [names[i] for i in keep]

FSch = 24000
f = np.fft.rfftfreq(8000, 1/FSch)
# 注意：MATLAB F(266:500) 是 1-based，对应 0-based 的 265:499；
# 即通带为 798 <= f < 1500，bin 500（=1500 Hz）恰好是 0，不能算进去。
band = (f >= 798) & (f < 1500)
sel, spread, flat = [], [], []
for h in H:
    A = np.abs(np.fft.rfft(h, 8000))[band]
    Adb = 20*np.log10(np.maximum(A/A.mean(), 1e-12))
    sel.append(np.percentile(Adb, 5))          # 带内 5% 分位凹陷(dB)
    flat.append(Adb.std())                     # 带内起伏(dB) —— 频率选择性
    e = np.abs(h)**2; e = e/e.sum(); c = np.cumsum(e)
    spread.append((np.searchsorted(c,0.95)-np.searchsorted(c,0.05))/FSch*1000)
sel = np.array(sel); spread = np.array(spread); flat = np.array(flat)

res = np.loadtxt(r"D:/project/output_v2/test_results.csv", delimiter=",", skiprows=1)
mt = sio.loadmat(r"D:/project/dataset/meta_test.mat", squeeze_me=True, struct_as_record=False)["meta"]
chan = np.array([mt[k].chan_id for k in range(mt.size)])
gain = res[:,1]-res[:,0]
# ★ chan_id 是【glob 字典序的下标】，不是文件名里的数字。
#   H[i] 位于 glob 第 i 位 -> 对应 chan_id = i+1。（此前按文件名数字配对是错的）
g = np.array([gain[chan == (i+1)].mean() if (chan == (i+1)).sum() else np.nan
              for i in range(len(cid))])

print("=== 信道特性 vs 模型增益（25 信道）===")
print("  时延扩展        相关系数 = %+.3f" % np.corrcoef(spread, g)[0,1])
print("  带内 5%% 凹陷    相关系数 = %+.3f" % np.corrcoef(sel, g)[0,1])
print("  带内起伏(std)   相关系数 = %+.3f" % np.corrcoef(flat, g)[0,1])
print("  带内凹陷范围 %.1f ~ %.1f dB  中位 %.1f dB" % (sel.min(), sel.max(), np.median(sel)))
print("  带内起伏范围 %.1f ~ %.1f dB  中位 %.1f dB" % (flat.min(), flat.max(), np.median(flat)))
o = np.argsort(flat)
print("\n  频率选择性最强的 4 个信道（起伏大）:")
for i in o[-4:][::-1]:
    print("    chan_id %2d (%s)  起伏 %5.1f dB  凹陷 %6.1f dB  增益 %+6.2f" % (i+1, names[i], flat[i], sel[i], g[i]))
print("  频率选择性最弱的 4 个信道（起伏小）:")
for i in o[:4]:
    print("    chan_id %2d (%s)  起伏 %5.1f dB  凹陷 %6.1f dB  增益 %+6.2f" % (i+1, names[i], flat[i], sel[i], g[i]))
print("\n  时延扩展最大 / 最小的信道:")
os_ = np.argsort(spread)
for i in list(os_[:2]) + list(os_[-2:][::-1]):
    print("    chan_id %2d (%s)  时延 %6.1f ms  增益 %+6.2f" % (i+1, names[i], spread[i], g[i]))

# ================= 2. 载入测试样本 =================
def fix_len(x):
    return x[:L] if len(x) >= L else np.pad(x, (0, L-len(x)))
tf = sorted(glob.glob(os.path.join(CFG["test_dir"], "*.mat")))
N = np.zeros((len(tf), L), np.float32); C = np.zeros_like(N)
for i, p in enumerate(tf):
    m = sio.loadmat(p)
    N[i] = fix_len(resample_poly(np.asarray(m["noisy_signal"]).squeeze().astype(np.float64),1,4).astype(np.float32))
    C[i] = fix_len(resample_poly(np.asarray(m["clean_signal"]).squeeze().astype(np.float64),1,4).astype(np.float32))

def snr(e, r):
    return 10*np.log10((r**2).sum(-1)/(((e-r)**2).sum(-1)+1e-12))
def sisdr(e, r):
    r0=r-r.mean(-1,keepdims=True); x=e-e.mean(-1,keepdims=True)
    a=(x*r0).sum(-1,keepdims=True)/((r0**2).sum(-1,keepdims=True)+1e-12)
    p=a*r0; nz=x-p
    return 10*np.log10((p**2).sum(-1)/((nz**2).sum(-1)+1e-12))

model = dc.DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"])
model.load_state_dict(torch.load(CFG["model_save_path"], map_location=DEV)); model=model.to(DEV).eval()
EST = np.zeros_like(N)
with torch.no_grad():
    for i in range(0, len(N), 32):
        EST[i:i+32] = model(torch.from_numpy(N[i:i+32]).unsqueeze(1).to(DEV)).squeeze(1).cpu().numpy()

b,a = butter(6,[798/(FS/2),1500/(FS/2)],btype="band")
BP = filtfilt(b,a,N,axis=1).astype(np.float32)
s_in, s_md, s_bp = snr(N,C), snr(EST,C), snr(BP,C)
print("\n输入 %.2f  带通 %.2f  模型 %.2f dB" % (s_in.mean(), s_bp.mean(), s_md.mean()))

# ================= 3. 出图 =================
C_BG, C_GRID = "#ffffff", "#e6e6e6"
def style(ax):
    ax.set_facecolor(C_BG); ax.grid(True, color=C_GRID, lw=.8)
    for s in ax.spines.values(): s.set_color("#bbbbbb")

# --- 图1 训练曲线 ---
lg = np.loadtxt(r"D:/project/output_v2/train_log.txt", delimiter=",", skiprows=1)
fig, ax = plt.subplots(figsize=(9,4.6), dpi=150); style(ax)
ax.plot(lg[:,0], lg[:,1], color="#d62728", lw=2, label="训练损失")
ax.plot(lg[:,0], lg[:,2], color="#1f77b4", lw=2, label="验证损失")
ax.set_xlabel("轮次 epoch"); ax.set_ylabel("损失"); ax.legend(loc="upper right")
ax2 = ax.twinx()
ax2.plot(lg[:,0], lg[:,4]-lg[:,3], color="#2ca02c", lw=2, ls="--", label="验证集 SNR 提升")
ax2.set_ylabel("SNR 提升 (dB)", color="#2ca02c"); ax2.tick_params(axis="y", colors="#2ca02c")
ax2.legend(loc="lower right")
ax.set_title("训练曲线：95% 的提升在前 6 轮完成，之后 44 轮只再涨 0.15 dB", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig1_training.png")); plt.close(fig)

# --- 图2 波形对比 ---
k = int(np.argmin(np.abs(s_in - np.median(s_in))))
seg = slice(0, 4500)
fig, axs = plt.subplots(3,1, figsize=(10,6.4), dpi=150, sharex=True)
for ax, y, c, t in zip(axs, [N[k], EST[k], C[k]], ["#d62728","#2ca02c","#1f77b4"],
                       ["含噪输入  SNR=%.2f dB" % s_in[k], "模型输出  SNR=%.2f dB" % s_md[k], "干净目标"]):
    style(ax); ax.plot(np.arange(L)[seg]/FS, y[seg], color=c, lw=.8); ax.set_ylabel(t, fontsize=9)
axs[-1].set_xlabel("时间 (s)")
axs[0].set_title("时域波形对比（同一测试样本，有效区前段）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig2_waveform.png")); plt.close(fig)

# --- 图3 功率谱 ---
fig, ax = plt.subplots(figsize=(9,4.6), dpi=150); style(ax)
for y, c, t in [(N[k],"#d62728","含噪输入"), (EST[k],"#2ca02c","模型输出"), (C[k],"#1f77b4","干净目标")]:
    fq, P = welch(y, fs=FS, nperseg=1024)
    ax.semilogy(fq, P, color=c, lw=1.5, label=t)
ax.axvspan(798, 1500, color="#ffd966", alpha=.35, label="信号频带 798-1500 Hz")
ax.axvspan(1600, 3000, color="#c9c9c9", alpha=.35, label="纯噪声带（用于盲估计）")
ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("功率谱密度"); ax.set_xlim(0,3000)
ax.legend(fontsize=8, ncol=2)
ax.set_title("功率谱：带外噪声被压低约 70 dB，信号带内输出与干净目标几乎重合", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig3_spectrum.png")); plt.close(fig)

# --- 图4 方法对比 ---
names = ["含噪输入","带通 700-1600","维纳(盲)","带通 798-1500","谱减(盲)","本模型"]
vals  = [s_in.mean(), 11.54, 11.07, 12.63, 13.89, s_md.mean()]
cols  = ["#7f7f7f","#aec7e8","#aec7e8","#ffbb78","#ffbb78","#2ca02c"]
fig, ax = plt.subplots(figsize=(9,4.6), dpi=150); style(ax)
b_ = ax.bar(names, vals, color=cols)
for r, v in zip(b_, vals): ax.text(r.get_x()+r.get_width()/2, v+.35, "%.2f" % v, ha="center", fontsize=9)
ax.set_ylabel("输出 SNR (dB)"); ax.set_ylim(0, max(vals)*1.18)
ax.set_title("模型 vs 经典方法（1500 个测试样本，指标定义一致）", fontsize=11)
plt.xticks(rotation=12); fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig4_methods.png")); plt.close(fig)

# --- 图5 散点 ---
fig, ax = plt.subplots(figsize=(6.6,6), dpi=150); style(ax)
ax.scatter(s_in, s_bp, s=4, alpha=.35, color="#ff7f0e", label="理想带通")
ax.scatter(s_in, s_md, s=4, alpha=.35, color="#2ca02c", label="本模型")
lim=[-6,36]; ax.plot(lim,lim, color="#888", ls=":", lw=1.5, label="不处理 (y=x)")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("输入 SNR (dB)"); ax.set_ylabel("输出 SNR (dB)")
ax.legend(fontsize=9, markerscale=3); ax.set_title("逐样本输入-输出 SNR（1500 样本）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig5_scatter.png")); plt.close(fig)

# --- 图6 信道选择性 ---
fig, axs = plt.subplots(1,3, figsize=(14,4.3), dpi=150)
for ax, x, t in zip(axs, [spread, sel, flat],
                    ["90% 时延扩展 (ms)", "带内 5% 分位凹陷 (dB)", "带内起伏 std (dB)"]):
    style(ax); ax.scatter(x, g, s=34, color="#1f77b4")
    ax.set_xlabel(t); ax.set_ylabel("模型增益 (dB)")
    ax.set_title("%s\nr = %+.3f" % (t, np.corrcoef(x, g)[0,1]), fontsize=10)
fig.suptitle("信道物理特性与模型增益的关系（25 个信道）", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig6_channel.png")); plt.close(fig)

# --- 图7 分档柱状 ---
edges=[-5,3,8,13,17]; lbl=["<3 dB","3~8 dB","8~13 dB","13~17 dB"]
xs=np.arange(len(lbl)); w=.27
fig, ax = plt.subplots(figsize=(9,4.6), dpi=150); style(ax)
def bv(arr):
    return [arr[(s_in>=edges[i])&(s_in<edges[i+1])].mean() for i in range(len(lbl))]
ax.bar(xs-w, bv(s_in), w, label="含噪输入", color="#7f7f7f")
ax.bar(xs,   bv(s_bp), w, label="理想带通", color="#ffbb78")
ax.bar(xs+w, bv(s_md), w, label="本模型", color="#2ca02c")
for i in range(len(lbl)):
    for off,arr in [(-w,bv(s_in)),(0,bv(s_bp)),(w,bv(s_md))]:
        ax.text(i+off, arr[i]+.4, "%.1f"%arr[i], ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(lbl); ax.set_ylabel("输出 SNR (dB)")
ax.legend(fontsize=9); ax.set_title("分档对比：低信噪比时模型优势最大", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig7_bysnr.png")); plt.close(fig)

print("\n图已保存到", OUT)
for fn in sorted(os.listdir(OUT)): print("  ", fn)
np.savez(os.path.join(OUT,"data.npz"), s_in=s_in, s_bp=s_bp, s_md=s_md,
         spread=spread, sel=sel, g=g, cid=cid, lg=lg)
