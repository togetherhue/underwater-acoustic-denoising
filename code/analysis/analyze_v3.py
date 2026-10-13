# -*- coding: utf-8 -*-
"""分析 2x2 泛化实验 + 消融结果"""
import os, glob, sys
import numpy as np
import scipy.io as sio

OUT = r"D:/project/output_v3"
DS  = r"D:/project/dataset_v3"

TESTS = [("testA", "testA_seench_newchirp", "新波形 + 见过的信道"),
         ("testB", "testB_newch_seenchirp", "见过的波形 + 新信道"),
         ("testC", "testC_newch_newchirp",  "新波形 + 新信道")]


def load(tag, key):
    p = os.path.join(OUT, "result_%s_%s.csv" % (tag, key))
    return np.loadtxt(p, delimiter=",", skiprows=1) if os.path.exists(p) else None


def meta_of(split):
    mt = sio.loadmat(os.path.join(DS, "meta_%s.mat" % split),
                     squeeze_me=True, struct_as_record=False)["meta"]
    return (np.array([mt[k].chirp_id for k in range(mt.size)]),
            np.array([mt[k].chan_id  for k in range(mt.size)]),
            np.array([mt[k].snr      for k in range(mt.size)]))


def main():
    tags = [t for t in ("full", "timeonly", "freqonly", "big")
            if os.path.exists(os.path.join(OUT, "summary_%s.csv" % t))]
    print("找到的结果 tag:", tags)

    # ---------- 1. 2x2 泛化表 ----------
    print("\n" + "=" * 84)
    print("2x2 泛化实验（每个测试集 5 档 SNR 均衡）")
    print("=" * 84)
    print("%-26s %7s %9s %9s %9s %9s" % ("测试集", "样本", "输入SNR", "输出SNR", "SNR提升", "SI-SDR提升"))
    print("-" * 84)
    for key, split, desc in TESTS:
        r = load("full", key)
        if r is None:
            print("%-26s (缺结果)" % desc); continue
        print("%-26s %7d %9.2f %9.2f %+9.2f %+9.2f" % (
            desc, len(r), r[:, 0].mean(), r[:, 1].mean(),
            (r[:, 1]-r[:, 0]).mean(), (r[:, 3]-r[:, 2]).mean()))
    print("-" * 84)
    print("对照：v2 模型在旧测试集（新波形+见信道，25 个信道）上 +13.66 dB / SI-SDR +13.62")
    print("      注意 testA 的训练集只有 18 个信道（v2 是 25 个），训练样本 7560（v2 是 7000）")

    # ---------- 2. 消融 ----------
    if len(tags) > 1:
        print("\n" + "=" * 84)
        print("消融实验")
        print("=" * 84)
        print("%-14s %10s %10s %10s %10s %10s" % ("变体", "参数量", "testA", "testB", "testC", "均值"))
        print("-" * 84)
        names = {"full": "时域+频域", "timeonly": "仅时域", "freqonly": "仅频域", "big": "加大模型"}
        # 参数量从日志里读不到，用固定值标注
        pcount = {"full": "57,657", "timeonly": "?", "freqonly": "?", "big": "?"}
        for t in tags:
            row = []
            for key, _, _ in TESTS:
                r = load(t, key)
                row.append((r[:, 1]-r[:, 0]).mean() if r is not None else float("nan"))
            print("%-14s %10s %10.2f %10.2f %10.2f %10.2f" % (
                names.get(t, t), pcount.get(t, "?"), row[0], row[1], row[2], np.nanmean(row)))

    # ---------- 3. 分组明细（full 模型）----------
    for key, split, desc in TESTS:
        r = load("full", key)
        if r is None: continue
        ch, ca, sn = meta_of(split)
        assert len(ch) == len(r), "%s 长度不一致 %d vs %d" % (key, len(ch), len(r))
        gain = r[:, 1] - r[:, 0]
        print("\n" + "=" * 84)
        print("[%s]  逐因素明细" % desc)
        print("=" * 84)
        print("  按真实 SNR 档:")
        print("    %-8s %5s %9s %9s %9s" % ("档位", "n", "输入", "输出", "提升"))
        for s in sorted(set(sn.tolist())):
            m = sn == s
            print("    %-8d %5d %9.2f %9.2f %+9.2f" % (
                s, m.sum(), r[m, 0].mean(), r[m, 1].mean(), gain[m].mean()))
        print("  按 chirp:")
        for c in sorted(set(ch.tolist())):
            m = ch == c
            print("    chirp %-3d %5d %9.2f %9.2f %+9.2f" % (
                c, m.sum(), r[m, 0].mean(), r[m, 1].mean(), gain[m].mean()))
        print("  按信道（前 6 / 后 6，按增益排序）:")
        cs = sorted(set(ca.tolist()))
        gg = np.array([gain[ca == c].mean() for c in cs])
        o = np.argsort(gg)
        for i in list(o[:3]) + list(o[-3:][::-1]):
            print("    chan %-3d %5d %9.2f %9.2f %+9.2f" % (
                cs[i], (ca == cs[i]).sum(), r[ca == cs[i], 0].mean(),
                r[ca == cs[i], 1].mean(), gg[i]))
        print("  分位: 最差 %+.2f  5%% %+.2f  中位 %+.2f  95%% %+.2f  最好 %+.2f" % (
            np.percentile(gain, 0), np.percentile(gain, 5), np.percentile(gain, 50),
            np.percentile(gain, 95), np.percentile(gain, 100)))
        print("  变差样本数: %d / %d" % ((gain < 0).sum(), len(gain)))


if __name__ == "__main__":
    main()
