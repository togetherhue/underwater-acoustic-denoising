#!/usr/bin/env bash
# α 轴的域内重训：把 8.1b「尾形 vs 强度」的分解从零样本升级为域内
# ============================================================================
# 【为什么需要】
# 8.1b 用等有效功率 SNR 的匹配条件（m_a12 / m_a18）把 α 与噪声强度解耦，
# 得到「α 越小分支收益越小」且「尾形解释 65%、强度解释 35%」。
# 但那仍是【零样本】—— 模型只在 base(α=1.5) 上训练过，所以那两个数里
# 仍混着"模型没见过该 α"的分布偏移成分。
#
# 本脚本在 m_a12 / m_a18 上从头重训 R4/R5，与 base(α=1.5) 做同设备、同配置对比，
# 回答：α 轴的收益差异是【尾形本身】造成的，还是【分布偏移】造成的？
#
# 预期：
#   若域内 Δ(α=1.2) 回到 base 的 ~0.79  -> 之前的压低是分布偏移
#   若域内 Δ(α=1.2) 仍接近零样本的 0.41 -> 尾形效应是真的
#
# 【设备】全部在 GPU（torch 2.14.1+cu126）。设备效应对 Δ 已单独测得
#   −0.091 ± 0.031 dB（base 条件 CPU vs GPU 配对），远小于待测差异。
#
# 【断点续跑】summary_sens_<cond>_<cfg>[_sN].csv 存在则跳过。
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_alpha.log"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

run_one () {   # $1=tag $2=pulse $3=seed $4=data $5=描述
  local tag="$1" pulse="$2" seed="$3" data="$4" desc="$5"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"; return 0
  fi
  echo "---- $desc  seed=$seed  tag=$tag ----" >> "$LOG"
  local extra=""
  [ "$pulse" = "1" ] && extra="--pulse-freq 1"
  "$PY" "$SRC/train_pamnet.py" --tag "$tag" --data "$data" \
      --norm gn --short 1 --long 1 --gate 1 --pulse "$pulse" \
      --freq-refine 2.0 $extra --seed "$seed" --outdir "$OUT" >> "$LOG" 2>&1
  echo "[完成] $tag  returncode=$?  $(date +%H:%M:%S)" | tee -a "$LOG"
}

echo "==== α 轴域内重训开始 $(date) ====" >> "$LOG"
for COND in a12 a18; do
  case $COND in
    a12) DATA="E:/project/data/ds_sens_matched/m_a12";;
    a18) DATA="E:/project/data/ds_sens_matched/m_a18";;
  esac
  for SEED in 20261013 2 3; do
    case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
    run_one "sens_${COND}_nopulse${SUF}" 0 "$SEED" "$DATA" "$COND R4 (无脉冲)"
    run_one "sens_${COND}_pf${SUF}"      1 "$SEED" "$DATA" "$COND R5 (p注入)"
  done
done
echo "==== α 轴域内重训结束 $(date) ====" >> "$LOG"
