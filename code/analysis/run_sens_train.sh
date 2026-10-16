#!/usr/bin/env bash
# 适用边界实验：域内重训（在异常条件下重新训练 R4/R5，区分"分布偏移"与"方法本身优劣"）
# 条件 a15_r10：α=1.5（与训练一致），突发到达率 10/s（训练条件的 2 倍）
#   —— 这是对"脉冲感知分支是否有价值"最干净的单因素操纵：背景噪声完全不变，
#      只有干扰密度翻倍。若冗余假设成立，分支的收益应随干扰变密而增大。
# 种子取 20261013 / 2 / 3，与基线前三个种子一致，便于按种子配对。
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="D:/project/ds_sens/a15_r10"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_r10.log"

export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12

echo "==== 域内重训 a15_r10 开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  echo "---- R4 (无脉冲) seed=$SEED ----" >> "$LOG"
  "$PY" "$SRC/train_pamnet.py" --tag "sens_r10_nopulse${SUF}" --data "$DATA" \
      --norm gn --short 1 --long 1 --gate 1 --pulse 0 --freq-refine 2.0 \
      --seed "$SEED" --outdir "$OUT" >> "$LOG" 2>&1
  echo "---- R5 (p注入) seed=$SEED ----" >> "$LOG"
  "$PY" "$SRC/train_pamnet.py" --tag "sens_r10_pf${SUF}" --data "$DATA" \
      --norm gn --short 1 --long 1 --gate 1 --pulse 1 --freq-refine 2.0 --pulse-freq 1 \
      --seed "$SEED" --outdir "$OUT" >> "$LOG" 2>&1
done
echo "==== 域内重训 a15_r10 结束 $(date) ====" >> "$LOG"
