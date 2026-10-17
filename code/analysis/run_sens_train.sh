#!/usr/bin/env bash
# 适用边界实验：域内重训（在异常条件下重新训练 R4/R5，区分"分布偏移"与"方法本身优劣"）
# 条件 a15_r10：α=1.5（与训练一致），突发到达率 10/s（训练条件的 2 倍）
#   —— 这是对"脉冲感知分支是否有价值"最干净的单因素操纵：背景噪声完全不变，
#      只有干扰密度翻倍。若冗余假设成立，分支的收益应随干扰变密而增大。
# 种子取 20261013 / 2 / 3，与基线前三个种子一致，便于按种子配对。
#
# 【断点续跑】已存在 summary_sens_r10_<tag>.csv 的训练直接跳过。
#   本环境会回收裸 nohup 的进程，长任务也可能被中断，
#   因此每次重启都从断点继续，不重复已完成的工作。
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="D:/project/ds_sens/a15_r10"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_r10.log"

export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12

run_one () {   # $1=tag  $2=pulse(0/1)  $3=seed  $4=描述
  local tag="$1" pulse="$2" seed="$3" desc="$4"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"
    return 0
  fi
  echo "---- $desc  seed=$seed  tag=$tag ----" >> "$LOG"
  local extra=""
  [ "$pulse" = "1" ] && extra="--pulse-freq 1"
  "$PY" "$SRC/train_pamnet.py" --tag "$tag" --data "$DATA" \
      --norm gn --short 1 --long 1 --gate 1 --pulse "$pulse" \
      --freq-refine 2.0 $extra --seed "$seed" --outdir "$OUT" >> "$LOG" 2>&1
  local rc=$?
  echo "[完成] $tag  returncode=$rc" | tee -a "$LOG"
  return $rc
}

echo "==== 域内重训 a15_r10 开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "sens_r10_nopulse${SUF}" 0 "$SEED" "R4 (无脉冲)"
  run_one "sens_r10_pf${SUF}"      1 "$SEED" "R5 (p注入)"
done
echo "==== 域内重训 a15_r10 结束 $(date) ====" >> "$LOG"
