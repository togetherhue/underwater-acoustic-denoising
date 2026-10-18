#!/usr/bin/env bash
# 适用边界实验：域内重训（在异常条件下重新训练 R4/R5，区分"分布偏移"与"方法本身优劣"）
# 条件 a15_r10：α=1.5（与训练一致），突发到达率 10/s（训练条件的 2 倍）
#   —— 对"脉冲感知分支是否有价值"最干净的单因素操纵：背景噪声不变，只有干扰密度翻倍。
# 种子取 20261013 / 2 / 3，与基线前三个种子一致，便于按种子配对。
#
# 【设备】torch 2.14.1+cu126，v3_model.DEVICE 自动选 cuda。
#   GPU 下每轮约 65s（CPU 为 600~1050s）。OMP_NUM_THREADS 降到 4 即可 —— 计算在 GPU，
#   主进程只做数据搬运与指标汇总，多开线程反而与 DataLoader worker 抢核。
#
# 【断点续跑】已存在 summary_sens_r10_<tag>.csv 的训练直接跳过。
#   本环境会回收裸 nohup 的进程，长任务也可能被中断，每次重启都从断点继续。
#
# 【为什么全部在 GPU 上重跑】实测同一配置在 CPU 与 GPU 上各训一次，
#   终值差 0.11~0.45 dB（testC 最大）—— 这是训练路径分叉（浮点归约顺序不同
#   → 梯度不同 → 15 轮 Adam 后模型不同），不是推理错误（同一 checkpoint
#   跨设备评估仅差 0.0004 dB）。该量级与待测效应（Δ≈0.78 dB）相当，
#   混用两种设备会产生设备混淆，因此 6 个全部在 GPU 上重跑。
#   CPU 结果已归档为 *_cpurun.* 保留作跨设备对照。
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="E:/project/data/ds_sens/a15_r10"
OUT="E:/project/data/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_r10.log"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

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
  echo "[完成] $tag  returncode=$rc  $(date +%H:%M:%S)" | tee -a "$LOG"
  return $rc
}

echo "==== 域内重训 a15_r10 开始（GPU） $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "sens_r10_nopulse${SUF}" 0 "$SEED" "R4 (无脉冲)"
  run_one "sens_r10_pf${SUF}"      1 "$SEED" "R5 (p注入)"
done
echo "==== 域内重训 a15_r10 结束 $(date) ====" >> "$LOG"
