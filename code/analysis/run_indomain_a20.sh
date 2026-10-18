#!/usr/bin/env bash
# 域内重训：a20_r5（α=2.0 近高斯背景，突发率 5/s 与 base 一致）
# ============================================================================
# 【为什么需要这一组】
# 8.11 用域内重训证明了 α 轴上"α=1.8 的尾形增强"是分布偏移假象、
# "α=1.2 的压低"只剩 0.24 dB 且不显著。但 α=2.0 这个端点没做。
# 零样本下 α=2.0 的 Δ = +1.297 dB（pulse_eff，5 种子，p=8e-18），如果域内重训后仍接近该值，
# 说明"背景越接近高斯、分支越没用"这条在端点上也成立（且不是分布偏移）。
#
# 注意：这是【未匹配功率】的 a20_r5（与 8.2 的零样本矩阵同一数据集）。
# 等功率匹配版本需要名义 SNR 低至 −24 dB，数据生成有数值风险，本组不做。
# 因此本组只能与同样未匹配的零样本 a20_r5（+0.36）比，不能与 base 直接比。
#
# 【断点续跑】已存在 summary_<tag>.csv 的跳过。
# 用法：bash run_indomain_a20.sh
# 输出 tag: sens_a20_nopulse{,_s2,_s3} / sens_a20_pf{,_s2,_s3}
# 【必须串行！】本脚本与另一个训练脚本**不能同时启动**。
#   2026-10-18 实测：r25 与 a20 两路同时起，第一个 run 在 loss.backward() 处
#   报 "RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED"
#   （主机内存分配失败，不是显存不够；GPU 只用了 4.2/6.1 GB）。
#   两个进程各带 3~4 个 DataLoader worker，启动瞬间的内存峰值撞到系统上限。
#   正确做法：一个跑完再起下一个（用 for 循环等待 summary 计数即可）。
#
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="E:/project/data/ds_sens/a20_r5"
OUT="E:/project/data/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_a20.log"

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

run_one () {   # $1=tag  $2=pulse(0/1)  $3=seed  $4=描述
  local tag="$1" pulse="$2" seed="$3" desc="$4"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"; return 0
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

echo "==== 域内重训 a20_r5（近高斯端点）开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "sens_a20_nopulse${SUF}" 0 "$SEED" "R4 (无脉冲)"
  run_one "sens_a20_pf${SUF}"      1 "$SEED" "R5 (p注入)"
done
echo "==== 域内重训 a20_r5 结束 $(date) ====" >> "$LOG"
