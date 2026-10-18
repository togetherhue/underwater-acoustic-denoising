#!/usr/bin/env bash
# 域内重训：a15_r25（α=1.5 与 base 相同，突发率 2.5/s = base 的一半）
# ============================================================================
# 【为什么需要这一组】
# 8.6 的结论是"rate 轴反向：突发越稀疏，脉冲分支收益越大"
#   base(5/s) Δ=+0.790 → 零样本 2.5/s Δ=+0.975 → 零样本 10/s Δ=+0.548
#   （数值取自 sens_summary_testA.json 的 pulse_eff，勿凭记忆写）
# 但域内重训只做了 rate=10 一个方向，确认了"密集端收益变小是真实条件效应"。
# 稀疏端（2.5/s）**没有域内验证** —— 如果那边域内重训后 Δ 也回到 base，
# 那"反向"就只剩 rate=10 一个点，结论必须收窄为"只有密集端成立"。
#
# 这是单因素操纵最干净的一组：α 与 base 完全一致（1.5），只有到达率减半。
#
# 【断点续跑】已存在 summary_<tag>.csv 的跳过。
# 用法：bash run_indomain_r25.sh
# 输出 tag: sens_r25_nopulse{,_s2,_s3} / sens_r25_pf{,_s2,_s3}
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
DATA="E:/project/data/ds_sens/a15_r25"
OUT="E:/project/data/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_r25.log"

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

echo "==== 域内重训 a15_r25（稀疏端）开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "sens_r25_nopulse${SUF}" 0 "$SEED" "R4 (无脉冲)"
  run_one "sens_r25_pf${SUF}"      1 "$SEED" "R5 (p注入)"
done
echo "==== 域内重训 a15_r25 结束 $(date) ====" >> "$LOG"
