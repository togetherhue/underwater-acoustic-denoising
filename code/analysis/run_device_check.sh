#!/usr/bin/env bash
# 设备效应对「脉冲分支效应 Δ」的影响：在 base 条件上用 GPU 重训 R4/R5
# ============================================================================
# 【为什么需要这一组】
# 域内重训（a15_r10）在 GPU 上完成，而 base 条件的全部结果在 CPU 上训练。
# 实测同一配置跨设备训练终值差 0.11~0.45 dB。若这个设备差对 Δ = R5 − R4 也有
# 同量级的影响，就无法判断
#       域内 Δ = 0.32  vs  base Δ = 0.79
# 究竟是"rate=10 条件本身让分支收益变小"，还是"GPU 与 CPU 的差异"。
#
# 本脚本在 base 条件上用 GPU 重训 R4/R5（与 CPU 存档相同的两个种子），
# 直接测出【同一条件、同一配置、仅设备不同】的 Δ 差异 = 设备效应。
#
# 用法：bash run_device_check.sh
# 输出 tag: devchk_nopulse / devchk_nopulse_s2 / devchk_pf / devchk_pf_s2
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="D:/project/dataset_pam/alphastable"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/device_check.log"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

run_one () {   # $1=tag $2=pulse $3=seed $4=描述
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
  echo "[完成] $tag  returncode=$?  $(date +%H:%M:%S)" | tee -a "$LOG"
}

echo "==== 设备效应核对（base 条件，GPU）开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "devchk_nopulse${SUF}" 0 "$SEED" "R4 (无脉冲)"
  run_one "devchk_pf${SUF}"      1 "$SEED" "R5 (p注入)"
done
echo "==== 设备效应核对结束 $(date) ====" >> "$LOG"
