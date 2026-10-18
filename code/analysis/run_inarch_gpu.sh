#!/usr/bin/env bash
# base 条件的架构排序域内验证（GPU）：R3 纯频域 + 纯时域 + 补第 3 个基线种子
# ============================================================================
# 【为什么必须做】
# 现有域内重训只覆盖 R4/R5，只回答了"有没有脉冲分支"。
# "架构排序 R5>R4>R3>纯时域"在域内是否保持，完全没验证。
# 这不是假想风险 —— 零样本矩阵里，高斯背景上 R4 就被 R3 反超过
# （24.22 vs 24.89）。若 base 域内也出现反超，"排序稳健"的结论要收窄。
#
# 【设备对齐】全部在 GPU 上跑，与 devchk_*（已有 2 个种子）严格同设备。
# 本脚本补齐 devchk 的第 3 个种子，使 GPU 基线也是 3 个，
# 与 R3/纯时域/input_norm 组的种子数一致，Δ 的跨种子检验才有意义。
#
# 用法：bash run_inarch_gpu.sh
# 输出 tag: gpu_fo{,_s2,_s3}(R3) / gpu_pn{,_s2,_s3}(纯时域)
#           devchk_nopulse_s3 / devchk_pf_s3
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="D:/project/dataset_pam/alphastable"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/inarch_gpu.log"

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

# ---- R3 纯频域掩码：无时域分支，靠频域精修 ----
run_fo () {   # $1=tag $2=seed $3=描述
  local tag="$1" seed="$2" desc="$3"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"; return 0
  fi
  echo "---- $desc  seed=$seed  tag=$tag ----" >> "$LOG"
  "$PY" "$SRC/train_pamnet.py" --tag "$tag" --data "$DATA" \
      --norm gn --short 0 --long 0 --gate 1 --pulse 0 \
      --freq-refine 2.0 --seed "$seed" --outdir "$OUT" >> "$LOG" 2>&1
  echo "[完成] $tag  returncode=$?  $(date +%H:%M:%S)" | tee -a "$LOG"
}

# ---- 纯时域（无脉冲分支、无频域精修）----
run_pn () {   # $1=tag $2=seed $3=描述
  local tag="$1" seed="$2" desc="$3"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"; return 0
  fi
  echo "---- $desc  seed=$seed  tag=$tag ----" >> "$LOG"
  "$PY" "$SRC/train_pamnet.py" --tag "$tag" --data "$DATA" \
      --norm gn --short 1 --long 1 --gate 1 --pulse 0 \
      --freq-refine 0.0 --seed "$seed" --outdir "$OUT" >> "$LOG" 2>&1
  echo "[完成] $tag  returncode=$?  $(date +%H:%M:%S)" | tee -a "$LOG"
}

# ---- R4 / R5（补第 3 个 GPU 基线种子，与 devchk 同参数）----
run_dc () {   # $1=tag $2=pulse $3=seed $4=描述
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

echo "==== 架构排序域内验证（base, GPU）开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_fo "gpu_fo${SUF}"  "$SEED" "R3 纯频域掩码"
  run_pn "gpu_pn${SUF}"  "$SEED" "纯时域（无脉冲分支）"
done
run_dc "devchk_nopulse_s3" 0 3 "R4 无脉冲（GPU 基线补种子3）"
run_dc "devchk_pf_s3"      1 3 "R5 p注入（GPU 基线补种子3）"
echo "==== 架构排序域内验证结束 $(date) ====" >> "$LOG"
