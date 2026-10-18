#!/usr/bin/env bash
# 输入尺度归一化（x / rms(x)）重训 —— base 条件，R4/R5 × 3 种子
# ============================================================================
# 【为什么必须做】
# 训练数据在 MATLAB 侧把 rms(clean) 归一化为 1，模型对输入整体尺度极度敏感：
#   s=1.00 → 25.58 dB（训练尺度）
#   s=0.38 →  9.69 dB（按 noisy rms 归一化就落在这里，损失 15.9 dB）
#   s=0.10 →  1.90 dB
# 缩放不改变 SNR，所以这纯粹是尺度敏感，不是任务变难。
# 实测数据拿不到 clean 信号，只能按 noisy 归一化 —— 不修无法部署。
#
# 盲自校准（4 个无参考准则动态范围仅 0.03%~0.7%）与物理估计（带外外推被
# α 稳定噪声的脉冲性打败，rms 估小约 300 倍）两条绕开的路都已失败，
# 唯一可靠解是在训练时就吃归一化输入。
#
# 【实现方式】数据层面：PAMDataset 把 noisy/clean 同除 noisy 的 rms。
# 模型零改动 → 与已有 checkpoint 结构完全兼容；metrics 是尺度不变量，
# 因此归一化域里算出的指标与原尺度等价。
#
# 【对照组】devchk_*（base 条件 GPU、无归一化，2 个种子）。
# 本脚本用同 3 个种子，其中 20261013 与 2 可与 devchk 逐对配对 ——
# 唯一变量就是 input_norm，完全不受设备影响。
#
# 用法：bash run_input_norm.sh
# 输出 tag: inm_nopulse{,_s2,_s3} / inm_pf{,_s2,_s3}
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="D:/project/dataset_pam/alphastable"
OUT="D:/project/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/input_norm.log"

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

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
      --freq-refine 2.0 $extra --seed "$seed" --input-norm 1 \
      --outdir "$OUT" >> "$LOG" 2>&1
  echo "[完成] $tag  returncode=$?  $(date +%H:%M:%S)" | tee -a "$LOG"
}

echo "==== 输入尺度归一化重训（base, GPU）开始 $(date) ====" >> "$LOG"
for SEED in 20261013 2 3; do
  case $SEED in 20261013) SUF="";; *) SUF="_s$SEED";; esac
  run_one "inm_nopulse${SUF}" 0 "$SEED" "R4 无脉冲 + 输入归一化"
  run_one "inm_pf${SUF}"      1 "$SEED" "R5 p注入 + 输入归一化"
done
echo "==== 输入尺度归一化重训结束 $(date) ====" >> "$LOG"
