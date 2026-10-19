#!/usr/bin/env bash
# 补齐 a20（α=2.0 近高斯端点）的种子 4、5 —— 8.14.2 的负结果目前只有 1 个有效种子
# ============================================================================
# 【为什么需要这一组】
# 8.14.2 记录的事实：a20 的 3 个种子 Δ = +1.685 / -0.861 / +0.711（均值 +0.512，
# sd 1.09），而 R4 跨种子极差 2.115 dB（R5 仅 0.43 dB）。
# 当时判定为"分支价值不可判"，原因是 R4 训练不稳定 —— 但只有 3 个种子的
# 极差估计本身极不稳（3 个样本的极差严重低估总体极差）。
# 补到 5 个种子后可以：
#   ① 重新估计 R4 / R5 的真实种子间 sd（4 自由度 vs 2）；
#   ② 判断 -0.861 这个离群种子是偶发还是常态；
#   ③ 用 5 种子重算 Δ 的均值与 t 检验 —— 这是 8.14.2 结论能否落到
#      "不可判"还是"确实为负"的关键。
#
# 【与 run_indomain_a20.sh 的关系】完全相同的训练配置（同数据、同超参），
# 只补 s4 / s5 两个后缀，不重跑已有的 1/2/3。
#
# 【断点续跑】已存在 summary_<tag>.csv 的跳过。
# 【必须串行！】不能与其他训练脚本并发（见 run_indomain_a20.sh 的 CUDNN 崩溃记录）。
# 用法：bash run_a20_seeds45.sh
# 输出 tag: sens_a20_nopulse_s4 / sens_a20_pf_s4 / sens_a20_nopulse_s5 / sens_a20_pf_s5
set -u
PY="/d/anaconda/python.exe"
SRC="C:/Users/32213/PycharmProjects/project1"
DATA="E:/project/data/ds_sens/a20_r5"
OUT="E:/project/data/output_pamnet"
LOG="E:/workbuddy/store/2026-10-12-22-50-08/sens_logs/train_a20_s45.log"

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

run_one () {   # $1=tag  $2=pulse(0/1)  $3=seed  $4=描述
  local tag="$1" pulse="$2" seed="$3" desc="$4"
  if [ -f "$OUT/summary_${tag}.csv" ]; then
    echo "[跳过] $tag 已完成" | tee -a "$LOG"; return 0
  fi
  echo "---- $desc  seed=$seed  tag=$tag  $(date +%H:%M:%S) ----" >> "$LOG"
  local extra=""
  [ "$pulse" = "1" ] && extra="--pulse-freq 1"
  "$PY" "$SRC/train_pamnet.py" --tag "$tag" --data "$DATA" \
      --norm gn --short 1 --long 1 --gate 1 --pulse "$pulse" \
      --freq-refine 2.0 $extra --seed "$seed" --outdir "$OUT" >> "$LOG" 2>&1
  local rc=$?
  echo "[完成] $tag  returncode=$rc  $(date +%H:%M:%S)" | tee -a "$LOG"
  return $rc
}

echo "==== a20 补种子 4/5 开始 $(date) ====" >> "$LOG"
for SEED in 4 5; do
  run_one "sens_a20_nopulse_s${SEED}" 0 "$SEED" "R4 (无脉冲) seed${SEED}"
  run_one "sens_a20_pf_s${SEED}"      1 "$SEED" "R5 (p注入) seed${SEED}"
done
echo "==== a20 补种子 4/5 结束 $(date) ====" >> "$LOG"
