#!/usr/bin/env bash
# =============================================================================
# 启动实验_v20服务器.sh —— v20 服务器全套实验（S1~S5）
#
# 一句话启动：
#     nohup bash ~/TP-MORL/启动实验_v20服务器.sh > ~/v20.log 2>&1 &
#
# 特性
#   * 跑批前自检：环境、数据集、奖励=计价的机械自证、git 远端可达
#   * 逐阶段运行，每段结束**自动 commit + push + 打标签**，备注里带上关键数字
#   * 断点续跑：已有汇总 CSV 的阶段自动跳过（FORCE=1 可强制重跑）
#   * 单段失败不中断后续（状态记进 RUN_STATUS.md），push 失败只告警
#
# 常用环境变量
#     STAGES="S1 S2"      只跑指定阶段（默认 S1 S2 S3 S4 S5）
#     FAST=1              冒烟模式：极小规模，十几分钟跑完全流程
#     FORCE=1             忽略已有结果，强制重跑
#     SEEDS="0 1 2 3 4"   覆盖主实验种子
#     NO_PUSH=1           只 commit 不 push
#
# 冻结协议（服务器实验期间不得修改）
#     env_gym.py / schedule.py / opportunity.py / reward.py / oracle / 评价指标
#     以及 gamma、树的棵数与深度、reward scaling、STOP、PBRS、prescreen、
#     机会场幅度与窗宽。本轮实验的卖点正是
#     "同一个 MDP + 同一个 reward + 同一个 oracle + 同一套评价，只换 learner"。
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

STAGES="${STAGES:-S1 S2 S3 S4 S5}"
FAST="${FAST:-0}"
FORCE="${FORCE:-0}"
NO_PUSH="${NO_PUSH:-0}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/v20_${STAMP}"
mkdir -p "$LOGDIR"

if [ "$FAST" = "1" ]; then
  SEEDS="${SEEDS:-0}"; EP_EACH=2; S1_SEEDS="0"; S1_ALPHAS="0 1.0"
  S5_EP=2; S5_ITER="--n-iter 5"; TAGPFX="v20-smoke"
else
  SEEDS="${SEEDS:-0 1 2 3 4}"; EP_EACH=20; S1_SEEDS="0 1 2 3 4"
  S1_ALPHAS="0 0.25 0.5 0.75 1.0"; S5_EP=20; S5_ITER=""; TAGPFX="v20"
fi

say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- 自检
selfcheck() {
  say "自检：环境 / 数据 / 口径 / git"
  python - <<'PY' || return 1
import sys, importlib
for m in ("numpy", "pandas", "scipy", "sklearn", "torch", "joblib"):
    importlib.import_module(m)
import os
need = ["data/processed/gm_dataset_v1/zones_v0/candidate_units.csv",
        "data/processed/gm_dataset_v1/temporal_v0/calib.json"]
missing = [p for p in need if not os.path.exists(p)]
assert not missing, f"缺数据文件：{missing}"
print("  依赖与数据集齐备")
PY
  # 奖励 = 计价的机械自证。不一致直接中止：上一轮就吃过
  # "环境按立项年付钱、oracle 按交付年计价"的亏，整组结论作废。
  python scripts/exp_fqi_local.py --stage L0 || {
    echo "  [中止] L0 口径自证未通过"; return 1; }
  git rev-parse --git-dir >/dev/null 2>&1 || { echo "  [中止] 不在 git 仓库里"; return 1; }
  if [ "$NO_PUSH" != "1" ]; then
    git ls-remote --exit-code origin >/dev/null 2>&1 \
      || echo "  [告警] 远端不可达，push 会失败（实验照跑，结果仍会 commit）"
  fi
  echo "  分支 $BRANCH  commit $(git rev-parse --short HEAD)"
}

# ---------------------------------------------------------------- 提交
# 用法：publish <阶段名> <备注正文>
publish() {
  local stage="$1"; shift
  local note="$*"
  git add -A >/dev/null 2>&1
  if git diff --cached --quiet; then
    echo "  [跳过提交] $stage 没有新文件"
    return 0
  fi
  git commit -q -m "v20 服务器 ${stage} 完成

${note}

分支 ${BRANCH} / 批次 ${STAMP}
冻结协议：env_gym / schedule / opportunity / reward / oracle / 评价指标未改动。" \
    && echo "  已提交 $(git rev-parse --short HEAD)"
  git tag -f "${TAGPFX}-${stage}-${STAMP}" -m "$stage" >/dev/null 2>&1
  if [ "$NO_PUSH" != "1" ]; then
    git push -q origin "HEAD:${BRANCH}" 2>&1 | tail -2 \
      && echo "  已推送 origin/${BRANCH}" || echo "  [告警] push 失败，结果已在本地 commit"
    git push -q -f origin "${TAGPFX}-${stage}-${STAMP}" >/dev/null 2>&1 || true
  fi
}

# 从汇总 CSV 里摘几行关键数字进提交备注 —— 备注要能独立读懂，
# 不能只写"完成"。
summarize() {
  python - "$@" <<'PY' 2>/dev/null || echo "（汇总文件未生成）"
import sys, pandas as pd
f, cols = sys.argv[1], sys.argv[2].split(",")
d = pd.read_csv(f, encoding="utf-8-sig")
cols = [c for c in cols if c in d.columns]
print(d[cols].round(4).to_string(index=False))
PY
}

record() {  # record <阶段> <状态> <耗时秒>
  printf '| %s | %s | %s 秒 | %s |\n' "$1" "$2" "$3" "$(date +%F\ %T)" >> RUN_STATUS.md
}

run_stage() {  # run_stage <名> <产物文件> <命令...>
  local name="$1"; local sentinel="$2"; shift 2
  if [[ "$STAGES" != *"$name"* ]]; then return 0; fi
  if [ -f "$sentinel" ] && [ "$FORCE" != "1" ]; then
    say "$name 已有结果（$sentinel），跳过。FORCE=1 可强制重跑"
    return 0
  fi
  say "$name 开始：$*"
  local t0; t0=$(date +%s)
  "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  local rc=${PIPESTATUS[0]}
  # DT 必须是全局的：下面各阶段的 record 在 run_stage 之外引用它，
  # 写成 local 会在 set -u 下直接报 unbound variable。
  DT=$(( $(date +%s) - t0 ))
  echo "$name 退出码 $rc，用时 ${DT} 秒"
  return $rc
}

# ---------------------------------------------------------------- 主流程
[ -f RUN_STATUS.md ] || printf '# v20 服务器实验状态\n\n| 阶段 | 状态 | 用时 | 时间 |\n|---|---|---|---|\n' > RUN_STATUS.md
selfcheck || { echo "自检未通过，终止。"; exit 1; }
DT=0; dt=0
say "批次 $STAMP  阶段 [$STAGES]  FAST=$FAST  种子 [$SEEDS]  每策略回合 $EP_EACH"

# ---- S1 合成世界：Where + When 机制 -----------------------------------------
run_stage S1 results_srv_s1/fqi_synth_summary.csv \
  python scripts/exp_fqi_local.py --stage L2 --out results_srv_s1 \
    --alphas $S1_ALPHAS --seeds $S1_SEEDS --ep-each "$EP_EACH"
rc=$?
if [[ "$STAGES" == *S1* ]]; then
  if [ $rc -eq 0 ]; then
    record S1 完成 "$DT"
    publish S1 "合成世界（24 地块 / 10 年 / 每年 2 名额），α 从 0 到 1，
方法：随机 / 纯空间排序 / 时间感知贪心 / Double-FQI / oracle。

$(summarize results_srv_s1/fqi_synth_summary.csv alpha,method,相对oracle,标准差,Where召回,When_MAE)

判读：α=0 是对照端（四者应当都接近 1.000）；α 增大时纯空间排序应当单调退化，
时序方法保持高位。**不要求** FQI 超过时间感知贪心 —— 本地已诊断：
小世界候选少、年内竞争不足，Q 的延续项在年内近似常数。"
  else record S1 失败 "$DT"; fi
fi

# ---- S2 光明区主实验（论文主表）---------------------------------------------
run_stage S2 results_srv_s2/fqi_real_summary.csv \
  python scripts/exp_fqi_local.py --stage L3 --out results_srv_s2 \
    --seeds $SEEDS --ep-each "$EP_EACH" --with-bc --dump-model
rc=$?
if [[ "$STAGES" == *S2* ]]; then
  if [ $rc -eq 0 ]; then
    record S2 完成 "$DT"
    publish S2 "光明区主实验（717 单元 / 25 年 / 每年 3 名额 / 只押 Floor / 初筛 K=20），
方法：随机 / 静态·近视贪心 / BC / 时间感知贪心 / Double-FQI / oracle，种子 [$SEEDS]。

$(summarize results_srv_s2/fqi_real_summary.csv method,n,相对oracle,标准差,挑错单元,放错年份)

本地参照（3 种子）：0.757 / 0.775 / 0.846 / 0.894 / 0.904 / 1.000。
只需确认三件事：① Double-FQI 是否仍稳定在约 0.90；② 是否仍高于时间感知贪心；
③ 放错年份是否仍明显低于时间感知贪心（本地 545 对 608）。
Q 模型已随结果落盘（fqi_model_seed*.joblib），S3-B 直接读取、不重训。"
  else record S2 失败 "$DT"; fi
fi

# ---- S3 机制分解 -------------------------------------------------------------
run_stage S3 results_srv_s3/s3_ladder_summary.csv \
  python scripts/exp_srv_mechanism.py --out results_srv_s3 \
    --model-dir results_srv_s2 --seeds $SEEDS
rc=$?
if [[ "$STAGES" == *S3* ]]; then
  if [ $rc -eq 0 ]; then
    record S3 完成 "$DT"
    publish S3 "机制分解。3A 信息阶梯（不训练）：空间/当期信息 → 加入未来时间信息 → 加入跨年协调。
**两档筛选分开报告，不得混用**：无初筛档本机实测 0.438 / 0.741 / 0.912 / 1.000，
而 Double-FQI 的 0.904 是初筛 K=20 档（该档 0.716 / 0.775 / 0.894 / 1.000），
拼在同一张阶梯里就是两套口径拼图。

$(summarize results_srv_s3/s3_ladder_summary.csv prescreen,method,相对oracle,挑错单元,放错年份)

3B Q 值诊断（读 S2 模型，未重训）：
$(summarize results_srv_s3/s3_q_diagnostics.csv seed,全局秩相关,年内秩相关,年内极差均值)

判读：年内秩相关 ≈ 1 说明延续项在年内近似常数、不改变 top-K（合成世界就是这样）；
明显小于 1 才说明 Q 在年内区分了候选。"
  else record S3 失败 "$DT"; fi
fi

# ---- S4 历史回测：数据可用性闸 -----------------------------------------------
if [[ "$STAGES" == *S4* ]]; then
  say "S4 历史回测：先过数据可用性闸"
  t0=$(date +%s); S4_OUT=results_srv_s4 python scripts/check_historical_data.py \
    2>&1 | tee "$LOGDIR/S4.log"; rc=${PIPESTATUS[0]}; DT=$(( $(date +%s) - t0 ))
  if [ $rc -eq 3 ]; then
    record S4 "BLOCKED（数据不足）" "$DT"
    publish S4 "历史回测 = **BLOCKED，数据不足，非代码问题**。

指南设计的是 1985–2009 训练 → 2010–2021 留出验证，本仓库数据不支持：
1. 时间覆盖：光明区街道年面板只有 2010–2018 有非零记录（共 34 条，4 个街道），
   2010 年前无记录、2019 年起全为 0 —— 没有 1985–2009 可训。
2. 粒度：记录是街道 × 年的汇总计划数；gm_renewal_units.csv 的 52 个具名项目
   没有 uid，候选单元表也没有街道字段，无法落到 717 个候选单元上。
3. 循环论证：calib.json 写明 hazard 来自 109 对公告→批准配对、tau_valid 来自
   配对时滞中位数、quota=3 来自光明区 2011–2018 计划公告均值 3.1 ——
   **这批记录已被用作标定输入**，再当留出集不成立。

解锁所需数据见 results_srv_s4/S4_STATUS.md（优先级：计划批次名单含单元边界+公告年份
→ 候选单元的街道归属字段 → 2019 年后的公告记录）。"
  else
    record S4 "异常（退出码 $rc）" "$DT"
  fi
fi

# ---- S5 未来规划情景 ---------------------------------------------------------
run_stage S5 results_srv_s5/s5_summary.csv \
  python scripts/exp_srv_scenarios.py --out results_srv_s5 \
    --seeds ${SEEDS%% *} --ep-each "$S5_EP" $S5_ITER
rc=$?
if [[ "$STAGES" == *S5* ]]; then
  if [ $rc -eq 0 ]; then
    record S5 完成 "$DT"
    publish S5 "未来规划情景（三个，不多做）：基线 / 基础设施提前 / 更新压力增强。

$(summarize results_srv_s5/s5_summary.csv scenario,相对oracle,与基线_Where重叠,与基线_When平移年)

**口径声明**：四个机会场通道全部是情景参数、无实证标定（单元表没有建成年份、
历年规划定位与设施投用年份），结论只能写成条件句，不得写成对光明区未来的预测。
另注：'立项年重心' 在配额年年咬满时是构造常数（恒为 (T−1)/2），不可当结果报，
有信息量的是与基线逐单元对比的 When 平移量。"
  else record S5 失败 "$DT"; fi
fi

say "全部阶段结束。状态表见 RUN_STATUS.md，日志在 $LOGDIR/"
tail -n 12 RUN_STATUS.md
