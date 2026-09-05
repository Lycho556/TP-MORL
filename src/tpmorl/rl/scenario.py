# -*- coding: utf-8 -*-
"""情景参数的统一改写与命名。

本项目的情景参数（预算、制度时序）是**模块级常量**，由入口脚本在 main 里改写。
`exp_opt_quality.py` 用 spawn 起子进程，子进程会重新 import 一遍模块、拿回默认
值，所以每个子进程都必须自己调用一次 `apply()`。以前只有 3 个预算类参数、在两
处各抄一遍；访谈落实后又多了 4 个制度参数（见 docs/访谈落实_v3.md 第 1、2、5
条），抄写式改写很容易漏，故集中到这里。

`inst_tag()` 给出制度参数的短标识，进入分母缓存文件名。**这一条是必须的**：
目标归一化的分母是"当前约束情景下参考策略集的可达上界"，制度窗口一改（例如
第 5 条的 tau_valid=2, tau_ext=1 法定窗口对照），可达上界随之改变；若缓存键里
不含制度参数，对照情景会静默复用基线情景的分母，两组结果不可比。
"""
INST_FIELDS = ("tau_valid", "tau_ext", "cooldown", "build_years")
BUDGET_FIELDS = ("budget", "carry", "growth")


_HORIZON = None      # 仅供 inst_tag() 入键；T 本身由各脚本传给 RenewalEnv


def apply(budget=None, carry=None, growth=None,
          tau_valid=None, tau_ext=None, cooldown=None, build_years=None,
          horizon=None):
    """把情景参数写回模块常量。None 表示沿用模块默认值，不改写。

    `horizon` 不改写任何常量，只登记进 `inst_tag()`：规划期长度改变可达上界，
    分母不可跨 T 复用。
    """
    global _HORIZON
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S

    if horizon is not None:
        _HORIZON = int(horizon)

    if budget is not None:
        env_gym.BUDGET = float(budget)
    if carry is not None:
        env_gym.CARRY_CAP = float(carry)
    if growth is not None:
        env_gym.FAR_GROWTH = float(growth)

    if tau_valid is not None:
        S.TAU_VALID = int(tau_valid)
    if tau_ext is not None:
        S.TAU_EXT = int(tau_ext)
    if cooldown is not None:
        S.COOLDOWN = parse_cooldown(cooldown)
    if build_years is not None:
        # 全体通道同值；分档差异化待数据／动作空间到位后再逐通道设
        S.BUILD_YEARS = int(build_years)
        S.BUILD_YEARS_BY_CHANNEL = {c: int(build_years)
                                    for c in S.BUILD_YEARS_BY_CHANNEL}


_ABSORB_WORDS = ("absorb", "inf", "t", "吸收态", "永久")


def parse_cooldown(v):
    """把冷却期规格解析成数值。

    接受整数年数，或 absorb/inf/T（不分大小写）表示"本规划期内不再申请"的吸收态。
    吸收态是依 2026-09 规划局实务答复的主设定，见 schedule.py 模块文档。
    """
    from tpmorl.env import schedule as S
    if isinstance(v, str) and v.strip().lower() in _ABSORB_WORDS:
        return S.COOLDOWN_ABSORB
    v = float(v)
    return S.COOLDOWN_ABSORB if v == float("inf") else int(v)


def cooldown_tag(v=None):
    """冷却期的文件名短标识。吸收态记 `DA`，数值档记 `D{年数}`。"""
    from tpmorl.env import schedule as S
    v = S.COOLDOWN if v is None else v
    return "DA" if not _np_isfinite(v) else f"D{int(v)}"


def _np_isfinite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def inst_tag():
    """当前制度参数的短标识，用于分母缓存文件名。"""
    from tpmorl.env import schedule as S
    y = "".join(f"{c}-{S.BUILD_YEARS_BY_CHANNEL[c]}"
                for c in sorted(S.BUILD_YEARS_BY_CHANNEL))
    t = "" if _HORIZON is None else f"T{_HORIZON}"
    return f"V{S.TAU_VALID}E{S.TAU_EXT}{cooldown_tag()}Y{y}{t}"


def describe():
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S
    return (f"年度预算 {env_gym.BUDGET:.0f}（结转上限 {env_gym.CARRY_CAP:g}×）  "
            f"配额 {S.QUOTA}  容积率年增 {env_gym.FAR_GROWTH:.0%}\n"
            f"有效期 {S.TAU_VALID}+{S.TAU_EXT} 年  "
            f"失效后 {'本规划期内不再申请（吸收态）' if not _np_isfinite(S.COOLDOWN) else f'冷却 {int(S.COOLDOWN)} 年'}"
            f"（无条文依据，依 2026-09 规划局实务答复）  "
            f"建设年限 {S.BUILD_YEARS_BY_CHANNEL}")


def add_args(ap):
    """给 argparse 加上情景参数。默认 None = 用模块默认值。"""
    ap.add_argument("--tau-valid", type=int, default=None,
                    help="计划有效期（年）。实测标定 3；现行法定窗口对照用 2")
    ap.add_argument("--tau-ext", type=int, default=None,
                    help="延期上限（年）。实测标定 2；现行法定窗口对照用 1")
    ap.add_argument("--cooldown", type=str, default=None,
                    help="失效后冷却期。默认 absorb=本规划期内不再申请（吸收态，"
                         "依 2026-09 规划局实务答复的主设定）；也可给年数作宽松"
                         "对照，敏感性方向 {2, 5, 吸收态}，0 档仅为最宽松极端参照")
    ap.add_argument("--build-years", type=int, default=None,
                    help="建设年限（年），全通道同值。默认按通道表取 5")
    ap.add_argument("--horizon", type=int, default=15,
                    help="规划期长度 T。建设年限延长后可能需要放宽，见第 3 条诊断")


def from_args(a):
    """从 argparse 结果取出 apply() 用的关键字字典。"""
    return dict(tau_valid=a.tau_valid, tau_ext=a.tau_ext,
                cooldown=a.cooldown, build_years=a.build_years)
