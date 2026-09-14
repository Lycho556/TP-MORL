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
INST_FIELDS = ("tau_valid", "tau_ext", "cooldown", "build_years", "gamma")
BUDGET_FIELDS = ("budget", "carry", "growth")


_HORIZON = None      # 仅供 inst_tag() 入键；T 本身由各脚本传给 RenewalEnv
_HORIZON_EVAL = None # 同上，对应 RenewalEnv 的 T_eval（尾部评价期）

_DEFAULTS = None     # 首次 reset()/apply() 时抓拍的模块出厂值


def _snapshot():
    """抓拍受 apply() 改写的全部模块常量的当前值。"""
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S
    return dict(BUDGET=env_gym.BUDGET, CARRY_CAP=env_gym.CARRY_CAP,
                FAR_GROWTH=env_gym.FAR_GROWTH, GAMMA=env_gym.GAMMA,
                BUDGET_MODE=env_gym.BUDGET_MODE, STAGE_INIT=env_gym.STAGE_INIT,
                TAU_VALID=S.TAU_VALID, TAU_EXT=S.TAU_EXT,
                COOLDOWN=S.COOLDOWN, BUILD_YEARS=S.BUILD_YEARS,
                QUOTA=S.QUOTA, HAZARD=tuple(S.HAZARD),
                BUILD_YEARS_BY_CHANNEL=dict(S.BUILD_YEARS_BY_CHANNEL))


def reset():
    """把全部情景常量恢复到出厂值。

    **在同一进程内连续处理多个情景时必须先调用本函数。** `apply()` 对 None 参数
    不改写、只沿用当前值，这意味着"上一个情景设过、这一个情景没设"的参数会静默
    继承下来——例如先跑 `cooldown=5` 的组、再跑 `cooldown=None` 的组，后者实际
    仍在 5 年冷却下运行，而它的 runs.json 会记成 None。

    这与 2026-09-12 定位的 `FAR_GROWTH` 泄漏是同一类缺陷（见 scale.reference_returns
    的说明）：**沉默的状态继承**。多进程池若复用 worker（Pool 默认行为），同样会踩到，
    故审计类脚本除调用本函数外还应设 `maxtasksperchild=1`。
    """
    global _DEFAULTS, _HORIZON, _HORIZON_EVAL
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S
    if _DEFAULTS is None:            # 进程内首次调用即出厂值，无需恢复
        _DEFAULTS = _snapshot()
        return
    d = _DEFAULTS
    env_gym.BUDGET, env_gym.CARRY_CAP = d["BUDGET"], d["CARRY_CAP"]
    env_gym.FAR_GROWTH, env_gym.GAMMA = d["FAR_GROWTH"], d["GAMMA"]
    S.TAU_VALID, S.TAU_EXT = d["TAU_VALID"], d["TAU_EXT"]
    S.COOLDOWN, S.BUILD_YEARS = d["COOLDOWN"], d["BUILD_YEARS"]
    S.BUILD_YEARS_BY_CHANNEL = dict(d["BUILD_YEARS_BY_CHANNEL"])
    env_gym.BUDGET_MODE, env_gym.STAGE_INIT = d["BUDGET_MODE"], d["STAGE_INIT"]
    S.QUOTA, S.HAZARD = d["QUOTA"], tuple(d["HAZARD"])
    _HORIZON = _HORIZON_EVAL = None


def apply(budget=None, carry=None, growth=None,
          tau_valid=None, tau_ext=None, cooldown=None, build_years=None,
          horizon=None, gamma=None, horizon_eval=None,
          quota=None, tau_approval=None, build_years_by_channel=None,
          budget_mode=None, stage_init=None):
    """把情景参数写回模块常量。None 表示沿用模块默认值，不改写。

    `horizon` 不改写任何常量，只登记进 `inst_tag()`：规划期长度改变可达上界，
    分母不可跨 T 复用。

    `horizon_eval` 同理：启用尾部评价会改变可达回报的上界，分母**不可**与
    闭区间口径共用缓存。若忘记入键，会静默复用旧分母——这正是必须显式登记的原因。
    """
    global _HORIZON, _HORIZON_EVAL, _DEFAULTS
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S

    if _DEFAULTS is None:        # 先于任何改写抓拍出厂值，供 reset() 恢复
        _DEFAULTS = _snapshot()

    if horizon is not None:
        _HORIZON = int(horizon)
    if horizon_eval is not None:
        # "auto" 先存标记，函数末尾待制度参数写完后再解析成整数
        _HORIZON_EVAL = ("auto" if str(horizon_eval).strip().lower() == "auto"
                         else int(horizon_eval))

    if budget is not None:
        env_gym.BUDGET = float(budget)
    if carry is not None:
        env_gym.CARRY_CAP = float(carry)
    if growth is not None:
        env_gym.FAR_GROWTH = float(growth)
    if gamma is not None:
        # 折现率改变可达上界（分母是折现回报的上界），故必须进 inst_tag()
        env_gym.GAMMA = float(gamma)

    if tau_valid is not None:
        S.TAU_VALID = int(tau_valid)
    if tau_ext is not None:
        S.TAU_EXT = int(tau_ext)
    if cooldown is not None:
        S.COOLDOWN = parse_cooldown(cooldown)
    if build_years is not None:
        # 全体通道同值；逐通道分档改用 build_years_by_channel 显式给出
        S.BUILD_YEARS = int(build_years)
        S.BUILD_YEARS_BY_CHANNEL = {c: int(build_years)
                                    for c in S.BUILD_YEARS_BY_CHANNEL}
    if build_years_by_channel is not None:
        # 必须写在 build_years 之后：两者同时给出时以逐通道表为准。
        # **口径声明**：通道分档没有实证依据（gm_renewal_units.csv 无更新方式字段，
        # 无法逐单元判定拆除重建／综合整治），故一切分档年限只能作为**情景参数**
        # 报告，不得写成实证标定值。见 docs/建议条目审计与改动清单_v15.md。
        d = parse_build_years_by_channel(build_years_by_channel)
        unknown = set(d) - set(S.BUILD_YEARS_BY_CHANNEL)
        if unknown:
            raise ValueError(f"未知通道 {sorted(unknown)}；"
                             f"可更新通道为 {sorted(S.BUILD_YEARS_BY_CHANNEL)}")
        S.BUILD_YEARS_BY_CHANNEL = {c: int(d.get(c, S.BUILD_YEARS_BY_CHANNEL[c]))
                                    for c in S.BUILD_YEARS_BY_CHANNEL}
    if quota is not None:
        # 配额改变可达上界（一年最多能立几个项），分母不可跨配额复用 → 入 inst_tag()
        S.QUOTA = int(quota)
    if tau_approval is not None:
        # 把逐年条件批准率整体替换为**常数风险率** 1/τ_A（几何分布，均值 τ_A 年）。
        # 默认档 HAZARD 是由累计获批数逐年反解的实测向量，本参数是它的敏感性对照，
        # 不是更精确的估计——写论文时必须说明这一档为几何近似。
        ta = float(tau_approval)
        if ta <= 0:
            raise ValueError(f"tau_approval 必须为正，收到 {ta}")
        S.HAZARD = tuple([min(1.0 / ta, 1.0)] * len(_DEFAULTS["HAZARD"]))
    if budget_mode is not None:
        m = str(budget_mode).strip().lower()
        if m not in ("upfront", "staged"):
            raise ValueError(f"budget_mode 只能是 upfront/staged，收到 {budget_mode}")
        env_gym.BUDGET_MODE = m
    if stage_init is not None:
        si = float(stage_init)
        if not 0.0 <= si <= 1.0:
            raise ValueError(f"stage_init 须在 [0,1]，收到 {si}")
        env_gym.STAGE_INIT = si

    # "auto" 必须在**最后**解析：auto_horizon_eval() 读 TAU_VALID/TAU_EXT/
    # BUILD_YEARS，而这几个常量到上面几行才写入。放在函数开头会算出旧制度下的值。
    if _HORIZON_EVAL == "auto":
        if _HORIZON is None:
            raise ValueError("horizon_eval='auto' 需要同时给出 horizon")
        _HORIZON_EVAL = auto_horizon_eval(_HORIZON)


def parse_build_years_by_channel(v):
    """解析 "1:5,2:3,3:3,5:3" 形式的逐通道建设年限；也接受字典。"""
    if isinstance(v, dict):
        return {int(k): int(x) for k, x in v.items()}
    out = {}
    for part in str(v).replace("，", ",").replace("：", ":").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"逐通道建设年限需写成 通道:年数，收到 {part!r}")
        k, x = part.split(":", 1)
        out[int(k)] = int(x)
    if not out:
        raise ValueError(f"未解析出任何通道，收到 {v!r}")
    return out


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
    # 尾部评价期只在被显式启用且真的长于决策期时入键：
    # 这样闭区间口径（T_eval==T）的键与既有分母缓存逐字一致，历史结果不作废。
    if _HORIZON_EVAL is not None and _HORIZON_EVAL != _HORIZON:
        t += f"X{_HORIZON_EVAL}"
    # γ 只在非默认值时入键：默认档保持与既有 _R7 分母缓存的键一致，不作废历史结果
    from tpmorl.rl import env_gym
    g = "" if env_gym.GAMMA == 0.95 else f"G{env_gym.GAMMA:g}".replace(".", "")
    # v15 新增三项，同样**只在非出厂值时**入键，既有缓存逐字不变。
    # 三者都改变可达上界，漏入键会让对照组静默复用基线分母（与 FAR_GROWTH 泄漏同类）：
    #   配额   —— 一年最多能立几个项，直接决定可达总量
    #   批准率 —— 决定有多少立项能走到完工
    #   资金口径 —— staged 下 Cost 改按支付年计入，目标本身的定义就变了
    d = _DEFAULTS or {}
    q = "" if S.QUOTA == d.get("QUOTA", S.QUOTA) else f"Q{int(S.QUOTA)}"
    hz = tuple(float(x) for x in S.HAZARD)
    a = "" if hz == tuple(d.get("HAZARD", hz)) else f"A{_hazard_tag(hz)}"
    if env_gym.BUDGET_MODE == d.get("BUDGET_MODE", env_gym.BUDGET_MODE) \
            and env_gym.STAGE_INIT == d.get("STAGE_INIT", env_gym.STAGE_INIT):
        m = ""
    else:
        m = f"M{env_gym.BUDGET_MODE[0].upper()}{env_gym.STAGE_INIT:g}".replace(".", "")
    return f"V{S.TAU_VALID}E{S.TAU_EXT}{cooldown_tag()}Y{y}{t}{g}{q}{a}{m}"


def _hazard_tag(hz):
    """批准率向量的短标识。常数向量（τ_A 档）记其倒数，便于人读；
    非常数向量退回 4 位十六进制摘要，保证不同向量不会撞键。"""
    import hashlib
    if len(set(hz)) == 1 and hz[0] > 0:
        return f"{1.0 / hz[0]:g}".replace(".", "")
    return hashlib.md5(repr(hz).encode()).hexdigest()[:4]


def describe():
    from tpmorl.rl import env_gym
    from tpmorl.env import schedule as S
    return (f"年度预算 {env_gym.BUDGET:.0f}（结转上限 {env_gym.CARRY_CAP:g}×）  "
            f"配额 {S.QUOTA}  容积率年增 {env_gym.FAR_GROWTH:.0%}\n"
            f"有效期 {S.TAU_VALID}+{S.TAU_EXT} 年  "
            f"失效后 {'本规划期内不再申请（吸收态）' if not _np_isfinite(S.COOLDOWN) else f'冷却 {int(S.COOLDOWN)} 年'}"
            f"（无条文依据，依 2026-09 规划局实务答复）  "
            f"建设年限 {S.BUILD_YEARS_BY_CHANNEL}  折现率 {env_gym.GAMMA:g}\n"
            f"批准率 {tuple(round(float(x), 3) for x in S.HAZARD)}  "
            f"资金口径 {env_gym.BUDGET_MODE}"
            + (f"（立项付 {env_gym.STAGE_INIT:.0%}，余额实施期内按年等额；"
               f"Cost 按支付年计入）" if env_gym.BUDGET_MODE == "staged"
               else "（立项当年全额；Cost 按完工年计入）")
            + f"\n分母缓存键 {inst_tag()}")


def auto_horizon_eval(T):
    """返回使管道必然排空的最短评价期。

    最坏路径（见 schedule.py 的转移）：第 `T-1` 年立项 → S1 至多
    `TAU_VALID+TAU_EXT` 年 → 获批入 S2 → 次年开工（1 年）→ S3 建设
    `BUILD_YEARS` 年 → S4。故

        T_eval = T + TAU_VALID + TAU_EXT + max(BUILD_YEARS) + 1

    **必须按组计算，不能全批写死一个常数**：T=15/τ=3+2/建设 5 得 26，
    但 `--horizon 20` 组需要 31，`--build-years 3` 组只需 24。写死 26 会把
    前者的管道截断在期内，那正是本次要修的错误的翻版。
    调用前必须已 `apply()` 好制度参数——本函数读的是**当时**的模块常量。
    """
    from tpmorl.env import schedule as S
    return int(T + S.TAU_VALID + S.TAU_EXT
               + max(S.BUILD_YEARS_BY_CHANNEL.values()) + 1)


def horizon_eval():
    """已登记的评价期（`apply()` 解析后的整数）；未启用则为 None。"""
    return _HORIZON_EVAL


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
    ap.add_argument("--build-years-by-channel", default=None,
                    help="逐通道建设年限，形如 1:5,2:3,3:3,5:3。与 --build-years "
                         "同时给出时以本项为准。**无实证依据**：单元表没有更新方式"
                         "字段，无法判定拆除重建／综合整治，故分档年限一律作情景"
                         "参数报告，不得写成标定值")
    ap.add_argument("--quota", type=int, default=None,
                    help="年度立项配额，默认 3。敏感性方向 {2, 3, 4, 6}。改变可达"
                         "上界，故入分母缓存键")
    ap.add_argument("--tau-approval", type=float, default=None,
                    help="平均审批时长 τ_A（年）。给出后把逐年条件批准率整体换成"
                         "常数风险率 1/τ_A（几何近似），敏感性方向 {1, 3, 5}。"
                         "默认档是由累计获批数反解的实测向量，本参数是对照而非"
                         "更精确的估计")
    ap.add_argument("--budget-mode", default=None, choices=["upfront", "staged"],
                    help="资金口径。upfront=立项当年全额支付（默认，与既往结果逐位"
                         "等价）；staged=立项付前期款、余额在实施期内按年等额支付，"
                         "且 Cost 目标改按实际支付年计入。upfront 下失效单元花掉了"
                         "预算却从不进入 Cost（实测差额 4290/13470），staged 下"
                         "两者按构造相等")
    ap.add_argument("--stage-init", type=float, default=None,
                    help="staged 口径下立项当年支付的比例，默认 0.2")
    ap.add_argument("--gamma", type=float, default=None,
                    help="年度折现率，默认 0.95。敏感性方向 {0.90, 0.95, 0.926}；"
                         "0.926 对应财政部社会折现率 8%%")
    ap.add_argument("--horizon", type=int, default=15,
                    help="规划期长度 T。建设年限延长后可能需要放宽，见第 3 条诊断")
    ap.add_argument("--horizon-eval", default=None,
                    help="评价期 T_eval：第 T 年起不再立项、不再进钱，仅推进状态机"
                         "并结算建成年释放的目标，直到管道排空。默认 None=闭区间"
                         "口径（等于 T，即旧结果）。推荐 auto——按本组的 T/有效期/"
                         "建设年限自动取最短排空期（T=15 主组得 26，horizon20 组得 "
                         "31）；也可给整数强制指定。依据见 "
                         "docs/跨期结转_口径修正_v1.md")


def from_args(a):
    """从 argparse 结果取出 apply() 用的关键字字典。"""
    return dict(tau_valid=a.tau_valid, tau_ext=a.tau_ext,
                cooldown=a.cooldown, build_years=a.build_years,
                gamma=getattr(a, "gamma", None),
                build_years_by_channel=getattr(a, "build_years_by_channel", None),
                quota=getattr(a, "quota", None),
                tau_approval=getattr(a, "tau_approval", None),
                budget_mode=getattr(a, "budget_mode", None),
                stage_init=getattr(a, "stage_init", None))
