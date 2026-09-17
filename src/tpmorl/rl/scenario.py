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
    from tpmorl.objectives import reward as R
    from tpmorl.env import opportunity as O
    return dict(CELL_COST_MODE=R.CELL_COST_MODE,
                A_PLAN=O.A_PLAN, A_INFRA=O.A_INFRA, A_AGE=O.A_AGE,
                A_READY=O.A_READY, OBS_OPPORTUNITY=O.OBS_OPPORTUNITY,
                FORESIGHT=O.FORESIGHT, FIELD_SEED=O.FIELD_SEED,
                SHARE_NOW=O.SHARE_NOW, SHARE_RAMP=O.SHARE_RAMP,
                ONSET_LO_FRAC=O.ONSET_LO_FRAC, ONSET_HI_FRAC=O.ONSET_HI_FRAC,
                RAMP_YEARS=O.RAMP_YEARS, OPP_SHAPE=O.OPP_SHAPE,
                WINDOW_YEARS_LO=O.WINDOW_YEARS_LO,
                WINDOW_YEARS_HI=O.WINDOW_YEARS_HI,
                BUDGET=env_gym.BUDGET, CARRY_CAP=env_gym.CARRY_CAP,
                FAR_GROWTH=env_gym.FAR_GROWTH, GAMMA=env_gym.GAMMA,
                BUDGET_MODE=env_gym.BUDGET_MODE, STAGE_INIT=env_gym.STAGE_INIT,
                REWARD_SHAPING=env_gym.REWARD_SHAPING,
                OBS_LIFECYCLE=env_gym.OBS_LIFECYCLE,
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
    env_gym.OBS_LIFECYCLE = d["OBS_LIFECYCLE"]
    env_gym.REWARD_SHAPING = d["REWARD_SHAPING"]
    S.QUOTA, S.HAZARD = d["QUOTA"], tuple(d["HAZARD"])
    from tpmorl.objectives import reward as R
    R.CELL_COST_MODE = d["CELL_COST_MODE"]
    # 机会场的全部情景常量一并复原。漏一项就会发生本模块文档所说的静默继承：
    # 上一组开着 A_PLAN=0.4、这一组没设，实际仍在 0.4 下跑而 runs.json 记成 None。
    from tpmorl.env import opportunity as O
    O.A_PLAN, O.A_INFRA = d["A_PLAN"], d["A_INFRA"]
    O.A_AGE, O.A_READY = d["A_AGE"], d["A_READY"]
    O.OBS_OPPORTUNITY, O.FORESIGHT = d["OBS_OPPORTUNITY"], d["FORESIGHT"]
    O.FIELD_SEED = d["FIELD_SEED"]
    O.SHARE_NOW, O.SHARE_RAMP = d["SHARE_NOW"], d["SHARE_RAMP"]
    O.ONSET_LO_FRAC, O.ONSET_HI_FRAC = d["ONSET_LO_FRAC"], d["ONSET_HI_FRAC"]
    O.RAMP_YEARS = d["RAMP_YEARS"]
    O.OPP_SHAPE = d["OPP_SHAPE"]
    O.WINDOW_YEARS_LO = d["WINDOW_YEARS_LO"]
    O.WINDOW_YEARS_HI = d["WINDOW_YEARS_HI"]
    _HORIZON = _HORIZON_EVAL = None


def apply(budget=None, carry=None, growth=None,
          tau_valid=None, tau_ext=None, cooldown=None, build_years=None,
          horizon=None, gamma=None, horizon_eval=None,
          quota=None, tau_approval=None, build_years_by_channel=None,
          budget_mode=None, stage_init=None, obs_lifecycle=None,
          cell_cost_mode=None,
          a_plan=None, a_infra=None, a_age=None, a_ready=None,
          obs_opportunity=None, foresight=None, field_seed=None,
          share_now=None, share_ramp=None, onset_lo=None, onset_hi=None,
          ramp_years=None, reward_shaping=None, opp_shape=None,
          window_years=None):
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
        _chk_build_years(int(build_years))
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
        for c, b in d.items():
            _chk_build_years(int(b), f"通道 {c} 的")
        S.BUILD_YEARS_BY_CHANNEL = {c: int(d.get(c, S.BUILD_YEARS_BY_CHANNEL[c]))
                                    for c in S.BUILD_YEARS_BY_CHANNEL}
    if quota is not None:
        # 配额改变可达上界（一年最多能立几个项），分母不可跨配额复用 → 入 inst_tag()
        q = int(quota)
        if q < 1:
            raise ValueError(f"quota 必须 >= 1，收到 {quota}；配额 0 意味着永不立项，"
                             "整批结果退化为空策略，不是有意义的敏感性档")
        S.QUOTA = q
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
    if cell_cost_mode is not None:
        # 拆除基数口径。bytype 把一刀切的每格基数换成按**被拆除现状类别**取值的
        # 向量（均值保持，见 reward.cell_base_vector）。成本结构变了 → 可达上界变了
        # → **必须**入 inst_tag，否则对照组会静默复用 flat 档的分母。
        cm = str(cell_cost_mode).strip().lower()
        if cm not in ("flat", "bytype"):
            raise ValueError(f"cell_cost_mode 只能是 flat/bytype，收到 {cell_cost_mode}")
        from tpmorl.objectives import reward as R
        R.CELL_COST_MODE = cm
    # ---- 机会场（v17）----
    # 四个幅度都改变可达上界（价值乘子抬高 Floor 的上界；实施条件改变有多少立项
    # 能走到完工），**都已入 inst_tag**。观测开关与前瞻刻意不入键（参考策略不读
    # 观测，两档可达上界按构造相同，共用分母才能直接相减）——与 OBS_LIFECYCLE 同理。
    from tpmorl.env import opportunity as O
    if a_plan is not None:
        O.A_PLAN = _chk_amp(a_plan, "a_plan")
    if a_infra is not None:
        O.A_INFRA = _chk_amp(a_infra, "a_infra")
    if a_age is not None:
        O.A_AGE = _chk_amp(a_age, "a_age")
    if a_ready is not None:
        # 上界 1.0：hazard_mult = 1 + A_READY·(2·ready − 1)，A_READY>1 时
        # ready<0.5 的年份会给出负因子，被 clip 成 0（该年永不获批）。那不是
        # "条件差一点"，而是"行政上完全冻结"，是另一种机制，不在本参数的读法内。
        O.A_READY = _chk_amp(a_ready, "a_ready", hi=1.0)
    if obs_opportunity is not None:
        O.OBS_OPPORTUNITY = bool(obs_opportunity)
    if foresight is not None:
        f = int(foresight)
        if f < 0:
            raise ValueError(f"foresight 不得为负，收到 {foresight}")
        O.FORESIGHT = f
    if field_seed is not None:
        O.FIELD_SEED = int(field_seed)
    if share_now is not None or share_ramp is not None:
        sn = O.SHARE_NOW if share_now is None else float(share_now)
        sr = O.SHARE_RAMP if share_ramp is None else float(share_ramp)
        if not (0.0 <= sn <= 1.0 and 0.0 <= sr <= 1.0 and sn + sr <= 1.0):
            raise ValueError(f"share_now + share_ramp 须 <= 1 且各自在 [0,1]，"
                             f"收到 {sn} + {sr}；余量是\"一直不好\"型，不可为负")
        if sr <= 0.0:
            raise ValueError("share_ramp = 0 意味着没有任何\"会变好\"的地块，"
                             "机会场退化为平稳场，该档测不到择时")
        O.SHARE_NOW, O.SHARE_RAMP = sn, sr
    if onset_lo is not None:
        O.ONSET_LO_FRAC = float(onset_lo)
    if onset_hi is not None:
        O.ONSET_HI_FRAC = float(onset_hi)
    if onset_lo is not None or onset_hi is not None:
        if not 0.0 <= O.ONSET_LO_FRAC <= O.ONSET_HI_FRAC <= 1.0:
            raise ValueError(f"须 0 <= onset_lo <= onset_hi <= 1，"
                             f"收到 {O.ONSET_LO_FRAC} / {O.ONSET_HI_FRAC}")
    if opp_shape is not None:
        if str(opp_shape) not in ("rising", "window"):
            raise ValueError(f"opp_shape 只能是 rising/window，收到 {opp_shape!r}")
        O.OPP_SHAPE = str(opp_shape)
    if window_years is not None:
        lo, hi = (float(window_years[0]), float(window_years[1])
                  if len(window_years) > 1 else float(window_years[0]))
        if not 0 < lo <= hi:
            raise ValueError(f"window_years 须 0 < lo <= hi，收到 {window_years}")
        O.WINDOW_YEARS_LO, O.WINDOW_YEARS_HI = lo, hi
    if ramp_years is not None:
        ry = float(ramp_years)
        if ry <= 0:
            raise ValueError(f"ramp_years 必须为正，收到 {ry}")
        O.RAMP_YEARS = ry

    if reward_shaping is not None:
        # 势函数型整形的强度。**不入 inst_tag**：参考策略不读整形项，可达的目标
        # 上界按构造不变，故整形组与主组共用分母、标量化回报可直接相减。
        rs = float(reward_shaping)
        if not 0.0 <= rs <= 10.0:
            raise ValueError(f"reward_shaping 须在 [0, 10]，收到 {reward_shaping}")
        env_gym.REWARD_SHAPING = rs

    if obs_lifecycle is not None:
        # 消融开关。刻意**不**进 inst_tag：参考策略不读观测，两组分母按构造相同，
        # 共用缓存才能让标量化回报直接相减（见 env_gym.OBS_LIFECYCLE 的说明）。
        env_gym.OBS_LIFECYCLE = bool(obs_lifecycle)

    # "auto" 必须在**最后**解析：auto_horizon_eval() 读 TAU_VALID/TAU_EXT/
    # BUILD_YEARS，而这几个常量到上面几行才写入。放在函数开头会算出旧制度下的值。
    if _HORIZON_EVAL == "auto":
        if _HORIZON is None:
            raise ValueError("horizon_eval='auto' 需要同时给出 horizon")
        _HORIZON_EVAL = auto_horizon_eval(_HORIZON)


AMP_MAX = 5.0           # 机会场幅度的硬上界，见 _chk_amp


def _chk_amp(v, what, lo=0.0, hi=AMP_MAX):
    """机会场幅度的取值校验。

    下界 0 = 关闭。负幅度意味着"机会越好、价值越低"，与四个场的读法相反，
    真要做这个对照应当换一个明确命名的参数，而不是让主参数取负值静默反向。
    上界 AMP_MAX：幅度 5 时时机乘子可达 6 倍，交付价值的量级差异会压过全部
    空间目标，标量化奖励退化为单目标——那不是"择时更重要"，是分母失效。
    """
    x = float(v)
    if not lo <= x <= hi:
        raise ValueError(f"{what} 须在 [{lo:g}, {hi:g}]，收到 {v}")
    return x


BUILD_YEARS_MAX = 100   # 见 _chk_build_years


def _chk_build_years(b: int, what: str = ""):
    """建设年限的取值校验。

    下界：b >= 1。b = 0 时开工与完工同年，闭式与状态机虽仍自洽，但"零年建成"
    不是可报告的情景；b < 0 会给出与状态相符性相反的剩余建设年（实测）。
    上界：状态机的 clock 为 int16，理论上界远大于此，但建设年限超过规划期加评价期
    已无规划含义（一个都不会完工），继续放大只会得到全空的指标表。故取 100 年
    作硬上界，把"手误多打一位"挡在批次开跑之前，而不是跑完才发现指标全空。
    """
    if not (1 <= b <= BUILD_YEARS_MAX):
        raise ValueError(f"{what}build_years 必须在 [1, {BUILD_YEARS_MAX}]，收到 {b}")


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
    # v16：拆除基数口径。bytype 改变成本结构（贵单元更贵、便宜单元更便宜），
    # 可达上界随之变，故只在非出厂值时入键，flat 档的键与既有缓存逐字不变。
    from tpmorl.objectives import reward as R
    k = "" if R.CELL_COST_MODE == d.get("CELL_COST_MODE", R.CELL_COST_MODE) \
        else f"K{R.CELL_COST_MODE[0].upper()}"
    # v17：机会场。全关时 tag() 返回空串，既有缓存键逐字不变。
    from tpmorl.env import opportunity as O
    return f"V{S.TAU_VALID}E{S.TAU_EXT}{cooldown_tag()}Y{y}{t}{g}{q}{a}{m}{k}{O.tag()}"


def _hazard_tag(hz):
    """批准率向量的短标识。常数向量（τ_A 档）记其倒数，便于人读；
    非常数向量退回 4 位十六进制摘要，保证不同向量不会撞键。"""
    import hashlib
    if len(set(hz)) == 1 and hz[0] > 0:
        return f"{1.0 / hz[0]:g}".replace(".", "")
    return hashlib.md5(repr(hz).encode()).hexdigest()[:4]


def _describe_opp():
    """机会场的人读摘要。**始终声明这是情景参数**，不给\"看起来像标定值\"的机会。"""
    from tpmorl.env import opportunity as O
    if not O.active():
        return ("\n机会场 关（四个幅度全 0，动态与 v16 逐位等价）  "
                f"观测{'开' if O.OBS_OPPORTUNITY else '关（置零）'}")
    return (f"\n机会场 开（**情景参数，无实证标定**）  "
            f"幅度 规划={O.A_PLAN:g} 设施={O.A_INFRA:g} 老化={O.A_AGE:g} "
            f"实施条件={O.A_READY:g}\n"
            f"  三型构成 早{O.SHARE_NOW:.0%}/等{O.SHARE_RAMP:.0%}/"
            f"从不{1 - O.SHARE_NOW - O.SHARE_RAMP:.0%}  "
            f"爬升起始 {O.ONSET_LO_FRAC:.0%}T~{O.ONSET_HI_FRAC:.0%}T  "
            f"时间常数 {O.RAMP_YEARS:g} 年  片区 {O.INFRA_CLUSTERS}\n"
            f"  观测{'开' if O.OBS_OPPORTUNITY else '关（置零）'}  "
            f"前瞻 {O.FORESIGHT} 年  场种子 {O.FIELD_SEED}")


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
            + (f"\n势函数整形 {env_gym.REWARD_SHAPING:g}"
               "（不改变最优策略集；只进标量奖励，不入分母键）"
               if env_gym.REWARD_SHAPING else "")
            + _describe_opp()
            + ("" if env_gym.OBS_LIFECYCLE else
               "\n**消融：生命周期观测特征（16..21 共 6 维）已置零**"
               "（位宽不变；不入分母缓存键，与主组共用分母）")
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
    ap.add_argument("--cell-cost-mode", default=None, choices=["flat", "bytype"],
                    help="拆除基数口径。flat=每格同价（默认，与既往结果逐位等价）；"
                         "bytype=按被拆除的现状用地类别取值（住宅>商业>工业>农地>"
                         "生态），倍率按候选池构成**均值保持**归一，故两档总成本尺度"
                         "相同、差异纯为类型间再分配。**倍率是假设不是标定值**："
                         "单元表无投资额字段、平台无逐类补偿标准数据集")
    ap.add_argument("--no-obs-lifecycle", action="store_true",
                    help="消融：把生命周期观测特征（剩余有效年/剩余建设年/期望交付"
                         "年数/slack/期内可交付标志/在建管道占比，共 6 维）**置零**。"
                         "位宽与网络形状不变，只切断这一段信息，用于把效果归因到"
                         "该信息本身。**不进分母缓存键**——参考策略不读观测，本组与"
                         "主组分母按构造相同、共用缓存，故两组的标量化回报可直接相减"
                         "（这是批次内单因子消融，比跨批次对比可辩护得多）")
    # ---- 机会场（v17）。四个幅度默认 None=不改写，出厂值 0=关闭 ----
    ap.add_argument("--a-plan", type=float, default=None,
                    help="上位规划对**立项可及性** P_init 的调制幅度（读当年）。"
                         "v18 起走\"能不能报上去\"这条通道，不再乘交付价值："
                         "法定图则未覆盖、不在更新单元计划名单里的地块，报上去"
                         "本就进不了流程。实现为逐年重抽的随机准入而非硬阈值——"
                         "规划只提高被纳入的机会，不保证纳入也不永久禁入。"
                         "**情景参数，无实证标定**：单元表没有历年规划定位字段")
    ap.add_argument("--a-infra", type=float, default=None,
                    help="周边设施成熟度对交付价值的加成（读**建成年**取值）。"
                         "v18 起这是**唯一保留的价值通道**，因为它最好辩护："
                         "价值在交付时点实现，地铁通了房子才值那个钱。"
                         "观测给的是**公布**层（提前 INFRA_ANNOUNCE_LEAD 年），"
                         "价值只认**建成**层——规划先产生预期，建成才兑现。"
                         "情景参数：无设施投用年份数据")
    ap.add_argument("--a-age", type=float, default=None,
                    help="更新必要性 Need 的幅度。**与建议的一处明示偏离**：建议把"
                         "Need 列为独立机制，但本环境里必要性没有独立作用位"
                         "（既非准入、也不改变客观收益），故并入**批准风险率**："
                         "越必要的项目在审批与协调中越顺利。引用建议 §9 时须按此标注。"
                         "情景参数：单元表无建成年份字段，基期年龄按分布抽样")
    ap.add_argument("--a-ready", type=float, default=None,
                    help="实施条件对**批准风险率** P_approve 的调制幅度（读当年），"
                         "在 [0,1]。报上去之后能不能批下来、推得动，与能不能报"
                         "上去（--a-plan）是两条通道。ready=0.5 处调制为中性")
    ap.add_argument("--no-obs-opportunity", action="store_true",
                    help="消融：把机会场四维观测特征（22..25）**置零**，位宽不变。"
                         "场仍作用于奖励与转移，只是策略看不见——用于分离\"信息\"与"
                         "\"机制\"。**不进分母缓存键**，与主组共用分母")
    ap.add_argument("--foresight", type=int, default=None,
                    help="观测里给出 t+K 年的机会场取值（K 年前瞻），默认 0=只看当期。"
                         "对应\"法定图则与设施计划已公布\"的信息档。同样不入分母键")
    ap.add_argument("--field-seed", type=int, default=None,
                    help="机会场的抽样种子（与回合种子无关：场在所有回合里是同一张图）。"
                         "换种子等于换一份规划安排，入分母缓存键")
    ap.add_argument("--share-now", type=float, default=None,
                    help="\"现在就好、以后不变\"型地块的占比，默认 0.30")
    ap.add_argument("--share-ramp", type=float, default=None,
                    help="\"现在一般、若干年后变好\"型的占比，默认 0.40。"
                         "余量为\"一直不好\"型——这一型必须存在，否则\"一直等\""
                         "就是最优解，退化解会冒充择时能力")
    ap.add_argument("--onset-lo", type=float, default=None,
                    help="爬升起始年的下界，按决策期 T 的比例给出，默认 0.15。"
                         "用比例而非年份，使 T=15 与 T=25 两档的场在相对时序上可比")
    ap.add_argument("--onset-hi", type=float, default=None,
                    help="爬升起始年的上界（T 的比例），默认 0.60")
    ap.add_argument("--opp-shape", default=None, choices=["rising", "window"],
                    help="机会曲线形状。rising=单调爬升（v17 口径，低→高→一直高）；"
                         "window=有限机会窗（低→升→峰→回落）。"
                         "单调场下最优立项年只有\"立刻\"或\"能拖多久拖多久\"两种，"
                         "P0 诊断实测加大幅度时\"拖满\"占比从 0.13 升到 0.31——"
                         "环境奖励的是无脑延后。有限窗让\"等太久\"由机会场自然惩罚")
    ap.add_argument("--window-years", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="逐单元机会窗宽度的抽样区间（年），默认 3 7。"
                         "对应\"全局 25 年期限 + 局部 3~7 年机会窗\"")
    ap.add_argument("--ramp-years", type=float, default=None,
                    help="逻辑斯蒂爬升的时间常数（年），默认 3。刻意不做成阶跃："
                         "机会改善在现实中是逐渐的，硬阈值会把择时变成查表")
    ap.add_argument("--reward-shaping", type=float, default=None,
                    help="势函数型奖励整形的强度（PBRS，Ng et al. 1999），默认 0=关闭。"
                         "**不改变最优策略集**，只把\"等待\"的梯度从机械扣分变成中性；"
                         "只加在标量奖励上，不进 11 维目标的落盘值，**不入分母缓存键**。"
                         "门槛实验里它是唯一把折现回报做到穷举最优 95%% 以上的一档"
                         "（3/3 种子），但并未把立项年推后，故作实验因子而非既定修复")
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
                stage_init=getattr(a, "stage_init", None),
                cell_cost_mode=getattr(a, "cell_cost_mode", None),
                # 未加 --no-obs-lifecycle 时传 None（不改写），而不是传 True：
                # apply() 对 None 一律不改写，出厂值由 reset() 保证为 True。
                # 传 True 会让"未指定"与"显式开启"在日志里无法区分。
                obs_lifecycle=(False if getattr(a, "no_obs_lifecycle", False)
                               else None),
                reward_shaping=getattr(a, "reward_shaping", None),
                a_plan=getattr(a, "a_plan", None),
                a_infra=getattr(a, "a_infra", None),
                a_age=getattr(a, "a_age", None),
                a_ready=getattr(a, "a_ready", None),
                foresight=getattr(a, "foresight", None),
                field_seed=getattr(a, "field_seed", None),
                share_now=getattr(a, "share_now", None),
                share_ramp=getattr(a, "share_ramp", None),
                onset_lo=getattr(a, "onset_lo", None),
                onset_hi=getattr(a, "onset_hi", None),
                ramp_years=getattr(a, "ramp_years", None),
                opp_shape=getattr(a, "opp_shape", None),
                window_years=getattr(a, "window_years", None),
                # 与 obs_lifecycle 同一处理：未加开关时传 None（不改写），
                # 而不是传 True——那会让"未指定"与"显式开启"在日志里无法区分。
                obs_opportunity=(False if getattr(a, "no_obs_opportunity", False)
                                 else None))
