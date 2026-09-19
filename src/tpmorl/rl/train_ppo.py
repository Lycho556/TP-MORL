"""train_ppo.py — 掩码指针策略 + PPO，扫权重得**权重扫描解集**。

术语约定（自评 v2 §P0-4）：**不得**称"Pareto 前沿"。加权和扫权重只能得到凸包上的
支撑解，且实测在 (Gdp, Floor) 投影上 7 个权重档有 6 个被其他档支配——它不是前沿。
输出文件名 `pareto_front.csv` 因下游脚本已依赖而保留，但论文与图表一律称"权重扫描解集"。

策略结构
    每年对全部 717 个单元打分 -> 用合规掩码屏蔽 -> 无放回自回归采样 QUOTA 个立项。
    这是指针网络式的组合动作，log-prob 为各次采样之和。掩码保证只在法定可动集合内采样，
    因此策略永远不会输出违反有效期/配额/冷却期的方案——约束是硬的，不靠惩罚项软化。

目标尺度归一化
    11 个目标量级相差 5 个数量级（Floor ~1e7 而 Cost ~1e2），不归一化则加权和退化为单目标。
    归一化因子取三个基线折扣回报的逐目标绝对值上界（`discounted_return.csv`），
    即"基线能达到的量级"，而非人工设定。

用法: PYTHONPATH=src python -m tpmorl.rl.train_ppo --iters 60
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn

from tpmorl.env import schedule as _SCH   # 按模块引用：情景会改写其常量
from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl import env_gym            # 需按模块引用，才能在 main 里改 BUDGET
from tpmorl.rl.env_gym import RenewalEnv, N_PAIR_FEAT

DEV = "cpu"


class Pointer(nn.Module):
    """掩码指针：对每个 (单元, 目标功能) 配对打分，末行是「今年到此为止」。

    ## stop_context：为什么需要它（门槛三的诊断结论）

    默认结构里 actor 是**逐行独立**打分的：

        score_i = score(enc(F_i))

    而 critic 拿到的是池化后的整集合信息：

        v = val( mean_i enc(F_i) )

    「到此为止」在 actor 眼里只是一行**普通候选**，它自己的特征里只有全局年份、
    停止标志与预算，**没有"所有地块此刻的时序状态如何"这个信息**。于是 actor
    做的其实是

        Score(STOP)  vs  Score(A)

    而不是

        Q(s, 等)     vs  Q(s, 做 A)

    这与实测吻合：训练后"动手"对"到此为止"的 logit 差约 8 个单位，而地块之间
    只差 0.04 —— 策略在"做哪个"上几乎没有分辨力，在"做不做"上却有一个很强的
    全局偏置。它会比较 A/B/C 谁更值得做，但不真正表达"此刻整个候选集合都不值得
    动，应当持有"。

    `stop_context=True` 时改为：先看完所有候选，再决定今年是否动手 ——

        c       = mean_i enc(F_i)              候选集合的池化表示
        score_i = w([enc(F_i), c])             普通候选：自身 + 集合上下文
        score_S = w_stop([enc(F_S), c])        停止行：**单独的头**

    停止行单列一个头，是因为"今年不动"与"做某个地块"根本不是同一类量，
    共用一个线性头会强迫它们落在同一个打分尺度上。

    默认 False 时，模块的构造顺序与参数量与旧版**完全一致**，故同种子下权重
    初始化逐位相同，历史结果不受影响。
    """

    def __init__(self, nf=N_PAIR_FEAT, h=64, stop_context=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.score = nn.Linear(h, 1)
        self.val = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.stop_context = bool(stop_context)
        if self.stop_context:
            # 只在开启时构造，故关闭时的初始化抽样序列与旧版逐位相同
            self.score_ctx = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(),
                                           nn.Linear(h, 1))
            self.score_stop = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(),
                                            nn.Linear(h, 1))

    def forward(self, F):
        """F 形状 (行数, 特征数)，**末行约定为「到此为止」**（env 与门槛世界同约定）。"""
        z = self.enc(F)
        c = z.mean(0)
        if not self.stop_context:
            return self.score(z).squeeze(-1), self.val(c).squeeze(-1)
        zc = torch.cat([z, c.detach().unsqueeze(0).expand_as(z)], dim=-1)
        s = self.score_ctx(zc).squeeze(-1)
        s = torch.cat([s[:-1], self.score_stop(zc[-1:]).squeeze(-1)], dim=0)
        return s, self.val(c).squeeze(-1)


def _step_mask(units_t, cost_t, chosen_units, left):
    """双重掩码：已选单元的所有行屏蔽（一单元一年一次），付不起的行屏蔽。

    「到此为止」行成本为 0，故永远可选——这是让「等」成为一个真实动作的地方。

    向量化实现：原先对最多 ~1985 个配对做 Python 逐行循环，而每次选取都要重算一遍，
    是训练的首要开销（cProfile 下占 37% tottime）。改为张量运算，语义等价——
    同种子下 Gdp/Eco/Floor 与旧实现逐位一致（Floor 相对误差 3e-7，来自累加次序）。

    实测加速**依配置而异**，因为旧实现的开销随候选集大小变化，而候选集在策略推迟时不收缩：
      - 无增长 α=0：  170s → 71s / 60 迭代（2.83 → 1.19 s/迭代，2.4×）
      - 年增5% α=0：  1148s → ~71s（19.1 → 1.19 s/迭代，16×）
      - 年增5% α=0.5：3007s → 71s（50.1 → 1.18 s/迭代，42×）
    要点不是倍数，而是**每迭代耗时不再依赖策略行为**，一律 ~1.2s，使多种子扫描可行。
    """
    m = cost_t <= left + 1e-6
    for u in chosen_units:                 # 每年至多 QUOTA 个，循环极短
        m &= units_t != u
    return m


def sample_action(logits, meta, cost, budget, k, greedy=False, units_t=None, cost_t=None):
    """在 (单元,目标) 对上自回归采样至多 k 个；选中 STOP 则当年结束。"""
    if units_t is None:
        units_t = torch.as_tensor(np.asarray([u for u, _ in meta], dtype=np.int64))
    if cost_t is None:
        cost_t = torch.as_tensor(np.asarray(cost, dtype=np.float64))
    picks, lp, ent, used = [], torch.zeros(()), torch.zeros(()), set()
    left = float(budget)
    for _ in range(k):
        m = _step_mask(units_t, cost_t, used, left)
        if not m.any():
            break
        z = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=z)
        a = torch.argmax(z) if greedy else d.sample()
        i = int(a)
        lp = lp + d.log_prob(a); ent = ent + d.entropy()
        if meta[i][0] < 0:                     # STOP：把余额留到明年
            picks.append(i); break
        picks.append(i); used.add(meta[i][0]); left -= float(cost[i])
    return picks, lp, ent


def logprob_of(logits, meta, cost, budget, picks, units_t=None, cost_t=None):
    """重算一组已选动作的 log-prob 与熵（PPO 多轮复用需要）。

    必须逐次复现与采样时相同的掩码序列，包括资金余额的递减——否则重要性比失真。
    """
    if units_t is None:
        units_t = torch.as_tensor(np.asarray([u for u, _ in meta], dtype=np.int64))
    if cost_t is None:
        cost_t = torch.as_tensor(np.asarray(cost, dtype=np.float64))
    lp, ent, used = torch.zeros(()), torch.zeros(()), set()
    left = float(budget)
    for a in picks:
        m = _step_mask(units_t, cost_t, used, left)
        z = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=z)
        lp = lp + d.log_prob(torch.tensor(a)); ent = ent + d.entropy()
        if meta[a][0] < 0:
            break
        used.add(meta[a][0]); left -= float(cost[a])
    return lp, ent


def run_episode(env, net, greedy=False, seed=None):
    env.reset(seed=seed)
    tr = dict(lp=[], v=[], r=[], ent=[], vec=[], X=[], meta=[], picks=[],
              cost=[], budget=[], units_t=[], cost_t=[])
    for t in range(env.T):
        X, meta, cost, units = env.pairs()
        Xt = torch.as_tensor(X)
        # 每年只建一次掩码用张量；PPO 多轮复用时直接复用，不重复转换
        units_t = torch.as_tensor(units)
        cost_t = torch.as_tensor(cost)      # 保持 float64：与旧实现的可负担性比较逐位一致
        b = env.budget
        with torch.no_grad():
            logits, v = net(Xt)
        picks, lp, ent = sample_action(logits, meta, cost, b, env.quota, greedy,
                                       units_t=units_t, cost_t=cost_t)
        _, r, done, info = env.step([meta[i] for i in picks])
        tr["lp"].append(lp.detach()); tr["v"].append(float(v)); tr["r"].append(r)
        tr["ent"].append(ent); tr["vec"].append(info["vec"])
        tr["X"].append(Xt); tr["meta"].append(meta); tr["picks"].append(picks)
        tr["cost"].append(cost); tr["budget"].append(b)
        tr["units_t"].append(units_t); tr["cost_t"].append(cost_t)
    # 尾部评价期（T_eval > T 时才进入）：不调用策略、不立项、不进钱，
    # 只推进状态机并结算建成年释放的 11 个目标。
    # 尾部的折现奖励折回最后一个决策步，这样 gae 的回报口径仍然正确，
    # 而轨迹长度保持 T——buf/gae/采样器/优势归一化全都无需改动。
    # vec 则按真实年份 append，evaluate() 里的 gamma**t 会自动给出正确折现。
    tail = 0.0
    for k in range(1, env.T_eval - env.T + 1):
        _, r, done, info = env.step([])
        tail += (env.gamma ** k) * r
        tr["vec"].append(info["vec"])
        if done:
            break
    if tail and tr["r"]:
        tr["r"][-1] += tail
    if env.T_eval > env.T:
        # 不依赖 auto_horizon_eval() 的推导正确：直接查管道是否真排空。
        # S1/S2/S3 还有存量就说明 T_eval 取短了，晚立项仍被截断——当场失败，
        # 而不是跑完 10 小时才在结果里发现。
        from tpmorl.rl.scenario import auto_horizon_eval
        open_ = int(((env.env.sigma >= 1) & (env.env.sigma <= 3)).sum())
        # 显式 raise 而非裸 assert：这一条是防"评价期取短"的唯一保险，若在
        # python -O 下失效，晚立项会被静默截断而结果看不出异常。
        if open_ != 0:
            raise AssertionError(
                f"评价期 T_eval={env.T_eval} 结束时仍有 {open_} 个单元在管道中"
                f"（S1/S2/S3），晚立项仍被截断。应取 {auto_horizon_eval(env.T)}")
    return tr


def gae(r, v, gamma=0.95, lam=0.95):
    adv, g = np.zeros(len(r)), 0.0
    vv = np.append(v, 0.0)
    for t in reversed(range(len(r))):
        d = r[t] + gamma * vv[t + 1] - vv[t]
        g = d + gamma * lam * g
        adv[t] = g
    return adv, adv + v


def train(env, iters=60, eps_per_iter=4, epochs=4, lr=3e-3, clip=0.2,
          stop_context=False,
          ent_c=0.01, vf_c=0.5, seed=0, drop_no_choice=False,
          advantage="gae", init_actor=None, actor_lr_scale=1.0,
          freeze_actor_iters=0, callback=None):
    torch.manual_seed(seed)
    net = Pointer(stop_context=stop_context).to(DEV)
    if init_actor is not None:
        # **唯一的改动点：actor 的初始化。** 环境、奖励、动作空间、特征、
        # PPO 超参一律不动 —— 这样若结果变好，可以干净地归因到初始化。
        #
        # 只加载 enc + score（actor），**不加载 val（critic）**：critic 的目标
        # 依赖策略本身，用监督预训练的表示去初始化它没有意义，而且会把
        # "actor 变好"与"critic 变好"两件事混在一起。
        sd = (init_actor if isinstance(init_actor, dict)
              else torch.load(init_actor, weights_only=False))
        sd = sd.get("state", sd)
        keep = {k: v for k, v in sd.items()
                if k.startswith("enc.") or k.startswith("score.")}
        missing = net.load_state_dict(keep, strict=False)
        if any(k.startswith(("enc.", "score.")) for k in missing.missing_keys):
            raise ValueError(f"actor 权重未能完整加载：{missing.missing_keys}")
        print(f"    [init] 已载入预训练 actor（{len(keep)} 个张量），critic 保持随机")
    if callback is not None:
        # 第 0 点 = 初始化本身。**必须在加载 init_actor 之后**，
        # 否则第 0 点求到的是随机网络，整条曲线的基准线就错了。
        callback(0, net)
    if actor_lr_scale == 1.0:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        # 给 actor（enc + score）单独的学习率。用于预训练初始化后的情形：
        # 实测 BC 预训练的 actor 在原学习率下会被 PPO 迅速推回近视解
        # （小世界档 A：BC 自己 0.932、带 0.14 的停止频率；经 PPO 后精确退回
        # 近视贪心的 0.900、停止频率 0.00）。调小 actor 步长是"保住已学到的
        # 空间选择、只让时序部分继续动"最直接的手段。
        act = (list(net.enc.parameters()) + list(net.score.parameters())
               + [p for n, p in net.named_parameters()
                  if n.startswith("score_")])
        act_ids = {id(p) for p in act}
        rest = [p for p in net.parameters() if id(p) not in act_ids]
        opt = torch.optim.Adam([dict(params=act, lr=lr * float(actor_lr_scale)),
                                dict(params=rest, lr=lr)])
    hist = []
    for it in range(iters):
        buf, RS = [], []
        for e in range(eps_per_iter):
            tr = run_episode(env, net, seed=seed * 1000 + it * 10 + e)
            if advantage == "gae":
                adv, ret = gae(np.array(tr["r"]), np.array(tr["v"]), env.gamma)
            elif advantage == "mc":
                # 精确蒙特卡洛回报，**不用 critic 做基线**（基线改为批内均值，
                # 在下面的 A 归一化里自动完成）。
                #
                # 为什么要有这一档：critic 读的是候选行的**均值池化** val(z.mean(0))。
                # 地块被逐个消耗后，均值分不清"还剩三个、都在窗口中段"与
                # "只剩一个、正在峰值"——而这恰好是判断"今年该不该等"所需的信息。
                # 若换成精确回报后择时行为出现，则偏差来自值函数基线，
                # 而不是 actor 结构、动作空间或环境。
                r = np.asarray(tr["r"], float)
                ret = np.zeros_like(r); run = 0.0
                for i in range(len(r) - 1, -1, -1):
                    run = r[i] + env.gamma * run
                    ret[i] = run
                adv = ret.copy()
            else:
                raise ValueError(f"advantage 只能是 gae/mc，收到 {advantage!r}")
            for t in range(env.T):
                # 无决策步（当年只有「到此为止」一个合法动作）可选择剔除。
                #
                # 为什么这不是"挑数据"而是修一处口径错误：这些步只有一个合法动作，
                # log-prob 恒为 0、重要性比恒为 1，**本来就不产生任何策略梯度**；
                # 但它们照样进入优势归一化的均值与标准差，也照样进值函数损失。
                # 门槛世界实测：训练到后期它们占到缓冲区的 47%，优势均值 −0.93，
                # 把"主动等"（优势 +0.41，方向正确）整个压到"立项"（+0.84）之下。
                # 也就是说策略不是学到了"等不好"，而是被一堆"无事可做"的年份带偏了。
                # 真实环境里 717 个单元、候选集几乎不会空，故这一项默认关闭、
                # 只在小世界诊断与名额紧张的情景里开。
                if drop_no_choice and len(tr["meta"][t]) <= 1:
                    continue
                buf.append((tr["X"][t], tr["meta"][t], tr["picks"][t],
                            tr["lp"][t], adv[t], ret[t],
                            tr["cost"][t], tr["budget"][t],
                            tr["units_t"][t], tr["cost_t"][t]))
            RS.append(sum(tr["r"]))
        A = np.array([b[4] for b in buf], dtype=np.float32)
        A = (A - A.mean()) / (A.std() + 1e-8)

        for _ in range(epochs):
            pl = vl = el = 0.0
            opt.zero_grad()
            for i, (X, meta, picks, lp_old, _, ret, cost, bdg, ut, ct) in enumerate(buf):
                logits, v = net(X)
                lp, ent = logprob_of(logits, meta, cost, bdg, picks,
                                     units_t=ut, cost_t=ct)
                ratio = torch.exp(lp - lp_old)
                a = torch.tensor(A[i])
                pl = pl + (-torch.min(ratio * a,
                                      torch.clamp(ratio, 1 - clip, 1 + clip) * a))
                vl = vl + (v - torch.tensor(ret, dtype=torch.float32)) ** 2
                el = el + ent
            n = len(buf)
            ((pl + vf_c * vl - ent_c * el) / n).backward()
            if it < int(freeze_actor_iters):
                # 冻结 actor 的前若干迭代：只让 critic 先拟合预训练策略的回报，
                # 避免"critic 还是随机的"时候用噪声优势去改写已学好的 actor。
                for nm, pp in net.named_parameters():
                    if (nm.startswith("enc.") or nm.startswith("score")) \
                            and pp.grad is not None:
                        pp.grad.zero_()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        hist.append(float(np.mean(RS)))
        if callback is not None:
            # 沿途求值用。一次训到最长迭代数、在检查点上求值，
            # 与"每个检查点各训一次"是同一条轨迹，但只付一次训练成本。
            callback(it + 1, net)
    return net, hist


def evaluate(env, net, n_ep=5, record=None):
    """贪心评估，返回 11 维折扣回报（乘回 scale 还原为原始量纲）。

    record 非空时同时把每年选中的 (单元,通道,目标) 写入该列表，用于分析策略学到了什么。
    """
    V = []
    for e in range(n_ep):
        tr = run_episode(env, net, greedy=True, seed=90000 + e)
        g = np.zeros(len(OBJ_NAMES))
        for t, vec in enumerate(tr["vec"]):
            g += (env.gamma ** t) * vec * env.scale
        V.append(g)
        if record is not None:
            for t, (meta, picks) in enumerate(zip(tr["meta"], tr["picks"])):
                # 当年一个单元都没立项 = 主动等待（含选 STOP 与付不起两种情形）
                real = [i for i in picks if meta[i][0] >= 0]
                if not real:
                    record.append(dict(ep=e, year=t, unit=-1, channel=0, target=-1,
                                       n_cells=0, cost=0.0,
                                       budget_before=env.budget_hist[t],
                                       released=env.released_hist[t],
                                       spent=env.spent_hist[t],
                                       stopped=int(any(meta[i][0] < 0 for i in picks))))
                for i in real:
                    u, tg = meta[i]
                    record.append(dict(ep=e, year=t, unit=u,
                                       channel=int(env.ch[u]), target=tg,
                                       n_cells=int(env.ncell[u]),
                                       cost=env.pair_cost(u, tg),
                                       budget_before=env.budget_hist[t],
                                       released=env.released_hist[t],
                                       spent=env.spent_hist[t],
                                       stopped=int(any(meta[j][0] < 0 for j in picks))))
    return np.mean(V, 0)


def eval_random(env, n_ep=5, seed=0):
    """同一 (单元,目标) 动作空间内的均匀随机策略——与前沿同底可比的参照。"""
    rng = np.random.default_rng(seed)
    V = []
    for e in range(n_ep):
        env.reset(seed=90000 + e)
        g = np.zeros(len(OBJ_NAMES))
        for t in range(env.T_eval):
            act = []
            if t < env.T:          # 尾部评价年不立项，与策略侧口径一致
                X, meta, cost, units = env.pairs()
                used, left = set(), env.budget
                order = rng.permutation(len(meta))
                for i in order:
                    if len(act) >= env.quota:
                        break
                    u = meta[i][0]
                    if u < 0 or u in used or cost[i] > left + 1e-6:
                        continue
                    act.append(meta[i]); used.add(u); left -= float(cost[i])
            _, r, done, info = env.step(act)
            g += (env.gamma ** t) * info["vec"] * env.scale
            if done:
                break
        V.append(g)
    return np.mean(V, 0)


def weight_vector(alpha):
    """alpha=1 全押经济/交付，alpha=0 全押生态/宜居；成本项始终受罚。"""
    w = {k: 0.10 for k in OBJ_NAMES}
    w.update(Gdp=alpha, Emp=alpha, Floor=alpha,
             Eco=1 - alpha, Aec=1 - alpha,
             Cost=0.20, Disrupt=0.20, Expire=0.20)
    return np.array([w[k] for k in OBJ_NAMES])


def main(ds, out, iters, alphas, budget=None, carry=None, growth=None,
         horizon=15, **inst):
    os.makedirs(out, exist_ok=True)
    from tpmorl.rl import scenario
    scenario.apply(budget=budget, carry=carry, growth=growth,
                   horizon=horizon, **inst)
    print(scenario.describe())
    # 分母取**当前约束情景**下参考策略集的可达上界（见 tpmorl/rl/scale.py 模块文档）。
    # 旧做法用 reward_v0/discounted_return.csv（无约束情景），失真跨度约 24 倍且方向不一致。
    from tpmorl.rl.scale import load_scale
    from tpmorl.rl import scenario as _scen
    scale = load_scale(ds, env_gym.BUDGET, env_gym.CARRY_CAP, env_gym.FAR_GROWTH)
    # 这条 CLI 路径不经 exp_opt_quality，也要跟随 scenario 登记的评价期，
    # 否则同一份代码从两个入口跑出两种口径。未登记时返回 None = 闭区间。
    _te = _scen.horizon_eval()

    rows, curves = [], {}
    e0 = RenewalEnv(ds, T=horizon, T_eval=_te,
                    weights=weight_vector(0.5), scale=scale)
    gr = eval_random(e0)
    rows.append(dict(alpha=-1.0, **{k: gr[i] for i, k in enumerate(OBJ_NAMES)}))
    print(f"随机(同动作空间)  Floor={gr[OBJ_NAMES.index('Floor')]/1e4:8.0f}万㎡  "
          f"Gdp={gr[OBJ_NAMES.index('Gdp')]:8.0f}  Eco={gr[OBJ_NAMES.index('Eco')]:9.0f}")

    for a in alphas:
        t0 = time.time()
        env = RenewalEnv(ds, T=horizon, T_eval=_te,
                         weights=weight_vector(a), scale=scale)
        net, hist = train(env, iters=iters)
        rec = []
        g = evaluate(env, net, record=rec)
        pd.DataFrame(rec).to_csv(os.path.join(out, f"selections_alpha{a:g}.csv"),
                                 index=False, encoding="utf-8-sig")
        rows.append(dict(alpha=a, **{k: g[i] for i, k in enumerate(OBJ_NAMES)}))
        curves[f"alpha={a}"] = hist
        print(f"alpha={a:.2f}  {time.time()-t0:5.0f}s  "
              f"Floor={g[OBJ_NAMES.index('Floor')]/1e4:8.0f}万㎡  "
              f"Gdp={g[OBJ_NAMES.index('Gdp')]:8.0f}  Eco={g[OBJ_NAMES.index('Eco')]:9.0f}  "
              f"最终回报={hist[-1]:.3f}")

    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(out, "pareto_front.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(os.path.join(out, "learning_curves.csv"),
                                index_label="iter", encoding="utf-8-sig")
    json.dump(dict(scale=dict(zip(OBJ_NAMES, scale.tolist())), iters=iters,
                   alphas=list(alphas), quota=int(_SCH.QUOTA),
                   budget=env_gym.BUDGET, carry_cap=env_gym.CARRY_CAP,
                   far_growth=env_gym.FAR_GROWTH),
              open(os.path.join(out, "train_config.json"), "w"), ensure_ascii=False, indent=1)
    print("\n" + P.round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="data/processed/gm_dataset_v1/rl_v0")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--carry", type=float, default=None)
    ap.add_argument("--growth", type=float, default=None)
    from tpmorl.rl import scenario
    scenario.add_args(ap)
    a = ap.parse_args()
    main(a.dataset, a.out, a.iters, a.alphas, a.budget, a.carry, a.growth,
         horizon=a.horizon, **scenario.from_args(a))
