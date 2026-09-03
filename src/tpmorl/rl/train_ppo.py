"""train_ppo.py — 掩码指针策略 + PPO，扫权重求 Pareto 前沿。

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

from tpmorl.env.schedule import QUOTA
from tpmorl.objectives.reward import OBJ_NAMES
from tpmorl.rl import env_gym            # 需按模块引用，才能在 main 里改 BUDGET
from tpmorl.rl.env_gym import RenewalEnv, N_PAIR_FEAT

DEV = "cpu"


class Pointer(nn.Module):
    def __init__(self, nf=N_PAIR_FEAT, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.score = nn.Linear(h, 1)
        self.val = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, F):
        z = self.enc(F)
        return self.score(z).squeeze(-1), self.val(z.mean(0)).squeeze(-1)


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
        picks, lp, ent = sample_action(logits, meta, cost, b, QUOTA, greedy,
                                       units_t=units_t, cost_t=cost_t)
        _, r, done, info = env.step([meta[i] for i in picks])
        tr["lp"].append(lp.detach()); tr["v"].append(float(v)); tr["r"].append(r)
        tr["ent"].append(ent); tr["vec"].append(info["vec"])
        tr["X"].append(Xt); tr["meta"].append(meta); tr["picks"].append(picks)
        tr["cost"].append(cost); tr["budget"].append(b)
        tr["units_t"].append(units_t); tr["cost_t"].append(cost_t)
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
          ent_c=0.01, vf_c=0.5, seed=0):
    torch.manual_seed(seed)
    net = Pointer().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        buf, RS = [], []
        for e in range(eps_per_iter):
            tr = run_episode(env, net, seed=seed * 1000 + it * 10 + e)
            adv, ret = gae(np.array(tr["r"]), np.array(tr["v"]), env.gamma)
            for t in range(env.T):
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
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        hist.append(float(np.mean(RS)))
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
                                       spent=env.spent_hist[t],
                                       stopped=int(any(meta[i][0] < 0 for i in picks))))
                for i in real:
                    u, tg = meta[i]
                    record.append(dict(ep=e, year=t, unit=u,
                                       channel=int(env.ch[u]), target=tg,
                                       n_cells=int(env.ncell[u]),
                                       cost=env.pair_cost(u, tg),
                                       budget_before=env.budget_hist[t],
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
        for t in range(env.T):
            X, meta, cost, units = env.pairs()
            act, used, left = [], set(), env.budget
            order = rng.permutation(len(meta))
            for i in order:
                if len(act) >= QUOTA:
                    break
                u = meta[i][0]
                if u < 0 or u in used or cost[i] > left + 1e-6:
                    continue
                act.append(meta[i]); used.add(u); left -= float(cost[i])
            _, r, done, info = env.step(act)
            g += (env.gamma ** t) * info["vec"] * env.scale
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
    scenario.apply(budget=budget, carry=carry, growth=growth, **inst)
    print(scenario.describe())
    # 分母取**当前约束情景**下参考策略集的可达上界（见 tpmorl/rl/scale.py 模块文档）。
    # 旧做法用 reward_v0/discounted_return.csv（无约束情景），失真跨度约 24 倍且方向不一致。
    from tpmorl.rl.scale import load_scale
    scale = load_scale(ds, env_gym.BUDGET, env_gym.CARRY_CAP, env_gym.FAR_GROWTH)

    rows, curves = [], {}
    e0 = RenewalEnv(ds, T=horizon, weights=weight_vector(0.5), scale=scale)
    gr = eval_random(e0)
    rows.append(dict(alpha=-1.0, **{k: gr[i] for i, k in enumerate(OBJ_NAMES)}))
    print(f"随机(同动作空间)  Floor={gr[OBJ_NAMES.index('Floor')]/1e4:8.0f}万㎡  "
          f"Gdp={gr[OBJ_NAMES.index('Gdp')]:8.0f}  Eco={gr[OBJ_NAMES.index('Eco')]:9.0f}")

    for a in alphas:
        t0 = time.time()
        env = RenewalEnv(ds, T=horizon, weights=weight_vector(a), scale=scale)
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
                   alphas=list(alphas), quota=QUOTA,
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
