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


def _step_mask(meta, chosen_units, n):
    """屏蔽已选单元的所有 (单元,目标) 行——一个单元一年只能立一次项。"""
    m = torch.ones(n, dtype=torch.bool)
    for i, (u, _) in enumerate(meta):
        if u in chosen_units:
            m[i] = False
    return m


def sample_action(logits, meta, k, greedy=False):
    """在 (单元,目标) 对上自回归采样至多 k 个，单元不重复。"""
    n = len(meta)
    picks, lp, ent, used = [], torch.zeros(()), torch.zeros(()), set()
    for _ in range(k):
        m = _step_mask(meta, used, n)
        if not m.any():
            break
        z = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=z)
        a = torch.argmax(z) if greedy else d.sample()
        picks.append(int(a)); used.add(meta[int(a)][0])
        lp = lp + d.log_prob(a); ent = ent + d.entropy()
    return picks, lp, ent


def logprob_of(logits, meta, picks):
    """重算一组已选动作的 log-prob 与熵（PPO 多轮复用需要）。"""
    n = len(meta)
    lp, ent, used = torch.zeros(()), torch.zeros(()), set()
    for a in picks:
        m = _step_mask(meta, used, n)
        z = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=z)
        lp = lp + d.log_prob(torch.tensor(a)); ent = ent + d.entropy()
        used.add(meta[a][0])
    return lp, ent


def run_episode(env, net, greedy=False, seed=None):
    env.reset(seed=seed)
    tr = dict(lp=[], v=[], r=[], ent=[], vec=[], X=[], meta=[], picks=[])
    for t in range(env.T):
        X, meta = env.pairs()
        Xt = torch.as_tensor(X)
        if len(meta) == 0:
            _, r, done, info = env.step([])
            tr["lp"].append(torch.zeros(())); tr["v"].append(0.0); tr["r"].append(r)
            tr["ent"].append(torch.zeros(())); tr["vec"].append(info["vec"])
            tr["X"].append(Xt); tr["meta"].append(meta); tr["picks"].append([])
            continue
        with torch.no_grad():
            logits, v = net(Xt)
        picks, lp, ent = sample_action(logits, meta, QUOTA, greedy)
        _, r, done, info = env.step([meta[i] for i in picks])
        tr["lp"].append(lp.detach()); tr["v"].append(float(v)); tr["r"].append(r)
        tr["ent"].append(ent); tr["vec"].append(info["vec"])
        tr["X"].append(Xt); tr["meta"].append(meta); tr["picks"].append(picks)
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
                if len(tr["meta"][t]) == 0:
                    continue
                buf.append((tr["X"][t], tr["meta"][t], tr["picks"][t],
                            tr["lp"][t], adv[t], ret[t]))
            RS.append(sum(tr["r"]))
        A = np.array([b[4] for b in buf], dtype=np.float32)
        A = (A - A.mean()) / (A.std() + 1e-8)

        for _ in range(epochs):
            pl = vl = el = 0.0
            opt.zero_grad()
            for i, (X, meta, picks, lp_old, _, ret) in enumerate(buf):
                logits, v = net(X)
                lp, ent = logprob_of(logits, meta, picks)
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
                for i in picks:
                    u, tg = meta[i]
                    record.append(dict(ep=e, year=t, unit=u,
                                       channel=int(env.ch[u]), target=tg,
                                       n_cells=int(env.ncell[u])))
    return np.mean(V, 0)


def eval_random(env, n_ep=5, seed=0):
    """同一 (单元,目标) 动作空间内的均匀随机策略——与前沿同底可比的参照。"""
    rng = np.random.default_rng(seed)
    V = []
    for e in range(n_ep):
        env.reset(seed=90000 + e)
        g = np.zeros(len(OBJ_NAMES))
        for t in range(env.T):
            X, meta = env.pairs()
            act, used = [], set()
            order = rng.permutation(len(meta))
            for i in order:
                if len(act) >= QUOTA:
                    break
                if meta[i][0] in used:
                    continue
                act.append(meta[i]); used.add(meta[i][0])
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


def main(ds, out, iters, alphas):
    os.makedirs(out, exist_ok=True)
    base = pd.read_csv(os.path.join(ds, "reward_v0", "discounted_return.csv"), index_col=0)
    scale = np.array(base[list(OBJ_NAMES)].abs().max(axis=0).values, dtype=float)
    scale[scale == 0] = 1.0

    rows, curves = [], {}
    e0 = RenewalEnv(ds, weights=weight_vector(0.5), scale=scale)
    gr = eval_random(e0)
    rows.append(dict(alpha=-1.0, **{k: gr[i] for i, k in enumerate(OBJ_NAMES)}))
    print(f"随机(同动作空间)  Floor={gr[OBJ_NAMES.index('Floor')]/1e4:8.0f}万㎡  "
          f"Gdp={gr[OBJ_NAMES.index('Gdp')]:8.0f}  Eco={gr[OBJ_NAMES.index('Eco')]:9.0f}")

    for a in alphas:
        t0 = time.time()
        env = RenewalEnv(ds, weights=weight_vector(a), scale=scale)
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
                   alphas=list(alphas), quota=QUOTA),
              open(os.path.join(out, "train_config.json"), "w"), ensure_ascii=False, indent=1)
    print("\n" + P.round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="data/processed/gm_dataset_v1/rl_v0")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    a = ap.parse_args(); main(a.dataset, a.out, a.iters, a.alphas)
