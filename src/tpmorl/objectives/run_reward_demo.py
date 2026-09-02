"""run_reward_demo.py — 在时序状态机上求 11 维奖励，比较三个基线策略。

用法: PYTHONPATH=src python -m tpmorl.objectives.run_reward_demo -T 15
这不是训练，只用于验证奖励函数可算、量级合理，并给出 RL 需要击败的基线。
"""
import argparse, json, os
import numpy as np, pandas as pd

from tpmorl.env.schedule import RenewalSchedule, QUOTA, S1, S2, S3, S4, S5
from tpmorl.objectives.reward import (Reward, FAR_CAP, CLASSES, OBJ_NAMES,
                                      OBJ_SPATIAL, OBJ_TEMPORAL, SIGN, CELL_AREA)

CH_CODE = {"P1 产业拆除重建": 1, "P2 旧居住通道": 2, "P3 商服提升": 3, "P5 农转用储备": 5}
# 各通道法定/惯例允许的目标功能（DN-1 索引）
ALLOWED = {1: [6, 3, 1],      # 工改工 I1 / 工改商 C / 工改居 R2
           2: [0, 1, 3],      # 旧居住 -> R1 / R2 / C
           3: [3],            # 商服提升 -> C
           5: [1, 6, 10]}     # 农转用 -> R2 / I1 / G


def load(ds):
    G = os.path.join(ds, "grid_100m")
    cls = np.load(os.path.join(G, "L0_class.npy"))
    road = np.load(os.path.join(G, "road_100m.npy"))
    water = np.load(os.path.join(G, "water_100m.npy"))
    uid = np.load(os.path.join(ds, "zones_v0", "unit_id.npy"))
    U = pd.read_csv(os.path.join(ds, "zones_v0", "candidate_units.csv"))
    U["ch_code"] = U["通道"].map(CH_CODE)
    Wd = os.path.join(ds, "tables")
    UUM = pd.read_csv(os.path.join(Wd, "UUM.csv"), header=None).values
    CM = pd.read_csv(os.path.join(Wd, "CM.csv"), header=None).values
    CCM = pd.read_csv(os.path.join(Wd, "CCM.csv"), header=None).values
    inside = (cls >= 1) & (cls <= 12)
    LU = np.zeros(cls.shape + (12,))
    for k in range(12):
        LU[..., k] = (cls == k + 1)
    rn = road / max(road.max(), 1e-9)
    wn = water / max(water.max(), 1e-9)
    return LU, cls, rn, wn, inside, uid, U, UUM, CM, CCM


def unit_cells(uid, u):
    return uid == u


def pick_target(R, ch, cur_hist, n_cells):
    """在通道允许的目标里选当期 Floor-Cost 最优者。"""
    best, bt = -np.inf, None
    for t in ALLOWED[int(ch)]:
        floor = R.floor_area(ch, n_cells)
        cost = sum(R.convert_cost(f, t, c) for f, c in cur_hist.items())
        v = floor / 1e4 - cost
        if v > best:
            best, bt = v, t
    return bt


def rollout(ds, T, policy, seed=0):
    LU, cls, road, water, inside, uid, U, UUM, CM, CCM = load(ds)
    R = Reward(UUM, CM, CCM, road, water, inside)
    env = RenewalSchedule(U["ch_code"].values, seed=seed)
    n = len(U)
    cells = {i: unit_cells(uid, U["uid"].iloc[i]) for i in range(n)}
    ncell = np.array([int(cells[i].sum()) for i in range(n)])

    if policy == "pressure":
        order = np.argsort(-U["压力"].values)
    elif policy == "floor":
        far = np.array([FAR_CAP.get(int(c), FAR_CAP[5]) for c in U["ch_code"]])
        order = np.argsort(-(far * ncell))
    else:
        order = np.random.default_rng(seed).permutation(n)

    traj, res_map = [], (LU * UUM[0]).sum(-1)
    for t in range(T):
        mi, ma = env.mask_initiate(), env.mask_advance()
        prev_sigma = env.sigma.copy()
        init = [u for u in order if mi[u]][:QUOTA]
        _, ev = env.step(initiate=init, advance=np.where(ma)[0])

        # S3 -> S4 完成的单元，落实用地变更并计入交付面积与转换成本
        done = np.where((prev_sigma == S3) & (env.sigma == S4))[0]
        floor = cost = 0.0
        for u in done:
            m = cells[u]
            hist = {k: int((LU[..., k][m] > 0).sum()) for k in range(12)}
            hist = {k: v for k, v in hist.items() if v}
            tgt = pick_target(R, U["ch_code"].iloc[u], hist, ncell[u])
            floor += R.floor_area(U["ch_code"].iloc[u], ncell[u])
            cost += sum(R.convert_cost(f, tgt, c) for f, c in hist.items())
            LU[m] = 0.0; LU[..., tgt][m] = 1.0

        dis = R.disrupt(env.sigma, uid, res_map)
        r = R.step_reward(LU, floor, cost, dis, ev["expired"])
        r["年"] = t; r["策略"] = policy
        traj.append(r)
    return R, pd.DataFrame(traj), traj


def main(ds, T, out):
    os.makedirs(out, exist_ok=True)
    frames, rets = [], {}
    for pol in ["pressure", "floor", "random"]:
        R, DF, traj = rollout(ds, T, pol)
        frames.append(DF)
        rets[pol] = R.discounted_return(traj)
    ALL = pd.concat(frames, ignore_index=True)
    ALL.to_csv(os.path.join(out, "reward_traj.csv"), index=False, encoding="utf-8-sig")
    RET = pd.DataFrame(rets).T[list(OBJ_NAMES)]
    RET.to_csv(os.path.join(out, "discounted_return.csv"), encoding="utf-8-sig")
    json.dump(dict(far_cap=FAR_CAP, gamma=R.gamma, T=T, pool_r=5,
                   obj_spatial=list(OBJ_SPATIAL), obj_temporal=list(OBJ_TEMPORAL),
                   sign={k: SIGN[k] for k in OBJ_NAMES}),
              open(os.path.join(out, "reward_config.json"), "w"),
              ensure_ascii=False, indent=1)

    pd.set_option("display.width", 200)
    print("折扣累计回报（已统一为越大越好，γ=0.95, T=%d）:" % T)
    print(RET.round(1).to_string())
    print("\n交付计容建面（万平方米，15 年累计，未折扣）:")
    for pol in rets:
        s = ALL[ALL["策略"] == pol]["Floor"].sum() / 1e4
        c = ALL[ALL["策略"] == pol]["Cost"].sum()
        print(f"  {pol:9s} {s:8.1f} 万平方米 | 转换成本 {c:9.0f} | "
              f"失效 {int(ALL[ALL['策略']==pol]['Expire'].sum())} 个")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("--out", default="data/processed/gm_dataset_v1/reward_v0")
    ap.add_argument("-T", type=int, default=15)
    a = ap.parse_args(); main(a.dataset, a.T, a.out)
