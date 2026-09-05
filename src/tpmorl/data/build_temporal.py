"""build_temporal.py — 生成时序状态机的年度动作掩码，并与光明区实证记录对照。

用法: PYTHONPATH=src python -m tpmorl.data.build_temporal --dataset data/processed/gm_dataset_v1 -T 15
输出: <dataset>/temporal_v0/  掩码序列、状态序列、事件表、标定参数
基线策略仅用于验证掩码机制可跑通并对照实证量级，不是训练出的策略。
"""
import argparse, json, os
import numpy as np, pandas as pd

from tpmorl.env.schedule import (RenewalSchedule, SNAME, HAZARD, TAU_VALID, TAU_EXT,
                                 QUOTA, BUILD_YEARS, COOLDOWN, ACTIONABLE_CHANNELS)

CH_CODE = {"P1 产业拆除重建": 1, "P2 旧居住通道": 2, "P3 商服提升": 3, "P5 农转用储备": 5}


def load_units(ds):
    U = pd.read_csv(os.path.join(ds, "zones_v0", "candidate_units.csv"))
    U["ch_code"] = U["通道"].map(CH_CODE)
    return U


def rollout(U, T, seed=0):
    """基线策略：按更新压力从高到低用满年度配额；有效期内的单元每年都推进。"""
    env = RenewalSchedule(U["ch_code"].values, seed=seed)
    order = np.argsort(-U["压力"].values)
    masks, sigmas, evs = [], [], []
    for t in range(T):
        mi, ma = env.mask_initiate(), env.mask_advance()
        masks.append(mi.copy()); sigmas.append(env.sigma.copy())
        init = [u for u in order if mi[u]][:QUOTA]
        _, ev = env.step(initiate=init, advance=np.where(ma)[0])
        ev["年"] = t; ev["可立项数"] = int(mi.sum()); ev["有效期内"] = int(ma.sum())
        evs.append(ev)
    return env, np.array(masks), np.array(sigmas), pd.DataFrame(evs)


def main(ds, T, seed):
    out = os.path.join(ds, "temporal_v0"); os.makedirs(out, exist_ok=True)
    U = load_units(ds)
    env, masks, sigmas, EV = rollout(U, T, seed)

    np.save(os.path.join(out, "mask_initiate_T.npy"), masks)
    np.save(os.path.join(out, "sigma_T.npy"), sigmas.astype("int8"))
    EV.to_csv(os.path.join(out, "events_baseline.csv"), index=False, encoding="utf-8-sig")

    calib = dict(
        hazard=list(HAZARD), tau_valid=TAU_VALID, tau_ext=TAU_EXT, quota=QUOTA,
        build_years=BUILD_YEARS,
        cooldown=("absorb" if COOLDOWN == float("inf") else int(COOLDOWN)),
        T=T, n_units=int(len(U)),
        actionable_channels=list(ACTIONABLE_CHANNELS),
        calibration_source=dict(
            hazard="109 对计划公告→单元规划批准配对的逐年条件批准率（实测）",
            tau_valid="配对时滞中位数 3.00 年（实测）",
            tau_ext="τ 上限 5 年覆盖 70.6% 实际案例（实测）",
            quota="光明区 2011-2018 计划公告均值 3.1、上限 6（实测）",
            build_years="无实测数据，超参数",
            cooldown="无条文依据；依 2026-09 规划局实务答复设为规划期内吸收态"))
    json.dump(calib, open(os.path.join(out, "calib.json"), "w"), ensure_ascii=False, indent=1)

    print(f"单元 {len(U)} 个 | 可更新通道内 {int(env.eligible.sum())} 个 | T={T}")
    print(f"掩码序列 {masks.shape} | 状态序列 {sigmas.shape}")
    print("\n终局状态分布:", env.counts())
    print("\n逐年事件:")
    print(EV[["年", "可立项数", "有效期内", "initiated", "approved", "started",
              "completed", "expired", "released"]].to_string(index=False))
    tot = EV[["initiated", "approved", "completed", "expired"]].sum()
    print(f"\n{T} 年累计: 立项 {tot['initiated']} | 获批 {tot['approved']} | "
          f"建成 {tot['completed']} | 失效 {tot['expired']}")
    print(f"实证对照（光明区 2010-2018）: 立项 34、获批 18 → 年均立项 3.8、年均获批 2.0")
    print(f"本次基线: 年均立项 {tot['initiated']/T:.1f}、年均获批 {tot['approved']/T:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    ap.add_argument("-T", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); main(a.dataset, a.T, a.seed)
