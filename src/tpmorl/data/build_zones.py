"""build_zones.py — 由 gm_dataset_v1/grid_100m 生成 zones_v0 制度通道分区与候选更新单元。

用法:  python -m tpmorl.data.build_zones --dataset data/processed/gm_dataset_v1
依据:  docs/zoning_design.md
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2

F6 = {"R": [1, 2], "C": [3, 4, 5], "I": [6, 7, 8], "U": [12], "A": [9], "E": [10, 11]}
F6NAME = {0: "限建", 1: "居住R", 2: "商服C", 3: "产业I", 4: "设施U", 5: "农业A", 6: "生态E"}
CHNAME = {0: "P0 刚性禁动", 1: "P1 产业拆除重建", 2: "P2 旧居住通道", 3: "P3 商服提升",
          4: "P4 设施存量", 5: "P5 农转用储备", 6: "P6 生态保育"}
FUNC2CH = {0: 0, 1: 2, 2: 3, 3: 1, 4: 4, 5: 5, 6: 6}   # 六类功能 -> 制度通道（v0 代理映射）
UNIT_TARGET = 12        # 候选单元目标格数，对标已批单元实证中位 4.4 ha 与 1-30 格区间
SPLIT_ABOVE = 30        # 超过此格数的连通体二次切分


def compress_func(L0_class):
    f6_of = {dn: i for i, (k, v) in enumerate(F6.items(), 1) for dn in v}
    FUNC = np.zeros_like(L0_class)
    for dn, f in f6_of.items():
        FUNC[L0_class == dn] = f
    FUNC[L0_class == 0] = 0
    FUNC[L0_class == 255] = 255
    return FUNC


def build_channels(FUNC, action_mask):
    inside = FUNC < 255
    CH = np.full(FUNC.shape, 255, "uint8")
    for f, c in FUNC2CH.items():
        CH[inside & (FUNC == f)] = c
    CH[inside & ~action_mask] = 0
    return CH, inside


def segment_units(CH, action_mask, channels=(1, 2, 3, 5), target=UNIT_TARGET, split_above=SPLIT_ABOVE):
    """同通道内连通域分割，再把巨型连通体按空间聚类切到目标尺度。"""
    lab = np.zeros(CH.shape, "int32")
    rows, nid = [], 1
    for ch in channels:
        m = (CH == ch) & action_mask
        cc, n = ndi.label(m, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            ys, xs = np.where(cc == i)
            sz = len(ys)
            k = max(1, int(round(sz / target))) if sz > split_above else 1
            if k == 1:
                parts = [np.ones(sz, bool)]
            else:
                _, cid = kmeans2(np.c_[ys, xs].astype(float), k, minit="++", seed=0)
                parts = [cid == j for j in range(k) if (cid == j).any()]
            for pm in parts:
                lab[ys[pm], xs[pm]] = nid
                rows.append(dict(uid=nid, 通道=CHNAME[ch], 格数=int(pm.sum()),
                                 row=float(ys[pm].mean()), col=float(xs[pm].mean())))
                nid += 1
    return lab, pd.DataFrame(rows)


def renewal_pressure(FUNC, road, inside, win=5):
    """更新压力代理 = z(邻域建成度) + z(可达性)。非权属数据，仅代理。"""
    built = np.isin(FUNC, [1, 2, 3, 4]).astype(float)
    nb = ndi.convolve(built, np.ones((win, win)) / win ** 2, mode="nearest")
    acc = 1 - np.clip(road / np.nanpercentile(road[inside], 95), 0, 1)
    z = lambda a: (a - np.nanmean(a[inside])) / np.nanstd(a[inside])
    return (z(nb) + z(acc)).astype("float32")


def main(ds):
    G, out = os.path.join(ds, "grid_100m"), os.path.join(ds, "zones_v0")
    os.makedirs(out, exist_ok=True)
    L0 = np.load(os.path.join(G, "L0_class.npy"))
    am = np.load(os.path.join(G, "action_mask.npy"))
    road = np.load(os.path.join(G, "road_100m.npy"))

    FUNC = compress_func(L0)
    CH, inside = build_channels(FUNC, am)
    lab, U = segment_units(CH, am)
    PRESS = renewal_pressure(FUNC, road, inside)
    U["压力"] = [float(np.nanmean(PRESS[lab == u])) for u in U["uid"]]
    U["距路m"] = [float(np.nanmean(road[lab == u])) for u in U["uid"]]

    np.save(os.path.join(out, "func6.npy"), FUNC)
    np.save(os.path.join(out, "channel_v0.npy"), CH)
    np.save(os.path.join(out, "unit_id.npy"), lab)
    np.save(os.path.join(out, "pressure_proxy.npy"), PRESS)
    U.to_csv(os.path.join(out, "candidate_units.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"code": range(7), "func6": [F6NAME[k] for k in range(7)],
                  "格数": [int((FUNC == k).sum()) for k in range(7)]}
                 ).to_csv(os.path.join(out, "func6_legend.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"code": range(7), "channel": [CHNAME[k] for k in range(7)],
                  "格数": [int((CH == k).sum()) for k in range(7)],
                  "占区内": [round(float((CH == k).sum() / inside.sum()), 4) for k in range(7)]}
                 ).to_csv(os.path.join(out, "channel_legend.csv"), index=False, encoding="utf-8-sig")
    print(f"候选单元 {len(U)} 个 | 格数中位 {U['格数'].median():.0f} | 1-30 格占比 {U['格数'].between(1,30).mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/gm_dataset_v1")
    main(ap.parse_args().dataset)
