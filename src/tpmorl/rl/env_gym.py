"""env_gym.py — 把时序状态机 + 多目标奖励包成一个 RL 环境。

一个 env step = 一年。动作 = 当年从合规单元中选出至多 QUOTA 个立项（自回归采样）。

动作是 **(单元, 目标功能) 对**，而非只选单元。

一开始曾把目标功能交给与基线相同的 myopic 规则（通道允许集内取 Floor-Cost 最优），
但那个规则是**空洞的**：容积率上限只由通道决定，同一单元各目标的 Floor 完全相同，
于是规则退化为"取转换成本最低者"＝原类重建，转换成本恒为 0、空间目标增量恒为 0
（实测评估中 Gdp/Eco/Cost 全为 0）。因此把目标功能并入动作空间由策略决定。
若要恢复"仅时序"的作用域，需要按目标类别而非通道标定容积率上限，
而已批单元规划表中没有规划功能字段，无法标定。
"""
import os
import numpy as np, pandas as pd

from tpmorl.env.schedule import RenewalSchedule, QUOTA, S3, S4, TAU_EXT
from tpmorl.objectives.reward import Reward, FAR_CAP, OBJ_NAMES, SIGN
from tpmorl.objectives.run_reward_demo import load, pick_target, ALLOWED

N_FEAT = 16
N_PAIR_FEAT = N_FEAT + 12 + 2
CH_ORDER = [1, 2, 3, 5]


class RenewalEnv:
    def __init__(self, ds, T=15, weights=None, scale=None, seed=0, gamma=0.95):
        (self.LU0, self.cls, self.road, self.water, self.inside,
         self.uid, self.U, self.UUM, self.CM, self.CCM) = load(ds)
        self.T, self.seed, self.gamma = T, seed, gamma
        self.n = len(self.U)
        self.ch = self.U["ch_code"].values
        # 单元格掩码只算一次：717 个单元各一张 171x161 布尔图，每回合重算会主导耗时
        self.cells = [self.uid == u for u in self.U["uid"].values]
        self.ncell = np.array([int(c.sum()) for c in self.cells])
        self.press = self.U["压力"].values.astype(float)
        self.press = self.press / max(self.press.max(), 1e-9)
        self.farcap = np.array([FAR_CAP.get(int(c), FAR_CAP[5]) for c in self.ch])
        self.weights = np.ones(len(OBJ_NAMES)) if weights is None else np.asarray(weights, float)
        self.scale = np.ones(len(OBJ_NAMES)) if scale is None else np.asarray(scale, float)

    def obs(self):
        F = np.zeros((self.n, N_FEAT), dtype=np.float32)
        F[:, 0] = self.press
        F[:, 1] = self.ncell / max(self.ncell.max(), 1)
        for j, c in enumerate(CH_ORDER):
            F[:, 2 + j] = (self.ch == c)
        for s in range(6):
            F[:, 6 + s] = (self.env.sigma == s)
        F[:, 12] = self.env.clock / max(TAU_EXT, 1)
        F[:, 13] = self.farcap / 10.0
        F[:, 14] = self.t / self.T
        F[:, 15] = self.mask_init.astype(np.float32)
        return F

    def pairs(self):
        """当年所有 (合规单元, 通道允许的目标功能) 组合及其特征。"""
        F = self.obs()
        idx = np.where(self.mask_init)[0]
        rows, meta = [], []
        for u in idx:
            for tg in ALLOWED[int(self.ch[u])]:
                cst = sum(self.CCM[f, tg] * c for f, c in self.hist0[u].items())
                x = np.zeros(N_PAIR_FEAT, dtype=np.float32)
                x[:N_FEAT] = F[u]
                x[N_FEAT + tg] = 1.0
                x[N_FEAT + 12] = cst / max(self.ncell[u], 1) / 100.0
                x[N_FEAT + 13] = self.farcap[u] / 10.0
                rows.append(x); meta.append((int(u), int(tg)))
        if not rows:
            return np.zeros((0, N_PAIR_FEAT), dtype=np.float32), []
        return np.stack(rows), meta

    def reset(self, seed=None):
        self.LU = self.LU0.copy()
        self.res_map = (self.LU * self.UUM[0]).sum(-1)
        self.R = Reward(self.UUM, self.CM, self.CCM, self.road, self.water,
                        self.inside, gamma=self.gamma)
        self.env = RenewalSchedule(self.ch, seed=self.seed if seed is None else seed)
        self.t = 0
        self.plan = {}          # 单元 -> 立项时选定的目标功能
        self.hist0 = [{k: v for k, v in
                       ((k, int((self.LU0[..., k][m] > 0).sum())) for k in range(12)) if v}
                      for m in self.cells]
        self.mask_init = self.env.mask_initiate()
        return self.obs()

    def step(self, actions):
        """actions: [(单元, 目标功能), ...]，至多 QUOTA 个。目标功能在立项时锁定。"""
        actions = list(actions)
        for u, tg in actions:
            self.plan[int(u)] = int(tg)
        prev = self.env.sigma.copy()
        ma = self.env.mask_advance()
        _, ev = self.env.step(initiate=[u for u, _ in actions], advance=np.where(ma)[0])

        done_u = np.where((prev == S3) & (self.env.sigma == S4))[0]
        floor = cost = 0.0
        for u in done_u:
            m = self.cells[u]
            hist = self.hist0[u]
            tgt = self.plan.get(int(u))
            if tgt is None:      # 保险：无记录时退回 myopic 规则
                tgt = pick_target(self.R, self.ch[u], hist, self.ncell[u])
            floor += self.R.floor_area(self.ch[u], self.ncell[u])
            cost += sum(self.R.convert_cost(f, tgt, c) for f, c in hist.items())
            self.LU[m] = 0.0
            self.LU[..., tgt][m] = 1.0

        dis = self.R.disrupt(self.env.sigma, self.uid, self.res_map)
        rv = self.R.step_reward(self.LU, floor, cost, dis, ev["expired"])
        vec = np.array([SIGN[k] * rv[k] for k in OBJ_NAMES], float) / self.scale
        r = float((self.weights / self.weights.sum() * vec).sum())

        self.t += 1
        self.mask_init = self.env.mask_initiate()
        return self.obs(), r, self.t >= self.T, dict(vec=vec, raw=rv, events=ev)
