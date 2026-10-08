#!/usr/bin/env python3
"""用 TP-MORL 真实的网络规模与前向形状，实测 CPU 与 A100 的单步耗时。

回答一个具体问题：把这套 PPO 搬到 GPU 上值不值得。
不猜，跑数字。在服务器上：  PYTHONPATH=src python3 scripts/bench_cpu_vs_gpu.py

口径说明：
  * 网络与 train_ppo.Pointer 逐字同构（22→64→64，score 头 + value 头）
  * n_pairs 取几档，覆盖候选对数量的实际范围
  * GPU 计时前做 warmup 并 torch.cuda.synchronize()，否则测到的是异步入队时间
  * 额外单独计一次 host→device 拷贝的开销——这项在真实训练里每步都要付，
    而它恰恰是小模型上 GPU 反而更慢的主因
"""
import time, statistics
import torch, torch.nn as nn

N_FEAT, H = 22, 64
REPEAT, WARMUP = 200, 20


class Pointer(nn.Module):
    def __init__(self, nf=N_FEAT, h=H):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.score = nn.Linear(h, 1)
        self.val = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, F):
        z = self.enc(F)
        return self.score(z).squeeze(-1), self.val(z.mean(0)).squeeze(-1)


def bench(dev, n_pairs, with_transfer=False):
    torch.manual_seed(0)
    m = Pointer().to(dev)
    src = torch.randn(n_pairs, N_FEAT)           # 真实场景里数据来自 numpy，在 host 上
    F = src if with_transfer else src.to(dev)
    ts = []
    for i in range(REPEAT + WARMUP):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = F.to(dev, non_blocking=True) if with_transfer else F
        with torch.no_grad():
            m(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= WARMUP:
            ts.append((time.perf_counter() - t0) * 1e6)   # 微秒
    return statistics.median(ts)


print(f"torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
n_par = sum(p.numel() for p in Pointer().parameters())
print(f"模型参数量: {n_par:,}（对照：ResNet-50 约 25,600,000）\n")

print(f"{'候选对数':>8} {'CPU(μs)':>10} {'GPU(μs)':>10} {'GPU+拷贝(μs)':>14} {'结论':>12}")
print("-" * 62)
for n in (200, 500, 1000, 2000, 5000):
    c = bench("cpu", n)
    if torch.cuda.is_available():
        g = bench("cuda", n)
        gt = bench("cuda", n, with_transfer=True)
        best = min(c, gt)
        verdict = "CPU 更快" if best == c else f"GPU 快 {c/gt:.1f}x"
        print(f"{n:>8} {c:>10.1f} {g:>10.1f} {gt:>14.1f} {verdict:>12}")
    else:
        print(f"{n:>8} {c:>10.1f} {'—':>10} {'—':>14} {'无 GPU':>12}")

print("""
怎么读这张表：
  * 看「GPU+拷贝」那列，不要看「GPU」列——真实训练每步都要把 host 上的观测
    搬进显存，那一列才是实际会付的代价。
  * 这套负载是 35 个 worker 各跑各的 PPO（任务并行）。即便单步 GPU 占优，
    35 个进程共享 2 块卡还要排队和切上下文，总吞吐未必比 128 核多进程高。
  * 真要用 GPU，有意义的改法不是「哪块空闲用哪块」，而是把 35 个 worker 的
    前向合批到一块卡上——那是重写训练循环，不是加个 device 参数。
""")
