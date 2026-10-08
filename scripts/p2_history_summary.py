"""p2_history_summary.py — Table D7: transfer with H years of past field changes in the state.

Reads results_p2/gen_s{S}.csv (H = 0, with baselines) and results_p2/gen_s{S}_h{H}.csv
(H = 1, 2, 3, 5) for training seeds S = 0-4 and writes results_p2/history_sensitivity.csv:
mean ratio on the ten unseen fields, range of the five seed means, difference to ranking on
announced plans, field-seed pairs above it (of 50), and the mean on the main field.
"""
import glob, os, re
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(REPO, "results_p2")
TEST = [f"field{i}" for i in range(1, 11)]


def main():
    base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P2, "gen_s*.csv")))
                      if re.search(r"gen_s\d+\.csv$", f)])
    B = base[base.seed == -1].pivot_table(index="eval_field", columns="rule", values="ratio")
    ann, pers = B.loc[TEST, "rh_announced"], B.loc[TEST, "persistence"].mean()
    rows = []
    for H in (0, 1, 2, 3, 5):
        X = base if H == 0 else pd.concat(
            [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P2, f"gen_s[0-9]_h{H}.csv")))])
        X = X[X.rule == "value_based_multi"]
        assert sorted(X.seed.unique()) == [0, 1, 2, 3, 4], H
        Xt = X[X.eval_field.isin(TEST)]
        sm = Xt.groupby("seed").ratio.mean()
        rows.append(dict(H=H, test_mean=Xt.ratio.mean(), seed_min=sm.min(), seed_max=sm.max(),
                         minus_ann=Xt.ratio.mean() - ann.mean(),
                         wins=int((Xt.ratio.values > ann.loc[Xt.eval_field].values).sum()),
                         above_pers=bool(Xt.ratio.mean() > pers),
                         main=X[X.eval_field == "main"].ratio.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(P2, "history_sensitivity.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
