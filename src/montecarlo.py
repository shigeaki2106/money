"""モンテカルロ: 「元手◯万円なら、各フェーズで月◯万円に届く確率」を試算する。

方法
  * 検証期間(2017〜, OOS)の月次リターンを「3か月ブロック」で復元抽出(ブロック・ブートストラップ)し、
    24か月の資産推移を 20,000 通り作る。好調・不調が続く性質をある程度残すためにブロックにする。
  * 利益は再投資(複利)。12か月・24か月の時点で、その1年の利益に 20.315% の税を差し引く。
  * 目標判定は「税引後」: 各フェーズの平均月収益 × (1-20.315%) ≥ 目標額。
  * 「保守的」シナリオ: 月次リターンの平均を半分に下げたもの(生存者バイアスや手法の劣化を考慮)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "reports" / "results"
TAX = 0.20315
N_SIM = 20_000
BLOCK = 3
SEED = 42

PHASES = [  # (名前, 開始月, 終了月, 目標下限[万円], 目標上限[万円])
    ("1〜3か月目", 1, 3, 1, 3),
    ("4〜12か月目", 4, 12, 3, 5),
    ("13〜18か月目", 13, 18, 8, 8),
    ("19〜24か月目", 19, 24, 10, 10),
]
CAPITALS = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]  # 万円

# 事前に決めたルール: 「インサンプル(2006-2016)でシャープ0.5以上」の手法を半々で組み合わせたもの等
TARGETS = {
    "個別株・乖離率逆張り": ["個別株(大型40銘柄)|乖離率逆張り"],
    "個別株・ボリンジャー逆張り": ["個別株(大型40銘柄)|ボリンジャー逆張り"],
    "個別株・2手法の組合せ": ["個別株(大型40銘柄)|乖離率逆張り", "個別株(大型40銘柄)|ボリンジャー逆張り"],
    "ETF(1321)・乖離率逆張り": ["日経225連動型上場投信(1321)|乖離率逆張り"],
    "ETF(1321)・ブレイクアウト": ["日経225連動型上場投信(1321)|ブレイクアウト"],
    "(比較)日経225を買って持つだけ": ["ETF(日経225)|買い持ち(比較用)"],
}


def monthly_returns(cols: list[str]) -> pd.Series:
    d = pd.read_csv(RES / "oos_daily_returns.csv", index_col=0, parse_dates=True)
    r = d[cols].fillna(0).mean(axis=1)          # 複数手法は資金を等分(日次で配分維持)
    m = (1 + r).resample("ME").prod() - 1
    return m.iloc[:-1] if m.index[-1] > r.index[-1] else m  # 途中月を除外


def simulate(m: np.ndarray, months=24, n=N_SIM, rng=None) -> np.ndarray:
    """ブロック・ブートストラップで月次リターン行列 [n, months] を作る。"""
    rng = rng or np.random.default_rng(SEED)
    nb = int(np.ceil(months / BLOCK))
    starts = rng.integers(0, len(m) - BLOCK + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(n, -1)[:, :months]
    return m[idx]


def paths(R: np.ndarray, capital: float) -> tuple[np.ndarray, np.ndarray]:
    """月次の資産と月次の損益(税引前)を返す。"""
    n, T = R.shape
    eq = np.full(n, capital, float)
    year_start = eq.copy()
    E, P = np.zeros((n, T)), np.zeros((n, T))
    for t in range(T):
        pnl = eq * R[:, t]
        eq = eq + pnl
        P[:, t] = pnl
        if (t + 1) % 12 == 0:
            gain = np.maximum(eq - year_start, 0)
            eq = eq - gain * TAX
            year_start = eq.copy()
        E[:, t] = eq
    return E, P


def phase_table(R: np.ndarray, capital_man: float) -> dict:
    cap = capital_man * 1e4
    E, P = paths(R, cap)
    net = np.where(P > 0, P * (1 - TAX), P) / 1e4   # 税引後の月損益(万円)。損失月は税なし
    out = {"元手(万円)": capital_man}
    all_ok = np.ones(len(R), bool)
    for name, a, b, lo, hi in PHASES:
        avg = net[:, a - 1:b].mean(axis=1)
        ok_lo = avg >= lo
        all_ok &= ok_lo
        out[f"{name} 月{lo}万以上"] = ok_lo.mean()
        if hi != lo:
            out[f"{name} 月{hi}万以上"] = (avg >= hi).mean()
        out[f"{name} 平均月収益の中央値(万)"] = np.median(avg)
    out["全フェーズ達成(下限)"] = all_ok.mean()
    total_target = sum(lo * (b - a + 1) for _, a, b, lo, _ in PHASES)
    out[f"2年累計で目標合計{total_target}万以上"] = (net.sum(axis=1) >= total_target).mean()
    out["マイナス月の割合"] = (P < 0).mean()
    end = E[:, -1] / 1e4
    out["2年後資産 中央値(万)"] = np.median(end)
    out["2年後資産 下位5%(万)"] = np.percentile(end, 5)
    out["2年後に元本割れの確率"] = (end < capital_man).mean()
    dd = (E / np.maximum.accumulate(np.concatenate([np.full((len(E), 1), cap), E], 1), axis=1)[:, 1:] - 1)
    out["2年間の最大下落率 中央値"] = np.median(dd.min(axis=1))
    out["2年間の最大下落率 下位5%"] = np.percentile(dd.min(axis=1), 5)
    return out


def required_capital(R: np.ndarray, prob: float, key: str) -> float:
    """key の確率が prob 以上になる最小の元手(万円)を二分探索。1億円でも届かなければ NaN。"""
    lo, hi = 10.0, 10_000.0
    if phase_table(R, hi)[key] < prob:
        return np.nan
    for _ in range(30):
        mid = (lo + hi) / 2
        if phase_table(R, mid)[key] >= prob:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    rows, req, stats = [], [], []
    for label, cols in TARGETS.items():
        m = monthly_returns(cols)
        stats.append({"手法": label, "月数": len(m), "平均月次リターン": m.mean(), "月次標準偏差": m.std(),
                      "月次勝率": (m > 0).mean(), "最悪月": m.min(), "最良月": m.max()})
        for scen, mm in [("基本(検証期間の実績)", m.to_numpy()),
                         ("保守的(平均リターン半分)", (m - m.mean() / 2).to_numpy())]:
            R = simulate(mm, rng=np.random.default_rng(SEED))
            for c in CAPITALS:
                rows.append({"手法": label, "シナリオ": scen, **phase_table(R, c)})
            keys = {"2年累計": "2年累計で目標合計138万以上", "19〜24か月目": "19〜24か月目 月10万以上"}
            req.append({"手法": label, "シナリオ": scen,
                        **{f"{k} 確率{int(p * 100)}%に必要な元手(万)": required_capital(R, p, v)
                           for k, v in keys.items() for p in (0.5, 0.8)}})
        print(label, "done")
    pd.DataFrame(stats).to_csv(RES / "mc_monthly_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / "mc_probabilities.csv", index=False)
    pd.DataFrame(req).to_csv(RES / "mc_required_capital.csv", index=False)


if __name__ == "__main__":
    main()
