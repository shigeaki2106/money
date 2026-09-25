"""日足ベースのシンプルなポートフォリオ型バックテストエンジン。

ルール(ルックアヘッド=未来の情報を使うことを防ぐ):
  * シグナルは当日の終値で判定し、翌営業日の始値で売買する。
  * 損切り(ストップ)は日中に安値が到達したら約定。窓を開けて下に始まった場合は始値で約定。
  * 1銘柄あたりの投資額 = 前日終値時点の資産 / 最大保有数(均等配分)。
  * 取引コスト(手数料+スプレッド+スリッページ)は片道 cost で一律控除。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- data
@dataclass
class Panel:
    """銘柄×日付の配列(分割・配当調整済みOHLC)。"""
    dates: pd.DatetimeIndex
    symbols: list[str]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray

    def df(self, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=self.dates, columns=self.symbols)

    def slice(self, start=None, end=None) -> "Panel":
        m = np.ones(len(self.dates), bool)
        if start is not None:
            m &= self.dates >= pd.Timestamp(start)
        if end is not None:
            m &= self.dates <= pd.Timestamp(end)
        return Panel(self.dates[m], self.symbols, self.O[m], self.H[m], self.L[m], self.C[m])


def load_symbol(sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{sym.replace('^', 'IDX_')}.csv", index_col=0, parse_dates=True)
    f = (df["AdjClose"] / df["Close"]).where(lambda s: s > 0)
    out = pd.DataFrame({c: df[c] * f for c in ["Open", "High", "Low", "Close"]})
    # 始値が欠損/0 の日は終値で代用、高安の整合性を補正
    out["Open"] = out["Open"].where(out["Open"] > 0, out["Close"])
    out["High"] = out[["Open", "High", "Close"]].max(axis=1)
    out["Low"] = out[["Open", "Low", "Close"]].min(axis=1)
    return out.dropna()


def load_panel(symbols: list[str], start="2005-01-01", end=None) -> Panel:
    frames = {s: load_symbol(s) for s in symbols}
    idx = sorted(set().union(*[f.index for f in frames.values()]))
    idx = pd.DatetimeIndex(idx)
    idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    arrs = {k: np.full((len(idx), len(symbols)), np.nan) for k in "OHLC"}
    for j, s in enumerate(symbols):
        f = frames[s].reindex(idx)
        for k, col in zip("OHLC", ["Open", "High", "Low", "Close"]):
            arrs[k][:, j] = f[col].to_numpy()
    return Panel(idx, list(symbols), arrs["O"], arrs["H"], arrs["L"], arrs["C"])


# ---------------------------------------------------------------- engine
@dataclass
class Signals:
    entry: np.ndarray            # bool[T,N] 当日終値で買いシグナル
    exit: np.ndarray             # bool[T,N] 当日終値で売りシグナル
    score: np.ndarray            # float[T,N] 同時シグナル時の優先度(大きい方を優先)
    max_hold: int = 0            # 最大保有日数(0なら無制限)
    stop_pct: float = 0.0        # 損切り幅(0なら無し)。0.08 = 買値から-8%


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series = field(default=None)


def run(panel: Panel, sig: Signals, max_pos: int = 1, cost: float = 0.001,
        init: float = 1.0, trade_start=None) -> Result:
    T, N = panel.C.shape
    O, H, L, C = panel.O, panel.H, panel.L, panel.C
    start_i = 0 if trade_start is None else int(np.searchsorted(panel.dates, pd.Timestamp(trade_start)))

    cash = init
    pos: dict[int, dict] = {}          # j -> {shares, entry_px, entry_i, stop}
    pend_entry: list[int] = []
    pend_exit: set[int] = set()
    last_px = np.full(N, np.nan)
    eq = np.full(T, np.nan)
    expo = np.zeros(T)
    trades = []
    prev_equity = init

    def close_pos(j, px, i, reason):
        nonlocal cash
        p = pos.pop(j)
        proceeds = p["shares"] * px * (1 - cost)
        cash += proceeds
        trades.append((panel.symbols[j], panel.dates[p["entry_i"]], panel.dates[i],
                       p["entry_px"], px, proceeds / p["cost_basis"] - 1, i - p["entry_i"], reason))

    for i in range(T):
        o, h, l, c = O[i], H[i], L[i], C[i]
        # --- 寄り付き: 売り → 買い
        for j in list(pend_exit):
            if j in pos and not np.isnan(o[j]):
                close_pos(j, o[j], i, "signal")
                pend_exit.discard(j)
        if pend_entry:
            alloc = prev_equity / max_pos
            for j in pend_entry:
                if len(pos) >= max_pos or np.isnan(o[j]) or j in pos:
                    continue
                spend = min(alloc, cash)
                if spend <= 0:
                    break
                px = o[j]
                shares = spend / (px * (1 + cost))
                cash -= spend
                pos[j] = {"shares": shares, "entry_px": px, "entry_i": i, "cost_basis": spend,
                          "stop": px * (1 - sig.stop_pct) if sig.stop_pct > 0 else -np.inf}
            pend_entry = []
        # --- 日中: 損切り
        if sig.stop_pct > 0:
            for j in list(pos):
                st = pos[j]["stop"]
                if not np.isnan(l[j]) and l[j] <= st:
                    close_pos(j, min(o[j], st) if pos[j]["entry_i"] != i else st, i, "stop")
                    pend_exit.discard(j)
        # --- 大引け: 評価
        valid = ~np.isnan(c)
        last_px[valid] = c[valid]
        mv = sum(p["shares"] * last_px[j] for j, p in pos.items())
        eq[i] = cash + mv
        expo[i] = mv / eq[i] if eq[i] > 0 else 0
        prev_equity = eq[i]
        if i < start_i:
            continue
        # 売りシグナル
        for j, p in pos.items():
            held = i - p["entry_i"]
            if sig.exit[i, j] or (sig.max_hold and held >= sig.max_hold):
                pend_exit.add(j)
        # 買いシグナル(空き枠ぶん、スコア順)
        free = max_pos - len(pos) + len(pend_exit & set(pos))
        if free > 0:
            cand = np.where(sig.entry[i] & valid)[0]
            cand = [j for j in cand if j not in pos]
            if cand:
                cand.sort(key=lambda j: -sig.score[i, j])
                pend_entry = cand[:free]

    tr = pd.DataFrame(trades, columns=["symbol", "entry_date", "exit_date", "entry_px",
                                       "exit_px", "ret", "days", "reason"])
    return Result(pd.Series(eq, panel.dates), tr, pd.Series(expo, panel.dates))


# ---------------------------------------------------------------- metrics
def metrics(res: Result, start=None, end=None) -> dict:
    eq = res.equity.dropna()
    if start is not None:
        eq = eq[eq.index >= pd.Timestamp(start)]
    if end is not None:
        eq = eq[eq.index <= pd.Timestamp(end)]
    tr = res.trades
    if start is not None:
        tr = tr[tr.entry_date >= pd.Timestamp(start)]
    if end is not None:
        tr = tr[tr.entry_date <= pd.Timestamp(end)]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    dr = eq.pct_change().dropna()
    vol = dr.std() * np.sqrt(245)
    sharpe = dr.mean() / dr.std() * np.sqrt(245) if dr.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    # 最長の下落期間(高値更新までの日数)
    under = dd < 0
    runs, cur = [], 0
    for u in under:
        cur = cur + 1 if u else 0
        runs.append(cur)
    mret = eq.resample("ME").last().pct_change().dropna()
    wins, losses = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    n = len(tr)
    return {
        "取引回数": n,
        "年間取引回数": n / years if years > 0 else np.nan,
        "勝率": len(wins) / n if n else np.nan,
        "平均利益": wins.mean() if len(wins) else 0.0,
        "平均損失": losses.mean() if len(losses) else 0.0,
        "損益比": (wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "期待値(1取引)": tr.ret.mean() if n else np.nan,
        "PF": (wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
        "平均保有日数": tr.days.mean() if n else np.nan,
        "年率リターン": cagr,
        "年率ボラ": vol,
        "シャープ": sharpe,
        "最大DD": mdd,
        "最長DD日数": int(max(runs)) if runs else 0,
        "MAR": cagr / -mdd if mdd < 0 else np.nan,
        "月次勝率": (mret > 0).mean() if len(mret) else np.nan,
        "平均月次リターン": mret.mean() if len(mret) else np.nan,
        "月次リターン標準偏差": mret.std() if len(mret) else np.nan,
        "最悪月": mret.min() if len(mret) else np.nan,
        "投資比率": res.exposure[eq.index].mean() if res.exposure is not None else np.nan,
    }


def buy_and_hold(panel: Panel, weights=None) -> Result:
    """等金額で日次リバランスせずに保有するベンチマーク(欠損は0リターン扱い)。"""
    r = pd.DataFrame(panel.C, index=panel.dates).ffill().pct_change().fillna(0.0)
    w = np.full(len(panel.symbols), 1 / len(panel.symbols)) if weights is None else weights
    growth = (1 + r).cumprod()
    eq = (growth * w).sum(axis=1)
    return Result(eq, pd.DataFrame(columns=["ret", "days", "entry_date"]), pd.Series(1.0, panel.dates))
