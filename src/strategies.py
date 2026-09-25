"""スイングトレード手法の定義。

どの手法も「当日の終値までの情報」だけでシグナルを作る(翌日始値で売買)。
各手法は params(dict) を受け取り Signals を返す。GRID はパラメータ探索範囲。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from engine import Panel, Signals


def _rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _b(x: pd.DataFrame) -> np.ndarray:
    return x.fillna(False).to_numpy(bool)


# ------------------------------------------------------------ 1. RSI(2) 押し目買い
def rsi_pullback(p: Panel, rsi_th=10, trend=200, exit_ma=5, max_hold=10, **_) -> Signals:
    """上昇トレンド(終値>長期線)中の短期的な売られすぎ(RSI2が低い)を買い、
    短期移動平均を上抜けたら利確。"""
    C = p.df(p.C)
    rsi = _rsi(C, 2)
    up = C > C.rolling(trend).mean()
    entry = up & (rsi < rsi_th)
    exit_ = C > C.rolling(exit_ma).mean()
    return Signals(_b(entry), _b(exit_), (-rsi).fillna(-999).to_numpy(), max_hold=max_hold)


# ------------------------------------------------------------ 2. ボリンジャーバンド逆張り
def bollinger_reversion(p: Panel, k=2.0, n=20, trend=200, max_hold=15, stop=0.08, **_) -> Signals:
    """終値が -kσ を下回ったら買い、中心線(移動平均)まで戻ったら売り。損切りあり。"""
    C = p.df(p.C)
    ma, sd = C.rolling(n).mean(), C.rolling(n).std()
    z = (C - ma) / sd
    entry = z < -k
    if trend:
        entry &= C > C.rolling(trend).mean()
    exit_ = C > ma
    return Signals(_b(entry), _b(exit_), (-z).fillna(-999).to_numpy(), max_hold=max_hold, stop_pct=stop)


# ------------------------------------------------------------ 3. 移動平均乖離率 逆張り
def ma_deviation(p: Panel, n=25, dev=-0.07, max_hold=20, stop=0.10, **_) -> Signals:
    """25日線からの乖離率が大きくマイナスになった(売られすぎ)ところを買い、
    移動平均まで戻ったら売り。日本の個人投資家に多い『乖離率』手法。"""
    C = p.df(p.C)
    ma = C.rolling(n).mean()
    d = C / ma - 1
    entry = d < dev
    exit_ = C > ma
    return Signals(_b(entry), _b(exit_), (-d).fillna(-999).to_numpy(), max_hold=max_hold, stop_pct=stop)


# ------------------------------------------------------------ 4. ブレイクアウト(順張り)
def breakout(p: Panel, n_high=20, n_low=10, trend=200, stop=0.0, **_) -> Signals:
    """直近 n_high 日の高値を終値で更新したら買い(順張り)、
    直近 n_low 日の安値を割ったら売り(トレーリング型の手仕舞い)。"""
    C, H, L = p.df(p.C), p.df(p.H), p.df(p.L)
    hh = H.rolling(n_high).max().shift(1)
    ll = L.rolling(n_low).min().shift(1)
    entry = C > hh
    if trend:
        entry &= C > C.rolling(trend).mean()
    exit_ = C < ll
    mom = C / C.shift(n_high) - 1
    return Signals(_b(entry), _b(exit_), mom.fillna(-999).to_numpy(), stop_pct=stop)


# ------------------------------------------------------------ 5. モメンタム・ローテーション
def momentum_rotation(p: Panel, lookback=120, top=5, buffer=3, market_ma=200,
                      market: pd.Series | None = None, **_) -> Signals:
    """毎週金曜に過去 lookback 日の上昇率ランキング上位 top 銘柄を保有。
    順位が top+buffer より下がったら入替え。市場(日経平均)が長期線を下回る間は現金。"""
    C = p.df(p.C)
    mom = C / C.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    fri = pd.Series(C.index.dayofweek == 4, index=C.index)
    # その週の最終営業日(金曜休場にも対応)
    wk = C.index.to_period("W-FRI")
    last_of_week = pd.Series(wk, index=C.index) != pd.Series(wk, index=C.index).shift(-1)
    reb = (fri | last_of_week).to_numpy()
    if market is not None and market_ma:
        m = market.reindex(C.index).ffill()
        ok = (m > m.rolling(market_ma).mean()).to_numpy()
    else:
        ok = np.ones(len(C), bool)
    rk = rank.to_numpy()
    entry = (rk <= top) & reb[:, None] & ok[:, None]
    exit_ = reb[:, None] & ((rk > top + buffer) | np.isnan(rk) | ~ok[:, None])
    return Signals(entry, exit_, mom.fillna(-999).to_numpy())


STRATEGIES = {
    "RSI2押し目買い": (rsi_pullback, {
        "rsi_th": [5, 10, 15, 20], "trend": [100, 200], "exit_ma": [5, 10], "max_hold": [5, 10]}),
    "ボリンジャー逆張り": (bollinger_reversion, {
        "k": [1.5, 2.0, 2.5], "trend": [0, 200], "max_hold": [10, 20], "stop": [0.05, 0.10]}),
    "乖離率逆張り": (ma_deviation, {
        "n": [25], "dev": [-0.05, -0.07, -0.10, -0.15], "max_hold": [10, 20], "stop": [0.0, 0.08, 0.15]}),
    "ブレイクアウト": (breakout, {
        "n_high": [20, 40, 60], "n_low": [10, 20], "trend": [0, 200], "stop": [0.0, 0.08]}),
    "モメンタム": (momentum_rotation, {
        "lookback": [60, 120, 250], "top": [3, 5], "buffer": [2, 5], "market_ma": [0, 200]}),
}


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())]
