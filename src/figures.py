"""レポート用のグラフを reports/figures/ に出力する。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from engine import load_panel, run  # noqa: E402
from experiment import COST, IS_END, IS_START, OOS_START, UNIVERSES  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "reports" / "results", ROOT / "reports" / "figures"

for f in font_manager.findSystemFonts():
    if "wqy-zenhei" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2, "axes.unicode_minus": False,
})
S1, S2, S3, GRAY, TEXT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"


def best_params(universe: str, strat: str) -> dict:
    s = pd.read_csv(RES / "summary.csv")
    row = s[(s["ユニバース"] == universe) & (s["手法"] == strat) & (s["期間"] == "IS")].iloc[0]
    return json.loads(row["パラメータ"])


def equity_figure(universe: str, strats: list[str], fname: str, title: str):
    u = UNIVERSES[universe]
    panel = load_panel(u["symbols"], start="2005-01-01")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bh = pd.DataFrame(panel.C, index=panel.dates).ffill()[IS_START:].pct_change().fillna(0).mean(axis=1)
    ax.plot((1 + bh).cumprod(), color=GRAY, lw=1.5, label="買って持つだけ(比較)")
    for s, c in zip(strats, [S1, S2, S3]):
        fn, _ = STRATEGIES[s]
        res = run(panel, fn(panel, **best_params(universe, s)), max_pos=u["max_pos"], cost=COST,
                  trade_start=IS_START)
        eq = res.equity[IS_START:]
        eq = eq / eq.iloc[0]
        ax.plot(eq, color=c, label=s)
        ax.annotate(s, (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TEXT, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}倍"))
    ax.axvspan(pd.Timestamp(OOS_START), panel.dates[-1], color="#2a78d6", alpha=0.06, lw=0)
    ax.axvline(pd.Timestamp(OOS_START), color=GRAY, lw=1, ls="--")
    y = ax.get_ylim()[1]
    ax.text(pd.Timestamp("2011-06-01"), y, "手法を作る期間(2006〜2016)", ha="center", va="top", color="#52514e")
    ax.text(pd.Timestamp("2021-10-01"), y, "検証期間(2017〜)", ha="center", va="top", color="#52514e")
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    ax.set_ylabel("資産(開始時=1倍, 対数目盛)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    ax.set_xlim(right=panel.dates[-1] + pd.Timedelta(days=900))
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def overfit_figure():
    g = pd.read_csv(RES / "grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, (u, gg) in zip(axes, g.groupby("ユニバース", sort=False)):
        for s, c in zip(gg["手法"].unique(), [S1, S2, S3, "#eda100", "#e87ba4"]):
            d = gg[gg["手法"] == s]
            ax.scatter(d["IS_シャープ"], d["OOS_シャープ"], s=22, color=c, alpha=0.8, label=s,
                       edgecolor="#fcfcfb", linewidth=0.8)
        ax.plot([-0.5, 1.2], [-0.5, 1.2], color=GRAY, lw=1, ls="--")
        ax.set_title(u, loc="left", color=TEXT)
        ax.set_xlabel("手法を作る期間のシャープレシオ")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("検証期間のシャープレシオ")
    fig.suptitle("全パラメータの「作成期間の成績」と「検証期間の成績」(点線より上=検証期間の方が良い)",
                 x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / "overfit_check.png", dpi=150)
    plt.close(fig)


def mc_figure():
    p = pd.read_csv(RES / "mc_probabilities.csv")
    col = "2年累計で目標合計138万以上"
    fig, ax = plt.subplots(figsize=(9, 4.4))
    items = [("個別株・乖離率逆張り", S1), ("ETF(1321)・ブレイクアウト", S2), ("(比較)日経225を買って持つだけ", GRAY)]
    for name, c in items:
        for scen, ls in [("基本(検証期間の実績)", "-"), ("保守的(平均リターン半分)", ":")]:
            d = p[(p["手法"] == name) & (p["シナリオ"] == scen)]
            ax.plot(d["元手(万円)"], d[col] * 100, color=c, ls=ls, marker="o", ms=5,
                    label=f"{name}({scen.split('(')[0]})")
    ax.axhline(50, color=GRAY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 300, 500, 1000, 2000, 3000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}万"))
    ax.set_ylim(0, 100)
    ax.set_ylabel("確率(%)")
    ax.set_xlabel("元手")
    ax.set_title("2年間の税引後利益が目標の合計(138万円)に届く確率", loc="left", color=TEXT, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "mc_probability.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    equity_figure("個別株(大型40銘柄)", ["乖離率逆張り", "ボリンジャー逆張り", "RSI2押し目買い"],
                  "equity_stocks.png", "個別株(大型40銘柄・最大5銘柄同時保有)の資産推移")
    equity_figure("ETF(日経225)", ["乖離率逆張り", "ブレイクアウト", "ボリンジャー逆張り"],
                  "equity_etf.png", "日経225(ETF想定)の資産推移")
    overfit_figure()
    mc_figure()


if __name__ == "__main__":
    main()
