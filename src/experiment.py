"""インサンプル最適化 → アウトオブサンプル検証 → ウォークフォワード を実行して結果を保存する。

  * 手法を作る期間(インサンプル, IS)  : 2006-01-04 〜 2016-12-30
  * 検証期間      (アウトオブサンプル, OOS): 2017-01-04 〜 データ最終日
  * ウォークフォワード(WF): 2012年〜毎年、「前年までのデータ」だけで最良パラメータを選び直してその年を運用
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Result, buy_and_hold, load_panel, load_symbol, metrics, run
from fetch_data import STOCKS
from strategies import STRATEGIES, grid

OUT = Path(__file__).resolve().parent.parent / "reports" / "results"
IS_START, IS_END = "2006-01-04", "2016-12-30"
OOS_START = "2017-01-04"
WF_FIRST_YEAR = 2012
COST = 0.001            # 片道0.1%(手数料0円の証券会社でも、スプレッド+滑りを見込む)

UNIVERSES = {
    # 日経平均(指数)を 1321/1330 等の日経225ETFの代理として使う(ETFは Yahoo 上 2009年〜しかないため)
    "ETF(日経225)": dict(symbols=["^N225"], max_pos=1, min_trades=20,
                        strategies=["RSI2押し目買い", "ボリンジャー逆張り", "乖離率逆張り", "ブレイクアウト"]),
    "個別株(大型40銘柄)": dict(symbols=list(STOCKS), max_pos=5, min_trades=100,
                          strategies=list(STRATEGIES)),
}
# OOS期間に実際のETFでも同じパラメータで確認する
ETF_CHECK = {"1321.T": "日経225連動型上場投信(1321)", "1305.T": "iFreeETF TOPIX(1305)"}


def sel_score(m: dict, min_trades: int) -> float:
    """パラメータ選択基準: シャープレシオ(取引回数が少なすぎるものは除外)。"""
    if not m["取引回数"] or m["取引回数"] < min_trades or np.isnan(m["シャープ"]):
        return -np.inf
    return m["シャープ"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market = load_symbol("^N225")["Close"]
    summary, grid_rows, oos_returns, wf_returns, wf_params = [], [], {}, {}, []

    for uname, u in UNIVERSES.items():
        panel = load_panel(u["symbols"], start="2005-01-01")
        last = panel.dates[-1]
        # ベンチマーク(買って持つだけ)
        bh = buy_and_hold(panel.slice(IS_START))
        for per, s, e in [("IS", IS_START, IS_END), ("OOS", OOS_START, None)]:
            m = metrics(bh, s, e)
            summary.append({"ユニバース": uname, "手法": "買い持ち(比較用)", "期間": per, "パラメータ": "-", **m})
        oos_returns[f"{uname}|買い持ち(比較用)"] = bh.equity.pct_change()[OOS_START:]

        for sname in u["strategies"]:
            fn, space = STRATEGIES[sname]
            kw = {"market": market} if sname == "モメンタム" else {}
            runs = {}
            for params in grid(space):
                sig = fn(panel, **params, **kw)
                res = run(panel, sig, max_pos=u["max_pos"], cost=COST, trade_start=IS_START)
                key = json.dumps(params, ensure_ascii=False)
                runs[key] = res
                mi, mo = metrics(res, IS_START, IS_END), metrics(res, OOS_START)
                grid_rows.append({"ユニバース": uname, "手法": sname, "パラメータ": key,
                                  **{f"IS_{k}": v for k, v in mi.items()},
                                  **{f"OOS_{k}": v for k, v in mo.items()}})
            g = pd.DataFrame([r for r in grid_rows if r["ユニバース"] == uname and r["手法"] == sname])
            # --- ISだけで最良パラメータを決める
            scores = {k: sel_score(metrics(runs[k], IS_START, IS_END), u["min_trades"]) for k in runs}
            best = max(scores, key=scores.get)
            res = runs[best]
            for per, s, e in [("IS", IS_START, IS_END), ("OOS", OOS_START, None)]:
                summary.append({"ユニバース": uname, "手法": sname, "期間": per, "パラメータ": best,
                                **metrics(res, s, e)})
            summary.append({"ユニバース": uname, "手法": sname, "期間": "OOS(全パラメータ中央値)",
                            "パラメータ": f"{len(runs)}通り",
                            **g.filter(like="OOS_").median(numeric_only=True)
                               .rename(lambda c: c[4:]).to_dict()})
            oos_returns[f"{uname}|{sname}"] = res.equity.pct_change()[OOS_START:]
            res.trades.assign(ユニバース=uname, 手法=sname).to_csv(
                OUT / f"trades_{uname}_{sname}.csv".replace("/", "_"), index=False)

            # --- 実際のETFで OOS 確認(同じパラメータ)
            if uname.startswith("ETF"):
                for sym, label in ETF_CHECK.items():
                    ep = load_panel([sym], start="2009-01-01")
                    er = run(ep, fn(ep, **json.loads(best)), max_pos=1, cost=COST, trade_start=OOS_START)
                    summary.append({"ユニバース": label, "手法": sname, "期間": "OOS", "パラメータ": best,
                                    **metrics(er, OOS_START)})
                    oos_returns[f"{label}|{sname}"] = er.equity.pct_change()[OOS_START:]

            # --- ウォークフォワード
            daily = {k: r.equity.pct_change() for k, r in runs.items()}
            pieces = []
            for y in range(WF_FIRST_YEAR, last.year + 1):
                sc = {k: sel_score(metrics(runs[k], IS_START, f"{y - 1}-12-31"), u["min_trades"])
                      for k in runs}
                pick = max(sc, key=sc.get)
                wf_params.append({"ユニバース": uname, "手法": sname, "年": y, "選ばれたパラメータ": pick})
                pieces.append(daily[pick][str(y)])
            wf = pd.concat(pieces)
            wf_returns[f"{uname}|{sname}"] = wf
            wf_res_eq = (1 + wf.fillna(0)).cumprod()
            wm = metrics(Result(wf_res_eq, pd.concat([runs[p["選ばれたパラメータ"]].trades[
                runs[p["選ばれたパラメータ"]].trades.entry_date.dt.year == p["年"]]
                for p in wf_params if p["ユニバース"] == uname and p["手法"] == sname]),
                pd.Series(np.nan, wf_res_eq.index)))
            summary.append({"ユニバース": uname, "手法": sname, "期間": f"WF({WF_FIRST_YEAR}〜)",
                            "パラメータ": "毎年再選択", **wm})
            print(f"[{uname}] {sname}: best={best}  IS Sharpe={scores[best]:.2f}  "
                  f"OOS Sharpe={metrics(res, OOS_START)['シャープ']:.2f}  WF Sharpe={wm['シャープ']:.2f}")

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "grid.csv", index=False)
    pd.DataFrame(wf_params).to_csv(OUT / "wf_params.csv", index=False)
    pd.DataFrame(oos_returns).to_csv(OUT / "oos_daily_returns.csv")
    pd.DataFrame(wf_returns).to_csv(OUT / "wf_daily_returns.csv")


if __name__ == "__main__":
    main()
