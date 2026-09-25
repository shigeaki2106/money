"""Yahoo Finance から日足データを取得して data/ にCSV保存する。

stooq.com はブラウザ認証(JavaScriptのボット対策)が入っており自動取得できないため、
Yahoo Finance の chart API を使う。
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 指数・ETF
ETFS = {
    "^N225": "日経平均(指数)",
    "1321.T": "日経225連動型上場投信",
    "1306.T": "TOPIX連動型上場投信",
    "1305.T": "iFreeETF TOPIX",
    "1330.T": "上場インデックスファンド225",
}

# 個別株: 2005年以前から上場している流動性の高い大型株(TOPIX Core30 中心)
# ※ 現在の大型株を選ぶので「生存者バイアス」がある点に注意(レポートに記載)
STOCKS = {
    "7203.T": "トヨタ自動車", "6758.T": "ソニーG", "8306.T": "三菱UFJ",
    "9984.T": "ソフトバンクG", "6861.T": "キーエンス", "9432.T": "NTT",
    "8316.T": "三井住友FG", "6501.T": "日立製作所", "8058.T": "三菱商事",
    "8001.T": "伊藤忠商事", "4063.T": "信越化学", "6098.T": "リクルート",
    "7974.T": "任天堂", "9433.T": "KDDI", "4502.T": "武田薬品",
    "8766.T": "東京海上", "6367.T": "ダイキン", "6954.T": "ファナック",
    "7267.T": "ホンダ", "8411.T": "みずほFG", "4568.T": "第一三共",
    "6902.T": "デンソー", "8031.T": "三井物産", "9983.T": "ファーストリテイリング",
    "6981.T": "村田製作所", "7751.T": "キヤノン", "6503.T": "三菱電機",
    "4661.T": "オリエンタルランド", "8801.T": "三井不動産", "6702.T": "富士通",
    "7011.T": "三菱重工", "5108.T": "ブリヂストン", "2914.T": "JT",
    "4452.T": "花王", "6752.T": "パナソニック", "8035.T": "東京エレクトロン",
    "9020.T": "JR東日本", "3382.T": "セブン&アイ", "6301.T": "コマツ",
    "4503.T": "アステラス製薬",
}

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d",
              "events": "div,splits", "includeAdjustedClose": "true"}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    res = json.loads(r.text)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": q["close"], "Volume": q["volume"],
        "AdjClose": res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"]),
    }, index=pd.to_datetime(res["timestamp"], unit="s").tz_localize("UTC")
       .tz_convert("Asia/Tokyo").normalize().tz_localize(None))
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    df = df[df["Volume"].fillna(0) > 0] if not symbol.startswith("^") else df
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    symbols = {**ETFS, **STOCKS}
    if len(sys.argv) > 1:
        symbols = {s: symbols.get(s, s) for s in sys.argv[1:]}
    for sym, name in symbols.items():
        try:
            df = fetch(sym)
            df.to_csv(DATA_DIR / f"{sym.replace('^', 'IDX_')}.csv")
            print(f"{sym:8s} {name:20s} {df.index[0].date()} 〜 {df.index[-1].date()}  {len(df)}行")
        except Exception as e:  # noqa: BLE001
            print(f"{sym:8s} {name:20s} 取得失敗: {e}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
