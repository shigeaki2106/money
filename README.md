# スイングトレード手法のバックテスト

日本株・ETF(日経225/TOPIX)で5種類のスイングトレード手法を過去約20年分(2006〜2026年)で検証し、
モンテカルロで「元手◯万円なら月◯万円に届く確率」を試算したものです。

- **レポート**: [reports/レポート_スイングトレード検証.md](reports/レポート_スイングトレード検証.md)
- データ: Yahoo Finance 日足(`data/`)

```bash
pip install -r requirements.txt
python src/fetch_data.py && python src/experiment.py && python src/montecarlo.py && python src/figures.py
```
