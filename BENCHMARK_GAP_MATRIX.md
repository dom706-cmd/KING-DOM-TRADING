# Scanner Benchmark Gap Matrix

## Benchmarks
- Trade Ideas
- TradingView
- TrendSpider
- Finviz
- Benzinga Pro

## Summary
KingDom is now stronger on live validation, explicit failure accounting, ORB resiliency, and scanner universe quality. It still trails the top scanner products in preset polish, news/catalyst integration breadth, explanation UX, and comparative analytics.

## Feature-by-feature matrix

| Capability | KingDom (current) | Trade Ideas | TradingView | TrendSpider | Finviz | Benzinga Pro | Gap |
|---|---|---|---|---|---|---|---|
| Real-time scan path | Yes, SIP-validated | Strong | Strong | Strong | Elite-tier real-time | Strong | Medium |
| Default scan presets | Basic + benchmark-inspired presets added | Excellent | Good | Good | Good | Good | Medium |
| Ranking quality | Combined score + ML + gate-aware ranking | Excellent | Good | Strong | Moderate | Moderate | Medium |
| Candidate explainability | Score breakdown + reasons + UI tags | Good | Moderate | Strong | Basic | Moderate | Medium |
| Scanner universe hygiene | Improved; cleaner NASDAQ source | Strong | Strong | Strong | Strong | Strong | Low-Medium |
| Monitor/alert tape UX | Present, improved state/why output | Strong | Moderate | Strong | Weak | Strong | Medium |
| News/catalyst richness | Partial | Moderate | Moderate | Strong | Weak | Excellent | High |
| Saved views/templates UX | Limited | Strong | Strong | Strong | Strong | Moderate | High |
| Comparative market context | Partial | Moderate | Moderate | Strong | Moderate | Strong | Medium-High |
| Observability/debuggability | Strong internal diagnostics | Moderate | Moderate | Moderate | Basic | Moderate | Low |

## What was improved in this pass
- Quiet single-instance startup path: second launch exits cleanly without traceback spam.
- Added benchmark-aware scan presets:
  - `tradeideas`
  - `tradingview`
  - `trendspider`
  - `finviz`
  - `benzinga`
- Added candidate-level `rank_why` explainability to score breakdowns.
- Surfaced scan preset in threshold summary UI.
- Improved alert tape wording and severity badges.

## Highest-priority remaining gaps
1. Saved preset management in the UI (create/save/load named scanners)
2. Richer catalyst/news ingestion and display
3. Better top-of-book / liquidity / spread diagnostics in ranking
4. Comparative watchlists (gappers, momentum, news, reversals, unusual volume)
5. Stronger “why now” / “why rejected” panels with structured explanations
6. More polished alert routing (sound tiers, desktop priority, cooldown controls)

## Recommended next implementation order
1. Saved preset manager + quick-select buttons
2. Dedicated candidate explanation drawer/modal
3. Ranking enhancements with spread/liquidity freshness penalties surfaced in UI
4. Scanner category tabs (gappers / momentum / news / reversal)
5. News/catalyst panel and richer monitor context cards
