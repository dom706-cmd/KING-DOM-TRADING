# Watchlist Plan Monitor — Standard Operating Procedure

## What This Is

A two-phase system:

**Phase 1 (now):** Monitor your pre-planned watchlist names using YOUR levels (entry, stop, target) instead of OR-derived ones. Each symbol gets a real-time **Plan Readiness score (0–100)** that tells you how favorable conditions are for your specific plan — not a generic signal.

**Phase 2 (future, automatic):** Every session silently logs data. After enough sessions (~200–300 labeled trades), you train a personal ML model on your own outcomes. The rule-based score gets replaced by a model trained on what actually worked *for your specific setups*.

---

## Pre-Market Prep — Parabolic Planner Scan (4am–9:29am)

Run this every morning starting at 4am. It auto-seeds from Alpaca Market Movers (top 25 gainers) and scores each stock on whether its PM move is institutional and still accelerating — not just big.

### What it looks for
- PM move ≥ 10% from previous close
- **Acceleration ratio**: is the move still building in the 6:30–9:30am window or did it peak early? (> 1.0 = still running, < 1.0 = peaked early — treat with caution)
- **Institutional Footprint Score**: avg trade size in PM vs 20-day historical baseline. High = block/institutional prints driving the move. Low = retail FOMO — weaker signal
- **Float turnover**: PM volume as % of shares outstanding — elevated = squeeze potential
- RVOL ≥ 2.0x and catalyst freshness from news

### How to run it
Select preset **`parabolic_watch`** in the Scan panel, or:
```bash
curl "http://localhost:8050/api/scan/start" -X POST \
  -H "Content-Type: application/json" \
  -d '{"preset":"parabolic_watch"}'
```

### Reading the output
| Field | What it means |
|---|---|
| `acceleration_ratio` | > 1.0 = still accelerating into open. Best candidates score > 1.2 |
| `Inst +X%` badge | Institutional factor above baseline. +50% or more = large block prints |
| `pm_move_pct` | Total PM move vs prev close |
| `float_turnover_pct` | PM volume / float — high = float exhaustion / squeeze pressure |
| `news_headline` | Catalyst driving the move — always check age |

### After the scan
1. Filter for `acceleration_ratio > 1.0` and `Inst` factor present
2. Add the best 2–5 names to Desk Watchlist with your planned entry/stop/side
3. **Do not chase names that peaked at 4–5am** (acceleration_ratio < 0.5) — they've already run
4. At 9:30am, check the **Auction** badge (see below) before acting

### The Opening Auction Signal (9:30am)
After the market opens, KingDom fetches the opening auction print for your parabolic and gap_and_go candidates and shows:

- 🟢 **Auction +X%** (green) — the stock opened *above* its PM price. Institutional MOO buyers were present. The move is confirmed. This is an entry window.
- 🔴 **Auction -X%** (red) — the stock faded at the opening print. Sellers absorbed at open. Treat as a potential fade, not a continuation.

> **Real-world example**: AKAN was running 80%+ pre-market. If the parabolic scanner had been running, it would have shown a strong `acceleration_ratio` and `Inst` factor. At open, a positive `Auction` badge would have confirmed whether institutional buyers took it higher or sellers dumped into the gap.

---

## Morning Intelligence — Pre-Market Research Card

Every time you open KingDom, the **Pre-Market Intelligence** card (🌅 Morning prep) at the top of the dashboard auto-loads in the background. It runs four parallel research modules so you walk in prepared.

### The 4 panels

| Panel | What it shows | How to use it |
|---|---|---|
| **Earnings Calendar** | Upcoming earnings in the next 3 days, filtered to stocks ≤ $30 | Earnings gaps are the most predictable movers — any name reporting today/tomorrow is high priority for Gap Orders |
| **Economic Events** | Next FOMC, CPI, NFP dates within 14 days | FOMC and CPI days suppress moves — reduce size or sit out if one is today |
| **FDA / Catalyst Newsflow** | Recent FDA approvals, PDUFA dates, Phase 3 results from the last 72h | Biotech with fresh FDA news = massive float + catalyst combo — pair with Know the Trade scoring |
| **52-Week Breakouts** | Stocks within 5% of 52-week high with RVOL ≥ 1.5x | New highs on volume = continuation play — add to Desk Watchlist if it fits your setup |

### Reading it
- Card loads automatically — no action needed
- Hit **Refresh** if you want to re-run mid-session (e.g. after a major news event)
- Stale data is fine for pre-market context — it's not live pricing, it's research

### When it matters most
- Check **Earnings** first thing — any stock with earnings today that shows up in your scan is not just a momentum play, it's a catalyst play (higher conviction)
- If **Economic Events** shows FOMC or CPI today → plan to be flat by 2pm ET or avoid large positions
- Use **52-Week Breakouts** to cross-reference your Gap Orders list — if a name is on both, grade it with Know the Trade

---

## Night-Before Prep — EOD Momentum Scan (Saturday Prep)

Run this the night before (or weekend) to generate Monday watchlist candidates from Friday's closes.

### What it looks for
- Day move 3–50% with close ≥ 60% of the day's range (closed near HOD)
- RVOL ≥ 1.5x (elevated relative volume)
- Avg daily dollar vol ≥ $1M (liquid enough to trade)
- Scans up to 5,000 symbols — no live market data needed
- **Snapshot pre-filter**: Alpaca Snapshots API quickly eliminates non-movers before fetching full history, making large universe scans significantly faster
- **News enrichment**: each candidate shows the most recent news headline + age (e.g. "14.7h ago") so you immediately know if there's a catalyst

### How to run it
In the UI, open the Scan panel and select preset **`saturday_prep`**, or via curl:
```bash
curl "http://localhost:8050/api/scan/start" -X POST \
  -H "Content-Type: application/json" \
  -d '{"preset":"saturday_prep"}'
```

### After the scan
1. Review top candidates — check the `news_headline` column for catalyst context
2. Save the best names to the **Desk Watchlist** tonight with your planned entry/stop/side
3. In the morning (pre-market, before 9:30), click **⚖ Analyze Watchlist** in the Gap Orders card
4. The analysis shows which names held their Friday move overnight and gives you status per symbol:
   - 🟢 **Intact** — setup still valid, price hasn't moved against you
   - 🟡 **Near Entry** — within 0.5% of your entry, watch closely at open
   - 🔵 **Triggered** — already at/through entry pre-market
   - 🔴 **Invalidated** — price crossed your stop overnight, skip it
5. The table also shows **4 Setup Quality indicators** for yesterday's session — use these to prioritize which Intact names deserve the most attention:

| Column | Green ✓ | Amber ~ | Red ✗ | What it tells you |
|---|---|---|---|---|
| **Close%** | ≥ 60% | 30–60% | < 30% | Where yesterday's close landed in the day's range. Near HOD (≥60%) = stock closed strong, sellers didn't take it back. Near LOD = weak close, setup less clean. |
| **RVOL** | ≥ 1.5× | 0.8–1.5× | < 0.8× | Yesterday's volume vs 20-day average. Elevated volume on the setup day confirms conviction in the move. |
| **News** | < 12h | 12–48h | No news | Age of the most recent headline. Hover the cell to see the headline text. A fresh catalyst (< 12h) means the story is still active going into the next session. |
| **Ext%** | 3–30% | 0–3% | > 30% | Yesterday's day move (open→close). > 30% = extended — harder to get a clean entry without chasing. 3–30% = sweet spot. |

6. Trim your list based on PM analysis, then start the monitor at open with **Watchlist Only** mode

---

## Pre-Session Setup (Night Before or Pre-Market)

### Step 1 — Build Your Desk Watchlist

In the dashboard, scroll to the **Desk Watchlist** card (left column, blue left border).

For each name you're watching, fill in:

| Field | Required | Notes |
|---|---|---|
| Symbol | ✅ | Ticker |
| Side | ✅ | long / short |
| Trigger Price | ✅ | Your planned entry level |
| Stop Price | ✅ | Your hard stop |
| Target Price | ⬜ | Optional — if blank, system computes 2R automatically |
| Notes | ⬜ | Catalyst, setup type, anything relevant |
| Session Date | ✅ | Today's date (auto-fills) |

> **Important:** Trigger price + stop price are the only hard requirements. The system computes risk (1R) from the distance between them. Everything else flows from that.

Click **Add** for each symbol.

The watchlist is server-backed — it survives browser refreshes and overnight.

---

## Know the Trade — Conviction Scoring (🎯 Know It)

After you run **⚖ Analyze Watchlist** in the Gap Orders card, every analyzed name gets a **🎯 Know It** button. This is your conviction engine — it scores the setup from 0–100 and gives you a letter grade with a position sizing recommendation.

### How it works

Click **🎯 Know It** on any analyzed gap order row. The system fetches live data and scores 6 dimensions:

| Component | Max pts | What it measures |
|---|---|---|
| **Catalyst quality + freshness** | 40 | Earnings/FDA/M&A > guidance > news. Fresh (< 2h) adds 18 pts. |
| **Float tier** | 28 | Micro (< 5M shares) = 28. Mega (> 200M) = 0. Lower float = more explosive. |
| **Live RVOL** | 25 | Volume relative to 20-day avg, pulled live at time of scoring. ≥ 10x = 25 pts. |
| **Bid/ask spread** | 12 | Tight spread = clean fills. > 1.5% spread = dangerous. |
| **ML model score** | 13 | Your personal model score (if trained). ≥ 75% = 13 pts. |
| **PM trend quality** | 8 | Is the stock holding its PM highs (long) or lows (short)? |

### Letter grades

| Grade | Score | Size multiplier | What to do |
|---|---|---|---|
| **A** | ≥ 78 | Full size (1.0×) | Lead with this name. Prioritize your max R. |
| **B** | ≥ 58 | Half size (0.5×) | Solid setup — half size, add on confirmation. |
| **C** | ≥ 38 | Quarter size (0.25×) | Mixed signals — quarter size or watch-only. |
| **D** | < 38 | Skip (0.0×) | Trade your A and B setups today. |

### Position sizing tiers

The modal shows three tiers based on your planned entry/stop distance (1R):

| Risk tier | 1R = $250 | 1R = $500 | 1R = $1,000 |
|---|---|---|---|
| Full shares (A grade) | $250 / risk/share | $500 / risk/share | $1,000 / risk/share |
| Graded shares | multiplied by grade | multiplied by grade | multiplied by grade |

**Example:** Entry $10.00, Stop $9.50 → risk/share = $0.50. At $500 1R: 1,000 full shares. B grade (0.5×) → 500 shares.

### When to use it
1. Run **⚖ Analyze Watchlist** on your gap orders
2. Click **🎯 Know It** on each Intact or Near Entry name
3. A/B grades = size up to your planned R. C = watch. D = skip entirely.
4. If you have 3 names and two are A-grade: lead both at full R, not equal weight
5. Re-score after open if conditions change (spread widens, RVOL fades, etc.)

---

## Morning Validation — System Health Check (Pre-Market)

Before you trade a single name, confirm the system is operating above benchmark. One click.

### How to run it

From the main dashboard, click **✅ Morning Validation** in the Quick Links panel (top-right of the page). A modal opens. Click **Run Validation**. The system runs 10 live checks in real time and streams the results as they complete.

### What it checks

| Check | Benchmark | Why it matters |
|---|---|---|
| **Alpaca API** | Response < 2s | Slow API = stale quotes = wrong entries |
| **Live quotes** | 5/5 symbols live | Confirms data feed is up |
| **Stream connection** | Connected, event < 30s ago | Real-time tape and L2 need this active |
| **L2 order book** | Streaming depth available | Walls and book pressure require SIP feed |
| **News feed** | ≥ 10 articles in last 24h | Catalyst scoring is only as good as the news pipeline |
| **ATR scanner** | ≥ 3 candidates returned | Confirms scanner-to-data pipeline is operational |
| **Know the Trade** | Returns A–D grade | Conviction engine is functioning end-to-end |
| **ORB grade distribution** | Multiple grade tiers (not all-F) | Confirms grading math is working and filters are correctly set |
| **Gap orders pipeline** | ≥ 1 gap candidate found | Pre-market gap analysis is operational |
| **Spread alerts** | ≥ 1 symbol subscribed | Halt prediction and spread explosion detection is live |

### Reading the score

| Grade | Score | Meaning |
|---|---|---|
| **A** | ≥ 85% | All systems operational. Trade your plan. |
| **B** | ≥ 70% | Minor issues. Check any warnings before entering size. |
| **C** | ≥ 50% | Multiple systems degraded. Reduce size, avoid illiquid names. |
| **D** | < 50% | System not ready. Diagnose before trading. |

### When to re-run

- After any server restart
- If quotes feel stale or scanner produces zero candidates unexpectedly
- After an Alpaca API outage or connectivity issue

---

## Tape Tracker — Level 2 Order Book

The Tape Tracker (`/tape`) now streams real Level 2 depth, showing where resting orders are sitting and where the real walls are — the same core data professional prop desks watch before every entry.

### How to access it

From the main dashboard Quick Links → **🎯 Tape Tracker →**. Or navigate directly to `/tape`.

### L2 DOM panel (left tab — default)

When you open the Tape Tracker, the right panel shows the **L2 Book** tab by default.

1. Enter a symbol in the input box and click **Track** (or press Enter)
2. The DOM loads and refreshes every 600ms

**What you see:**

```
ASK SIDE (red — sellers)
  $10.48 |       ████  1,200
  $10.45 |     ██████  2,800
  $10.43 | 🧱 ████████████  12,400  ← ask WALL
  ────────────────────────────────
  $10.41   MID  $0.04 spread
  ────────────────────────────────
  $10.40 |  ████████  4,100
  $10.38 |   ██████  2,300
  $10.35 | 🧱 ████████████  11,800  ← bid WALL  (support)
BID SIDE (green — buyers)
```

**Walls** (🧱) — any level where resting size is ≥ 3× the median size on that side AND ≥ 200 shares. These are significant resting orders — price tends to pause, bounce, or absorb at these levels.

**Book pressure bar** — shows what percentage of visible depth is on the bid side. > 60% = buyers are stacked (bullish). < 40% = sellers are stacked (bearish).

### Time & Sales

Below the DOM, the **Time & Sales** feed shows every print streaming from the live tape:

| Column | What it is |
|---|---|
| Time | Print timestamp |
| Price | Trade price (green ▲ = uptick, red ▼ = downtick) |
| Size | Share count |
| Tick | Direction arrow |

**Large prints** (≥ 5× median size for that session) appear in **amber** — these are blocks and institutional sweeps worth noting.

### Book pressure on tape cards

When you add symbols to the main tape grid, each card now shows:
- `Book 68%bid` — 68% of visible depth is resting bids (bullish)
- `Bid wall $10.35` — nearest significant support level
- `Ask wall $10.43` — nearest significant resistance level

### Alpaca SIP requirement

Full L2 depth requires Alpaca's **SIP data feed**. If your account is on IEX (free tier), the DOM will show an amber notice and fall back to L1 NBBO (best bid/ask + size only). To unlock full depth, upgrade your Alpaca plan to include SIP data. The tape tracker is fully functional either way — you just get single-level vs multi-level book.

### Switching to EOD Check

The right panel has two tabs. Click **📋 EOD Check** to switch to the end-of-day gut check (hold overnight decision). Click **📊 L2 Book** to return to the order book.

---

## Session Start (Pre-Market or Open)

### Step 2 — Start the Watchlist Monitor

In the **Monitor** section, find the **Source** dropdown.

Select: **Watchlist Only (premarket)**

Set **Monitor Top N** as needed (default 10 — set higher if you have more names).

Click **Start Monitor**.

What happens behind the scenes:
- Backend fetches your full desk watchlist including entry/stop/target/side
- Each symbol gets seeded with YOUR levels (not OR levels)
- System subscribes to live tape for all symbols
- Plan Readiness scoring begins immediately on every refresh

> If any symbol fails to seed (e.g. no market data yet pre-market), it shows in the **Skipped** warning row in red — other symbols still load fine.

---

## During the Session — Reading the Monitor

### The Plan Readiness Score

Each watchlist symbol shows a **grade + score** instead of GO/WAIT/PASS.

| Display | Meaning |
|---|---|
| `A 82` | Strong conditions — tape aligned, near entry, good time window |
| `B 63` | Decent setup, some friction (spread, VWAP misalign, etc.) |
| `C 48` | Conditions mixed — not ideal to act yet |
| `D 29` | Not aligned — price far away or conditions unfavorable |

### State Labels (left column of monitor row)

| State | Meaning |
|---|---|
| **watch** | Price not yet near your entry level |
| **NEAR ENTRY 🟡** | Price within 0.5R of your entry — getting close, start paying attention |
| **AT ENTRY 🟢** | Price has reached your planned entry level — your setup is live |
| **STOPPED 🔴** | Price crossed your stop level — plan invalidated |

### The Score Breakdown (hover or check `plan_readiness_breakdown` in dev tools)

The score is built from 6 components:

| Component | Weight | What It Measures |
|---|---|---|
| Entry Proximity | 30% | How close price is to your entry (peaks at entry, decays with distance) |
| Tape Alignment | 20% | Tape live + momentum direction matches your side |
| Spread Health | 15% | Spread as % of your risk — can you fill cleanly at your level? |
| Time of Day | 15% | Open impulse (9:30–9:50) scores highest; mid-day lowest |
| VWAP Alignment | 10% | Price on the right side of VWAP for your direction |
| Catalyst | 10% | Fresh catalyst boosts; no catalyst is neutral (not a dealbreaker) |

### WL Badge in the Candidates Table

Any symbol that's on your desk watchlist shows a gold **WL** badge in the candidates table and a gold row outline. This lets you see at a glance which scan candidates you already have pre-planned levels for.

---

## What to Do at Each State

### watch — Price far from entry
Nothing actionable. Monitor score trend over time. If score is rising, conditions are building.

### NEAR ENTRY — Within 0.5R
Start watching the tape closely. Check:
- Is the score ≥ 60 (B or better)?
- Is the time bucket favorable (open impulse or morning trend)?
- Is spread tight enough to fill at your level?

If yes — get ready. This is your pre-entry window.

### AT ENTRY — Entry level reached
Your planned level is live. This is when YOU decide to act based on the tape.

The system is NOT auto-trading — it's telling you the level is there. Your read of the tape in that moment is the final input.

### STOPPED — Stop crossed
Plan is done. No action needed from the monitor. The system logs this as an outcome automatically.

---

## Data Collection — Running in the Background

Every session, the system silently builds your personal training dataset.

**What gets logged (automatically, no action needed):**

Every time a watchlist symbol is within 1.5R of your entry, a snapshot is logged to the `plan_snapshots` table with:
- Your plan levels (entry, stop, target, side)
- Market context at that moment (spread, VWAP delta, time bucket, tape live, catalyst score)
- The rule-based Plan Readiness score at that moment

**What gets labeled (automatically):**

When price crosses your target → snapshot labeled `target_reached`
When price crosses your stop → snapshot labeled `stopped_out`
If neither triggers → you can label it `neither` at session end (manual step, future automation)

**You never have to touch this.** Just run sessions normally and the data accumulates.

---

## Phase 2 — Training Your Personal Model (Future)

Once you have enough labeled sessions (~200–300 snapshots minimum, more is better), you'll run:

```bash
# from the kingdom directory
python ml/build_plan_model.py   # (to be built — uses plan_snapshots_for_training())
```

This trains a gradient boosting classifier on your own labeled data:
- **Features:** all the context columns at snapshot time
- **Label:** `target_reached` (1) vs `stopped_out` / `neither` (0)
- **Output:** a model that predicts probability of YOUR plan hitting YOUR target

The Plan Readiness score in the monitor gets replaced by this model's output. At that point the system is scoring your setups based on what has historically worked for *your specific entry style, levels, and timing preferences* — not generic rules.

---

## SIP Market Events — Real-Time Halt & LULD Feed

Kingdom is wired to the full Alpaca SIP feed (included with Elite at $100/mo). Three live signals are surfaced in real time and update every 5 seconds.

### Market Events Card

A dedicated **Market Events** card in the dashboard shows:

**Active Halts** — every trading halt on your monitored symbols with a plain-English reason and color code:

| Code | Label | Color | What to do |
|---|---|---|---|
| **T1** | News Pending | 🟢 Green | Most bullish halt — news is about to drop. Prepare for gap-up on resume. |
| **T2** | News Released | 🟣 Purple | News is out. Wait for first print to confirm direction. |
| **T5** | Single Stock Pause | 🟡 Amber | 5-min volatility pause. Watch the resume — momentum often continues. |
| **LUDP** | LULD Pause | 🟡 Amber | Limit Up/Limit Down triggered. 5-min countdown. Track collar for resume range. |
| **T12** | NASDAQ Info Req | ⬜ Grey | NASDAQ asked company for info. Neutral — don't rush in. |
| **H4** | Non-Compliance | 🔴 Red | Delisting risk. Do not trade. |
| **H10** | SEC Suspension | 🔴 Red | SEC-ordered halt. Do not trade under any circumstances. |
| **IPO1** | IPO Not Trading | 🟣 Purple | Pre-open IPO. Watch for first print — that's your reference. |
| **MWC1** | Circuit Breaker L1 | 🔴 Red | 7% market-wide drop. 15-min halt. Sit flat. |
| **MWC2** | Circuit Breaker L2 | 🔴 Red | 13% market-wide drop. 15-min halt. Sit flat. |
| **MWC3** | MARKET CLOSED | 🔴 Red | 20% drop. Market closed for the day. Done. |

**LULD Price Bands** — the real-time limit_up and limit_down collars for each monitored symbol, sourced from the SIP feed. These are the exact price levels where the exchange will halt trading if breached.

- Amber highlight = price is within 3% of the limit_up band → you are approaching the parabolic halt zone. Tighten your stop or exit before the halt triggers.
- On a LUDP halt resume, the auction collar is typically set at these bands — they define the resume window.

### SSR Flag in ⚖ Analyze Watchlist

The gap analysis table now includes a red **SSR** pill on any stock that has dropped ≥10% from its prior close. This is Reg SHO Short Sale Restriction — short selling is restricted to uptick only for the rest of that session.

**Practical impact:**
- You cannot short at the bid — you must short on an uptick
- Momentum shorts become harder to fill cleanly
- If SSR is active, widen your short entry cushion or pass and watch for a long fade setup instead

### How it all connects

The SIP signals flow directly into your pre-trade checklist:

1. **Before entering a gap trade** → check the LULD columns in Analyze Watchlist. A stock with a tight limit_up band is one bad print away from a halt.
2. **On a LUDP halt** → wait for the 5-min clock to run, watch the resume auction. The limit_up/limit_down bands in the Market Events card ARE the collar.
3. **On a T1 halt** → news is the catalyst. When the halt clears, the first print tells you direction. This is the highest-quality halt for gap-and-go.
4. **SSR active on your short** → your fills will be slower. Either pass or use a limit order above the market on uptick.

---

## Rebuilt Plans — Invalidated Gap Setups

When a gap order setup gets invalidated overnight (price crosses your stop before 9:30 AM), the system no longer just marks it red and discards it. Instead, it automatically **rebuilds** a fresh plan from the last 30 minutes of pre-market bar data (9:00–9:29 AM).

### What "Rebuilt" means

The Analyze Watchlist table shows a **🔄 Rebuilt** status badge instead of ✗ Invalidated when a valid replacement plan can be constructed. The entry/stop/target are replaced with levels derived from the PM range right before the open — the most relevant reference frame for the gap-and-go.

| Status | Meaning |
|---|---|
| ✗ Invalidated | Price crossed your original stop. No valid replacement found. Skip. |
| 🔄 Rebuilt | Original setup died but a fresh plan exists from the last 30 min of PM bars. |

### When to act on a Rebuilt plan

A Rebuilt plan is a **secondary signal, not a primary one**. Use it as a contingency:

1. The original catalyst is still valid (T1 halt, earnings, FDA — something real)
2. The rebuilt entry is tighter than the original (smaller risk = better R setup)
3. The stock has not already made its full move pre-market
4. Score the rebuilt plan with **🎯 Know It** the same way you would any other setup

If the rebuilt plan grades B or higher and the catalyst is still fresh → treat it as a live setup at open.

If the rebuilt plan grades C or D → the structure is weak. Watch-only or skip.

---

## Troubleshooting

**"Desk watchlist is empty" error when starting monitor**
→ You haven't added any symbols to the Desk Watchlist. Add them first (Step 1).

**Symbol shows in Skipped (red warning)**
→ That symbol had no market data available when the session started (common pre-market for thinly traded names or private companies). Other symbols still load. Try adding it manually via the symbols field.

**Score is stuck at 0**
→ Entry or stop price is missing from the desk watchlist entry. Edit the symbol and add the levels.

**Plan Readiness showing but score seems off**
→ Check the `plan_readiness_breakdown` in the monitor row. The component breakdown shows exactly what's dragging the score (spread, VWAP misalign, time bucket, etc.).

**Want to update a plan mid-session**
→ Remove the symbol from the desk watchlist and re-add with new levels. Then stop and restart the monitor to re-seed with the updated plan. The new levels take effect immediately.

---

## Quick Reference — Daily Flow

```
Night before (or weekend)
  └─ Run saturday_prep scan → review EOD momentum candidates
  └─ Desk Watchlist → save top names with entry/stop/target/side

Pre-market (first thing — before anything else)
  └─ Dashboard → ✅ Morning Validation → Run Validation
       Grade A/B = systems operational, proceed normally
       Grade C   = check warnings, reduce size on illiquid names
       Grade D   = diagnose before trading (API/data issue)

Pre-market 4am–9:29am
  └─ 🌅 Morning Prep card auto-loads → check Earnings (today/tomorrow), Economic Events, FDA news, 52wk highs
  └─ FOMC or CPI today? → reduce size or sit flat. Earnings on a name? → priority candidate
  └─ Run parabolic_watch → auto-seeds from Market Movers
  └─ Filter: acceleration_ratio > 1.0, Inst factor present, fresh catalyst
  └─ Add best 2–5 names to Desk Watchlist
  └─ Skip names that peaked at 4–5am (low acceleration_ratio)

Pre-market (before 9:30)
  └─ Gap Orders → ⚖ Analyze Watchlist → trim names that invalidated overnight
  └─ ML% column shows model score for each name — use it to prioritize
  └─ Setup Quality columns (Close%, RVOL, News, Ext%) — green ✓ = setup criteria met
  └─ Click 🎯 Know It on each Intact/Near Entry name → get A/B/C/D conviction grade
  └─ A/B = trade at grade-sized R. C = watch-only. D = skip. Lead with your A setups.
  └─ Gap Orders → ⚡ Generate Orders → check for gap-open slippage flags
  └─ Tape Tracker → L2 Book → track your top 2–3 names
       Watch bid/ask walls before entering — large ask wall at your entry = resistance
       Book pressure < 40% on a long = sellers stacked, wait for absorption
       Large amber print in T&S near your entry = institutional interest

Market open (9:30am)
  └─ Check Auction badge on parabolic/gap_and_go candidates:
       🟢 Auction +X% = confirmed, institutional buyers at open → entry window
       🔴 Auction -X% = fading, sellers absorbed at open → treat as fade
  └─ Monitor → Source: "Watchlist Only (premarket)" → Start Monitor

During session
  └─ Watch for NEAR ENTRY state + score ≥ B
  └─ Inst badge confirms large block prints → higher conviction
  └─ L2 Book: if a wall clears (large resting ask disappears) on your long → momentum entry
  └─ L2 Book: if book pressure flips < 40% while you're in a long → tighten stop
  └─ AT ENTRY = your level is live, your read of tape makes the call
  └─ STOPPED = plan done, system logs outcome

End of session
  └─ Nothing required — data collection is automatic

After ~200+ sessions
  └─ Train personal ML model on accumulated plan_snapshots data
  └─ Score becomes model-driven, tuned to your setups

Pre-market (any session with a halt)
  └─ Market Events card → Active Halts section lights up with reason code + color
       T1 (green)  = news pending → most bullish halt, price usually gaps higher on resume
       LUDP (amber) = LULD pause → 5-min clock, watch for collar resume range
       H4/H10 (red) = compliance/SEC → do not trade, delisting risk
       MWC1-3 (red) = circuit breaker → market-wide halt, stand down
  └─ LULD Bands section shows limit_up / limit_down for all monitored symbols
       Amber = price within 3% of a band → parabolic halt zone, tighten stop or exit
  └─ SSR badge in ⚖ Analyze Watchlist → red SSR pill means short selling restricted to uptick only
       Gap-down ≥10% from prior close triggers SSR — factor into short entries
```
