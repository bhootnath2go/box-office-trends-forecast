# Box office trends forecast

Forecasts the opening-day and opening-weekend box office of a film from Google Trends search interest and advance-booking data, calibrated on a year of Indian and Hollywood releases. The first film it was built for is **Ramayana: Part 1** (release Thu 5 Nov 2026).

## Current forecast: Ramayana: Part 1

Ramayana is tracked the way a Hollywood tentpole is in its final week. A search model fitted on Hollywood openings sets the starting point. Each region's own ticket pre-sales, read against comps at the same point before release, then update it. Allotted screens cap each region. All figures are gross, in ₹ crore, with dollars converted at ₹93.

| | India (home market) | Overseas | Worldwide |
|---|---|---|---|
| Day 1 (Thu 5 Nov, previews included) | 166 (114–244) | 78 | **244** (range 168–354) |
| Opening weekend (Thu–Sun) | 475 | 222 | **697** (range 481–1,011) |

Search inputs and pre-sales are as of 2 Nov 2026 (T-3). Search alone would give ₹1,806 cr for the worldwide weekend; pre-sales are running far behind that.

## How it works

1. **Global opening per day.** A log-log ridge regression of worldwide opening-weekend gross per day on worldwide Google Trends interest (28-day average to T-3). It is fitted on 15 Hollywood releases from Oct 2025 to Oct 2026. Leave-one-out median error is 33% and R² is 0.54. Ramayana's search level (0.42× the reference term) gives ₹450 cr a day, with a P10–P90 range of ₹186–1,008 cr.
2. **Opening weekend** = per day × 4 (Thu–Sun).
3. **Day 1** = per day × 3 × the Hollywood opening-day share of a 3-day weekend (Friday plus previews ÷ Fri–Sun). The comps are Wicked: For Good, Spider-Man: Brand New Day, The Odyssey, Toy Story 5 and Michael; the median share is 47%. The verified figures are in `data/hollywood_opening_days.csv`.
4. **Home/overseas split.** India is treated as the home market, at the Hollywood median domestic share of the global opening (45%).
5. **Pre-sales blend.** India: Day-1 gross is regressed on Day-1 advance at the same T-day, log-log, across 5 comps (leave-one-out median error 28%). Ramayana's ₹28.5 cr at T-3 (Sacnilk, 2 Nov, incl. blocked seats) implies ₹145 cr. Overseas: Ramayana's US pre-sales run 80–85% of Dhurandhar 2's at the same stage (Pinkvilla via Sacnilk). Applied to D2's $21.25M overseas opening, that gives ₹163 cr for the weekend. Each region's search and pre-sales estimates are combined in log space, weighted by inverse variance. Pre-sales get about 80% of the weight. Inputs are in `data/ramayana_advance.csv` and `data/overseas_presales.csv`.
6. **Screen cap.** India can't sell more seats than it has: each day is capped at 9,000 screens × 5 shows × 250 seats × ₹416 average ticket (from Ramayana's 2 Nov advance) = ₹468 cr gross. 16,440 Day-1 shows at 1,840 cinemas were on sale in Sacnilk's sample on 2 Nov. No scenario hits the cap now. Overseas is capped the same way using Sony Pictures' reported ~50,000 international screens × 5 shows × 200 seats × $11 = ₹5,115 cr a day, which no scenario reaches. No per-country screen split has been published; the 9,000 / 50,000 figures are pre-release reports from Variety India and Sacnilk.

**Search interest:** each film's Google Trends *topic* (all searches for the film, in any wording) is fetched with a steady anchor term ("sudoku") in the same request, so every film sits on one scale.

`data/model_output.json` also keeps the search-only forecast (`search_only`) and the earlier India-style bookings forecast (`india_style_reference`) for comparison.

## Run it

```bash
pip install -r requirements.txt
python model.py                  # fits models, writes data/model_output.json, prints the forecast
python report/export.py          # writes the dashboard CSVs in report/
```

Refresh Google Trends (cached per film; only missing series are fetched, and Google rate-limits hard, so expect pauses):

```bash
python fetch_trends.py --topic   # India, film topics  -> data/trends_topic.csv
python fetch_trends.py --ww      # worldwide, film topics -> data/trends_ww.csv
```

Topic ids are resolved automatically and cached in `data/topics.csv`. Check new rows there: auto-matching picked a song for *Awarapan 2* and "Alphabet" for *Alpha*, which were fixed by hand.

To update Ramayana's bookings, add rows to `data/ramayana_advance.csv` (India) and edit `data/overseas_presales.csv`. The pre-sales model uses the latest India row and needs comp snapshots in `data/advance_bookings.csv` at the same T-day.

Note: scikit-learn was blocked on the original machine, so the ridge regression and k-NN are plain numpy.

## Files

| Path | What it is |
|---|---|
| `model.py` | Feature building, models, blend, forecast |
| `fetch_trends.py` | Google Trends fetcher (pytrends), anchored, cached |
| `data/films_india.csv`, `data/films_hollywood.csv` | Comp films with first-pass box office figures and sources |
| `data/boxoffice_verified.csv` | Second-pass verified figures (Day 1–4, weekend, worldwide), which override the above |
| `data/advance_bookings.csv` | Comps' Day-1 advance-booking snapshots and actual Day 1 |
| `data/ramayana_advance.csv` | Ramayana's advance-booking snapshots |
| `data/overseas_presales.csv` | Ramayana's overseas pre-sales and its US pace vs Dhurandhar 2 |
| `data/ramayana_facts.md` | Release facts with sources |
| `data/topics.csv` | Google Trends topic id per film |
| `data/trends_topic*.csv`, `data/trends_ww*` | Search series used by the model |
| `data/trends.csv`, `data/trends_raw/` | Earlier search-string series, kept for comparison |
| `data/hollywood_opening_days.csv` | Verified US opening days (incl. previews) and 3-day weekends for the Day-1 share |
| `data/model_output.json` | Full model output: forecast, fits, backtests, inputs |
| `report/` | Dashboard page (`index.html`, written for a Claude dashboard artifact) and its CSV datasets |

## Known limits and ideas for contributors

- **Small sample:** 15 Hollywood comps for the headline model, and 5 verified opening days for the Day-1 share. More comps would tighten the range most.
- **Single-film anchors:** the T-8 growth path and the overseas conversion lean on Dhurandhar 2, the only comp with a comparable two-week booking window. More long-window comps would help most.
- **Inconsistent trade figures:** sources disagree (net vs gross, blocked seats, previews). Each choice is noted in the CSV `notes` / `source_url` columns.
- **Overseas pre-sales:** one comp (Dhurandhar 2) anchors the overseas pre-sales read, and Middle East bookings had not opened by 31 Oct.
- **Ideas:** add YouTube trailer views; collect advance curves for more films; backtest the full pipeline on past releases; and model lifetime gross from the opening.

Box office figures are from Sacnilk, Box Office Mojo, The Numbers, Koimoi, Business Today and other trade press. Every row carries its source URL. These are trade estimates, not official studio numbers.

## License

MIT
