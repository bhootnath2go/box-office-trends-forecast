# Box office trends forecast

Forecasts the opening-day and opening-weekend box office of a film from Google Trends search interest and advance-booking data, calibrated on a year of Indian and Hollywood releases. The first film it was built for is **Ramayana: Part 1** (release Thu 5 Nov 2026).

## Current forecast: Ramayana: Part 1

Ramayana is modelled as a Hollywood tentpole: the model is fitted on Hollywood openings only and driven by worldwide search interest. No Indian comp, advance booking or walk-in multiple feeds it. All figures are gross, in ₹ crore, with dollars converted at ₹93.

| | India (home market) | Overseas | Worldwide |
|---|---|---|---|
| Day 1 (Thu 5 Nov, previews included) | 277 (max 486) | 335 | **611** (range 281–1,276) |
| Opening weekend (Thu–Sun) | 791 | 956 | **1,747** (range 844–4,087) |

Search inputs are as of 28 Oct 2026 (T-8).

## How it works

1. **Global opening per day.** A log-log ridge regression of worldwide opening-weekend gross per day on worldwide Google Trends interest (28-day average to T-8). It is fitted on 15 Hollywood releases from Oct 2025 to Oct 2026. Leave-one-out median error is 26% and R² is 0.55. Ramayana's search level (0.33× the reference term) gives ₹437 cr a day, with a P10–P90 range of ₹211–1,022 cr.
2. **Opening weekend** = per day × 4 (Thu–Sun).
3. **Day 1** = per day × 3 × the Hollywood opening-day share of a 3-day weekend (Friday plus previews ÷ Fri–Sun). The comps are Wicked: For Good, Spider-Man: Brand New Day, The Odyssey, Toy Story 5 and Michael; the median share is 47%. The verified figures are in `data/hollywood_opening_days.csv`.
4. **Home/overseas split.** India is treated as the home market, at the Hollywood median domestic share of the global opening (45%).
5. **Screen cap.** India can't sell more seats than it has: each day is capped at 9,000 screens × 5 shows × 250 seats × ₹432 average ticket (from Ramayana's advance sales) = ₹486 cr gross. Only the high-case Day 1 hits it. Overseas is capped the same way using Sony Pictures' reported ~50,000 international screens × 5 shows × 200 seats × $11 = ₹5,115 cr a day, which no scenario reaches. No per-country screen split has been published; the 9,000 / 50,000 figures are pre-release reports from Variety India and Sacnilk.

**Search interest:** each film's Google Trends *topic* (all searches for the film, in any wording) is fetched with a steady anchor term ("sudoku") in the same request, so every film sits on one scale.

The earlier bookings-based (India-style) forecast is still computed and saved under `india_style_reference` in `data/model_output.json` for comparison. It does not feed the headline numbers.

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

To update Ramayana's bookings, add rows to `data/ramayana_advance.csv`. The latest row is used as "now". The model assumes that row is T-8; if you add a later snapshot, adjust the growth step in `advance_model()`.

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
- **Overseas pre-sales:** Sacnilk's widely quoted "$2M worldwide" includes India; the overseas-only figure (~$0.95M) is used.
- **Ideas:** add YouTube trailer views; collect advance curves for more films; backtest the full pipeline on past releases; and model lifetime gross from the opening.

Box office figures are from Sacnilk, Box Office Mojo, The Numbers, Koimoi, Business Today and other trade press. Every row carries its source URL. These are trade estimates, not official studio numbers.

## License

MIT
