# Box office trends forecast

Forecasts the opening-day and opening-weekend box office of a film from Google Trends search interest and advance-booking data, calibrated on a year of Indian and Hollywood releases. The first film it was built for is **Ramayana: Part 1** (release Thu 5 Nov 2026).

## Current forecast: Ramayana: Part 1

All figures are gross, in ₹ crore. India gross = Sacnilk net × 1.18 (GST). Dollars convert at ₹93. Wednesday 4 Nov paid previews are folded into Day 1.

| | India | Overseas | Worldwide |
|---|---|---|---|
| Day 1 (Thu 5 Nov) | 162 | 126 | **288** (range 144–511) |
| Opening weekend (Thu–Sun) | 590 | 503 | **1,094** (range 461–2,406) |

Inputs as of 28 Oct 2026 (T-8). The forecast moves with every new booking snapshot, so rerun it close to release.

## How it works

**India Day 1** comes from Ramayana's own bookings:

1. Day-1 advance at T-8: ₹14.05 cr gross, including blocked seats (Sacnilk).
2. Grown to release morning at Ramayana's own pace: ~11% a day from T-13 to T-8, held to T-1, then the comps' median last-day jump.
3. Multiplied by the comps' median walk-in multiple (actual Day 1 ÷ final advance, 2.4×).
4. Paid previews added at the comps' median preview share of Day 1.

**Overseas Day 1** is an equal-weight log-average of two estimates:

- Overseas pre-sales (~$0.95M at T-10) grown to T-6 at Ramayana's booking pace, then converted at Dhurandhar 2's pre-sales-to-Day-1 multiple.
- A ridge regression of outside-India opening gross on worldwide Google Trends interest across 34 comps. No Indian/Hollywood label is used.

**Opening weekend** = Day 1 × the Thu–Sun day pattern of Thursday-release comps (Dhurandhar: The Revenge, Kantara: Chapter 1, Spider-Man: Brand New Day).

**Search interest**: each film's Google Trends *topic* (all searches for the film, in any wording) is fetched with a steady anchor term in the same request ("Hanuman Chalisa" for India, "sudoku" worldwide). Values are read as multiples of the anchor, which puts every film on one scale. Features are measured up to T-8, the furthest point Ramayana could be observed at.

A search-only India model (`web_in_mean28` + star flag) is also fitted and reported for reference, but India's forecast uses the bookings model.

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
| `data/model_output.json` | Full model output: forecast, fits, backtests, inputs |
| `report/` | Dashboard page (`index.html`, written for a Claude dashboard artifact) and its CSV datasets |

## Known limits and ideas for contributors

- **Small sample:** 34 comps. Search interest alone explains little of Indian Day-1 variation (leave-one-out error ~70%), which is why India leans on bookings.
- **Single-film anchors:** the T-8 growth path and the overseas conversion lean on Dhurandhar 2, the only comp with a comparable two-week booking window. More long-window comps would help most.
- **Inconsistent trade figures:** sources disagree (net vs gross, blocked seats, previews). Each choice is noted in the CSV `notes` / `source_url` columns.
- **Overseas pre-sales:** Sacnilk's widely quoted "$2M worldwide" includes India; the overseas-only figure (~$0.95M) is used.
- **Ideas:** add YouTube trailer views; collect advance curves for more films; backtest the full pipeline on past releases; and model lifetime gross from the opening.

Box office figures are from Sacnilk, Box Office Mojo, The Numbers, Koimoi, Business Today and other trade press. Every row carries its source URL. These are trade estimates, not official studio numbers.

## License

MIT
