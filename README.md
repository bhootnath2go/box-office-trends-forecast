# Box office trends forecast

Forecasts the opening-day and opening-weekend box office of a film from Google Trends search interest and advance-booking data, calibrated on a year of Indian and Hollywood releases. The first film it was built for is **Ramayana: Part 1** (release Thu 5 Nov 2026).

## Current forecast: Ramayana: Part 1

Sentiment is the main input. Each market's opening comes from how much people in that market search for the film, whether that interest is still rising, and Wikipedia views. The model is fitted on 34 Hollywood and Indian releases from the past year. Pre-sales are a check, not an input, and allotted screens cap each region. All figures are gross, in ₹ crore, with dollars converted at ₹93.

| | India | Overseas | Worldwide |
|---|---|---|---|
| Day 1 (Thu 5 Nov, previews included) | 143 (54–259) | 178 (68–324) | **320** (range 122–583) |
| Opening weekend (Thu–Sun) | 407 (155–741) | 508 (194–925) | **916** (range 349–1,666) |

By market, opening weekend: India 407, North America 255 ($27.4M), rest of world 171, UK 41, Australia/NZ 30, Gulf 11.

Search and Wikipedia data run to 3 Nov 2026 (T-2, partial day).

**Checks (not inputs):**

| Check | Says | Sentiment model |
|---|---|---|
| Pre-sales, India Day 1 (₹32.5 cr booked at T-2) | 133 | 143 |
| Pre-sales, overseas weekend (US pace vs Dhurandhar 2) | 181 | 508 |
| Search + pre-sales blend, worldwide weekend | 725 | 916 |
| One worldwide search number, worldwide weekend | 1,579 | 916 |

## How it works

`market_model.py`, one equation per market (India, North America, UK, Gulf, Australia/NZ, rest of world):

log(opening per day) = market level + market slope × log(search) + c × log(momentum) + d × log(Wikipedia views)

- **Search:** the Google Trends film topic in that country (India, US, UK, UAE, Australia), 28-day average to T-2, as a multiple of a steady anchor term in the same request.
  - Rest of world uses Wikipedia views in Spanish, French, German, Portuguese, Japanese, Chinese and Arabic instead. Worldwide search for an Indian-audience film is mostly searches from India, which inflated the old model about 5×.
- **Momentum:** last-7-day average ÷ 28-day average. A film still rising into release opens bigger than one that peaked at its trailer.
- **Wikipedia views:** daily views across all language editions. These are absolute counts, not relative scores.
- **Market level and slope:** these capture ticket prices, moviegoing habit and anchor-term popularity in each market. Hollywood and Indian films are pooled; they differ only by where their audience searches.
- **Weekend** = per day × 4 (Thu–Sun). **Day 1** = 35% of the weekend, the Hollywood opening-day share (`data/hollywood_opening_days.csv`).
- **Bias correction:** summed across markets, the raw model came in below actuals on past films (median actual ÷ predicted 1.44). The base is scaled by that factor so it sits in the middle of the model's own errors.
- **Range:** the 10th–90th percentile of the model's leave-one-out errors on past films' worldwide totals. It is skewed upward because the model underpredicts the biggest Hollywood openings (Super Mario Galaxy, Zootopia 2).
- **Screen caps:** India 9,000 screens × 5 shows × 250 seats × ₹402 = ₹452 cr a day. Overseas 50,000 screens × 5 × 200 × $11 = ₹5,115 cr a day. No scenario reaches either cap.

**Backtest** (34 comps, each predicted without itself, median error on the worldwide opening):

| | All | Indian-audience | Hollywood |
|---|---|---|---|
| This model | **38%** | **31%** | 49% |
| One worldwide search number | 63% | 70% | 42% |

A per-market search slope beat one shared slope (38% vs 54%). Trailer like ratios were tried and dropped: automatic matching picked wrong videos for most films. Drishyam 3 is left out because Google rejected its country-level requests. Several variants were tried on this same backtest, so the 38% is slightly optimistic.

`data/model_output.json` also keeps the search + pre-sales blend (`presales_blend`), the Hollywood-only search model (`search_only`) and the earlier India bookings model (`india_style_reference`) for comparison.

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
python fetch_trends.py --geo=US,GB,AE,AU   # per country -> data/trends_geo.csv
python fetch_wiki.py             # Wikipedia pageviews -> data/wiki_views.csv
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
| `market_model.py` | Market-by-market sentiment model (the headline) |
| `fetch_wiki.py` | Wikipedia pageview fetcher |
| `data/market_openings.csv` | Opening weekend by market (North America, UK, Gulf, Australia/NZ, overseas, India) for each comp |
| `data/trends_geo.csv`, `data/wiki_views.csv` | Per-country search and Wikipedia views |
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
