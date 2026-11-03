"""Write the dashboard's CSV datasets from data/model_output.json (market-by-market sentiment model)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model as M
R = ROOT / "report"
o = json.loads((ROOT / "data" / "model_output.json").read_text())
f, h, ps, mm = o["forecast"], o["hollywood"], o["presales"], o["market_model"]
lab = {"ww_d1": ("Worldwide", "Day 1"), "ww_wknd": ("Worldwide", "Opening weekend"),
       "india_d1": ("India", "Day 1"), "india_wknd": ("India", "Opening weekend"),
       "overseas_d1": ("Overseas", "Day 1"), "overseas_wknd": ("Overseas", "Opening weekend")}
pd.DataFrame([{"metric": k, "region": r, "window": w, **{s: round(f[k][s], 1) for s in ("low", "base", "high")}}
              for k, (r, w) in lab.items()]).to_csv(R / "forecast.csv", index=False)
st, c = mm["stats"], mm["coefs"]
drv = [("fit_market", "Leave-one-out median error, this model (34 comps)", st["market"]["mdape_all"]),
       ("fit_baseline", "Same test, one worldwide search number (old approach)", st["worldwide_baseline"]["mdape_all"]),
       ("fit_indian", "This model on Indian-audience comps (old approach in brackets: " + f"{st['worldwide_baseline']['mdape_indian']:.0%})", st["market"]["mdape_indian"]),
       ("fit_hollywood", "This model on Hollywood comps (old approach in brackets: " + f"{st['worldwide_baseline']['mdape_hollywood']:.0%})", st["market"]["mdape_hollywood"])]
drv += [(f"mkt_{r['market']}", f"{r['market']} opening weekend", r["wknd_cr"]) for r in mm["ramayana"]]
drv += [("el_" + m, f"Search response, {m} (% gross per 1% search)", c[f"search_{m}"]) for m in ("India", "North America", "UK")]
drv += [("momentum", "Momentum response (% gross per 1% rise, last week vs 28 days)", c["momentum"]),
        ("d1_share", "Day 1 as a share of the Thu–Sun weekend", mm["d1_of_weekend"]),
        ("ceiling", "India screen ceiling per day: 9,000 screens × 5 shows × 250 seats × ₹402", h["india_ceiling_per_day"]),
        ("os_ceiling", "Overseas screen ceiling per day: 50,000 screens (Sony) × 5 shows × 200 seats × $11", h["overseas_ceiling_per_day"]),
        ("india_presales_d1", "Check: India Day 1 implied by pre-sales (₹32.5 cr booked at T-2)", ps["india_presales_d1"]),
        ("os_presales_wknd", "Check: overseas weekend implied by pre-sales (US pace vs Dhurandhar 2)", ps["overseas_presales_wknd"]),
        ("presales_blend_wknd", "Check: worldwide weekend, search + pre-sales blend", o["presales_blend"]["ww_wknd"]["base"]),
        ("search_only_wknd", "Check: worldwide weekend, one worldwide search number", o["search_only"]["ww_wknd"]["base"])]
pd.DataFrame(drv, columns=["key", "label", "value"]).round(3).to_csv(R / "drivers.csv", index=False)
print(pd.read_csv(R / "forecast.csv").to_string()); print(pd.read_csv(R / "drivers.csv").to_string())

# comps: worldwide opening per day, actual vs this model (leave-one-out) vs the worldwide-search baseline, Rs crore
bt = pd.DataFrame(mm["backtest"])
to_cr = M.USD_INR / 10
ct = pd.DataFrame({"title": bt["title"], "industry": bt["industry"], "per_day": (bt["actual"] * to_cr).round(1),
                   "per_day_loo": (bt["pred"] * to_cr).round(1), "per_day_ww": (bt["base_pred"] * to_cr).round(1)})
ram = pd.DataFrame([{"title": M.RAMA, "industry": "Upcoming", "per_day": np.nan,
                     "per_day_loo": round(f["ww_wknd"]["base"] / h["days"], 1), "per_day_ww": np.nan}])
ct = pd.concat([ram, ct.sort_values("per_day", ascending=False)])
ct.to_csv(R / "comps.csv", index=False)
print(ct.to_string())
