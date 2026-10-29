"""Write the dashboard's CSV datasets from data/model_output.json (Hollywood tentpole model)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model as M
R = ROOT / "report"
o = json.loads((ROOT / "data" / "model_output.json").read_text())
f, h = o["forecast"], o["hollywood"]
lab = {"ww_d1": ("Worldwide", "Day 1"), "ww_wknd": ("Worldwide", "Opening weekend"),
       "india_d1": ("India", "Day 1"), "india_wknd": ("India", "Opening weekend"),
       "overseas_d1": ("Overseas", "Day 1"), "overseas_wknd": ("Overseas", "Opening weekend")}
pd.DataFrame([{"metric": k, "region": r, "window": w, **{s: round(f[k][s], 1) for s in ("low", "base", "high")}}
              for k, (r, w) in lab.items()]).to_csv(R / "forecast.csv", index=False)
drv = [("search_ww", "Ramayana worldwide search, 28-day average to T-8 (× reference term)", h["ramayana_features"]["web_ww_mean28"]),
       ("elasticity", "% more opening gross per 1% more worldwide search (Hollywood fit)", h["coefs"]["web_ww_mean28"]),
       ("fit", "Leave-one-out median error, Hollywood comps", h["mdape_ridge"]),
       ("per_day", "Worldwide opening gross per day", h["per_day"]["base"]),
       ("days", "Opening weekend days (Thu–Sun)", h["days"]),
       ("d1_share", "Opening day (incl. previews) share of a 3-day weekend", h["d1_share"]["base"]),
       ("dom_share", "Home market's share of the global opening", h["dom_share"]),
       ("ceiling", "India screen ceiling per day: 9,000 screens × 5 shows × 250 seats × ₹432", h["india_ceiling_per_day"])]
pd.DataFrame(drv, columns=["key", "label", "value"]).round(3).to_csv(R / "drivers.csv", index=False)
print(pd.read_csv(R / "forecast.csv").to_string()); print(pd.read_csv(R / "drivers.csv").to_string())

# Hollywood comps: search, actual vs leave-one-out predicted opening per day, opening-day share
fw = M.trend_features(pd.read_csv(ROOT / "data" / "trends_ww.csv"))["web_ww_mean28"]
bt = pd.DataFrame(h["backtest"]).rename(columns={"actual": "per_day", "pred_ridge": "per_day_loo"})
sh = pd.DataFrame(h["d1_share_comps"]).set_index("title")["share"]
ver = {r["title"]: r["verdict"] for r in o["comp_table"]}
bt["verdict"], bt["search_ww"], bt["d1_share"] = bt["title"].map(ver), bt["title"].map(fw).round(3), bt["title"].map(sh)
ram = {"title": M.RAMA, "verdict": "Upcoming", "search_ww": round(fw[M.RAMA], 3), "per_day": np.nan,
       "per_day_loo": round(h["per_day"]["base"], 1), "d1_share": np.nan}
cols = ["title", "verdict", "search_ww", "per_day", "per_day_loo", "d1_share"]
ct = pd.concat([pd.DataFrame([ram]), bt.sort_values("search_ww", ascending=False)[cols]])[cols].round({"per_day": 1, "per_day_loo": 1})
ct.to_csv(R / "comps.csv", index=False)
print(ct.to_string())
