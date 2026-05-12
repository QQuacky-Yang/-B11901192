
"""
CVA recovery analysis: longitudinal mixed-model implementation

Input:
    CVA data_勿刪_try GEE_0609.xlsx
Sheet:
    Data

Main question:
    Do stroke patients show recovery over time, and which clinical factors are associated
    with ADL recovery trajectory?

Main method:
    Linear Mixed Effects Model:
        ADL_it ~ time + baseline covariates + random intercept for patient ID

Notes:
    - Cox regression is not used because there is no clear event-time or censoring structure.
    - GEE is not used here because the requested alternative model is Mixed Effects Modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

xlsx_path = Path("CVA data_勿刪_try GEE_0609.xlsx")
df = pd.read_excel(xlsx_path, sheet_name="Data", na_values=["#NULL!", "NULL", ""])
df = df.rename(columns={"@3T_hour": "T_hour"})

id_str = df["ID"].astype(str)
df["stroke_group"] = pd.Series(pd.NA, index=df.index, dtype="object")
df.loc[id_str.str.startswith("1"), "stroke_group"] = "ischemic"
df.loc[id_str.str.startswith("2"), "stroke_group"] = "hemorrhagic"

def make_long(df, outcome_name, col_map):
    frames = []
    for time_label, col in col_map.items():
        tmp = df[[
            "ID", "stroke_group", "ageCE365", "sex", "CVA_type", "NIHSS", "GCS", "T_hour", col
        ]].copy()
        tmp = tmp.rename(columns={col: outcome_name})
        tmp["time_label"] = time_label
        tmp["time_num"] = {"pre": 0, "P": 1, "PP": 2}[time_label]
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)
    long["ID"] = long["ID"].astype(str)
    long["sex"] = long["sex"].astype("category")
    long["CVA_type"] = long["CVA_type"].astype("category")
    return long

adl_long = make_long(df, "ADL", {
    "pre": "ADL_total",
    "P": "ADL_total_P",
    "PP": "ADL_total_PP",
})

summary = adl_long.groupby("time_label", observed=False)["ADL"].agg(["count", "mean", "std", "median"]).loc[["pre", "P", "PP"]]
print("\nADL summary by time:")
print(summary)

def paired_t(a, b):
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    t, p = stats.ttest_rel(tmp["a"], tmp["b"])
    return len(tmp), (tmp["b"] - tmp["a"]).mean(), t, p

print("\nPaired ADL changes:")
for name, a, b in [
    ("pre to P", df["ADL_total"], df["ADL_total_P"]),
    ("P to PP", df["ADL_total_P"], df["ADL_total_PP"]),
    ("pre to PP", df["ADL_total"], df["ADL_total_PP"]),
]:
    n, mean_change, t, p = paired_t(a, b)
    print(f"{name}: n={n}, mean change={mean_change:.2f}, t={t:.3f}, p={p:.4g}")

model_df_A = adl_long.dropna(subset=["ADL", "ageCE365", "sex", "CVA_type", "T_hour"])
model_A = smf.mixedlm(
    "ADL ~ time_num + ageCE365 + C(sex) + C(CVA_type) + T_hour",
    model_df_A,
    groups=model_df_A["ID"]
).fit(reml=False, method="lbfgs", maxiter=300)
print("\nModel A:")
print(model_A.summary())

model_df_B = adl_long.dropna(subset=["ADL", "ageCE365", "sex", "CVA_type", "T_hour", "NIHSS"])
model_B = smf.mixedlm(
    "ADL ~ time_num*T_hour + time_num*NIHSS + ageCE365 + C(sex) + C(CVA_type)",
    model_df_B,
    groups=model_df_B["ID"]
).fit(reml=False, method="lbfgs", maxiter=300)
print("\nModel B:")
print(model_B.summary())

plot_data = adl_long.dropna(subset=["ADL"]).groupby("time_label", observed=False)["ADL"].agg(["mean", "sem"]).loc[["pre", "P", "PP"]]
x = np.arange(3)
plt.figure(figsize=(7, 5))
plt.errorbar(x, plot_data["mean"], yerr=plot_data["sem"], marker="o", capsize=4)
plt.xticks(x, ["Pre", "P", "PP"])
plt.xlabel("Time point")
plt.ylabel("ADL total score")
plt.title("Mean ADL Recovery Trajectory")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("adl_mean_trajectory.png", dpi=200)
