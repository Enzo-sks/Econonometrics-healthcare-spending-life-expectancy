"""Modèle 6 : effets fixes pays + année, dépense en log et retardée.

Spécification principale :
    LifeExp_it = b1*ln(TotalExp_{i,t-k}) + b2*log_gdp_it + b3*Schooling_it
                 + alpha_i + lambda_t + u_it
Erreurs standard clustérisées par pays.

Entrée  : data/processed/controlled_regression_sample.csv (produit par data_clean.py)
Sortie  : output/step_03/model6_results.json et model6_summary.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "controlled_regression_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "step_03"

Y = "Q('Life expectancy')"
CONTROLS = "log_gdp + Schooling"
FE = "C(Country) + C(Year)"


def build_sample(path: Path = CONTROL_DATA_PATH, lag: int = 1) -> pd.DataFrame:
    """Ajoute ln(Total expenditure) et son retard calculé au sein de chaque pays.

    Le retard n'est conservé que si l'observation t-lag existe réellement
    (années exactement espacées de `lag`), pour éviter de relier des années non consécutives.
    """
    df = pd.read_csv(path)
    df = df[df["Total expenditure"] > 0].sort_values(["Country", "Year"]).copy()
    df["log_te"] = np.log(df["Total expenditure"])

    grouped = df.groupby("Country")
    df["log_te_lag"] = grouped["log_te"].shift(lag)
    year_gap = df["Year"] - grouped["Year"].shift(lag)
    df.loc[year_gap != lag, "log_te_lag"] = np.nan
    return df.dropna(subset=["log_te_lag"]).reset_index(drop=True)


def fit_cluster(formula: str, data: pd.DataFrame):
    """OLS avec erreurs standard clustérisées par pays."""
    groups = pd.factorize(data["Country"])[0]
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})


def summarize(model, term: str) -> dict:
    ci = model.conf_int().loc[term]
    return {
        "coef": float(model.params[term]),
        "std_err": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "n_obs": int(model.nobs),
    }


def within_variation(data: pd.DataFrame) -> dict:
    """Part de la variation de ln(dépense) qui est intra-pays."""
    total_sd = data["log_te"].std()
    within_sd = (data["log_te"] - data.groupby("Country")["log_te"].transform("mean")).std()
    return {"sd_total": float(total_sd), "sd_within": float(within_sd)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    summaries: list[str] = []

    # --- Modèle 6 principal : retard de 1 an --------------------------------
    df1 = build_sample(lag=1)
    main_model = fit_cluster(f"{Y} ~ log_te_lag + {CONTROLS} + {FE}", df1)
    results["model6_lag1_log"] = summarize(main_model, "log_te_lag")
    summaries.append("=== Modèle 6 : ln(dépense) retardée de 1 an ===")
    summaries.append(str(main_model.summary().tables[1]).split("C(Country)")[0])

    # --- Comparaison à échantillon identique : effet contemporain en log -----
    contemp = fit_cluster(f"{Y} ~ log_te + {CONTROLS} + {FE}", df1)
    results["contemporaneous_log_same_sample"] = summarize(contemp, "log_te")

    # --- Autres horizons (2 et 3 ans) ---------------------------------------
    for lag in (2, 3):
        dfk = build_sample(lag=lag)
        mk = fit_cluster(f"{Y} ~ log_te_lag + {CONTROLS} + {FE}", dfk)
        results[f"model6_lag{lag}_log"] = summarize(mk, "log_te_lag")

    # --- Extension : interaction avec le statut de développement ------------
    df1["Developed"] = (df1["Status"].str.strip() == "Developed").astype(int)
    inter = fit_cluster(
        f"{Y} ~ log_te_lag + log_te_lag:Developed + {CONTROLS} + {FE}", df1
    )
    results["interaction_developed"] = {
        "slope_developing": summarize(inter, "log_te_lag"),
        "difference_developed_minus_developing": summarize(inter, "log_te_lag:Developed"),
    }

    # --- Diagnostic : variation intra-pays ----------------------------------
    results["within_variation_log_te"] = within_variation(df1)
    results["sample"] = {
        "rows_lag1": int(len(df1)),
        "countries_lag1": int(df1["Country"].nunique()),
    }

    (OUTPUT_DIR / "model6_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "model6_summary.txt").write_text("\n".join(summaries), encoding="utf-8")

    for name, res in results.items():
        print(name, json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
