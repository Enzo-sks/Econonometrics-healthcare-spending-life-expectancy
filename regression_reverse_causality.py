"""Reverse causality and health-shock indicators.

Motivation (project proposal): a health crisis lowers life expectancy and the government
reacts by raising health spending. This pushes the expenditure/life-expectancy association
DOWN, so the true positive effect of spending would be underestimated.

Shock indicator: shock_t = 1 if life expectancy fell by at least SHOCK_DROP years
between t-1 and t (exact consecutive calendar years). Default SHOCK_DROP = 1.0.

PART A - Is there reverse causality? (does a shock make spending jump?)
    A1: dX_t ~ shock_t                      year fixed effects
    A2: dX_t ~ shock_t + shock_{t-1}        year fixed effects
    A3: dX_t ~ dLE_{t-1}                    year fixed effects
    (dX_t = X_t - X_{t-1}; first differences already remove country levels.)

PART B - Does the lagged-expenditure effect change once shocks are handled?
    Linear model with X_{t-1}, estimated pooled OLS and with country+year fixed effects.
    B0: baseline (X_{t-1} + log GDP + schooling [+ development status in pooled])
    B1: B0 + shock_{t-1} + HIV/AIDS_{t-1}
    B2: B1 on the sample WITHOUT rows where shock_{t-1} = 1
    B3: B1 on the sample WITHOUT countries that ever have a shock
    Only information dated t-1 or earlier is used as a control (no post-treatment controls).
    Rows are never dropped on the current outcome (shock_t) in B2, to avoid selecting on Y.

Standard errors are clustered by country. Run: python regression_reverse_causality.py
Options: --input PATH, --shock-drop 1.0
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data/processed/life_expectancy_final_clean.csv'
OUT = ROOT / 'output/reverse_causality'
X, Y, HIV = 'Total expenditure', 'Life expectancy', 'HIV/AIDS'


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.duplicated(['Country', 'Year']).any():
        raise ValueError('Duplicate Country-Year records')
    df.loc[df.Country.isin(['France', 'Canada']), 'Status'] = 'Developed'
    for c in [Y, X, 'GDP', 'Schooling', HIV]:
        df[c] = pd.to_numeric(df[c], errors='raise')
    df['log_gdp'] = np.log(df.GDP)
    df['is_developped'] = df.Status.eq('Developed').astype(int)
    return df


def lag(df, col, k, name):
    """Value of `col` in year t-k for the same country (exact calendar year, else NaN)."""
    prev = df[['Country', 'Year', col]].rename(columns={col: name}).assign(Year=lambda d: d.Year + k)
    return df.merge(prev, on=['Country', 'Year'], how='left', validate='one_to_one')


def build(df, drop):
    df = lag(df, Y, 1, 'le_l1')
    df = lag(df, Y, 2, 'le_l2')
    df = lag(df, X, 1, 'x_l1')
    df = lag(df, HIV, 1, 'hiv_l1')
    df['dX'] = df[X] - df.x_l1
    df['dLE'] = df[Y] - df.le_l1                  # change t-1 -> t
    df['dLE_l1'] = df.le_l1 - df.le_l2            # change t-2 -> t-1
    df['shock'] = np.where(df.dLE.notna(), (df.dLE <= -drop).astype(float), np.nan)
    df['shock_l1'] = np.where(df.dLE_l1.notna(), (df.dLE_l1 <= -drop).astype(float), np.nan)
    return df


def ols(data, y, terms, fe):
    """fe: 'pooled' | 'year' | 'twfe'. Returns fit with country-clustered SE and the key terms."""
    X_ = data[terms].astype(float)
    if fe in ('year', 'twfe'):
        X_ = X_.join(pd.get_dummies(data.Year, prefix='yr', drop_first=True, dtype=float))
    if fe == 'twfe':
        X_ = X_.join(pd.get_dummies(data.Country, prefix='cty', drop_first=True, dtype=float))
    X_ = sm.add_constant(X_, has_constant='add')
    if np.linalg.matrix_rank(X_) != X_.shape[1]:
        raise ValueError(f'Rank-deficient design for {terms} ({fe})')
    groups = pd.factorize(data.Country)[0]
    return sm.OLS(data[y], X_).fit(cov_type='cluster', cov_kwds={'groups': groups})


def table(model, terms):
    ci = model.conf_int()
    keep = [t for t in model.params.index if t in terms or t == 'const']
    return pd.DataFrame({'term': keep, 'coefficient': model.params[keep].values,
                         'cluster_SE': model.bse[keep].values,
                         't_statistic': model.tvalues[keep].values,
                         'p_value': model.pvalues[keep].values,
                         'ci95_low': ci.loc[keep, 0].values, 'ci95_high': ci.loc[keep, 1].values})


def run(data, name, y, terms, fe, key):
    m = ols(data, y, terms, fe)
    print(f'\n{name} [{fe}]')
    print(table(m, terms).to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    print(f'N = {int(m.nobs)}; Countries = {data.Country.nunique()}; R2 = {m.rsquared:.6f}')
    return {'model': name, 'fe': fe, 'n': int(m.nobs), 'countries': int(data.Country.nunique()),
            'key_term': key, 'coefficient': m.params[key], 'cluster_SE': m.bse[key],
            'p_value': m.pvalues[key], 'ci95_low': m.conf_int().loc[key, 0],
            'ci95_high': m.conf_int().loc[key, 1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=INPUT)
    ap.add_argument('--shock-drop', type=float, default=1.0,
                    help='Fall in life expectancy (years) that defines a shock')
    args = ap.parse_args()

    df = build(load(args.input), args.shock_drop)
    print(f'Shock = life expectancy falls by >= {args.shock_drop} year(s) vs previous year')
    ev = df[df.shock == 1]
    print(f'Shock country-years: {len(ev)} in {ev.Country.nunique()} countries')

    results_a, results_b = [], []

    # ---------------- PART A: reverse-causality tests ----------------
    a = df.dropna(subset=['dX', 'shock', 'shock_l1', 'dLE_l1']).reset_index(drop=True)
    print(f'\n===== PART A: does a health shock make spending jump? (N = {len(a)}) =====')
    results_a.append(run(a, 'A1_dX_on_shock_t', 'dX', ['shock'], 'year', 'shock'))
    results_a.append(run(a, 'A2_dX_on_shock_t_and_t-1', 'dX', ['shock', 'shock_l1'], 'year', 'shock'))
    results_a.append(run(a, 'A3_dX_on_dLE_t-1', 'dX', ['dLE_l1'], 'year', 'dLE_l1'))

    # ---------------- PART B: lagged expenditure with shock controls ----------------
    b = df.dropna(subset=['x_l1', 'shock_l1', 'hiv_l1', 'shock']).reset_index(drop=True)
    never = ~b.Country.isin(b.loc[(b.shock == 1) | (b.shock_l1 == 1), 'Country'].unique())
    base_pooled = ['x_l1', 'log_gdp', 'Schooling', 'is_developped']
    base_fe = ['x_l1', 'log_gdp', 'Schooling']
    print(f'\n===== PART B: effect of X_(t-1) on life expectancy (common sample N = {len(b)}) =====')
    for fe, base in [('pooled', base_pooled), ('twfe', base_fe)]:
        results_b.append(run(b, 'B0_baseline', Y, base, fe, 'x_l1'))
        results_b.append(run(b, 'B1_plus_shock_hiv', Y, base + ['shock_l1', 'hiv_l1'], fe, 'x_l1'))
        no_shock = b[b.shock_l1 == 0]
        results_b.append(run(no_shock, 'B2_drop_shock_rows', Y, base + ['hiv_l1'], fe, 'x_l1'))
        no_ctry = b[never]
        results_b.append(run(no_ctry, 'B3_drop_shock_countries', Y, base + ['hiv_l1'], fe, 'x_l1'))

    pd.set_option('display.width', 250)
    OUT.mkdir(parents=True, exist_ok=True)
    ra, rb = pd.DataFrame(results_a), pd.DataFrame(results_b)
    ra.to_csv(OUT / 'part_a_reverse_causality.csv', index=False)
    rb.to_csv(OUT / 'part_b_shock_controls.csv', index=False)
    print('\nPART A SUMMARY\n' + ra.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nPART B SUMMARY (coefficient on X_(t-1))\n' + rb.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
