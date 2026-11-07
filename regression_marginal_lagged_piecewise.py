"""Piecewise-linear OLS with lagged government health expenditure.

Implements the plan in todo.md:
    LifeExp_it = b0 + b1*X_{i,t-L} + b2*max(X_{i,t-L} - k, 0)
                 + g1*log(GDPpc_it) + g2*Schooling_it + g3*Developed_i + u_it
- k is fixed at the median of Total expenditure in the 2,702-row baseline sample (8.76).
- Contemporaneous (L=0), 1-year and 3-year lags are estimated on ONE COMMON SAMPLE
  (rows where X_t, X_{t-1} and X_{t-3} all exist), so differences come from timing only.
- Conventional OLS (as in the other scripts, no fixed effects, Adult Mortality excluded).
  Country-clustered standard errors are reported next to the OLS ones.

Run: python regression_lagged_piecewise.py [--input PATH] [--lags 0 1 3] [--knot 8.76]
Optional: --full-source builds lags from the full replaced source (needs data/raw files).
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data/processed/life_expectancy_final_clean.csv'
OUT = ROOT / 'output/lagged_piecewise'
X = 'Total expenditure'
Y = 'Life expectancy'
CONTROLS = ['log_gdp', 'Schooling', 'is_developped']


def load_base(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.duplicated(['Country', 'Year']).any():
        raise ValueError('Duplicate Country-Year records')
    df.loc[df.Country.isin(['France', 'Canada']), 'Status'] = 'Developed'
    for c in [Y, X, 'GDP', 'Schooling']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if not df.GDP.gt(0).all() or not df[X].gt(0).all():
        raise ValueError('GDP and expenditure must be positive')
    df['log_gdp'] = np.log(df.GDP)
    df['is_developped'] = df.Status.eq('Developed').astype(int)
    return df


def add_lags(df, lags, full_source=False):
    """Exact calendar-year lags: X_{t-L} is matched on Country and Year-L."""
    if full_source:
        from data_process import load_replaced_data
        source, _, _ = load_replaced_data()
    else:
        source = df
    base = source[['Country', 'Year', X]]
    for lag in lags:
        if lag == 0:
            df['x_L0'] = df[X]
            continue
        prev = base.rename(columns={X: f'x_L{lag}'}).assign(Year=lambda d: d.Year + lag)
        df = df.merge(prev, on=['Country', 'Year'], how='left', validate='one_to_one')
    return df


def coef_table(model, se_label):
    """Same layout as regression_marginal_effect.py."""
    ci = model.conf_int()
    return pd.DataFrame({
        'term': model.params.index,
        'coefficient': model.params.values,
        se_label: model.bse.values,
        't_statistic': model.tvalues.values,
        'p_value': model.pvalues.values,
        'ci95_low': ci[0].values,
        'ci95_high': ci[1].values,
    })


def fit(sample, lag, knot):
    """Fit OLS (conventional SE) and country-clustered SE; print in the same style as the other scripts."""
    xname = f'expenditure_lag{lag}'
    aname = f'expenditure_above_knot_lag{lag}'
    x = sample[f'x_L{lag}']
    d = pd.DataFrame({xname: x, aname: np.maximum(x - knot, 0),
                      **{c: sample[c] for c in CONTROLS}})
    design = sm.add_constant(d.astype(float), has_constant='add')
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f'lag {lag}: rank-deficient design matrix')
    ols = sm.OLS(sample[Y], design).fit()
    groups = pd.factorize(sample.Country)[0]
    clu = sm.OLS(sample[Y], design).fit(cov_type='cluster', cov_kwds={'groups': groups})
    countries = int(sample.Country.nunique())

    # Slope above the knot = b1 + b2, with its test.
    names = list(ols.params.index)
    contrast = np.zeros(len(names))
    contrast[[names.index(xname), names.index(aname)]] = 1
    t_ols, t_clu = ols.t_test(contrast), clu.t_test(contrast)
    high = float(np.squeeze(t_ols.effect))
    high_p_ols = float(np.squeeze(t_ols.pvalue))
    high_p_clu = float(np.squeeze(t_clu.pvalue))

    name = f'lag{lag}_piecewise'
    ff = lambda v: f'{v:.6g}'
    print(f'\n{name}')
    print(coef_table(ols, 'OLS_SE').to_string(index=False, float_format=ff))
    print(f'N = {int(ols.nobs)}; Countries = {countries}; '
          f'R2 = {ols.rsquared:.6f}; Adjusted R2 = {ols.rsquared_adj:.6f}')
    print(f'Slope above knot (b1+b2) = {high:.6f}; p-value (OLS) = {high_p_ols:.6g}')
    print(f'\n{name} - country-clustered standard errors')
    print(coef_table(clu, 'cluster_SE').to_string(index=False, float_format=ff))
    print(f'Slope above knot (b1+b2) = {high:.6f}; p-value (clustered) = {high_p_clu:.6g}')
    return {
        'model': name, 'lag': lag, 'n': int(ols.nobs), 'countries': countries,
        'X_coefficient': ols.params[xname],
        'nonlinear_term': aname,
        'nonlinear_coefficient': ols.params[aname],
        'nonlinear_p_value': ols.pvalues[aname],
        'nonlinear_p_value_cluster': clu.pvalues[aname],
        'slope_above_knot': high,
        'slope_above_p_value': high_p_ols,
        'slope_above_p_value_cluster': high_p_clu,
        'R2': ols.rsquared, 'adjusted_R2': ols.rsquared_adj,
        'AIC': ols.aic, 'BIC': ols.bic,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=INPUT)
    ap.add_argument('--lags', type=int, nargs='+', default=[0, 1, 3])
    ap.add_argument('--knot', type=float, default=None,
                    help='Fixed threshold; default = baseline median of Total expenditure')
    ap.add_argument('--full-source', action='store_true')
    args = ap.parse_args()
    if any(l < 0 for l in args.lags):
        raise ValueError('Lags must be non-negative')

    base = load_base(args.input)
    knot = args.knot if args.knot is not None else float(base[X].median())
    data = add_lags(base.copy(), args.lags, args.full_source)
    common = data.dropna(subset=[f'x_L{l}' for l in args.lags]).reset_index(drop=True)
    print(f'Baseline rows: {len(base)}; common sample: {len(common)} rows, '
          f'{common.Country.nunique()} countries; fixed knot = {knot:.6f} percentage points')

    results = [fit(common, l, knot) for l in args.lags]
    comparison = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / 'lagged_piecewise_results.csv', index=False)
    pd.set_option('display.width', 250)
    print('\nLAG COMPARISON (common sample)\n' + comparison.to_string(index=False))


if __name__ == '__main__':
    main()
