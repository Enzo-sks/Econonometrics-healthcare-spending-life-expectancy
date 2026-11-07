"""Run six nonlinear OLS regressions without Adult Mortality.

Run: D:/Anaconda/python.exe regression_marginal_effect.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
X = 'Total expenditure'
CONTROLS = ['log_gdp', 'Schooling', 'is_developped']


def run_regression(data, name, nonlinear_term, controls):
    """Fit conventional OLS and print the requested regression statistics only."""
    terms = [X, nonlinear_term, *controls]
    design = sm.add_constant(data[terms].astype(float), has_constant='add')
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f'{name}: rank-deficient design matrix')
    model = sm.OLS(data['Life expectancy'], design).fit()
    ci = model.conf_int()
    coefficients = pd.DataFrame({
        'term': model.params.index,
        'coefficient': model.params.values,
        'OLS_SE': model.bse.values,
        't_statistic': model.tvalues.values,
        'p_value': model.pvalues.values,
        'ci95_low': ci[0].values,
        'ci95_high': ci[1].values,
    })
    print(f'\n{name}')
    print(coefficients.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    print(f'N = {int(model.nobs)}; Countries = {data.Country.nunique()}; '
          f'R2 = {model.rsquared:.6f}; Adjusted R2 = {model.rsquared_adj:.6f}')
    return {
        'model': name,
        'n': int(model.nobs),
        'countries': data.Country.nunique(),
        'X_coefficient': model.params[X],
        'nonlinear_term': nonlinear_term,
        'nonlinear_coefficient': model.params[nonlinear_term],
        'nonlinear_p_value': model.pvalues[nonlinear_term],
        'R2': model.rsquared,
        'adjusted_R2': model.rsquared_adj,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT/'data/processed/life_expectancy_final_clean.csv')
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data.columns = data.columns.str.strip()
    required = ['Country', 'Year', 'Status', 'Life expectancy', X, 'GDP', 'Schooling']
    if data[required].isna().any().any() or data.duplicated(['Country', 'Year']).any():
        raise ValueError('Missing model values or duplicate country-year records')
    if not data.GDP.gt(0).all() or not data[X].gt(0).all():
        raise ValueError('GDP and expenditure must be positive')
    if set(data.Status) != {'Developed', 'Developing'}:
        raise ValueError('Unexpected Status labels')
    data['log_gdp'] = np.log(data.GDP)
    data['is_developped'] = data.Status.eq('Developed').astype(int)
    if not np.isfinite(data[['Life expectancy', X, *CONTROLS]].to_numpy()).all():
        raise ValueError('Nonfinite model values')

    results = []

    # Model 1: Full sample, original expenditure plus its squared term.
    data['expenditure_squared'] = data[X]**2
    results.append(run_regression(data, 'overall_quadratic',
                                  'expenditure_squared', CONTROLS))

    # Model 2: Full sample, original expenditure plus its natural logarithm.
    data['log_expenditure'] = np.log(data[X])
    results.append(run_regression(data, 'overall_linear_log',
                                  'log_expenditure', CONTROLS))

    # Model 3: Full sample, original expenditure plus its reciprocal (1/X).
    data['inverse_expenditure'] = 1/data[X]
    results.append(run_regression(data, 'overall_linear_inverse',
                                  'inverse_expenditure', CONTROLS))

    # Model 4: Full sample, piecewise linear regression at the expenditure median.
    knot = float(data[X].median())
    data['expenditure_above_median'] = np.maximum(data[X]-knot, 0)
    row = run_regression(data, 'overall_piecewise',
                         'expenditure_above_median', CONTROLS)
    print(f'Piecewise knot = {knot:.6f} percentage points')
    row['knot_pp'] = knot
    results.append(row)

    # Model 5: Developed countries, expenditure plus its squared term.
    # Omit the development dummy because it is constant within this group.
    developed = data.loc[data.Status.eq('Developed')]
    results.append(run_regression(developed, 'developed_quadratic',
                                  'expenditure_squared', CONTROLS[:-1]))

    # Model 6: Developing countries, expenditure plus its squared term.
    # Omit the development dummy because it is constant within this group.
    developing = data.loc[data.Status.eq('Developing')]
    results.append(run_regression(developing, 'developing_quadratic',
                                  'expenditure_squared', CONTROLS[:-1]))

    comparison = pd.DataFrame(results)
    print('\nSIX-MODEL COMPARISON\n'+comparison.to_string(index=False))


if __name__ == '__main__':
    main()
