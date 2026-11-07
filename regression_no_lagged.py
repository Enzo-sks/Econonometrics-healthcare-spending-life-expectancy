
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'data/processed/life_expectancy_final_clean.csv'


def run_model(sample, name, terms):
    x = sm.add_constant(sample[terms].astype(float), has_constant='add')
    y = sample['Life expectancy']
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite model values')
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('Rank deficient design')
    model = sm.OLS(y, x).fit()
    ci = model.conf_int()
    result = pd.DataFrame({'term': model.params.index, 'coefficient': model.params.values,
        'standard_error': model.bse.values, 't_statistic': model.tvalues.values,
        'p_value': model.pvalues.values, 'ci95_low': ci[0].values, 'ci95_high': ci[1].values})
    beta = float(model.params['Total expenditure'])
    row = {'model': name, 'n': int(model.nobs), 'countries': sample.Country.nunique(),
        'expenditure_coefficient': beta, 'OLS_SE': model.bse['Total expenditure'],
        'p_value': model.pvalues['Total expenditure'],
        'ci95_low': ci.loc['Total expenditure', 0], 'ci95_high': ci.loc['Total expenditure', 1],
        'R2': model.rsquared, 'adjusted_R2': model.rsquared_adj}
    print('\n'+name+'\n'+result.to_string(index=False))
    print(f'N = {int(model.nobs)}; R2 = {model.rsquared:.6f}; Adjusted R2 = {model.rsquared_adj:.6f}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    args = parser.parse_args()
    df = pd.read_csv(args.input); df.columns = df.columns.str.strip()
    required = ['Life expectancy', 'Total expenditure', 'GDP', 'Schooling', 'Adult Mortality', 'Status']
    if df.duplicated(['Country', 'Year']).any():
        raise ValueError('Duplicate Country-Year records')
    df['Status_original'] = df.Status
    correction = df.Country.isin(['France', 'Canada']) & df.Status.ne('Developed')
    df.loc[correction, 'Status'] = 'Developed'
    for col in required[:-1]:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not set(df.Status.dropna()).issubset({'Developed', 'Developing'}):
        raise ValueError('Unrecognized development label')
    # Common sample: require mortality even for the model excluding that predictor.
    missing = df[required].isna().any(axis=1)
    nonpositive = df.GDP.notna() & df.GDP.le(0)
    excluded = missing | nonpositive
    sample = df.loc[~excluded].copy().reset_index(drop=True)
    sample['log_gdp'] = np.log(sample.GDP)
    sample['is_developped'] = sample.Status.eq('Developed').astype(int)
    specifications = [
        ('with_adult_mortality', ['Total expenditure', 'log_gdp', 'Schooling', 'Adult Mortality', 'is_developped']),
        ('without_adult_mortality', ['Total expenditure', 'log_gdp', 'Schooling', 'is_developped'])]
    comparison = pd.DataFrame([run_model(sample, name, terms) for name, terms in specifications])
    assert comparison.n.nunique() == 1
    print('\nCOMPARISON\n'+comparison.to_string(index=False))


if __name__ == '__main__':
    main()
