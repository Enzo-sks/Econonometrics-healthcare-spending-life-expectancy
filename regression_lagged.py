"""Conventional pooled OLS with previous-calendar-year expenditure."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--measure',default='Domestic general government health expenditure (% of GDP), SH.XPD.GHED.GD.ZS')
    parser.add_argument('--lag',type=int,default=1)
    args=parser.parse_args()
    data=pd.read_csv(args.input or ROOT/f'data/processed/official_expenditure_model_lag{args.lag}.csv')
    out=args.output or ROOT/f'output/official_expenditure_lag{args.lag}_ols'
    out.mkdir(parents=True,exist_ok=True)
    controls=['log_gdp','Schooling','Adult Mortality','is_developped']
    lag=args.lag
    lag_variable=f'total_expenditure_lag{lag}'
    terms=[lag_variable,*controls]
    required=['Life expectancy',*terms]
    if data[required].isna().any().any() or not np.isfinite(data[required].to_numpy()).all():
        raise ValueError('Missing or nonfinite regression data')
    if not data.Year.sub(data.expenditure_source_year).eq(lag).all():
        raise ValueError(f'Lag is not exactly {lag} years')
    model=sm.OLS(data['Life expectancy'],sm.add_constant(data[terms].astype(float))).fit()
    ci=model.conf_int()
    result=pd.DataFrame({'term':model.params.index,'coefficient':model.params.to_numpy(),
        'OLS_SE':model.bse.to_numpy(),'t_statistic':model.tvalues.to_numpy(),
        'p_value':model.pvalues.to_numpy(),'ci95_low':ci.iloc[:,0].to_numpy(),
        'ci95_high':ci.iloc[:,1].to_numpy()})
    result.to_csv(out/'coefficients.csv',index=False)
    (out/'ols_summary.txt').write_text(model.summary().as_text(),encoding='utf-8')
    # A same-sample comparison isolates timing from the loss of first-year records.
    comparison=[]
    for x in ['Total expenditure',lag_variable]:
        fit=sm.OLS(data['Life expectancy'],sm.add_constant(data[[x,*controls]].astype(float))).fit()
        comparison.append({'expenditure_variable':x,'n':int(fit.nobs),'coefficient':fit.params[x],
            'OLS_SE':fit.bse[x],'p_value':fit.pvalues[x],'r_squared':fit.rsquared})
    pd.DataFrame(comparison).to_csv(out/'same_sample_timing_comparison.csv',index=False)
    stats={'lag_years':lag,'n_observations':int(model.nobs),'n_countries':int(data.Country.nunique()),
        'year_min':int(data.Year.min()),'year_max':int(data.Year.max()),
        'r_squared':model.rsquared,'adjusted_r_squared':model.rsquared_adj,
        'covariance':'conventional OLS','country_fixed_effects':False,'year_fixed_effects':False,
        'expenditure_measure':args.measure}
    (out/'model_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    rows='\n'.join(f'| {r.term} | {r.coefficient:.6f} | {r.OLS_SE:.6f} | {r.p_value:.6g} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] |'
                   for r in result.itertuples(index=False))
    (out/'summary.md').write_text(f'''# {lag}-year-lag expenditure OLS

`LifeExpectancy_it = intercept + beta1 TotalExpenditure_i,t-{lag} + beta2 ln(GDPpc_it) + beta3 Schooling_it + beta4 AdultMortality_it + beta5 Developed_i + error_it`

{int(model.nobs)} observations, {data.Country.nunique()} countries; {data.Year.min()}–{data.Year.max()}. Conventional OLS, no fixed effects. Controls are current-year values. Expenditure measure: {args.measure}.

| Variable | Coefficient | OLS SE | p value | 95% CI |
| --- | ---: | ---: | ---: | --- |
{rows}

R2={model.rsquared:.6f}; adjusted R2={model.rsquared_adj:.6f}. The lag coefficient describes a conditional delayed association, not a causal effect. Conventional SE do not address repeated-country error dependence or heteroskedasticity. Adult mortality may be an expenditure channel. Same-sample contemporaneous comparison is saved separately; both models use exactly these observations.
''',encoding='utf-8')
    print(result.to_string(index=False)); print(json.dumps(stats,indent=2))
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__=='__main__':
    main()
