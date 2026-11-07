"""Conventional pooled OLS with previous-calendar-year expenditure."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent


def prepare_lag_data(lag):
    """Build the original exact-year lag sample in memory without writing files."""
    from data_process import load_replaced_data
    sample=pd.read_csv(ROOT/'data/processed/life_expectancy_final_clean.csv')
    sample.columns=sample.columns.str.strip()
    sample.loc[sample.Country.isin(['France','Canada']),'Status']='Developed'
    required=['Life expectancy','Total expenditure','GDP','Schooling','Adult Mortality','Status']
    sample=sample.loc[sample[required].notna().all(axis=1) & sample.GDP.gt(0)].copy()
    sample['log_gdp']=np.log(sample.GDP)
    sample['is_developped']=sample.Status.eq('Developed').astype(int)
    full,_,_=load_replaced_data()
    previous=full[['Country','Year','Total expenditure']].rename(columns={
        'Year':'expenditure_source_year','Total expenditure':f'total_expenditure_lag{lag}'})
    previous['Year']=previous.expenditure_source_year+lag
    merged=sample.merge(previous,on=['Country','Year'],how='left',validate='one_to_one')
    return merged.loc[merged[f'total_expenditure_lag{lag}'].notna()].reset_index(drop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--measure',default='Domestic general government health expenditure (% of general government expenditure), SH.XPD.GHED.GE.ZS')
    parser.add_argument('--lag',type=int,default=1)
    args=parser.parse_args()
    if args.lag<1:
        raise ValueError('Lag must be positive')
    data=pd.read_csv(args.input) if args.input else prepare_lag_data(args.lag)
    controls=['log_gdp','Schooling','Adult Mortality','is_developped']
    lag=args.lag
    lag_variable=f'total_expenditure_lag{lag}'
    terms=[lag_variable,*controls]
    required=['Life expectancy','Total expenditure',*terms]
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
    # A same-sample comparison isolates timing from the loss of first-year records.
    comparison=[]
    for x in ['Total expenditure',lag_variable]:
        fit=sm.OLS(data['Life expectancy'],sm.add_constant(data[[x,*controls]].astype(float))).fit()
        comparison.append({'expenditure_variable':x,'n':int(fit.nobs),'coefficient':fit.params[x],
            'OLS_SE':fit.bse[x],'p_value':fit.pvalues[x],
            'ci95_low':fit.conf_int().loc[x,0],'ci95_high':fit.conf_int().loc[x,1],
            'r_squared':fit.rsquared,'adjusted_r_squared':fit.rsquared_adj})
    print(result.to_string(index=False))
    print(f'N = {int(model.nobs)}; Countries = {data.Country.nunique()}; R2 = {model.rsquared:.6f}; Adjusted R2 = {model.rsquared_adj:.6f}')
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__=='__main__':
    main()
