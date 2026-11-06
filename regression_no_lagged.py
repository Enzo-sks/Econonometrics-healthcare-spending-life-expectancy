"""Requested pooled OLS on externally replaced life expectancy and GDP per capita.

Run: D:/Anaconda/python.exe regression_updated_data.py
"""
from pathlib import Path
import hashlib
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'data/processed/life_expectancy_final_clean.csv'
OUT=ROOT/'output/official_expenditure_ols'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--expenditure-measure',default='Domestic general government health expenditure (% of GDP), SH.XPD.GHED.GD.ZS')
    args=parser.parse_args()
    # Local paths permit reusing the same specification for distinct source versions.
    input_path=args.input
    output_dir=args.output
    output_dir.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(input_path.read_bytes()).hexdigest()
    df=pd.read_csv(input_path)
    df.columns=df.columns.str.strip()
    required=['Life expectancy','Total expenditure','GDP','Schooling','Adult Mortality','Status']
    if df.duplicated(['Country','Year']).any():
        raise ValueError('Duplicate Country-Year records')
    df['Status_original']=df.Status
    correction=df.Country.isin(['France','Canada']) & df.Status.ne('Developed')
    df.loc[correction,'Status']='Developed'
    df.loc[correction,['Country','Year','Status_original','Status']].to_csv(output_dir/'status_corrections.csv',index=False)
    for col in required[:-1]:
        df[col]=pd.to_numeric(df[col],errors='raise')
    if not set(df.Status.dropna()).issubset({'Developed','Developing'}):
        raise ValueError('Unrecognized development label')
    missing=df[required].isna().any(axis=1)
    nonpositive=df.GDP.notna() & df.GDP.le(0)
    excluded=missing|nonpositive
    removed=df.loc[excluded,['Country','Year']].copy()
    removed['missing_model_variables']=df.loc[excluded,required].apply(lambda r:';'.join(r.index[r.isna()]),axis=1)
    removed['nonpositive_GDP']=nonpositive[excluded]
    removed.to_csv(output_dir/'removed_rows.csv',index=False)
    sample=df.loc[~excluded].copy().reset_index(drop=True)
    sample['log_gdp']=np.log(sample.GDP)
    sample['is_developped']=sample.Status.eq('Developed').astype(int)
    terms=['Total expenditure','log_gdp','Schooling','Adult Mortality','is_developped']
    x=sm.add_constant(sample[terms].astype(float),has_constant='add')
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(sample['Life expectancy']).all():
        raise ValueError('Nonfinite model values')
    if np.linalg.matrix_rank(x)!=x.shape[1]:
        raise ValueError('Design matrix is rank deficient')
    model=sm.OLS(sample['Life expectancy'],x).fit()
    ci=model.conf_int()
    result=pd.DataFrame({'term':model.params.index,'coefficient':model.params.to_numpy(),
        'standard_error':model.bse.to_numpy(),'t_statistic':model.tvalues.to_numpy(),
        'p_value':model.pvalues.to_numpy(),'ci95_low':ci.iloc[:,0].to_numpy(),'ci95_high':ci.iloc[:,1].to_numpy()})
    result.to_csv(output_dir/'ols_coefficients.csv',index=False)
    sample.to_csv(output_dir/'regression_sample.csv',index=False)
    (output_dir/'ols_summary.txt').write_text(model.summary().as_text(),encoding='utf-8')
    control=x.drop(columns='Total expenditure')
    xr=sm.OLS(sample['Total expenditure'],control).fit().resid
    yr=sm.OLS(sample['Life expectancy'],control).fit().resid
    beta=float(model.params['Total expenditure'])
    assert np.isclose((xr@yr)/(xr@xr),beta,atol=1e-9)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(xr,yr,s=10,alpha=.25,color='#2667a8')
    grid=np.linspace(xr.min(),xr.max(),100)
    ax.plot(grid,beta*grid,color='#c54a3c',label=f'Adjusted slope = {beta:.4f} years/pp')
    ax.axhline(0,color='gray',lw=.6); ax.axvline(0,color='gray',lw=.6)
    ax.set(xlabel='Total expenditure residual (percentage points)',
           ylabel='Life expectancy residual (years)',title='Adjusted expenditure–life expectancy association')
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(output_dir/'partial_regression.png',dpi=160); plt.close(fig)
    audit={'input':str(input_path),'input_sha256':digest,'input_rows':len(df),
        'expenditure_measure':args.expenditure_measure,
        'missing_by_model_variable':{c:int(df[c].isna().sum()) for c in required},
        'rows_with_missing_model_variables':int(missing.sum()),'nonpositive_GDP_rows':int(nonpositive.sum()),
        'excluded_rows':int(excluded.sum()),'sample_rows':len(sample),'countries':int(sample.Country.nunique()),
        'status_corrected_rows':int(correction.sum()),'developed_rows':int(sample.is_developped.sum()),
        'developing_rows':int(sample.is_developped.eq(0).sum()),
        'r_squared':model.rsquared,'adjusted_r_squared':model.rsquared_adj,
        'covariance':'conventional OLS','country_fixed_effects':False,'year_fixed_effects':False}
    (output_dir/'model_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    lines=['| Variable | Coefficient | OLS SE | p value | 95% CI |','| --- | ---: | ---: | ---: | --- |']
    for r in result.itertuples(index=False):
        lines.append(f'| {r.term} | {r.coefficient:.6f} | {r.standard_error:.6f} | {r.p_value:.6g} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] |')
    text=f'''# Updated-data pooled OLS / 新数据OLS

`LifeExpectancy_it = beta0 + beta1 TotalExpenditure_it + beta2 ln(GDPperCapita_it) + beta3 Schooling_it + beta4 AdultMortality_it + beta5 is_developped_i + u_it`

Input: {input_path.name}. Expenditure measure: {args.expenditure_measure}. Natural logarithm of GDP per capita. is_developped=1 for Developed and 0 for Developing, with prior user-confirmed France/Canada corrections. Only model-variable missingness or nonpositive GDP causes exclusion; no imputation or numerical reconstruction. No fixed effects, interactions or quadratic terms. Conventional OLS only.

{len(df)} input rows; {int(excluded.sum())} excluded; {len(sample)} observations, {sample.Country.nunique()} countries retained. Missing counts overlap; see model_audit.json and removed_rows.csv.

{chr(10).join(lines)}

R2={model.rsquared:.5f}; adjusted R2={model.rsquared_adj:.5f}.

The expenditure coefficient is life-expectancy years per expenditure percentage point, conditional on the four controls. Its conventional p value is {model.pvalues['Total expenditure']:.6g}. A positive coefficient is not sufficient to establish a positive relationship if uncertainty includes zero. GDP coefficient is years per log unit: a 1% GDP increase corresponds approximately to beta_logGDP/100 years. Developed coefficient is a level difference, not a slope interaction.

This is an association, not a causal effect. Adult mortality may be an outcome channel of expenditure, so conditioning on it can remove part of the relationship. Remaining source-data quality issues and repeated-country error dependence are not resolved by replacing GDP/life expectancy. Conventional SE do not address those issues.

The partial plot residualizes both expenditure and life expectancy on the other four regressors. Its slope is verified to equal the expenditure coefficient. Negative residuals do not represent negative original expenditure.
'''
    (output_dir/'summary.md').write_text(text,encoding='utf-8')
    assert digest==hashlib.sha256(input_path.read_bytes()).hexdigest()
    print(result.to_string(index=False))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
