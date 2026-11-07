"""Prepare exact-calendar-year official expenditure lag on the no-lag sample."""
from pathlib import Path
import hashlib
import json
import argparse
import pandas as pd
from data_process import load_replaced_data

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lag',type=int,default=1)
    args=parser.parse_args()
    if args.lag<1:
        raise ValueError('Lag must be positive')
    lag=args.lag
    current=ROOT/'output/government_share_ols/regression_sample.csv'
    out=ROOT/f'output/government_share_lag{lag}_preparation'
    output=out/'regression_data.csv'
    out.mkdir(parents=True,exist_ok=True)
    full,replacement_audit,source_paths=load_replaced_data()
    hashes={**replacement_audit['input_hashes'],current.name:hashlib.sha256(current.read_bytes()).hexdigest()}
    sample=pd.read_csv(current)
    if full.duplicated(['Country','Year']).any():
        raise ValueError('Duplicate source keys')
    prev=full[['Country','Year','Total expenditure']].rename(
        columns={'Year':'expenditure_source_year','Total expenditure':f'total_expenditure_lag{lag}'})
    prev['Year']=prev.expenditure_source_year+lag
    merged=sample.merge(prev,on=['Country','Year'],how='left',sort=False,validate='one_to_one',indicator=True)
    missing=merged[f'total_expenditure_lag{lag}'].isna()
    dropped=merged.loc[missing,['Country','Year','_merge']].copy()
    dropped['reason']=dropped._merge.astype(str).map({'left_only':'no_previous_calendar_year_record','both':'previous_year_expenditure_missing'})
    dropped.to_csv(out/'excluded_rows.csv',index=False)
    valid=merged.loc[~missing].drop(columns='_merge').copy()
    valid['expenditure_source_year']=valid.expenditure_source_year.astype(int)
    assert valid.Year.sub(valid.expenditure_source_year).eq(lag).all()
    valid.to_csv(output,index=False)
    for p in [*source_paths,current]:
        assert hashes[p.name]==hashlib.sha256(p.read_bytes()).hexdigest()
    stats={'lag_years':lag,'initial_rows':len(sample),'removed_missing_lag':int(missing.sum()),'retained_rows':len(valid),
        'countries':int(valid.Country.nunique()),'excluded_reasons':dropped.reason.value_counts().to_dict(),
        'source_hashes':hashes,'measure':'Domestic government health expenditure (% general government expenditure), SH.XPD.GHED.GE.ZS',
        'note':f'Retains no-lag complete cases for comparison; uses full source for exact t-{lag} matching.'}
    (out/'audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
