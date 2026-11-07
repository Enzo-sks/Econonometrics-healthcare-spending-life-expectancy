
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/raw'
PROCESSED=ROOT/'data/processed'
ALIASES={
    'Bahamas':'Bahamas, The','Bolivia (Plurinational State of)':'Bolivia',
    'Congo':'Congo, Rep.',"Côte d'Ivoire":"Cote d'Ivoire",
    "Democratic People's Republic of Korea":"Korea, Dem. People's Rep.",
    'Democratic Republic of the Congo':'Congo, Dem. Rep.','Egypt':'Egypt, Arab Rep.',
    'Gambia':'Gambia, The','Iran (Islamic Republic of)':'Iran, Islamic Rep.',
    'Kyrgyzstan':'Kyrgyz Republic',"Lao People's Democratic Republic":'Lao PDR',
    'Micronesia (Federated States of)':'Micronesia, Fed. Sts.',
    'Republic of Korea':'Korea, Rep.','Republic of Moldova':'Moldova',
    'Saint Kitts and Nevis':'St. Kitts and Nevis','Saint Lucia':'St. Lucia',
    'Saint Vincent and the Grenadines':'St. Vincent and the Grenadines',
    'Slovakia':'Slovak Republic','Swaziland':'Eswatini',
    'The former Yugoslav republic of Macedonia':'North Macedonia',
    'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',
    'United Republic of Tanzania':'Tanzania','United States of America':'United States',
    'Venezuela (Bolivarian Republic of)':'Venezuela, RB','Viet Nam':'Vietnam','Yemen':'Yemen, Rep.',
}
ALTERNATIVES={
    'Turkey':['Turkey','Turkiye','Türkiye'], 'Czechia':['Czechia','Czech Republic'],
    'Sao Tome and Principe':['Sao Tome and Principe','São Tomé and Príncipe'],
    "Côte d'Ivoire":["Côte d'Ivoire","Cote d'Ivoire","Côte d’Ivoire"],
    'Somalia':['Somalia','Somalia, Fed. Rep.'],
}
MODEL_COLUMNS=['Life expectancy','Total expenditure','GDP','Schooling','Adult Mortality','Status']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wide(path,years,skiprows=0,indicator=None):
    wide=pd.read_csv(path,skiprows=skiprows)
    wide.columns=wide.columns.str.strip()
    if indicator and set(wide['Indicator Code'].dropna())!={indicator}:
        raise ValueError(f'Unexpected indicator in {path.name}')
    cols=[c for c in wide if c.isdigit() and int(c) in years]
    return wide.melt(id_vars=['Country Name'],value_vars=cols,
                     var_name='year',value_name='value').rename(columns={'Country Name':'country_name'})


def replace_column(data,source,column,label):
    source=source[['country_name','year','value']].copy()
    source['country_name']=source.country_name.str.strip()
    year=pd.to_numeric(source.year,errors='raise')
    if year.isna().any() or not year.mod(1).eq(0).all():
        raise ValueError(f'Invalid years: {label}')
    source['year']=year.astype(int)
    source['value']=pd.to_numeric(source.value,errors='raise')
    if source.duplicated(['country_name','year']).any():
        raise ValueError(f'Duplicate country/year source keys: {label}')
    names=set(source.country_name)
    aliases=ALIASES.copy()
    for country,options in ALTERNATIVES.items():
        aliases[country]=next((n for n in options if n in names),country)
    keys=data[['Country','Year']].copy()
    keys['country_name']=keys.Country.map(lambda n:n if n in names else aliases.get(n,n))
    matched=keys.merge(source.rename(columns={'year':'Year'}),on=['country_name','Year'],
                       how='left',sort=False,validate='many_to_one',indicator=True)
    assert matched.Country.equals(data.Country) and matched.Year.equals(data.Year)
    log=matched[['Country','Year','country_name','value','_merge']].copy()
    log['old_value']=data[column]
    log['status']='replaced'
    log.loc[matched._merge.eq('left_only'),'status']='unmatched_country_year'
    log.loc[matched._merge.eq('both') & matched.value.isna(),'status']='source_missing'
    data[column]=matched.value.to_numpy()
    return log.status.value_counts().to_dict()


def load_replaced_data():
    """Replace the three original columns in memory; write no files."""
    paths=[RAW/'kaggle_life_expectancy.csv',RAW/'life_expectancy_by_country.csv',
           RAW/'gdp_per_capita.csv',RAW/'API_SH.XPD.GHED.GE.ZS_DS2_en_csv_v2_8352.csv']
    hashes={p.name:sha256(p) for p in paths}
    data=pd.read_csv(paths[0])
    data.columns=data.columns.str.strip()
    data['Country']=data.Country.str.strip()
    data['Year']=pd.to_numeric(data.Year,errors='raise')
    if data[['Country','Year']].isna().any().any() or data.duplicated(['Country','Year']).any():
        raise ValueError('Invalid country/year identifiers')
    untouched=data.drop(columns=['Life expectancy','GDP','Total expenditure']).copy()
    stats={'raw_rows':len(data),'raw_countries':int(data.Country.nunique()),'input_hashes':hashes}
    years=set(data.Year)
    stats['life_replacement']=replace_column(data,pd.read_csv(paths[1]),'Life expectancy','life')
    stats['gdp_replacement']=replace_column(data,read_wide(paths[2],years),'GDP','gdp_per_capita')
    stats['expenditure_replacement']=replace_column(data,read_wide(paths[3],years,skiprows=4,
            indicator='SH.XPD.GHED.GE.ZS'),'Total expenditure','expenditure')
    pd.testing.assert_frame_equal(data.drop(columns=['Life expectancy','GDP','Total expenditure']),untouched)
    return data,stats,paths


def main():
    PROCESSED.mkdir(parents=True,exist_ok=True)
    data,stats,paths=load_replaced_data()
    hashes=stats['input_hashes']
    expenditure_missing=data['Total expenditure'].isna()
    expenditure_clean=data.loc[~expenditure_missing].copy()
    # Same user-confirmed classification correction used by the latest regression.
    corrected=expenditure_clean.Country.isin(['France','Canada']) & expenditure_clean.Status.ne('Developed')
    expenditure_clean.loc[corrected,'Status']='Developed'
    expenditure_clean['Status']=expenditure_clean.Status.astype('string').str.strip().replace('',pd.NA)
    if not set(expenditure_clean.Status.dropna()).issubset({'Developed','Developing'}):
        raise ValueError('Unexpected Status value')
    for c in MODEL_COLUMNS[:-1]:
        expenditure_clean[c]=pd.to_numeric(expenditure_clean[c],errors='raise')
    missing=expenditure_clean[MODEL_COLUMNS].isna().any(axis=1)
    numeric=expenditure_clean[MODEL_COLUMNS[:-1]]
    invalid=(numeric.notna() & ~np.isfinite(numeric)).any(axis=1)
    bad_gdp=expenditure_clean.GDP.notna() & expenditure_clean.GDP.le(0)
    exclude=missing|invalid|bad_gdp
    final=expenditure_clean.loc[~exclude].sort_values(['Country','Year']).reset_index(drop=True)
    target=PROCESSED/'life_expectancy_final_clean.csv'
    final.to_csv(target,index=False)
    final['log_gdp']=np.log(final.GDP)
    final['is_developped']=final.Status.eq('Developed').astype(int)
    assert not final[MODEL_COLUMNS].isna().any().any()
    assert np.isfinite(final[MODEL_COLUMNS[:-1]+['log_gdp']].to_numpy()).all()
    assert len(data)-len(final)==int(expenditure_missing.sum())+int(exclude.sum())
    for path in paths:
        assert sha256(path)==hashes[path.name]
    print("done")


if __name__=='__main__':
    main()
