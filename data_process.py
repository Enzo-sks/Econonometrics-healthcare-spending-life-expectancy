"""Replace life expectancy, GDP per capita and official health spending; clean OLS data.

Run: D:/Anaconda/python.exe data_process.py
Only pandas and numpy are required. All inputs are read-only.
Total expenditure now means domestic government health expenditure (% of GDP),
SH.XPD.GHED.GD.ZS, NOT expenditure as a share of the government budget.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/raw'
PROCESSED=ROOT/'data/processed'
LOGS=ROOT/'output/data_process'
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
    log.to_csv(LOGS/f'{label}_replacement_log.csv',index=False)
    log.loc[matched.value.isna()].to_csv(LOGS/f'{label}_missing.csv',index=False)
    data[column]=matched.value.to_numpy()
    return log.status.value_counts().to_dict()


def main():
    PROCESSED.mkdir(parents=True,exist_ok=True)
    LOGS.mkdir(parents=True,exist_ok=True)
    paths=[RAW/'kaggle_life_expectancy.csv',RAW/'life_expectancy_by_country.csv',
           RAW/'gdp_per_capita.csv',RAW/'API_SH.XPD.GHED.GD.ZS_DS2_en_csv_v2_789.csv']
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
            indicator='SH.XPD.GHED.GD.ZS'),'Total expenditure','expenditure')
    pd.testing.assert_frame_equal(data.drop(columns=['Life expectancy','GDP','Total expenditure']),untouched)
    # Preserve a merged, unfiltered 22-column version before status corrections.
    data.to_csv(PROCESSED/'data_merged.csv',index=False)
    expenditure_missing=data['Total expenditure'].isna()
    data.loc[expenditure_missing,['Country','Year']].to_csv(LOGS/'missing_expenditure_rows.csv',index=False)
    expenditure_clean=data.loc[~expenditure_missing].copy()
    expenditure_clean.to_csv(PROCESSED/'data_expenditure_clean.csv',index=False)
    stats['missing_expenditure_rows_removed']=int(expenditure_missing.sum())
    # Same user-confirmed classification correction used by the latest regression.
    corrected=expenditure_clean.Country.isin(['France','Canada']) & expenditure_clean.Status.ne('Developed')
    status_log=expenditure_clean.loc[corrected,['Country','Year','Status']].copy()
    status_log['corrected_status']='Developed'
    status_log.to_csv(LOGS/'status_corrections.csv',index=False)
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
    dropped=expenditure_clean.loc[exclude,['Country','Year']].copy()
    dropped['missing_model_variables']=expenditure_clean.loc[exclude,MODEL_COLUMNS].apply(
        lambda r:';'.join(r.index[r.isna()]),axis=1)
    dropped['nonpositive_GDP']=bad_gdp[exclude]
    dropped['nonfinite_numeric']=invalid[exclude]
    dropped.to_csv(LOGS/'other_model_exclusions.csv',index=False)
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
    stats.update({'expenditure_clean_rows':len(expenditure_clean),'additional_model_rows_removed':int(exclude.sum()),
        'total_rows_removed':len(data)-len(final),'final_rows':len(final),'final_countries':int(final.Country.nunique()),
        'status_corrections':int(corrected.sum()),'year_min':int(final.Year.min()),'year_max':int(final.Year.max()),
        'model_variables':MODEL_COLUMNS,'output':str(target),'inputs_unchanged':True,
        'expenditure_indicator':'SH.XPD.GHED.GD.ZS: Domestic general government health expenditure (% GDP)'})
    (LOGS/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (LOGS/'README.md').write_text(f'''# 统一数据处理

运行：`D:/Anaconda/python.exe data_process.py`。独立文件，不依赖此前替换脚本或已生成的中间数据。

依次按国家年份替换寿命、人均GDP、政府卫生支出，使用明确国家别名，不做模糊匹配。源数据重复键直接报错。未匹配/源缺失留空，不回填旧数据、不插值、不做位数修复。只读原始文件并验证哈希不变。

1. data/processed/data_merged.csv：三列替换，保留全部2938行、原22列（列名去首尾空格）。
2. data/processed/data_expenditure_clean.csv：只删除新Total expenditure缺失，22列，不产生冗余指标列。
3. data/processed/life_expectancy_final_clean.csv：继续只删除当前OLS变量缺失/数值无效记录，按此前确认纠正法国加拿大Status，保留原22列。未使用变量缺失不删行；log_gdp和is_developped由回归脚本生成。

Total expenditure现为政府卫生支出占GDP比例，不是占政府预算比例。gdp_per_capita.csv未提供币种/价格基年元数据，不额外推断。数据替换不等于确认其他Kaggle列质量。

{json.dumps(stats,ensure_ascii=False,indent=2)}

所有替换及删行日志位于本目录。没有运行回归、创建滞后或修改原有分析入口。
''',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
