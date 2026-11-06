# 当前保留的代码

全部命令在项目目录运行，使用 `D:/Anaconda/python.exe`。

1. `python data_process.py`：读取四个原始CSV，替换寿命、人均GDP与政府卫生支出，按模型变量清理缺失。最终文件：`data/processed/life_expectancy_final_clean.csv`。保留原22列。
2. `python regression_updated_data.py`：最新数据的无滞后常规OLS，输出 `output/official_expenditure_ols`。
3. `python create_official_expenditure_lag.py --lag 1`：准备1年滞后；将1改为3可准备3年滞后。
4. `python regression_lagged_expenditure.py --lag 1`：运行1年滞后常规OLS；将1改为3可运行3年滞后。

先做无滞后回归，再准备滞后数据。滞后脚本使用完整的 `data_merged.csv` 匹配过去年份，不跨国家或跨错年份。

当前控制变量：log(GDP per capita)、Schooling、Adult Mortality、is_developped；不含固定效应，只报告常规OLS。
Total expenditure现在为政府卫生支出占GDP比例（SH.XPD.GHED.GD.ZS），不是占政府总支出的比例。

旧流程代码已归档，数据、原始文件和历史结果保留。
