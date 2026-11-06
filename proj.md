### Research question

This project examines whether increasing general government health expenditure as a share of total government expenditure improves national life expectancy. It also investigates whether this relationship differs between developed and developing countries and whether there is evidence of diminishing marginal returns.

The main explanatory variable is `Total expenditure`, defined as general government expenditure on health as a percentage of total government expenditure.

### Data and cleaning

We use only the original Kaggle life expectancy dataset and do not merge any additional datasets.

The original dataset contains 2,938 country-year observations covering 2000–2015, with no duplicate country-year records.

The first cleaning stage removed:

- 10 observations with missing life expectancy;
- 226 observations with missing total expenditure.

This produced a baseline sample of 2,702 observations from 181 countries.

For models containing GDP and education controls, observations with missing GDP or schooling were removed. We also constructed the logarithm of GDP. The final control-variable sample contains 2,327 observations from 157 countries.

### Model 1: Simple pooled OLS

The first model is:
$$
LifeExpectancy_{it} =\beta_0+\beta_1TotalExpenditure_{it}+u_{it}
$$
The estimated coefficient on total expenditure is 0.8369, with a p-value below 0.001.

This indicates that a one-percentage-point difference in the healthcare expenditure share is associated with approximately 0.84 additional years of life expectancy.

However, this is only a simple association across countries and years and should not be interpreted as a causal effect.

### Model 2: Adding GDP and schooling

The second model controls for `log(GDP)` and `Schooling`:
$$
LifeExpectancy_{it} =\beta_0+\beta_1TotalExpenditure_{it} +\beta_2\log(GDP_{it}) +\beta_3Schooling_{it}+u_{it}
$$
The estimated total expenditure coefficient becomes -0.0514, with a p-value of 0.352.

The positive association found in the simple regression therefore disappears after controlling for income and education.

This suggests that the initial positive association may partly reflect differences in national development. Richer countries generally have better education, higher healthcare expenditure and longer life expectancy.

### Model 3: Country and year fixed effects

The third model adds country and year fixed effects:
$$
LifeExpectancy_{it} =\beta_1TotalExpenditure_{it} +\beta_2\log(GDP_{it}) +\beta_3Schooling_{it} +\alpha_i+\lambda_t+u_{it}
$$
Country fixed effects control for time-invariant characteristics of each country, including geography, history and long-term institutional differences.

Year fixed effects control for shocks affecting all countries in the same year.

The estimated total expenditure coefficient is -0.0323, with a p-value of 0.224 and a 95% confidence interval from -0.0843 to 0.0197.

The model does not provide evidence that increasing the healthcare expenditure share within the same country is associated with a significant contemporaneous increase in life expectancy.

### Model 4: Curvature and diminishing returns

To test for diminishing marginal returns, we added a quadratic expenditure term:
$$
 LifeExpectancy_{it} =\beta_1X_{it}+\beta_2X_{it}^2 +\text{controls}+\alpha_i+\lambda_t+u_{it}
$$


Here, $X$ is the centered value of total expenditure.

The quadratic coefficient is 0.0151, with a p-value of 0.119 and a 95% confidence interval from -0.0039 to 0.0341.

Positive diminishing marginal returns would normally require a negative quadratic coefficient. Instead, the estimated coefficient is positive, implying a weak U-shaped pattern. However, the quadratic term is not statistically significant, so the U-shaped relationship should not be treated as established.

The current results therefore do not support positive diminishing marginal returns.

### Model 5: Development-status interaction

To test whether the relationship differs between developed and developing countries, we estimated:
$$
LifeExpectancy_{it} =\beta_1TotalExpenditure_{it} +\beta_2(TotalExpenditure_{it}\times Developed_i) +\text{controls}+\alpha_i+\lambda_t+u_{it}
$$
The sample contains 129 developing countries and 28 developed countries.

The estimated slopes are:

- Developing countries: -0.0041, p-value 0.900;
- Developed countries: -0.0845, p-value 0.057;
- Difference between the two slopes: -0.0804, p-value 0.144.

Although the estimated slope is more negative for developed countries, the interaction test does not provide sufficient evidence that the two slopes are statistically different.

### Current conclusion

The results show a consistent pattern.

A simple comparison across countries indicates a significant positive association between the government healthcare expenditure share and life expectancy. However, this association disappears after controlling for income, education, country fixed effects and year fixed effects.

This suggests that the initial positive association primarily reflects differences in development levels across countries rather than changes in healthcare expenditure within the same country.

We also find no robust evidence of positive diminishing marginal returns or a statistically significant difference between developed and developing countries.

These results do not prove that healthcare expenditure has no effect on life expectancy. Possible explanations include delayed effects, limited within-country variation, measurement error and the fact that total expenditure measures a government budget share rather than per-capita healthcare expenditure.

### Next steps

The next stages of the analysis are:

1. Measure the amount of within-country variation in total expenditure.
2. Determine whether fixed-effects estimates rely on only a small number of countries with substantial expenditure changes.
3. Test whether the results are sensitive to extreme observations.
4. Estimate a lagged model because healthcare expenditure may take time to affect life expectancy.