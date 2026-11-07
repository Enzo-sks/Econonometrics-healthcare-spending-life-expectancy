All six models use conventional OLS, exclude **Adult Mortality**, and include no country or year fixed effects. Overall models control for log GDP per capita, Schooling, and development status; within-group models control for the first two.

Let $X$ denote government healthcare expenditure as a percentage of total government expenditure. In the table, $b$ is the coefficient on $X$, and $c$ is the coefficient on the additional nonlinear term.

| Model                          | Healthcare expenditure specification | $b$       | $c$       | p-value for $c$ | $R^2$    | Observations |
| ------------------------------ | ------------------------------------ | --------- | --------- | --------------- | -------- | ------------ |
| Overall quadratic              | $bX+cX^2$                            | 0.161948  | 0.001216  | 0.6141          | 0.695015 | 2,702        |
| Overall linear plus logarithm  | $bX+c\log(X)$                        | 0.176937  | 0.149959  | 0.7621          | 0.694997 | 2,702        |
| Overall linear plus reciprocal | $bX+c/X$                             | 0.196036  | 0.216609  | 0.8768          | 0.694989 | 2,702        |
| Overall piecewise linear       | $bX+c\max(X-8.7600,0)$               | 0.270742  | −0.112662 | 0.1372          | 0.695236 | 2,702        |
| Developed-country quadratic    | $bX+cX^2$                            | −0.219782 | 0.013506  | **0.0129**      | 0.740654 | 496          |
| Developing-country quadratic   | $bX+cX^2$                            | 0.151079  | 0.002536  | 0.3385          | 0.599440 | 2,206        |

None of the additional nonlinear terms in the overall models is significant at the 10% level. The developed-country quadratic term is significantly positive, indicating increasing rather than diminishing slopes. The developing-country quadratic term is not significant. Therefore, these results do **not consistently support diminishing marginal returns**.

For the piecewise model, the estimated slopes are:

| Quantity                                           | Estimate  | 95% confidence interval | p-value |
| -------------------------------------------------- | --------- | ----------------------- | ------- |
| Slope below 8.76%                                  | 0.270742  | [0.157317, 0.384166]    | <0.001  |
| Slope above 8.76%                                  | 0.158080  | [0.092073, 0.224087]    | <0.001  |
| High-expenditure slope minus low-expenditure slope | −0.112662 | [−0.261232, 0.035908]   | 0.1372  |

Both slopes are significantly positive, but their difference is not statistically significant.

For the next stage, we will use the **piecewise linear specification with lagged healthcare expenditure**:

$$
\begin{aligned} LifeExpectancy_{it} ={}&\beta_0+\beta_1X_{i,t-L} +\beta_2\max(X_{i,t-L}-k,0)\\ &+\gamma_1\log(GDPpc_{it}) +\gamma_2Schooling_{it} +\gamma_3Developed_i+u_{it}, \end{aligned}
$$
where $L$ is the lag length.

We choose this specification because:

- **It is easy to interpret:** $\beta_1$ measures the low-expenditure slope, while $\beta_1+\beta_2$ measures the high-expenditure slope.
- **It directly tests slope attenuation:** a negative $\beta_2$ indicates a smaller slope at higher expenditure levels, without imposing that result.
- **It has the best in-sample fit among the four overall nonlinear models**, with the lowest AIC and BIC. However, its current curvature evidence remains insufficient.

We will compare contemporaneous, one-year-lagged, and three-year-lagged expenditure using a **common sample and a fixed threshold**. This will help assess whether the association emerges with a delay. Lagging expenditure alone does not establish causality, and all lag results should be reported rather than selecting only significant ones.

  

2:03