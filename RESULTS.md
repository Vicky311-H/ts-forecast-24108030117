\# Experiment Results



Student ID: 24108030117



Fingerprint: `e461b6a28e6d`



\## Main Results



| Model | CV MASE Mean | Holdout MASE |

|---|---:|---:|

| Seasonal Naive | 0.776118 | 0.911098 |

| Ridge-AR | 0.885758 | 0.544148 |

| NLinear | 0.951650 | 0.586816 |



Seasonal Naive achieved the lowest mean CV MASE, while Ridge-AR achieved the lowest terminal holdout MASE.



\## Leakage Counterfactual



For NLinear:



\- Correct CV MASE: 0.951650

\- Leaky CV MASE: 0.866085

\- Apparent improvement: 8.99%



The leaky version uses statistics computed from the complete sequence, including future observations, so the apparent improvement cannot be treated as valid forecasting performance.



\## Residual Diagnostics



For all models and all folds, the lag-1 Ljung-Box test gave p-values greater than 0.05.



Therefore, no significant lag-1 residual autocorrelation was detected. Because each forecast horizon contains only four residuals, this should not be interpreted as proof that the residuals are white noise.



\## Supplemental Task



Mean horizon-wise CV MAE:



\- Direct: 1.968077

\- Recursive: 2.163857



Direct reduced the mean horizon-wise CV MAE by approximately 9.05%, although Recursive performed better at horizons 1 and 4.

