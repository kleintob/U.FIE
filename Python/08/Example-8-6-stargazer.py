import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

k401ksubs = woo.dataWoo('401ksubs')

# subsetting data:
k401ksubs_sub = k401ksubs[k401ksubs['fsize'] == 1]

# Model 1: OLS with plain SE
reg_ols_plain = smf.ols(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                        data=k401ksubs_sub)
results_ols_plain = reg_ols_plain.fit()

# Model 2: OLS with robust SE (HC0)
reg_ols_robust = smf.ols(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                         data=k401ksubs_sub)
results_ols_robust = reg_ols_robust.fit(cov_type='HC0')

# Model 3: WLS with plain SE
wls_weight = list(1 / k401ksubs_sub['inc'])
reg_wls_plain = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                        weights=wls_weight, data=k401ksubs_sub)
results_wls_plain = reg_wls_plain.fit()

# Model 4: WLS with robust SE (HC0)
reg_wls_robust = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                         weights=wls_weight, data=k401ksubs_sub)
results_wls_robust = reg_wls_robust.fit(cov_type='HC0')

# Create stargazer table with all four models
stargazer = Stargazer([results_ols_plain, results_ols_robust, 
                       results_wls_plain, results_wls_robust])

# Customize the table
stargazer.title('Net Financial Assets Regression Results')
stargazer.custom_columns(['OLS (Plain SE)', 'OLS (Robust SE)', 
                         'WLS (Plain SE)', 'WLS (Robust SE)'], [1, 1, 1, 1])
stargazer.covariate_order(['Intercept', 'inc', 'I((age - 25) ** 2)', 'male', 'e401k'])
stargazer.rename_covariates({'Intercept': 'Constant', 
                            'inc': 'Income',
                            'I((age - 25) ** 2)': 'Age-25 squared',
                            'male': 'Male',
                            'e401k': '401k eligible'})
stargazer.show_degrees_of_freedom(False)

# Save HTML table to file
html_output = stargazer.render_html()
with open('Python/08/Example-8-6-table.html', 'w') as f:
    f.write('<html><head><style>table {font-family: Arial, sans-serif; border-collapse: collapse; margin: 20px;} td, th {padding: 8px; text-align: center;}</style></head><body>')
    f.write(html_output)
    f.write('</body></html>')
print("HTML table saved to: Python/08/Example-8-6-table.html")

# Also print LaTeX for journal submission
print("\n" + "="*80)
print("LaTeX output (for journal articles):")
print("="*80)
print(stargazer.render_latex())
