import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

k401ksubs = woo.dataWoo('401ksubs')

# subsetting data:
k401ksubs_sub = k401ksubs[k401ksubs['fsize'] == 1]

# OLS without robust SE:
reg_ols_plain = smf.ols(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                        data=k401ksubs_sub)
results_ols_plain = reg_ols_plain.fit()

# OLS with robust SE (HC0):
reg_ols = smf.ols(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                  data=k401ksubs_sub)
results_ols_robust = reg_ols.fit(cov_type='HC0')

# WLS without robust SE:
wls_weight = list(1 / k401ksubs_sub['inc'])
reg_wls = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                  weights=wls_weight, data=k401ksubs_sub)
results_wls_plain = reg_wls.fit()

# WLS with robust SE:
reg_wls_robust = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                         weights=wls_weight, data=k401ksubs_sub)
results_wls_robust = reg_wls_robust.fit(cov_type='HC0')


# Create journal-style regression table
def format_coef(coef, se, pval):
    """Format coefficient with stars and SE in parentheses"""
    stars = ''
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.1:
        stars = '*'
    
    coef_str = f'{coef:.4f}{stars}'
    se_str = f'({se:.4f})'
    return coef_str, se_str


# Variable names for display
var_labels = {
    'Intercept': 'Constant',
    'inc': 'Income',
    'I((age - 25) ** 2)': 'Age-25 squared',
    'male': 'Male',
    'e401k': '401k eligible'
}

# Build table
print('\n' + '='*90)
print('Table: Net Financial Assets Regression Results')
print('='*90)
print(f'{"Variable":<20} {"(1)":<15} {"(2)":<15} {"(3)":<15} {"(4)":<15}')
print(f'{"":<20} {"OLS":<15} {"OLS":<15} {"WLS":<15} {"WLS":<15}')
print(f'{"":<20} {"Plain SE":<15} {"Robust SE":<15} {"Plain SE":<15} {"Robust SE":<15}')
print('-'*90)

# Get all results
results = [results_ols_plain, results_ols_robust, results_wls_plain, results_wls_robust]

# Print coefficients
for var in results_ols_plain.params.index:
    label = var_labels.get(var, var)
    print(f'{label:<20}', end='')
    
    for res in results:
        coef_str, _ = format_coef(res.params[var], res.bse[var], res.pvalues[var])
        print(f'{coef_str:<15}', end='')
    print()
    
    # Print standard errors
    print(f'{"":<20}', end='')
    for res in results:
        _, se_str = format_coef(res.params[var], res.bse[var], res.pvalues[var])
        print(f'{se_str:<15}', end='')
    print()

print('-'*90)

# Print model statistics
print(f'{"R-squared":<20}', end='')
for res in results:
    print(f'{res.rsquared:.4f}{"":<10}', end='')
print()

print(f'{"Adj. R-squared":<20}', end='')
for res in results:
    print(f'{res.rsquared_adj:.4f}{"":<10}', end='')
print()

print(f'{"Observations":<20}', end='')
for res in results:
    print(f'{int(res.nobs):<15}', end='')
print()

print('='*90)
print('Notes: Standard errors in parentheses.')
print('*** p<0.01, ** p<0.05, * p<0.1')
print('='*90)