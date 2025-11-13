data(fertil1, package='wooldridge')

# Detailed OLS results including interaction terms
summary( lm(kids ~ educ + age + I(age^2) + black +
              east + northcen + west + farm + othrural + town + smcity +
              y74 + y76 + y78 + y80 + y82 + y84,  data=fertil1) )
