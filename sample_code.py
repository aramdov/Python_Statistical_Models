### Normality Test for Linear Regression
result = anderson(workingsample['price'])
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    


### Polynomial explanatory variables with sklearn and fitting a linear regression, also change color of plot for visibility
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)
variable = train_set['variable'].values
x_dist = x_dist.reshape(-1,1)

variable_polynomial = polynomial_features.fit_transform(variable)
variable_polynomial.shape
y_real = df['y'].values

Poly_regression = sm.OLS(y_real, variable_polynomial).fit()
ypred_graph = test_poly.predict(variable_polynomial)
ypred_graph.shape
plt.scatter(train_set['variable'], y_real)
plt.plot(train_set['variable'], ypred_graph)
test_poly.summary()



### Vanilla Function to quickly test MAE of statsmodel regression on test set
def linearmodel_error(model):
    test_real = out_test_set['price']
    fn_test_set = out_test_set.copy()
    del fn_test_set['price']
    test_pred = model.predict(fn_test_set)
    modelMAE = mean_absolute_percentage_error(test_real, test_pred)
    return modelMAE
    
    
    
### Polynomial regression splines, cubic and natural, with plot of fit
from patsy import dmatrix
x_cubic = dmatrix('bs(x, knots=(1200, 2750, 3200))', {'x': train_set['variable']]}) # Kink points with knots = () parameter
fit_cubic = sm.GLM(train_set['y'], x_cubic).fit()    # can fit with GLM or OLS estimator
fit_cubic.summary()

x_natural = dmatrix('cr(x, knots=(1200, 2750, 3200))', {'x': train_set['variable']})
fit_natural = sm.GLM(train_set['y'], x_cubic).fit()
fit_natural.summary()

# For predicting on test set, transform test data 'variable' values into dmatrix with knots then use .predict function
# Example
test_x_cubic = dmatrix('bs(x, knots=(1200, 2750, 3200))', {'x': test_set['variable']})
ypred_x_cubic = ols_fit_cubic.predict(test_x_cubic)
mean_absolute_percentage_error(test_set['y'], ypred_x_cubic)

# Spline lines for graph
xp = np.linspace(train_set['variable'].min(), train_set['variable'].max())
line_cubic = fit_cubic.predict(dmatrix('bs(xp, knots=(1200, 2750, 3200))', {'xp': xp}))
line_natural = fit_natural.predict(dmatrix('bs(xp, knots=(1200, 2750, 3200))', {'xp': xp}))

plt.plot(xp, line_cubic, color='r', label='Cubic spline')
plt.plot(xp, line_natural, color='g', label='Natural spline')
plt.legend()
plt.scatter(train_set['variable'], train_set['y'], facecolor='None', edgecolor='k', alpha=0.05)
plt.xlabel('Variable')
plt.ylabel('Y Value')
plt.show()



### Linearity test for linear regressions
def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
