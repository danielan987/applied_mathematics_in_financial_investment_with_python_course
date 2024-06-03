import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Define the stock and market symbols
stock_symbol = 'AAPL'  
market_symbol = 'IVV' 

# Define the time period
start_date = '2020-01-01'
end_date = dt.datetime.now().strftime('%Y-%m-%d')

# Fetch the data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
market_data = yf.download(market_symbol, start=start_date, end=end_date)

# Calculate daily returns
stock_returns = stock_data['Adj Close'].pct_change().dropna()
market_returns = market_data['Adj Close'].pct_change().dropna()

# Align the data
returns = pd.concat([stock_returns, market_returns], axis=1).dropna()
returns.columns = ['Stock', 'Market']

# Prepare the data for regression
X = returns['Market'].values.reshape(-1, 1)
y = returns['Stock'].values

# Perform the linear regression
model = LinearRegression().fit(X, y)
alpha = model.intercept_
beta = model.coef_[0]

# Predict the stock returns using the market returns
predictions = model.predict(X)

# Calculate the Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y, predictions, squared=False)


# Plot the residuals to check for linearity
residuals = y - predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=returns['Market'], y=residuals, color='purple', label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Market Returns')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residuals Plot')
plt.show()


# Calculate the R-squared value
r_squared = r2_score(y, predictions)

# Metrics
print("Root Mean Squared Error:", rmse)
print(f'Alpha (intercept): {alpha}')
print(f'Beta (slope): {beta}')
print(f'R-squared: {r_squared}')

# Plot the regression line using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=returns['Market'], y=returns['Stock'], color='blue', label='Data points')
sns.lineplot(x=returns['Market'], y=predictions, color='red', label='Regression line')
plt.xlabel('Market Returns')
plt.ylabel('Stock Returns')
plt.legend()
plt.title(f'Regression of {stock_symbol} returns against {market_symbol} returns')
plt.show()
