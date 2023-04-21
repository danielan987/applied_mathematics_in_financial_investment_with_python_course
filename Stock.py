#set up
#git remote add origin https://github.com/danielan987/IVV_Stock.git

#checking

#update script to git
#git push --all

# https://python.plainenglish.io/measure-stock-volatility-using-betas-in-python-d6411612e7bd

import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

##make sure XIC.TO
####TO MAKE CHANGES, UPDATE THIS
# symbols = [stock, market]
symbols = ['IMCG', 'IVV']

# Create a dataframe of historical stock prices
# Enter dates as yyyy-mm-dd or yyyy-m-dd
# The date entered represents the first historical date prices will
#   be returned
data = yf.download(symbols, '2010-2-15')['Adj Close']
print(data)
# Convert historical stock prices to daily percent change
price_change = data.pct_change()
print(price_change)

# Deletes row one containing the NaN
df = price_change.drop(price_change.index[0])

####TO MAKE CHANGES, AND THIS
# Create arrays for x and y variables in the regression model
x = np.array(df['VWO']).reshape((-1, 1))
y = np.array(df['IVV'])

# Define the model and type of regression
model = LinearRegression().fit(x, y)

# Prints the beta to the screen
print('Beta: ', model.coef_)

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2)
# polynomial_features_array = poly.fit_transform(linear_features_array)
# model.fit(polynomial_features_array, y_train)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

# r2_score(y_true,y_predicted)
print(r2_score(x, y))

sns.regplot(data=data, x=x, y=y, marker='+')  # marker just to make it look pretty,not convention
plt.show()
# x_estimator=np.mean can also look cleaner, make sure to check for convention before using

# if it's straight and not slanted in one direction, support linearity assumption
sns.residplot(data=data, x=x, y=y)
plt.show()

# if it's curved, order=2 in sns.regplot()
