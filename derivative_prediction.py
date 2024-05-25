import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_derivatives(data):
    data['Close_1d'] = data['Close'].diff()  # First derivative
    data['Close_2d'] = data['Close_1d'].diff()  # Second derivative
    data.dropna(inplace=True)
    return data

def prepare_dataset(data):
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    features = data[['Close', 'Close_1d', 'Close_2d']]
    target = data['Target']
    return features, target

# Download and prepare data
stock_data = download_stock_data('AAPL', '2020-01-01', '2023-01-01')
stock_data = calculate_derivatives(stock_data)
X, y = prepare_dataset(stock_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
