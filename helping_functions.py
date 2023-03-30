import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_sequences(data, n_steps):
    X = []
    y = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def data_scaling(data, scaler):
    df = pd.read_csv(data, parse_dates=["Date"])
    df.dropna(inplace=True)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.drop("Open", axis=1)
    df = df.drop("High", axis=1)
    df = df.drop("Low", axis=1)
    df = df.drop("Adj Close", axis=1)
    df = df.drop("Volume", axis=1)
    df = df.sort_values("Date").set_index("Date")
    # print(df)
    scaled_data_train = scaler.fit_transform(df)
    return scaled_data_train, df


def graph(predicted_prices,actual_prices=None, actual_prices_df = None):
    actual_prices = actual_prices[-len(predicted_prices):]
    actual_dates = actual_prices_df.index[-len(predicted_prices):]
    predicted_dates = actual_prices_df.index[-len(predicted_prices):]
    actual_dates = actual_dates[:len(actual_prices)]
    plt.plot(actual_dates, actual_prices[:, -1], label='Actual Value')
    plt.plot(predicted_dates, predicted_prices[:, -1], label='Predicted Value')
    plt.title('Actual vs Predicted Stock closing value')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
