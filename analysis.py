import pandas as pd

def calculate_daily_returns(df):
    """
    Adds a 'Daily Return' column using 'Close' prices.
    """
    df['Daily Return'] = df['Close'].pct_change()
    return df


def calculate_rolling_volatility(df, window=7):
    """
    Adds a 'Rolling Volatility' column using standard deviation of daily returns.
    """
    if 'Daily Return' not in df.columns:
        df = calculate_daily_returns(df)
    df['Rolling Volatility'] = df['Daily Return'].rolling(window=window).std()
    return df

from sklearn.linear_model import LinearRegression
import numpy as np

def predict_next_day_price(df):
    """
    Predicts next day's price using linear regression on closing prices.
    """
    df = df.copy()
    df['day'] = np.arange(len(df))
    
    X = df[['day']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(df)]])
    predicted_price = model.predict(next_day)[0]
    
    return predicted_price
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def predict_lstm(df, look_back=10, epochs=10):
    """
    Predict next day's price using LSTM.
    """
    data = df['Close'].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create training data
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i - look_back:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # Predict next value
    last_sequence = data_scaled[-look_back:]
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    return predicted_price
def predict_last_n_days(df, look_back=10, predict_days=30):
    """
    Simulates predicting each of the last N days using rolling LSTM forecast.
    The model trains on data up to day i and predicts day i+1.
    """
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    import numpy as np

    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    actual = []
    predicted = []
    dates = []

    for i in range(len(scaled_data) - predict_days, len(scaled_data)):
        train_data = scaled_data[:i]
        X_train, y_train = [], []

        for j in range(look_back, len(train_data)):
            X_train.append(train_data[j - look_back:j, 0])
            y_train.append(train_data[j, 0])

        if len(X_train) == 0:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        last_window = scaled_data[i - look_back:i].reshape(1, look_back, 1)
        pred_scaled = model.predict(last_window, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        actual_price = data[i][0]
        predicted.append(pred_price)
        actual.append(actual_price)
        dates.append(df.index[i].date())


    return dates, np.array(actual), np.array(predicted)

def predict_future_days(df, look_back=10, future_days=7, epochs=10):
    """
    Predicts the next N days of prices using an LSTM model.
    """
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    import numpy as np
    import pandas as pd

    data = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create input sequence
    X = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, scaled_data[look_back:], epochs=epochs, batch_size=32, verbose=0)

    # Forecast future days
    forecast_input = scaled_data[-look_back:].reshape(1, look_back, 1)
    predicted_prices = []

    for _ in range(future_days):
        pred_scaled = model.predict(forecast_input, verbose=0)
        predicted_prices.append(pred_scaled[0, 0])
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred_scaled[0, 0]]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

    # Create dates for future
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

    return future_dates, predicted_prices
import ta

def add_technical_indicators(df):
    """
    Adds MA7, MA30, RSI, MACD to the DataFrame.
    """
    df = df.copy()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    # Add RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Add MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD Signal'] = macd.macd_signal()
    return df
