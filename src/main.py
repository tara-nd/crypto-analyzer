from fetch_data import get_historical_market_data, get_live_price
from analysis import calculate_daily_returns, calculate_rolling_volatility, predict_lstm
from visualize import plot_price

def main():
    coin_id = 'bitcoin'
    
    # Get current price
    live_price = get_live_price(coin_id)
    print(f"💰 Live price of {coin_id.capitalize()}: ${live_price:.2f}")

    # Get historical data and prepare it
    df = get_historical_market_data(coin_id, days=90)
    df = calculate_daily_returns(df)
    df = calculate_rolling_volatility(df)

    # Predict using LSTM
    predicted_price = predict_lstm(df, look_back=10, epochs=20)
    print(f"🤖 LSTM Predicted next-day price: ${predicted_price:.2f}")

    # Plot current price trend
    plot_price(df, coin_id)

if __name__ == "__main__":
    main()
