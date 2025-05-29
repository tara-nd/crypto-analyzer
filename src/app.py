import streamlit as st
from fetch_data import get_live_price, get_historical_market_data
from analysis import (
    calculate_daily_returns,
    calculate_rolling_volatility,
    predict_lstm,
    predict_last_n_days,
    predict_future_days,
    add_technical_indicators
)
from visualize import (
    plot_price,
    plot_daily_returns,
    plot_volatility,
    plot_actual_vs_predicted,
    plot_future_forecast,
    plot_moving_averages,
    plot_rsi,
    plot_macd
)

# === App Config ===
st.set_page_config(page_title="Crypto Analyzer", layout="wide")
st.title("🚀 Crypto Price Movement & Prediction Tool")

# === Controls ===
coin = st.selectbox("Select a Cryptocurrency", ["bitcoin", "ethereum", "dogecoin"])
days = st.slider("Select Days of Historical Data", 30, 180, 90)

# === Indicator Checkboxes ===
st.subheader("📐 Technical Indicators")
show_ma = st.checkbox("Show Moving Averages (MA7 & MA30)")
show_rsi = st.checkbox("Show RSI (Relative Strength Index)")
show_macd = st.checkbox("Show MACD")

# === Run Analysis ===
if st.button("Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        live_price = get_live_price(coin)
        st.success(f"💰 Live price of {coin.capitalize()}: ${live_price:.2f}")

        df = get_historical_market_data(coin, days=days)
        df = calculate_daily_returns(df)
        df = calculate_rolling_volatility(df)
        df = add_technical_indicators(df)

        # === Summary ===
        st.subheader("📊 Summary Panel")
        if len(df) >= 2:
            price_yesterday = df['Close'].iloc[-2]
            price_today = df['Close'].iloc[-1]
            price_change = ((price_today - price_yesterday) / price_yesterday) * 100
        else:
            price_change = 0
            price_today = df['Close'].iloc[-1]

        volatility = df['Rolling Volatility'].iloc[-1]
        vol_level = "Low" if volatility < 0.01 else "Medium" if volatility < 0.03 else "High"

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${price_today:,.2f}", f"{price_change:+.2f}%")
        col2.metric("24h Change", f"{price_change:+.2f}%", delta_color="inverse")
        col3.metric("Volatility", vol_level)

        # === Charts ===
        st.subheader("📈 Price Chart")
        plot_price(df, coin)

        st.subheader("📊 Daily Returns")
        plot_daily_returns(df, coin)

        st.subheader("📉 Rolling Volatility")
        plot_volatility(df, coin)

        if show_ma:
            plot_moving_averages(df, coin)
        if show_rsi:
            plot_rsi(df, coin)
        if show_macd:
            plot_macd(df, coin)

        # === Predictions ===
        st.subheader("🤖 LSTM Prediction")
        predicted = predict_lstm(df, look_back=10, epochs=20)
        st.success(f"Predicted next-day price: ${predicted:.2f}")

        st.subheader("📉 Actual vs Predicted (Last 30 Days)")
        dates, actual, predicted_vals = predict_last_n_days(df, look_back=10, predict_days=30)
        plot_actual_vs_predicted(dates, actual, predicted_vals, title=f"{coin.capitalize()} - Last 30 Days Prediction Accuracy")

        st.subheader("🔮 Next 7 Days Price Forecast")
        future_dates, future_prices = predict_future_days(df, look_back=10, future_days=7, epochs=20)
        plot_future_forecast(future_dates, future_prices, title=f"{coin.capitalize()} - 7 Day LSTM Forecast")
        st.dataframe({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Predicted Price (USD)": [f"${p:,.2f}" for p in future_prices]
        })

        # === Chatbot Section ===
        st.subheader("💬 Ask CryptoBot")
        user_input = st.text_input("Ask me something about this crypto")
        if user_input:
            response = generate_response(user_input, coin, df, predicted, vol_level)
            st.info(response)

# === Chatbot Logic ===
def generate_response(user_input, coin, df, predicted_price, volatility):
    user_input = user_input.lower()

    if "price" in user_input and "today" in user_input:
        latest_price = df['Close'].iloc[-1]
        return f"The current price of {coin.capitalize()} is ${latest_price:,.2f}."

    if "predict" in user_input or "tomorrow" in user_input:
        return f"My model predicts {coin.capitalize()}'s price tomorrow will be around ${predicted_price:,.2f}."

    if "volatility" in user_input:
        return f"The recent market volatility for {coin.capitalize()} is categorized as '{volatility}'."

    if "help" in user_input:
        return "You can ask me:\n- What's the price today?\n- Predict tomorrow's price\n- How volatile is the market?"

    return "I'm not sure how to answer that. Try asking about the price, volatility, or prediction."
