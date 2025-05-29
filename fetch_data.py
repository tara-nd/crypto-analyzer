from pycoingecko import CoinGeckoAPI
import pandas as pd

cg = CoinGeckoAPI()

def get_live_price(coin_id='bitcoin'):
    """
    Fetch current price in USD for a given coin ID.
    """
    data = cg.get_price(ids=coin_id, vs_currencies='usd')
    return data[coin_id]['usd']

def get_historical_market_data(coin_id='bitcoin', days=90):
    """
    Fetches daily historical market data (last N days).
    """
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df.drop(columns=['timestamp'])
    df['Close'] = df['price']
    df.drop(columns=['price'], inplace=True)
    return df

