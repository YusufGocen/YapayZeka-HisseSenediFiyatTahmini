import yfinance as yf

# --------------------------------------------------
# Veri Çekme Fonksiyonu
# --------------------------------------------------
def load_stock_data(ticker):
    data = yf.download(ticker, start="2018-01-01")
    close_prices = data[["Close"]].dropna()
    return close_prices
