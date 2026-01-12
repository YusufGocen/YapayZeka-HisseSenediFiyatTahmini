import pandas as pd
from datetime import date
from analysis.model_analysis import train_and_evaluate


# --------------------------------------------------
# SON 5 GÜNLÜK GEÇMİŞ TEST
# --------------------------------------------------
def last_5_days_test(close_prices, models):
    results_list = []
    today = date.today()
    close_prices = close_prices[close_prices.index.date <= today]
    last_5_days = close_prices.tail(5)

    for i in range(5):
        test_date = last_5_days.index[i].date()
        real_price = last_5_days.iloc[i, 0]

        train_prices = close_prices.iloc[:-(5 - i)]

        results, best_name, scaler, _, _ = train_and_evaluate(
            train_prices,
            models,
            silent=True
        )

        best_model = results[best_name]["model"]

        last_known_price = train_prices.iloc[-1, 0]
        last_input = scaler.transform([[last_known_price]])
        predicted_price = best_model.predict(last_input)[0]

        error = abs(real_price - predicted_price)
        error_pct = (error / real_price) * 100

        results_list.append({
            "Tarih": test_date,
            "Gerçek Fiyat": round(real_price, 2),
            "Tahmin Fiyat": round(predicted_price, 2),
            "Hata (TL)": round(error, 2),
            "Hata (%)": round(error_pct, 2)
        })

    df = pd.DataFrame(results_list)

    avg_mae = df["Hata (TL)"].mean()
    avg_pct = df["Hata (%)"].mean()

    return df, avg_mae, avg_pct





# --------------------------------------------------
# Moving Average
# --------------------------------------------------

def calculate_moving_averages(close_prices):
    close = close_prices["Close"]

    last_close = close.iloc[-1].item()
    ma5 = close.rolling(5).mean().iloc[-1].item()
    ma20 = close.rolling(20).mean().iloc[-1].item()

    return last_close, ma5, ma20


# --------------------------------------------------
# Bollinger Bandı
# --------------------------------------------------

def calculate_bollinger(close_prices, window=20):
    close = close_prices["Close"]

    ma = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return (
        lower.iloc[-1].item(),
        ma.iloc[-1].item(),
        upper.iloc[-1].item()
    )


def bollinger_status(price, lower, upper):
    if price > upper:
        return "ÜST BANT ÜZERİNDE"
    elif price < lower:
        return "ALT BANT ALTINDA"
    else:
        return "Bant İÇİNDE"


def last_5_days_bollinger_analysis(close_prices, models):
    from analysis.model_analysis import train_and_evaluate

    results_list = []

    last_5_days = close_prices.tail(5)

    for i in range(5):
        date = last_5_days.index[i].date()

        train_prices = close_prices.iloc[:-(5 - i)]

        results, best_name, scaler, _, _ = train_and_evaluate(
            train_prices,
            models,
            silent=True
        )

        best_model = results[best_name]["model"]

        last_close = train_prices.iloc[-1, 0]
        predicted = best_model.predict(
            scaler.transform([[last_close]])
        )[0]

        lower, mid, upper = calculate_bollinger(train_prices)

        status = bollinger_status(predicted, lower, upper)

        results_list.append((date, status))

    return results_list



# --------------------------------------------------
# Gelecek Gün Tahmini
# --------------------------------------------------

def future_prediction(best_model, scaler, test_price):

    choice = input(
        "\nYarın için kapanış fiyatı tahmini yapmak ister misiniz? (Evet/Hayır): "
    ).lower()

    if choice == "evet":
        future_input = scaler.transform([[test_price]])
        prediction = best_model.predict(future_input)[0]

        print("\n----------------------------------------")
        print("GELECEK TAHMİNİ")
        print("----------------------------------------")
        print(f"Yarın İçin Tahmin Edilen Kapanış Fiyat: {prediction:.2f} TL")
        print("(Bu değer istatistiksel bir tahmindir, yatırım tavsiyesi değildir.)")
    else:
        print("\nGelecek tahmini yapılmadı.")
