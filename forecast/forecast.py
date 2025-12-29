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
# Geçmiş Test
# --------------------------------------------------

def past_test(model, scaler, train_prices, test_price, test_date, streamlit_mode=False):

    last_known_price = train_prices.iloc[-1, 0]
    last_input = scaler.transform([[last_known_price]])

    predicted_price = model.predict(last_input)[0]
    error = abs(test_price - predicted_price)

    if streamlit_mode:
        return {
            "date": test_date,
            "real": test_price,
            "predicted": predicted_price,
            "error": error
        }

    else:
        print("\nGEÇMİŞ TEST SONUCU")
        print(f"Tarih          : {test_date}")
        print(f"Gerçek Fiyat   : {test_price:.2f}")
        print(f"Tahmin         : {predicted_price:.2f}")
        print(f"Hata           : {error:.2f}")


    if streamlit_mode:
        return predicted_price, error

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
