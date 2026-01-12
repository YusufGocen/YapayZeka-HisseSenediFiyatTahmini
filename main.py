from data.data import load_stock_data
from models.models import get_models
from analysis.model_analysis import train_and_evaluate
from forecast.forecast import future_prediction
from visual.visual import plot_results
from forecast.forecast import last_5_days_test
from forecast.forecast import calculate_moving_averages,calculate_bollinger, bollinger_status,last_5_days_bollinger_analysis


# --------------------------------------------------
# BIST 30 Listesi
# --------------------------------------------------
BIST30 = [
    "AKBNK", "ARCLK", "ASELS", "BIMAS", "DOHOL",
    "EKGYO", "EREGL", "FROTO", "GARAN", "GUBRF",
    "HEKTS", "ISCTR", "KCHOL", "KOZAL", "PETKM",
    "PGSUS", "SAHOL", "SISE", "TAVHL", "TCELL",
    "THYAO", "TOASO", "TTKOM", "TUPRS", "YKBNK"
]

print("=" * 65)
print("BIST 30 - YAPAY ZEKA İLE HİSSE FİYAT TAHMİN SİSTEMİ")
print("=" * 65)

for i, stock in enumerate(BIST30, 1):
    print(f"{i} - {stock}")

choice = int(input("\nAnaliz edilecek hisse numarasını giriniz: "))
stock = BIST30[choice - 1]
ticker = stock + ".IS"

print(f"\nSeçilen Hisse: {stock}")

# --------------------------------------------------
# Veri Çekme
# --------------------------------------------------
close_prices = load_stock_data(ticker)

# --------------------------------------------------
# GEÇMİŞ TEST İÇİN SON GÜNÜ AYIR
# --------------------------------------------------
test_date = close_prices.index[-1].date()
test_price = close_prices.iloc[-1, 0]
train_prices = close_prices.iloc[:-1]

print(f"Tahmin Edilecek Tarih : {test_date}")
print(f"Gerçek Kapanış Fiyatı : {test_price:.2f} TL")

# --------------------------------------------------
# Model Eğitimi
# --------------------------------------------------
models = get_models()
results, best_name, scaler, X_test, y_test = train_and_evaluate(train_prices, models)

best_model = results[best_name]["model"]

print("\nEN BAŞARILI MODEL:", best_name)

# --------------------------------------------------
# SON 5 GÜNLÜK ORTALAMA PERFORMANS
# --------------------------------------------------
print("\n========================================")
print("SON 5 GÜNLÜK ORTALAMA PERFORMANS")
print("========================================")

df_5, avg_mae, avg_pct = last_5_days_test(
    close_prices,
    models
)

print(df_5.to_string(index=False))

print("\n----------------------------------------")
print(f"- Ortalama Mutlak Hata (MAE): {avg_mae:.2f} TL")
print(f"- Ortalama Yüzdelik Hata    : %{avg_pct:.2f}")

# --------------------------------------------------
# Grafik
# --------------------------------------------------
print("\nGrafikler oluşturuluyor...")
print("- Gerçek vs Tahmin Değerleri Grafiği")
print("- En iyi model için zaman serisi karşılaştırması")

plot_results(
    results[best_name]["y_test"],
    results[best_name]["preds"],
    stock,
    best_name
)

# --------------------------------------------------
# Moving Average
# --------------------------------------------------

last_close, ma5, ma20 = calculate_moving_averages(close_prices)
print("----------------------------------------")
print("\nHAREKETLİ ORTALAMA DEĞERLERİ")
print("----------------------------------------")
print(f"Son Kapanış Fiyatı          : {last_close:.2f} TL")
print(f"5 Günlük Hareketli Ortalama : {ma5:.2f} TL")
print(f"20 Günlük Hareketli Ortalama: {ma20:.2f} TL")

last_price = close_prices.iloc[-1, 0]

if last_price > ma5:
    print("Durum: Fiyat 5 günlük ortalamanın ÜZERİNDE")
else:
    print("Durum: Fiyat 5 günlük ortalamanın ALTINDA")

# --------------------------------------------------
# Bollinger Bandı
# --------------------------------------------------

lower, mid, upper = calculate_bollinger(close_prices)

predicted_price = results[best_name]["model"].predict(
    scaler.transform([[last_close]])
)[0]

status = bollinger_status(predicted_price, lower, upper)

print("\nBOLLINGER BANDI DEĞERLERİ")
print("----------------------------------------")
print(f"Alt Bant  : {lower:.2f} TL")
print(f"Orta Bant : {mid:.2f} TL")
print(f"Üst Bant  : {upper:.2f} TL")
print(f"\nTahmin Edilen Fiyat : {predicted_price:.2f} TL")
print(f"Durum: {status}")

#Orta Bant = Son 20 günün kapanış fiyatlarının ortalaması
#Üst Bant = Orta Bant + (2 × Standart Sapma)
#Alt Bant = Orta Bant − (2 × Standart Sapma)

analysis_5 = last_5_days_bollinger_analysis(close_prices, models)

print("\nSON 5 GÜNLÜK TAHMİNLERİN BOLLINGER ANALİZİ")
print("----------------------------------------")

for d, s in analysis_5:
    print(f"{d} → {s}")

# --------------------------------------------------
# Gelecek Tahmini
# --------------------------------------------------

future_prediction(best_model, scaler, test_price)
