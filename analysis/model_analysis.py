import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Model Eğitimi & Değerlendirme
# --------------------------------------------------
def train_and_evaluate(train_prices, models, silent=False):
    # Özellik & Etiket
    X = train_prices.shift(1).dropna()
    y = train_prices.iloc[1:]

    # Ölçekleme
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.1, shuffle=False
    )

    results = {}

    if not silent:
        print("\n----------------------------------------")
        print("MODEL PERFORMANSLARI")
        print("----------------------------------------")

    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "preds": preds,
            "y_test": y_test
        }

        if not silent:
            print(f"{name}")
            print(f"  MAE  : {mae:.2f} TL")
            print(f"  RMSE : {rmse:.2f} TL")
            print(f"  R²   : {r2:.2f}")
            print("-" * 40)

    best_name = min(results, key=lambda x: results[x]["MAE"])

    return results, best_name, scaler, X_test, y_test
