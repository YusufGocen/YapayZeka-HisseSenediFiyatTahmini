import matplotlib.pyplot as plt

# --------------------------------------------------
# Grafik Çizimi
# --------------------------------------------------
def plot_results(y_test, preds, stock, model_name, streamlit_mode=False):

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(y_test.values, label="Gerçek Kapanış Fiyatı")
    ax.plot(preds, label="Model Tahmini")

    ax.set_title(f"{stock} - Gerçek vs Tahmin ({model_name})")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Fiyat (TL)")

    ax.legend()
    ax.grid(True)

    if streamlit_mode:
        return fig
    else:
        plt.show()
