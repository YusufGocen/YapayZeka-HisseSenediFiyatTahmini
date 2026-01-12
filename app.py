import streamlit as st
import pandas as pd
import numpy as np
from forecast.forecast import last_5_days_test
from data.data import load_stock_data
from models.models import get_models
from analysis.model_analysis import train_and_evaluate
from forecast.forecast import past_test
from visual.visual import plot_results
from datetime import datetime, timedelta

# --------------------------------------------------
# Sayfa Ayarları
# --------------------------------------------------

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

st.set_page_config(
    page_title="Yapay Zeka ile Hisse Tahmini",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E6E9EF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .custom-label {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    .custom-value {
        font-size: 1.25rem;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 12px;
    }
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid rgba(0, 0, 0, 0.05);   /* ince çizgi */
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06); /* yumuşak gölge */
        transition: all 0.25s ease;
        height: 100%;
    }
    .card-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .card-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
    }
    .date-header {
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .error-badge {
        display: inline-block;
        margin-top: 10px;
        background-color: #FEE2E2;
        color: #DC2626;
        font-size: 13px;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 600;
    }
    div.stButton > button {
        background-color: #01549F;
        color: white;
        border-radius: 10px;
        height: 45px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #013E78;
        color: white;
    }
    .blue-card {
        background-color: #01549F;
        color: white;
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 15px;
    }
    .column-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 20px;
        height: 40px;
        display: flex;
        align-items: center;
    }
    .prediction-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 40px;
        margin-top: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid #F1F5F9;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03); /* Hafif iç gölge */
        width: 100%;
    }
    .prediction-label {
        color: #6B7280;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .prediction-value-container {
        display: flex;
        align-items: center;
        gap: 25px;
    }
    .prediction-price {
        font-size: 4.5rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -2px;
        line-height: 1;
    }
    .prediction-badge {
        padding: 10px 22px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .disclaimer-box {
        margin-top: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #854d0e;
        background-color: #fefce8;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #fef08a;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("## 📈 BIST 30 Hisseleri İçin Yapay Zeka ile Fiyat Tahmin Sistemi")

# --------------------------------------------------
# Hisse Seçimi
# --------------------------------------------------
BIST30 = [
    "AKBNK", "ARCLK", "ASELS", "BIMAS", "DOHOL",
    "EKGYO", "EREGL", "FROTO", "GARAN", "GUBRF",
    "HEKTS", "ISCTR", "KCHOL", "KOZAL", "PETKM",
    "PGSUS", "SAHOL", "SISE", "TAVHL", "TCELL",
    "THYAO", "TOASO", "TTKOM", "TUPRS", "YKBNK"
]

col_select, col_btn = st.columns([4, 1])

with col_select:
    stock = st.selectbox("Hisse Seçiniz", BIST30)

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Analizi Başlat")

# --------------------------------------------------
# Analiz
# --------------------------------------------------
if run:
    with st.spinner("Veriler alınıyor ve modeller eğitiliyor..."):
        ticker = stock + ".IS"
        close_prices = load_stock_data(ticker)

        test_date = close_prices.index[-1].date()
        test_price = close_prices.iloc[-1, 0]
        train_prices = close_prices.iloc[:-1]

        models = get_models()
        results, best_name, scaler, X_test, y_test = train_and_evaluate(train_prices, models)
        best_model = results[best_name]["model"]

        past = past_test(best_model, scaler, train_prices, test_price, test_date, streamlit_mode=True)
        df_5, avg_mae, avg_pct = last_5_days_test(close_prices, models)

        st.session_state["analysis_results"] = {
            "close_prices": close_prices,
            "test_date": test_date,
            "past": past,
            "best_model": best_model,
            "train_prices": train_prices,
            "df_5": df_5,
            "avg_mae": avg_mae,
            "avg_pct": avg_pct,
            "results": results,
            "best_name": best_name,
            "y_test": y_test,
            "stock": stock,
            "scaler": scaler
        }
        st.session_state.prediction_done = False
        st.success("Analiz tamamlandı")

# --------------------------------------------------
# GÖRÜNTÜLEME
# --------------------------------------------------
if st.session_state.analysis_results:
    r = st.session_state.analysis_results
    y_test = r["y_test"]
    results = r["results"]
    best_name = r["best_name"]
    stock = r["stock"]

    # ÜST KARTLAR
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='card'><div class='card-title'>Test Tarihi</div><div class='card-value'>{r['test_date']}</div></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='card'><div class='card-title'>Gerçek Kapanış</div><div class='card-value'>{r['past']['real']:.2f} TL</div></div>",
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div class='card'><div class='card-title'>Model Tahmini</div><div class='card-value'>{r['past']['predicted']:.2f} TL</div></div>",
            unsafe_allow_html=True)
    with c4:
        st.markdown(
            f"<div class='card'><div class='card-title'>Mutlak Hata</div><div class='card-value'>{r['past']['error']:.2f} TL</div></div>",
            unsafe_allow_html=True)

    #  SON 5 GÜNLÜK PERFORMANS

    st.subheader("📊 Son 5 Günlük Model Performansı")
    df_5 = r["df_5"]
    avg_mae = r["avg_mae"]
    avg_pct = r["avg_pct"]
    cols = st.columns(5)

    for i, row in df_5.reset_index().iterrows():
        with cols[i]:
            st.markdown(f"""
            <div class="custom-card" style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                <div class="date-header">📅 {row['Tarih']}</div>
                <div class="custom-label">Gerçek Fiyat</div>
                <div class="custom-value">{row["Gerçek Fiyat"]:.2f} TL</div>
                <div class="custom-label">Tahmin Fiyat</div>
                <div class="custom-value">{row["Tahmin Fiyat"]:.2f} TL</div>
                <div style="font-size: 0.9rem; font-weight: 600;">● {row["Hata (TL)"]:.2f} TL Hata</div>
                <div class="error-badge" style="margin-top: 10px; margin-bottom: 10px;">
                    Hata Payı: <span style="color:#1f1f1f;">%{row['Hata (%)']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>"f"""
    <div class="card">
        <div class="card-title">📌 Son 5 Günlük Ortalama</div>
        <div class="card-value">MAE: {avg_mae:.2f} TL</div>
        <div class="card-value">Ortalama Hata: %{avg_pct:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    #  GRAFİK VE PERFORMANS
    # ==============================================================================

    st.write("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown(
            '<br><p style="font-size:1.5rem; font-weight:700; color:#1F2937; margin-bottom:15px;">Gerçek vs Tahmin Grafiği</p>',
            unsafe_allow_html=True)

        fig = plot_results(y_test, results[best_name]["preds"], stock, best_name, streamlit_mode=True)
        if fig is not None:
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with right_col:
        st.markdown(
            '<br><br><p style="font-size:1.5rem; font-weight:700; color:#1F2937; margin-bottom:15px;">Model Performans Metrikleri</p>',
            unsafe_allow_html=True)

        perf_df = pd.DataFrame([
            {
                "Model": name,
                "MAE": round(res["MAE"], 2),
                "RMSE": round(res["RMSE"], 2),
                "R²": round(res["R2"], 2)
            }
            for name, res in results.items()
        ])

        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown(f'<div class="blue-card" style="margin-top:12px;">En Başarılı Model: {best_name}</div>',
                    unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

    # ==============================================================================
    #  GELECEK FİYAT FORECASTING
    # ==============================================================================

    st.write("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-weight:700; font-size:1.8rem; color:#1F2937; margin-bottom:15px;'>Gelecek Fiyat Tahmini.</div>",
        unsafe_allow_html=True)

    forecast_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    with st.container(border=True):
        col_txt, col_btn = st.columns([3.5, 1.5])

        with col_txt:
            st.markdown(
                f"<div style='font-weight:600; font-size:1.5rem; color:#1F2937;'>{forecast_date} Tarihi için Model Tahminini Oluşturun.</div>",
                unsafe_allow_html=True)

        with col_btn:
            if st.button("🚀 Analiz Et", type="primary", use_container_width=True, key="final_predict_btn"):
                st.session_state.prediction_done = True

        if st.session_state.get('prediction_done', False):
            try:
                r = st.session_state.analysis_results
                model, scaler, data = r["best_model"], r["scaler"], r["close_prices"]

                window_size = 60
                last_window = data.iloc[-window_size:].values
                last_window_scaled = scaler.transform(last_window)
                X_input = last_window_scaled.reshape(1, window_size, -1) if r["best_name"] in ["LSTM", "GRU"] else \
                last_window_scaled[-1].reshape(1, -1)

                t_pred_raw = model.predict(X_input)
                t_pred = float(t_pred_raw.item() if isinstance(t_pred_raw, np.ndarray) else t_pred_raw)
                l_price = float(data.iloc[-1, 0])
                c_pct = ((t_pred - l_price) / l_price) * 100

                badge_bg = '#eefbf3' if t_pred > l_price else '#fff5f5'
                badge_color = '#16A34A' if t_pred > l_price else '#DC2626'
                arrow = '↑' if t_pred > l_price else '↓'

                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-label">MODEL TAHMİNİ</div>
                        <div class="prediction-value-container">
                            <div class="prediction-price">₺{t_pred:.2f}</div>
                            <div class="prediction-badge" style="background-color: {badge_bg}; color: {badge_color};">
                                {arrow} %{abs(c_pct):.2f}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Hesaplama sırasında bir hata oluştu: {e}")

        st.markdown("""
            <div class="disclaimer-box">
                <span style="font-size: 20px;">⚠️</span>
                <span style="font-size: 0.9rem; font-weight:500;">
                    <strong>Yasal Uyarı:</strong> Bu değer istatistiksel bir tahmindir, yatırım tavsiyesi değildir.
                </span>
            </div>
        """, unsafe_allow_html=True)