import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 📌 En üstte olması gereken Streamlit sayfa yapılandırması
st.set_page_config(page_title="📊 Garanti Bankası Finansal Tahmin Sistemi", layout="wide")

# Makine öğrenmesi modelini yükleme
try:
    model = joblib.load("models/finans_tahmin_modeli.pkl")
except:
    model = None
    st.warning("📢 Uyarı: Model dosyası yüklenemedi. Lütfen model dosyanızın doğru klasörde olduğundan emin olun.")

st.title("📊 Garanti Bankası Finansal Tahmin Sistemi")

st.markdown("Bu uygulama, **Garanti Bankası'nın finansal göstergelerine dayalı tahminler yapar.**")

# Kullanıcı girişleri
faiz_orani = st.number_input("📌 TCMB Faiz Oranı (%)", min_value=0.0, max_value=50.0, value=30.0)
enflasyon_orani = st.number_input("📌 Yıllık Enflasyon (%)", min_value=0.0, max_value=100.0, value=50.0)

# Tahmin butonu
if st.button("📈 Tahmin Yap"):
    if model:
        df_yeni = pd.DataFrame({"faiz_orani": [faiz_orani], "enflasyon_orani": [enflasyon_orani]})
        tahmin = model.predict(df_yeni)
        st.success(f"📈 Tahmini Net Kâr: {tahmin[0]:,.2f} Milyar TL")
    else:
        st.error("❌ Model dosyası bulunamadı. Lütfen modeli oluşturup 'models' klasörüne ekleyin.")
