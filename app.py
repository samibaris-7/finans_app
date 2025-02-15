import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

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

        # 📊 Tahmin sonucunu göster
        st.success(f"📈 Tahmini Net Kâr: {tahmin[0]:,.2f} Milyar TL")

        # 📊 Grafik Ekleme (Daha Estetik)
        fig, ax = plt.subplots(figsize=(6, 4))  # Grafik boyutu artırıldı
        bars = ax.bar(["Tahmin Edilen Net Kâr"], [tahmin[0]], color="#4CAF50", alpha=0.85, edgecolor="black", linewidth=1.5)
        
        # Çubukların üstüne değerleri ekle
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:,.2f} Milyar TL", ha='center', fontsize=12, fontweight='bold', color="black")

        ax.set_ylabel("Milyar TL", fontsize=12, fontweight='bold')
        ax.set_title("📊 Finansal Tahmin Sonucu", fontsize=14, fontweight='bold')

        # Arka plana grid ekleyelim
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        st.pyplot(fig)

    else:
        st.error("❌ Model dosyası bulunamadı. Lütfen modeli oluşturup 'models' klasörüne ekleyin.")
