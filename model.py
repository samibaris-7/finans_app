import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 📌 Örnek finansal veriler oluştur
data = {
    "faiz_orani": [10, 15, 20, 25, 30, 35, 40, 45],
    "enflasyon_orani": [5, 10, 15, 20, 25, 30, 35, 40],
    "net_kar": [50, 55, 60, 65, 70, 75, 80, 85]  # Örnek net kâr değerleri
}

df = pd.DataFrame(data)

# 📌 Bağımsız ve bağımlı değişkenleri ayır
X = df[["faiz_orani", "enflasyon_orani"]]
y = df["net_kar"]

# 📌 Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Lineer Regresyon Modelini Eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Modeli 'models' klasörüne kaydet
joblib.dump(model, "models/finans_tahmin_modeli.pkl")

print("✅ Model başarıyla oluşturuldu ve 'models' klasörüne kaydedildi.")


