import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


veri = {
    'EvYaşı': [1, 0, 3, 0, 5, 7, 8, 2, 1, 9, 4, 3, 2, 5, 6, 7, 3, 1, 10, 15, 7, 30, 25, 10, 15, 7, 30, 25],
    'OdaSayısı': [1, 8, 6, 3, 2, 1, 2, 3, 4, 3, 5, 4, 3, 4, 3, 5, 4, 3, 4, 3, 5, 4, 3, 4, 3, 5, 4, 3],
    'Büyüklük': [100, 150, 120, 200, 180, 100, 150, 120, 200, 180, 100, 150, 120, 200, 180, 100, 150, 120, 200, 180, 100, 150, 120, 200, 180, 100, 150, 120],
    'Fiyat': [800, 450, 620, 700, 680, 500, 750, 720, 600, 480, 500, 850, 820, 700, 580, 500, 550, 620, 200, 180, 100, 150, 120, 200, 180, 100, 150, 120]
}

df = pd.DataFrame(veri)


X = df[['EvYaşı', 'OdaSayısı', 'Büyüklük']]
Y = df['Fiyat']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=35)


model = LinearRegression()
model.fit(X_train, Y_train)

tahminler = model.predict(X_test)


mse = mean_squared_error(Y_test, tahminler)
mae = mean_absolute_error(Y_test, tahminler)

print("MSE:", mse)
print("MAE:", mae)


print("Gerçek Fiyatlar:", Y_test.values)
print("Tahmin Edilen Fiyatlar:", tahminler)


plt.scatter(df['EvYaşı'], df['Fiyat'])
plt.xlabel('Ev Yaşı')
plt.ylabel('Fiyat')
plt.title('Ev Yaşı ve Fiyat Arasındaki İlişki')
plt.show()
