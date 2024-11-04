import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data", alpha=0.5)
# Лінійна регресія
plt.plot(X, y_lin_pred, color="green", label="Linear Regression")
# Поліноміальна регресія
sorted_idx = X[:, 0].argsort()
plt.plot(X[sorted_idx], y_poly_pred[sorted_idx], color="red", label="Polynomial Regression (degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Лінійна та поліноміальна регресія")
plt.legend()
plt.show()

# Оцінка моделей
mse_linear = mean_squared_error(y, y_lin_pred)
r2_linear = r2_score(y, y_lin_pred)

mse_poly = mean_squared_error(y, y_poly_pred)
r2_poly = r2_score(y, y_poly_pred)


print(f"Лінійна регресія - MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}")
print(f"Поліноміальна регресія - MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}, coef{poly_reg.coef_}, intercept{poly_reg.intercept_}")
