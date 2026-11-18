import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Display results
print("Linear Regression")
print("MSE:", mean_squared_error(y, linear_pred))
print("R2 Score:", r2_score(y, linear_pred))

print("\nPolynomial Regression")
print("MSE:", mean_squared_error(y, poly_pred))
print("R2 Score:", r2_score(y, poly_pred))

# Plot graph
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, linear_pred, color='red', label='Linear Regression')
plt.plot(X, poly_pred, color='green', label='Polynomial Regression')

plt.title("Linear vs Polynomial Regression")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.grid()
plt.show()
