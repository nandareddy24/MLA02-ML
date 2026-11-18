import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load housing dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("Actual Prices:", y_test[:10])
print("\nPredicted Prices:", y_pred[:10])

print("\nMean Squared Error:",
      mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

# Predict new house
new_house = X_test[:1]

prediction = model.predict(new_house)

print("\nPredicted House Price:",
      prediction[0], "Hundred Thousand Dollars")

# Plot graph
plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("House Price Prediction")
plt.grid()
plt.show()
