import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100
square_footage = np.random.uniform(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
prices = 50000 + (150 * square_footage) + (20000 * bedrooms) + (15000 * bathrooms) + np.random.normal(0, 30000, n_samples)

data = pd.DataFrame({
    'Square_Footage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

X = data[['Square_Footage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("=" * 60)
print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 60)
print(f"\nModel Coefficients:")
print(f"  Square Footage: ${model.coef_[0]:.2f} per sq ft")
print(f"  Bedrooms: ${model.coef_[1]:.2f} per bedroom")
print(f"  Bathrooms: ${model.coef_[2]:.2f} per bathroom")
print(f"  Intercept: ${model.intercept_:.2f}")

print(f"\nTraining Metrics:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
print(f"  MAE: ${mean_absolute_error(y_train, y_pred_train):,.2f}")

print(f"\nTesting Metrics:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
print(f"  MAE: ${mean_absolute_error(y_test, y_pred_test):,.2f}")

sample_house = pd.DataFrame({
    'Square_Footage': [2500],
    'Bedrooms': [4],
    'Bathrooms': [2.5]
})
predicted_price = model.predict(sample_house)[0]
print(f"\nSample Prediction:")
print(f"  House: 2500 sq ft, 4 bedrooms, 2.5 bathrooms")
print(f"  Predicted Price: ${predicted_price:,.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(data['Square_Footage'], data['Price'], alpha=0.6)
axes[0].set_xlabel('Square Footage')
axes[0].set_ylabel('Price ($)')
axes[0].set_title('Square Footage vs Price')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(data['Bedrooms'], data['Price'], alpha=0.6, color='orange')
axes[1].set_xlabel('Bedrooms')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Bedrooms vs Price')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(y_test, y_pred_test, alpha=0.6, color='green')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Actual Price ($)')
axes[2].set_ylabel('Predicted Price ($)')
axes[2].set_title('Actual vs Predicted Prices')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'house_price_analysis.png'")