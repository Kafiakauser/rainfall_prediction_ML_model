# Import libraries
import pandas as pd               # To handle dataset (like Excel/CSV data)
import numpy as np                # For numerical operations
from sklearn.model_selection import train_test_split  # To split data into training & testing
from sklearn.linear_model import LinearRegression     # Our ML algorithm
from sklearn.metrics import mean_squared_error, r2_score  # To check model performance
import matplotlib.pyplot as plt    # For plotting graphs
import seaborn as sns              # For better-looking plots

# 1. Load dataset (Excel file)
# Make sure the Excel file is in the same folder as this Python file
file_path = r"C:\Users\Hp\OneDrive\Documents\ML-model\weatherAUS.csv"
df = pd.read_csv(file_path)

# 2. Select only the columns we need
# Features: Humidity3pm, Temp3pm, WindSpeed3pm
# Target: Rainfall (continuous value in mm)
df = df[['Humidity3pm', 'Temp3pm', 'WindSpeed3pm', 'Rainfall']]

# 3. Remove rows with missing (NaN) values
# Models can't handle missing data, so we drop them
df.dropna(inplace=True)

# 4. Split into Features (X) and Target (y)
X = df[['Humidity3pm', 'Temp3pm', 'WindSpeed3pm']]  # Inputs
y = df['Rainfall']                                  # Output we want to predict

# 5. Split the data into training (80%) and testing (20%)
# random_state=42 → ensures the split is always the same every time you run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Create the Linear Regression model
model = LinearRegression()

# 7. Train (fit) the model on the training data
model.fit(X_train, y_train)

# 8. Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)

# 9. Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)   # Mean Squared Error
rmse = np.sqrt(mse)                        # Root Mean Squared Error (same units as rainfall)
r2 = r2_score(y_test, y_pred)               # R² Score (closer to 1 = better)

# Print results
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 10. Visualize the predictions vs actual values
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')  # Scatter plot
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")
plt.show()

# 11. Try a manual prediction
# Format: [[Humidity3pm, Temp3pm, WindSpeed3pm]]
sample_data = [[70, 25, 15]]  # Example: 70% humidity, 25°C temp, 15 km/h wind
predicted_rainfall = model.predict(sample_data)[0]
print(f"Predicted rainfall for {sample_data[0]} → {predicted_rainfall:.2f} mm")
