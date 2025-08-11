# Rainfall Prediction (Practice Project)

This is a **practice machine learning project** where I built a simple **Rainfall Prediction Model** using historical weather data from Australia.  
The goal was to understand the workflow of creating, training, and evaluating a machine learning model.

## Dataset
- Source: `weatherAUS.csv`
- Contains historical weather data from various locations in Australia.
- Key features used in this model:
  - **Humidity3pm**
  - **Temp3pm**
  - **WindSpeed3pm**
- Target variable:
  - **Rainfall** (in millimeters)

## Model
- **Algorithm**: Linear Regression (from `scikit-learn`)
- **Libraries used**:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Steps in the Project
1. Load and clean the dataset (remove missing values).
2. Select relevant features and target variable.
3. Split data into training and testing sets.
4. Train a Linear Regression model.
5. Evaluate using RMSE and R² Score.
6. Visualize predictions.
7. Allow manual predictions for custom input values.

## Example Output
RMSE: 7.84
R² Score: 0.08
Predicted rainfall for [70, 25, 15] → 4.81 mm

## Purpose
This project is purely for learning and practicing **basic ML model creation**.  
It’s not optimized for real-world forecasting and should be seen as an educational example.

---
