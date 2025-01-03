import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, RocCurveDisplay
from sklearn.datasets import fetch_california_housing


df = pd.read_csv('housing_data.csv')
df.head()

# Check for missing values
df.isnull().sum()
df.fillna(df.mean(numeric_only=True), inplace=True)
# Encode categorical variables (e.g., neighborhood)
df = pd.get_dummies(df, drop_first=True)
data = {'Location_Countryside': [1, 0, 0, 1, 0],
        'Location_Downtown': [0, 1, 0, 0, 0],
        'Location_Suburb': [0, 0, 1, 0, 1],
        'Location_CityCenter': [0, 0, 0, 0, 0]}
location = pd.DataFrame(data)

# Define features and target variable
X = df.drop('Price', axis=1)
y = df['Price']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False,
random_state=42)

df['Location_Combined'] = location['Location_Countryside'].astype(str) + location['Location_Downtown'].astype(str) + location['Location_Suburb'].astype(str)
df['Location_Combined'] = df['Location_Combined'].replace({'100': 'Countryside', '010': 'Downtown', '001': 'Suburb', '000':'CityCenter'})
df = df.drop(['Location_Countryside', 'Location_Downtown', 'Location_Suburb'], axis=1)
#print(df)

# Initialize the model
rf_model = RandomForestRegressor()
# Define hyperparameters for tuning
param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [10, 20, 30],
'min_samples_split': [2, 5, 10]
}
rf_model.fit(X_train, y_train)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
# Best parameters and model performance
print(grid_search.best_params_)

# Predict on the test set
y_pred = grid_search.predict(X_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Predicted: {y_pred}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
