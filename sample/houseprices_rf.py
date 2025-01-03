import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('~/Desktop/[1]Masters/Intelligent Systems/Project/project/housing_data.csv')
df.head()

# Check for missing values
df.isnull().sum()
# Handle missing data (example: fill missing values with the mean)
df.fillna(df.mean(numeric_only=True), inplace=True)
# Encode categorical variables (e.g., neighborhood)
df = pd.get_dummies(df, drop_first=True)

# Plot correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
# Plot the distribution of house prices
sns.histplot(df['Price'], kde=True)
plt.show()

# Define features and target variable
X = df.drop('Price', axis=1)
y = df['Price']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# Initialize the model
rf_model = RandomForestRegressor()
# Define hyperparameters for tuning
param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [10, 20, 30],
'min_samples_split': [2, 5, 10]
}
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")