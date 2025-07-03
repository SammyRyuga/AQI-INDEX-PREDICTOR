# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import scipy.stats as stats
import pylab

# Load the dataset
df = pd.read_csv('city_day.csv')

# Display basic information
print("Dataset Info:")
df.info()
print("\nDataset Description:")
print(df.describe())
print("\nColumns:", list(df.columns))

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...")

# Jointplots for key pollutants vs AQI
sns.jointplot(x="PM2.5", y="AQI", data=df, alpha=0.5)
plt.savefig('jointplot_pm25_aqi.png')
plt.close()

sns.jointplot(x="PM10", y="AQI", data=df, alpha=0.5)
plt.savefig('jointplot_pm10_aqi.png')
plt.close()

sns.jointplot(x="NO2", y="AQI", data=df, alpha=0.5)
plt.savefig('jointplot_no2_aqi.png')
plt.close()

sns.jointplot(x="O3", y="AQI", data=df, alpha=0.5)
plt.savefig('jointplot_o3_aqi.png')
plt.close()

# Pairplot for all variables (Note: May be slow for large datasets)
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.5})
plt.savefig('pairplot_all.png')
plt.close()

# Linear model plot for PM2.5 vs PM10
sns.lmplot(y="PM2.5", x="PM10", data=df, scatter_kws={'alpha': 0.3})
plt.savefig('lmplot_pm25_pm10.png')
plt.close()

# Data Preprocessing
print("\nPreprocessing Data...")

# Check missing values before imputation
print("\nMissing values per column (Before):")
print(df.isnull().sum())

# Select features and target
features = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
numerical_features = features[1:]  # Exclude 'City'
X = df[features]
y = df['AQI']

# Drop rows with missing AQI
valid = y.notna()
X = X[valid]
y = y[valid]

# Encode City as dummy variables
X = pd.get_dummies(X, columns=['City'], drop_first=True)

# Impute missing values in numerical features with median
imputer = SimpleImputer(strategy='median')
X[numerical_features] = imputer.fit_transform(X[numerical_features])

# Check missing values after imputation
print("\nMissing values in features (After):")
print(X.isnull().sum().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Training the Model
print("\nTraining Linear Regression Model...")
lm = LinearRegression()
lm.fit(X_train, y_train)

# Display coefficients
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(cdf)

# Predictions
predictions = lm.predict(X_test)
print("\nSample Predictions (first 5):", predictions[:5])

# Model Evaluation
print("\nModel Evaluation Metrics:")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, predictions)))
print("R² Score:", r2_score(y_test, predictions))

# Example prediction for a sample input
sample = X_test.iloc[0].values.reshape(1, -1)
sample_pred = lm.predict(sample)[0]
print(f"\nSample Prediction: Predicted AQI = {sample_pred:.2f}")

# Visualizations
print("\nGenerating Visualizations...")

# Actual vs Predicted AQI
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.grid(True)
plt.savefig('actual_vs_predicted_aqi.png')
plt.close()

# Residuals
residuals = y_test - predictions

# Residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.savefig('residual_distribution.png')
plt.close()

# Q-Q Plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=pylab)
plt.title("Q-Q Plot of Residuals")
plt.savefig('qq_plot_residuals.png')
plt.close()

print("\nAll visualizations saved as PNG files.")