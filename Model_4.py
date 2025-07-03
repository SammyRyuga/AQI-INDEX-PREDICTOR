#!/usr/bin/env python3
# Shebang

#AQI Index Predictor using Linear Regression

import pandas as pd  # For data manipul
import numpy as np  # For numerical ops
import matplotlib.pyplot as plt  # For plot
import seaborn as sns  # For plot
from sklearn.model_selection import train_test_split, cross_val_score  # For data splitting and crossvalidation
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For model eval
from sklearn.feature_selection import SelectKBest, f_regression  # For feature selection (unused in this version)
import math
import scipy.stats as stats  # used in Q-Q plot
import warnings
warnings.filterwarnings('ignore')  # No warnings, for clear consle

# AQI Predictor class
class AQIPredictor:
    def __init__(self):
        '''Initialize the AQI Predictor'''
        self.model = LinearRegression()  # Initialize LR model
        self.scaler = StandardScaler()  # Initialize feature scaler
        self.imputer = SimpleImputer(strategy='median')  # Uses medians
        self.feature_selector = None  # Placeholder for feature (not used yet)
        self.selected_features = None  # Placeholder for selected features (not used yet)
        self.city_encoder = None  # Placeholder for city (not used yet)
        self.is_trained = False  # Track if model is trained
    
    def load_and_preprocess_data(self, csv_path='city_day.csv'):
        # Display header
        print("=" * 60)
        print("🌍 AQI LINEAR REGRESSION PREDICTOR")
        print("=" * 60)
        
        # Load dataset 
        print("📊 Loading dataset...")
        self.df = pd.read_csv(csv_path)  # Read CSV into a pandas DataFrame
        
        # Confirm loaing
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Dataset shape: {self.df.shape}")  # Print number of rows and columns
        print(f"🏙️ Cities in dataset: {self.df['City'].nunique()}")  # Print number of unique cities
        
        # Display basic dataset info
        print("\n📋 Dataset Info:")
        print(f"Columns: {list(self.df.columns)}")  # List all column names
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")  # Show date range
        
        # Missing values
        print(f"\n🔍 Data Quality Check:")
        missing_counts = self.df.isnull().sum()  # Count missing values per column
        print("Missing values per column:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(self.df)*100:.1f}%)")  # Print columns with missing values and percentages
        
        return self.df  # Return the loaded DataFrame
    
    def prepare_features(self, target_col='AQI'):
        """Prepare features for modeling by cleaning data"""
        print("\n🔧 Preparing features for modeling...")
        
        # Defineing pollutant features
        pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        
        # Filter for features available in the dataset
        available_features = [col for col in pollutant_features if col in self.df.columns]
        print(f"📊 Available pollutant features: {available_features}")
        
        # Create feature matrix (X) and target vector (y)
        X = self.df[available_features + ['City']].copy()  # Include pollutants and City column
        y = self.df[target_col].copy()  # Copy AQI column as target
        
        # Remove rows where target (AQI) is missing
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"✅ Valid samples after removing missing AQI: {len(X)}")
        
        # Encoding cities
        print("🏙️ Encoding cities...")
        X_encoded = pd.get_dummies(X, columns=['City'], prefix='City', drop_first=True)  # Encode cities, drop first to avoid multicollinearity
        
        # Impute missing values in pollutant features
        print("🔄 Imputing missing values...")
        pollutant_cols = [col for col in available_features if col in X_encoded.columns]
        X_encoded[pollutant_cols] = self.imputer.fit_transform(X_encoded[pollutant_cols])  # Impute missing values with median
        
        print(f"✅ Final feature matrix shape: {X_encoded.shape}")
        
        return X_encoded, y, available_features  # Return encoded features, target, and available pollutant features
    
    def feature_analysis(self, X, y, pollutant_features):
        """Analyze feature importance through correlation analysis"""
        print("\n📊 Feature Analysis...")
        
        # Calculate correlation of each pollutant feature w AQI
        feature_importance = {}
        for feature in pollutant_features:
            if feature in X.columns:
                correlation = np.corrcoef(X[feature], y)[0, 1]  # Compute correlation
                feature_importance[feature] = abs(correlation)  # Store absolute correlation
        
        # Sort features by correlation (descendingorder)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Display feature importance
        print("✔️ Feature Importance (Correlation with AQI):")
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
        
        # Choose top 8 features
        top_features = [feature for feature, _ in sorted_features[:8]]
        
        return top_features, feature_importance  # Return top features 
    
    def train_model(self, X, y, test_size=0.2):
        """Train the LR model"""
        print("\n🤖 Training Linear Regression Model...")
        
        # Split data into training set and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )  # 80-20 split
        
        print(f"📊 Training set size: {len(X_train)}")
        print(f"📊 Testing set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit and transform training data
        X_test_scaled = self.scaler.transform(X_test)  # Transform test data using same scaler
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Store data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
        self.feature_names = [col for col in X.columns]  # Store feature names
        
        self.is_trained = True  # Mark model as trained
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred  # Return splits and predictions
    
    def evaluate_model(self):
        """Evaluate model performance using multiple metrics"""
        print("\n📈 Model Performance Evaluation:")
        print("=" * 40)
        
        # Calculate training metrics
        train_r2 = r2_score(self.y_train, self.y_train_pred)  # R2 score for training data
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)  # Mean Absolute Error
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))  # Root Mean Squared Error
        
        # Calculate testing metrics
        test_r2 = r2_score(self.y_test, self.y_test_pred)  # R2 score for test data
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)  # Mean Absolute Error
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))  # Root Mean Squared Error
        
        # Show training 
        print("🏋️ Training Performance:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        
        # Show testing 
        print("\n🎯 Testing Performance:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # Perform 5-fold cross-validation
        X_scaled = self.scaler.transform(self.X_train)  # Scale training data
        cv_scores = cross_val_score(self.model, X_scaled, self.y_train, cv=5, scoring='r2')  # Compute R² for 5 folds
        print(f"\n🔄 5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Interpret model performance
        if test_r2 > 0.8:
            print("✅ Excellent model performance!")
        elif test_r2 > 0.6:
            print("✅ Good model performance!")
        elif test_r2 > 0.4:
            print("⚠️ Moderate model performance")
        else:
            print("❌ Poor model performance - consider feature engineering")
        
        # Return metrics as a dict
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
    
    def analyze_coefficients(self):
        """Analyze and display model coefficients"""
        print("\n🔍 Linear Regression Coefficients Analysis:")
        print("=" * 50)
        
        # Model coefficients and intercept
        coefficients = self.model.coef_  # Coefficients for each feature
        intercept = self.model.intercept_  # Model intercept (bias term)
        
        # Create DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)  # Sort by absolute coefficient value
        
        # Display intercept and top 10 features
        print(f"📊 Model Intercept: {intercept:.4f}")
        print("\n🎯 Top 10 Most Important Features:")
        print(coef_df.head(10).to_string(index=False))
        
        # Explain coefficient interpretation
        print("\n💡 Coefficient Interpretation:")
        print("  Positive coefficients: Increase in feature leads to higher AQI")
        print("  Negative coefficients: Increase in feature leads to lower AQI")
        print("  Larger absolute values: More influence on AQI prediction")
        
        return coef_df  # Return coefficient DataFrame
    
    def create_visualizations(self):
        """Create graphs for model analysis"""
        print("\n📊 Generating Visualizations...")
        
        # Set matplotlib backend for non-interactive plotting
        plt.switch_backend('Agg')  # Use Agg backend for compatibility
        
        # Select plotting style
        available_styles = plt.style.available  # Get available matplotlib styles
        style = 'ggplot'  # Default style
        if 'seaborn-v0_8' in available_styles:
            style = 'seaborn-v0_8'  # Preferred style
        plt.style.use(style)  # Applying selected style
        sns.set_palette("husl")  # Setting colour
        
        # Creating figure with multiple subplots
        plt.figure(figsize=(20, 15))  # 6 subplots
        
        # Graph 1: Actual vs Predicted AQI (Training)
        plt.subplot(2, 3, 1)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue', label='Training Data')  # Scatter plot
        plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)  # Diagonal line
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Training: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 2: Actual vs Predicted AQI (Testing)
        plt.subplot(2, 3, 2)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green', label='Testing Data')  # Scatter plot
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)  # Diagonal line
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Testing: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 3: Residuals vs Predicted AQI
        plt.subplot(2, 3, 3)
        residuals = self.y_test - self.y_test_pred  # Calculate residuals
        plt.scatter(self.y_test_pred, residuals, alpha=0.6)  # Scatter plot of residuals
        plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line at y=0
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted AQI')
        plt.grid(True, alpha=0.3)
        
        # Graph 4: Residual Distribution
        plt.subplot(2, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')  # Histogram of residuals
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')  # Mean line
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 5: Q-Q Plot for Residuals
        plt.subplot(2, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)  # Q-Q plot to check normality
        plt.title('Q-Q Plot: Residuals vs Normal Distribution')
        plt.grid(True, alpha=0.3)
        
        # Graph 6: Feature Importance (Pollutant Coefficients)
        plt.subplot(2, 3, 6)
        coef_df = self.analyze_coefficients()  # Get coefficients
        top_features = coef_df.head(10)  # Select top 10 features
        
        # Filter for pollutant features (exclude cities)
        pollutant_coefs = top_features[~top_features['Feature'].str.contains('City')]
        if len(pollutant_coefs) > 0:
            plt.barh(pollutant_coefs['Feature'], pollutant_coefs['Coefficient'])  # Horizontal bar plot
            plt.xlabel('Coefficient Value')
            plt.title('Top Pollutant Feature Coefficients')
            plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()  # Subplot spacing
        plt.savefig('aqi_prediction_analysis.png', dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
        print("✅ Visualizations saved as 'aqi_prediction_analysis.png'")
        
        # Display plot non-interactively
        plt.show(block=False)  # Non-blocking display
        plt.pause(1)  # Brief pause to ensure rendering
        plt.close('all')  # Close all figures to free memory
    
    def predict_single_sample(self, sample_data):
        """Predict AQI for a single sample"""
        if not self.is_trained:
            print("❌ Model not trained yet!")  # Check if model is trained
            return None
        
        # Prepare sample data
        sample_df = pd.DataFrame([sample_data])  # Convert input dictionary to DataFrame
        
        # Ensure all training features are present
        for col in self.feature_names:
            if col not in sample_df.columns:
                sample_df[col] = 0  # Set missing values to 0
        
        # Reorder columns to match training data
        sample_df = sample_df[self.feature_names]
        
        # Scale the sample
        sample_scaled = self.scaler.transform(sample_df)  # Apply same scaling as training
        
        # Make prediction
        prediction = self.model.predict(sample_scaled)[0]  # Predict AQI for the sample
        
        return prediction  # Return predicted AQI
    
    def interactive_prediction(self):
        """Interactive interface for AQI prediction"""
        print("\n🎯 Interactive AQI Prediction")
        print("=" * 40)
        
        if not self.is_trained:
            print("❌ Model not trained yet!")  # Check if model is trained
            return
        
        print("Enter pollutant values to predict AQI:")
        
        # Get available cities from encoded columns
        city_columns = [col for col in self.feature_names if col.startswith('City_')]
        cities = [col.replace('City_', '') for col in city_columns]
        
        sample_data = {}  # Initialize dictionary for user input
        
        # Prompt for pollutant values
        pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'NH3']
        
        for pollutant in pollutants:
            if pollutant in self.feature_names:
                try:
                    value = float(input(f"Enter {pollutant} value (μg/m³): "))  # Get user input
                    sample_data[pollutant] = value
                except ValueError:
                    print(f"Invalid input for {pollutant}, using 0")  # Handle wrong input
                    sample_data[pollutant] = 0
        
        # Prompt for city
        if cities:
            print(f"\nAvailable cities: {cities}")  # Show cities
            city = input("Enter city name (or press Enter for default): ").strip()
            
            # Initialize city columns to 0
            for city_col in city_columns:
                sample_data[city_col] = 0
            
            # Set selected city to 1 if valid
            if f"City_{city}" in city_columns:
                sample_data[f"City_{city}"] = 1
        
        # Predict AQI
        predicted_aqi = self.predict_single_sample(sample_data)
        
        if predicted_aqi is not None:
            print(f"\n🎯 Predicted AQI: {predicted_aqi:.2f}")  # Display predicted AQI
            
            # Categorize AQI based on standard ranges
            if predicted_aqi <= 50:
                category = "Good 😊"
                color = "Green"
            elif predicted_aqi <= 100:
                category = "Moderate 😐"
                color = "Yellow"
            elif predicted_aqi <= 150:
                category = "Unhealthy for Sensitive Groups 😷"
                color = "Orange"
            elif predicted_aqi <= 200:
                category = "Unhealthy 😨"
                color = "Red"
            elif predicted_aqi <= 300:
                category = "Very Unhealthy 🤢"
                color = "Purple"
            else:
                category = "Hazardous 💀"
                color = "Maroon"
            
            print(f"🏷️ AQI Category: {category}")  # Display AQI category
            print(f"🎨 Color Code: {color}")  # Display color code
    
    def generate_report(self):
        """Generate a comprehensive model performance report"""
        print("\n📖 COMPREHENSIVE MODEL REPORT")
        print("=" * 60)
        
        if not self.is_trained:
            print("❌ Model not trained yet!")  # Check if model is trained
            return
        
        # Display model summary
        print("🤖 MODEL SUMMARY:")
        print(f"  Algorithm: Linear Regression")
        print(f"  Features: {len(self.feature_names)}")  # Number of features
        print(f"  Training samples: {len(self.y_train)}")  # Training set size
        print(f"  Testing samples: {len(self.y_test)}")  # Test set size
        
        # Get and display performance metrics
        metrics = self.evaluate_model()
        
        # Display model equation
        print(f"\n📐 MODEL EQUATION:")
        print(f"  AQI = {self.model.intercept_:.4f} + Σ(coefficient × feature)")
        
        # Analyze and display coefficients
        coef_df = self.analyze_coefficients()
        
        # Display prediction accuracy
        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"  The model explains {metrics['test_r2']*100:.1f}% of AQI variance")
        print(f"  Average prediction error: ±{metrics['test_mae']:.1f} AQI units")
        
        # Provide recommendations based on model
        print(f"\n💡 RECOMMENDATIONS:")
        if metrics['test_r2'] > 0.7:
            print("  ✅ Model shows good predictive performance")
            print("  ✅ Suitable for AQI prediction tasks")
        else:
            print("  ⚠️ Consider collecting more data")
            print("  ⚠️ May need feature engineering")
        
        # Display insights on influential pollutants
        print(f"\n📊 DATA INSIGHTS:")
        top_pollutants = coef_df[~coef_df['Feature'].str.contains('City')].head(3)  # Top 3 non-city features
        print("  Most influential pollutants:")
        for _, row in top_pollutants.iterrows():
            print(f"    - {row['Feature']}: {row['Coefficient']:.4f}")

# Main function to run the AQI Predictor
def main():
    """Main function to run the AQI Predictor"""
    # Initialize the predictor
    predictor = AQIPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_and_preprocess_data('city_day.csv')
        
        # Prepare features
        X, y, pollutant_features = predictor.prepare_features()
        
        # Perform feature analysis
        top_features, feature_importance = predictor.feature_analysis(X, y, pollutant_features)
        
        # Train the model
        predictor.train_model(X, y)
        
        # Evaluate model performance
        predictor.evaluate_model()
        
        # Generate visualizations
        predictor.create_visualizations()
        
        # Generate report
        predictor.generate_report()
        
        # Interactive menu loop
        while True:
            print("\n" + "="*50)
            print("OPTIONS:")
            print("1. Make a prediction")
            print("2. View model report")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                predictor.interactive_prediction()  # User interactive
            elif choice == '2':
                predictor.generate_report()  # Display model 
            elif choice == '3':
                print("👋 Thank you for using AQI Predictor!")  # Exit 
                break
            else:
                print("❌ Invalid choice. Please try again.")  # Handle invalid input
                
    except FileNotFoundError:
        print("❌ Error: 'city_day.csv' file not found!")  # Handle missing CSV 
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Handle exceptions

if __name__ == "__main__":
    main()