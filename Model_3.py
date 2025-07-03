#!/usr/bin/env python3
"""
AQI Index Predictor using Linear Regression
"""

#importing libs
import pandas as pd                                       #data   
import numpy as np                                        #data
import matplotlib.pyplot as plt                           #graph
import seaborn as sns                                     #graph
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import math
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')              #ignore warning, keep console clean

class AQIPredictor:
    def __init__(self):
        """Initialize the AQI Predictor with necessary components"""
        self.model = LinearRegression()  #model
        self.scaler = StandardScaler()  #scaler
        self.imputer = SimpleImputer(strategy='median')  #missingvalue imputer
        self.feature_selector = None  #feature selector (not used currently)
        self.selected_features = None  #selected features storage
        self.city_encoder = None  #city encoder (not used currently)
        self.is_trained = False  #model training status
    
    def load_and_preprocess_data(self, csv_path='city_day.csv'):
        """Load and preprocess the AQI dataset from CSV file"""
        # Display header
        print("=" * 60)
        print("🌍 AQI LINEAR REGRESSION PREDICTOR")
        print("=" * 60)
        
        # Load dataset
        print("📊 Loading dataset...")
        self.df = pd.read_csv(csv_path)
        
        # Display loading success message
        print(f"✅ Dataset loaded successfully!")
        # Display dataset information
        print(f"📈 Dataset shape: {self.df.shape}")
        print(f"🏙️ Cities in dataset: {self.df['City'].nunique()}")
        
        # Display basic info
        print("\n📋 Dataset Info:")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Check data quality
        print(f"\n🔍 Data Quality Check:")
        missing_counts = self.df.isnull().sum()
        print("Missing values per column:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def prepare_features(self, target_col='AQI'):
        """Prepare features for modeling by cleaning and encoding data"""
        print("\n🔧 Preparing features for modeling...")
        
        # Define pollutant features
        pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        
        # Check available features exist in dataset
        available_features = [col for col in pollutant_features if col in self.df.columns]
        print(f"📊 Available pollutant features: {available_features}")
        
        # Create feature matrix
        X = self.df[available_features + ['City']].copy()
        y = self.df[target_col].copy()
        
        # Remove rows with missing target values
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"✅ Valid samples after removing missing AQI: {len(X)}")
        
        # One-hot encode city names
        print("🏙️ Encoding cities...")
        X_encoded = pd.get_dummies(X, columns=['City'], prefix='City', drop_first=True)
        
        # Impute missing values for pollutant features
        print("🔄 Imputing missing values...")
        pollutant_cols = [col for col in available_features if col in X_encoded.columns]
        X_encoded[pollutant_cols] = self.imputer.fit_transform(X_encoded[pollutant_cols])
        
        print(f"✅ Final feature matrix shape: {X_encoded.shape}")
        
        return X_encoded, y, available_features
    
    def feature_analysis(self, X, y, pollutant_features):
        """Analyze feature importance through correlation analysis"""
        print("\n📊 Feature Analysis...")
        
        # Calculate feature importance based on correlation with target
        feature_importance = {}
        for feature in pollutant_features:
            if feature in X.columns:
                correlation = np.corrcoef(X[feature], y)[0, 1]
                feature_importance[feature] = abs(correlation)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("✔️ Feature Importance (Correlation with AQI):")
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
        
        # Select top features
        top_features = [feature for feature, _ in sorted_features[:8]]
        
        return top_features, feature_importance
    
    def train_model(self, X, y, test_size=0.2):
        """Train the linear regression model with provided data"""
        print("\n🤖 Training Linear Regression Model...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"📊 Training set size: {len(X_train)}")
        print(f"📊 Testing set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Store data for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
        self.feature_names = [col for col in X.columns]
        
        self.is_trained = True
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
    
    def evaluate_model(self):
        """Evaluate model performance using multiple metrics"""
        print("\n📈 Model Performance Evaluation:")
        print("=" * 40)
        
        # Calculate training metrics
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        
        # Calculate testing metrics
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        
        # Display training performance
        print("🏋️ Training Performance:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        
        # Display testing performance
        print("\n🎯 Testing Performance:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # Perform cross-validation
        X_scaled = self.scaler.transform(self.X_train)
        cv_scores = cross_val_score(self.model, X_scaled, self.y_train, cv=5, scoring='r2')
        print(f"\n🔄 5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Provide performance interpretation
        if test_r2 > 0.8:
            print("✅ Excellent model performance!")
        elif test_r2 > 0.6:
            print("✅ Good model performance!")
        elif test_r2 > 0.4:
            print("⚠️ Moderate model performance")
        else:
            print("❌ Poor model performance - consider feature engineering")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
    
    def analyze_coefficients(self):
        """Analyze and display model coefficients for feature importance"""
        print("\n🔍 Linear Regression Coefficients Analysis:")
        print("=" * 50)
        
        # Get model coefficients and intercept
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Create coefficient dataframe
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        # Display results
        print(f"📊 Model Intercept: {intercept:.4f}")
        print("\n🎯 Top 10 Most Important Features:")
        print(coef_df.head(10).to_string(index=False))
        
        # Provide interpretation
        print("\n💡 Coefficient Interpretation:")
        print("  Positive coefficients: Increase in feature leads to higher AQI")
        print("  Negative coefficients: Increase in feature leads to lower AQI")
        print("  Larger absolute values: More influence on AQI prediction")
        
        return coef_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations for model analysis"""
        print("\n📊 Generating Visualizations...")
        
        # Ensure matplotlib uses a suitable backend
        plt.switch_backend('Agg')  # Use Agg backend for non-interactive plotting
        
        # Check available styles and select a valid one
        available_styles = plt.style.available
        style = 'ggplot'  # Default style
        if 'seaborn-v0_8' in available_styles:
            style = 'seaborn-v0_8'  # Preferred style if available
        plt.style.use(style)  # Set the selected style
        sns.set_palette("husl")  # Use husl color palette
        
        # Create new figure
        plt.figure(figsize=(20, 15))
        
        # 1. Actual vs Predicted (Training)
        plt.subplot(2, 3, 1)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue', label='Training Data')
        plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Training: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Testing)
        plt.subplot(2, 3, 2)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green', label='Testing Data')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Testing: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        plt.subplot(2, 3, 3)
        residuals = self.y_test - self.y_test_pred
        plt.scatter(self.y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted AQI')
        plt.grid(True, alpha=0.3)
        
        # 4. Residual distribution
        plt.subplot(2, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Q-Q plot
        plt.subplot(2, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot: Residuals vs Normal Distribution')
        plt.grid(True, alpha=0.3)
        
        # 6. Feature importance
        plt.subplot(2, 3, 6)
        coef_df = self.analyze_coefficients()
        top_features = coef_df.head(10)
        
        # Filter for pollutant features
        pollutant_coefs = top_features[~top_features['Feature'].str.contains('City')]
        if len(pollutant_coefs) > 0:
            plt.barh(pollutant_coefs['Feature'], pollutant_coefs['Coefficient'])
            plt.xlabel('Coefficient Value')
            plt.title('Top Pollutant Feature Coefficients')
            plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('aqi_prediction_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'aqi_prediction_analysis.png'")
        
        # Display plot
        plt.show(block=False)  # Non-blocking show
        plt.pause(1)  # Brief pause to ensure rendering
        plt.close('all')  # Close all figures to free memory
    
    def predict_single_sample(self, sample_data):
        """Predict AQI for a single sample"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        # Prepare sample data
        sample_df = pd.DataFrame([sample_data])
        
        # Ensure all required columns exist
        for col in self.feature_names:
            if col not in sample_df.columns:
                sample_df[col] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        sample_df = sample_df[self.feature_names]
        
        # Scale the sample
        sample_scaled = self.scaler.transform(sample_df)
        
        # Make prediction
        prediction = self.model.predict(sample_scaled)[0]
        
        return prediction
    
    def interactive_prediction(self):
        """Interactive interface for AQI prediction"""
        print("\n🎯 Interactive AQI Prediction")
        print("=" * 40)
        
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        print("Enter pollutant values to predict AQI:")
        
        # Get available cities
        city_columns = [col for col in self.feature_names if col.startswith('City_')]
        cities = [col.replace('City_', '') for col in city_columns]
        
        sample_data = {}
        
        # Get pollutant values
        pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'NH3']
        
        for pollutant in pollutants:
            if pollutant in self.feature_names:
                try:
                    value = float(input(f"Enter {pollutant} value (μg/m³): "))
                    sample_data[pollutant] = value
                except ValueError:
                    print(f"Invalid input for {pollutant}, using 0")
                    sample_data[pollutant] = 0
        
        # Get city
        if cities:
            print(f"\nAvailable cities: {cities[:10]}...")  # Show first 10
            city = input("Enter city name (or press Enter for default): ").strip()
            
            # Set city encoding
            for city_col in city_columns:
                sample_data[city_col] = 0
            
            if f"City_{city}" in city_columns:
                sample_data[f"City_{city}"] = 1
        
        # Make prediction
        predicted_aqi = self.predict_single_sample(sample_data)
        
        if predicted_aqi is not None:
            print(f"\n🎯 Predicted AQI: {predicted_aqi:.2f}")
            
            # Determine AQI category
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
                category = "Hazardous ☠️"
                color = "Maroon"
            
            print(f"🏷️ AQI Category: {category}")
            print(f"🎨 Color Code: {color}")
    
    def generate_report(self):
        """Generate a comprehensive model performance report"""
        print("\n📖 COMPREHENSIVE MODEL REPORT")
        print("=" * 60)
        
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        # Display model summary
        print("🤖 MODEL SUMMARY:")
        print(f"  Algorithm: Linear Regression")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Testing samples: {len(self.y_test)}")
        
        # Get and display performance metrics
        metrics = self.evaluate_model()
        
        # Display model equation
        print(f"\n📐 MODEL EQUATION:")
        print(f"  AQI = {self.model.intercept_:.4f} + Σ(coefficient × feature)")
        
        # Analyze coefficients
        coef_df = self.analyze_coefficients()
        
        # Display prediction accuracy
        print(f"\n🎯 PREDICTION ACCURACY:")
        print(f"  The model explains {metrics['test_r2']*100:.1f}% of AQI variance")
        print(f"  Average prediction error: ±{metrics['test_mae']:.1f} AQI units")
        
        # Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if metrics['test_r2'] > 0.7:
            print("  ✅ Model shows good predictive performance")
            print("  ✅ Suitable for AQI prediction tasks")
        else:
            print("  ⚠️ Consider collecting more data")
            print("  ⚠️ May need feature engineering")
        
        # Display data insights
        print(f"\n📊 DATA INSIGHTS:")
        top_pollutants = coef_df[~coef_df['Feature'].str.contains('City')].head(3)
        print("  Most influential pollutants:")
        for _, row in top_pollutants.iterrows():
            print(f"    - {row['Feature']}: {row['Coefficient']:.4f}")

def main():
    """Main function to run the AQI Predictor"""
    # Initialize predictor
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
        
        # Interactive menu
        while True:
            print("\n" + "="*50)
            print("OPTIONS:")
            print("1. Make a prediction")
            print("2. View model report")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                predictor.interactive_prediction()
            elif choice == '2':
                predictor.generate_report()
            elif choice == '3':
                print("👋 Thank you for using AQI Predictor!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
    except FileNotFoundError:
        print("❌ Error: 'city_day.csv' file not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()