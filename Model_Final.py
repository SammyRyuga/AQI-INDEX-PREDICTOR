#!/usr/bin/env python3
# Shebang

# AQI Index Predictor using Linear Regression

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced plotting
from sklearn.model_selection import train_test_split, cross_val_score  # For data splitting and cross-validation
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For model evaluation
from sklearn.feature_selection import SelectKBest, f_regression  # For feature selection (unused in this version)
import math
import scipy.stats as stats  # Used in Q-Q plot
import warnings
import streamlit as st  # For Streamlit interface
warnings.filterwarnings('ignore')  # No warnings, for clear console

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
        self.df = None  # Store DataFrame globally
    
    def load_and_preprocess_data(self):
        if self.df is not None and st.session_state.get('page') == "Home":
            st.write("✅ Dataset loaded successfully! 🌟")
            st.write(f"📈 Dataset shape: {self.df.shape}")  # Print number of rows and columns
            st.write(f"🏙️ Cities in dataset: {self.df['City'].nunique()}")  # Print number of unique cities
            
            # Display basic dataset info
            st.write("\n📋 Dataset Info:")
            st.write(f"Columns: {list(self.df.columns)}")  # List all column names
            st.write(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")  # Show date range
            
            # Missing values
            st.write(f"\n🔍 Data Quality Check:")
            missing_counts = self.df.isnull().sum()  # Count missing values per column
            for col, count in missing_counts.items():
                if count > 0:
                    st.write(f"  {col}: {count} ({count/len(self.df)*100:.1f}%)")  # Print columns with missing values and percentages
        elif st.session_state.get('page') == "Home":
            st.write("⚠️ Please upload a file to continue. 📤")
        return self.df  # Return the loaded DataFrame
    
    def prepare_features(self, target_col='AQI'):
        """Prepare features for modeling by cleaning data"""
        if self.df is None:
            st.error("❌ No dataset loaded. Please upload a file on the Home page. 🚫")
            return None, None, None
        
        st.write("\n🔧 Preparing features for modeling...")
        
        # Define pollutant features
        pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        
        # Filter for features available in the dataset
        available_features = [col for col in pollutant_features if col in self.df.columns]
        st.write(f"📊 Available pollutant features: {available_features}")
        
        # Create feature matrix (X) and target vector (y)
        X = self.df[available_features + ['City']].copy()  # Include pollutants and City column
        y = self.df[target_col].copy()  # Copy AQI column as target
        
        # Remove rows where target (AQI) is missing
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        st.write(f"✅ Valid samples after removing missing AQI: {len(X)} 🌱")
        
        # Encoding cities
        st.write("🏙️ Encoding cities...")
        X_encoded = pd.get_dummies(X, columns=['City'], prefix='City', drop_first=True)  # Encode cities, drop first to avoid multicollinearity
        
        # Impute missing values in pollutant features
        st.write("🔄 Imputing missing values...")
        pollutant_cols = [col for col in available_features if col in X_encoded.columns]
        X_encoded[pollutant_cols] = self.imputer.fit_transform(X_encoded[pollutant_cols])  # Impute missing values with median
        
        st.write(f"✅ Final feature matrix shape: {X_encoded.shape} ✨")
        
        return X_encoded, y, available_features  # Return encoded features, target, and available pollutant features
    
    def feature_analysis(self, X, y, pollutant_features):
        """Analyze feature importance through correlation analysis"""
        if X is None or y is None:
            st.error("❌ No data available for analysis. Please upload a file. 🚫")
            return None, None
        
        st.write("\n📊 Feature Analysis...")
        
        # Calculate correlation of each pollutant feature with AQI
        feature_importance = {}
        for feature in pollutant_features:
            if feature in X.columns:
                correlation = np.corrcoef(X[feature], y)[0, 1]  # Compute correlation
                feature_importance[feature] = abs(correlation)  # Store absolute correlation
        
        # Sort features by correlation (descending order)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Display feature importance
        st.write("✔️ Feature Importance (Correlation with AQI):")
        for feature, importance in sorted_features:
            st.write(f"  {feature}: {importance:.4f} 🌿")
        
        # Choose top 8 features
        top_features = [feature for feature, _ in sorted_features[:8]]
        
        return top_features, feature_importance  # Return top features 
    
    def train_model(self, X, y, test_size=0.2):
        """Train the LR model"""
        if X is None or y is None:
            st.error("❌ No data available for training. Please upload a file. 🚫")
            return None, None, None, None, None, None
        
        st.write("\n🤖 Training Linear Regression Model... 🚀")
        
        # Split data into training set and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )  # 80-20 split
        
        st.write(f"📊 Training set size: {len(X_train)} 📚")
        st.write(f"📊 Testing set size: {len(X_test)} 📈")
        
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
        if not self.is_trained or self.y_test is None or self.y_test_pred is None:
            st.error("❌ Model not trained or data unavailable. Please train the model first. 🚫")
            return None
        
        st.write("\n📈 Model Performance Evaluation: 🌟")
        st.write("=" * 40)
        
        # Calculate training metrics
        train_r2 = r2_score(self.y_train, self.y_train_pred)  # R2 score for training data
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)  # Mean Absolute Error
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))  # Root Mean Squared Error
        
        # Calculate testing metrics
        test_r2 = r2_score(self.y_test, self.y_test_pred)  # R2 score for test data
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)  # Mean Absolute Error
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))  # Root Mean Squared Error
        
        # Show training 
        st.write("🏋️ Training Performance:")
        st.write(f"  R² Score: {train_r2:.4f} 🎯")
        st.write(f"  MAE: {train_mae:.4f} 📏")
        st.write(f"  RMSE: {train_rmse:.4f} 📉")
        
        # Show testing 
        st.write("\n🎯 Testing Performance:")
        st.write(f"  R² Score: {test_r2:.4f} 🎯")
        st.write(f"  MAE: {test_mae:.4f} 📏")
        st.write(f"  RMSE: {test_rmse:.4f} 📉")
        
        # Perform 5-fold cross-validation
        X_scaled = self.scaler.transform(self.X_train)  # Scale training data
        cv_scores = cross_val_score(self.model, X_scaled, self.y_train, cv=5, scoring='r2')  # Compute R² for 5 folds
        st.write(f"\n🔄 5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) ✅")
        
        # Interpret model performance
        if test_r2 > 0.8:
            st.write("✅ Excellent model performance! 🌟")
        elif test_r2 > 0.6:
            st.write("✅ Good model performance! 🌱")
        elif test_r2 > 0.4:
            st.write("⚠️ Moderate model performance 😐")
        else:
            st.write("❌ Poor model performance - consider feature engineering 💡")
        
        # Return metrics as a dict
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
    
    def analyze_coefficients(self):
        """Analyze and display model coefficients"""
        if not self.is_trained:
            st.error("❌ Model not trained yet. Please train the model first. 🚫")
            return None
        
        st.write("\n🔍 Linear Regression Coefficients Analysis: ✨")
        st.write("=" * 50)
        
        # Model coefficients and intercept
        coefficients = self.model.coef_  # Coefficients for each feature
        intercept = self.model.intercept_  # Model intercept (bias term)
        
        # Create DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)  # Sort by absolute coefficient value
        
        # Display intercept
        st.write(f"📊 Model Intercept: {intercept:.4f} 📏")
        
        # Display top 10 features in a styled table
        st.write("\n🎯 Top 10 Most Important Features:")
        st.dataframe(coef_df.head(10).style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]}]
        ))
        
        # Explain coefficient interpretation
        st.write("\n💡 Coefficient Interpretation:")
        st.write("  Positive coefficients: Increase in feature leads to higher AQI 🌡️")
        st.write("  Negative coefficients: Increase in feature leads to lower AQI ❄️")
        st.write("  Larger absolute values: More influence on AQI prediction 📈")
        
        return coef_df  # Return coefficient DataFrame
    
    def create_visualizations(self):
        """Create graphs for model analysis"""
        if not self.is_trained:
            st.error("❌ Model not trained yet. Please train the model first. 🚫")
            return
        
        st.write("\n📊 Generating Visualizations... 🎨")
        
        # Set matplotlib backend for non-interactive plotting
        plt.switch_backend('Agg')  # Use Agg backend for compatibility
        
        # Select plotting style
        available_styles = plt.style.available  # Get available matplotlib styles
        style = 'ggplot'  # Default style
        if 'seaborn-v0_8' in available_styles:
            style = 'seaborn-v0_8'  # Preferred style
        plt.style.use(style)  # Applying selected style
        sns.set_palette("husl")  # Setting color palette
        
        # Creating figure with multiple subplots
        plt.figure(figsize=(20, 15))  # 6 subplots
        
        # Graph 1: Actual vs Predicted AQI (Training)
        plt.subplot(2, 3, 1)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue', label='Training Data')  # Scatter plot
        plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)  # Diagonal line
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Training: Actual vs Predicted AQI 🌱')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 2: Actual vs Predicted AQI (Testing)
        plt.subplot(2, 3, 2)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green', label='Testing Data')  # Scatter plot
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)  # Diagonal line
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Testing: Actual vs Predicted AQI 🌿')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 3: Residuals vs Predicted AQI
        plt.subplot(2, 3, 3)
        residuals = self.y_test - self.y_test_pred  # Calculate residuals
        plt.scatter(self.y_test_pred, residuals, alpha=0.6)  # Scatter plot of residuals
        plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line at y=0
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted AQI 📉')
        plt.grid(True, alpha=0.3)
        
        # Graph 4: Residual Distribution
        plt.subplot(2, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')  # Histogram of residuals
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')  # Mean line
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals 📊')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 5: Q-Q Plot for Residuals
        plt.subplot(2, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)  # Q-Q plot to check normality
        plt.title('Q-Q Plot: Residuals vs Normal Distribution 📈')
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
            plt.title('Top Pollutant Feature Coefficients 🌡️')
            plt.grid(True, alpha=0.3)
        
        # Save and display plot
        plt.tight_layout()  # Subplot spacing
        st.pyplot(plt)  # Display plot in Streamlit
        plt.close()  # Close figure to free memory
    
    def predict_single_sample(self, sample_data):
        """Predict AQI for a single sample"""
        if not self.is_trained:
            st.error("❌ Model not trained yet! 🚫")  # Check if model is trained
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
        if not self.is_trained:
            st.error("❌ Model not trained yet! 🚫")  # Check if model is trained
            return
        
        st.write("\n🎯 Interactive AQI Prediction 🌍")
        st.write("=" * 40)
        
        st.write("Enter pollutant values to predict AQI: 🌱")
        
        # Get available cities from encoded columns
        city_columns = [col for col in self.feature_names if col.startswith('City_')]
        cities = [col.replace('City_', '') for col in city_columns]
        
        sample_data = {}  # Initialize dictionary for user input
        
        # Prompt for pollutant values
        pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2', 'NH3']
        with st.form(key='prediction_form'):
            for pollutant in pollutants:
                if pollutant in self.feature_names:
                    sample_data[pollutant] = st.number_input(f"Enter {pollutant} value (μg/m³): ", value=0.0)
            
            # Prompt for city
            if cities:
                st.write(f"\nAvailable cities: {cities} 🏙️")  # Show cities
                city = st.text_input("Enter city name (or leave blank for default): ").strip()
                
                # Initialize city columns to 0
                for city_col in city_columns:
                    sample_data[city_col] = 0
                
                # Set selected city to 1 if valid
                if f"City_{city}" in city_columns:
                    sample_data[f"City_{city}"] = 1
            
            submit_button = st.form_submit_button(label="Predict AQI 🚀")
        
        if submit_button:
            predicted_aqi = self.predict_single_sample(sample_data)
            if predicted_aqi is not None:
                st.write(f"\n🎯 Predicted AQI: {predicted_aqi:.2f} 🌡️")  # Display predicted AQI
                
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
                
                st.write(f"🏷️ AQI Category: {category} 🎨")  # Display AQI category
                st.write(f"🎨 Color Code: {color} 🌈")  # Display color code
    
    def generate_report(self):
        """Generate a comprehensive model performance report"""
        if not self.is_trained:
            st.error("❌ Model not trained yet! 🚫")  # Check if model is trained
            return
        
        st.write("\n📖 COMPREHENSIVE MODEL REPORT 🌟")
        st.write("=" * 60)
        
        # Display model summary
        st.write("🤖 MODEL SUMMARY:")
        st.write(f"  Algorithm: Linear Regression 📊")
        st.write(f"  Features: {len(self.feature_names)} 🌿")  # Number of features
        st.write(f"  Training samples: {len(self.y_train)} 📚")  # Training set size
        st.write(f"  Testing samples: {len(self.y_test)} 📈")  # Test set size
        
        # Get and display performance metrics
        metrics = self.evaluate_model()
        
        # Display model equation
        st.write(f"\n📐 MODEL EQUATION:")
        st.write(f"  AQI = {self.model.intercept_:.4f} + Σ(coefficient × feature) 📉")
        
        # Analyze and display coefficients
        coef_df = self.analyze_coefficients()
        
        # Display prediction accuracy
        st.write(f"\n🎯 PREDICTION ACCURACY:")
        st.write(f"  The model explains {metrics['test_r2']*100:.1f}% of AQI variance 🌡️")
        st.write(f"  Average prediction error: ±{metrics['test_mae']:.1f} AQI units 📏")
        
        # Provide recommendations based on model
        st.write(f"\n💡 RECOMMENDATIONS:")
        if metrics['test_r2'] > 0.7:
            st.write("  ✅ Model shows good predictive performance 🌱")
            st.write("  ✅ Suitable for AQI prediction tasks 🚀")
        else:
            st.write("  ⚠️ Consider collecting more data 📤")
            st.write("  ⚠️ May need feature engineering 💡")
        
        # Display insights on influential pollutants
        st.write(f"\n📊 DATA INSIGHTS:")
        top_pollutants = coef_df[~coef_df['Feature'].str.contains('City')].head(3)  # Top 3 non-city features
        st.write("  Most influential pollutants:")
        for _, row in top_pollutants.iterrows():
            st.write(f"    - {row['Feature']}: {row['Coefficient']:.4f} 🌡️")

# Main Streamlit app
def main():
    """Main function to run the AQI Predictor"""
    predictor = AQIPredictor()
    
    # Navigation sidebar
    st.sidebar.title("🌍 Navigation Menu")
    page = st.sidebar.radio("Go to", ["Home", "Graphs", "Make a Prediction", "View Model Report"], index=0)
    st.session_state['page'] = page  # Store current page in session state
    
    # Credits & Help sidebar
    with st.sidebar:
        st.title("📜 Credits & Help")
        with st.expander("Credits"):
            st.write("🎉 Developed with ❤️ by Akshat Goel, Anwita Padhi, Samanyu Pattanayak, Samruddh Om Bahanwal, Shruti Deepak:")
            st.write("### About Us")
            st.write("We are a passionate team dedicated to improving air quality awareness using data science. Our mission is to provide actionable insights through innovative tools like the AQI Predictor. 🌱")
            st.write("- Streamlit for the interactive UI 🌐")
            st.write("- Pandas and NumPy for data processing 📊")
            st.write("- Scikit-learn for machine learning 🤖")
            st.write("- Matplotlib and Seaborn for visualizations 🎨")
            st.write("Special thanks to ISTE for assistance! 🚀")
        with st.expander("Help"):
            st.write("📚 How to Use:")
            st.write("- Upload a CSV file (e.g., city_day.csv) on the Home page 📤")
            st.write("- Navigate using the left menu 🌍")
            st.write("- Use 'Make a Prediction' to input pollutant values and predict AQI 🌡️")
            st.write("- View graphs or reports for detailed analysis 📊")
            st.write("For issues, contact support at sammyryuga@gmail.com ✉️")
        with st.expander("Team"):
            st.write("👥 Meet the Team:")
            st.write("- **Akshat Goel**")
            st.write("  - LinkedIn: [linkedin.com/in/akshat-goel](https://www.linkedin.com/in/akshat-goel-b13054323/)")
            st.write("  - Instagram: [@akshatgoel_1105006](https://www.instagram.com/akshatgoel_1105006/)")
            st.write("- **Anwita Padhi**")
            st.write("  - LinkedIn: [linkedin.com/in/anwita-padhi](https://www.linkedin.com/in/anwita-padhi-187576321/)")
            st.write("  - Instagram: [@anwitapadhi](https://www.instagram.com/anwitapadhi/)")
            st.write("- **Samanyu Pattanayak**")
            st.write("  - LinkedIn: [linkedin.com/in/samanyu-pattanayak](https://www.linkedin.com/in/samanyu-pattanayak-8757551a9/)")
            st.write("  - Instagram: [@sammyryuga](https://www.instagram.com/sammyryuga/)")
            st.write("- **Samruddh Om Bahanwal**")
            st.write("  - LinkedIn: [linkedin.com/in/samruddh-om-bahanwal]( https://www.linkedin.com/in/samruddh-om-bahanwal-48a93a228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)")
            st.write("  - Instagram: [@samruddh_om](https://www.instagram.com/samruddh_om/)")
            st.write("- **Shruti Deepak**")
            st.write("  - LinkedIn: [linkedin.com/in/shruti](https://www.linkedin.com/in/shruti-deepak-956820362/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
            st.write("  - Instagram: [@silent_silverkitty](https://instagram.com/silent_silverkitty)")
    
    # File uploader at the app level with a unique key
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="unique_file_uploader_2025")
    if uploaded_file is not None and predictor.df is None:
        try:
            predictor.df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)} 🚫")
    
    # Only proceed if data is loaded
    if predictor.df is None:
        st.write("⚠️ Please upload a file on the Home page to proceed. 📤")
        predictor.load_and_preprocess_data()  # Show prompt on Home
        return
    
    # Prepare features
    X, y, pollutant_features = predictor.prepare_features()
    if X is None or y is None or pollutant_features is None:
        return
    
    # Perform feature analysis
    top_features, feature_importance = predictor.feature_analysis(X, y, pollutant_features)
    if top_features is None or feature_importance is None:
        return
    
    # Train the model
    X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = predictor.train_model(X, y)
    if X_train is None or X_test is None or y_train is None or y_test is None or y_train_pred is None or y_test_pred is None:
        return
    
    # Evaluate model performance
    predictor.evaluate_model()
    
    # Page content based on sidebar selection
    st.empty()  # Clear previous content
    if page == "Home":
        predictor.load_and_preprocess_data()  # Show upload and info only on Home
    elif page == "Graphs":
        predictor.create_visualizations()
    elif page == "Make a Prediction":
        predictor.interactive_prediction()
    elif page == "View Model Report":
        predictor.generate_report()

if __name__ == "__main__":
    main()