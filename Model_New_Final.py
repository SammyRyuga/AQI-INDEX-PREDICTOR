#!/usr/bin/env python3
# AQI Index Predictor using Linear Regression

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import warnings
import streamlit as st
import time
warnings.filterwarnings('ignore')

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #2980b9;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stNumberInput>input[type="number"]::-webkit-inner-spin-button,
    .stNumberInput>input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    .stNumberInput>input[type="number"] {
        -moz-appearance: textfield;
        transition: background-color 0.3s ease;
    }
    .stNumberInput>input[type="number"]:hover {
        background-color: #f0f0f0;
    }
    .stNumberInput>input[type="number"]:active,
    .stNumberInput>input[type="number"]:focus {
        background-color: white;
        outline: none;
    }
    details > summary {
        transition: all 0.2s ease;
        color: #2980b9;
        font-weight: bold;
    }
    details > summary:hover {
        background-color: #f0f0f0 !important;
        color: #3498db !important;
    }
    details > summary:active,
    details > summary:focus {
        background-color: transparent !important;
        color: #2980b9 !important;
    }
    details[open] > summary {
        background-color: transparent !important;
        color: #2980b9 !important;
    }
    </style>
""", unsafe_allow_html=True)

class AQIPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.selected_features = None
        self.city_encoder = None
        self.is_trained = False
        self.df = None
        self.top_features = None
        self.feature_importance = None
        self.pollutant_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.feature_names = None
    
    def prepare_features(self, target_col='AQI'):
        if self.df is None:
            return None, None, None
        
        pollutant_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        available_features = [col for col in pollutant_features if col in self.df.columns]
        
        if not available_features:
            st.error("No pollutant features found in dataset")
            return None, None, None
        
        X = self.df[available_features + ['City']].copy()
        y = self.df[target_col].copy()
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            st.error("No valid data after removing missing AQI values")
            return None, None, None
        
        X_encoded = pd.get_dummies(X, columns=['City'], prefix='City', drop_first=True)
        pollutant_cols = [col for col in available_features if col in X_encoded.columns]
        X_encoded[pollutant_cols] = self.imputer.fit_transform(X_encoded[pollutant_cols])
        
        return X_encoded, y, available_features
    
    def feature_analysis(self, X, y, pollutant_features):
        if X is None or y is None:
            return None, None
        
        feature_importance = {}
        for feature in pollutant_features:
            if feature in X.columns:
                try:
                    correlation = np.corrcoef(X[feature], y)[0, 1]
                    if not np.isnan(correlation):
                        feature_importance[feature] = abs(correlation)
                except:
                    continue
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:8]]
        return top_features, feature_importance
    
    def train_model(self, X, y, test_size=0.2):
        if X is None or y is None:
            return None, None, None, None, None, None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Preparing data for training...")
            time.sleep(0.3)
            progress_bar.progress(25)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
            
            status_text.text("Scaling features...")
            progress_bar.progress(50)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            status_text.text("Training model...")
            progress_bar.progress(75)
            self.model.fit(X_train_scaled, y_train)
            
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
            self.feature_names = [col for col in X.columns]
            self.is_trained = True
            
            status_text.text("Training complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
        finally:
            progress_bar.empty()
            status_text.empty()
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
    
    def evaluate_model(self):
        if not self.is_trained or self.y_test is None or self.y_test_pred is None:
            return None
        
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        
        X_scaled = self.scaler.transform(self.X_train)
        cv_scores = cross_val_score(self.model, X_scaled, self.y_train, cv=5, scoring='r2')
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
    
    def analyze_coefficients(self):
        if not self.is_trained:
            return None
        
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def create_visualizations(self):
        if not self.is_trained:
            st.error("Model not trained yet. Please train the model first")
            return
        
        st.markdown("<div class='section-header'>Model Visualizations</div>", unsafe_allow_html=True)
        
        # Use context manager for figure
        fig = plt.figure(figsize=(20, 15))
        
        try:
            # Set style
            available_styles = plt.style.available
            style = 'ggplot'
            if 'seaborn-v0_8' in available_styles:
                style = 'seaborn-v0_8'
            plt.style.use(style)
            sns.set_palette("husl")
            
            # Plot 1: Training Actual vs Predicted
            plt.subplot(2, 3, 1)
            plt.scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue', label='Training Data')
            plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
            plt.xlabel('Actual AQI')
            plt.ylabel('Predicted AQI')
            plt.title('Training: Actual vs Predicted AQI')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Testing Actual vs Predicted
            plt.subplot(2, 3, 2)
            plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green', label='Testing Data')
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual AQI')
            plt.ylabel('Predicted AQI')
            plt.title('Testing: Actual vs Predicted AQI')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            plt.subplot(2, 3, 3)
            residuals = self.y_test - self.y_test_pred
            plt.scatter(self.y_test_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted AQI')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted AQI')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Residuals Distribution
            plt.subplot(2, 3, 4)
            plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Distribution of Residuals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Q-Q Plot
            plt.subplot(2, 3, 5)
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot: Residuals vs Normal Distribution')
            plt.grid(True, alpha=0.3)
            
            # Plot 6: Feature Coefficients
            plt.subplot(2, 3, 6)
            coef_df = self.analyze_coefficients()
            top_features = coef_df.head(10)
            pollutant_coefs = top_features[~top_features['Feature'].str.contains('City')]
            if len(pollutant_coefs) > 0:
                plt.barh(pollutant_coefs['Feature'], pollutant_coefs['Coefficient'])
                plt.xlabel('Coefficient Value')
                plt.title('Top Pollutant Feature Coefficients')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        finally:
            plt.close(fig)
    
    def predict_single_sample(self, sample_data):
        if not self.is_trained:
            st.error("Model not trained yet")
            return None
        
        try:
            sample_df = pd.DataFrame([sample_data])
            for col in self.feature_names:
                if col not in sample_df.columns:
                    sample_df[col] = 0
            sample_df = sample_df[self.feature_names]
            sample_scaled = self.scaler.transform(sample_df)
            prediction = self.model.predict(sample_scaled)[0]
            # Ensure predicted AQI is non-negative
            return max(0, prediction)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def interactive_prediction(self):
        if not self.is_trained:
            st.error("Model not trained yet")
            return
        
        st.markdown("<div class='section-header'>Interactive AQI Prediction</div>", unsafe_allow_html=True)
        st.write("Enter city and pollutant values to predict AQI")
        
        city_columns = [col for col in self.feature_names if col.startswith('City_')]
        cities = [col.replace('City_', '') for col in city_columns]
        
        # Get available pollutant features (exclude city dummies)
        available_pollutants = [col for col in self.pollutant_features if col in self.feature_names]
        
        sample_data = {}
        with st.form(key='prediction_form'):
            if cities:
                city = st.selectbox("Select city (or choose 'None' for default):", ["None"] + cities)
                selected_city = None
                if city != "None":
                    selected_city = city
                    st.markdown(f"**Selected city**: {selected_city}")
                else:
                    st.markdown("**No city selected**, using default")
                
                for city_col in city_columns:
                    sample_data[city_col] = 0
                if selected_city:
                    sample_data[f"City_{selected_city}"] = 1
            
            for pollutant in available_pollutants:
                sample_data[pollutant] = st.number_input(
                    f"Enter {pollutant} value (μg/m³): ",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    format="%.1f"
                )
            
            submit_button = st.form_submit_button(label="Predict AQI")
        
        if submit_button:
            if any(value < 0 for value in sample_data.values() if isinstance(value, (int, float))):
                st.error("Negative values are not allowed for pollutant inputs. Please enter non-negative values.")
                return
            
            predicted_aqi = self.predict_single_sample(sample_data)
            if predicted_aqi is not None:
                st.markdown(f"**Predicted AQI**: <span style='color: #e74c3c; font-size: 24px;'>{predicted_aqi:.2f}</span>", unsafe_allow_html=True)
                if predicted_aqi <= 50:
                    category = "Good"
                    color = "Green"
                elif predicted_aqi <= 100:
                    category = "Moderate"
                    color = "Yellow"
                elif predicted_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                    color = "Orange"
                elif predicted_aqi <= 200:
                    category = "Unhealthy"
                    color = "Red"
                elif predicted_aqi <= 300:
                    category = "Very Unhealthy"
                    color = "Purple"
                else:
                    category = "Hazardous"
                    color = "Maroon"
                st.markdown(f"**AQI Category**: <span style='color: {color}; font-weight: bold;'>{category}</span>", unsafe_allow_html=True)
                st.markdown(f"**Color Code**: <span style='color: {color}; font-weight: bold;'>{color}</span>", unsafe_allow_html=True)
    
    def generate_report(self):
        if not self.is_trained:
            st.error("Model not trained yet")
            return
        
        st.markdown("<div class='section-header'>Comprehensive Model Report</div>", unsafe_allow_html=True)
        st.divider()
        
        st.subheader("Dataset Information")
        st.write(f"**Dataset shape**: {self.df.shape}")
        st.write(f"**Cities in dataset**: {self.df['City'].nunique()}")
        st.write(f"**Columns**: {list(self.df.columns)}")
        if 'Date' in self.df.columns:
            st.write(f"**Date range**: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        st.markdown("**Data Quality Check**:")
        missing_counts = self.df.isnull().sum()
        has_missing = False
        for col, count in missing_counts.items():
            if count > 0:
                has_missing = True
                st.write(f"  - {col}: {count} ({count/len(self.df)*100:.1f}%)")
        if not has_missing:
            st.write("  - No missing values detected")
        
        st.subheader("Preparing Features for Modeling")
        if self.pollutant_features:
            st.write(f"**Available pollutant features**: {self.pollutant_features}")
            st.write(f"**Valid samples after removing missing AQI**: {len(self.X_train) + len(self.X_test)}")
            st.write("**Encoding cities using one-hot encoding**")
            st.write("**Imputing missing values using median strategy**")
            st.write(f"**Final feature matrix shape**: {self.X_train.shape}")
        
        st.subheader("Feature Analysis")
        if self.feature_importance is not None:
            st.markdown("**Feature Importance (Correlation with AQI)**:")
            for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  - {feature}: {importance:.4f}")
        
        st.subheader("Training Linear Regression Model")
        if self.X_train is not None:
            st.write(f"**Training set size**: {len(self.X_train)}")
            st.write(f"**Testing set size**: {len(self.X_test)}")
        
        st.subheader("Model Performance Evaluation")
        st.divider()
        metrics = self.evaluate_model()
        if metrics is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
            with col2:
                st.metric("Test MAE", f"{metrics['test_mae']:.2f}")
            with col3:
                st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
            
            st.markdown("**Detailed Metrics**:")
            metrics_df = pd.DataFrame({
                "Metric": ["R² Score", "MAE", "RMSE"],
                "Training": [f"{metrics['train_r2']:.4f}", f"{metrics['train_mae']:.4f}", f"{metrics['train_rmse']:.4f}"],
                "Testing": [f"{metrics['test_r2']:.4f}", f"{metrics['test_mae']:.4f}", f"{metrics['test_rmse']:.4f}"]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            st.write(f"**5-Fold Cross-Validation R² Score**: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
            
            if metrics['test_r2'] > 0.8:
                st.success("✅ Excellent model performance")
            elif metrics['test_r2'] > 0.6:
                st.info("ℹ️ Good model performance")
            elif metrics['test_r2'] > 0.4:
                st.warning("⚠️ Moderate model performance")
            else:
                st.error("❌ Poor model performance - consider feature engineering")
        
        st.subheader("Model Summary")
        st.write(f"**Algorithm**: Linear Regression")
        st.write(f"**Features**: {len(self.feature_names)}")
        st.write(f"**Training samples**: {len(self.y_train)}")
        st.write(f"**Testing samples**: {len(self.y_test)}")
        
        st.subheader("Model Equation")
        st.latex(r"AQI = \beta_0 + \sum_{i=1}^{n} \beta_i \cdot x_i")
        st.write(f"**Intercept (β₀)**: {self.model.intercept_:.4f}")
        
        st.subheader("Linear Regression Coefficients Analysis")
        st.divider()
        coef_df = self.analyze_coefficients()
        if coef_df is not None:
            st.markdown("**Top 10 Most Important Features**:")
            st.dataframe(coef_df.head(10), use_container_width=True)
            
            st.markdown("**Coefficient Interpretation**:")
            st.write("- Positive coefficients: Increase in feature leads to higher AQI")
            st.write("- Negative coefficients: Increase in feature leads to lower AQI")
            st.write("- Larger absolute values: More influence on AQI prediction")
        
        st.subheader("Prediction Accuracy")
        if metrics:
            st.write(f"**The model explains** {metrics['test_r2']*100:.1f}% **of AQI variance**")
            st.write(f"**Average prediction error**: ±{metrics['test_mae']:.1f} AQI units")
        
        st.subheader("Recommendations")
        if metrics and metrics['test_r2'] > 0.7:
            st.write("✅ Model shows good predictive performance")
            st.write("✅ Suitable for AQI prediction tasks")
        else:
            st.write("⚠️ Consider collecting more data")
            st.write("⚠️ May need feature engineering or non-linear models")
        
        st.subheader("Data Insights")
        coef_df = self.analyze_coefficients()
        if coef_df is not None:
            top_pollutants = coef_df[~coef_df['Feature'].str.contains('City')].head(3)
            st.markdown("**Most influential pollutants**:")
            for _, row in top_pollutants.iterrows():
                st.write(f"  - {row['Feature']}: {row['Coefficient']:.4f}")

def load_predictor(df: pd.DataFrame) -> AQIPredictor:
    """Load and train predictor with error handling"""
    try:
        predictor = AQIPredictor()
        predictor.df = df
        X, y, pollutant_features = predictor.prepare_features()
        
        if X is None or y is None:
            raise ValueError("Failed to prepare features from the provided DataFrame.")
        
        predictor.pollutant_features = pollutant_features
        top_features, feature_importance = predictor.feature_analysis(X, y, pollutant_features)
        predictor.top_features = top_features
        predictor.feature_importance = feature_importance
        predictor.train_model(X, y)
        
        return predictor
    except Exception as e:
        st.error(f"Error initializing predictor: {str(e)}")
        raise

def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    
    st.markdown("<div class='main-title'>🌍 AQI Predictor</div>", unsafe_allow_html=True)
    st.markdown("Welcome to the Air Quality Index (AQI) Predictor! Upload a dataset, predict AQI based on pollutant levels, and explore detailed model insights.", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("🌍 **AQI Predictor**")
    st.sidebar.title("Navigation Menu")
    options = ["Home", "Graphs", "View Model Report"]
    page = st.sidebar.radio("Go to", options, index=options.index(st.session_state.current_page))
    st.session_state.current_page = page
    
    with st.sidebar:
        st.title("Credits & Help")
        with st.expander("Credits"):
            st.markdown("""
                **Developed by**:
                - Akshat Goel
                - Anwita Padhi
                - Samanyu Pattanayak
                - Samruddh Om Bahanwal
                - Shruti Deepak
                
                **About Us**  
                We are a passionate team dedicated to improving air quality awareness using data science.
                
                **Technologies Used**:
                - Streamlit for the interactive UI
                - Pandas and NumPy for data processing
                - Scikit-learn for machine learning
                - Matplotlib and Seaborn for visualizations
                - Special thanks to ISTE for assistance
            """)
        
        with st.expander("Help"):
            st.markdown("""
                **How to Use:**
                1. Upload a CSV file containing air quality data
                2. Navigate using the left menu
                3. Use Home page to input pollutant values and predict AQI
                4. View graphs or reports for detailed analysis
                
                **Required CSV Format**:
                - Must contain 'City' and 'AQI' columns
                - Should have pollutant columns like PM2.5, PM10, NO2, etc.
                
                **For Support**: sammyryuga@gmail.com
            """)
        
        with st.expander("Team"):
            st.markdown("""
                **Meet the Team:**
                - **Akshat Goel**  
                  [LinkedIn](https://www.linkedin.com/in/akshat-goel-b13054323/) | [Instagram](https://www.instagram.com/akshatgoel_1105006/)
                - **Anwita Padhi**  
                  [LinkedIn](https://www.linkedin.com/in/anwita-padhi-187576321/) | [Instagram](https://www.instagram.com/anwitapadhi/)
                - **Samanyu Pattanayak**  
                  [LinkedIn](https://www.linkedin.com/in/samanyu-pattanayak-8757551a9/) | [Instagram](https://www.instagram.com/sammyryuga/)
                - **Samruddh Om Bahanwal**  
                  [LinkedIn](https://www.linkedin.com/in/samruddh-om-bahanwal-48a93a228/) | [Instagram](https://www.instagram.com/samruddh_om/)
                - **Shruti Deepak**  
                  [LinkedIn](https://www.linkedin.com/in/shruti-deepak-956820362/) | [Instagram](https://instagram.com/silent_silverkitty)
            """)
    
    # File upload
    uploaded_file = st.file_uploader("**Choose a CSV file**", type="csv", help="Upload a CSV file containing City, AQI, and pollutant columns")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate dataset
            required_cols = ['City', 'AQI']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()
            
            # Check if dataset changed
            if 'df' not in st.session_state or not st.session_state.df.equals(df):
                st.session_state.df = df
                with st.spinner("Loading dataset and training model..."):
                    st.session_state.predictor = load_predictor(df)
                st.success("✅ Dataset loaded and model trained successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.stop()
    
    if 'df' not in st.session_state or st.session_state.predictor is None:
        st.info("👆 Please upload a CSV file to proceed")
        st.markdown("""
        **Expected CSV Format:**
        - `City`: Name of the city
        - `AQI`: Air Quality Index value
        - Pollutant columns: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
        - Optional: `Date` column
        """)
        st.stop()
    
    predictor = st.session_state.predictor
    
    # Page routing
    if page == "Home":
        predictor.interactive_prediction()
    elif page == "Graphs":
        predictor.create_visualizations()
    elif page == "View Model Report":
        predictor.generate_report()

if __name__ == "__main__":
    main()
