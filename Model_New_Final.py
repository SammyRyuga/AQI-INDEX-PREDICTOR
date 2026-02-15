#!/usr/bin/env python3
# Shebang

# AQI Index Predictor using Linear Regression
# Entrypoint for Streamlit Community Cloud: Run with `streamlit run app.py`

import pandas as pd
import numpy as np
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
        
        X = self.df[available_features + ['City']].copy()
        y = self.df[target_col].copy()
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
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
                correlation = np.corrcoef(X[feature], y)[0, 1]
                feature_importance[feature] = abs(correlation)
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:8]]
        return top_features, feature_importance
    
    def train_model(self, X, y, test_size=0.2):
        if X is None or y is None:
            return None, None, None, None, None, None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Preparing data for training...")
        time.sleep(0.5)
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
        time.sleep(0.5)
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
        
        st.markdown("<div class='section-header'>Generating Visualizations</div>", unsafe_allow_html=True)
        plt.switch_backend('Agg')
        available_styles = plt.style.available
        style = 'ggplot'
        if 'seaborn-v0_8' in available_styles:
            style = 'seaborn-v0_8'
        plt.style.use(style)
        sns.set_palette("husl")
        
        plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 3, 1)
        plt.scatter(self.y_train, self.y_train_pred, alpha=0.6, color='blue', label='Training Data')
        plt.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Training: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.6, color='green', label='Testing Data')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Testing: Actual vs Predicted AQI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        residuals = self.y_test - self.y_test_pred
        plt.scatter(self.y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted AQI')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot: Residuals vs Normal Distribution')
        plt.grid(True, alpha=0.3)
        
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
        st.pyplot(plt)
        plt.close()
    
    def predict_single_sample(self, sample_data):
        if not self.is_trained:
            st.error("Model not trained yet")
            return None
        
        sample_df = pd.DataFrame([sample_data])
        for col in self.feature_names:
            if col not in sample_df.columns:
                sample_df[col] = 0
        sample_df = sample_df[self.feature_names]
        sample_scaled = self.scaler.transform(sample_df)
        prediction = self.model.predict(sample_scaled)[0]
        # Ensure predicted AQI is non-negative
        return max(0, prediction)
    
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
                    min_value=0.0,  # Prevent negative values in UI
                    value=0.0,
                    step=0.1,
                    format="%.1f"
                )
            
            submit_button = st.form_submit_button(label="Predict AQI")
        
        if submit_button:
            # Check for negative values (redundant with min_value but added for robustness)
            if any(value < 0 for value in sample_data.values() if isinstance(value, (int, float))):
                st.error("Negative values are not allowed for pollutant inputs. Please enter non-negative values.")
                return
            
            predicted_aqi = self.predict_single_sample(sample_data)
            if predicted_aqi is not None:
                st.markdown(f"**Predicted AQI**: <span style='color: #e74c3c'>{predicted_aqi:.3f}</span>", unsafe_allow_html=True)
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
                st.markdown(f"**AQI Category**: <span style='color: {color}'>{category}</span>", unsafe_allow_html=True)
                st.markdown(f"**Color Code**: <span style='color: {color}'>{color}</span>", unsafe_allow_html=True)
    
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
        st.write(f"**Date range**: {self.df['Date'].min()} to {self.df['Date'].max()}")
        st.markdown("**Data Quality Check**:")
        missing_counts = self.df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                st.write(f"  - {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        st.subheader("Preparing Features for Modeling")
        if self.pollutant_features:
            st.write(f"**Available pollutant features**: {self.pollutant_features}")
            st.write(f"**Valid samples after removing missing AQI**: {len(self.X_train) + len(self.X_test)}")
            st.write("**Encoding cities**")
            st.write("**Imputing missing values**")
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
            metrics_df = pd.DataFrame({
                "Metric": ["R² Score", "MAE", "RMSE"],
                "Training": [f"{metrics['train_r2']:.4f}", f"{metrics['train_mae']:.4f}", f"{metrics['train_rmse']:.4f}"],
                "Testing": [f"{metrics['test_r2']:.4f}", f"{metrics['test_mae']:.4f}", f"{metrics['test_rmse']:.4f}"]
            })
            st.markdown("**Performance Metrics**:")
            st.dataframe(metrics_df.style.set_properties(**{'text-align': 'center'}))
            st.write(f"**5-Fold Cross-Validation R² Score**: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
            if metrics['test_r2'] > 0.8:
                st.markdown('<p style="color:green; font-weight:bold;">Excellent model performance</p>', unsafe_allow_html=True)
            elif metrics['test_r2'] > 0.6:
                st.markdown('<p style="color:blue; font-weight:bold;">Good model performance</p>', unsafe_allow_html=True)
            elif metrics['test_r2'] > 0.4:
                st.markdown('<p style="color:orange; font-weight:bold;">Moderate model performance</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red; font-weight:bold;">Poor model performance - consider feature engineering</p>', unsafe_allow_html=True)
        
        st.subheader("Model Summary")
        st.write(f"**Algorithm**: Linear Regression")
        st.write(f"**Features**: {len(self.feature_names)}")
        st.write(f"**Training samples**: {len(self.y_train)}")
        st.write(f"**Testing samples**: {len(self.y_test)}")
        
        st.subheader("Model Equation")
        st.write(f"**AQI** = {self.model.intercept_:.4f} + Σ(coefficient × feature)")
        
        st.subheader("Linear Regression Coefficients Analysis")
        st.divider()
        coef_df = self.analyze_coefficients()
        if coef_df is not None:
            st.write(f"**Model Intercept**: {self.model.intercept_:.4f}")
            st.markdown("**Top 10 Most Important Features**:")
            st.dataframe(coef_df.head(10).style.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]}]
            ))
            st.markdown("**Coefficient Interpretation**:")
            st.write("- Positive coefficients: Increase in feature leads to higher AQI")
            st.write("- Negative coefficients: Increase in feature leads to lower AQI")
            st.write("- Larger absolute values: More influence on AQI prediction")
        
        st.subheader("Prediction Accuracy")
        st.write(f"**The model explains** {metrics['test_r2']*100:.1f}% **of AQI variance**")
        st.write(f"**Average prediction error**: ±{metrics['test_mae']:.1f} AQI units")
        
        st.subheader("Recommendations")
        if metrics['test_r2'] > 0.7:
            st.write("- Model shows good predictive performance")
            st.write("- Suitable for AQI prediction tasks")
        else:
            st.write("- Consider collecting more data")
            st.write("- May need feature engineering")
        
        st.subheader("Data Insights")
        coef_df = self.analyze_coefficients()
        top_pollutants = coef_df[~coef_df['Feature'].str.contains('City')].head(3)
        st.markdown("**Most influential pollutants**:")
        for _, row in top_pollutants.iterrows():
            st.write(f"  - {row['Feature']}: {row['Coefficient']:.4f}")

@st.cache_resource
def load_predictor(_df: pd.DataFrame) -> AQIPredictor:
    predictor = AQIPredictor()
    predictor.df = _df
    X, y, pollutant_features = predictor.prepare_features()
    if X is None or y is None:
        raise ValueError("Failed to prepare features from the provided DataFrame.")
    predictor.pollutant_features = pollutant_features
    top_features, feature_importance = predictor.feature_analysis(X, y, pollutant_features)
    predictor.top_features = top_features
    predictor.feature_importance = feature_importance
    predictor.train_model(X, y)
    return predictor

def main():
    # Initialize session state for page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    st.markdown("<div class='main-title'>AQI Predictor</div>", unsafe_allow_html=True)
    st.markdown("Welcome to the Air Quality Index (AQI) Predictor! Upload a dataset, predict AQI based on pollutant levels, and explore detailed model insights.", unsafe_allow_html=True)
    
    st.sidebar.markdown("🌍 **AQI Predictor**")
    st.sidebar.title("Navigation Menu")
    options = ["Home", "Graphs", "View Model Report"]
    page = st.sidebar.radio("Go to", options, index=options.index(st.session_state.current_page))
    st.session_state.current_page = page
    
    with st.sidebar:
        st.title("Credits & Help")
        with st.expander("Credits"):
            st.markdown("""
                **Developed by** Akshat Goel, Anwita Padhi, Samanyu Pattanayak, Samruddh Om Bahanwal, Shruti Deepak:
                
                **About Us**  
                We are a passionate team dedicated to improving air quality awareness using data science. Our mission is to provide actionable insights through innovative tools like the AQI Predictor.
                
                - Streamlit for the interactive UI
                - Pandas and NumPy for data processing
                - Scikit-learn for machine learning
                - Matplotlib and Seaborn for visualizations
                - Special thanks to ISTE for assistance
            """)
        with st.expander("Help"):
            st.markdown("""
                **How to Use:**
                - Upload a CSV file (e.g., city_day.csv) on the Home page
                - Navigate using the left menu
                - Use Home page to input pollutant values and predict AQI
                - View graphs or reports for detailed analysis
                - For issues, contact support at **sammyryuga@gmail.com**
            """)
        with st.expander("Team"):
            st.markdown("""
                **Meet the Team:**
                - **Akshat Goel**  
                  - LinkedIn: [linkedin.com/in/akshat-goel](https://www.linkedin.com/in/akshat-goel-b13054323/)  
                  - Instagram: [@akshatgoel_1105006](https://www.instagram.com/akshatgoel_1105006/)
                - **Anwita Padhi**  
                  - LinkedIn: [linkedin.com/in/anwita-padhi](https://www.linkedin.com/in/anwita-padhi-187576321/)  
                  - Instagram: [@anwitapadhi](https://www.instagram.com/anwitapadhi/)
                - **Samanyu Pattanayak**  
                  - LinkedIn: [linkedin.com/in/samanyu-pattanayak](https://www.linkedin.com/in/samanyu-pattanayak-8757551a9/)  
                  - Instagram: [@sammyryuga](https://www.instagram.com/sammyryuga/)
                - **Samruddh Om Bahanwal**  
                  - LinkedIn: [linkedin.com/in/samruddh-om-bahanwal](https://www.linkedin.com/in/samruddh-om-bahanwal-48a93a228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)  
                  - Instagram: [@samruddh_om](https://www.instagram.com/samruddh_om/)
                - **Shruti Deepak**  
                  - LinkedIn: [linkedin.com/in/shruti](https://www.linkedin.com/in/shruti-deepak-956820362/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
                  - Instagram: [@silent_silverkitty](https://instagram.com/silent_silverkitty)
            """)
    
    # Handle file upload with session state persistence
    uploaded_file = st.file_uploader("**Choose a CSV file**", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'df' not in st.session_state or not st.session_state.df.equals(df):
                st.session_state.df = df
                st.cache_resource.clear()
                st.success("Dataset loaded and model updated successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    if 'df' not in st.session_state:
        st.info("Please upload a file to proceed")
        st.stop()
    
    # Load cached predictor
    try:
        predictor = load_predictor(st.session_state.df)
    except ValueError:
        st.error("Failed to prepare the model from the uploaded dataset. Please check the data format.")
        st.stop()
    
    # Clear any previous empty containers
    st.empty()
    
    if page == "Home":
        predictor.interactive_prediction()
    elif page == "Graphs":
        predictor.create_visualizations()
    elif page == "View Model Report":
        predictor.generate_report()

if __name__ == "__main__":
    main()
