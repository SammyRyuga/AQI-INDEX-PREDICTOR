# 🌍 AQI Index Predictor using Linear Regression

An interactive, feature-rich Air Quality Index (AQI) predictor powered by Linear Regression and built with **Streamlit**. This tool analyzes air quality using pollutant data and provides real-time predictions with detailed visualizations and reports.

> 📌 Developed with ❤️ by
> 
>  [Akshat Goel](https://www.linkedin.com/in/akshat-goel-b13054323/),
>  [Anwita Padhi](https://www.linkedin.com/in/anwita-padhi-187576321/),
> [Samanyu Pattanayak](https://www.linkedin.com/in/samanyu-pattanayak-8757551a9/),
> [Samruddh Om Bahanwal](https://www.linkedin.com/in/samruddh-om-bahanwal-48a93a228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app), and
> [Shruti Deepak](https://www.linkedin.com/in/shruti-deepak-956820362/).

---

## 🚀 Features

- 📈 Linear Regression model for AQI prediction
- 🌡️ Supports key pollutants: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `NH3`, and more
- 📊 Rich visualizations:
  - Actual vs Predicted AQI
  - Residual plots
  - Q-Q plot for normality check
  - Feature importance (coefficients)
- 🏙️ City-wise prediction with searchable dropdown
- ✍️ Interactive AQI prediction form with enhanced UI
- 📋 Detailed model performance report with styled output
- 📤 CSV upload with automatic data preprocessing
- 🔁 5-Fold Cross-Validation for model robustness
- 🎨 Improved UI with custom styling and hover effects

---

## 📁 Project Structure

- `Model_Final.py`: Streamlit app and AQI Predictor class used by Codespaces.
- `requirements.txt`: List of dependencies required to run the project.
- `LICENSE`: MIT License file for the project.
- `README.md`: This file, providing an overview and setup instructions.

---

## 🛠️ Installation

To run the AQI Index Predictor locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SammyRyuga/AQI-INDEX-PREDICTOR
   cd AQI-INDEX-PREDICTOR
   ```

2. **Install and launch the Codespaces entry point**
   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run Model_Final.py
   ```

   Upload a CSV containing `City`, `AQI`, and pollutant columns such as `PM2.5`
   and `PM10`. `Date` is optional; when provided, its range appears in the report.

## Regression tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

These tests run the Streamlit app with synthetic CSV uploads and check prediction
inputs, graphs, and reports with and without the optional `Date` column.
